"""CPU only: does an 8-byte-per-weight SPLINE_W forward table pass the certificate? (9 November 2026)

Codex's 9 November v3 report (proposal 2) packs each weight's four cubic coefficients into 64 bits instead of 128. The
5060 Ti lab (``tools/spline_read_lab.py`` variant ``packed8``) shows the read path is worth it. This script asks
whether a concrete format keeps the forward's values within production's certificate.

The forward and w' read one table today (w' differentiates it). A packed table would be a second, values-only table
for the forward; w' keeps the float32 table for its slopes. So the value parts of the certificate decide (the slope
parts are reported too, for the case where w' read the packed table as well).

Each format stores the cubic in the midpoint basis s = t - 1/2 (a0 + a1 s + a2 s^2 + a3 s^3; |s| <= 1/2 shrinks the
higher terms' share), from production's float64 Hermite coefficients:

- ``mN``: a float with N mantissa bits (round to nearest, own exponent);
- ``bN@ref+k``: a signed N-bit integer in steps of 2^(e - k), e the exponent of the reference coefficient of the same
  weight (block floating point: no exponent bits of its own);
- ``xN``: a signed N-bit fixed-point integer against the weight's shared exponent E (8 bits, stored once per weight):
  E = 1 + floor(log2 max_i |a_i|), steps 2^(E - N + 1), so nothing clips;
- ``cN``: a signed N-bit fixed-point integer against the column's exponent (per species pair and weight, over all
  intervals; a separate small table of Q floats per pair, read alongside), steps 2^(E_col - N + 1);
- ``econ``: the cubic economized to a quadratic first (s^3 -> 3 s / 16, error <= |a3| / 32), three coefficients.

The quantized coefficients go back to production's float32 power basis in t for the scorer (``spline._Scorer``:
error over the network's own float32 error, passes at <= 1; production's target 0.5).

    python tools/spline_packed_certificate.py CHECKPOINT OUTPUT.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastmace.rewrites import radial as rt  # noqa: E402
from fastmace.rewrites.radial import spline as sp  # noqa: E402

SPECIES = (38, 22, 8)  # Sr, Ti, O

# name: (a0, a1, a2, a3) codes, total bits (sign + 8 exponent bits for each float code)
FORMATS = {
    "f32 midpoint (reference)": (("m", 23), ("m", 23), ("m", 23), ("m", 23)),
    "a0 m15 only": (("m", 15), ("m", 23), ("m", 23), ("m", 23)),
    "a1 m15 only": (("m", 23), ("m", 15), ("m", 23), ("m", 23)),
    "a2 b10@a0 only": (("m", 23), ("m", 23), ("b", 10, 0, 6), ("m", 23)),
    "a3 b6@a0 only": (("m", 23), ("m", 23), ("m", 23), ("b", 6, 0, 2)),
    "S1: E8 x22 x16 x11 x7": (("x", 22), ("x", 16), ("x", 11), ("x", 7)),
    "S2: E8 x21 x17 x11 x7": (("x", 21), ("x", 17), ("x", 11), ("x", 7)),
    "S3: E8 x20 x16 x12 x8": (("x", 20), ("x", 16), ("x", 12), ("x", 8)),
    "S4: E8 x24 x14 x11 x7": (("x", 24), ("x", 14), ("x", 11), ("x", 7)),
    "S5 econ: E8 x24 x18 x14": (("x", 24), ("x", 18), ("x", 14), None),
    "S6 econ: E8 x26 x16 x14": (("x", 26), ("x", 16), ("x", 14), None),
    "S7 econ: E8 x25 x17 x14": (("x", 25), ("x", 17), ("x", 14), None),
    "C1: col c24 c20 c12 c8": (("c", 24), ("c", 20), ("c", 12), ("c", 8)),
    "C2: col c22 c18 c14 c10": (("c", 22), ("c", 18), ("c", 14), ("c", 10)),
    "C3: col c20 c18 c14 c12": (("c", 20), ("c", 18), ("c", 14), ("c", 12)),
    "C4 econ: col c24 c22 c18": (("c", 24), ("c", 22), ("c", 18), None),
    "econ unquantized (floor)": (("m", 52), ("m", 52), ("m", 52), None),
    "S8 econ: E7 x24 x19 x14": (("x", 24), ("x", 19), ("x", 14), None),
    "S9 econ: E8 x23 x19 x14": (("x", 23), ("x", 19), ("x", 14), None),
    "S10 econ: E8 x24 x19 x13": (("x", 24), ("x", 19), ("x", 13), None),
    "S11 econ: E8 x22 x20 x14": (("x", 22), ("x", 20), ("x", 14), None),
    "S12 econ: E6 x22 x20 x16": (("x", 22), ("x", 20), ("x", 16), None),
    "S13 econ: E6 x23 x20 x15": (("x", 23), ("x", 20), ("x", 15), None),
    "S14 econ: E6 x22 x21 x15": (("x", 22), ("x", 21), ("x", 15), None),
    "S15 econ: E6 x21 x21 x16": (("x", 21), ("x", 21), ("x", 16), None),
    "A: m15 m15 b10@a0 b6@a0": (("m", 15), ("m", 15), ("b", 10, 0, 6), ("b", 6, 0, 2)),
    "B: m15 m15 b10@a1 b6@a1": (("m", 15), ("m", 15), ("b", 10, 1, 6), ("b", 6, 1, 2)),
    "C: m19 m11 b10@a0 b6@a0": (("m", 19), ("m", 11), ("b", 10, 0, 6), ("b", 6, 0, 2)),
    "D: m17 m13 b10@a0 b6@a0": (("m", 17), ("m", 13), ("b", 10, 0, 6), ("b", 6, 0, 2)),
    "E econ: m19 m13 b14@a0": (("m", 19), ("m", 13), ("b", 14, 0, 8), None),
}


def midpoint(c):
    """t-basis [.., 4, Q] -> s = t - 1/2 basis."""
    c0, c1, c2, c3 = c.unbind(-2)
    return torch.stack([c0 + c1 / 2 + c2 / 4 + c3 / 8, c1 + c2 + 0.75 * c3, c2 + 1.5 * c3, c3], dim=-2)


def power(a):
    a0, a1, a2, a3 = a.unbind(-2)
    return torch.stack([a0 - a1 / 2 + a2 / 4 - a3 / 8, a1 - a2 + 0.75 * a3, a2 - 1.5 * a3, a3], dim=-2)


def exponent(x):
    return torch.floor(torch.log2(x.abs().clamp_min(1e-300)))


def quantize(a, codes):
    a = a.clone()
    if codes[3] is None:  # economize: s^3 ~ 3 s / 16 on [-1/2, 1/2]
        a[..., 1, :] += 3.0 / 16.0 * a[..., 3, :]
        a[..., 3, :] = 0.0
    ref = a.clone()
    out = a.clone()
    clipped = 0
    live = [i for i, code in enumerate(codes) if code is not None]
    shared = exponent(a[..., live, :].abs().amax(dim=-2)) + 1  # E: every |a_i| < 2^E
    column = exponent(a[..., live, :].abs().amax(dim=-2).amax(dim=1, keepdim=True)) + 1  # per (pair, q)
    for i, code in enumerate(codes):
        if code is None:
            continue
        x = a[..., i, :]
        if code[0] in ("x", "c"):
            step = torch.exp2((shared if code[0] == "x" else column) - code[1] + 1)
            top = 2 ** (code[1] - 1) - 1
            q = torch.round(x / step)
            clipped += int((q.abs() > top + 1).sum())  # top + 1: rounding at the edge, clamped
            out[..., i, :] = q.clamp(-top, top) * step
        elif code[0] == "m":
            step = torch.exp2(exponent(x) - code[1])
            out[..., i, :] = torch.where(x == 0, x, torch.round(x / step) * step)
        else:
            _, bits, r, k = code
            step = torch.exp2(exponent(ref[..., r, :]) - k)
            top = 2 ** (bits - 1) - 1
            q = torch.round(x / step)
            clipped += int((q.abs() > top).sum())
            out[..., i, :] = q.clamp(-top, top) * step
    return out, clipped


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("output", type=Path)
    p.add_argument("--interaction", type=int, default=1)
    args = p.parse_args(argv)
    torch.set_num_threads(8)
    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False).float()
    numbers = [int(z) for z in model.atomic_numbers]
    columns = [numbers.index(z) for z in SPECIES]
    pairs = [(s, t) for s in range(len(columns)) for t in range(len(columns))]
    interaction = model.interactions[args.interaction]
    chain64 = rt._GenericChain(model, interaction, columns, torch.float64, torch.device("cpu"))
    chain32 = rt._GenericChain(model, interaction, columns, torch.float32, torch.device("cpu"))
    weight = chain64.folded
    weight64 = weight.detach().double()
    r_max = float(model.r_max)
    started = time.time()
    spline, cert = sp.fit(chain64, chain32, weight, pairs, rt.R_LO, r_max, rt.CERTIFY_POINTS, rt.R_PHYSICAL,
                          seed=args.interaction)
    grid = [tuple(g) for g in spline["grid"]]
    print(f"production Hermite: {cert['intervals']} intervals, certificate {cert['worst']:.3f} {cert['parts']} "
          f"({time.time() - started:.0f} s)", flush=True)
    scorer = sp._Scorer(chain64, chain32, weight64, pairs, rt.R_LO, r_max, rt.CERTIFY_POINTS, rt.R_PHYSICAL,
                        args.interaction)
    coef64 = sp.coefficients(chain64, weight64, pairs, grid)  # [P, I, 4, Q] float64, t basis
    a = midpoint(coef64)
    result = dict(tool="spline_packed_certificate", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  intervals=cert["intervals"], production=dict(worst=cert["worst"], parts=cert["parts"]), formats={})
    e = exponent(a.abs().amax(dim=-2))
    e = e[a.abs().amax(dim=-2) > 0]
    print(f"shared exponents span {int(e.min())} .. {int(e.max())} (a 7-bit field holds 128)", flush=True)
    result["shared_exponent_span"] = [int(e.min()), int(e.max())]
    for name, codes in FORMATS.items():
        q, clipped = quantize(a, codes)
        c = scorer.certificate(scorer.relative(power(q).to(torch.float32), grid))
        value = max(v for k, v in c["parts"].items() if k.startswith("value"))
        bits = sum(0 if code is None else (9 + code[1] if code[0] == "m" else code[1]) for code in codes)
        bits += (7 if name.startswith("S8") else 6 if " E6 " in name else 8) if any(cd is not None and cd[0] == "x" for cd in codes) else 0
        result["formats"][name] = dict(codes=[list(cd) if cd else None for cd in codes], bits=bits, clipped=clipped,
                                       value_worst=value, worst=c["worst"], parts=c["parts"])
        print(f"{name:28s} {bits:3d} bits  value {value:.3f}  all {c['worst']:.3f}  clipped {clipped}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    if not math.isfinite(result["production"]["worst"]):
        raise SystemExit("production certificate not finite")


if __name__ == "__main__":
    main()
