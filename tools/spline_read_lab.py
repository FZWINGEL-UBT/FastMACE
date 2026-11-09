"""Lab only: other table read paths for the SPLINE_W forward, timed by ``tools/spline_table_bound.py`` (9 Nov 2026).

Two ways of fetching fewer table bytes into L1, both on the STAGE_EDGES kernel. The production kernel is unchanged;
these variants exist only here:

- ``bspline`` (Codex 9 Nov v2, proposal 2): a cubic B-spline on the C2 spline's refined grid (851 intervals,
  ``results/bspline-reuse-bound-20261109/c2_certificate.json``). The table holds one row of Q floats per knot,
  ``[pairs * (intervals + 3), Q]``. An edge in interval k reads rows k .. k + 3, four scalar loads per weight, and
  weighs them with the uniform cubic basis of t. Neighbouring intervals share three rows. Non-uniform band edges
  would need per-interval basis weights; those cost a few FMAs per edge, not per weight, and are left out. The table
  is random (only the time counts).
- ``packed8`` (Codex 9 Nov v3, proposal 2): production's grid with 8 bytes per weight instead of 16, in the format
  that passes the value certificate (``tools/spline_packed_certificate.py``, S12): the cubic in s = t - 1/2,
  economized to a quadratic (s^3 -> 3 s / 16), then a 6-bit shared exponent E and signed fixed-point a0 (22 bits),
  a1 (20), a2 (16) in steps 2^(E - 21), 2^(E - 19), 2^(E - 15). Layout per weight: x = a0 | (a2 >> 6) << 22,
  y = a1 | (a2 & 63) << 20 | (E + bias) << 26. The table is encoded from the captured production table, so its
  message can be compared with production's.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BASIS = ("            const float u1 = 1.f - t, tt = t * t;\n"
         "            const float b0 = u1 * u1 * u1 * (1.f / 6.f);\n"
         "            const float b1 = fmaf(tt, fmaf(t, 0.5f, -1.f), 2.f / 3.f);\n"
         "            const float b2 = fmaf(tt, fmaf(t, -0.5f, 0.5f), fmaf(t, 0.5f, 1.f / 6.f));\n"
         "            const float b3 = tt * t * (1.f / 6.f);\n")
CENTRE = "            const float s = t - 0.5f, s64 = 64.f * s;\n"
BITS = (22, 20, 16)  # a0, a1, a2 (packed8)


def c2_grid(path=Path("results/bspline-reuse-bound-20261109/c2_certificate.json")):
    """The C2 spline's refined grid (the last ``c2_refined`` round)."""
    return [tuple(g) for g in json.loads(Path(path).read_text())["c2_refined"][-1]["grid"]]


def _replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError(f"spline_read_lab: expected {count} of {old!r}, found {source.count(old)}")
    return source.replace(old, new)


def transform(source, read, bias=0):
    """The STAGE_EDGES forward source with the table read of ``read`` ("bspline" or "packed8"; ``bias``: packed8's
    exponent bias)."""
    if "s_tq[TILE]" not in source:
        raise ValueError("spline_read_lab: the variants need the STAGE_EDGES kernel")
    kind = {"bspline": "float", "packed8": "uint2"}[read]
    source = _replace(source, "const float4* __restrict__ table", f"const {kind}* __restrict__ table")
    source = _replace(source, "const float4* tq = table + s_tq[k] + ch;",
                      f"const {kind}* tq = table + s_tq[k] + ch;\n" + (BASIS if read == "bspline" else CENTRE))
    source = source.replace("const float4* ahead = table", f"const {kind}* ahead = table")  # PREFETCH_AHEAD
    if read == "bspline":  # (intervals + 3) knot rows per pair
        source = _replace(source, "tq_k = ((int64_t)pairs[e] * intervals + iv) * Q;",
                          "tq_k = ((int64_t)pairs[e] * (intervals + 3) + iv) * Q;")
    pattern = (r"const float4 c = __ldg\(tq \+ (\d+)\);\n\s*const float w = fmaf\(t, fmaf\(t, fmaf\(t, c\.w, c\.z\), "
               r"c\.y\), c\.x\);")
    loads = re.findall(pattern, source)
    if not loads:
        raise ValueError("spline_read_lab: no table reads found")

    def bspline(m):
        o = m.group(1)
        return (f"const float w = fmaf(b3, __ldg(tq + 3 * Q + {o}), fmaf(b2, __ldg(tq + 2 * Q + {o}), "
                f"fmaf(b1, __ldg(tq + Q + {o}), b0 * __ldg(tq + {o}))));")

    def packed8(m):  # w = 2^(E - 21) (q0 + s (4 q1 + s 64 q2))
        o = m.group(1)
        return (f"const uint2 c = __ldg(tq + {o});\n"
                f"                const int q0 = (int)(c.x << 10) >> 10, q1 = (int)(c.y << 12) >> 12;\n"
                f"                const int q2 = ((int)c.x >> 22) * 64 + (int)((c.y >> 20) & 63u);\n"
                f"                const float sc = __uint_as_float(((c.y >> 26) + {106 - bias}u) << 23);\n"
                f"                const float w = sc * fmaf(s, fmaf(s64, (float)q2, 4.f * (float)q1), (float)q0);")

    return re.sub(pattern, bspline if read == "bspline" else packed8, source), len(loads)


def ghoist(source):
    """The ``ghoist`` falsifier (third 9 Nov report, "G-hoisting"): every output's CG sum regrouped by sender feature,
    acc_o += (w x_i) g_oi, with the channel-independent g_oi = sum_j c_oij e_j read from a per-edge shared array. The
    instruction shape of the real rewrite (which would compute g at staging from the tile's harmonics); here g is
    filled with harmonics, so the output is wrong. Returns (source, number of g values per edge)."""
    if "s_tq[TILE]" not in source:
        raise ValueError("spline_read_lab: the variants need the STAGE_EDGES kernel")
    count = [0]

    def output(m):
        o, expr = m.group(1), m.group(2)
        xs = sorted(set(re.findall(r"\bx(\d)\b", expr)))
        lines = []
        for i in xs:
            lines.append(f"acc[{o}] = fmaf(w * x{i}, s_g[gk + {count[0]}], acc[{o}]);")
            count[0] += 1
        return "\n                ".join(lines)

    source = re.sub(r"acc\[(\d+)\] = fmaf\(w, (.*), acc\[\1\]\);", output, source)
    gd = (count[0] + 3) // 4 * 4
    source = _replace(source, "__shared__ __align__(16) float s_sh[TILE * ED];",
                      f"__shared__ __align__(16) float s_sh[TILE * ED];\n    constexpr int GD = {gd};\n"
                      "    __shared__ __align__(16) float s_g[TILE * GD];")
    fill = "for (int i = ch; i < count_in * ED; i += C) s_sh[i] = __ldg(sh + base * ED + i);"
    source = _replace(source, fill, fill + "\n        for (int i = ch; i < count_in * GD; i += C) "
                      "s_g[i] = __ldg(sh + base * ED + (i & 15));")
    she = "const float* she = s_sh + s_k[k] * ED;"
    source = _replace(source, she, she + "\n            const int gk = s_k[k] * GD;")
    return source, count[0]


def encode_packed8(table):
    """The packed8 table ``int32 [rows, Q, 2]`` of a production table ``float32 [rows, Q, 4]`` (t basis) and its
    exponent bias."""
    import torch
    c = table.double()
    c0, c1, c2, c3 = c.unbind(-1)
    a0, a1, a2 = c0 + c1 / 2 + c2 / 4 + c3 / 8, c1 + c2 + 0.75 * c3 + 3.0 / 16.0 * c3, c2 + 1.5 * c3
    top = torch.stack([a0, a1, a2]).abs().amax(dim=0)
    e = torch.floor(torch.log2(top.clamp_min(1e-300))) + 1
    live = top > 0
    emin, emax = int(e[live].min()), int(e[live].max())
    e = torch.where(live, e, torch.full_like(e, emin))
    bias = -emin
    if emax + bias > 63 or 106 - bias < 1:
        raise ValueError(f"packed8: exponents {emin} .. {emax} do not fit 6 bits")
    q = []
    for a, n in zip((a0, a1, a2), BITS):
        lim = 2 ** (n - 1) - 1
        q.append(torch.round(a / torch.exp2(e - n + 1)).clamp(-lim, lim).to(torch.int64))
    q0, q1, q2 = q
    x = (q0 & 0x3FFFFF) | ((q2 >> 6) & 0x3FF) << 22
    y = (q1 & 0xFFFFF) | (q2 & 63) << 20 | (e.to(torch.int64) + bias) << 26
    packed = torch.stack([x, y], dim=-1)
    packed = torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed).to(torch.int32)
    return packed.contiguous(), bias


def variant(kernel, read, table=None, grid=None):
    """A copy of the captured ``SplineForward`` with the table read of ``read``, compiled, and its table: for
    "bspline" a random one on ``grid`` (default: the C2 grid); for "packed8" ``table`` (production's) encoded.
    Returns (kernel, table, sources)."""
    import torch
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin
    from fastmace.kernels.oeq_streaming.spline_forward import SplineForward

    schedule, blocks, old_grid, r_max, device, path_scale, shared_locator = kernel.init_args
    grid = (c2_grid() if grid is None else grid) if read == "bspline" else old_grid
    k = SplineForward(schedule, blocks, grid, r_max, device, path_scale, shared_locator,
                      block_order=kernel.block_order)
    k.order = kernel.order
    if not k.stage_edges:
        raise SystemExit("spline_read_lab: STAGE_EDGES is off (set spline_forward:STAGE_EDGES=True)")
    q = len(schedule.paths) * schedule.channels
    bias = 0
    if read == "ghoist":
        new_table = table
    elif read == "bspline":
        pairs = table.shape[0] // kernel.intervals
        g = torch.Generator(device=device).manual_seed(0)
        new_table = torch.randn(pairs * (k.intervals + 3), q, device=device, generator=g) * 0.05
    else:
        new_table, bias = encode_packed8(table)
    capability = torch.cuda.get_device_capability(device)
    functions, loads, sources = [], 0, []
    k.lab_registers = []
    for i, source in enumerate(k.sources):
        new, n = ghoist(source) if read == "ghoist" else transform(source, read, bias)
        sources.append(new)
        loads += n
        module = _check(driver.cuModuleLoadData(compile_cubin(new, capability, ())))
        function = _check(driver.cuModuleGetFunction(module, f"spline_forward{i}".encode()))
        k._modules.append(module)
        functions.append(function)
        k.lab_registers.append(int(_check(driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, function))))
    k._functions = functions
    print(f"{read}: {k.intervals} intervals, {loads} table reads (ghoist: g values) per edge, "
          f"registers {k.lab_registers} "
          f"(production {kernel.registers}), table {new_table.numel() * 4 / 2 ** 20:.1f} MiB, bias {bias}",
          flush=True)
    return k, new_table, sources
