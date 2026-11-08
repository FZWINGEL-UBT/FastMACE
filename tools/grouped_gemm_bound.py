"""Grouped-GEMM bound (9 November 2026): what one launch per product group could save over today's per-path launches.

OMAT-0 medium on SrTiO3 10x10x10 (5,000 atoms) runs four product groups with one CUTLASS 3xTF32 launch per path,
M = 5,000 x (1, 3, 5, 7) rows each (the H200 round-6 breakdown, results/h200-fastpath-20261108/round6/breakdown):

- interaction 2's ``linear_div`` forward, K = 256 / 384 / 384 / 256, N = 128 (``fm_tf32x3_remap_div``, 145 us);
- its input gradient, K = 128, N = 256 / 384 / 384 / 256 (four of ``fm_tf32x3_rr``'s six launches);
- interaction 1's species GEMM, K = 192, N = 128 (``fm_tf32x3_species``, 99 us), and its gradient, K = 128, N = 192
  (101 us).

A grouped GEMM (one persistent launch over every path's tiles) can at best run the group's tiles as one well-filled
launch of the same arithmetic. So the bound times, inside CUDA graphs (launch gaps as in production):

- ``seq``: the four products as four ``tf32x3.mm_into`` launches (the 64x64 mainloop every one of these kernels uses);
- ``one``: one ``mm_into`` launch of M = 80,000 rows with the group's mean K (or N), so the same multiply-adds.

``seq - one`` per group is the most a grouped kernel could save (its own per-tile overhead not counted). The
remap, select and species epilogues are left out: the same in both, so they cancel in the difference.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

ATOMS = 5000
SIZES = (1, 3, 5, 7)
# name: (K per path, N per path); M = ATOMS * size
GROUPS = {
    "linear_div": ((256, 384, 384, 256), (128,) * 4),
    "linear_div_grad": ((128,) * 4, (256, 384, 384, 256)),
    "species": ((192,) * 4, (128,) * 4),
    "species_grad": ((128,) * 4, (192,) * 4),
}


def operands(m, k, n, device):
    a = torch.randn(m, k, device=device)
    b = torch.randn(k, n, device=device) / k ** 0.5
    return a, b, torch.empty(m, n, device=device)


def graph(launches, repeat):
    """A CUDA graph replaying ``launches`` (a list of (a, b, out)) ``repeat`` times."""
    from fastmace.kernels.cutlass import tf32x3
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for a, b, out in launches:  # warm: the kernels load before capture
            tf32x3.mm_into(a, b, out)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(repeat):
            for a, b, out in launches:
                tf32x3.mm_into(a, b, out)
    return g


def timed(g, repeat):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / repeat  # us per group


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=21)
    p.add_argument("--repeat", type=int, default=20, help="group launches per graph replay")
    args = p.parse_args()
    from fastmace.kernels.cutlass import tf32x3
    device = torch.device("cuda")
    if not tf32x3.available(device):
        raise SystemExit(f"the CUTLASS tf32x3 kernels do not run on {torch.cuda.get_device_name()}")
    torch.manual_seed(0)
    result = {"gpu": torch.cuda.get_device_name(), "sms": torch.cuda.get_device_properties(0).multi_processor_count,
              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rounds": args.rounds,
              "repeat": args.repeat, "groups": {}}
    total_saving = 0.0
    for name, (ks, ns) in GROUPS.items():
        ms = [ATOMS * s for s in SIZES]
        seq = [operands(m, k, n, device) for m, k, n in zip(ms, ks, ns)]
        rows = sum(ms)
        madds = sum(m * k * n for m, k, n in zip(ms, ks, ns))
        if len(set(ks)) == 1:  # N varies (or nothing does): one launch at the mean N
            k = ks[0]
            n = madds // (rows * k)
        else:
            n = ns[0]
            k = madds // (rows * n)
        assert rows * k * n == madds and k % 4 == 0 and n % 4 == 0, (name, k, n)
        one = [operands(rows, k, n, device)]
        graphs = {"seq": graph(seq, args.repeat), "one": graph(one, args.repeat)}
        times = {key: [] for key in graphs}
        for r in range(args.rounds):  # alternate the order every round
            for key in (("seq", "one") if r % 2 == 0 else ("one", "seq")):
                times[key].append(timed(graphs[key], args.repeat))
        med = {key: statistics.median(v[1:]) for key, v in times.items()}  # the first round warms
        tiles_seq = sum(-(-m // 64) * -(-nn // 64) for m, nn in zip(ms, ns))
        tiles_one = -(-rows // 64) * -(-n // 64)
        saving = med["seq"] - med["one"]
        total_saving += saving
        result["groups"][name] = {"paths": [{"m": m, "k": kk, "n": nn} for m, kk, nn in zip(ms, ks, ns)],
                                  "one": {"m": rows, "k": k, "n": n}, "tiles": {"seq": tiles_seq, "one": tiles_one},
                                  "us": {key: round(v, 2) for key, v in med.items()},
                                  "us_spread": {key: [round(min(v[1:]), 2), round(max(v[1:]), 2)]
                                                for key, v in times.items()},
                                  "saving_us": round(saving, 2), "one_over_seq": round(med["one"] / med["seq"], 4)}
        print(f"{name}: seq {med['seq']:.1f} us ({tiles_seq} tiles in 4 launches), one {med['one']:.1f} us "
              f"({tiles_one} tiles), saving {saving:.1f} us (x {med['one'] / med['seq']:.4f})", flush=True)
        del graphs, seq, one
        torch.cuda.empty_cache()
    result["total_saving_us"] = round(total_saving, 2)
    print(f"total: grouped launches save at most {total_saving:.1f} us per force call", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "grouped_gemm_bound.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
