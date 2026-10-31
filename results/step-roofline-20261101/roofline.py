"""Roofline of the force call's largest kernel groups from op_profile.py's records: achieved FLOP rate and minimum
DRAM traffic per group, from the operators' shapes (no performance counters on the workstations).

- cuBLAS ``ampere_sgemm*`` and the matrix kernels under ``aten::mm``: 2 M K N FLOP, 4 (M K + K N + M N) bytes.
- ``segment_gemm`` (``gemm``) under ``fastmace::joined_products`` and ``gated_products_input_grads``: MH-1's
  linear_2 per segment, rows = atoms x (2l + 1) for l = 0..3, K = N = 512 (the launch times scale with the rows).
- ``gated_input_grads``: per gated segment l (D = 2l + 1), rows = atoms D, K = N = 512: 2 rows K N FLOP; bytes:
  the gradient rows read, the gated rows read, dgated written (4 rows N each) and the gates (4 atoms N).
- ``wrebuild_backward`` and ``fused_folded_forward``/``oeq_projected_backward_wprime``: per edge the w rebuild,
  2 x 65 x 1280 FLOP (exact); their tensor-product sums are not counted (lower bound).
    python roofline.py OPS.json... [--peak-tflops 26.7 --peak-gbs 360]
"""

import argparse
import json
from collections import defaultdict

MUL = 512
GATED_DIMS = (3, 5, 7)  # MH-1's gated segments l = 1, 2, 3
W_REBUILD_FLOP_PER_EDGE = 2 * 65 * 1280


def group_of(kernel):
    if kernel.startswith("ampere_sgemm") or "cutlass" in kernel and "sgemm" in kernel:
        return "cuBLAS sgemm"
    if kernel == "gemm":
        return "segment_gemm"
    if kernel == "gated_input_grads":
        return "gated_input_grads"
    if kernel == "wrebuild_backward":
        return "wrebuild_backward"
    if kernel in ("fused_folded_forward", "oeq_projected_backward_wprime"):
        return kernel
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("records", nargs="+")
    p.add_argument("--peak-tflops", type=float, default=26.7, help="FP32 non-tensor peak (RTX 4000 Ada: 26.7)")
    p.add_argument("--peak-gbs", type=float, default=360.0, help="DRAM bandwidth (RTX 4000 Ada: 360 GB/s)")
    args = p.parse_args()
    for path in args.records:
        d = json.load(open(path))
        n, edges = d["atoms"], d["edges"]
        ms = defaultdict(float)
        flop = defaultdict(float)
        nbytes = defaultdict(float)
        for op in d["ops"]:
            for kernel, t in op["kernels"]:
                g = group_of(kernel)
                if g is None:
                    continue
                ms[g] += t
                if g == "cuBLAS sgemm" and op["op"] in ("aten::mm", "aten::addmm") and len(op["shapes"]) >= 2:
                    a, b = op["shapes"][-2:] if op["op"] == "aten::addmm" else op["shapes"][:2]
                    m, k, nn = a[0], a[1], b[1]
                    flop[g] += 2 * m * k * nn
                    nbytes[g] += 4 * (m * k + k * nn + m * nn)
        # gated_input_grads: one launch per gated segment per call of the op
        calls = sum(1 for op in d["ops"] if op["op"] == "fastmace::gated_products_input_grads")
        rows_atoms = [op["shapes"][0][0] for op in d["ops"] if op["op"] == "fastmace::gated_products_input_grads"]
        for atoms in rows_atoms:
            for dim in GATED_DIMS:
                rows = atoms * dim
                flop["gated_input_grads"] += 2 * rows * MUL * MUL
                nbytes["gated_input_grads"] += 4 * (3 * rows * MUL + atoms * MUL)
        # segment_gemm: each joined_products call is linear_2's four products (l = 0..3: rows atoms x 1, 3, 5, 7,
        # K = N = 512; the launch times scale with the rows), and the scalars' input gradient inside
        # gated_products_input_grads is one more (atoms rows, 512 -> 512). The call's atoms are the region chunk's.
        joined = sum(1 for op in d["ops"] if op["op"] == "fastmace::joined_products")
        chunk_atoms = rows_atoms[0] if rows_atoms else n
        for _ in range(joined):
            rows = chunk_atoms * (1 + sum(GATED_DIMS))
            flop["segment_gemm"] += 2 * rows * MUL * MUL
            nbytes["segment_gemm"] += 4 * (2 * rows * MUL + 4 * MUL * MUL)
        for atoms in rows_atoms:
            flop["segment_gemm"] += 2 * atoms * MUL * MUL
            nbytes["segment_gemm"] += 4 * (2 * atoms * MUL + MUL * MUL)
        for g in ("wrebuild_backward", "fused_folded_forward", "oeq_projected_backward_wprime"):
            if ms[g]:
                flop[g] += edges * W_REBUILD_FLOP_PER_EDGE
        total = sum(t for op in d["ops"] for _, t in op["kernels"])
        print(f"\n{d['structure']} {n} atoms, {edges} edges, profiled call {total:.1f} ms "
              f"(step {d['step_ms']:.1f} ms); {calls} gated-grad calls, {joined} joined_products launches")
        print(f"| Group | ms | share | us/atom | TFLOP/s | % of {args.peak_tflops} | min GB/s | % of {args.peak_gbs} |")
        for g in ("gated_input_grads", "segment_gemm", "cuBLAS sgemm", "wrebuild_backward", "fused_folded_forward",
                  "oeq_projected_backward_wprime"):
            if not ms[g]:
                continue
            s = ms[g] / 1000.0
            tf = flop[g] / s / 1e12 if flop[g] else float("nan")
            gbs = nbytes[g] / s / 1e9 if nbytes[g] else float("nan")
            print(f"| {g} | {ms[g]:.1f} | {100 * ms[g] / total:.1f} % | {ms[g] * 1000 / n:.3f} | {tf:.2f} | "
                  f"{100 * tf / args.peak_tflops:.0f} % | {gbs:.0f} | {100 * gbs / args.peak_gbs:.0f} % |")


if __name__ == "__main__":
    main()
