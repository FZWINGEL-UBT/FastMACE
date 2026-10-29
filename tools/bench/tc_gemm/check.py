"""Correctness check of ``libfm_tc_gemm.so`` (run before any timing): every variant, tile and B layout, dense and
joined (segment_gemm's strided batched layout), against a float64 product, plus CUDA-graph replay = eager, plus
the patched ``tf32x3_gemms`` launches against Triton's tf32x3. Prints one line per case and exits 1 on a failure.

    FASTMACE_SCREEN_TC_GEMM_LIB=.../libfm_tc_gemm.so python check.py
"""

import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch, SPEC §14)
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cutlass_tc  # noqa: E402

# Max relative error bounds against float64 (Gaussian operands): FP32-class for the 3x variants, TF32 for tf32.
BOUND = {"rz3": 5e-6, "rn3": 5e-6, "rn4": 5e-6, "tf32": 5e-3}
failures = []


def report(name, ok, detail):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)
    if not ok:
        failures.append(name)


def rel(got, ref):
    return float((got.double() - ref).abs().max() / ref.abs().max())


def main():
    torch.manual_seed(0)
    dev = "cuda"
    print(torch.cuda.get_device_name(), torch.__version__, flush=True)
    for variant in cutlass_tc.VARIANTS:
        for tile in (0, 1):
            for m, k, n in ((216, 512, 512), (1000, 96, 160), (37, 33, 65), (5, 32, 32)):
                a = torch.randn(m, k, device=dev)
                w = torch.randn(k, n, device=dev)
                ref = a.double() @ w.double()
                for col in (False, True):
                    b = w.t().contiguous().t() if col else w
                    out = torch.full((m, n), float("nan"), device=dev)
                    cutlass_tc.dense(variant, a, b, out, tile=tile)
                    torch.cuda.synchronize()
                    e = rel(out, ref)
                    report(f"dense {variant} t{tile} {m}x{k}x{n} col={col}", e < BOUND[variant], f"{e:.2e}")
            # joined: 7 atoms x dim 3 rows of K into an output of width W at column offset 10, as segment_gemm
            atoms, dim, k, n, width, offset = 300, 3, 64, 96, 10 + 3 * 96 + 6, 10
            x = torch.randn(atoms * dim, k, device=dev)
            w = torch.randn(k, n, device=dev)
            out = torch.full((atoms, width), float("nan"), device=dev)
            cutlass_tc.joined(variant, x, 0, dim * k, k, w, out, offset, width, n, atoms * dim, dim, tile=tile)
            torch.cuda.synchronize()
            ref = (x.double() @ w.double()).reshape(atoms, dim * n)
            got = out[:, offset:offset + dim * n]
            e = rel(got, ref)
            untouched = bool(out[:, :offset].isnan().all() and out[:, offset + dim * n:].isnan().all())
            report(f"joined {variant} t{tile}", e < BOUND[variant] and untouched, f"{e:.2e}, rest untouched={untouched}")
    # capture: a graph replay equals the eager result bitwise
    a, w = torch.randn(1512, 512, device=dev), torch.randn(512, 512, device=dev)
    eager, graphed = torch.empty(1512, 512, device=dev), torch.empty(1512, 512, device=dev)
    cutlass_tc.dense("rn3", a, w, eager)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        cutlass_tc.dense("rn3", a, w, graphed)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    graphed.zero_()
    with torch.cuda.graph(g):
        cutlass_tc.dense("rn3", a, w, graphed)
    graphed.zero_()
    g.replay()
    torch.cuda.synchronize()
    report("graph replay == eager", bool(torch.equal(eager, graphed)), "bitwise")
    # the patched record against Triton's tf32x3 (both 3xTF32: rounding-level apart)
    from fastmace.kernels.triton import segment_gemm, tf32x3_gemm
    from fastmace.kernels.triton import tile_rule

    x, w = torch.randn(1944 * 3, 128, device=dev), torch.randn(128, 128, device=dev)
    rows = [x, torch.randn(1944, 64, device=dev)]
    weights = [w, torch.randn(64, 256, device=dev)]
    weights_t = [t.t().contiguous() for t in weights]
    triton_out = segment_gemm.joined_products(rows, weights, weights_t, [0, 384], [3, 1], 640, "tf32x3")
    triton_dense = tf32x3_gemm.tf32x3_mm(x, w)
    cutlass_tc.install("rn3")
    try:
        ours = segment_gemm.joined_products(rows, weights, weights_t, [0, 384], [3, 1], 640, "tf32x3")
        ours_dense = tf32x3_gemm.tf32x3_mm(x, w)
        torch.cuda.synchronize()
    finally:
        pass
    ref = torch.cat([(x.double() @ w.double()).reshape(1944, 384), rows[1].double() @ weights[1].double()], 1)
    for name, got, base in (("joined", ours, triton_out), ("dense", ours_dense, triton_dense)):
        r = ref if name == "joined" else x.double() @ w.double()
        report(f"patched record {name} vs float64", rel(got, r) < 5e-6, f"cutlass {rel(got, r):.2e}, "
               f"triton {rel(base, r):.2e}")
    print("block size", tile_rule.BLOCK_K, flush=True)
    print("FAILED: " + ", ".join(failures) if failures else "ALL OK", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
