"""Cheap float32 checks for the fp32 ideas of 28 October 2026 (plans/fp32-ideas-20261028.md), one GPU per process.

Modes (``--mode``, several allowed; each prints one JSON line per measurement):

- ``bw``: effective bandwidth of a read-write elementwise kernel against the tensor size (1 MB to 256 MB), with
  the tensor reused in a loop. Where the size fits L2 the loop hits L2; the ratio to the large sizes bounds
  what keeping atom-level intermediates in L2 could give (idea 1).
- ``chain``: a chain of ``--chain-length`` elementwise kernels over a NbCl5-sized atom tensor ([1944, 8192],
  64 MB), once over the whole tensor and once in ``--chunks`` row blocks (each block's chain runs before the next
  block), both captured in one CUDA graph. This is idea 1's real ceiling with the extra launches counted.
- ``gemmchunk``: the atom GEMMs [rows, 512] x [512, 512] of NbCl5's four l blocks (1944, 5832, 9720, 13608 rows),
  whole and split into 2, 4 and 8 row blocks, in CUDA graphs (the cost of idea 1's smaller GEMMs).
- ``strassen``: one Strassen-Winograd level for [rows, 512] x [512, 512], with the weight-side sums precomputed
  (the weights are fixed), against one cuBLAS product; time and the error against float64 (idea 5).
- ``emu``: the main atom GEMM shapes in float32 as torch runs them, timed, with their error against float64.
  Run it once plainly and once with ``CUBLAS_EMULATE_SINGLE_PRECISION=1`` (and ``CUBLAS_EMULATION_STRATEGY=eager``)
  in the environment: cuBLAS's BF16x9 FP32 emulation (idea 6). The process records the variables it saw.

Measurement only; no FastMACE import. TF32 stays off (the production setting).
"""

from __future__ import annotations

import argparse
import json
import os

import torch

ROWS = (1944, 5832, 9720, 13608)  # NbCl5 3x3x3 atoms x (2l+1), l = 0..3


def _gpu():
    p = torch.cuda.get_device_properties(0)
    return {"gpu": p.name, "capability": f"{p.major}.{p.minor}", "l2_mb": getattr(p, "L2_cache_size", 0) / 2**20,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "env": {k: os.environ[k] for k in os.environ if k.startswith("CUBLAS_")}}


def _emit(record):
    print(json.dumps(record), flush=True)


def _time(fn, repeat=20, inner=1):
    """Median ms per call of ``fn`` (each sample ``inner`` calls) with CUDA events."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / inner)
    samples.sort()
    return samples[len(samples) // 2]


def _graph(fn):
    """``fn`` captured in a CUDA graph; returns the replay."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


def mode_bw(args):
    for mb in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128, 256):
        n = mb * 2**20 // 4
        x = torch.rand(n, device="cuda")
        y = torch.empty_like(x)
        ms = _time(lambda: torch.mul(x, 1.0001, out=y), repeat=15, inner=20)
        _emit({"mode": "bw", "mb": mb, "ms": ms, "gb_s": 2 * n * 4 / ms / 1e6})


def _chain(x, outs):
    """``len(outs)`` dependent elementwise kernels (read one tensor, write one), as an atom gate chain does."""
    current = x
    for i, out in enumerate(outs):
        if i % 3 == 0:
            torch.sigmoid(current, out=out)
        elif i % 3 == 1:
            torch.mul(current, x, out=out)
        else:
            torch.add(current, x, alpha=0.5, out=out)
        current = out


def mode_chain(args):
    atoms, width = 1944, 8192  # 512 channels x 16 components
    x = torch.rand(atoms, width, device="cuda")
    outs = [torch.empty_like(x) for _ in range(args.chain_length)]
    whole = _graph(lambda: _chain(x, outs))
    base = _time(whole)
    _emit({"mode": "chain", "chunks": 1, "kernels": args.chain_length, "mb": x.numel() * 4 / 2**20, "ms": base})
    for chunks in args.chunks:
        bounds = [atoms * i // chunks for i in range(chunks + 1)]

        def chunked():
            for a, b in zip(bounds[:-1], bounds[1:]):
                _chain(x[a:b], [o[a:b] for o in outs])

        ms = _time(_graph(chunked))
        _emit({"mode": "chain", "chunks": chunks, "kernels": args.chain_length * chunks,
               "chunk_mb": x[: bounds[1]].numel() * 4 / 2**20, "ms": ms, "ratio": ms / base})


def mode_gemmchunk(args):
    w = torch.randn(512, 512, device="cuda") / 512**0.5
    for rows in ROWS:
        a = torch.randn(rows, 512, device="cuda")
        out = torch.empty(rows, 512, device="cuda")
        base = _time(_graph(lambda: torch.mm(a, w, out=out)))
        _emit({"mode": "gemmchunk", "rows": rows, "chunks": 1, "ms": base,
               "tflops": 2 * rows * 512 * 512 / base / 1e9})
        for chunks in (2, 4, 8):
            bounds = [rows * i // chunks for i in range(chunks + 1)]

            def chunked():
                for lo, hi in zip(bounds[:-1], bounds[1:]):
                    torch.mm(a[lo:hi], w, out=out[lo:hi])

            ms = _time(_graph(chunked))
            _emit({"mode": "gemmchunk", "rows": rows, "chunks": chunks, "ms": ms, "ratio": ms / base})


def mode_block(args):
    """An atom block as the model runs it: per l, GEMM, gate (sigmoid times the value), GEMM, residual add, over
    NbCl5's four l blocks (64 MB per tensor in all); whole, and with every l block split into the same atom
    fraction (the chain of one atom chunk runs before the next; idea 1 with its GEMMs counted)."""
    weights = [torch.randn(512, 512, device="cuda") / 512**0.5 for _ in range(2)]
    xs = [torch.randn(rows, 512, device="cuda") for rows in ROWS]
    bufs = [[torch.empty_like(x) for _ in range(4)] for x in xs]

    def chain(lo, hi):
        for x, (y, z, u, v) in zip(xs, bufs):
            a, b = x.shape[0] * lo // 1944, x.shape[0] * hi // 1944
            torch.mm(x[a:b], weights[0], out=y[a:b])
            torch.mul(torch.sigmoid(y[a:b]), y[a:b], out=z[a:b])
            torch.mm(z[a:b], weights[1], out=u[a:b])
            torch.add(u[a:b], x[a:b], out=v[a:b])

    base = _time(_graph(lambda: chain(0, 1944)))
    _emit({"mode": "block", "chunks": 1, "ms": base})
    for chunks in args.chunks:
        bounds = [1944 * i // chunks for i in range(chunks + 1)]

        def chunked():
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                chain(lo, hi)

        ms = _time(_graph(chunked))
        _emit({"mode": "block", "chunks": chunks, "ms": ms, "ratio": ms / base})


def mode_streams(args):
    """Idea 4's overlap factor: pairs of independent kernels on one stream against two streams (both captured in
    one CUDA graph): a GEMM with a 64 MB elementwise kernel (compute-bound with memory-bound), and the two GEMMs of
    two l blocks (both compute-bound), at NbCl5's sizes."""
    w = torch.randn(512, 512, device="cuda") / 512**0.5
    a = {rows: torch.randn(rows, 512, device="cuda") for rows in ROWS}
    out = {rows: torch.empty(rows, 512, device="cuda") for rows in ROWS}
    x = torch.rand(1944, 8192, device="cuda")
    y = torch.empty_like(x)
    side = torch.cuda.Stream()
    pairs = {
        "gemm13608+pointwise64MB": (lambda: torch.mm(a[13608], w, out=out[13608]),
                                    lambda: torch.mul(x, 1.0001, out=y)),
        "gemm5832+pointwise64MB": (lambda: torch.mm(a[5832], w, out=out[5832]), lambda: torch.mul(x, 1.0001, out=y)),
        "gemm9720+gemm13608": (lambda: torch.mm(a[9720], w, out=out[9720]),
                               lambda: torch.mm(a[13608], w, out=out[13608])),
        "gemm1944+gemm5832": (lambda: torch.mm(a[1944], w, out=out[1944]), lambda: torch.mm(a[5832], w, out=out[5832])),
    }
    for name, (first, second) in pairs.items():
        alone = [_time(_graph(first)), _time(_graph(second))]

        def serial():
            first()
            second()

        def concurrent():
            main = torch.cuda.current_stream()
            side.wait_stream(main)
            first()
            with torch.cuda.stream(side):
                second()
            main.wait_stream(side)

        one, two = _time(_graph(serial)), _time(_graph(concurrent))
        _emit({"mode": "streams", "pair": name, "first_ms": alone[0], "second_ms": alone[1], "one_stream_ms": one,
               "two_streams_ms": two, "ratio": two / one,
               "hidden_share_of_shorter": (one - two) / min(alone)})


def _winograd(a, b_sums, half):
    """One Strassen-Winograd level of ``a @ b`` (square 2x2 blocks of b, rows of a split in two), with b's sums
    (``b_sums``: B11, B12, B21, B22, T1..T4) precomputed."""
    rows = a.shape[0] // 2
    a11, a12 = a[:rows, :half], a[:rows, half:]
    a21, a22 = a[rows:, :half], a[rows:, half:]
    b11, b12, b21, b22, t1, t2, t3, t4 = b_sums
    s1 = a21 + a22
    s2 = s1 - a11
    s3 = a11 - a21
    s4 = a12 - s2
    m1 = a11 @ b11
    m2 = a12 @ b21
    m3 = s4 @ b22
    m4 = a22 @ t4
    m5 = s1 @ t1
    m6 = s2 @ t2
    m7 = s3 @ t3
    u2 = m1 + m6
    u3 = u2 + m7
    u4 = u2 + m5
    top = torch.cat((m1 + m2, u4 + m3), dim=1)
    bottom = torch.cat((u3 - m4, u3 + m5), dim=1)
    return torch.cat((top, bottom), dim=0)


def mode_strassen(args):
    w = torch.randn(512, 512, device="cuda", dtype=torch.float64) / 512**0.5
    half = 256
    b11, b12, b21, b22 = w[:half, :half], w[:half, half:], w[half:, :half], w[half:, half:]
    t1 = b12 - b11
    t2 = b22 - t1
    t3 = b22 - b12
    t4 = t2 - b21
    sums32 = [m.float().contiguous() for m in (b11, b12, b21, b22, t1, t2, t3, t4)]
    w32 = w.float()
    for rows in ROWS[1:]:
        a64 = torch.randn(rows, 512, device="cuda", dtype=torch.float64)
        a = a64.float()
        exact = a.double() @ w32.double()
        scale = (a.double().abs() @ w32.double().abs()).max()
        plain = torch.mm(a, w32)
        win = _winograd(a, sums32, half)
        base = _time(lambda: torch.mm(a, w32))
        ms = _time(lambda: _winograd(a, sums32, half))
        _emit({"mode": "strassen", "rows": rows, "mm_ms": base, "winograd_ms": ms, "ratio": ms / base,
               "mm_err": float((plain.double() - exact).abs().max() / scale),
               "winograd_err": float((win.double() - exact).abs().max() / scale)})


def mode_emu(args):
    shapes = [(r, 512, 512) for r in ROWS] + [(9720, 384, 512), (13608, 512, 256), (1944, 2048, 384),
                                              (56320, 68, 1280)]
    for m, k, n in shapes:
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda") / k**0.5
        exact = a.double() @ b.double()
        scale = (a.double().abs() @ b.double().abs()).max()
        out = torch.mm(a, b)
        ms = _time(_graph(lambda: torch.mm(a, b)), repeat=20)
        _emit({"mode": "emu", "m": m, "k": k, "n": n, "ms": ms, "tflops": 2 * m * k * n / ms / 1e9,
               "err": float((out.double() - exact).abs().max() / scale)})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", nargs="+", required=True,
                   choices=("bw", "chain", "block", "streams", "gemmchunk", "strassen", "emu"))
    p.add_argument("--chain-length", type=int, default=6)
    p.add_argument("--chunks", type=int, nargs="+", default=[2, 4, 8, 16])
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _emit({"mode": "gpu", **_gpu()})
    for mode in args.mode:
        globals()[f"mode_{mode}"](args)


if __name__ == "__main__":
    main()
