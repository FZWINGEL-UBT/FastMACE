"""Tensor-core GEMM screen (owner, 29 October 2026): every block-linear GEMM of one real MH-1 force call, replayed
through each backend, timed and compared with float64.

The products are recorded as ``tools/tf32x3_accuracy.py`` records them: a ``ForceField`` with
``RewriteConfig(tf32x3_gemms="on")`` (float32, compiled evaluation plan, no CUDA graphs), one warm-up evaluation,
then one evaluation whose dense (``tf32x3_gemm._launch``) and joined (``segment_gemm._gemm``) launches are copied
with their real operands, forward products and input gradients alike. Sub-tile products (K or N < 32) stay on SIMT
IEEE in every mode and are not recorded.

Modes, each on the same operands:

- ``ieee_default``: what the default plan runs: dense products on cuBLAS IEEE; joined products by
  ``tile_rule.segment_gemm_choice`` on the Triton IEEE ``segment_gemm`` or on cuBLAS plus the copy into the
  joined layout (emulated: gather the rows, ``mm``, copy into the output);
- ``cublas_batched``: joined products as one cuBLAS strided batched GEMM each, straight into the joined layout
  (``cutlass_tc.cublas_joined``); dense products on cuBLAS;
- ``cublas_ieee`` / ``cublas_tf32``: cuBLAS on the gathered, contiguous operands (pure GEMM time, no layout;
  TF32 is a rate reference);
- ``triton_tf32x3``: the record as it is (Triton ``tf32x3``);
- ``cutlass_<variant>`` (``cutlass_tc``: rz3, rn3, rn4, tf32) at the tile rule's tile, and ``cutlass_rz3_t0`` /
  ``_t1`` at both tiles, written straight into the joined layout;
- ``cutlass_simt`` (``_t0``, ``_t1``): CUTLASS SIMT IEEE FP32 straight into the joined layout (idea 2's IEEE
  alternative to the Triton ``segment_gemm``);
- ``fp16x3_prepacked`` (idea 3's first check): A and W split once, outside the timing, into power-of-two
  normalised FP16 big and 2^11-scaled small parts (per row of A, per column of W), then
  ``C = Ah Wh + 2^-11 [Ah | Al] [Wl; Wh]`` as two cuBLAS FP16 GEMMs with FP32 accumulation and output
  (``torch.mm``/``addmm`` with ``out_dtype``); ``fp16x3_split`` times the activation split alone (unfused).

Timing: one CUDA graph per product and mode holding ``--reps`` launches, replayed ``--iters`` times per block;
modes cycle in a rotated order over ``--blocks`` blocks in one process; the median per launch counts. Operands
repeat, so small products run L2-warm in every mode alike. Errors: max relative, relative Frobenius and the
cancellation-scaled max |c - c64| / sum |a b| (``tf32x3_accuracy.errors``). The two largest products also get
opt-T01 §6's synthetic cancellation operands. Output: ``<output>/<structure>.json``, ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[1]
MODES = ("ieee_default", "cublas_batched", "cublas_ieee", "cublas_tf32", "triton_tf32x3", "cutlass_rz3", "cutlass_rz3_t0",
         "cutlass_rz3_t1", "cutlass_rn3", "cutlass_rn4", "cutlass_tf32", "cutlass_simt", "cutlass_simt_t0",
         "cutlass_simt_t1", "fp16x3_prepacked", "fp16x3_split")
SMALL_SCALE = 2.0 ** 11


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="Fresh directory")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structures", nargs="+", default=["NaCl", "NaNbO3", "NbCl5"])
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--reps", type=int, default=10, help="launches per graph")
    p.add_argument("--iters", type=int, default=10, help="graph replays per block")
    p.add_argument("--blocks", type=int, default=7)
    p.add_argument("--rattle-A", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--provider", default="", help="Recorded label: machine and provider, e.g. 'trillian'")
    return p.parse_args(argv)


class Product:
    """One recorded product: gathered A [M, K], B [K, N] and the launch's own arguments."""

    def __init__(self, kind, a, b, launch):
        self.kind, self.a, self.b, self.launch = kind, a, b, launch
        self.m, self.k = a.shape
        self.n = b.shape[1]

    @property
    def flops(self):
        return 2 * self.m * self.k * self.n


def record(torch, ff, atoms):
    from fastmace.kernels.triton import segment_gemm, tf32x3_gemm

    dense, joined = tf32x3_gemm._launch, segment_gemm._gemm
    products = []

    def launch(a, b, out):
        dense(a, b, out)
        if tf32x3_gemm.uses_tf32x3_tile(a.shape[1], b.shape[1]):
            products.append(Product("dense", a.contiguous().clone(), b.clone(), dict(shape=tuple(out.shape))))

    def gemm(a, a_offset, a_n, a_m, b, c, c_offset, c_n, c_m, rows, dim, choice, precision="ieee"):
        joined(a, a_offset, a_n, a_m, b, c, c_offset, c_n, c_m, rows, dim, choice, precision)
        if precision == "tf32x3":
            k, n = b.shape
            left = a.reshape(-1)[a_offset:].as_strided((rows // dim, dim, k), (a_n, a_m, 1)).reshape(rows, k)
            products.append(Product("joined", left.clone(memory_format=torch.contiguous_format), b.clone(), dict(
                a=a.clone(), a_offset=a_offset, a_n=a_n, a_m=a_m, c_shape=tuple(c.shape), c_offset=c_offset,
                c_n=c_n, c_m=c_m, rows=rows, dim=dim)))

    tf32x3_gemm._launch, segment_gemm._gemm = launch, gemm
    try:
        ff.evaluate(atoms)
        torch.cuda.synchronize()
    finally:
        tf32x3_gemm._launch, segment_gemm._gemm = dense, joined
    return products


def _split16(torch, x, dim):
    """Power-of-two normalisation along ``dim`` (rows of A: dim=1; columns of W: dim=0), then FP16 big and
    2^11-scaled small parts. Returns (big, small, scale) with x = (big + small / 2^11) / scale."""
    peak = x.abs().amax(dim=dim, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    scale = torch.exp2(-torch.floor(torch.log2(peak)))
    y = x * scale
    big = y.half()
    small = ((y - big.float()) * SMALL_SCALE).half()
    return big, small, scale


class Runner:
    """Callables per mode for one product, with their output tensor."""

    def __init__(self, torch, product, sms, cutlass_tc, tile_rule, segment_gemm):
        self.torch, self.p, self.sms = torch, product, sms
        self.cutlass_tc, self.tile_rule, self.segment_gemm = cutlass_tc, tile_rule, segment_gemm
        p = product
        self.c = torch.empty((p.m, p.n), device=p.a.device)
        if p.kind == "joined":
            self.jc = torch.zeros(p.launch["c_shape"], device=p.a.device)
        ah, al, sa = _split16(torch, p.a, 1)
        wh, wl, sw = _split16(torch, p.b, 0)
        self.f16 = dict(ah=ah, x2=torch.cat([ah, al], 1), wh=wh, w2=torch.cat([wl, wh], 0), unscale=1 / (sa * sw))
        self.g1 = torch.empty((p.m, p.n), device=p.a.device)

    def joined_view(self):
        p, L = self.p, self.p.launch
        atoms = L["rows"] // L["dim"]
        return self.jc.reshape(-1)[L["c_offset"]:].as_strided((atoms, L["dim"], p.n), (L["c_n"], L["c_m"], 1)) \
            .reshape(L["rows"], p.n)

    def fn(self, mode):
        """(callable, result getter) for ``mode``, or None if the mode does not apply."""
        torch, p = self.torch, self.p
        L = p.launch
        if mode == "cublas_batched":
            if p.kind == "dense":
                return (lambda: torch.mm(p.a, p.b, out=self.c)), (lambda: self.c)
            return (lambda: self.cutlass_tc.cublas_joined(L["a"], L["a_offset"], L["a_n"], L["a_m"], p.b, self.jc,
                                                          L["c_offset"], L["c_n"], L["c_m"], L["rows"], L["dim"])),                 self.joined_view
        if mode == "cublas_ieee":
            return (lambda: torch.mm(p.a, p.b, out=self.c)), (lambda: self.c)
        if mode == "cublas_tf32":
            def run():
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.mm(p.a, p.b, out=self.c)
                torch.backends.cuda.matmul.fp32_precision = "ieee"
            return run, (lambda: self.c)
        if mode == "fp16x3_prepacked":
            f = self.f16

            def run():
                torch.mm(f["ah"], f["wh"], out_dtype=torch.float32, out=self.g1)
                torch.addmm(self.g1, f["x2"], f["w2"], alpha=1 / SMALL_SCALE, out_dtype=torch.float32, out=self.c)
            return run, (lambda: self.c * f["unscale"])
        if mode == "fp16x3_split":
            return (lambda: _split16(torch, p.a, 1)), None
        if mode.startswith("cutlass_"):
            parts = mode.split("_")
            variant = parts[1]
            tile = int(parts[2][1]) if len(parts) > 2 else None
            if p.kind == "dense":
                return (lambda: self.cutlass_tc.dense(variant, p.a, p.b, self.c, tile=tile)), (lambda: self.c)
            return (lambda: self.cutlass_tc.joined(variant, L["a"], L["a_offset"], L["a_n"], L["a_m"], p.b, self.jc,
                                                   L["c_offset"], L["c_n"], L["c_m"], L["rows"], L["dim"], tile)), \
                self.joined_view
        if mode == "triton_tf32x3":
            if p.kind == "dense":
                return (lambda: self._triton_dense()), (lambda: self.c)
            choice = self.tile_rule.GemmChoice("triton")
            return (lambda: self.segment_gemm._gemm(L["a"], L["a_offset"], L["a_n"], L["a_m"], p.b, self.jc,
                                                    L["c_offset"], L["c_n"], L["c_m"], L["rows"], L["dim"], choice,
                                                    "tf32x3")), self.joined_view
        if mode == "ieee_default":
            if p.kind == "dense":
                return (lambda: torch.mm(p.a, p.b, out=self.c)), (lambda: self.c)
            choice = self.tile_rule.segment_gemm_choice(L["rows"], p.n, p.k, self.sms)
            if choice.backend == "triton":
                return (lambda: self.segment_gemm._gemm(L["a"], L["a_offset"], L["a_n"], L["a_m"], p.b, self.jc,
                                                        L["c_offset"], L["c_n"], L["c_m"], L["rows"], L["dim"],
                                                        choice, "ieee")), self.joined_view
            view = self.joined_view()

            def run():
                left = L["a"].reshape(-1)[L["a_offset"]:].as_strided(
                    (L["rows"] // L["dim"], L["dim"], p.k), (L["a_n"], L["a_m"], 1)).reshape(L["rows"], p.k)
                view.copy_(left @ p.b)
            return run, self.joined_view
        raise ValueError(mode)

    def _triton_dense(self):
        from fastmace.kernels.triton import tf32x3_gemm

        tf32x3_gemm._launch(self.p.a, self.p.b, self.c)


def errors(torch, tf32x3_accuracy, got, a, b):
    a64, b64 = a.double(), b.double()
    return tf32x3_accuracy.errors(torch, got, a64 @ b64, a64.abs() @ b64.abs())


def cancellation_operands(torch, m, k, n, device, seed):
    """opt-T01 §6: A[:, 2j+1] = -A[:, 2j] (1 + 1e-3 u), B[2j+1, :] = B[2j, :]."""
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(m, k, device=device, generator=g)
    b = torch.randn(k, n, device=device, generator=g)
    u = torch.rand(m, k // 2, device=device, generator=g)
    a[:, 1::2] = -a[:, 0::2] * (1 + 1e-3 * u)
    b[1::2, :] = b[0::2, :]
    return a, b


def time_modes(torch, runners, modes, reps, iters, blocks):
    """{mode: [median µs per launch per product]} (None where a mode does not apply)."""
    graphs = {}
    stream = torch.cuda.Stream()
    for i, r in enumerate(runners):
        for mode in modes:
            fn = r.fn(mode)[0]
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fn()
                fn()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            if mode == "cublas_tf32":  # the matmul mode is read at launch, so the capture keeps TF32
                torch.backends.cuda.matmul.fp32_precision = "tf32"
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(reps):
                    fn()
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            graphs[i, mode] = g
    samples = {(i, m): [] for (i, m) in graphs}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for block in range(blocks):
        order = list(modes[block % len(modes):]) + list(modes[:block % len(modes)])
        if block % 2:
            order.reverse()
        for i in range(len(runners)):
            for mode in order:
                g = graphs[i, mode]
                g.replay()
                start.record()
                for _ in range(iters):
                    g.replay()
                end.record()
                end.synchronize()
                samples[i, mode].append(start.elapsed_time(end) * 1e3 / (iters * reps))
    return {mode: [statistics.median(samples[i, mode]) for i in range(len(runners))] for mode in modes}


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    import torch

    import fastmace as fm
    from fastmace.config import RewriteConfig
    from fastmace.kernels.triton import segment_gemm, tile_rule

    sys.path.insert(0, str(TOOLS))
    sys.path.insert(0, str(HERE))
    import ab_step_time  # noqa: E402
    import cutlass_tc  # noqa: E402
    import tf32x3_accuracy  # noqa: E402

    args.output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    cutlass_dir = Path(os.environ[cutlass_tc.ENV_LIB]).parent
    summary = dict(tool="tc_gemm_screen", created_utc=datetime.now(timezone.utc).isoformat(),
                   host=os.uname().nodename, provider=args.provider, python=sys.executable,
                   fastmace_commit=ab_step_time.fastmace_commit(), model=args.model,
                   gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                   sms=torch.cuda.get_device_properties(0).multi_processor_count, torch=torch.__version__,
                   cuda=torch.version.cuda, cutlass=(cutlass_dir / "cutlass_commit.txt").read_text().strip()
                   if (cutlass_dir / "cutlass_commit.txt").exists() else None,
                   nvcc=(cutlass_dir / "nvcc.txt").read_text().strip() if (cutlass_dir / "nvcc.txt").exists() else None,
                   reps=args.reps, iters=args.iters, blocks=args.blocks, modes=args.modes, structures={})
    import triton

    summary["triton"] = triton.__version__
    sms = tile_rule.device_sms("cuda")
    model = fm.load(args.model)
    for spec in args.structures:
        name, atoms = ab_step_time.read_structure(spec, None)
        rng = np.random.default_rng(args.seed)
        atoms.positions += rng.uniform(-args.rattle_A, args.rattle_A, atoms.positions.shape)
        ff = fm.ForceField(model, atoms, precision="float32", rewrites=RewriteConfig(tf32x3_gemms="on", grouped_launch="off"))
        ff.evaluate(atoms)
        products = record(torch, ff, atoms)
        del ff
        torch.cuda.empty_cache()
        runners = [Runner(torch, p, sms, cutlass_tc, tile_rule, segment_gemm) for p in products]
        times = time_modes(torch, runners, args.modes, args.reps, args.iters, args.blocks)
        rows = []
        for i, (p, r) in enumerate(zip(products, runners)):
            row = dict(kind=p.kind, m=p.m, k=p.k, n=p.n, dim=p.launch.get("dim", 1),
                       tile_rule=cutlass_tc.tile_for(p.launch.get("rows", p.m) // p.launch.get("dim", 1), p.n,
                                                     p.launch.get("dim", 1), sms),
                       tile_rule_simt=cutlass_tc.tile_for(p.launch.get("rows", p.m) // p.launch.get("dim", 1), p.n,
                                                          p.launch.get("dim", 1), sms, "simt"),
                       us={mode: times[mode][i] for mode in args.modes}, errors={})
            for mode in args.modes:
                fn, get = r.fn(mode)
                if get is None:
                    continue
                fn()
                torch.cuda.synchronize()
                row["errors"][mode] = errors(torch, tf32x3_accuracy, get(), p.a, p.b)
            rows.append(row)
        totals = {mode: sum(times[mode]) for mode in args.modes}
        largest = sorted(range(len(products)), key=lambda i: -products[i].flops)[:2]
        synthetic = []
        for i in largest:
            p = products[i]
            a, b = cancellation_operands(torch, p.m, p.k, p.n, p.a.device, args.seed)
            fake = Product("dense", a, b, dict(shape=(p.m, p.n)))
            r = Runner(torch, fake, sms, cutlass_tc, tile_rule, segment_gemm)
            entry = dict(m=p.m, k=p.k, n=p.n, errors={})
            for mode in args.modes:
                fn, get = r.fn(mode)
                if get is None:
                    continue
                fn()
                torch.cuda.synchronize()
                entry["errors"][mode] = errors(torch, tf32x3_accuracy, get(), a, b)
            synthetic.append(entry)
        worst = {mode: max((row["errors"][mode][2] for row in rows if mode in row["errors"]), default=None)
                 for mode in args.modes}
        ratios = {mode: [row["errors"][mode][2] / row["errors"]["cublas_ieee"][2] for row in rows
                         if mode in row["errors"] and row["errors"]["cublas_ieee"][2] > 0]
                  for mode in args.modes if "cublas_ieee" in args.modes}
        stem = Path(str(name)).stem
        result = dict(structure=stem, natoms=len(atoms), products=len(products), total_us=totals,
                      largest=[dict(m=products[i].m, k=products[i].k, n=products[i].n, kind=products[i].kind,
                                    us={mode: times[mode][i] for mode in args.modes}) for i in largest],
                      worst_cancellation_scaled=worst,
                      cancellation_ratio_to_ieee={m: dict(median=statistics.median(v), max=max(v)) for m, v in
                                                  ratios.items() if v},
                      synthetic_cancellation=synthetic, rows=rows)
        (args.output / f"{stem}.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        summary["structures"][stem] = {k: v for k, v in result.items() if k != "rows"}
        base = totals.get("ieee_default")
        print(f"{stem}: {len(products)} products", flush=True)
        for mode in args.modes:
            rel_ = f"  {totals[mode] / base:.3f}x default" if base else ""
            ratio = result["cancellation_ratio_to_ieee"].get(mode)
            acc = f"  cancel/ieee median {ratio['median']:.2f} max {ratio['max']:.2f}" if ratio else ""
            print(f"  {mode:18s} {totals[mode]:10.1f} us{rel_}{acc}", flush=True)
        for entry in result["largest"]:
            print(f"  largest [{entry['m']}, {entry['k']}] @ [{entry['k']}, {entry['n']}] ({entry['kind']}): " +
                  ", ".join(f"{m} {u:.1f}" for m, u in entry["us"].items()), flush=True)
        del runners, products
        torch.cuda.empty_cache()
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


if __name__ == "__main__":
    main()
