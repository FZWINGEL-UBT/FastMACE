"""Pipeline stages of the tf32x3 Triton GEMMs on the Ada (conductor screen, 12 November 2026): same-process lab.

capacity_scan ``point`` on the production plan; every launch of tf32x3_gemm's and species_gemm's Triton kernel is
recorded (launches outside a CUDA-graph capture keep cloned inputs; launches inside a capture are counted, which
gives each signature's launches per replayed step). After the point's runs, every recorded launch is replayed alone
with num_stages 3 and 2 (production: tf32x3_gemm 3 via tile_rule.NUM_STAGES, species_gemm 2), in ROUNDS rounds
of ABBA order, each round timing one step's worth of launches per family with CUDA events.

Output STAGES_OUT/stages-<STAGES_TAG>.json: per family and stage count, µs per step (median over rounds), the
3-stage / 2-stage ratio, and the signatures (grid, shapes, launches per step).

Usage: the capacity_scan point arguments; env STAGES_OUT, STAGES_TAG, ROUNDS (20), MEMORY_FRACTION.
"""
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ["FM_SRC"]) / "tools" / "bench"))
import capacity_scan as cs  # noqa: E402

import fastmace as fm  # noqa: E402
import fastmace.model.compile as compile_  # noqa: E402

OUT = Path(os.environ["STAGES_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
TAG = os.environ.get("STAGES_TAG", "x")
ROUNDS = int(os.environ.get("ROUNDS", "20"))

_configure = compile_.configure_allocator


def configure_allocator(*args, **kwargs):
    result = _configure(*args, **kwargs)
    import torch

    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMORY_FRACTION", "0.9")))
    return result


compile_.configure_allocator = configure_allocator

import torch  # noqa: E402

from fastmace.kernels.triton import species_gemm, tf32x3_gemm  # noqa: E402

records = {}  # signature -> {"family", "kernel", "grid", "args", "kwargs", "captured": n, "eager": n}


def _sig(family, grid, args, kwargs):
    parts = [family, tuple(grid)]
    for a in args:
        parts.append((tuple(a.shape), tuple(a.stride()), str(a.dtype)) if isinstance(a, torch.Tensor) else a)
    parts.append(tuple(sorted((k, v) for k, v in kwargs.items())))
    return repr(parts)


class Recorder:
    def __init__(self, family, kernel):
        self.family, self.kernel = family, kernel

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            key = _sig(self.family, grid, args, kwargs)
            rec = records.setdefault(key, {"family": self.family, "kernel": self.kernel, "grid": tuple(grid),
                                           "args": None, "kwargs": dict(kwargs), "captured": 0, "eager": 0})
            if torch.cuda.is_current_stream_capturing():
                rec["captured"] += 1
            else:
                rec["eager"] += 1
                if rec["args"] is None:
                    rec["args"] = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
            return launch(*args, **kwargs)

        return run


for module, family in ((tf32x3_gemm, "tf32x3_gemm"), (species_gemm, "species_gemm")):
    real = module._kernel

    def patched(real=real, family=family):
        return Recorder(family, real())

    module._kernel = patched


def lab():
    usable = [r for r in records.values() if r["captured"] and r["args"] is not None]
    missing = [r for r in records.values() if r["captured"] and r["args"] is None]
    info = {"tag": TAG, "rounds": ROUNDS, "signatures": len(records), "usable": len(usable),
            "captured_without_eager_sample": len(missing),
            "launches_per_step": {f: sum(r["captured"] for r in usable if r["family"] == f)
                                  for f in ("tf32x3_gemm", "species_gemm")},
            "production_stages": {"tf32x3_gemm": tf32x3_gemm.tile_rule.NUM_STAGES,
                                  "species_gemm": species_gemm.NUM_STAGES}}

    def step(family, stages):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for r in usable:
            if r["family"] != family:
                continue
            kwargs = dict(r["kwargs"], num_stages=stages)
            for _ in range(r["captured"]):
                r["kernel"][r["grid"]](*r["args"], **kwargs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000.0

    results = {}
    for family in ("tf32x3_gemm", "species_gemm"):
        times = {3: [], 2: []}
        for stages in (3, 2, 3, 2):  # compile and warm both
            step(family, stages)
        for i in range(ROUNDS):
            order = (3, 2, 2, 3) if i % 2 == 0 else (2, 3, 3, 2)
            for stages in order:
                times[stages].append(step(family, stages))
        m3, m2 = statistics.median(times[3]), statistics.median(times[2])
        results[family] = {"us_per_step_3": round(m3, 1), "us_per_step_2": round(m2, 1),
                           "ratio_2_over_3": round(m2 / m3, 4) if m3 else None,
                           "abba_half_ratios": [round(statistics.mean(times[2][j::2]) / statistics.mean(times[3][j::2]), 4)
                                                for j in (0, 1)] if m3 else None,
                           "spread_3": [round(min(times[3]), 1), round(max(times[3]), 1)],
                           "spread_2": [round(min(times[2]), 1), round(max(times[2]), 1)]}
    info["results"] = results
    info["signature_list"] = [{"family": r["family"], "grid": r["grid"], "captured": r["captured"],
                               "shapes": [tuple(a.shape) for a in r["args"] if isinstance(a, torch.Tensor)]}
                              for r in usable]
    (OUT / f"stages-{TAG}.json").write_text(json.dumps(info, indent=1, default=repr))
    print("stages", TAG, json.dumps({k: info[k] for k in ("launches_per_step", "production_stages", "usable",
                                                          "captured_without_eager_sample", "results")}), flush=True)


original_run = fm.Simulation.run
calls = {"n": 0}


def run(self, *args, **kwargs):
    result = original_run(self, *args, **kwargs)
    calls["n"] += 1
    if calls["n"] == 2:
        try:
            lab()
        except BaseException as e:  # noqa: BLE001
            import traceback
            print("stages FAILED", type(e).__name__, str(e)[:500], traceback.format_exc()[-1500:], flush=True)
    return result


fm.Simulation.run = run

if __name__ == "__main__":
    sys.exit(cs.main())
