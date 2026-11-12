"""PyTorch 2.15 trial (12 November 2026): compatibility probe of loupMD's stack, one check after another.

Run once per environment (2.14.1 production, 2.15 trial) on the same GPU. Each check is timed and records
"ok", or the exception and its last frames, and the probe goes on to the next check where that makes sense:

  import         torch, triton, cuequivariance(_torch), openequivariance, fastmace (+ versions, NVRTC/CUPTI)
  nvrtc          our NVRTC path (fastmace.kernels.oeq_streaming.runtime.compile_cubin) on a small kernel for
                 this GPU, loaded with cuModuleLoadData and launched on a torch tensor
  cutlass        each shipped CUTLASS cubin (tf32x3 sm_80/sm_90, hybrid) through cuModuleLoadData: loads on its
                 capability, CUDA_ERROR_NO_BINARY_FOR_GPU elsewhere (the loader path itself is exercised)
  compile        torch.compile (Inductor) of a small pointwise + reduction function against eager
  graph          CUDA-graph capture and replay of that compiled function
  force          one OMAT-0 medium float32 force + stress call (fm.ForceField(...).evaluate) on SrTiO3 3x3x3,
                 which builds the production plan: NVRTC w' kernels, Inductor, the CUDA-graph path

Usage: python compat.py OUTPUT.json  (PYTHONPATH=<loupMD checkout>, FASTMACE_MODELS set)
"""
import json
import os
import sys
import time
import traceback

out = {"checks": {}}


def check(name):
    def wrap(fn):
        t0 = time.perf_counter()
        try:
            value = fn()
            out["checks"][name] = {"status": "ok", "seconds": round(time.perf_counter() - t0, 3), "value": value}
        except BaseException as e:  # noqa: BLE001
            out["checks"][name] = {"status": "error", "seconds": round(time.perf_counter() - t0, 3),
                                   "error": f"{type(e).__name__}: {e}"[:3000],
                                   "frames": traceback.format_exc().splitlines()[-30:]}
        print(name, out["checks"][name]["status"], out["checks"][name]["seconds"], flush=True)
        return fn
    return wrap


@check("import")
def _import():
    import fastmace  # noqa: F401  (before torch)
    from fastmace.model.compile import configure_allocator

    configure_allocator()  # before any CUDA allocation, as a production process does (the simulation check needs it)
    import cuequivariance
    import cuequivariance_torch

    from fastmace import _patches
    from fastmace.model.loading import check_oeq

    record = _patches.prepare_oeq_runtime(os.environ["FASTMACE_CACHE"])  # as the planner does, before OEq imports
    import openequivariance
    import torch
    check_oeq()  # the hybrid's availability check (the OEq host extension built): HybridUnavailableError otherwise
    import triton
    from cuda.bindings import nvrtc

    v = nvrtc.nvrtcVersion()
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
            "triton": triton.__version__, "cuequivariance": cuequivariance.__version__,
            "cuequivariance_torch": cuequivariance_torch.__version__,
            "openequivariance": getattr(openequivariance, "__version__", "?"), "nvrtc": f"{v[1]}.{v[2]}",
            "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
            "cudnn": torch.backends.cudnn.version(), "oeq_runtime": record.detail if hasattr(record, "detail")
            else str(record)}


@check("nvrtc")
def _nvrtc():
    import numpy as np
    import torch
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming import runtime

    src = r'''extern "C" __global__ void axpy(float a, const float* x, float* y, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = a * x[i] + y[i]; }'''
    cubin = runtime.compile_cubin(src, torch.cuda.get_device_capability())
    torch.cuda.current_stream()
    module = runtime._check(driver.cuModuleLoadData(cubin))
    fn = runtime._check(driver.cuModuleGetFunction(module, b"axpy"))
    n = 4096
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.ones(n, device="cuda")
    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    import ctypes
    runtime._check(driver.cuLaunchKernel(fn, n // 256, 1, 1, 256, 1, 1, 0, stream,
                                         ((2.0, x.data_ptr(), y.data_ptr(), n),
                                          (ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)), 0))
    torch.cuda.synchronize()
    err = float((y - (2 * x + 1)).abs().max())
    assert err == 0.0, err
    return {"cubin_bytes": len(cubin), "max_error": err, "np": np.__version__}


@check("cutlass")
def _cutlass():
    import torch
    from cuda.bindings import driver

    from fastmace.kernels.cutlass import tf32x3

    torch.cuda.current_stream()
    result = {}
    for kind, table in (("tf32x3", tf32x3.CUBINS), ("hybrid", tf32x3.HYBRID_CUBINS)):
        for cap, path in table.items():
            if not path.exists():
                result[f"{kind}-sm{cap[0]}{cap[1]}"] = "missing"
                continue
            err, module = driver.cuModuleLoadData(path.read_bytes())
            result[f"{kind}-sm{cap[0]}{cap[1]}"] = "loaded" if int(err) == 0 else str(err)
    result["available_here"] = bool(tf32x3.available(torch.device("cuda")))
    return result


def _fn(x, w):
    import torch
    h = torch.nn.functional.silu(x @ w) * x.sigmoid()
    return h.sum(dim=1), (h * h).mean()


@check("compile")
def _compile():
    import torch
    torch.manual_seed(0)
    x = torch.randn(4096, 256, device="cuda")
    w = torch.randn(256, 256, device="cuda") / 16
    f = torch.compile(_fn, dynamic=False)
    t0 = time.perf_counter()
    a = f(x, w)
    torch.cuda.synchronize()
    first = time.perf_counter() - t0
    b = _fn(x, w)
    return {"first_call_s": round(first, 3), "max_error": max(float((p - q).abs().max()) for p, q in zip(a, b))}


@check("graph")
def _graph():
    import torch
    torch.manual_seed(0)
    x = torch.randn(4096, 256, device="cuda")
    w = torch.randn(256, 256, device="cuda") / 16
    f = torch.compile(_fn, dynamic=False)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            f(x, w)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = f(x, w)
    x.copy_(torch.randn_like(x))
    g.replay()
    torch.cuda.synchronize()
    ref = _fn(x, w)
    return {"max_error": max(float((p - q).abs().max()) for p, q in zip(y, ref))}


@check("force")
def _force():
    import numpy as np
    import torch
    from ase import Atoms

    import fastmace as fm

    cell = Atoms("SrTiO3", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5),
                                             (0, 0.5, 0.5)], cell=[3.905] * 3, pbc=True).repeat((3, 3, 3))
    cell.rattle(0.03, seed=1)
    t0 = time.perf_counter()
    field = fm.ForceField(fm.load("omat0-medium"), cell, precision="float32")
    build = time.perf_counter() - t0
    t0 = time.perf_counter()
    energy, forces, stress = field.evaluate(cell, stress=True)
    torch.cuda.synchronize()
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    energy2, forces2, stress2 = field.evaluate(cell, stress=True)
    torch.cuda.synchronize()
    second = time.perf_counter() - t0
    forces = np.asarray(forces)
    np.save(os.environ.get("COMPAT_FORCES", "/dev/null"), forces)
    return {"build_s": round(build, 3), "first_eval_s": round(first, 3), "second_eval_s": round(second, 3),
            "energy": float(energy), "fmax": float(np.abs(forces).max()), "stress": np.asarray(stress).tolist(),
            "repeat_force_diff": float(np.abs(np.asarray(forces2) - forces).max()),
            "plan": getattr(getattr(field, "plan", None), "describe", lambda: None)()}


@check("simulation")
def _simulation():
    """The planner's production path (fm.Simulation, fallbacks refused, so a broken tier raises instead of quietly
    becoming the reference tier): build, 20 NVT steps, the tier and engine it chose."""
    import torch
    from ase import Atoms

    import fastmace as fm
    from fastmace.config import FallbackConfig

    cell = Atoms("SrTiO3", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5),
                                             (0, 0.5, 0.5)], cell=[3.905] * 3, pbc=True).repeat((6, 6, 6))
    field = fm.ForceField(fm.load("omat0-medium"), cell, precision="float32")
    t0 = time.perf_counter()
    sim = fm.Simulation(cell, field, seed=1, fallback=FallbackConfig(fallback="refuse"),
                        integrator=fm.integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.002))
    build = time.perf_counter() - t0
    sim.run(steps=20)
    torch.cuda.synchronize()
    text = sim.plan.describe()
    assert "tier reference" not in text, "planner fell back to the reference tier"
    return {"build_s": round(build, 2), "engine": type(sim._engine).__name__, "plan_head": text.splitlines()[:3]}


with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=1, default=str)
print("compat done", json.dumps({k: v["status"] for k, v in out["checks"].items()}))
