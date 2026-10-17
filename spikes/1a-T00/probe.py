"""1a-T00 spike probe: build the OEq host extension and evaluate MH-1 hybrid forces.

Usage (inside the hybrid environment's Python):

    probe.py --mode MODE --cache CACHE --out RESULT_DIR [--checkpoint PATH]

MODE says how the OEq runtime is prepared before OpenEquivariance is imported:

- ``script``: nothing; the process was started by oeq_no_nvcc.sh.
- ``bare``: nothing; plain user environment (expected to fail or fall back).
- ``inproc``: candidate (a), ``_oeq_runtime.prepare_in_process`` with ctypes
  preloading of the cu13 libraries.
- ``inproc-nopreload``: candidate (a) without the ctypes preloading.
- ``reexec``: candidate (b), ``_oeq_runtime.prepare_by_reexec``.
- ``launched``: candidate (c), the process was started by the launcher
  ``_oeq_runtime`` ``__main__``; nothing is done here.

The probe loads MH-1 (head matpes_r2scan), converts it to the cuEquivariance/
OpenEquivariance hybrid in float32 as
Experiments/Benchmark_MACE_GPU_OPTIMIZATIONS/scripts/common.py load_calculator
(..., enable_oeq=True) does, evaluates forces on rattled NaCl 2x2x2 and writes
forces.npy and result.json with the evidence (environment, loaded libraries,
extension path, ldd, nvcc sentinel log).
"""
import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True,
                    choices=["script", "bare", "inproc", "inproc-nopreload", "reexec", "launched"])
parser.add_argument("--cache", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--set-driver-cache", action="store_true",
                    help="set CUDA_CACHE_PATH=CACHE/cuda-driver-cache in-process, before torch is imported")
parser.add_argument("--checkpoint", type=Path, default=Path.home() / "models" / "mace-mh-1.model")
args = parser.parse_args()

START = time.time()
ENV_AT_START = dict(os.environ)

if args.set_driver_cache:
    (args.cache / "cuda-driver-cache").mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_CACHE_PATH"] = str(args.cache / "cuda-driver-cache")

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))
if args.mode in ("inproc", "inproc-nopreload", "reexec"):
    from fastmace.kernels import _oeq_runtime
    if args.mode == "reexec":
        _oeq_runtime.prepare_by_reexec(args.cache)
    else:
        _oeq_runtime.prepare_in_process(args.cache, preload=args.mode == "inproc")

WATCHED = ["CUDA_HOME", "CUDA_PATH", "CPATH", "PATH", "LD_LIBRARY_PATH", "TORCH_EXTENSIONS_DIR",
           "OEQ_JIT_EXTENSION", "MAX_JOBS", "OEQ_NVCC_ATTEMPTS", "CUDACXX", "PYTORCH_NVCC", "CXX",
           "CUDA_CACHE_PATH", "CUDA_CACHE_DISABLE", "TMPDIR", "XDG_CACHE_HOME", "MACE_COMPILE_CACHE",
           "FASTMACE_OEQ_PREPARED"]
result = {"mode": args.mode, "pid": os.getpid(), "argv": sys.argv, "python": sys.executable,
          "env_at_start": {k: ENV_AT_START.get(k) for k in WATCHED},
          "env_before_oeq_import": {k: os.environ.get(k) for k in WATCHED}}

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.utils.cpp_extension as cpp_extension  # noqa: E402

result["torch"] = torch.__version__
result["cpp_extension_CUDA_HOME"] = cpp_extension.CUDA_HOME

t0 = time.time()
error = None
try:
    import openequivariance as oeq
    from openequivariance._torch import extlib
except Exception as exc:  # noqa: BLE001 - the probe records any failure
    error = repr(exc)
result["oeq_import_s"] = round(time.time() - t0, 2)
result["oeq_import_error"] = error
if error is None:
    result["BUILT_EXTENSION"] = getattr(oeq, "BUILT_EXTENSION", None)
    result["BUILT_EXTENSION_ERROR"] = repr(getattr(oeq, "BUILT_EXTENSION_ERROR", None))
    result["USE_PRECOMPILED_EXTENSION"] = extlib.USE_PRECOMPILED_EXTENSION
    try:
        so = oeq.torch_ext_so_path()
    except Exception as exc:  # noqa: BLE001
        so = f"error: {exc!r}"
    result["extension_so"] = so
    if so and os.path.isfile(so):
        result["ldd"] = subprocess.run(["ldd", so], capture_output=True, text=True).stdout.splitlines()
        build_ninja = Path(so).parent / "build.ninja"
        if build_ninja.is_file():
            result["build_ninja"] = [line for line in build_ninja.read_text().splitlines()
                                     if line.startswith(("cxx", "cflags", "ldflags", "cuda_"))]


def loaded_libraries():
    keys = ("libcudart", "libnvrtc", "libcublas", "libcuda.so", "libtorch_tp_jit")
    found = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 6 and any(k in parts[5] for k in keys):
            found.add(parts[5])
    return sorted(found)


forces = None
if result.get("BUILT_EXTENSION") is True:
    from ase.build import bulk

    @contextlib.contextmanager
    def hybrid_keeps_last_layer_irreps():
        # Copied from common._hybrid_keeps_last_layer_irreps (baseline tag).
        import mace.cli.convert_e3nn_hybrid as hybrid
        transfer, extract = hybrid.transfer_symmetric_contractions, hybrid.extract_config_mace_model
        keep = [False]

        def extract_config(model):
            config = extract(model)
            keep[0] = bool(config.get("keep_last_layer_irreps", False))
            return config

        def transfer_from_config(*a, keep_last_layer_irreps=False, **kw):
            return transfer(*a, keep_last_layer_irreps=keep[0], **kw)

        hybrid.transfer_symmetric_contractions, hybrid.extract_config_mace_model = (
            transfer_from_config, extract_config)
        try:
            yield
        finally:
            hybrid.transfer_symmetric_contractions, hybrid.extract_config_mace_model = transfer, extract

    from mace.calculators import mace_mp
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with hybrid_keeps_last_layer_irreps():
        calc = mace_mp(model=str(args.checkpoint), device="cuda", default_dtype="float32",
                       head="matpes_r2scan", enable_cueq=True, enable_oeq=True)
    result["calc_enable_cueq"] = bool(getattr(calc, "_enable_cueq", False))
    result["calc_enable_oeq"] = bool(getattr(calc, "_enable_oeq", False))
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    atoms.rattle(stdev=0.05, seed=17)
    atoms.calc = calc
    forces = atoms.get_forces()
    result["energy_eV"] = float(atoms.get_potential_energy())
    result["n_atoms"] = len(atoms)
    result["max_abs_force"] = float(np.abs(forces).max())
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "forces.npy", forces)
    np.save(args.out / "positions.npy", atoms.get_positions())

result["loaded_libraries"] = loaded_libraries()
attempts = os.environ.get("OEQ_NVCC_ATTEMPTS")
result["nvcc_attempts"] = (Path(attempts).read_text().splitlines()
                           if attempts and os.path.exists(attempts) else [])
driver_cache = os.environ.get("CUDA_CACHE_PATH")
result["cuda_driver_cache_files"] = (
    sorted(str(p.relative_to(driver_cache)) for p in Path(driver_cache).rglob("*"))[:20]
    if driver_cache and os.path.isdir(driver_cache) else None)
result["wall_s"] = round(time.time() - START, 1)
result["ok"] = forces is not None
args.out.mkdir(parents=True, exist_ok=True)
(args.out / "result.json").write_text(json.dumps(result, indent=1, default=str))
print(json.dumps({k: result.get(k) for k in ("mode", "ok", "BUILT_EXTENSION", "oeq_import_s",
                                              "oeq_import_error", "extension_so", "wall_s")}))
print("BUILT_EXTENSION_ERROR", result.get("BUILT_EXTENSION_ERROR"))
