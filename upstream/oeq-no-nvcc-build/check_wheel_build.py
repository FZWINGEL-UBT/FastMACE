"""Check the OEq wheel-CUDA build flags: build OEq 0.7.0's host extension sources with them.

Runs in a fresh TORCH_EXTENSIONS_DIR, with CUDA_HOME/CUDA_PATH removed and PATH without any
nvcc, using only include_paths("cpu") plus wheel_build_flags(). Then loads the result.
"""
import importlib.util
import os
import sys
import sysconfig
import tempfile
import time

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "wheel_cuda", os.path.join(here, "oeq/openequivariance/openequivariance/_torch/extlib/wheel_cuda.py"))
wc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wc)

import pybind11  # noqa: E402
import torch  # noqa: E402
from torch.utils.cpp_extension import include_paths, library_paths, load  # noqa: E402

flags = wc.wheel_build_flags()
print("flags", flags)
assert flags is not None
inc, ld = flags
oeq = os.path.dirname(importlib.util.find_spec("openequivariance").origin)
root = os.path.dirname(oeq)
src = [f"{oeq}/extension/libtorch_tp_jit.cpp", f"{oeq}/extension/json11/json11.cpp"]
libdir = sysconfig.get_config_var("LIBDIR")
pyl = f"python{sys.version_info.major}.{sys.version_info.minor}"
ld = [f"-Wl,--no-as-needed,-rpath,{libdir}", f"-L{libdir}", f"-l{pyl}"] + ld + [
    "-Wl,-rpath," + library_paths("cpu")[0]]
os.environ["TORCH_EXTENSIONS_DIR"] = tempfile.mkdtemp(prefix="oeq-wheel-check-", dir=os.path.expanduser("~/tmp"))
t = time.time()
mod = load("libtorch_tp_jit", src, extra_cflags=["-O3", "-DCUDA_BACKEND"],
           extra_include_paths=[f"{oeq}/extension/backend", pybind11.get_include()] + include_paths("cpu") + inc,
           extra_ldflags=ld, verbose=False)
print("built", mod.__file__, round(time.time() - t, 1), "s")
torch.ops.load_library(mod.__file__)
print("loaded; DeviceProp ok:", hasattr(mod, "DeviceProp"))
os.system(f"ldd {mod.__file__} | grep -E 'cuda|nvrtc|cublas'")
