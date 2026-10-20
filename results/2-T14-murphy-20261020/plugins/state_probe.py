"""2-T14: process state before and after the provenance_git simulation test."""
import os, sys
import fastmace  # noqa: F401
import torch
import pytest
sys.path[:0] = ["tests", "tests/cpu"]

def state():
    from fastmace import _patches
    s = {"threads": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
         "dtype": str(torch.get_default_dtype()), "patches": sorted(r.name for r in _patches.applied()),
         "profiling_executor": torch._C._jit_set_profiling_executor(True) if False else None,
         "mkldnn": torch.backends.mkldnn.enabled, "flush_denormal": None,
         "deterministic": torch.are_deterministic_algorithms_enabled(),
         "grad": torch.is_grad_enabled(), "mods": sorted(m for m in sys.modules if m.split(".")[0] in
                                                          ("mace", "cuequivariance_torch", "cuequivariance_ops_torch",
                                                           "openequivariance", "torch_dftd", "graph_longrange"))}
    s["env"] = dict(os.environ)
    return s

before = state()
code = pytest.main(["-p", "no:cacheprovider", "-q", "-p", "no:xdist", "--basetemp", sys.argv[1],
                    "tests/cpu/test_provenance_git.py::test_a_simulation_builds_the_block_once_across_ten_runs"])
after = state()
for k in before:
    if k == "env":
        diff = {v: (before["env"].get(v), after["env"].get(v)) for v in set(before["env"]) | set(after["env"])
                if before["env"].get(v) != after["env"].get(v)}
        print("ENV diff:", diff)
    elif k == "mods":
        print("new modules:", sorted(set(after[k]) - set(before[k]))[:60])
    elif before[k] != after[k]:
        print("CHANGED", k, before[k], "->", after[k])
print("pytest exit", code)
