"""Run a tool script with module constants set first: with_constants.py mod:NAME=VALUE[,...] script args..."""
import importlib
import runpy
import sys

import fastmace  # noqa: F401  (soft file locks before torch)

for item in sys.argv[1].split(","):
    target, value = item.split("=", 1)
    module, name = target.split(":")
    setattr(importlib.import_module(module), name, eval(value))
    print(f"constant {target} = {value}", flush=True)
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
