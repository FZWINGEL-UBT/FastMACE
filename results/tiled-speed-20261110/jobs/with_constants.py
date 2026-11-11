"""Run a tool with module constants set first: with_constants.py MODULE:NAME=VALUE ... -- TOOL [ARGS...].

Values are Python literals (ast.literal_eval). The tool runs as __main__ (runpy), so its own argparse sees ARGS.
Used by the host-memory probe of the tiled speed switches (10 November 2026).
"""

import ast
import importlib
import runpy
import sys

split = sys.argv.index("--")
for item in sys.argv[1:split]:
    target, value = item.split("=", 1)
    module, name = target.split(":")
    mod = importlib.import_module(module)
    if not hasattr(mod, name):
        raise SystemExit(f"{module} has no {name}")
    setattr(mod, name, ast.literal_eval(value))
    print(f"set {module}.{name} = {getattr(mod, name)!r}", flush=True)

def _host_stats():
    """The pinned host allocator's statistics at exit (bytes in use, reserved and their peaks, allocations)."""
    try:
        import json

        import torch

        stats = torch.cuda.memory.host_memory_stats()
        keep = {k: v for k, v in stats.items() if any(s in k for s in ("bytes", "segment", "num_host"))}
        print("host_memory_stats " + json.dumps(keep, sort_keys=True), flush=True)
    except Exception as error:  # the probe's numbers only
        print(f"host_memory_stats unavailable: {error!r}", flush=True)


import atexit  # noqa: E402

atexit.register(_host_stats)
tool = sys.argv[split + 1]
sys.argv = [tool] + sys.argv[split + 2:]
sys.path.insert(0, tool.rsplit("/", 1)[0] if "/" in tool else ".")
runpy.run_path(tool, run_name="__main__")
