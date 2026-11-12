"""wrap_frac.py FRACTION SCRIPT ARGS...: run SCRIPT with torch.cuda.set_per_process_memory_fraction(FRACTION) first,
so an allocation past the fraction raises instead of oversubscribing the card (WSL; copied from the tiled-speed lane)."""
import fastmace  # noqa: F401  (before torch: soft file locks)
import runpy
import sys

import torch

fraction = float(sys.argv[1])
torch.cuda.set_per_process_memory_fraction(fraction)
print(f"wrap_frac: memory fraction {fraction} of {torch.cuda.mem_get_info()[1] / 2**30:.2f} GiB", flush=True)
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
