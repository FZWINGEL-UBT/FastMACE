"""Where the launch fix's 6 % comes from: main's screen kernel against this tree's, with the marker store on and off
(the kernel source patched in this process only). NbCl5 6x6x20, 10 ABBA-style rounds, CUDA events."""

import importlib.util
import statistics
import sys

import fastmace  # noqa: F401

sys.path.insert(0, "tools/bench")
import numpy as np  # noqa: E402
import tiled_capacity  # noqa: E402
import torch  # noqa: E402

from fastmace.neighbors import pair_search as new  # noqa: E402

spec = importlib.util.spec_from_file_location("_old_ps", sys.argv[1] + "/fastmace/neighbors/pair_search.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

source = open(new.__file__).read().replace(
    "        if (tl.program_id(0) == 0) & (tl.program_id(1) == 0):  # the launch ran (``check_launched``)\n"
    "            tl.store(launched_ptr, 1)\n", "")
assert "tl.store(launched_ptr, 1)" not in source
source = source.replace("        check_launched(launched, natoms)" + chr(10), "")  # nothing sets the marker here
target = sys.argv[2] + "/_nomark_ps.py"
open(target, "w").write(source)
spec = importlib.util.spec_from_file_location("_nomark_ps", target)
nomark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nomark)

atoms = tiled_capacity.supercell("NbCl5", (6, 6, 20), 0.05, 0.0)
cell = torch.tensor(atoms.cell.array, dtype=torch.float64, device="cuda")
images = torch.tensor(np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij"), -1).reshape(-1, 3),
                      dtype=torch.float64, device="cuda")
vectors = images @ cell
wrapped = torch.tensor(atoms.get_positions(wrap=True), dtype=torch.float64, device="cuda")
sides = {name: (lambda m=m: m.screen_codes(wrapped, vectors, 6.3, {})) for name, m in
         (("main", old), ("new", new), ("new_nomark", nomark))}
for run in sides.values():
    run()
times = {name: [] for name in sides}
for _ in range(10):
    for name in list(sides) + list(reversed(list(sides))):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        sides[name]()
        stop.record()
        torch.cuda.synchronize()
        times[name].append(start.elapsed_time(stop))
base = statistics.median(times["main"])
for name, values in times.items():
    print(name, round(statistics.median(values), 2), "ms", round(statistics.median(values) / base, 4))
