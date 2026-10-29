import csv
import io
from collections import defaultdict

blocks, label = defaultdict(lambda: defaultdict(dict)), None
for line in open("ncu_cmp.txt"):
    if line.startswith("=="):
        label = line[3:].strip()
        continue
    if not line.startswith('"') or label is None:
        continue
    row = next(csv.reader(io.StringIO(line)))
    if len(row) < 15:
        continue
    kernel, metric, value = row[4][:40], row[12], row[14]
    blocks[label][kernel][metric] = value
short = {"gpu__time_duration.sum": "us", "lts__t_bytes.sum": "L2 MB", "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum": "L1 ld MB",
         "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "mem %",
         "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM %", "sm__warps_active.avg.pct_of_peak_sustained_active": "occ %",
         "launch__registers_per_thread": "regs", "launch__waves_per_multiprocessor": "waves", "launch__grid_size": "grid",
         "launch__block_size": "block", "smsp__inst_executed.sum": "Minst", "lts__t_requests_srcunit_tex_op_red.sum": "red k"}
for label, kernels in blocks.items():
    print("==", label)
    for kernel, metrics in kernels.items():
        parts = []
        for key, name in short.items():
            v = metrics.get(key)
            if v is None:
                continue
            try:
                f = float(v.replace(",", ""))
                if name in ("us",):
                    f /= 1e3
                if name in ("L2 MB", "L1 ld MB"):
                    f /= 1e6
                if name == "Minst":
                    f /= 1e6
                if name == "red k":
                    f /= 1e3
                parts.append(f"{name} {f:.4g}")
            except ValueError:
                parts.append(f"{name} {v}")
        print("  ", kernel.ljust(40), "; ".join(parts))
