"""Which CUDA libraries are mapped at each import stage, and where the extension's sonames resolve.

Usage: loaded_libs.py [EXTENSION_SO]
Shows whether torch has already loaded libcudart/libnvrtc/libcublas from the
cu13 wheel before OEq's extension is loaded (then the dynamic loader reuses them
by soname), and what a fresh dlopen of the extension would resolve to without
them (ld.so.cache: a system CUDA here, nothing on the workstations).
"""
import subprocess
from pathlib import Path

KEYS = ("libcudart", "libnvrtc.so", "libcublas.so", "libcublasLt")


def mapped(stage):
    found = sorted({line.split()[5] for line in Path("/proc/self/maps").read_text().splitlines()
                    if len(line.split()) >= 6 and any(k in line.split()[5] for k in KEYS)})
    print(f"{stage}: {found or 'none'}")


mapped("start")
import torch  # noqa: E402

mapped("after import torch")
torch.cuda.init()
torch.zeros(1, device="cuda")
mapped("after CUDA init")
torch.ones(4, 4, device="cuda") @ torch.ones(4, 4, device="cuda")
mapped("after a cuBLAS matmul")
print(subprocess.run(["sh", "-c", "ldconfig -p | grep -E 'libnvrtc.so.13|libcudart.so.13' || echo none"],
                     capture_output=True, text=True).stdout)
