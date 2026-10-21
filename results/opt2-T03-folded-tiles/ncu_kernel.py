"""OPT2-T03: the folded input-gradient kernel at the padded edge counts of NaCl, NbCl5 and NaNbO3 3x3x3, once
per tile rule, for ncu (``ncu --kernel-name kernel``): each rule's launch is the last of 3 (warm), and NVTX-free
ordering is fixed: for each structure, fixed then waves.

    ncu --kernel-name kernel --launch-skip 2 --launch-count ... python ncu_kernel.py
"""

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

# isort: split
import torch

from fastmace.kernels.triton import skinny_linear, tile_rule

EDGES = {"NaCl": 8192, "NbCl5": 56320, "NaNbO3": 82688}  # padded edges (OPT-T09)
generator = torch.Generator(device="cuda").manual_seed(0)
weight = torch.randn((1280, 65), device="cuda", generator=generator)
sms = tile_rule.device_sms("cuda")
for name, rows in EDGES.items():
    grad = torch.randn((rows, 1280), device="cuda", generator=generator)
    for rule in ("fixed", "waves"):
        print(name, rows, rule, skinny_linear.tile_choice(rows, sms, rule), flush=True)
        for _ in range(3):
            skinny_linear.folded_linear_input_grad(grad, weight, rule)
        torch.cuda.synchronize()
