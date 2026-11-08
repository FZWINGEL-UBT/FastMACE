import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import fastmace as fm
from fastmace.rewrites.radial import _GenericChain, R_LO, R_PHYSICAL

ckpt = "/home/fzwin/models/mace-omat-0-medium.model"
loaded = fm.load(ckpt, trusted=True)
model = loaded.model
inter0 = model.interactions[0]
inter1 = model.interactions[1]
species_cols = [7, 21, 37]  # O, Ti, Sr
pairs_6 = [(i, j) for i in range(3) for j in range(i, 3)]

r64 = torch.linspace(R_LO, float(model.r_max), 2000, dtype=torch.float64)[:-1]
r32 = r64.to(torch.float32)
r64 = r32.to(torch.float64)
phys = r64 >= R_PHYSICAL
phys15 = r64 >= 1.5

for idx, inter in [(0, inter0), (1, inter1)]:
    chain64 = _GenericChain(model, inter, species_cols, torch.float64, torch.device("cpu"))
    chain32 = _GenericChain(model, inter, species_cols, torch.float32, torch.device("cpu"))
    W_rad = chain64.folded[:, :64].double()  # [Q, 64]

    print(f"\n=== Interaction {idx} (Q={W_rad.shape[0]}) ===")
    # Evaluate exact and f32 network once per pair
    pair_data = []
    for s, t in pairs_6:
        with torch.no_grad():
            g64, dg64 = chain64.with_derivative(r64, s, t)
            g32, dg32 = chain32.with_derivative(r32, s, t)
        a64, da64 = g64[:, :64], dg64[:, :64]
        a32, da32 = g32[:, :64].double(), dg32[:, :64].double()
        w64, dw64 = a64 @ W_rad.T, da64 @ W_rad.T
        w32, dw32 = (a32.float() @ W_rad.float().T).double(), (da32.float() @ W_rad.float().T).double()
        pair_data.append((a64, da64, w64, dw64, w32, dw32))

    for region_name, mask in [("all [0.3, 6.0)", slice(None)), ("phys [1.0, 6.0)", phys), ("phys15 [1.5, 6.0)", phys15)]:
        print(f"  Region: {region_name}")
        for R in [8, 12, 16, 20, 24, 28, 32]:
            worst_val_ratio = 0.0
            worst_slp_ratio = 0.0
            max_abs_w_err = 0.0
            max_abs_dw_err = 0.0
            for a64, da64, w64, dw64, w32, dw32 in pair_data:
                am, dam = a64[mask], da64[mask]
                wm, dwm = w64[mask], dw64[mask]
                w32m, dw32m = w32[mask], dw32[mask]
                w_scale = wm.abs().amax(dim=0).clamp_min(1e-30)
                dw_scale = dwm.abs().amax(dim=0).clamp_min(1e-30)
                net_w_err = ((w32m - wm).abs() / w_scale).max().item()
                net_dw_err = ((dw32m - dwm).abs() / dw_scale).max().item()

                # SVD of [am; dam * h_char]
                X = torch.cat([am, dam * 0.05], dim=0)
                _, _, Vh = torch.linalg.svd(X, full_matrices=False)
                V_R = Vh[:R, :]  # [R, 64]
                # Project in float32:
                a_proj = (am @ V_R.T).float().double()
                da_proj = (dam @ V_R.T).float().double()
                W_proj = (V_R @ W_rad.T).float().double()
                w_hat = (a_proj.float() @ W_proj.float()).double()
                dw_hat = (da_proj.float() @ W_proj.float()).double()

                err_w = ((w_hat - wm).abs() / w_scale).max().item()
                err_dw = ((dw_hat - dwm).abs() / dw_scale).max().item()
                worst_val_ratio = max(worst_val_ratio, err_w / net_w_err)
                worst_slp_ratio = max(worst_slp_ratio, err_dw / net_dw_err)
                max_abs_w_err = max(max_abs_w_err, (w_hat - wm).abs().max().item())
                max_abs_dw_err = max(max_abs_dw_err, (dw_hat - dwm).abs().max().item())
            print(f"    R={R:2d}: val_ratio={worst_val_ratio:.3f}, slp_ratio={worst_slp_ratio:.3f}, max_dw_err={max_abs_dw_err:.3e}")
