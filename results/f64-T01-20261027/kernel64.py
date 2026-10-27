import torch

from fastmace.kernels.triton.radial_table import evaluate, radial_table_backward, radial_table_forward

torch.manual_seed(0)
P, K, T, J = 2, 6, 17, 65
for dtype in (torch.float32, torch.float64):
    coef = torch.randn(P, K, T, J, dtype=dtype, device="cuda")
    dcoef = torch.randn(P, K, T, J, dtype=dtype, device="cuda")
    r_lo, r_max = 0.3, 6.0
    starts = torch.linspace(r_lo, r_max, K + 1, dtype=dtype, device="cuda")[:-1]
    scales = torch.full((K,), 2.0 / ((r_max - r_lo) / K), dtype=dtype, device="cuda")
    n_bins = 8 * K
    bin_scale = n_bins / (r_max - r_lo)
    bins = (torch.arange(n_bins, device="cuda") // 8).to(torch.int32)
    r = (r_lo + (r_max - r_lo) * torch.rand(1000, dtype=dtype, device="cuda"))
    pair = torch.randint(0, P, (1000,), dtype=torch.int32, device="cuda")
    g, density = radial_table_forward(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max)
    ref = evaluate(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max)
    print(dtype, "forward", g.dtype, ((torch.cat([g, density], 1) - ref).abs().max() / ref.abs().max()).item())
    grad_g = torch.randn(1000, J - 1, dtype=dtype, device="cuda")
    grad_d = torch.randn(1000, 1, dtype=dtype, device="cuda")
    out = radial_table_backward(r, pair, dcoef, starts, scales, bins, grad_g, grad_d, r_lo, bin_scale, r_max)
    slopes = evaluate(r, pair, dcoef, starts, scales, bins, r_lo, bin_scale, r_max, derivative=True)
    refb = (torch.cat([grad_g, grad_d], 1) * slopes).sum(1)
    print(dtype, "backward", ((out.reshape(-1) - refb).abs().max() / refb.abs().max()).item())
