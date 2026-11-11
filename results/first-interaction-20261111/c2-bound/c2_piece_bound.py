import glob, torch, math, json, sys
found = None
for p in sorted(glob.glob('/home/fzwin/.cache/fastmace/radial-tables/*.pt'), key=lambda p: -__import__('os').path.getmtime(p)):
    try:
        e = torch.load(p, map_location='cpu', weights_only=True)
    except Exception:
        continue
    if 'tables' not in e or 'coef' not in e['tables']: continue
    c = e['tables']['coef']
    if c.dim() == 4 and c.shape[2] == 6 and c.shape[3] == 66:
        found = (p, e); break
p, e = found
t = e['tables']; coef = t['coef'].double(); dcoef = t['dcoef'].double(); scales = t['scales'].double()
print(p, tuple(coef.shape), 'certificate worst', e['certificate'].get('worst'), 'r', t['r_lo'], t['r_max'])
P, K, T, J = coef.shape
widths = 2.0 / scales
# column scales over the whole domain (lenient): max |value| ~ max over pieces of sum |c_k|
vscale = coef.abs().sum(2).amax(dim=1)  # [P, J]
dscale = dcoef.abs().sum(2).amax(dim=1)
def need(series, scale, keep, tol):
    # split a piece into m: Chebyshev term k shrinks ~ m^-k; truncating at degree keep-1 leaves sum_{k>=keep} |c_k| m^-k
    rel = series.abs() / scale[:, None, None, :].clamp_min(1e-300)  # [P, K, T, J]
    out = torch.full((K,), float('nan'), dtype=torch.float64)
    for m in range(1, 5000):
        err = sum(rel[:, :, k] / m ** k for k in range(keep, series.shape[2])).amax(dim=(0, 2))  # [K]
        out = torch.where(torch.isnan(out) & (err <= tol), torch.full_like(out, float(m)), out)
        if not torch.isnan(out).any():
            break
    return out
starts = t['starts'].double()
phys = starts >= 1.0
print('pieces below 1 A:', int((~phys).sum()), 'from 1 A:', int(phys.sum()))
for tol in (1e-7, 4.8e-7, 1e-6):
    mv2 = need(coef, vscale, 3, tol); md2 = need(dcoef, dscale, 3, tol); md3 = need(dcoef, dscale, 4, tol)
    for name, m in (("value quad", mv2), ("slope quad", md2), ("slope cubic", md3)):
        print(f"tol {tol:g} {name}: total {int(m.sum())}, below 1 A {int(m[~phys].sum())}, from 1 A {int(m[phys].sum())}, "
              f"max m from 1 A {int(m[phys].max())} at r {float(starts[phys][m[phys].argmax()]):.3f}")
print('current table MB', P * K * (T + T - 1) * J * 4 / 2**20)
