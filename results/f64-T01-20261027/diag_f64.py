import copy
import sys

sys.path[:0] = ["tests/cpu", "tests"]
import torch  # noqa: E402

from fastmace.rewrites import radial  # noqa: E402
from test_radial_tables import _registry, PACKAGE, context, tiny  # noqa: E402

model, batch = tiny("mh1")
regs = _registry()
asked = {name: "on" for name in PACKAGE}
asked |= {name: "off" for name in regs.names() if name not in asked}
m = copy.deepcopy(model)
regs.apply(m, regs.resolve(m, asked, strict=False), context(batch), original=model)
interaction = m.interactions[0]
columns = list(range(len(m.atomic_numbers)))
pairs = [(s, t) for s in columns for t in columns][:1]
chain = radial._Chain(m, interaction, columns, torch.float64, torch.device("cpu"))
r_max = float(m.r_max)
print("r_max", r_max)
tables = radial.fit(chain, pairs, r_max, table_dtype=torch.float64)
table = radial._table_module(tables, columns, torch.device("cpu"))
print("pieces", table.coef.shape[1], "fit_error", tables["fit_error"])
starts = table.starts
widths = torch.diff(torch.cat([starts, torch.tensor([r_max], dtype=starts.dtype)]))
print("narrowest pieces at", starts[widths.argsort()[:8]].tolist())
r = torch.linspace(radial.R_LO, r_max - 1e-9, 40001, dtype=torch.float64)
pair = torch.zeros(r.shape, dtype=torch.int32)
tab, dtab = radial._table_values(table, r, pair)
truth, dtruth = chain.with_derivative(r, *pairs[0])
for name, a, b in (("value", tab, truth), ("slope", dtab, dtruth)):
    err = ((a - b).abs() / b.abs().amax(dim=0)).amax(dim=1)
    worst = int(err.argmax())
    print(name, f"max rel err {err.max():.2e} at r={float(r[worst]):.6f}")
    edges = torch.linspace(radial.R_LO, r_max, 9)
    for lo, hi in zip(edges[:-1].tolist(), edges[1:].tolist()):
        sel = (r >= lo) & (r < hi)
        print(f"  [{lo:.2f},{hi:.2f}) {err[sel].max():.2e}")
