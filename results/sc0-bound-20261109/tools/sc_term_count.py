import cuequivariance as cue
from cuequivariance.group_theory.descriptors import symmetric_contraction
irr_in = cue.Irreps("O3", "128x0e+128x1o+128x2e+128x3o"); irr_out = cue.Irreps("O3", "128x0e+128x1o")
ep = symmetric_contraction(irr_in, irr_out, range(1, 4)); poly = ep.polynomial
print(poly)
for name, p in [("raw", poly), ("sym", poly.symmetrize_for_identical_operands())]:
    print(name, [(stp.num_operands, stp.num_paths) for _, stp in p.operations])
# minimal straight-line forward: per (out segment C, weight segment a) group S = sum coef * monomial; out += w_a * S
sym = poly.symmetrize_for_identical_operands()
paths = 0; groups = set(); monos = set(); pairs = set()
for op, stp in sym.operations:
    for path in stp.paths:
        idx = path.indices  # (a, b..., C)
        a, xs, c = idx[0], tuple(sorted(idx[1:-1])), idx[-1]
        nz = int((path.coefficients != 0).sum()) if hasattr(path.coefficients, "sum") else 1
        paths += nz
        groups.add((len(xs), a, c)); monos.add(xs)
        if len(xs) == 3: pairs.add(xs[:2])
print("segment shapes in", [stp.operands[1].segments[:3] for _, stp in sym.operations][:1])
print("nonzero coefficient terms", paths, "groups (a,C)", len(groups), "distinct monomials", len(monos), "pairs", len(pairs))
from collections import Counter
mc, mca, deg = set(), set(), Counter()
per_mc = Counter()
for op, stp in sym.operations:
    for path in stp.paths:
        idx = path.indices; a, xs, c = idx[0], tuple(sorted(idx[1:-1])), idx[-1]
        mc.add((xs, c)); mca.add((xs, c, a)); per_mc[(xs, c)] += 1; deg[len(xs)] += 1
d3 = {m for m in monos if len(m) == 3}; d2 = {m for m in monos if len(m) == 2}
print("terms by degree", dict(deg), "monomials deg2/deg3", len(d2), len(d3))
print("distinct (monomial, C)", len(mc), "distinct (monomial, C, a)", len(mca), "max a per (monomial,C)", max(per_mc.values()))
# Horner on the weight side: out_C = sum_a w_a * S_aC ; S_aC = sum coef*m.  vs  out_C = sum_m m * (sum_a coef w_a)
print("instr straight-line (monomial-shared): FMUL", len(d2|{m[:2] for m in d3}) + len(d3), "+ FFMA terms", sum(deg.values()), "+ FFMA groups", len(groups))
print("instr weight-first: per atom-channel combine sum_a coef*w_a for each (m,C):", len(mca), "FFMA, then", len(mc), "FFMA, + monomial FMULs")
