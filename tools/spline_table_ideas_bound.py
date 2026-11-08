"""CPU only: bounds for reading fewer spline-table bytes per edge in the SPLINE_W forward (8 November 2026).

On the H200 the forward spends about 45 % of its time reading the table (``tools/spline_table_bound.py``: 785 µs real,
433 µs with every edge on one slab; results/omat0-breakthroughs-assessment-20261108). Three directions, for
MACE-OMAT-0 medium on SrTiO3 (interaction 2, 9 ordered species pairs):

``certify``: the spline as production fits it (``rewrites.radial.spline.fit`` with the interaction's folded last layer,
CPU, float64 chain), then the same certificate (``_Scorer``: per pair and column, error over the column's largest value,
divided by the float32 network's own error; values and r-slopes; passes at <= 1) for
- compressed tables: coefficients c1..c3 (or c2, c3, or all) rounded to float16 / bfloat16, c0 in float32; bytes per
  coefficient set 16 -> 12, 10 or 8;
- per-pair grids: each pair's own worst score, and the coarsening factor that would bring it to the production target
  (slope error ~ h^3), i.e. how many fewer intervals a per-pair fit could use.

``reuse``: an L1 model of the forward's slab reads. SrTiO3 10x10x10 with Gaussian displacements (``--sigma``), the
production grid and the ordered-pair slabs; the H200 schedule as R128 runs it (one receiver per 128-thread block,
``--resident`` blocks per SM, blocks to SMs round-robin by block index, the resident blocks' edges interleaved) and an
LRU of ``--l1-slabs`` slabs (20 KB each) per SM. Orderings: natural atom order; Morton order; each SM a compact run of
the Morton order (a block -> receiver permutation); the same by (species, Morton); each with every receiver's edges
also sorted by slab. A grid coarsened per pair by the ``certify`` factors is modelled the same way.

    python tools/spline_table_ideas_bound.py CHECKPOINT OUTPUT.json [--skip-certify]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastmace.rewrites import radial as rt  # noqa: E402
from fastmace.rewrites.radial import spline as sp  # noqa: E402

SPECIES = (38, 22, 8)  # Sr, Ti, O
STRUCTURE = "results/loup-a100-20261103/SrTiO3.cif"
SM, BYTES_PER_SLAB = 132, 1280 * 4 * 4


def quantise(coef, which, dtype):
    out = coef.clone()
    for k in which:
        out[:, :, k] = coef[:, :, k].to(dtype).to(torch.float32)
    return out


def certify(model, interaction_index):
    numbers = [int(z) for z in model.atomic_numbers]
    columns = [numbers.index(z) for z in SPECIES]
    pairs = [(s, t) for s in range(len(columns)) for t in range(len(columns))]
    interaction = model.interactions[interaction_index]
    chain64 = rt._GenericChain(model, interaction, columns, torch.float64, torch.device("cpu"))
    chain32 = rt._GenericChain(model, interaction, columns, torch.float32, torch.device("cpu"))
    weight = chain64.folded  # [W, 0]: the chain's [h c, c] columns
    r_max = float(model.r_max)
    started = time.time()
    spline, cert = sp.fit(chain64, chain32, weight, pairs, rt.R_LO, r_max, rt.CERTIFY_POINTS, rt.R_PHYSICAL,
                          seed=interaction_index)
    print(f"fit: {cert['intervals']} intervals, certificate {cert['worst']:.3f} ({time.time() - started:.0f} s)",
          flush=True)
    scorer = sp._Scorer(chain64, chain32, weight.double(), pairs, rt.R_LO, r_max, rt.CERTIFY_POINTS, rt.R_PHYSICAL,
                        interaction_index)
    coef, grid = spline["coef"], spline["grid"]
    variants = {"float32 (production)": (coef, 16)}
    for name, dtype in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        variants[f"c2,c3 {name}"] = (quantise(coef, (2, 3), dtype), 12)
        variants[f"c1..c3 {name}"] = (quantise(coef, (1, 2, 3), dtype), 10)
        variants[f"all {name}"] = (quantise(coef, (0, 1, 2, 3), dtype), 8)
    compressed = {}
    for name, (c, size) in variants.items():
        relative = scorer.relative(c, grid)
        cc = scorer.certificate(relative)
        compressed[name] = dict(bytes_per_set=size, worst=cc["worst"], parts=cc["parts"])
        print(f"{name:22s} {size:2d} B  worst {cc['worst']:.3g}  {cc['parts']}", flush=True)
    relative = scorer.relative(coef, grid)
    per_pair = {}
    for k, (s, t) in enumerate(pairs):
        worst = max(v[scorer.regions[name]].max().item() for (kk, j, name), v in relative.items() if kk == k)
        bands = []
        for a, b, n in grid:
            inside = (scorer.r64 >= a) & (scorer.r64 < b)
            w = max((v[inside & scorer.regions[name]].max().item()
                     for (kk, j, name), v in relative.items() if kk == k and (inside & scorer.regions[name]).any()),
                    default=0.0)
            factor = max(1.0, (sp.TARGET / max(w, 1e-30)) ** (1 / 3))  # slope error ~ h^3
            bands.append(dict(start=a, end=b, intervals=n, score=w, coarsen=factor,
                              intervals_at_target=max(1, math.ceil(n / factor))))
        per_pair[f"{SPECIES[s]}-{SPECIES[t]}"] = dict(worst=worst, bands=bands,
                                                      intervals_at_target=sum(x["intervals_at_target"] for x in bands))
        print(f"pair {SPECIES[s]}-{SPECIES[t]}: worst {worst:.3f}, intervals at target "
              f"{per_pair[f'{SPECIES[s]}-{SPECIES[t]}']['intervals_at_target']} of {cert['intervals']}", flush=True)
    return dict(intervals=cert["intervals"], certificate=cert["worst"], grid=grid, compressed=compressed,
                per_pair=per_pair)


# ------------------------------------------------------------------------------------------------------ reuse model
def structure(sigma, seed):
    import numpy as np
    from ase.io import read
    from ase.neighborlist import neighbor_list

    atoms = read(STRUCTURE).repeat((10, 10, 10))
    rng = np.random.default_rng(seed)
    atoms.positions += rng.normal(0.0, sigma, atoms.positions.shape)
    i, j, d = neighbor_list("ijd", atoms, 6.0)
    return atoms, i, j, d


def slab_ids(numbers, i, j, d, grid, per_pair_factor=None):
    """Global slab index per edge: ordered pair (receiver species, sender species) x interval."""
    import numpy as np
    index = {z: k for k, z in enumerate(SPECIES)}
    si = np.array([index[int(z)] for z in numbers])
    pair = si[i] * len(SPECIES) + si[j]
    iv, _, _ = sp.locate(torch.tensor(d, dtype=torch.float32), grid)
    iv = iv.numpy()
    if per_pair_factor is not None:  # a per-pair grid: band offsets kept, intervals merged by the pair's factor
        offsets, out = [], iv.copy()
        start = 0
        for a, b, n in grid:
            offsets.append((start, start + n))
            start += n
        for p, bands in per_pair_factor.items():
            mask = pair == p
            for (lo, hi), f in zip(offsets, bands):
                m = mask & (iv >= lo) & (iv < hi)
                out[m] = lo + np.floor((iv[m] - lo) / max(1.0, float(f))).astype(iv.dtype)
        iv = out
    return pair * 100000 + iv, si


def morton(positions, cell):
    import numpy as np
    frac = (positions @ np.linalg.inv(cell)) % 1.0
    q = np.minimum((frac * 1024).astype(np.int64), 1023)
    code = np.zeros(len(q), dtype=np.int64)
    for bit in range(10):
        for axis in range(3):
            code |= ((q[:, axis] >> bit) & 1) << (3 * bit + axis)
    return code


def simulate(order_blocks, edges_of, resident, capacity, compact):
    """Hits / reads of an LRU per SM. ``order_blocks``: receivers in block order. Round-robin: SM s runs blocks
    s, s + SM, ...; compact: SM s runs a contiguous run of ``order_blocks``. Resident blocks' edges interleave."""
    n = len(order_blocks)
    if compact:
        per = math.ceil(n / SM)
        runs = [order_blocks[s * per:(s + 1) * per] for s in range(SM)]
    else:
        runs = [order_blocks[s::SM] for s in range(SM)]
    hits = reads = 0
    for run in runs:
        cache = OrderedDict()
        for w in range(0, len(run), resident):
            lists = [edges_of[r] for r in run[w:w + resident]]
            for k in range(max(len(x) for x in lists)):
                for x in lists:
                    if k < len(x):
                        key = x[k]
                        reads += 1
                        if key in cache:
                            hits += 1
                            cache.move_to_end(key)
                        else:
                            cache[key] = None
                            if len(cache) > capacity:
                                cache.popitem(last=False)
    return hits / reads


def reuse(grid, per_pair_factor, sigma, seed, resident, capacities):
    import numpy as np
    atoms, i, j, d = structure(sigma, seed)
    order_edges = np.lexsort((j, i))  # receiver-sorted CSR, senders ascending
    i, j, d = i[order_edges], j[order_edges], d[order_edges]
    results = {}
    code = morton(atoms.positions, atoms.cell.array)
    for grid_name, factors in (("production grid", None), ("per-pair coarsened grid", per_pair_factor)):
        if grid_name != "production grid" and not factors:
            continue
        slab, si = slab_ids(atoms.numbers, i, j, d, grid, factors)
        bounds = np.searchsorted(i, np.arange(len(atoms) + 1))
        plain = [list(slab[bounds[a]:bounds[a + 1]]) for a in range(len(atoms))]
        by_slab = [sorted(x) for x in plain]
        touched = len(set(slab.tolist()))
        natural = list(range(len(atoms)))
        mort = list(np.argsort(code, kind="stable"))
        spec_mort = list(np.lexsort((code, si)))
        orders = {"natural": (natural, False), "morton": (mort, False), "morton, SM-compact": (mort, True),
                  "species+morton, SM-compact": (spec_mort, True)}
        entry = dict(touched_slabs=touched, touched_mb=touched * BYTES_PER_SLAB / 1e6, edges=int(len(slab)), hits={})
        for cap in capacities:
            for name, (order, compact) in orders.items():
                for edge_order, lists in (("csr", plain), ("slab-sorted", by_slab)):
                    h = simulate(order, lists, resident, cap, compact)
                    entry["hits"][f"L1 {cap} slabs | {name} | {edge_order}"] = h
                    print(f"{grid_name}: L1 {cap} slabs, {name:28s} {edge_order:11s} hit rate {h:.4f}", flush=True)
        results[grid_name] = entry
        print(f"{grid_name}: {touched} slabs touched ({entry['touched_mb']:.1f} MB)", flush=True)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("output", type=Path)
    p.add_argument("--interaction", type=int, default=1)
    p.add_argument("--sigma", type=float, default=0.05, help="Gaussian displacement per component, Angstrom")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resident", type=int, default=6, help="R128 blocks resident per SM")
    p.add_argument("--l1-slabs", type=int, nargs="*", default=[5, 10])
    args = p.parse_args(argv)
    torch.set_num_threads(8)
    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False).float()
    result = dict(tool="spline_table_ideas_bound", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  sigma=args.sigma, resident=args.resident)
    result["certify"] = cert = certify(model, args.interaction)
    factors = {}
    for k, (name, entry) in enumerate(cert["per_pair"].items()):
        factors[k] = [b["coarsen"] for b in entry["bands"]]
    result["reuse"] = reuse(cert["grid"], factors, args.sigma, args.seed, args.resident, args.l1_slabs)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
