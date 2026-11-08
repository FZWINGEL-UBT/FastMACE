import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from ase.io import read
import fastmace as fm
from fastmace.rewrites.radial import _GenericChain, R_LO, R_PHYSICAL, spline as sp_mod

# Load SrTiO3 10x10x10
atoms = read("/mnt/d/Margraf/FastMACE/.claude/worktrees/omat0-breakthroughs-20261108/results/loup-a100-20261103/SrTiO3.cif") * (10, 10, 10)
rng = np.random.default_rng(20261007)
atoms.positions += rng.normal(0.0, 0.05, size=atoms.positions.shape)  # 300K thermal rattle

# Build neighbour list at cutoff 6.0
from matscipy.neighbours import neighbour_list
senders, receivers, unit_shifts = neighbour_list("ijS", pbc=(True, True, True), cell=atoms.cell, positions=atoms.positions, cutoff=6.0)
keep = ~((senders == receivers) & np.all(unit_shifts == 0, axis=1))
senders, receivers, unit_shifts = senders[keep], receivers[keep], unit_shifts[keep]

# Species: O=8 (0), Ti=22 (1), Sr=38 (2)
nums = atoms.get_atomic_numbers()
uniq, species = np.unique(nums, return_inverse=True)
print(f"Atoms: {len(atoms)}, active edges (<6.0A): {len(senders)}")

# Compute distances r
cell = np.asarray(atoms.cell, dtype=np.float64)
vecs = atoms.positions[receivers] - atoms.positions[senders] + unit_shifts @ cell
r = np.linalg.norm(vecs, axis=1).astype(np.float32)

# Spline grid from OMAT-0 medium
grid = [(0.3, 0.5, 157), (0.5, 0.75, 112), (0.75, 1.0, 87), (1.0, 1.5, 124), (1.5, 2.0, 54), (2.0, 2.5, 31), (2.5, 6.0, 166)]
r_t = torch.from_numpy(r)
iv, t, inv = sp_mod.locate(r_t, grid)
iv = iv.numpy()

# Symmetric pair index (6 pairs) and unsymmetric (9 pairs)
s_sp = species[senders]
r_sp = species[receivers]
pair_9 = s_sp * 3 + r_sp
pair_6 = np.minimum(s_sp, r_sp) * 3 + np.maximum(s_sp, r_sp)

# Order 1: Current FastMACE order: (receiver, sender_species, sender)
order_curr = np.lexsort((senders, s_sp, receivers))

# Order 2: Intra-receiver radial order: (receiver, sender_species, iv)
order_rad = np.lexsort((iv, s_sp, receivers))

# Order 3: Intra-receiver radial order without sender_species split (for Interaction 2): (receiver, pair_6, iv)
order_rad6 = np.lexsort((iv, pair_6, receivers))

for name, order, pairs in [
    ("Current (rcv, snd_sp, snd) + 9 pairs", order_curr, pair_9),
    ("Current (rcv, snd_sp, snd) + 6 sym pairs", order_curr, pair_6),
    ("Intra-rcv radial (rcv, snd_sp, r) + 6 sym pairs", order_rad, pair_6),
]:
    slabs = pairs[order] * 731 + iv[order]
    rcvs = receivers[order]
    # 1. In spline_forward0: sequential edge walk per receiver: how often does edge k have the SAME slab as edge k-1 (or within last 2 edges)?
    same_prev1 = np.sum((slabs[1:] == slabs[:-1]) & (rcvs[1:] == rcvs[:-1]))
    same_prev2 = np.sum(((slabs[2:] == slabs[1:-1]) | (slabs[2:] == slabs[:-2])) & (rcvs[2:] == rcvs[:-2]))
    # 2. In wprime_backward: 4 warps per block process 4 consecutive edges (e, e+1, e+2, e+3), and 32 edges per block:
    n4 = len(slabs) // 4
    s4 = slabs[:n4*4].reshape(n4, 4)
    uniq_per_4 = np.array([len(set(row)) for row in s4]).mean()
    n32 = len(slabs) // 32
    s32 = slabs[:n32*32].reshape(n32, 32)
    uniq_per_32 = np.array([len(set(row)) for row in s32]).mean()
    # Total unique slabs across entire structure
    total_uniq = len(np.unique(slabs))
    print(f"\n=== {name} ===")
    print(f"  Total unique slabs touched (each 20.48 KB): {total_uniq} ({total_uniq * 20.48 / 1024:.2f} MB)")
    print(f"  spline_forward sequential slab reuse (edge k == edge k-1): {same_prev1 / len(slabs) * 100:.1f}%")
    print(f"  wprime_backward unique slabs per 4-warp cohort (4 edges): {uniq_per_4:.2f} / 4.00 ({uniq_per_4 * 20.48:.1f} KB working set vs 32 KB L1)")
    print(f"  wprime_backward unique slabs per 32-edge CTA:             {uniq_per_32:.2f} / 32.00 ({uniq_per_32 * 20.48:.1f} KB)")

# Also check radial distribution of edges!
print("\n=== Edge Radial Distribution in SrTiO3 10x10x10 ===")
for r_a, r_b in [(1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.0), (5.0, 5.5), (5.5, 5.8), (5.8, 6.0)]:
    cnt = np.sum((r >= r_a) & (r < r_b))
    print(f"  r in [{r_a:.1f}, {r_b:.1f}) A: {cnt:6d} edges ({cnt / len(r) * 100:5.1f}%)")
