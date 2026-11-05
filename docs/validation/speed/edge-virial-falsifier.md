# Edge-virial stress falsifier (30 October 2026)

**Verdict: stop.**

- **Accuracy:** the edge virial `W = -Σ_e r_e ⊗ dE/dr_e` is exact. In float32 it is as accurate as the strain
  path.
- **Speed and memory:** there is nothing to gain. The force call without any stress is at most 1.4 % faster than
  the strain path's at 1,944 atoms, and 0.3 % at 9,216 atoms. Peak memory is the same to within 3 MiB.
- **Proposal:** the edge virial can only be slower than "no stress", so it is not proposed as the NPT default.
- **What was skipped:** no engine path was built, so there is no NPT A/B and no gate run.

## The bound

The strain path differentiates the energy by a symmetric strain ε of positions and cell. MACE's
`get_symmetric_displacement` builds:

- strained positions, `[N, 3] @ [3, 3]`;
- a strained cell;
- shifts `unit_shifts @ cell`, of size `[E, 3]`.

The single backward then carries the gradient through these to ε, in the same pass as the forces. An edge virial
drops the strain leaf. In its place it needs the gradient by the edge vectors, which the backward computes anyway
on its way to the positions, plus an `[E, 3] × [E, 3] → [3, 3]` reduction. So the force call **without stress** is
a lower bound on the edge virial's time and memory, and (strain − none) is the most it can save.

`results/edge-virial-20261030/stress_cost.py` measures that bound:

- Two engines of the default plan (MinimalPlanner, hybrid float32) in one process: NPT, which has the strain path,
  and NVT, which has no stress.
- Their force fields' calls are timed in 8 alternating blocks of 20, on the same positions and edges. These are
  eager calls of the compiled model, not graph replays, so per-kernel launch costs count fully.
- The peak memory of one call is measured above what was allocated before it.

RTX 4000 Ada (leto), FastMACE 701feeaa, MH-1 r2SCAN head:

| Cell | Atoms | Edges | Strain path (ms) | No stress (ms) | none/strain | Call peak (MiB), strain / none |
|---|---|---|---|---|---|---|
| NaCl 3×3×3 | 216 | 6,912 | 7.879 | 7.542 | 0.957 | 83 / 83 |
| NaNbO3 3×3×3 | 1,080 | 83,808 | 15.938 | 15.698 | 0.985 | 690 / 689 |
| NbCl5 3×3×3 | 1,944 | – | 19.983 | 19.707 | 0.986 | 681 / 680 |
| NbCl5 4×4×8 | 9,216 | 242,176 | 91.489 | 91.253 | **0.997** | 3,187 / 3,184 |

- **Small cells:** NaCl's 4 % is the launch cost of the strain path's few extra small kernels in an eager call.
  In the NPT step these are graph replays, which cost a few microseconds each.
- **The brief's sizes:** at 1,944 and 9,216 atoms the ceiling is 1.4 % and 0.3 % of the force call. The NPT step
  also integrates the cell and rebuilds the neighbour list, so its share is lower still.
- **Memory:** the strain path keeps an `[E, 3]` and an `[N, 3]` tensor. That is 3 MiB at 242,176 edges, against
  the 3.1 GiB call peak.
- **The NbCl5 1,944-atom row:** its JSON entry was overwritten by the 9,216-atom cell, which had the same key
  (fixed since). Its numbers come from `stress-cost-run.log`.

## Accuracy

`results/edge-virial-20261030/edge_virial_accuracy.py` uses MACE's own edge path: `compute_edge_forces` and
`get_atomic_virials_stresses`, as LAMMPS uses them. It compares that with `compute_stress`.

Setup:
- e3nn modules on the CPU, in float32 and float64.
- The force/stress gate's configurations: 20 per structure, random full-cell strain up to 2 %, then a 0.05 Å
  Gaussian rattle, seed 20261030.
- leto's CPU with the GPU hidden.

Max |Δσ| over the 20 configurations and the nine components, in meV/Å³:

| Structure | Atoms | edge64 − strain64 | strain32 − strain64 | edge32 − strain64 | edge32 − strain32 |
|---|---|---|---|---|---|
| NaCl | 216 | 3.3e-14 | 7.2e-5 | 5.9e-5 | 1.4e-5 |
| NbCl5 | 1,944 | 1.0e-13 | 8.1e-5 | 6.8e-5 | 3.9e-5 |
| NaNbO3 | 1,080 | 2.8e-13 | 3.5e-4 | 2.4e-4 | 2.1e-4 |

- **Method:** in float64 the two stresses agree to rounding, so the method is exact for these models, whose
  energy depends on the positions only through the edge vectors.
- **float32:** the edge virial's error against float64 is at or below the strain path's own on every structure.
- **Forces:** they agree to within 5e-15 eV/Å in float64. In float32 the two paths' errors are equal.
- **Sign:** MACE's `atomic_virials` are −dE/dε per atom, so the stress is −Σ atomic_virials / V.

The models this would apply to keep the strain path anyway. PolarMACE's k-space term and D3 depend on the cell
directly, so they would have needed it regardless.

## What the brief asked and what ran

- **Accuracy on 20 configurations of NaCl, NbCl5 and NaNbO3:** done.
- **NPT step time and peak memory, A/B in both orders at 1,944 and 9,216 atoms:** replaced by the bound above.
  An engine path (an edge-vector gradient in the `wprime_backward` / `chunked_conv_backward` epilogue) would be
  needed for that A/B. Its best case is under 1.4 % of the force call, with no memory change, so it was not built.
  The sm_120 confirmation was not run for the same reason.

## Records

`results/edge-virial-20261030/`:
- `edge_virial_accuracy.py` and `stress_cost.py`;
- `leto/NaCl.json`, `leto/NbCl5.json` and `leto/NaNbO3.json` (accuracy, every configuration);
- `leto/stress_cost.json` and `leto/stress-cost-run.log` (the bound).

Environments on leto: NAS `mace_cueq_latest` for the CPU accuracy run, NAS `mace_cueq_oeq` (hybrid) for the GPU
bound. Source copy 701feeaa: the accuracy and bound scripts at that commit, with the library's code as on main
plus the tile order. The bound's cells are below the tile order's L2 threshold, except NbCl5 at 9,216 atoms, where
both engines ran the same order.
