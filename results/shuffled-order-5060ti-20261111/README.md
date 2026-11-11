# Atom order: shuffled against ASE's repeat order, RTX 5060 Ti (11 November 2026)

Source: the conductor's B4 (11 November 2026). The question is whether arbitrary inputs need a spatial sort at setup.
The stop rule: if shuffled atoms are within 3 % of ordered ones, no sort is needed. If they are slower by more than
that, record the size and propose the sort as a ledger item, without building it.

- **Machine:** local PC, WSL2, NVIDIA GeForce RTX 5060 Ti 16 GB, driver 621.14, `~/environments/mace_cueq_oeq`.
- **Source:** FastMACE main `f31d4f50`.
- **Model and plan:** OMAT-0 medium, float32, the planner's default plan.
- **Structure:** SrTiO3 24×24×24 (69,120 atoms).
- **Runs:** `tools/bench/capacity_scan.py point`, one engine per process, NVT 300 K, 1 fs, warm-up 100 steps, 600 timed
  steps. [jobs/shuffled_point.py](jobs/shuffled_point.py) wraps it:
  - With `SHUFFLE=1` it permutes the supercell's atoms (numpy `default_rng(0)`) right after `Atoms.repeat`, before the
    velocities are drawn.
  - It caps the process at a CUDA memory fraction of 0.9.
- **Process order:** ordered, shuffled, shuffled, ordered, which covers both orders.
- **Guards:** one `tools/ci/local_gpu.py run` reservation, [jobs/job.sh](jobs/job.sh). The watchdog kills a process
  on any of these, and none fired:
  - process-tree RSS above 14.8 GiB;
  - card above 16,000 MiB;
  - a new soft-lockup or RCU-stall kernel line;
  - wall time above 3× the expected.
- **Peaks:** card 8,987 MiB (2,990 MiB of it Windows), RSS 11.4 GiB, steady reserved 5.57 GiB.

## Result

**Shuffled atoms are 16.7 % slower per MD step.** That is past the 3 % rule, so a setup sort goes into the ledger as a
proposal (`setup-spatial-sort`). Nothing is built.

| Run | Atom order | ms/step |
| --- | --- | ---: |
| o1 (first process) | ASE repeat order | 224.07 |
| s1 | shuffled | 259.27 |
| s2 | shuffled | 261.18 |
| o2 (last process) | ASE repeat order | 221.90 |

Shuffled / ordered: 260.2 / 223.0 = **1.167**. Pairwise, the ratios are 1.157 (s1 / o1) and 1.177 (s2 / o2).

Both orders build the same list size: 5,114,880 edges within the cutoff at step 0, and 6.256 M candidates within
cutoff plus skin. The two trajectories differ slightly, because the velocities are drawn after the permutation. The
gap is the memory layout: neighbours of a receiver are scattered across the atom arrays, so the per-edge gathers of
sender features and positions lose locality. The engine's receiver sort (`sort_receivers`) orders edges by receiver
but does not renumber the atoms (no spatial sort exists in `fastmace/` today).

## What a setup sort would be (proposal, not built)

Renumber the atoms once at setup in a space-filling order, for example Morton or Hilbert bins of the cell list. Keep
the permutation, and map positions, forces, velocities and per-atom outputs back at the API boundary. Inputs from
`ase.build` (repeat order) and MD restarts of those are already local, so the cost falls on inputs whose order
is not spatial, such as packmol boxes, structures concatenated per species, or randomly generated configurations.

Only this one size was measured. The gap will likely grow with the cell, as the arrays outgrow the L2 cache, but that
is not measured.
