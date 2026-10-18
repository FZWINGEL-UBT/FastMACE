# 1a-T13: CUDA-graph engine against the research engine

Plan task 1a-T13, 18 October 2026. FastMACE commit `452f687` (branch
`fastmace/1a-T13`), baseline ResearchModule tag
`research-engine-baseline-20261017` (commit `d2faa9fd`), exported with
`common.py` and the OEq launcher (`~/fastmace-baseline/1a-T18` on the PC,
`/mnt/homes/zwingel/fastmace-baseline/1a-T18` on the NAS). Test file:
`tests/gpu/test_engine_gpu.py`; JSON summaries in
`results/1a-T13-engine-20261018/`.

## Set-up

- MH-1 (`mh1-r2scan`, head `matpes_r2scan`), NaCl 3×3×3 (216 atoms),
  Maxwell–Boltzmann velocities at 300 K (NumPy seed 11), NVT 300 K,
  timestep 0.5 fs, friction 0.02, seed 20261018, skin 0.3 Å, GPU neighbour
  list, edge bucket 5 %. NPT: BAOAB, hydrostatic, 0 GPa, ASE's default
  barostat.
- Rewrites as main.py runs them (1a-T11's cases). Plain cuEq float64: the
  MH-1 package without reassociation, the linear family and the one-graph
  strain. The hybrid in float32 adds reassociation, the segmented first sum
  and streaming convolutions on receiver-sorted edges.
- Both sides run in fresh processes. The research engine runs
  `workflow.GPUEngine` exactly as `main.py` builds it. The hybrid baseline
  goes through `oeq_no_nvcc.sh`. FastMACE builds `GraphEngine` directly,
  through the tests' helper, as the planner will.
- Machines: the PC's RTX 5060 Ti (driver 610.88, WSL2,
  `~/environments/mace_cueq_latest` and `mace_cueq_oeq`), and leto's
  RTX 4000 Ada (driver 595.58.03, the NAS environments of the same names).
  Both use torch 2.14.1+cu130, MACE 0.3.16, cuEquivariance 0.12.0 and
  OEq 0.7.0.

## Results

Largest position difference over the recorded steps (every 50 steps), in Å.
PC numbers are from the full GPU suite in one process (216 passed, 2 skipped),
leto's from `tests/gpu/test_engine_gpu.py` alone (16 passed).

| Comparison | PC | leto |
| --- | --- | --- |
| **Plain cuEq float64, NVT** | | |
| captured replay vs eager, 200 steps (same padded sizes) | 5.0e-14 | 7.1e-14 |
| FastMACE vs FastMACE, repeat run, 500 steps | 2.5e-13 | 2.2e-13 |
| baseline vs baseline, repeat run, 500 steps | 3.2e-13 | 2.9e-13 |
| pipelined vs checked, 500 steps | 2.5e-13 | 2.9e-13 |
| speculative vs pipelined, 200 steps | 1.2e-13 | 7.1e-14 |
| **FastMACE vs baseline, checked, 500 steps** | **2.5e-13** | **2.9e-13** |
| **FastMACE vs baseline, pipelined, 500 steps** | **2.9e-13** | **2.5e-13** |
| **Plain cuEq float64, NPT (BAOAB, hydrostatic)** | | |
| FastMACE vs baseline, checked / pipelined, 500 steps | 8.9e-15 / 7.1e-15 | 5.3e-15 / 5.3e-15 |
| **Hybrid float32, NVT** | | |
| captured replay vs eager, 200 steps | 5.8e-8 | 4.2e-8 |
| FastMACE vs FastMACE, repeat run, 500 steps | 9.8e-8 | 1.0e-7 |
| pipelined vs checked, 500 steps | 9.3e-8 | 1.0e-7 |
| baseline vs baseline (two processes), 500 steps | 1.1e-7 | 7.8e-8 |
| **FastMACE vs baseline, 500 steps** | **8.7e-8** | **8.4e-8** |

The test limits are 1e-12 Å for float64. For the hybrid, the limit is
three times the baseline's own scatter (at least 1e-6 Å).

Both sides make the same number of neighbour rebuilds and use the same
padded edge sizes. Plain cuEq NVT has 15 builds in checked mode and 19 in
pipelined mode (18 early rebuilds), all at 7168 edges.

Every check of the tests holds on both machines:
- asynchronous output (`snapshot().wait()`) is bit-identical to `sync()`
  in every mode;
- capture warm-ups leave every runtime tensor unchanged, at the same
  address, for AC, C, B, ACP and ACB;
- the mixed-process check refuses a compiled plain-cuEq engine after a
  compiled hybrid one, and still allows an uncompiled one.

### Graph nodes

Node counts per graph, FastMACE / baseline:

| Graph | Plain NVT (PC) | Plain NVT (leto) | Plain NPT (PC) | Plain NPT (leto) | Hybrid NVT (PC) | Hybrid NVT (leto) |
| --- | --- | --- | --- | --- | --- | --- |
| AC | 64 / 64 | 64 / 64 | 229 / 229 | 229 / 229 | 64 / 64 | 64 / 64 |
| C | 23 / 23 | 23 / 23 | 28 / 28 | 28 / 28 | 23 / 23 | 23 / 23 |
| ACP | 73 / 73 | 73 / 73 | 354 / 354 | 354 / 354 | | |
| B (7168 edges) | 240 / 238 | 236 / 234 | 295 / 295 | 291 / 291 | 315 / 317 | 293 / 293 |

AC, C and ACP agree exactly. B differs by a few nodes in some runs. In
every run the kernel sequences are identical once cuEquivariance's choice
of kernel variant is taken out:

- **cuEq chooses kernel variants at run time.** Each segmented-polynomial
  operation launches one of several variants (`segmented_polynomial_algo1_4`,
  `…_algo2_2`, `…_algo1_1`, or a pair `…_split1_of_2_…`/`…_split2_of_2_…`).
  The choice differs between processes and between engines in one process.
- The research engine's own B graph had between 238 and 242 nodes in four runs
  on the PC.
- The test therefore compares node sequences by kernel name, with each cuEq
  segmented-polynomial launch (or split pair) counted as one node
  (`kernel_sequence` in the test).

This choice is also why plain cuEq float64 is **not bitwise reproducible
between engines**, not even between two runs of the research engine. The
variant fixes the order of the sums, so trajectories from one seed differ
by about 1e-13 Å after 500 steps. FastMACE against the baseline stays
inside that run-to-run scatter. Within one engine, replays are exact: the
tests' deterministic Lennard-Jones force field gives bitwise equality
between replayed and eager steps.

### Rough step times

These are a sanity check only, measured over the last 100 steps of a run,
across processes. Speed acceptance belongs to T19.

| Case | PC FastMACE / baseline (ms) | leto FastMACE / baseline (ms) |
| --- | --- | --- |
| plain cuEq float64 NVT, checked | 77.8 / 77.8 | 73.2 / 73.2 |
| plain cuEq float64 NVT, pipelined | 77.9 / 77.7 | 73.2 / 73.6 |
| plain cuEq float64 NPT, pipelined | 78.3 / 78.2 | 74.5 / 74.5 |
| hybrid float32 NVT, checked | 3.76 / 3.69–3.74 | 3.46 / 3.50 |
| hybrid float32 NVT, pipelined | 3.53 / – | 3.53 / – |

Float64 is slow on both GPUs (consumer-class FP64 throughput).
