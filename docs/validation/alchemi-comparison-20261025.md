# FastMACE 1.0.0 against NVIDIA ALCHEMI Toolkit 0.2.0: MD step time (26 October 2026)

Post-1.0 single-structure speed and accuracy comparison (`run_id`: `post1-alchemi-20261025`), closing the
unmeasured ALCHEMI row in `docs/design/2026-10-positioning.md`.

**Result:** on single-structure NVT Langevin MD (`OMAT-0 medium`, FP32, TF32 off, 1 fs, skin 0.3 Å) on the
reference RTX 5060 Ti (`sm_120`), **FastMACE 1.0.0 is 9.5–9.6× faster per steady-state MD step on `NaCl` (216
atoms: 1.65 ms vs 15.7–15.9 ms) and 2.75–2.77× faster per step on `NbCl5` (1,944 atoms: 9.90–9.92 ms vs
27.2–27.4 ms)**, with **10–14× shorter setup/compile time** (23–32 s vs 324–326 s) and **31 % lower peak GPU
memory on `NbCl5`** (1,018 MiB vs 1,474 MiB).

## 1. Setup and fairness

- **Hardware:** local reference PC (`Fabi-PC`), NVIDIA GeForce RTX 5060 Ti (`sm_120`, 16 GiB), WSL2 Ubuntu
  24.04, under a single quiet `tools/ci/local_gpu.py` reservation (`pid 8133`, 00:04–00:44 UTC, 26 October
  2026).
- **Shared environment (`~/environments/fastmace_alchemi_compare`):** both engines run in the exact same Python
  3.12.3 environment so PyTorch, CUDA, cuEquivariance and MACE versions are identical:
  - `torch==2.14.1+cu130`, `mace-torch==0.3.16`, `cuequivariance==0.13.0`,
    `cuequivariance-torch==0.13.0`, `cuequivariance-ops-torch-cu13==0.12.0`,
    `openequivariance==0.7.0`, `fastmace-oeq-ext==0.7.0.post1`, `fastmace==1.0.0`,
    `nvalchemi-toolkit==0.2.0`, `nvalchemi-toolkit-ops==0.5.0`, `warp-lang==1.18.0`,
    `ase==3.29.0`, `numpy==2.3.5`.
  - **Note on NVIDIA's optional MACE pin:** `nvalchemi-toolkit[mace]==0.2.0` pins `mace-torch==0.3.15`. We
    installed `nvalchemi-toolkit[cu13]==0.2.0` with `mace-torch==0.3.16` so both engines evaluate the exact
    same MACE 0.3.16 model implementation (`pip check` clean).
- **Model and structures:** `mace-omat-0-medium.model` (SHA-256
  `d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a`), $r_{\max} = 6.0$ Å, 3×3×3 r²SCAN
  supercells from `Structures/Benchmark/R2SCAN`:
  - `NaCl216`: 216 atoms, cubic $16.764$ Å cell;
  - `NbCl5_1944`: 1,944 atoms.
  - Identical initial positions and Maxwell–Boltzmann velocities at 300 K (`seed=0`, zeroed centre-of-mass
    momentum; input SHA-256 verified in every worker record).
- **Dynamics protocol:** NVT BAOAB Langevin at 300 K, $\Delta t = 1.0$ fs, friction $0.01\text{ fs}^{-1}$,
  Verlet neighbour skin $0.3$ Å (search radius $6.3$ Å), TF32 off (`allow_tf32 = False`,
  `float32_matmul_precision = "highest"`), no stress computation.
  - **FastMACE 1.0.0:** `fm.Simulation` default graph-tier plan (`backend="hybrid"`, `dtype="float32"`,
    float64 energy accumulation, CUDA graphs, exact generic MACE rewrites).
  - **ALCHEMI 0.2.0:** `MACEWrapper.from_checkpoint(..., device="cuda:0", enable_cueq=True, dtype=torch.float32, compile_model=True)`
    with `NeighborListHook(wrapper.model_config.neighbor_config, skin=0.3, stage=DynamicsStage.BEFORE_COMPUTE)`
    and `NVTLangevin`. Both vendor-recommended performance options (`enable_cueq=True` and
    `compile_model=True`) are enabled.
- **Timing protocol (`tools/bench/alchemi_compare.py`):**
  - Every timed run executes in a fresh child process.
  - Two alternating rounds cover both build/execution orders:
    - Round 1: `fastmace` then `alchemi` on `NaCl216`, followed by `fastmace` then `alchemi` on `NbCl5_1944`.
    - Round 2: `alchemi` then `fastmace` on `NaCl216`, followed by `alchemi` then `fastmace` on `NbCl5_1944`.
  - Each run performs 300 warm-up steps (`warmup_s` measured separately from `setup_s`), followed by 4 timed
    blocks of 300 steps (1,200 timed steps per run).
  - **Validity guards:** each timed block verifies that the integrator's committed step counter advanced by
    exactly 300 steps and that `torch._dynamo.utils.counters` recorded zero new frame captures or recompiles
    across the timed window (`compile_counter_delta_during_timing == {}`).

## 2. Static energy and force validation

Before timing, both engines were checked at step 0 against the uncompiled float64 `e3nn` reference
(`MACECalculator(..., default_dtype="float64", enable_cueq=False)`):

| Structure | Engine | Energy (eV) | Energy error vs FP64 (eV/atom) | Max force error (eV/Å) | RMS force error (eV/Å) |
| --- | --- | ---: | ---: | ---: | ---: |
| `NaCl216` | FP64 reference | −729.885165 | 0 | 0 | 0 |
| `NaCl216` | **FastMACE 1.0.0** | −729.885198 | **−1.52×10⁻⁷** | 1.54×10⁻⁶ | 5.15×10⁻⁷ |
| `NaCl216` | **ALCHEMI 0.2.0** | −729.886230 | −4.93×10⁻⁶ | 1.54×10⁻⁶ | 4.90×10⁻⁷ |
| `NbCl5_1944` | FP64 reference | −8600.953694 | 0 | 0 | 0 |
| `NbCl5_1944` | **FastMACE 1.0.0** | −8600.954070 | **−1.93×10⁻⁷** | 7.08×10⁻⁵ | 9.75×10⁻⁶ |
| `NbCl5_1944` | **ALCHEMI 0.2.0** | −8600.932617 | +1.08×10⁻⁵ | 7.62×10⁻⁵ | 9.78×10⁻⁶ |

- Both engines produce forces within normal float32 round-off of the float64 reference.
- FastMACE accumulates per-atom energies in float64, keeping total energy error 32–56× smaller than ALCHEMI's
  float32 energy reduction.

## 3. Steady-state MD step times and setup overhead

All 8 timing runs passed every validity guard (`valid: true`, 1,200/1,200 timed steps completed, zero Dynamo
captures/recompiles during timing).

| Case | Round & order | Engine | Block medians (ms/step) | 4-block times (ms/step) | Setup (s) | Warm-up 300 steps (s) | Peak CUDA memory (MiB) | Speed-up (ALCHEMI / FastMACE) |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `NaCl216` (216 atoms) | R1 (FM → AL) | **FastMACE 1.0.0** | **1.651** | 1.655, 1.646, 1.622, 1.661 | 23.3 | 0.61 | 315.9 | **9.53×** |
| `NaCl216` (216 atoms) | R1 (FM → AL) | ALCHEMI 0.2.0 | 15.724 | 15.300, 16.149, 16.789, 14.857 | 325.8 | 10.81 | 318.1 | 1.00× |
| `NaCl216` (216 atoms) | R2 (AL → FM) | ALCHEMI 0.2.0 | 15.895 | 15.323, 15.579, 17.902, 16.211 | 324.4 | 10.40 | 318.1 | 1.00× |
| `NaCl216` (216 atoms) | R2 (AL → FM) | **FastMACE 1.0.0** | **1.652** | 1.652, 1.644, 1.652, 1.654 | 23.9 | 0.60 | 315.9 | **9.62×** |
| `NbCl5_1944` (1,944 atoms) | R1 (FM → AL) | **FastMACE 1.0.0** | **9.901** | 9.901, 9.902, 9.896, 9.901 | 31.8 | 3.14 | 1,018.1 | **2.77×** |
| `NbCl5_1944` (1,944 atoms) | R1 (FM → AL) | ALCHEMI 0.2.0 | 27.399 | 27.164, 27.419, 27.573, 27.380 | 325.1 | 30.27 | 1,474.0 | 1.00× |
| `NbCl5_1944` (1,944 atoms) | R2 (AL → FM) | ALCHEMI 0.2.0 | 27.230 | 27.279, 27.133, 27.380, 27.181 | 323.9 | 13.89 | 1,474.0 | 1.00× |
| `NbCl5_1944` (1,944 atoms) | R2 (AL → FM) | **FastMACE 1.0.0** | **9.921** | 9.902, 9.913, 9.932, 9.930 | 23.4 | 3.18 | 1,018.1 | **2.74×** |

## 4. Why FastMACE is faster on single trajectories (and where ALCHEMI differs)

1. **Whole-step CUDA graphs vs compiled per-step dispatch (`NaCl216`: 1.65 ms vs 15.7–15.9 ms, 9.5–9.6×).**
   - ALCHEMI compiles the model (`torch.compile`) and the neighbour-list hook (`mode="max-autotune-no-cudagraphs"`),
     and runs a Python step loop (`BEFORE_STEP`, `pre_update`, `BEFORE_COMPUTE`, `compute` with `detach()` and
     `copy_()`, `post_update`) on every step. On 216 atoms, kernel launch and Python/hook dispatch dominate
     ALCHEMI's 15.8 ms step, just as they did in TorchSim `reduce-overhead` (16.3 ms on the same cell and model).
   - FastMACE captures the entire integrator + force step into pipelined CUDA graphs (`ACP` and `B`), eliminating
     host launch overhead.
2. **Exact model rewrites + cuEq/OEq hybrid (`NbCl5_1944`: 9.90 ms vs 27.2–27.4 ms, 2.74–2.77×).**
   - On 1,944 atoms GPU compute dominates, and ALCHEMI's compiled cuEquivariance path is faster than TorchSim's
     (27.3 ms vs 39.4 ms), but FastMACE's receiver-streaming OEq convolution, reassociated first interaction,
     species-skip linear and block-linear rewrites cut the GPU execution time by a further 2.75× and reduce peak
     allocated memory from 1,474 MiB to 1,018 MiB.
3. **Process startup / compilation time (23–32 s vs 324–326 s).**
   - Even with warm Inductor/Triton/Warp caches on disk, ALCHEMI's `MACEWrapper(enable_cueq=True, compile_model=True)`
     spends ~325 s per fresh process tracing the cuEquivariance `autograd-in-graph` forward + backward graph in
     Dynamo. FastMACE traces the forward pass only (with OEq outside the autograd compile graph and
     `dynamic_edges=on`), starting in 23–32 s.
4. **API notes encountered in `nvalchemi-toolkit 0.2.0` with `mace-torch 0.3.16`:**
   - `MACEWrapper.from_checkpoint` passes `checkpoint_path` directly to `download_mace_mp_checkpoint`, which in
     MACE 0.3.16 passes non-preset strings to `urllib.request.urlopen`; a local checkpoint requires a `file://`
     URI (`checkpoint.resolve().as_uri()`).
   - `NeighborListHook` documents `stage=None` as defaulting to `DynamicsStage.BEFORE_COMPUTE`, but
     `register_hook` raises `TypeError` unless `stage=DynamicsStage.BEFORE_COMPUTE` is passed explicitly.
   - `BaseDynamics.compute` only writes `forces` and `energy` back into `Batch` when `AtomicData` pre-allocates
     `forces` and `energy` tensors.

## 5. Limits of this comparison

- This benchmark measures **one trajectory on one GPU** (FastMACE's target workload). It does not measure
  ALCHEMI's in-flight batching across many concurrent structures or multi-GPU domain decomposition, which
  FastMACE 1.0 does not implement.
- Tested on `OMAT-0 medium` (`mace-omat-0-medium.model`); MH-1 (`mace-mh-1.model`), where FastMACE's additional
  MH-1-specific rewrites (`factor_species`, `fold_cutoff`, `density_first`, `joint_radial_input`, `gated_rows`,
  `segment_gemm`, `species_tables`, `fold_readout_tail`, `fold_product_linear`) apply, was not timed here.
- Scripts and records: `tools/bench/alchemi_compare.py` and `results/post1-alchemi-20261025/`.
