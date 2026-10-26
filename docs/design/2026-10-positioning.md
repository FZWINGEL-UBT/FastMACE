# FastMACE positioning against other GPU MLIP MD engines (October 2026)

Task 6-T09 (phase 6), answering the SPEC §18 risk "Positioning vs existing tools": re-check the competitors'
feature sets before 1.0 and decide whether to offer the force field as a backend to them. Written
25 October 2026. Feature claims for other tools come from their public pages, cited by URL. Speed claims come
only from our own measurements, which are labelled with hardware and versions. Vendor speed figures are given
as the vendor's own claims.

## 1. Tools compared

| Tool | Version checked | Kind |
| --- | --- | --- |
| **FastMACE** | main, phase 5 accepted (25 Oct 2026) | Python library: one structure, CUDA graphs, exact MACE rewrites, cuEq/OEq hybrid, planner |
| **TorchSim** (`torch-sim-atomistic`) | 0.6.2 (PyPI, 23 Sep 2026) | PyTorch-native batched MD/relaxation engine, many MLIPs |
| **MACE ML-IAP in LAMMPS** (`mace-torch` + `pair_style mliap unified`) | mace-torch 0.3.15/0.3.16 | MACE's own LAMMPS path, Kokkos, cuEq, multi-GPU |
| **mace-torch ASE calculator** (`MACECalculator`) | 0.3.16 | MACE's reference calculator, driven by ASE |
| **Symmetrix / Symmetrix-XL** | Symmetrix-XL 0.1.1 | C++/Kokkos MACE evaluator, LAMMPS pair style plus ASE path |
| **NVIDIA ALCHEMI Toolkit** (`nvalchemi-toolkit`) | 0.2.0 (PyPI, 7 Aug 2026) | batched GPU MD/relaxation framework, MACE among its models |
| **mace-million** | research code (arXiv 2610.23371) | memory-lean exact MACE evaluator in LAMMPS ML-IAP, million-atom MD |
| **Symmetrix-Polar** | research patch | PolarMACE layer for Symmetrix-XL |

Sources:
- TorchSim: <https://pypi.org/project/torch-sim-atomistic/>, <https://torchsim.github.io/torch-sim/>,
  <https://github.com/TorchSim/torch-sim/releases>, paper <https://arxiv.org/abs/2508.06628>.
- MACE ML-IAP: <https://mace-docs.readthedocs.io/en/latest/guide/lammps_mliap.html>,
  <https://github.com/ACEsuit/mace/releases>.
- Symmetrix(-XL): <https://github.com/wcwitt/symmetrix>, <https://github.com/amphilagus/symmetrix-xl>,
  conda recipe <https://github.com/conda-forge/staged-recipes/pull/35010>.
- ALCHEMI: <https://github.com/NVIDIA/nvalchemi-toolkit>,
  <https://nvidia.github.io/nvalchemi-toolkit/userguide/about/intro.html>.
- mace-million: <https://github.com/ShiQiaoL/mace-million>. Symmetrix-Polar: <https://github.com/Yong-Q/symmetrix-polar>.

## 2. Feature comparison

"Ours" marks measurements by this project. "Claim" marks a tool's own statement. "?" means not checked or not
documented.

| | Single-structure step speed (evidence) | Batching | Integrators | NPT | MC / species change | Restart | D3 | PolarMACE | Install needs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **FastMACE** | Ours, RTX 5060 Ti, 3×3×3 cells, FP32: MH-1 NaCl 2.90 ms, NbCl5 21.3 ms; OMAT-0 NaCl 1.62 ms, NbCl5 9.79 ms per 1 fs step (`docs/validation/torchsim-comparison-20261025.md`) | No (non-goal for 1.0, SPEC §1.2; one structure at a time) | NVE, Langevin, BAOAB, CSVR, Berendsen, Nosé–Hoover, MTK (`fastmace/integrators/`), relaxations, anneal/quench | Yes (MTK, Langevin barostat); qualified per SPEC | Yes: transactional species swaps (O↔Cl swap-MC) without recapture (SPEC §8.5) | Complete simulation state incl. RNG and exchanged species (SPEC §10.6); bit-exact with `restart_interval 0` | Yes, own GPU D3(BJ) in MD, relaxation and quench (decision 43) | Yes, CUDA-graph capturable k-space path, upstream bug fixes | `pip install`, no `nvcc`; NVIDIA only; extras `[oeq]`, `[polar]`, `[d3]`; Python 3.10–3.12 |
| **TorchSim 0.6.2** | Ours, same GPU and cells: 3.8–10× slower than FastMACE per single-structure step (best compile mode); 8 batched NaCl copies 4.7 ms per structure, still 2.9× FastMACE's single step | Yes, its core design: "evolving multiple systems concurrently" (paper); autobatching with memory management | NVE, NVT (Langevin, Nosé–Hoover chain), FIRE, gradient descent | Yes: Langevin and C-rescale variants (iso-, aniso-, triclinic), Nosé–Hoover NPT (docs) | Yes: `swap_mc_init`/`swap_mc_step`, batched and ragged systems (docs, 0.5.0 notes) | Trajectory reporting; state slicing and concatenation | Via models or ALCHEMI ops (not checked) | No MACE-POLAR wrapper seen; MACE's wrapper fails with OEq (ours) | `pip`, Python ≥ 3.12 (ours: did not install in our 3.10 env); many model extras |
| **MACE ML-IAP + LAMMPS** | Ours (Symmetrix study): Symmetrix-XL's README puts ML-IAP 5× slower per atom-step than Symmetrix-XL, which is 1.07–2.0× slower than FastMACE on our cells | Not in one evaluation (one system per LAMMPS run) | All of LAMMPS (`fix nve/nvt/npt/langevin`, many more) | Yes, LAMMPS barostats | Yes, LAMMPS `fix atom/swap`, `fix gcmc` (each change re-evaluates the model) | LAMMPS restart files | LAMMPS `pair_style dispersion/d3` combinable via `hybrid/overlay` (not checked with ML-IAP) | Not documented for ML-IAP; MH-1 supported since 0.3.15 | Build LAMMPS from source with Kokkos, ML-IAP, Python, MPI; docs call the interface "in beta testing" |
| **mace-torch ASE calculator** | Ours (research engine, SPEC §3): FastMACE 3–22× faster in NVT | No | All of ASE (CPU-side loop) | ASE NPT integrators | Any ASE/Python MC | ASE trajectories, no full-state restart | `torch-dftd` SumCalculator | Yes (reference implementation, with known bugs FastMACE fixes) | `pip install mace-torch`; the simplest |
| **Symmetrix-XL 0.1.1** | Ours (OMAT-0 medium, same GPU, native LAMMPS loop): 2.0× (NaCl), 1.07× (NaNbO3), 1.38× (NbCl5) slower than FastMACE; MH-1 not supported (MACE 0.3.15 notes); ASE path 1.9–2.5× slower (leto) | No | LAMMPS's, or ASE's | Via LAMMPS | Via LAMMPS | Via LAMMPS | Via LAMMPS | Only through the third-party Symmetrix-Polar patch | Prebuilt arch-specific wheels (e.g. sm_86, sm_89, sm_120), or a C++/Kokkos/CUDA build; conda recipe pending. Its edge: far lower memory, MPI multi-GPU, CPU and AMD |
| **NVIDIA ALCHEMI Toolkit 0.2.0** | Ours, same GPU and cells (`OMAT-0 medium`, `enable_cueq=True, compile_model=True`, FP32): NaCl 15.7–15.9 ms (9.5–9.6× slower than FastMACE's 1.65 ms), NbCl5 27.2–27.4 ms (2.74–2.77× slower than FastMACE's 9.90–9.92 ms); 324–326 s per-process compile setup (`docs/validation/alchemi-comparison-20261025.md`) | Yes, batch-first with in-flight batching | NVE, NVT, NPT (MTK) | Yes | Not documented | ? | Yes, DFT-D3(BJ) and Ewald/PME composable with MLIPs | No | `pip install nvalchemi-toolkit` |
| **mace-million** | Not measured; targets million-atom cells on one GPU | No | Via LAMMPS | Via LAMMPS | Via LAMMPS | Via LAMMPS | Via LAMMPS | No | Patched LAMMPS build plus Python env |

## 3. What is distinct about FastMACE

Stated without exaggeration:

1. **Fastest single-structure MACE MD step we have measured.** On our own GPUs and cells it beats NVIDIA ALCHEMI
   0.2.0 (2.7–9.6×), TorchSim 0.6.2 (3.8–10×), Symmetrix-XL's LAMMPS loop (1.07–2.0×) and stock ASE (3–22×).
   The margin is largest on small cells (hundreds of atoms), where launch overhead dominates. That is exactly the
   regime of long single trajectories such as anneal/quench. On 1,000–2,000 atom cells Symmetrix-XL is close.
2. **Exactness with self-checks.** Every rewrite is exact to rounding and checked against the original module
   at setup. An accuracy gate runs on the user's structure through the captured path, with safe fallback.
   No other tool listed here gates its fast path against MACE on the user's own input.
3. **Dynamic species inside a captured graph.** Transactional species swaps keep CUDA-graph speed. TorchSim
   also has swap MC, but at its slower per-step speed. LAMMPS `fix atom/swap` has no graph capture.
4. **PolarMACE and MH-1 at graph speed.** This holds with upstream bug fixes (MACE issue 1642, PR 1711/1762
   ports). Symmetrix-XL has no MH-1, and PolarMACE exists there only as a third-party patch. TorchSim did
   not run the OEq path.
5. **Complete, reproducible restarts and provenance:** RNG state, exchanged species, model hash, plan and
   versions.
6. **Pip install without a compiler.** Unlike LAMMPS-based paths, no LAMMPS/Kokkos build is needed.

## 4. Where others are ahead

- **Batched throughput:** TorchSim and ALCHEMI. For many small independent structures (screening,
  relaxations), they are the right tool. FastMACE has no batching before 2.x.
- **Scale:** LAMMPS ML-IAP, Symmetrix-XL and mace-million all offer multi-GPU domain decomposition, and
  Symmetrix-XL and mace-million use far less memory per atom. Million-atom cells are out of FastMACE's scope.
- **Model breadth:** TorchSim and ALCHEMI run many MLIP families (MatterSim, Fairchem/UMA, SevenNet, ORB,
  NequIP, AIMNet2...). FastMACE is MACE 0.3.x only.
- **Ecosystem and integrator breadth:** LAMMPS has decades of fixes, computes and analysis, PLUMED and
  enhanced sampling. TorchSim offers more barostat variants, phonopy, pymatgen I/O and differentiable
  simulation.
- **Hardware:** Symmetrix(-XL) runs on CPUs and AMD GPUs. FastMACE is NVIDIA only.
- **Python version:** TorchSim needs Python ≥ 3.12. FastMACE supports 3.10–3.12, which is an advantage on our
  workstations. It may become a liability if MACE upstream drops 3.10 (it already has on develop).
- **Maturity and community:** TorchSim has a paper and many contributors, and ALCHEMI has NVIDIA behind it.
  FastMACE is pre-1.0 with one owner.

## 5. Recommendations

### README pitch

- Lead with the niche: "the fastest way to run one long MACE trajectory on one NVIDIA GPU, exact to MACE
  within float32 round-off, checked on your structure".
- Quote measured numbers with hardware: "3.8–10× faster per step than TorchSim 0.6.2 and 1.1–2× faster than
  Symmetrix-XL in LAMMPS on 216–1,944-atom cells, RTX 5060 Ti". Link the validation records. Do not claim
  generality beyond the measured sizes and GPUs.
- Name the differentiators: species swaps at graph speed, MH-1 and PolarMACE, D3, full-state restarts,
  `pip install` without `nvcc`.
- Add a short "use something else when" section: many structures (TorchSim, ALCHEMI), >100k atoms or multi-GPU
  (LAMMPS ML-IAP, Symmetrix-XL), non-MACE models, AMD/CPU. Honest routing builds trust and avoids
  mismatched benchmarks.
- Update the stale README status (it still says 20 October, phase 2).

### Offer the force field as a backend?

- **TorchSim: yes, as a low-cost post-1.0 item.** A `FastMACEModel` implementing
  `torch_sim.models.ModelInterface` would give TorchSim users the rewritten model (exact rewrites, hybrid
  kernels) for single structures, and bring FastMACE visibility.
  - Caveats: TorchSim needs Python ≥ 3.12 (FastMACE must test on 3.12).
  - Our biggest gains come from CUDA graphs over the whole step, which a force-only backend loses. Expect
    model-level gains only, unless batch-size-1 graph capture is exposed.
  - Prototype it first and measure it, then decide.
- **ALCHEMI: later, only on demand.** It has the same model-wrapper pattern (`BaseModelMixin`) and the same
  caveats, and its user base is batch-first.
- **LAMMPS ML-IAP: not for 1.0.** The SPEC already lists it under 2.x. It would add a LAMMPS/Kokkos build
  matrix and multi-GPU ghost exchange. That competes with MACE's own path and with Symmetrix-XL on their home
  ground, where our per-step margin on large cells is small.
- **Symmetrix: no backend.** Instead, offer upstream the exact rewrites (species skip, reassociated first
  interaction) as ideas or patches where they apply.
- **Upstream MACE:** continue to report and port bug fixes. Contributing rewrites upstream (as MACE PR 1641's
  fused convolution shows) narrows the gap to the reference calculator. That is good for users and keeps
  FastMACE's value in graphs, the planner and dynamic species.

### Re-check

Repeat this comparison before 1.0 release. Measure TorchSim's NPT path and a Symmetrix-XL MH-1 release if one
appears, and add an ALCHEMI single-structure timing. Use the same cells, GPU and alternating-process protocol
as on 25 October.
