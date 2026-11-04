# OMAT-0 A100 profile at 4b70e3f2 (4 November 2026)

- Machine: Modal, NVIDIA A100-SXM4-80GB (driver 610.57.04); image recipe C hybrid (`pip-freeze.txt`).
- Source: `4b70e3f2` (tip of `fastmace/omat0-strided-tables`, merged into main as `9ed7f559`); model MACE-OMAT-0
  medium, float32, hybrid backend, default plan; Benchmark_loup SrTiO3 10×10×10 (5,000 atoms).
- Command: `modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 60 --commit 4b70e3f2
  --command "bash tools/modal/omat0_a100_profile.sh"`; wall 442 s, about $0.31 (`modal.json`).
- Benchmark protocol (`loup/`, one 20-step block per fresh process), µs/atom/step:
  NVE 2.726, 2.935, 2.967; Langevin 2.950, 2.794, 2.944. Against `results/omat0-a100-profile-20261103/`
  (3501a499): NVE 3.64, Langevin 3.59. The spread between processes is wider than on 3 November.
- Step breakdown (`breakdown/`, Langevin, skin 0.3 Å): force graph 13.69 ms per replay (18.43 on 3 November);
  `streaming_conv_wprime_backward` 4.37 ms, fused forward 3.70 ms, CUTLASS 3×TF32 GEMMs 1.85 ms.
- `sources.log`: the kernel-source step failed (`'Namespace' object has no attribute 'timestep_fs'`), the same
  `tools/module_profile.py` bug as on 3 November.
