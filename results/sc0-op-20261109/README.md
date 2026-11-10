# SC_GENERATED: interaction 0's division and symmetric contraction as generated kernels (9 November 2026)

Codex report v3 proposal 5 (`FUSED_LAYER0_SC`), engine integration. The bound and the lab are in
[sc0-bound-20261109](../sc0-bound-20261109/README.md): on the H200 the four interaction-0 kernels (division,
cuEq `uniform_1d` contraction, its backward, the division's backward) take 179.5 µs, and the generated straight-line
pair takes 82.4 µs.

Branch `fastmace/sc0-op` (from main a6430eb6). The integration is in four files:

- `fastmace/kernels/cuda/sc_generated.py` renders the contraction's merged terms (480 terms, 410 monomials, 61 weight × output
  groups for OMAT-0 medium) as one forward (1 / (D + 1) fused) and one backward (dm, dD) NVRTC kernel. The custom
  ops are `fastmace::sc_generated_forward` and `_backward`.
- `fastmace/rewrites/sc_generated.py` holds `SC_GENERATED`, `SC_GENERATED_GPUS` and `route`. When routed,
  interaction 0's compose_skip branch returns the undivided message and passes `(density,)` in the product's `sc`
  slot. The density kind leaves that slot None. Product 0's subclass runs the generated kernels. Any other `sc` runs
  the original forward.
- `RewriteConfig.sc_generated` (auto = with `tf32x3_gemms` on the listed GPUs) is routed by `rewrites.precision`.
- Checks: `tests/gpu/test_sc_generated_cuda.py`, `tools/sc_generated_check.py` and
  `tools/modal/sc_generated_job.sh`.

## H200 ([h200/](h200/run.log))

- Machine and environment: Modal H200 (`NVIDIA H200`, driver 610.57.04, sm_90, CUDA UMD 13.3), profile `sixth`, the
  Modal hybrid image (torch 2.14.1+cu130).
- Source: FastMACE `5c5ade67`, slim tarball.
- Model and precision: OMAT-0 medium (SHA-256 `d4b14be9…`), hybrid backend, float32.
- Workload: Benchmark_loup, SrTiO3 10×10×10 (5,000 atoms), Langevin 300 K, 1 fs, skin 0.5.
- Side a is the H200 defaults of the source; side b adds `fastmace.rewrites.sc_generated:SC_GENERATED=True`.
- Wall time 453 s, about $0.57.

| Step | Result |
| --- | --- |
| GPU test (`tests/gpu/test_sc_generated_cuda.py`) | 4 passed. Values and gradients match cuEq on the divided features (unprojected and projected weight). Any other `sc` keeps the original forward, bit for bit. The compiled forward matches eager. |
| Engine check (`tools/sc_generated_check.py`) | Product 0 is routed and interaction 0 is marked. Forward 80 registers, backward 96, no local memory. On a copy rattled by 0.05 Å, forces against SC_GENERATED off differ by up to 3.6e-6 eV/Å (same build twice: 2.9e-6). Energy differs by 4.0e-5 eV (same build: 4.0e-5). |
| Force/stress gate (`tools/force_stress_gate.py`, limit 3) | **passes**. Ratios (SrTiO3 3×3×3 / NaCl / NbCl5): forces 0.850 / 0.920 / 0.959, stress 0.415 / 0.778 / 1.209. Energy: SrTiO3 1.8 meV (under the 0.01 eV floor), NaCl 0.63, NbCl5 1.75. |
| MD step A/B, order ab (ABAB, 6 × 256 steps) | a 3.4795 ms, b 3.3959 ms, **b/a 0.9760**, valid |
| MD step A/B, order ba (ABBA) | b 3.3712 ms, a 3.4645 ms, **b/a 0.9731**, valid |

The step got 84–93 µs faster, against 97 µs saved at kernel level in the lab. It is faster in both orders and the
gate passes, so the switch is **adopted on the H200** under decision 66.

## Local RTX 5060 Ti ([local-5060ti/](local-5060ti/check.json))

The run went through `tools/ci/local_gpu.py run`, using WSL `mace_cueq_oeq` and the same source before the
adoption commit.

- GPU test: 4 passed.
- The engine check needed `COMPOSE_SKIP`, `COMPOSE_FAST` and `tf32x3_gemms=on`, which are not 5060 Ti defaults. It
  ran on SrTiO3 6×6×6 (1,080 atoms).
- Product 0 is routed, with 80 / 96 registers and no spills.
- Forces against off differ by 3.2e-6 eV/Å (same build: 3.0e-6).

## A100 ([a100/](a100/run.log))

- Machine and environment: Modal `NVIDIA A100 80GB PCIe` (driver 610.57.04, sm_80, CUDA UMD 13.3), profile `sixth`.
- Same source, image, model, workload and steps as the H200; a = the A100 defaults.
- Wall time 623 s, about $0.43.
- Alex was not reachable (csnhr refused connections for about 25 minutes), so the run went to Modal.

| Step | Result |
| --- | --- |
| GPU test | 4 passed |
| Engine check | Routed: forward 79 registers, backward 96, no local memory. Forces against off differ by 4.2e-6 eV/Å (same build: 2.4e-6). |
| Force/stress gate (limit 3) | **passes**. Forces 0.972 / 0.854 / 0.977, stress 0.701 / 0.898 / 1.237. Energy: SrTiO3 2.4 meV and NbCl5 0.5 meV (both under the 0.01 eV floor), NaCl 1.34. |
| MD step A/B, order ab | a 7.5005 ms, b 7.2643 ms, **b/a 0.9685**, valid |
| MD step A/B, order ba | b 7.2667 ms, a 7.5287 ms, **b/a 0.9652**, valid |

It is faster in both orders and the gate passes, so the switch is **adopted for the A100 family** (decision 66).
`SC_GENERATED_GPUS` lists `NVIDIA A100-SXM4-80GB` and `NVIDIA A100 80GB PCIe`, and `GPUModels` matches every A100
variant by family.
