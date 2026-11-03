# Benchmark_loup on the A100: why Symmetrix-XL's MD step was faster (3 November 2026)

**Background.** MargrafGroup's Benchmark_loup (private repository, commit `89ab3d8`) compared complete MD steps on
an NVIDIA A100 80GB PCIe:
- **Setup:** MACE-OMAT-0 medium, SrTiO3 10×10×10 (5,000 atoms), float32, skin 0.5 Å, 1 fs, from rest.
- **Protocol:** 5 warm-up and 20 timed steps, 3 fresh processes.
- **Result:** Symmetrix-XL under LAMMPS took 5.00–5.01 µs/atom/step. FastMACE's MD engine (`fm.Simulation`, commit
  `7430fb51`) took 5.35 (NVE) and 5.29 (Langevin).

The owner asked why, and this note answers it.

## Verdict

**FastMACE ran without its radial tables on that GPU.** Decision 66 turns `generic_radial_tables` on by exact device
name, and the A100 had never been measured, so the plan left it off. It also left `generic_chunked_conv_backward`
off; that choice was right, since the chunked backward is slower on the A100.

**With the radial tables on, FastMACE is about 10 % faster than the plan the benchmark ran:**
- Same-process A/Bs, both build orders:
  - SrTiO3 5,000 atoms on the A100 80GB PCIe: 0.902 / 0.901;
  - NaCl 8,000 atoms on an A100-SXM4-80GB: 0.917 / 0.917.
- Scaled to the benchmark's PCIe protocol time, that is about 4.8 µs/atom/step, against Symmetrix-XL's 5.00.
  - This is an estimate: 5.33 × 0.90, not measured under the protocol on the PCIe card.
  - The second job got an SXM4 card, where the protocol with the tables gave 4.50 (NVE) and 4.43 (Langevin).

## Measurements (Modal, labelled per GPU)

**Job 1, A100 80GB PCIe** (the benchmark's GPU; source `2d3583fc`)

The benchmark's protocol (`tools/bench/loup_md.py`), 3 fresh processes each:

| Plan | NVE µs/atom/step | Langevin µs/atom/step |
|---|---:|---:|
| Default (as in the benchmark) | 5.325, 5.326, 5.331 | 5.266, 5.273, 5.265 |
| + `radial_tables=on chunked_conv_backward=on` | 5.996, 5.997, 6.002 | 5.944, 5.950, 5.946 |

The default reproduces the benchmark's 5.35 / 5.29.

Same-process A/Bs (`tools/ab_rewrites.py`):
- SrTiO3, 300 K, skin 0.5, 1 fs, 8 blocks of 200 steps;
- b/a in the build orders ab / ba.

| Side b | b/a | Step a → b |
|---|---|---|
| `radial_tables=on chunked_conv_backward=on` | 1.120 / 1.115 | 27.4 → 30.7 ms |
| `radial_tables=on` | **0.902 / 0.901** | 27.5 → 24.8 ms |

**Job 2, A100-SXM4-80GB** (Modal assigned the SXM4 card; source `c658a7b7`)
- The protocol with `radial_tables=on`: NVE 4.493, 4.496, 4.496; Langevin 4.429, 4.432, 4.436 µs/atom/step.
  - The default was not run on this card, so these numbers are not a ratio.
  - The paper's Symmetrix-XL value on an A100-SXM4 is 5.80.
- A/B on NaCl 10×10×10 (8,000 atoms), `radial_tables=on`: **0.917 / 0.917** (22.05 → 20.20 ms).

## Where the default step's time goes (A100 80GB PCIe, `tools/step_breakdown.py`)

**The step is GPU-bound:** 26.9 of 27.0 ms per step is the force graph, and the host adds nothing.

Kernel time of one force call, 26.5 ms:

| Category | ms | Share | Note |
|---|---:|---:|---|
| Interaction 2: convolution backward (streaming) | 5.32 | 20 % | 44 % occupancy |
| Radial networks: folded last layer, input gradient | 4.40 | 17 % | 13 % occupancy |
| Radial networks: folded last layer, forward GEMM | 3.78 | 14 % | 83 % of FP32 peak |
| Atom-level linear layers | 3.42 | 13 % | 69 % of peak |
| Radial networks: other GEMMs | 2.17 | 8 % | |
| Interaction 2: convolution forward | 1.79 | 7 % | |
| Radial networks: elementwise | 1.68 | 6 % | |
| Atom-level elementwise | 1.51 | 6 % | |
| Edge elementwise | 1.12 | 4 % | |
| Interaction 1: reassociated edge sums | 0.80 | 3 % | |
| Symmetric contraction | 0.44 | 2 % | |

- **The radial networks take 45 % of the kernel time.** The radial tables replace exactly these networks.
- **The chunked backward is slow on the A100.** Its folded input gradient becomes 183 kernels at about 2 %
  occupancy, 19.8 ms in the eager profile. On the A100's 108 SMs and 40 MB L2, the chunks are too small.
  - On the 5060 Ti and RTX 4000 Ada the chunked backward measured 0.92–0.97, so it stays on there.
- **Without the tables, the next target is the folded input-gradient kernel**, at 13 % occupancy on the A100.

## Why Symmetrix-XL is close without tables

- **Kernels generated for the model and GPU.** Symmetrix-XL compiles a device artifact for the model, the precision
  and sm_80 (`jit-r1-gen11-f32`, 8 blocks per SM) before the run.
- **Hand-written forward and backward.** It keeps 1.2 GB on the GPU against FastMACE's 5.4 GB, because it stores no
  autograd intermediates per edge.
- **Precision.** Its force error against MACE + cuEquivariance float32 is about twice FastMACE's: 0.0167 against
  0.0077 meV/Å in the benchmark.

## Consequence

- **The A/B evidence:** under decision 66, `generic_radial_tables` measured faster on two A100 models, in both build
  orders:
  - "NVIDIA A100 80GB PCIe": SrTiO3;
  - "NVIDIA A100-SXM4-80GB": NaCl.

  Each card has one structure; the 5060 Ti and RTX 4000 Ada had two each.
- **The pending change:** adding both names to `DECISION_66_GENERIC_RADIAL_TABLES_GPUS` is a default change, and
  waits for the owner.
- **`generic_chunked_conv_backward`:** stays off on the A100. The data for that is the PCIe SrTiO3 A/B with both
  records, 1.12; the chunked backward was not measured alone.

## Records

`results/loup-a100-20261103/`:
- `job1-a100-80gb-pcie/`: the protocol JSONs, the A/Bs, and the breakdowns without traces or kernel sequences;
- `job2-a100-sxm4-80gb/`: the second job, on the A100-SXM4-80GB;
- the structure, `SrTiO3.cif` (a = 3.905 Å, as in the benchmark).

Scripts: `tools/modal/loup_a100.sh`, `tools/modal/loup_a100_b.sh`, `tools/bench/loup_md.py`. Modal cost: about $2.7
for both jobs; this month's metered total afterwards was $5.70.
