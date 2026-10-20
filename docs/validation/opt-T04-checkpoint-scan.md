# OPT-T04: checkpoint scan for exact skips

Plan: `plans/opt-1.md` OPT-T04 ("a checkpoint scan for zero weight blocks and unused paths or heads").
CPU only, report only. Tool: `tools/checkpoint_scan.py`. Raw output: `results/opt-T04-checkpoint-scan/summary.json`.

**Result: nothing worth building.** The learned weights of the ten benchmark models and the five MACE-OFF
models have no zero blocks, no zero linear paths and no species with a zero contraction block. The exact
zeros in the checkpoints are almost all in the symmetric contractions' fixed coupling tensors (`U_matrix_*`
buffers), whose sparsity the cuEquivariance kernel already exploits. The only exact skips found are worth
below 0.5 % of an MH-1 step together.

## Method

The scan loads each pickled MACE model on the CPU (`torch.load`, no conversion, no rewrites). For every
checkpoint behind the labels `BENCHMARK_LABELS` and `MACE_OFF_LABELS` (`fastmace/model/labels.py`), it records:

- the share of exact zeros in all parameters and buffers, and the tensors that are entirely zero;
- 2-D tensors with all-zero rows or columns;
- for every e3nn `o3.Linear`, each instruction's (path's) weight block, and whether it is all zero;
- for every symmetric contraction, per element (species) and weight tensor, whether its block is all zero;
- heads and readout shapes.

Local PC, WSL2, `~/environments/fastmace_cpu` (CPU torch), checkpoints from `~/models`, 21 October 2026.

## Findings

| Checkpoint (labels) | Parameters + buffers | Exact zeros | in learned parameters | Zero linear paths | Species with a zero contraction block |
| --- | --- | --- | --- | --- | --- |
| `2023-12-03-mace-128-L1_epoch-199` (mp0-medium) | 5.52 M | 14.8 % | 0 | 0 of 10 | none |
| `MACE_MPtrj_2022.9` (mp0-large) | 16.70 M | 4.9 % | (not split) | 0 of 10 | none |
| mp0b3, mpa0, omat0, matpes-pbe, matpes-r2scan (medium) | 9.90 M each | 8.3 % | (not split) | 0 of 10 | none |
| `mace-mh-1` (mh1-omat, mh1-mp, mh1-r2scan) | 7.35 M | 11.4 % | 14,171 of 6.44 M (0.22 %) | 0 of 20 | none (one element group) |
| MACE-OFF23 small / medium / large, 23b, 24 medium | 0.89–6.91 M | 21–36 % | 0 (medium) | 0 of 10 | none |

- **Where the zeros are.** In every model, the largest zero shares are in
  `products.*.symmetric_contractions.contractions.*.U_matrix_{1,2,3}`: 93.8–99.6 % zero. These are the
  generalised Clebsch–Gordan coupling tensors, which are fixed, not learned. cuEquivariance's
  `uniform_1d` symmetric contraction already works on their sparse form. The only all-zero tensor is
  `scale_shift.shift` (a buffer, 0 for these models).
- **MH-1's learned zeros.** In MH-1's first product, the contraction for the 1o output has 24 of its 512
  feature channels zero over the whole basis (`weights_max`, `weights.0`; 23 in `weights.1`). So 24 of the
  512 channels of that product's l = 1 output are exactly zero before `products.0.linear`. Removing them
  would shorten that block linear's l = 1 product by K = 24 of 512 (4.7 %), and nothing else. Estimate:
  under 0.1 ms per NbCl5 step, below the noise floor.
- **Heads.** MH-1 has six heads. A run uses one, but the readouts compute all of them: the linear readout
  is 512 → 6 (3,072 weights), and the non-linear readout is 512 → 96 → 6 (16 hidden channels per head).
  Slicing them to one head is exact. Estimate: the readouts take about 0.2 GFLOP forward and twice that in
  the backward per force call on NbCl5 3×3×3 (1944 atoms), and five sixths of it could go. That is
  roughly 0.05 ms, or 0.2 % of the step.
- **Species-only work** is already hoisted out of the step by `factor_species` (MH-1 class) and
  `species_skip` (MP-0/density family).

## Recommendation

Do not build OPT-T04 rewrites. The two exact skips found (MH-1's 24 zero l = 1 channels, and the unused
heads' readout columns) are worth well under 0.5 % together, below the 1.1 % local noise floor, and each
would need its own record, self-check and pins. Record it in `docs/rewrites.md` as screened.
