# OPT9-T03/T04/T05: the record `wprime_backward`, gates and per-GPU A/Bs (29 October 2026)

Plan: `plans/opt-9.md` (OPT9-T03 onward). It follows the falsifier and ceilings in
`docs/validation/opt9-T01.md`. Decision 66 (29 October 2026) allows a default by exact GPU name where
same-process whole-step A/Bs in both build orders measured a speed-up. Runs and scripts:
`results/opt9-T03-wprime-backward-20261029/`.

**Verdict:**
- The record is exact to float32 rounding.
- The gate passed and NVE drift is unchanged.
- **The whole MD step is 18.5–18.8 % faster on the A100 (NaNbO3) and 11.1 % faster (NbCl5)**, in both build
  orders, with every block pair favouring the record.
- **Local RTX 5060 Ti: 6.0–6.3 % faster (NaNbO3), 2.9–3.0 % (NbCl5) and 5.4–6.5 % (NaCl)**, every block pair
  favouring the record.
- The RTX 4000 Ada was not measured (see "Not done").

## Per-GPU A/B table (the owner's input for decision 66)

`tools/ab_rewrites.py`, a = `wprime_backward=off`, b = `on`. MH-1 hybrid float32 NVT at 300 K, 12 blocks of
300 steps after 300 warm-up steps, everything else the default plan. Median step in ms. Order `ab` uses the abab
block pattern and order `ba` uses abba.

| GPU (exact device name) | Cell | order ab: a / b, b/a | order ba: a / b, b/a | block pairs | proposed |
| --- | --- | --- | --- | --- | --- |
| `NVIDIA A100-SXM4-80GB` (Modal) | NaNbO3 | 11.33 / 9.24, **0.815** | 11.35 / 9.22, **0.812** | 0.811–0.817 | on |
| | NbCl5 | 12.56 / 11.16, **0.889** | 12.55 / 11.15, **0.889** | 0.887–0.894 | |
| | NaCl | not run (job time limit) | | | |
| `NVIDIA GeForce RTX 5060 Ti` (local) | NaNbO3 | 14.47 / 13.56, **0.937** | 14.64 / 13.77, **0.940** | 0.934–0.944 | on |
| | NbCl5 | 18.25 / 17.71, **0.970** | 18.29 / 17.76, **0.971** | 0.946–0.996 | |
| | NaCl | 2.536 / 2.400, **0.946** | 2.572 / 2.405, **0.935** | 0.929–0.960 | |
| `NVIDIA RTX 4000 Ada Generation` | — | not measured | | | off (unmeasured; T01 predicts a loss) |

**Proposed device-name list for a decision-66 default:** `("NVIDIA A100-SXM4-80GB", "NVIDIA GeForce RTX 5060 Ti")`.
The A100-SXM4-40GB (T02's ceiling GPU) was not A/B'd and stays off by decision 66's exact-name rule. Off everywhere else, with the off reason
naming the GPU.

The A100 gain is larger than OPT9-T02's estimate (11 % NaNbO3, 7 % NbCl5) and even than its NaNbO3 ceiling
(0.828). The ceiling stubbed only the input gradient and the table backward. The record also drops the `dw`
chunk stores, the chunk loop's events and side stream, and the 66-column table backward's launch. The estimate
subtracted OPT9-T01's standalone `wdot`, measured with an L2 flush before every call; inside the step the
receiver-order `G'` gathers evidently cost less. Note that this A100 is the SXM4-80GB; T02's ceiling ran on the
SXM4-40GB.

## What was built

**Kernel** (`fastmace/kernels/oeq_streaming/wprime_backward.py`).
- `oeq_projected_backward_wprime` is generated from the streaming schedule with the vendored Jinja
  environment; `factorized_projected.cuh` stays byte-identical, as `test_kernels_oeq_streaming.py` pins it.
- Its body is the vendored edge-owned `oeq_projected_backward`, so `dx` and `dsh` are the same sums in the same
  order.
- The `dweights` store is replaced by `radial_gradient += weight_gradient * w'`, with
  `w' = sum_t T_t(x) G'[group, t, weight_index]` read with `__ldg`. The lane sums are reduced like `dsh`, and
  lane 0 writes `dr[e]`.
- Each edge locates its piece with the tables' rule: one bin gather, then
  `x = (max(r, r_lo) - start) * scale - 1`.
- At or beyond the cutoff (the skin and the padding) and below `r_lo` the slope is zero and `G'` is not read.
- One launch covers all edges: no chunks, no side stream, no `dw`.

**Derivative table.** `G' = C' [W, b]^T` per (pair, piece) in float32, `[P*K, 10, Q]`. This is
`w_table.fold_tables`' derivative half without its padding term and zero group (a CPU test checks the
equality). NaNbO3 has 586 groups and 30 MB; NbCl5 has 213 groups and 11 MB.

**Edge density.** Its gradient reaches `r` through the table's own backward on the density column alone
(`dcoef[..., -1:]`, with an `[E, 0]` folded gradient): one small launch in place of the 66-column one.

**Autograd.** One function, `_WPrimeConvolution`, takes the edge lengths and returns `(message, edge density)`:
- The forward is unchanged: the table, then `fused_folded_forward` (or the cuBLAS product when it is off), which
  stores `w`.
- The backward is the kernel above plus the density column's backward.
- The MH-1 folded forward takes this path when the interaction has `wprime_backward` attached, its edge features
  are the lengths and its backward is chunked.

**Record** `wprime_backward` (`fastmace/rewrites/radial`):
- requires `radial_tables` and `chunked_conv_backward`;
- exactness `rounding` (the derivative series and the folded product are reassociated); default `off`; not
  deterministic (the sender gradients' atomics); refuses float64 tables and a CPU model; the size guard is
  `W_TABLE_MAX_BYTES`;
- its self-check runs, per interaction, a random 720-edge graph (a twelfth of the edges beyond the cutoff)
  through the record and through the chain (table, `folded_streaming_convolution` with its chunked backward),
  and compares the message, the edge density and the gradients to r, x and sh within SPEC §6.4's float32 limit.

**`RewriteConfig.wprime_backward`:**
- `auto` rule `never_automatic`;
- `on` needs `radial_tables` and `chunked_conv_backward` (`rewrite-requirement-missing`);
- `on` needs float32 outside reproducible plans (`wprime-backward-needs-float32`).

**Docs and tests:** the switch map, the golden plan texts, `test_config.py`, `test_report.py` and
`test_reproducible.py`.

## Exactness

**CPU** (`tests/cpu/test_wprime_backward.py`):
- The `G'` dot against `radial_table_backward(dw [W, b])`, with float64 tables whose `dcoef` is `coef`'s
  Chebyshev derivative, agrees within 1e-12 relative (float32: 2e-5).
- Edges at and beyond the cutoff and below `r_lo` give exactly zero.
- A float64 `gradcheck` of the radial chain with the density column passes, and the dot plus the density-only
  backward equal its gradient within 1e-11. The density's share is checked to be non-negligible.
- The generated source is checked for both layouts, and NVRTC compilation for sm_120, sm_89 and sm_80 (run
  locally in `mace_cueq_oeq`; the CPU environment has no NVRTC and skips it).
- The record and the planner are tested.

Targeted CPU files (`fastmace_cpu`, `-n 6`): 431 passed, 36 skipped (planner, plan texts, config, report,
reproducible, radial tables, MH-1 records, registry, fused forward, streaming kernels, this record). ruff is
clean.

**GPU** (`tests/gpu/test_wprime_backward_cuda.py`). The setup is synthetic: degree-10 float32 tables over 9
pairs and 12 pieces, K = 65, 1,500 edges with 70 padding edges, skin edges and an edge at exactly `r_max`. It
covers both layouts, the cuBLAS and the fused forward, and a CUDA graph.

| Check | RTX 5060 Ti (sm_120) | A100-SXM4-80GB (sm_80) |
| --- | --- | --- |
| message, edge density | bitwise the chain's | passed |
| `dsh` | bitwise the chain's | passed |
| `dx` (atomic sender sums) | ≤ 2.1e-7 of its largest value | passed |
| `dL/dr` vs a float64 reference from the same `dw` | record 2.5–3.2e-7, chain 4.5–5.3e-7 | passed (≤ 1e-5) |
| CUDA graph replay | message, `dsh`, `dr` bitwise; `dx` to rounding | passed |

The record's `dL/dr` is closer to float64 than the chain's. The chain rounds `da = dw [W, b]` to float32
before the table backward; the record does not form `da`. The A100 run used the test's first version, which held
only the record to the float64 reference. That version's synthetic table had a non-contiguous `dcoef`
(`chebder`'s strides), which misled the Triton table kernels of the chain (not the record). The fix is commit
`8e647113`, and the local run above is on the fixed test.

## Gates (local RTX 5060 Ti, driver 610.88, `mace_cueq_oeq`; pinned source `523b0890`; `local-523b0890/`)

**Force/stress gate** (`tools/force_stress_gate.py --engine fastmace --rewrites wprime_backward=on`, MH-1, 3
configurations per structure and cell mode): **passed** on all three cells. The plan resolved
`wprime_backward = on`. The table gives the error against float64, with its ratio to the float32 ASE
calculator's error in brackets:

| Cell | energy (eV) | forces (eV/Å) | stress (eV/Å³) |
| --- | ---: | ---: | ---: |
| NaCl | 6.4e-5 (0.086) | 3.6e-6 (0.92) | 3.3e-8 (0.37) |
| NbCl5 | 7.9e-5 (0.11) | 8.4e-5 (1.04) | 4.1e-8 (0.43) |
| NaNbO3 | 3.9e-4 (0.32) | 1.3e-4 (1.02) | 1.9e-7 (0.55) |

These are the same as `fused_folded_forward`'s gate (`docs/validation/opt8-T02.md`) to within the gate's
scatter.

**NVE drift** (`tools/bench/nve_drift.py`, 20 ps at 1 fs, 300 K; drift in meV/atom/ps, rms of the conserved
quantity in meV/atom):

| Cell | off | on |
| --- | --- | --- |
| NaCl | −1.3e-6, rms 2.75e-4 | −3.7e-7, rms 2.75e-4 |
| NbCl5 | 1.3e-6, rms 3.80e-4 | 7.8e-7, rms 3.79e-4 |

## Runs

- **Modal A100** (`modal-a100/`): `NVIDIA A100-SXM4-80GB`, driver 610.57.04, image torch 2.14.1+cu130; job
  `ap-b3W2cK7qJvHgBBqxwMyYan`, source `fe58a0b8`; 1,267 s, about $0.96 estimated (ledger metered before $25.88).
  It ran under the October override (cap $28). The GPU test passed first, then NaNbO3 and NbCl5 in both orders.
  NaCl was skipped by the script's 19-minute start limit.
- **Local RTX 5060 Ti** (`local-523b0890/`, raw files in WSL `~/fastmace-runs/opt9-T03/`): the GPU test, the
  gate and the drift through `tools/ci/local_gpu.py run`; the A/Bs with `--quiet` (NaNbO3 on 29 October
  02:08–02:15 UTC; the PC rebooted on its own during NbCl5, which was rerun with NaCl at 08:47–09:07 UTC). No
  compile while timed, both sides' plans as requested.

## Not done

- **RTX 4000 Ada (murphy, leto):** not measured. murphy's GPU was idle at 01:13 CET on 29 October, but
  deploying the source there (writing under `/mnt/homes/zwingel` over ssh) was refused by this session's
  permission settings. `results/opt9-T03-wprime-backward-20261029/ada.sh` runs the GPU test, the gate and the six
  A/Bs from an archive of the branch. Until it runs, the Ada stays off: T01 measured the added `G'` gathers at
  0.76 / 0.71 of the removed work there, against 0.21 / 0.17 on the A100.
- NaCl on the A100.
- MH-1 family models (POLAR-1-S, OMOL-0) were not run.
- **CPU tests after the merge of main (`030e2fe2`, merge `4859055f`) were not run.** The owner's rule of
  29 October allows no CPU tests on the PC, and this session may not write to the workstations. The conflicts
  were the golden plan texts, `test_report.py`'s count, the switch map and the plans. The new
  `species_residual` and `wprime_backward` off fields were recomputed by hand with `plan_text._wrap`. Before the
  merge, 431 targeted CPU tests passed on the PC.
- The full CPU suite (merge tier) is the conductor's.
