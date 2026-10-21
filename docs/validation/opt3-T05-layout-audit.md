# OPT3-T05: layout audit

Plan task OPT3-T05 (`plans/opt-3.md`). Audit and design: `docs/design/opt3-T05-layout-audit.md`. Branch
`fastmace/opt3-T05`. Results: `results/opt3-T05-20261021/`.

**Result:**
- **Measured.** Pure data movement is 2.4 % of NaCl's force kernels and 0.75 % of NbCl5's.
- **Kept.** Two exact removals, `direct_node_feats` and `folded_constants`, about 11 µs per NaCl force
  call. Both pass the gate. They give no gain that can be resolved from noise: NaCl is 0.5–0.8 % faster
  locally, inside the noise, and neither structure gains on the L4. Both stay opt-in.
- **Not exact.** The rest needs a gathered-row GEMM (rounding, already used on the large cells) or moves a
  transpose without removing it (interaction 1, screened and reverted).

## What was built

- **`direct_node_feats`** (MH-1 family; exact, deterministic, opt-in, requires `fold_cutoff`). Without
  LAMMPS, the folded forward skips the `[node_feats, node_attrs]` join for `handle_lammps`, which is the
  identity then. Its strided slices made the streaming convolution copy its input `[n, 512]`, forward and
  backward.
- **`folded_constants`** (same class). The folded last radial layer's `[W, b]` and its halves are buffers
  built once. A new op, `fastmace::folded_linear_input_grad_parts`, runs the same kernel on the given
  halves.
- **`tools/layout_probe.py`:** the CPU op, shapes and time of every copy-only kernel, from one profiled
  eager force call.
- **`tools/modal/opt3_t05.sh`:** the L4 A/B job.

## Checks

| Check | Result |
| --- | --- |
| CPU (`tests/cpu/test_layout_records.py` and the touched config, planner, report and MH-1 tests) | energy and forces **bitwise** with and without each record and both together (tiny MH-1-type model, float64); the records are opt-in, refused without `fold_cutoff`, and self-checked; `folded_linear` with given halves is bitwise the same; ruff clean |
| Layout probe, NaCl (local) | pure data movement 59.4 → **47.8 µs** per force call; the `[216, 512]` copies, the `[1280, 65]` cat and the `[1280, 64]` and `[1280]` copies are gone |
| Force gate (`--rewrites direct_node_feats=on folded_constants=on`) | **passed**: NaCl energy 0.05, forces 1.15, stress 0.32; NbCl5 0.19, 1.00, 0.70 (limit 2) |

## Speed

Same-process A/B (`tools/ab_rewrites.py`):
- **Run:** MH-1 hybrid float32 NVT at 300 K, 3×3×3, 300 warm-up steps, 12 alternating blocks of 300
  steps.
- **Sides:** a is the default plan; b adds both records.
- **Validity:** all blocks valid, nothing compiled while timed.

| Structure | Order, pattern | Local RTX 5060 Ti b/a (a → b, ms) | Modal L4 b/a (a → b, ms) |
| --- | --- | --- | --- |
| NaCl | a-b, ABAB | **0.9921** (2.9936 → 2.9699) | 1.0034 (4.0648 → 4.0785) |
| NaCl | b-a, ABBA | **0.9948** (2.9751 → 2.9597) | 1.0005 (4.1298 → 4.1320) |
| NbCl5 | a-b, ABAB | 1.0020 (21.2113 → 21.2535) | 1.0008 (33.6860 → 33.7125) |
| NbCl5 | b-a, ABBA | 1.0020 (21.2842 → 21.3274) | 0.9950 (34.1252 → 33.9556) |

- **Noise floors (local):** 1.1 % on NaCl, 0.4 % on NbCl5.
- **NaCl, locally:** both orders are faster, by more than the expected 0.4 %, but inside the noise.
- **NbCl5, locally:** within noise.
- **The L4:** the two orders disagree on both structures, so no effect.
- **Default candidate?** The rule (`plans/opt-3.md`) needs a gain on sm_120 and no loss on the L4. Neither
  holds clearly, so this is the owner's call.

**Machines:**
- **local:** RTX 5060 Ti (sm_120), WSL2, `~/environments/mace_cueq_oeq`, pinned copy
  `~/fastmace-runs/src-3aff996` (the branch's code; `fce27e8` and `9f519ea` add only reverts of the
  screened record, the script and docs). Each job held its own `tools/ci/local_gpu.py` reservation of
  at most 15 minutes, 21 October 2026.
- **Modal:** an NVIDIA L4 (sm_89, driver 610.57.04, CUDA 13.3), not murphy. Image pinned from
  `requirements-hybrid.txt`, source `9f519ea`, 19:56–20:17 UTC, 20.7 min. Estimated GPU cost $0.28.

## Screened and reverted

**`lm_major_outer_sum`** (exact): the reassociated first interaction's per-slot sums stored (l, m)-major,
so that each l's GEMM operand is a view.
- **What it did:** it removed the input transposes, but the transpose moved to the join and the gradient
  slices, because the GEMM's input and output are the same size. NbCl5's interaction-1 layout kernels went
  from 93.2 to 85.9 µs per force call (0.03 % of the step).
- **History:** commits `4dcaa30` and `37ef837`, reverted in `fce27e8`.
