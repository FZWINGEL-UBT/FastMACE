# OPT3-T04: NaCl's graph gaps (launch-count reduction); closed, not adopted

Optimisation round 3 (`plans/opt-3.md`). OPT-T10 (`docs/design/opt-T10-step-breakdown.md`) found 174 µs of
gaps between the nodes of NaCl's force graph, 264 launches per replay. Goal: cut launches exactly; estimate
3–5 % on NaCl. Date: 21 October 2026. **Result: no exact candidate with a gain and no loss; closed.**

## Where NaCl's launches are (OPT-T10's kernel sequence, 256 kernels of an eager force call)

| Origin | Kernels | Notes |
| --- | ---: | --- |
| Inductor-generated Triton | 123 | many on 216 rows or scalars: argmax of the one-hot species, `arange`, `zeros`, `ones_like`, `scatter_add` into fresh zeros, `index_put`, `cat` pieces; 50 of them in 8 runs of ≥ 3 between GEMMs |
| GEMMs (cuBLAS, FastMACE Triton) | 87 | |
| cuBLAS memsets, one before each small GEMM | 25 | 0.3–0.5 µs each, plus their gaps |
| eager `fill_` and `copy_` | 12 | zero-fills of [216], [8192, 3], [216, 512], [1] |
| FastMACE and OEq custom ops | 9 | |

## The candidate built: `combo_kernels` (Inductor's combo kernels, pointwise only)

The record `combo_kernels` (new family `fastmace.rewrites.inductor`, registered last) marks the model.
`compile_model` then compiles it with `torch.compile(options={"combo_kernels": True,
"combo_kernels_pointwise_only": True})`. The enabled records are in the compile-cache key. It is opt-in
(`never_automatic`).

### Launches and gaps (probe, NaCl, MH-1 hybrid float32, local RTX 5060 Ti, one process per side)

| Side | Kernels per force-graph replay | Gaps inside a replay |
| --- | ---: | ---: |
| off (two processes) | 264, 266 | 156.2, 157.0 µs |
| on (two processes) | 211, 211 | 129.1, 127.4 µs |

The combos remove 53 launches and about 29 µs of gaps per step.

### Step A/Bs (`tools/ab_rewrites.py`, same process, 12 blocks of 300 steps, both build orders)

- **Local:** RTX 5060 Ti, driver 610.88, WSL2, `mace_cueq_oeq`, FastMACE `e3614cc` (pinned).
- **Modal L4:** sm_89, driver 610.57.04, native Linux, recipe-C image, FastMACE `669ecee`. "Modal L4, not murphy". 29 min, $0.39 of GPU list price.

| Machine | Structure | ab: a → b ms/step (b/a, pairs b faster) | ba: a → b ms/step (b/a, pairs b faster) |
| --- | --- | --- | --- |
| local | NaCl | 2.999 → 2.972 (0.991, 10/12) | 2.970 → 2.916 (0.982, 12/12) |
| local | NbCl5 | 21.327 → 21.835 (1.024, 0/12) | 21.331 → 21.899 (1.027, 0/12) |
| Modal L4 | NaCl | 4.341 → 4.302 (0.991, 11/12) | 4.286 → 4.244 (0.990, 11/12) |
| Modal L4 | NbCl5 | 35.555 → 36.241 (1.019, 0/12) | 35.399 → 36.098 (1.020, 0/12) |

On native Linux the gain on NaCl is about 1 %, and NbCl5 loses about 2 % on both cards. The gap reduction is
real, but on large cells the combined kernels run slower than the separate ones. A generic screen with
`combo_kernel_per_subkernel_blocks` (each sub-kernel keeps its own block size; local, one order, an uncommitted
screening copy `c03a511+per_subkernel_blocks`) did not remove the loss: NaCl 2.978 → 2.942 ms (0.988, 12/12),
NbCl5 21.325 → 21.806 ms (1.023, 0/12).

### Exactness (reproducible mode)

`results/opt3-T04-graph-gaps/reproducible_check.py` ran plain cuEq MH-1 float32, `reproducible=True` (2-T16), on
NaCl 3×3×3, 200 NVT steps, off, off and on, each in a fresh process (local, `mace_cueq_latest`, FastMACE
`c03a511`):

| Compared with the first "off" | positions | momenta | forces | potential energy |
| --- | --- | --- | --- | --- |
| second "off" | bitwise | bitwise | bitwise | bitwise |
| "on" | bitwise | bitwise | bitwise | **differs by 2.1e-6 eV** (−1473.8717015 against −1473.8717036 eV) |

The trajectory is unchanged, but the energy the run reports is not. So the record is not exact. It is labelled
`rounding`; the record exists for future screens. On the hybrid backend, which isn't bitwise repeatable, the
CUDA test found "on" within the "off" repeat's scatter (L4: 9.5e-7 against 1.67e-6; local: passed).

## Why it is closed

- The task asks for exact cuts that stay bitwise in the reproducible mode. Inductor's combo kernels fail on the
  energy.
- A size rule (combos only for small cells, computed from atoms and the SM count) would avoid NbCl5's loss. It
  would not make the record exact, and the gain left is about 1 %, below the 3–5 % estimate.

## Not attempted, for a later round

- **The 25 cuBLAS memsets before small GEMMs.** They belong to cuBLAS's kernel choice; removing them means
  another GEMM (rounding) or a different cuBLAS path.
- **Constants recomputed every step:** the species index (argmax of the one-hot attributes), `arange`,
  `ones_like` for the autograd seed. Precomputed buffers would be exact, but they need model-level rewrites
  and touch MC's species changes (phase 3).
- **The eager zero-fills before scatter accumulations:** fusing them into their consumers is exact, but it
  is per-op work. OPT3-T05 (layout audit: copies and transposes) overlaps it.

## Records

- `results/opt3-T04-graph-gaps/local-20261021/`: probe JSONs, A/B JSONs and logs, the per-subkernel screen,
  and the reproducible-mode logs and `repro-compare.json`.
- `results/opt3-T04-graph-gaps/modal-L4-669ecee/`: the L4 A/Bs, `modal.json` and the CUDA check log.
- Scripts: `probe_combo.py`, `reproducible_check.py`, `local.sh`, and `tools/modal/opt3_t04.sh`.
