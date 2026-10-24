# OPT4-T08: per-stage precision sensitivity of the force call

Optimisation round 4 (`plans/opt-4.md`), a task the owner asked for on 24 October 2026. **The question:** if a
single stage of the force call ran at reduced precision, with every other stage on the production FP32 path,
would the forces still pass the force gate? The gate allows at most 2× the error of MACE's float32 calculator,
both measured against float64. **The owner's idea:** the accuracy loss comes from a few crucial stages, and
the others may tolerate lower precision.

This is a measurement only: no kernels, no rewrites, untimed. Runs were on 24 October 2026, 14:04–15:23 CEST.

**Result: no stage with a real share of the step passes the gate on its own in any 16-bit format (bf16 or
fp16), whether in storage or in compute.**
- fp16 misses the gate by 2–1,100×. bf16 misses by 7× up to more than 30,000×.
- TF32 passes in only one place: the backward of interaction 0's symmetric contraction on MH-1. That is a
  part of the symmetric contraction's 3–6 % of the force call. The same stage fails on omat0-medium, and
  interaction 1's fails on NaNbO3's stress.
- The only stage that tolerates fp16 storage everywhere is the ZBL pair repulsion, which costs nothing.

The sensitivity is not concentrated in a few stages: every stage that carries GEMM work fails by itself.

## Method

**Tool:** `tools/precision_screen.py` (commit `35761a1`). One process per model and structure, on the local GPU:

1. **The gate's own copies of the cell.** `selfcheck.make_copies` with a variable cell gives 2 rattled copies
   (0.05 Å) and 2 strained (±2 %, full) and rattled copies. Stress is therefore gated too.
2. **The float64 reference and the float32 yardstick of `fastmace.reference`.** Both are MACE's calculator on
   the stock record (cuEquivariance), and the yardstick accumulates its energy in float64. Each is built,
   evaluated and freed before the next.
3. **The screened path.** The same float32 yardstick calculator, built on MACE's e3nn modules
   (`backend="e3nn"`). There every GEMM is a torch matmul that TF32 can reach, and the neighbour sums are
   separate `scatter_sum` calls.
   - TorchScript's executor optimisation is off, and deterministic algorithms are on.
   - Two unperturbed evaluations are bitwise equal in every case.
   - Unperturbed, it scores gate ratios of 0.85–1.03 against the cuEq yardstick (table below). It is a valid
     stand-in for the FP32 path.
4. **One stage perturbed per evaluation, nothing else.** The perturbed path is scored with the gate's own
   `selfcheck.copy_errors` and `selfcheck.verdict`, against the float64 reference, with the cuEq float32
   yardstick as the denominator. The RMS force error is recorded as well.

**Stages.** Hooks on MACE's modules, per interaction (`i0`, `i1`):

| Stage | Modules (MH-1; omat0-medium has `linear` for `linear_1`/`linear_2` and no species embeddings or gate) |
| --- | --- |
| radial-basis | `radial_embedding` (Bessel basis and cutoff) |
| spherical-harmonics | `spherical_harmonics` |
| node-embedding | `node_embedding` |
| zbl | `pair_repulsion_fn` |
| radial-mlp | `conv_tp_weights` and `density_fn` (forward and its derivative in the backward) |
| species-embeddings | `source_embedding`, `target_embedding` |
| linears | `linear_up`, `linear_1`, `linear_2` |
| conv-tp | `conv_tp` (the per-edge message) |
| scatter | the two `scatter_sum` calls inside the interaction (message and density) |
| gate | `equivariant_nonlin` |
| skip-residual | `skip_tp`, `linear_res` |
| symmetric-contraction | `products[i].symmetric_contractions` |
| product-linear | `products[i].linear` |
| readout | `readouts[i]` |

**Modes** (`dt` is bf16 or fp16):

| Mode | What changes | Forward / backward |
| --- | --- | --- |
| `out-dt` | the stage's outputs rounded through `dt`; the gradient passes unchanged | forward |
| `grad-dt` | the gradients the stage's backward returns (with respect to its inputs) rounded through `dt` | backward |
| `saved-dt` | the tensors autograd saves inside the stage, stored in `dt` | backward |
| `tf32-fwd` | TF32 for the stage's GEMMs in the forward | forward |
| `tf32-bwd` | TF32 for the stage's GEMMs in its backward only. The stage is one autograd node that recomputes itself in FP32 and differentiates with TF32 on for that nested call. | backward |
| `autocast-dt` | the stage under `torch.autocast` (its GEMMs in `dt`) | both |
| `cast-dt` | the whole stage in `dt`: modules copied to `dt`, inputs cast in, outputs cast out | both |

**Cases:**
- MH-1 (`mh1-r2scan`, `mace-mh-1.model`) on NaCl, NbCl5 and NaNbO3, all 3×3×3.
  - NaNbO3's interaction 1 ran out of GPU memory at 3×3×3 (82,688 edges, about 16 GB in e3nn). Its
    interaction-1 stages ran at 2×2×2 (320 atoms) instead, as the brief allows.
- omat0-medium on NaCl and NbCl5, 3×3×3.

### Caveats

- **The gate's floors decide some NaCl verdicts.** On NaCl the yardstick's largest force error is 3.5e-6
  eV/Å, below the gate's force floor of 1e-5. The energy floor of 0.01 meV/atom is about 14× the yardstick's.
  - So a NaCl row can pass with a force ratio up to about 2.9 and a large energy ratio.
  - The tables mark such floor-only passes separately. The large cells have no such slack: their yardstick
    errors are 5e-5 to 1.3e-4 eV/Å.
- **The production path has less headroom than the screened one.** Production MH-1 already gates at force
  ratio 1.16 on NaCl and 1.01 on NbCl5 (`docs/validation/opt-def-2.md`). A stage that brings the screened
  path to 1.9 would not fit beside production's own error.
- **The production kernels differ.** The symmetric contraction is cuEquivariance, interaction 2's convolution
  is OEq's streaming kernel, and the radial last layer is folded. The screen measures what precision the
  arithmetic of each stage tolerates. It does not test those kernels.
- **TF32 and autocast reach only torch matmuls.**
  - e3nn's linears and tensor products are einsums, which autocast leaves in FP32. Their autocast rows are
    `=` (unchanged); TF32 does reach them.
  - `=` in the TF32 columns means the stage has no GEMM.
  - The readout's `cast` mode fails inside TorchScript (`err`).
- **Stages without an input gradient.** Node embedding, the species embeddings and interaction 0's skip path
  take the one-hot species as input. Their backward produces no gradient, so `grad` and `tf32-bwd` cannot
  change them, and their `tf32-bwd` "passes" carry no information.

## Results

The worst force ratio (production / yardstick, largest component) over the cases. **Bold**: the gate passes in
every case with every ratio ≤ 2. *Italic*: passes only through a floor. `inf`: a non-finite force (fp16
overflow). `=`: unchanged.

### MH-1 (NaCl, NbCl5, NaNbO3)

| Stage | out bf16 | out fp16 | grad bf16 | grad fp16 | saved bf16 | saved fp16 | tf32 fwd | tf32 bwd | autocast bf16 | autocast fp16 | cast bf16 | cast fp16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| radial-basis | 7,970 | 1,050 | 557 | 73.8 | 1,420 | 171 | = | = | = | = | 30,800 | 3,270 |
| spherical-harmonics | 207 | 27.1 | 20.7 | 3.19 | 90 | 14.4 | = | = | = | = | 566 | 75.3 |
| node-embedding | 130 | 5.44 | = | = | = | = | 21.3 | (no grad) | = | = | 101 | 22.3 |
| zbl | **1.03** | **1.03** | 3.56 | **1.03** | 6.41 | **1.24** | = | = | = | = | 315 | inf |
| i0-radial-mlp | 176 | 23.0 | 266 | 30.7 | 619 | 79.1 | 599 | 44.7 | 5,550 | 626 | 6,240 | 645 |
| i0-species-embeddings | 47.0 | 30.7 | = | = | = | = | 29.1 | (no grad) | = | = | 277 | 8.24 |
| i0-linears | 305 | 36.6 | 219 | 28.3 | 22.4 | 3.02 | 51.2 | 23.5 | = | = | 708 | 91.4 |
| i0-conv-tp | 146 | 21.9 | 199 | 23.8 | 41.3 | 4.02 | = | = | 335 | 45.7 | 335 | 45.7 |
| i0-scatter | 364 | 42.9 | 238 | 24.9 | = | = | = | = | = | = | 1,260 | 222 |
| i0-gate | 153 | 19.4 | 97.4 | 11.5 | 54.9 | 6.10 | = | = | 27.7 | 3.61 | 366 | 39.5 |
| i0-skip-residual | 44.4 | 5.42 | = | = | = | = | 10.8 | (no grad) | = | = | 60.8 | 28.9 |
| i0-symmetric-contraction | 241 | 27.7 | 143 | 13.8 | 112 | 14.4 | 3.45 | **1.25** | 429 | 70.5 | 427 | 74.0 |
| i0-product-linear | 81.1 | 8.18 | 109 | 14.5 | 25.8 | 1.94 | 23.1 | 16.4 | = | = | 377 | 48.8 |
| i0-readout | 1.03 | 1.03 | 43.8 | 9.99 | 45.6 | 4.78 | 1.03 | 34.0 | = | = | 71.2 | 27.3 |
| i1-radial-mlp | 117 | 16.9 | 128 | 18.2 | 472 | 70.4 | 476 | 25.0 | 3,320 | 480 | 3,320 | 464 |
| i1-species-embeddings | 183 | 3.48 | = | = | = | = | 14.7 | (no grad) | = | = | 42.3 | 20.4 |
| i1-linears | 276 | 27.6 | 166 | 19.9 | 34.2 | 5.92 | 35.2 | 17.3 | = | = | 522 | 64.6 |
| i1-conv-tp | 140 | 17.5 | 150 | 16.8 | 89.5 | 12.1 | 2.13 | 1.95 | 307 | 44.9 | 343 | 47.0 |
| i1-scatter | 220 | 28.2 | 111 | 14.4 | = | = | = | = | = | = | 824 | 94.9 |
| i1-gate | 119 | 12.1 | 77.6 | 8.06 | 56.6 | 7.83 | = | = | 44.3 | 5.67 | 230 | 25.2 |
| i1-skip-residual | 134 | 15.8 | 117 | 17.1 | 6.85 | 2.96 | 27.3 | 6.16 | = | = | 314 | 42.3 |
| i1-symmetric-contraction | 18.0 | 2.26 | 67.8 | 7.72 | 64.5 | 9.19 | 4.04 | 1.50 | 129 | 17.1 | 151 | 20.8 |
| i1-product-linear | 16.9 | 2.46 | 56.7 | 8.18 | 8.65 | 1.87 | *2.60* | 5.46 | = | = | 97.9 | 10.7 |
| i1-readout | 1.03 | 1.03 | 44.0 | 5.26 | 90.9 | 11.4 | 3.07 | 24.3 | = | = | err | err |

- The readout's `out` rows show 1.03 for forces because a rounded output changes only the energy. Forces are
  its gradient, and nothing downstream saves it. Its energy ratio fails.
- Rows whose force ratio is ≤ 2 but which are not bold fail on stress or energy in some case:
  - i1-symmetric-contraction `tf32-bwd`: NaNbO3 stress 2.38;
  - i1-conv-tp `tf32-bwd`: NbCl5 stress 3.42;
  - i1-product-linear `saved-fp16`: NaCl stress 3.78.

### omat0-medium (NaCl, NbCl5)

| Stage | out bf16 | out fp16 | grad bf16 | grad fp16 | saved bf16 | saved fp16 | tf32 fwd | tf32 bwd | autocast bf16 | autocast fp16 | cast bf16 | cast fp16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| radial-basis | 5,820 | 715 | 387 | 58.7 | 1,710 | 215 | = | = | = | = | 2.6e11 | 133,000 |
| spherical-harmonics | 169 | 22.1 | 24.7 | 3.42 | 129 | 18.0 | = | = | = | = | 682 | 91.0 |
| node-embedding | 48.5 | 4.05 | = | = | = | = | 19.2 | (no grad) | = | = | 81.2 | 16.4 |
| zbl | **0.99** | **0.99** | **1.20** | **1.00** | *2.18* | **1.11** | = | = | = | = | 257 | 47.4 |
| i0-radial-mlp | 491 | 66.6 | 368 | 36.4 | 274 | 35.5 | 360 | 65.8 | 3,180 | 453 | 3,050 | 436 |
| i0-linears | 284 | 35.7 | 199 | 26.5 | 14.4 | 5.36 | 141 | 30.3 | = | = | 1,570 | 194 |
| i0-conv-tp | 445 | 54.3 | 213 | 26.2 | 53.2 | 8.83 | = | = | 837 | 96.2 | 837 | 96.2 |
| i0-scatter | 748 | 102 | 253 | 31.5 | = | = | = | = | = | = | 3,530 | 526 |
| i0-skip-residual | 362 | 46.9 | 226 | 26.4 | 190 | 2.84 | 32.2 | 13.4 | 539 | 59.5 | 581 | 76.4 |
| i0-symmetric-contraction | 306 | 28.8 | 108 | 12.6 | 112 | 14.2 | 32.3 | 3.92 | 639 | 85.7 | 529 | 96.3 |
| i0-product-linear | 186 | 26.9 | 121 | 15.6 | 15.6 | 2.26 | 33.8 | 10.4 | = | = | 518 | 62.1 |
| i0-readout | 0.99 | 0.99 | 10.8 | 3.15 | 16.4 | 5.38 | = | = | = | = | 40.8 | 32.5 |
| i1-radial-mlp | 180 | 23.2 | 574 | 74.6 | 524 | 54.4 | 318 | 82.2 | 4,190 | 400 | 3,780 | 407 |
| i1-linears | 446 | 46.7 | 143 | 19.3 | 17.9 | 6.03 | 43.0 | 26.7 | = | = | 605 | 98.8 |
| i1-conv-tp | 160 | 20.5 | 206 | 24.6 | 130 | 15.9 | 2.93 | 1.61 | 425 | 57.4 | 433 | 57.6 |
| i1-scatter | 288 | 30.4 | 142 | 19.9 | = | = | = | = | = | = | 996 | 116 |
| i1-skip-residual | 69.0 | 11.0 | 54.2 | 6.37 | 57.7 | 1.23 | 11.2 | 4.81 | 220 | 21.1 | 220 | 25.8 |
| i1-symmetric-contraction | 197 | 25.9 | 197 | 20.4 | 185 | 18.9 | 4.00 | 1.36 | 542 | 59.9 | 588 | 67.1 |
| i1-product-linear | 68.4 | 6.92 | 194 | 21.6 | 9.11 | 2.94 | 20.8 | 10.4 | = | = | 345 | 41.2 |
| i1-readout | 0.99 | 0.99 | 76.8 | 11.0 | 459 | 57.1 | 16.3 | 46.2 | = | = | 977 | 106 |

- The radial basis `tf32-bwd` row errored in the tool on this model, because its radial embedding returns a
  `None` output. The stage has no GEMM (its `tf32-fwd` is `=`), so the row is `=` by construction. The tool is
  fixed.
- On omat0-medium the symmetric contraction's `tf32-bwd` fails: interaction 0 on NaCl's forces (3.92) and
  interaction 1 on NbCl5's stress (2.28).

### The unperturbed screened path (gate ratios against the cuEq yardstick)

| Case | Energy | Forces | Stress | Yardstick's largest force error (eV/Å) |
| --- | ---: | ---: | ---: | ---: |
| MH-1 NaCl 3×3×3 | 0.99 | 1.03 | 1.01 | 3.5e-6 |
| MH-1 NbCl5 3×3×3 | 1.01 | 1.00 | 0.97 | 8.9e-5 |
| MH-1 NaNbO3 3×3×3 | 1.01 | 1.00 | 1.03 | 1.3e-4 |
| MH-1 NaNbO3 2×2×2 | 1.03 | 1.00 | 0.92 | 5.0e-5 |
| omat0-medium NaCl 3×3×3 | 0.85 | 0.93 | 0.97 | 4.3e-6 |
| omat0-medium NbCl5 3×3×3 | 0.97 | 0.99 | 0.97 | 8.9e-5 |

## Ranked: stages that pass on their own, with their share of the step

Shares are from the MH-1 force call on the local RTX 5060 Ti (`plans/opt-4.md` §1, main `ccab528`, eager
profiler; NaCl / NbCl5 / NaNbO3).

**Stages that pass on their own and take a real share of the step: none.**

The passing rows, all of them:

| Rank | Stage and mode | Worst ratios (E / F / S) | Share of the force call | Can it be used? |
| --- | --- | --- | --- | --- |
| 1 | ZBL: fp16 output and gradient storage, bf16 output | 1.02 / 1.03 / 1.09 | ≈ 0 (a few elementwise kernels inside "edge geometry", 2.6 / 2.7 / 4.9 % for the whole group) | passes, but nothing to gain |
| 2 | i0 symmetric contraction, TF32 backward (MH-1 only) | 1.01 / 1.25 / 0.89 | a part of the symmetric contraction, which is 3.8 / 5.5 / 3.2 % for both interactions, forward and backward | not general: fails on omat0-medium (forces 3.92 on NaCl). In production it is cuEquivariance's kernel, which has no TF32 switch. |

The nearest misses (≤ 3× in every case), for the owner's question:

| Stage and mode | Worst ratio and where | Share |
| --- | --- | --- |
| i1 symmetric contraction, TF32 backward | stress 2.38 (MH-1 NaNbO3), 2.28 (omat NbCl5) | ≈ 2–3 % |
| i1 conv-tp, TF32 forward / backward | forces 2.13 (NaNbO3), stress 3.42 (NbCl5) | convolution 15 / 17 / 25 % (interaction 2's streaming OEq is most of it) |
| i1 product linear, fp16 output; TF32 forward; fp16 saved | forces 2.46 (NaCl, floor), 2.17 (NaNbO3); stress 3.78 (NaCl) | part of the atom linears, 48 / 43 / 27 % |
| i1 symmetric contraction, fp16 output | forces 2.26 (NaCl, floor), 2.01 (NaNbO3) | ≈ 2–3 % |

## What it says about the owner's idea

- **The error is not concentrated.** Apart from ZBL, every stage that does real arithmetic fails by itself:
  - in fp16 storage by 2–1,100×;
  - in bf16 storage by 7–8,000×;
  - in TF32 forward GEMMs by 2–600×.
- **The most sensitive stages are on the edge side, where the geometry enters:**
  - the radial basis: fp16 output 715–1,050×, bf16 5,820–7,970×;
  - the radial MLP: TF32 forward 318–599×, bf16 autocast 3,180–5,550×;
  - the neighbour sums: fp16 output 28–102×.
  These quantities carry the distance dependence that the forces differentiate, and the error passes straight
  into every force.
- **The readout's backward is the seed of the whole force backward.** TF32 there costs 24–46×, although the
  readout's forward tolerates rounding of its output, which only shifts the energy.
- **The least sensitive are interaction 1's late atom-side stages** (symmetric contraction, product linear,
  conv-tp in TF32). On MH-1 those land within 2–4× of the gate. For output rounding, MH-1's interaction-0
  versions are 3–12× more sensitive (fp16: symmetric contraction 27.7 against 2.26, product linear 8.18
  against 2.46). On omat0-medium both interactions are alike (fp16 output 7–29×).
- **The brief's expectation is borne out.** fp16 and TF32 keep about 11 significant bits (relative rounding
  5e-4), bf16 about 8 (4e-3). Rounding at that level anywhere on the force path, apart from ZBL and one MH-1
  backward, adds 2× to more than 1,000× the float32 calculator's whole force error. So the damping on the way
  to the forces is far below the 1e3–1e4 a 16-bit stage would need.

**Recommendation:** no lower-precision stage is worth building under the current gate. A mixed-precision path
would need a different accuracy contract (a looser gate), which is the owner's decision, not an optimisation.

## Records

- **Machine:** local PC, WSL2, NVIDIA GeForce RTX 5060 Ti (sm_120, 16 GB).
- **Environment:** `~/environments/mace_cueq_latest` (torch 2.14.1+cu130, MACE 0.3.16, cuEquivariance 0.12).
- **Backends:** float64 reference and float32 yardstick on cuEq (stock record
  `fastmace.reference/0.3.16/mh1/763155208cd1ef8c` for MH-1); screened path e3nn float32.
- **Commits:** FastMACE `35761a1` (branch `opt4-T08-precision-screen` from main `1b568f7`); no baseline engine
  is involved.
  - Most rows were measured with the tool as committed.
  - The `tf32-bwd` rows were remeasured after an earlier backward-window method proved to leak TF32 into
    other branches' backward. The engine runs higher sequence numbers first, so an early stage's window
    spanned the whole later backward. Those rows were discarded.
  - The final method recomputes the stage in its own autograd node. It was checked on the CPU, where TF32 has
    no effect: forces and stress came out bitwise identical for all 24 stages.
- **GPU slots:** each run was one `tools/ci/local_gpu.py` reservation, untimed, without `--quiet`, 14:04–15:23
  CEST.
  - MH-1 NbCl5 3×3×3 peaked at 11.4 GB and NaNbO3 at 9.0 GB.
  - NaNbO3's interaction 1 at 3×3×3 ran out of memory after 36 rows and was rerun at 2×2×2.
- **Results:** `results/opt4-T08-precision-screen-20261024/`:
  - `summary.json`: per case, every stage × mode with ratios, verdict and worst errors;
  - `tables.md`: the aggregated tables.
  - The per-copy errors (3.4 MB) stay outside git in `~/fastmace-opt4-t08/merged.json`.

```bash
python tools/precision_screen.py --model mh1-r2scan --structures NaCl NbCl5 --output <dir>
```
