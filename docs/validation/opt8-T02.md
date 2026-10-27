# OPT8-T02/T03/T04: the record `fused_folded_forward`, default on for MH-1 (27 October 2026)

Plan: `plans/opt-8.md`. The probe it comes from is OPT8-T01b (`docs/validation/opt8-T01.md`). The owner asked for
the record ("Yes build it"); after the gates it made it **the default for MH-1** ("Make it default for mh-1",
owner decision 64; OPT8-T04 below). Runs and scripts:
`results/opt8-T02-fused-folded-forward-20261027/`.

**Verdict: exact, gate passed, NVE drift unchanged.**

| Cell | Local RTX 5060 Ti (sm_120), b/a, both build orders | murphy RTX 4000 Ada (sm_89), b/a, both build orders | Colab L4 (sm_89), b/a, order ab |
| --- | --- | --- | --- |
| NaNbO3 | **0.928 / 0.927** (−7.2 %) | **0.958 / 0.958** (−4.2 %) | 0.977 |
| NbCl5 | **0.970 / 0.964** (−3.3 %) | **0.983 / 0.982** (−1.8 %) | 0.991 |
| NaCl | 0.991 / 1.000 (neutral) | **0.980 / 0.983** (−1.9 %) | not run |

- **POLAR-1-S:** takes the record; the A/B is neutral (1.004).
- **OMOL-0:** takes the record, but it is **3.5 % slower** there (1.035).

## What was built

**`fastmace/kernels/oeq_streaming/fused_forward.py`.** The kernel is generated from the streaming schedule, like
the vendored kernels, and compiled with NVRTC through the existing runtime; there is no new dependency.
- One kernel builds `w = a @ [W, b].T` and the streaming message.
- It stores `w` for the unchanged chunked backward.
- Edges past the last CSR row (the neighbour list's padding) also get their `w` rows, so the backward sees what
  the chain built.
- Both node layouts (`ir_mul`, `mul_ir`) are supported; any uvu problem with one edge channel per path and
  channels divisible by 32; float32 only.
- Custom op `fastmace::fused_folded_forward`.

**Launch rule (general, documented in the module).** It is derived from the problem and from the device's opt-in
shared memory per block and SM count only; there is no table and no shape cache.
- **Paths per part:** at most 40 / 8 (the part's `w` tile is at most 40 registers per lane).
- **Parts:** the fewest that fit the `W` slice (row stride 33) beside at least 4 warps' staging.
- **Warps:** up to 12 beside the slice (168 registers per thread on a 64K-register SM).
- **Grid:** `ceil(2 × SMs / slices)` blocks per slice.
- **Constants:** a batch of 8 edges, chunks of 2 receivers from a per-slice counter, and 64 padding edges per
  tail chunk.

For MH-1 on sm_120 the rule gives the configuration OPT8-T01 found best by its sweep: two parts, 12 warps,
74 KB of shared memory, 9 blocks per slice.

**Wiring.**
- `folded_streaming_convolution(..., fused=None)`: the autograd function takes the fused op in its forward and
  keeps today's backward.
- The MH-1 folded forward passes the interaction's attached `fused_folded_forward`.
- **Record** `fused_folded_forward` (`fastmace/rewrites/mh1`):
  - requires `chunked_conv_backward`;
  - exactness `exact`; default `auto` (decision 64; `off` until then);
  - workload constraints OEq, sorted receivers and the per-engine edge counter;
  - not deterministic (the backward's atomics);
  - refuses a float64 model.
- **`RewriteConfig.fused_folded_forward`:**
  - `auto` rule `never_automatic`;
  - `on` requires `chunked_conv_backward`, otherwise it is refused;
  - `on` in a float64 plan is refused (`fused-folded-forward-needs-float32`): float64 plans keep the cuBLAS
    product.
- **Docs and tests touched:** both tables of `docs/design/switch-map.md`, `CHANGELOG.md`, and the golden plan
  texts (the field shows as off).

It works with and without `radial_tables`: it reads the folded layer's input `a`, whichever op produced it.

**Exactness.** The same FP32 sums in the same order: each `w` element is a k-ordered chain of fused
multiply-adds, and each receiver's edges are summed in CSR order with the vendored angular terms.
- `tests/gpu/test_fused_folded_forward_cuda.py` checks both layouts and folded widths 9 and 65, with padding
  edges and a receiver without edges.
- The stored `w` equals cuBLAS's `a @ W.T` to rounding, and was **bitwise** equal in all four cases on the RTX
  5060 Ti.
- The message is bitwise the streaming forward's on the stored `w`, and bitwise the chain's.
- The gradients through `folded_streaming_convolution` equal the chain's. `dx` matches to 1e-6 of its largest
  value: the atomic sender sums differ between any two runs.
- A CUDA graph replays the result bitwise.
- On the recorded MH-1 operands (OPT8-T01b), `w`, the message, `dsh` and `da` were bitwise the chain's.

**CPU tests** (`tests/cpu/test_fused_folded_forward.py`, plus the record lists in `test_rewrites_mh1.py`,
`test_config.py` and `test_report.py`): the launch rule on sm_120, sm_89, A100 and a 48 KB device; the path
balancing; the generated source for both layouts; refusals; NVRTC compilation for sm_120 and sm_89; the record;
and the planner's resolution and refusals. Targeted runs with `mace_cueq_latest` (`-n 4` to `-n 6`, CPU only):
306 passed, 20 skipped in the planner, report, plan-text and record files; ruff clean.

## Gates (local RTX 5060 Ti, driver 610.88, `mace_cueq_oeq`; pinned source `9eb1ab1e`; `local-9eb1ab1e/`)

**Force/stress gate** (`tools/force_stress_gate.py --engine fastmace --rewrites fused_folded_forward=on`, MH-1,
3 configurations per structure and cell mode): **passed** on all three cells. Error against float64, and its
ratio to the float32 ASE calculator's error:

| Cell | energy (eV) | forces (eV/Å) | stress (eV/Å³) |
| --- | ---: | ---: | ---: |
| NaCl | 6.4e-5 (0.085) | 3.7e-6 (0.97) | 3.3e-8 (0.36) |
| NbCl5 | 8.0e-5 (0.11) | 8.4e-5 (1.06) | 3.6e-8 (0.35) |
| NaNbO3 | 4.0e-4 (0.32) | 1.3e-4 (1.01) | 2.1e-7 (0.67) |

**NVE drift** (`tools/bench/nve_drift.py`, 20 ps at 1 fs, 300 K; meV/atom/ps, rms of the conserved quantity in
meV/atom):

| Cell | off | on |
| --- | --- | --- |
| NaCl | 1.1e-6, rms 2.7e-4 | 1.5e-7, rms 2.7e-4 |
| NbCl5 | −9.1e-6, rms 3.8e-4 | −2.7e-6, rms 3.8e-4 |

**Same-process A/B** (`tools/ab_rewrites.py`, a = `fused_folded_forward=off`, b = `on`, MH-1 hybrid float32 NVT
300 K, 12 blocks of 300 steps). Median step in ms; `ab` uses the abab block pattern, `ba` abba:

| Cell | order ab: a / b, b/a | order ba: a / b, b/a | pair ratios |
| --- | --- | --- | --- |
| NaCl | 2.679 / 2.654, 0.991 | 2.708 / 2.709, 1.000 | 0.97–1.02 |
| NbCl5 | 19.24 / 18.67, **0.970** | 19.31 / 18.62, **0.964** | 0.95–1.00 |
| NaNbO3 | 15.69 / 14.56, **0.928** | 15.73 / 14.58, **0.927** | 0.91–0.94 |

These match T01b's subgraph savings: 0.66 ms (NbCl5) and 1.10 ms (NaNbO3) per force call, against measured
0.58–0.69 ms and 1.13–1.15 ms per step.

## Other GPUs and models

**murphy, RTX 4000 Ada Generation (sm_89, driver 595.99.02, NAS `mace_cueq_oeq`; source `059b2cfa`;
`ada-murphy-059b2cfa/`, script `ada.sh`).** Run by the conductor on 27 October 2026, 19:08–19:36 UTC, after the
idle check (no GPU compute process, no other user's job); every job repeated the check.
- **Force/stress gate** (`--engine fastmace --rewrites fused_folded_forward=on`, reference `mace_cueq_latest`):
  passed. Ratios to the float32 ASE calculator's error, energy / forces / stress: NaCl 0.70 / 1.03 / 0.98,
  NbCl5 0.12 / 1.04 / 0.35, NaNbO3 0.29 / 0.99 / 0.52.
- **GPU test** (the child script of `tests/gpu/test_fused_folded_forward_cuda.py`, run directly with the hybrid
  environment because it has no pytest; the first `check` entry in `run.log` is that missing pytest): passed.
  `w` and the message are bitwise equal to the chain for K = 65; for K = 9 they differ by FP32 rounding from
  cuBLAS on sm_89 and pass the tolerances.
- **A/B** (12 blocks of 300 steps, both build orders; median ms, a = off, b = on):

  | Cell | Order ab: a / b, b/a | Order ba: a / b, b/a | Pair ratios |
  | --- | --- | --- | --- |
  | NaNbO3 | 18.13 / 17.37, **0.958** | 18.28 / 17.50, **0.958** | 0.955–0.963 |
  | NbCl5 | 21.92 / 21.54, **0.983** | 22.01 / 21.62, **0.982** | 0.981–0.984 |
  | NaCl | 2.867 / 2.811, **0.980** | 2.859 / 2.809, **0.983** | 0.976–0.985 |

  Every block pair favours the record on all three cells. Unlike the local GPU, NaCl gains too.

**Colab L4 (sm_89, driver 580.82.07; `colab-l4-9eb1ab1e/`).** The workstations were busy (leto with the
conductor's check, leeloo and coraline with other users), so the Ada check ran on Colab. One A/B per cell,
order ab:
- **NbCl5:** 31.74 → 31.47 ms (0.991; pairs 0.984–1.002);
- **NaNbO3:** 26.66 → 26.06 ms (0.977; pairs 0.963–0.988).

The L4's lower FP32 rate and bandwidth shrink the gain. Run from Colab session `fm-l4-20261027-173343-2ef458`,
stopped afterwards; 1.2 compute units.

**MH-1 family**, both on the local GPU:
- **POLAR-1-S, NbCl5:**
  - The plan resolves the record on, and the probe `attach_probe.py` shows the kernel attached (5 paths, one
    part).
  - A/B **1.004** (pairs 0.99–1.01): neutral.
  - The probe saw the kernel run only in the setup self-checks, not in the full-size force call of that eager
    probe. Why was not investigated; the A/B timing is the measurement.
- **OMOL-0 XL, NbCl5:**
  - The default plan fails at setup on this model **without the new record**: `radial_tables` fails its
    self-check (relative errors 1.4–2.2e-4 > 1e-4 in interactions 1 and 2; `omol-nbcl5-ab-default-plan.log`).
    `ab_rewrites` does not re-plan, so both sides ran with `radial_tables=off`.
  - The record attaches to all four interactions: folded width 129, 5 parts and 7 warps for the 17-path
    interactions, shared-memory bound.
  - A/B **1.035** (pairs 1.031–1.044): **slower**. The wider `a` (129) costs the GEMM more than the `w` read
    saves, with fewer warps.

## Decisions for the owner

1. **Default or not.** Decided: on by default for MH-1 (decision 64, OPT8-T04 below). On this PC it is −7.2 % (NaNbO3) and −3.3 %
   (NbCl5) per MD step; on the RTX 4000 Ada (murphy) −4.2 %, −1.8 % and −1.9 % (NaCl), on an Ada-class L4
   −2.3 % and −0.9 %. It is neutral on NaCl locally and on POLAR-1-S, and +3.5 % on OMOL-0. A default would need a rule: the MH-1 width, or a gain model from the folded width and the
   edge count.
2. **Separate from this record: OMOL-0's default plan fails at setup.** `radial_tables` (default on since
   decision 62) fails its self-check on `omol0-xl` here. The planner's self-check revision (SPEC §9.3 rule 13)
   should re-plan without it, but `tools/ab_rewrites.py` does not. Production through `fastmace run`, or any
   path that does not revise, would stop on it.

## Not done

- No Colab `ba` order and no NaCl on the L4 (the murphy run covers sm_89 in both orders).
- MH-1 family models were not run on Ada.
- The POLAR-1-S hot path was not traced further.
- The CPU full suite: per the conductor, it runs before merging.

## OPT8-T04: default for MH-1 (owner decision 64)

The owner chose "Make it default for mh-1" after the gates above and the murphy check. Rule
`on_for_mh1_float32_unless_reproducible`:

- **On:** a label-table label of the `mh-1` checkpoint (`mh1-r2scan`, `mh1-omat`, `mh1-mp`, found by label or by
  SHA-256 and head), a float32 plan that is not reproducible, and `chunked_conv_backward` on.
- **Off (auto):** every other model, with the stage-2 reason "not MH-1, decision 64"; float64 and reproducible
  plans. POLAR-1-S (neutral) and OMOL-0 XL (3.5 % slower) are the measured reasons; other models were not timed.
- **Explicit `"on"`:** still enables it for any float32 model with `chunked_conv_backward`; float64 refuses it.
- MH-1 heads outside the label table (for example `spice_wB97M`) and bare models planned without a label have
  no label, so they keep it off unless asked (as decision 58's POLAR-1 defaults).

Planner, switch map, SPEC §18 decision 64, CHANGELOG and the CPU tests changed; the golden plan texts were
regenerated. No new GPU measurement: the A/Bs above are the measured basis.
