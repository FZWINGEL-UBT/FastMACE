# Fifth list, items D, E and F, and the rebuild-`a` falsifier (30–31 October 2026)

The fifth list of [plans/capacity-ideas-20261030.md](../../plans/capacity-ideas-20261030.md), through the conductor,
plus the conductor's follow-up "rebuild `a` in the backward instead of saving it". MH-1 (`mh1-r2scan`) hybrid
float32, default plan unless named.

**GPUs (decision 66: every result by GPU):**

| Label | Machine | GPU | Driver | Environment |
| --- | --- | --- | --- | --- |
| sm_120 | local PC, WSL2 | RTX 5060 Ti 16 GB | 610.88 | `mace_cueq_oeq`, torch 2.14.1+cu130 |
| Ada | leeloo | RTX 4000 Ada Generation 20 GB (sm_89) | 595.99.02 | NAS `mace_cueq_oeq`, torch 2.14.1+cu130 |
| A100 | Modal | A100-SXM4-40GB | | **not run**: refused by the Modal budget guard (below) |

FastMACE: `59ed08d2` for the A/Bs and gates of D, E and F, `feabaa53` for the falsifier's profile, `77a17974` for
the gates with `gate_epilogue` on and NaCl E2, `0300d53d` for the ledgers. Baseline for the bitwise test: the
research engine at `research-engine-baseline-20261017` (export `1a-T18`).

## Verdict

| Item | sm_120 | Ada | Verdict |
| --- | --- | --- | --- |
| **D.** First reassociation through `segment_gemm`'s joined products | neutral, b/a 0.993–1.002 | prototype 0.25 % slower at NaNbO3 8,640 atoms (both orders) | **Not adopted** ("adopt only if not slower anywhere"). The prototype stays in `tools/bench/reference_kernels.py`. |
| **E1.** `gated_products` with K outer over a stacked tile | 0.1–1.5 % faster | 0.4–2.5 % faster | **Adopted** (it is `gate_epilogue`'s kernel; the option stays opt-in) |
| **E2.** `gate_epilogue` on against off, with E1's kernel | 1.2–4.3 % faster on NbCl5 and NaNbO3, NaCl neutral | 1.8–4.7 % faster on NbCl5 and NaNbO3, NaCl neutral | **A default candidate** on both GPUs: gates passed with it on. The A100 leg, which decides global against per-GPU, is pending budget. |
| **F.** `edge_outer_sum` backward storing `grad_y` once per edge block | neutral, 0.994–1.005 | neutral, 0.999–1.004 | **Not adopted, reverted** (`8a118537`); exact, no gain |
| **Rebuild `a`** (the first interaction's radial-table output, 260 B per edge) | | at the peak at both limits; rebuilding costs about 0.45 % of a force call | **Build** (decision-67 memory mode): `a` is 0.47–0.55 GiB of the limit cells' peak |

**The gates passed** on both GPUs, default plan and `gate_epilogue` on (ratios energy/forces/stress to the float32
ASE reference's spread, limit 1):

| GPU | Plan | NaCl | NbCl5 | NaNbO3 |
| --- | --- | --- | --- | --- |
| sm_120 | default (`59ed08d2`, with E1 and F) | 0.16 / 0.93 / 0.40 | 0.08 / 1.05 / 0.35 | 0.49 / 1.01 / 0.75 |
| sm_120 | `gate_epilogue` on | 0.16 / 0.92 / 0.39 | 0.08 / 1.05 / 0.36 | 0.49 / 1.01 / 0.62 |
| Ada | default (`59ed08d2`) | 0.85 / 0.96 / 0.95 | 0.12 / 1.04 / 0.36 | 0.29 / 0.98 / 0.59 |
| Ada | `gate_epilogue` on | 0.78 / 0.98 / 0.92 | 0.12 / 1.03 / 0.36 | 0.29 / 1.00 / 0.52 |

The forces' ratio above 1 on NbCl5 is inside the gate's limit there (forces limit 2x the ASE spread, or 1e-5 eV/A).

**GPU tests** on sm_120 (`59ed08d2`): `tests/gpu/test_kernels_triton_cuda.py` and
`tests/gpu/test_gated_products_cuda.py`, 23 passed, including `edge_outer_sum`'s backward bitwise against the
research engine and `test_stacked_tile_has_no_register_spills` for D = 3, 5, 7. CPU suite on murphy: 222 passed,
26 skipped, `test_layering.py` included.

## How the A/Bs were made

**Same-process A/B**, `tools/ab_rewrites.py --prepare`, NVT at 300 K, both build orders (ab with pattern abab, ba with
abba). Small cells: NaCl 3x3x3 (216 atoms), NbCl5 3x3x3 (1,944), NaNbO3 4x4x3 (1,920), 12 blocks of 300 steps. Big
cells: NbCl5 4x4x8 (9,216), NaNbO3 6x6x6 (8,640), 100 warm-up steps and 8 blocks of 100 steps. Every run was valid:
nothing compiled while timed, no other process on the GPU.

**Kernel swaps** (D, E1, F): `tools/bench/ab_swap.py` (new) runs side a with `main`'s code for one module attribute
(`tools/bench/reference_kernels.py`) and side b with the library's. It records each engine's side at its build and
sets the side before every step, warm-up re-captures included. So the `--a-constant` bug found in `ab_rewrites.py`
on 31 October (a constant set only around the build) does not apply to it; E2 compares `RewriteConfig` values.

**The trace confirms the sides** (sm_120, NbCl5 3x3x3, the A/Bs' warm-up; `rtx5060ti/trace/`). `ab_swap` counts every
Python call of a swapped attribute by side, phase and the code that ran:

| Item | Side a (reference) | Side b (library) | Wrong side |
| --- | --- | --- | --- |
| F `_triton_backward` | 7 at build, 4 in steps | 7 at build, 4 in steps | 0 |
| E1 `_fused` | 36 at build, 24 in steps | 36 at build, 24 in steps | 0 |
| D `FirstInteractionReassociation.forward` | Dynamo graph [0/0]: 28 `joined_products`, 20 `cat` | graph [0/1]: 18 `joined_products`, 22 `cat` | 0; no recompile after the build |

D's forward is traced by Dynamo, so the counter cannot run inside it (a global mutation there recompiles during the
capture); its sides come from `TORCH_LOGS=recompiles,graph_code`.

## D. The first reassociation through joined products

**What the prototype does.** In the first interaction's reassociation, the per-path products go through
`segment_gemm.joined_products`: one Triton launch per path where the shape rule picks Triton, written into the joined
output. `main` runs one matmul per path, then a `cat`.

**Same-process A/B** (`ab_swap`, side a = the prototype, side b = the library; b/a > 1 means the prototype is
faster):

| Cell | Atoms | sm_120 ab / ba | Ada ab / ba |
| --- | ---: | --- | --- |
| NaCl | 216 | 0.998 / 0.993 | 0.995 / 1.001 |
| NbCl5 | 1,944 | 0.999 / 1.002 | 1.002 / 1.001 |
| NaNbO3 | 1,920 | 1.000 / 1.001 | 1.000 / 0.998 |
| NbCl5 | 9,216 | | 1.001 / 1.001 |
| NaNbO3 | 8,640 | | **0.997 / 0.998** |

The conductor's rule was to adopt D only if it is not slower anywhere. On Ada it is 0.2–0.3 % slower at NaNbO3 8,640
atoms in both orders, and nowhere more than 0.2 % faster. So D is not adopted.

## E. `gated_products`: K outer over a stacked tile

**The change** (`fastmace/kernels/triton/gated_products.py`). `gate_epilogue`'s kernel computes, per gated segment,
`dgated = (g @ Wᵀ) * gate` and `dgate = Σ_m (g @ Wᵀ)_m * gated_m`. `main`'s kernel looped over the D components
outside and K inside, so it loaded each weight tile D times. The new kernel stacks every component of `BLOCK_A` atoms
into one tile of `DP * BLOCK_A` rows (`DP` = D rounded up to a power of two, `BLOCK_A = 128 // DP`), runs one K loop
with one `tl.dot` per K block, and reduces `dgate` over the components in the epilogue. The arithmetic per output
element is the same; the order of the K sum is unchanged. No register spills for D = 3, 5, 7 (new GPU test).

**E1, the kernel against `main`'s, `gate_epilogue` on both sides** (`ab_swap`, b = the new kernel):

| Cell | Atoms | sm_120 ab / ba | Ada ab / ba |
| --- | ---: | --- | --- |
| NbCl5 | 1,944 | 0.989 / 0.993 | 0.978 / 0.975 |
| NaNbO3 | 1,920 | 0.991 / 0.985 | 0.993 / 0.991 |
| NbCl5 | 9,216 | 0.998 / 0.997 | 0.995 / 0.994 |
| NaNbO3 | 8,640 | 0.999 / 0.996 | 0.995 / 0.996 |

**E2, `gate_epilogue` on against the default (off)** (`ab_rewrites.py --b gate_epilogue=on`):

| Cell | Atoms | sm_120 ab / ba | Ada ab / ba |
| --- | ---: | --- | --- |
| NaCl | 216 | 1.000 / 1.005 | 0.998 / 1.003 |
| NbCl5 | 1,944 | 0.983 / 0.982 | 0.976 / 0.975 |
| NaNbO3 | 1,920 | 0.982 / 0.988 | 0.982 / 0.981 |
| NbCl5 | 9,216 | **0.958 / 0.957** | **0.953 / 0.953** |
| NaNbO3 | 8,640 | 0.972 / 0.968 | 0.968 / 0.969 |

NaCl's ratios lie inside its block ranges (0.98–1.03 on sm_120, 0.997–1.010 on Ada): neutral. The gain grows with the
cell. Before E1, `gate_epilogue` was not a win (`docs/validation/opt10-gate-epilogue.md`).

**The A100 leg** (`tools/modal/speed_def_e2.sh`: the gate with `gate_epilogue` on and the E2 A/Bs) was refused by the
budget guard on 31 October: this month's metered cost was $26.69, the job's worst case $3.42, against the $27
limit. It needs next month's credit or the owner's override. Until then, decision 66 would put `gate_epilogue` on by
exact device name for the RTX 5060 Ti and the RTX 4000 Ada only.

## F. `edge_outer_sum`'s backward, one `grad_y` store per edge block

**The change** (measured at `59ed08d2`, reverted at `8a118537`): the backward kernel kept `grad_y` for its D
components in registers and stored them once as a `[BLOCK_E, BLOCK_D]` tile, instead of D strided stores. Bitwise
equal to the research engine's (GPU test).

**Same-process A/B** (`ab_swap`, b = the new store):

| Cell | Atoms | sm_120 ab / ba | Ada ab / ba |
| --- | ---: | --- | --- |
| NaCl | 216 | 0.998 / 0.999 | 0.999 / 1.004 |
| NbCl5 | 1,944 | 0.996 / 1.002 | 1.002 / 1.000 |
| NaNbO3 | 1,920 | 1.004 / 0.997 | 1.001 / 0.999 |
| NbCl5 | 9,216 | 1.001 / 1.005 | 1.001 / 1.000 |
| NaNbO3 | 8,640 | 1.004 / 0.994 | 1.001 / 1.001 |

No gain on either GPU, so the kernel is back to `main`'s (as item B of the fourth list: the conductor's rule for a
neutral change). The A/B stays reproducible: `reference_kernels.edge_outer_sum_backward` is `main`'s kernel.

## The rebuild-`a` falsifier

**The idea** (conductor, 30 October). The first interaction's radial table output `a` ([E, 65] float32, 260 B per
edge: the folded radial input with the cutoff folded in) is saved for `edge_outer_sum`'s backward. Rebuilding it there
from the edge lengths would free it for the rest of the call. The conditions: build it only if `a` is alive at the
peak with `recompute_node_segments` on at the limits, and the rebuild costs at most 3 % when on.

**Cost of the rebuild** (Ada, NbCl5 4x4x8, 9,216 atoms, 270,336 padded edges, 1000 K; `tools/step_breakdown.py`'s
nsys pass, 10 force-graph replays with every graph node traced; `rtx4000ada/profile/`). Each force call runs the
radial table forward twice, once per interaction: 427–504 µs per launch, against 101.9 ms per force call. Rebuilding
interaction 0's table in the backward is one more launch: **about 0.45 % of the force call**.

**Is `a` at the peak?** `tools/bench/memory_ledger.py` (now with `--rewrites`) with `recompute_conv_weights` and
`recompute_node_segments` on, Ada; the blocks alive at the largest live total while one step captures again into a
fresh pool (`rtx4000ada/ledger/`):

| Cell | Atoms | Padded edges | Peak over the step's start | `radial_table_forward` alive at the peak |
| --- | ---: | ---: | ---: | ---: |
| NbCl5 5x5x12 | 21,600 | 596,736 | 3.16 GiB | 0.144 GiB (260 B/edge), 4.6 % |
| NbCl5 6x6x27 (the sm_120 limit) | 69,984 | 1,935,360 | 10.18 GiB | 0.469 GiB, 4.6 % |
| NaCl 20x20x21 (the sm_120 limit) | 67,200 | 2,257,920 | 9.91 GiB | 0.547 GiB, 5.5 % |

One block is alive, interaction 0's: with `recompute_conv_weights` on, interaction 1's convolution rebuilds its
weights on chip and keeps no table output. The 51,840-atom ledger of CAP-T12 (default plan) showed the same
block (0.348 GiB).

**Verdict: build.** It is at the peak at both limits, and costs about 0.45 % when on. As a decision-67 memory mode it
is planned only when the plan does not fit; the rewrite follows on its own branch.

## Records

`results/speed-def-20261030/`:

- `local.sh`, `leeloo.sh`: the run scripts; `rtx5060ti/jobs.log`, `rtx4000ada/jobs.log`: idle checks and exits.
- `rtx5060ti/ab/`, `rtx4000ada/ab/`: every A/B record (`tools/ab_rewrites.py` JSON).
- `rtx5060ti/trace/`: `ab_swap`'s traces and D's recompile log.
- `*/gate-summary.txt`, `*/gate-epi-summary.txt`: the gates' per-structure summaries.
- `rtx4000ada/profile/`: the 9,216-atom profile (summary, kernel sequence, nsys trace, gzipped).
- `rtx4000ada/ledger/`: the three ledgers.
