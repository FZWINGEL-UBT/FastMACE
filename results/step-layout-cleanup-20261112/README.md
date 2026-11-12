# step-layout-cleanup: time we don't own removed from the captured MD step (12 November 2026)

Brief (conductor, 12 November 2026): remove the work and copies that the torch 2.15 CUDA-graph attribution found in
the captured step outside our kernels ([torch-215-trial §6a](../torch-215-trial-20261112/README.md), branch
`fastmace/torch-215-trial` `511957f4`, Ada profile on `b7c19b19`). Branch `fastmace/step-layout-cleanup`, based on main
`1d491a67`. Items 1–2 here are bitwise work removals (decision 66: one not-slower A/B on any GPU; no A100 or H200
run). Items 3–4 wait for the A100 attribution (conductor); items 5–6 follow below.

| | SrTiO3 10³ (5,000 atoms) | NbCl5 3³ (1,944 atoms) |
|---|---|---|
| Ada MD step b/a, ab / ba (items 1 + 2 against main's code) | **0.9317 / 0.9276** (17.29 → 16.11 ms) | **0.9260 / 0.9293** (3.60 → 3.33 ms) |
| of which `JOINED_LAYOUT` on the Triton backend alone | 0.9586 / 0.9532 | 0.9697 / 0.9729 |
| Time we don't own per step (2.15 attribution, §6a `b7c19b19` → `718caab6`) | 3,778 → **2,273 µs** | 634 → **342 µs** |
| Pure-layout Inductor kernels (ledger bound `i0-copies-zero-fill`, layout part) | 849 → **73 µs** | 128 → **23 µs** |
| Species-only work (ledger `species-only-kernels`) | 126 → **11 µs** (the radix sort gone) | 57 → **5 µs** |
| Exactness | bitwise: the same products, sums and divisions; only where they are read from and written to | same |

The attribution numbers include main's compose-outside-graph fix (the CUTLASS compose, 301 / 278 µs on `b7c19b19`),
which landed between §6a's profile and this branch; every row below is per source line, so the two are apart.

## Item 1: the species plan as derived state of the species

**What ran every step.** `species_gemm.species_plan` (the stable `argsort` of the atoms' species, a cub radix sort,
and an `index_add` count) was called by every grouped species product: COMPOSE_SKIP's `_composed`
(`rewrites/mh1/modules.py`) and the generic `SpeciesSkip` (`SPECIES_GEMM`). Inside the captured step it replayed every
step: 112 / 49 µs (SrTiO3 / NbCl5) for inputs that change only with the species.

**The change** (`kernels/triton/species_gemm.py`, `model/forces.py`, both module forwards):

- `ForceField._setup_plans` builds the plan once per column set (`species_gemm.plan_of`: kind, order, counts) for every
  module that groups by species (`plan_columns()`), into fixed buffers.
- The plan is derived state of the species: `species_plan.<g>.kind / .order / .counts`, depends on `species`,
  registered after `node_attrs` (whose one-hot it reads). A species event (host transaction, MC swap or transmutation,
  also inside an attempt graph: the refresh is capturable) rebuilds it in place; a snapshot restore (a capture's
  warm-up, a rejected transaction) brings it back with the species.
- `_bind_streaming_count` binds it onto the modules before each whole-system call (`species_gemm.bind_plan`, a
  host-side buffer swap like the species table's); the tiled call unbinds it (`plans=False`), and a released one-hot
  has none.
- A forward reads the bound plan (`bound_plan`) only where its rows are exactly the plan's (not a tile, not the real
  rows of a ghost-padded call); otherwise it sorts as before.
- **No version-counter cache.** The registry refreshes the plan explicitly, so the capture's snapshot restore (same
  values, new versions) is no species change. The trap compose-outside-graph found did bite once, elsewhere: the
  plan buffers bound onto the skip were part of `_compose_versions` (every buffer of the skip), so on `c6511d8e` the
  captured bind saw "changed parameters" and the four d884 compose GEMMs were back in the graph
  ([superseded/attribution-c6511d8e](superseded/attribution-c6511d8e/)). Fixed on `718caab6`: the plan buffers are
  no compose input; regression test `test_a_restored_species_plan_on_the_skip_composes_nothing` (fails without the
  fix: two compositions).

## Item 2: the layout copies around the species products

**`generic/modules.py:211` (first interaction, 563 / 80 µs) and `tabled_outer_sum.py:739` (its backward, 358 / 19
µs).** On the Ada the tf32x3 backend is Triton, and `JOINED_LAYOUT` (the per-slot sums written d-major so that each
path is a row group read in place by `species_mm_rows`) required CUTLASS. So the Ada took the per-path route: each
path's `summed[:, :, l-block].transpose(1, 2).reshape(...)` (Inductor `clone_slice_species_mm_transpose`, three kernels),
`species_mm`, and `torch.cat` of the outputs (`cat_view_24`); the backward assembled the gradient through
`slice_backward` and transposes. (The 8 November Ada A/B of `JOINED_LAYOUT`, a tie, measured it inert for that reason.)

- `species_mm_rows` and `species_mm_joined` on Triton launch the kernel on strided views: each path's rows are read in
  place and written straight into the joined output (the kernel takes any strides); the gradient likewise. Before,
  the Triton fallback copied every input view and every output block.
- `JOINED_LAYOUT` applies on the Triton backend too; the RTX 4000 Ada is added to `JOINED_LAYOUT_GPUS`.
- Bitwise: the same kernel on the same values (tests below compare against the copied route with `torch.equal`,
  forward and gradient, eager and compiled).

**`generic/modules.py:530` (interaction 2, 481 / 64 µs forward, 313 / 65 µs backward).** `reshape_irreps(message)` is
cuEquivariance's ir_mul cat of `[n, dim, mul]` pieces: with one multiplicity that is the message's own memory order.
`_reshaped` returns it as a view (`linear.modules.reshape_view`, already used on the CUTLASS paths), so Inductor no
longer writes a second copy fused with the division. Left at :530: the join of the linear's block outputs fused with
the division (`add_cat_div_view`, 207 / 34 µs) and the division's backward reduction for the density
(`red_fused_add_div_mul_neg_sum`, 182 / 24 µs): compute, not layout (item 5 tries the CUTLASS epilogue for them).

## A/B (RTX 4000 Ada, murphy; OMAT-0 medium float32, production plan)

`tools/bench/ab_swap.py`, same process, both build orders, SrTiO3 10×10×10 and NbCl5, 1 fs, skin 0.5, 200 warm-up and
6 × 256 timed steps. Side a runs main's code (`tools/bench/reference_kernels.py`: `no_bound_plan`, `reshape_copy`, and
`JOINED_LAYOUT=False`); side b the branch. [jobs/ada.sh](jobs/ada.sh) jobs `ab-*` and `jl-*`, records in
[ada-murphy/](ada-murphy/), commit `718caab6`.

| | SrTiO3 ab | SrTiO3 ba | NbCl5 ab | NbCl5 ba |
|---|---:|---:|---:|---:|
| Items 1 + 2 (plan, joined layout, reshape view) | 0.9317 | 0.9276 | 0.9260 | 0.9293 |
| `JOINED_LAYOUT` alone (ab_rewrites, side a off) | 0.9586 | 0.9532 | 0.9697 | 0.9729 |

Potential energy per atom after the timed steps agrees to 1e-7 eV between the sides (same trajectory start; forces
are not bitwise between two builds of this plan, scatter atomics). Swap trace: 0 calls on the wrong side.

**Tool fix found on the way:** `ab_swap.py` appended its own `--a-constant ab_swap:SIDE='a'` after the caller's, and
argparse keeps only the last occurrence, so a caller's side-a constants were silently dropped. It now joins the
caller's list (`tests/cpu/test_ab_swap_constants.py`). Earlier `ab_swap` runs that passed `--a-constant` should be
checked (compose-outside-graph did not).

Superseded records ([superseded/](superseded/)): run 1 on `c6511d8e` (the side-a `JOINED_LAYOUT=False` dropped, so it
measured plan + view only: 0.9747 / 0.9687, 0.9598 / 0.9614) and run 2 on `c819ccd8` (both sides with the compose
regression: 0.9333 / 0.9283, 0.9314 / 0.9325; JOINED_LAYOUT alone 0.9583 / 0.9529, 0.9738 / 0.9759).

## Attribution after the change (leeloo, torch 2.15 profiling env, `718caab6`)

Same recipe as §6a (`jobs/ada.sh prof`: t215 `profile_point.py` with `comment_origin`, PYSTACKS, 5 annotated steps;
`attribution.py`). Records: [ada-leeloo/](ada-leeloo/).

| Site (µs per step) | SrTiO3 before → after | NbCl5 before → after |
|---|---|---|
| `generic/modules.py:211` (first interaction copies + cat) | 563 → **0** | 80 → **0** |
| `generic/modules.py:530` forward | 481 → 207 | 64 → 34 |
| `generic/modules.py:530` backward | 313 → 182 | 65 → 24 |
| `tabled_outer_sum.py:739` backward gather + transpose | 358 → **0** | 19 → **0** |
| species plan radix sort in the graph (`lean_forward`, eager ATen) | 114 → **0** | 50 → **0** |
| cuEq inside the graph at that frame (moved to `forces.py:1261`, unchanged work) | 111 → 111 | 17 → 16 |
| `tabled_outer_sum.py:600` Memset (item 3) | 192 → 188 | 17 → 15 |
| `tf32x3_gemm.py:171` clone / slice_backward (item 4) | 185 → 220 | 30 → 35 |
| `tabled_outer_sum.py:757` | 150 → 164 | 17 → 8 |
| GPU kernel time per step | 16,878 → 15,195 | 3,372 → 2,851 |

## Tests

- `tests/cpu/test_species_plan_state.py` (new): `plan_of` equals the forward's plan (radix and counting); a bound plan
  covers its rows only and is never saved; the composed forward reads it bitwise and sorts no more; the force field
  binds one plan per column set, a species refresh rebuilds it in place (the one-hot first) and equals a fresh
  construction (`check_against`, outputs bitwise), a snapshot restore brings it back, the tiled bind and a released
  one-hot unbind it.
- `tests/cpu/test_step_layout_cleanup.py` (new): `_reshaped` is a view with reshape's values (ir_mul) and keeps the
  copy elsewhere; the joined layout runs on the Triton backend and equals the per-path route.
- `tests/gpu/test_step_layout_cleanup_cuda.py` (new; Ada, murphy): `species_mm_rows` reading and writing strided views
  equals the copied route bitwise (ieee and tf32x3, eager and compiled, forward and gradient); the compiled joined
  route has no cat, clone or sort kernel (Inductor code), where the old route has its cat.
- `test_compose_outside_graph.py`: the regression test above. `test_compose_skip.py`, `test_gpu_models.py`: the Ada in
  `JOINED_LAYOUT_GPUS`.
- Targeted CPU runs on murphy (CUDA hidden): 61 + 67 passed; GPU tests 28 passed. Full suite: see SPEC entry.

## Items 3 and 4 (not started; conductor: wait for the A100 attribution)

- **Item 3, the outer-sum zero-fill** (`tabled_outer_sum.py:600`, 188 / 15 µs): needed as the kernel stands. The
  forward accumulates each slot with `tl.atomic_add` from every edge chunk that holds part of the slot's segment, and
  slots without edges (a receiver without senders of a species) must read zero. Folding the zeroing into the write
  needs a store-once segmented kernel, which changes the summation order: a rounding change.
- **Item 4, `tf32x3_gemm.py:171`** (220 / 35 µs): the BlockLinear's input rows `x[:, segment].reshape(n * dim, mul)`
  are not a uniform-stride matrix, so Inductor copies them (and assembles the gradient by `slice_backward`). On the
  Ada, item 5's CUTLASS backend takes it to 26 / 5 µs; on the A100 and H200 (CUTLASS already) the A100 rows decide.

## Item 5: the CUTLASS backend with LINEAR_DIV on the Ada (owner-approved; decision 66, gate passes)

**The first A/B measured nothing.** Side b set `tf32x3_gemm.CUTLASS_CAPABILITIES` to include 8.9, as
ada-5060-screen did on 10 November. But `cutlass.tf32x3.available()` accepts only capabilities with their own
cubin (8.0, 9.0), so `backend()` stayed "triton" on both sides: b/a 1.0053 / 0.9979 SrTiO3, 0.9995 / 0.9992 NbCl5,
and the profile was the baseline's kernel for kernel ([superseded/ada-murphy-item5-null](superseded/ada-murphy-item5-null/)).
**The same holds for ada-5060-screen's "CUTLASS tf32x3 on sm_89" (1.0025 / 0.9998) and `SEGMENT_ROWS` (0.9967 /
0.9981) rows: both ran on Triton.** `cutlass.tf32x3.FAMILY_CAPABILITIES` (an A/B constant) now lets another 8.x
device load the sm_80 cubin (its SASS runs on the whole 8.x family).

**A/B with CUTLASS really on** (murphy, `4ae5164c`, [ada-murphy-item5/](ada-murphy-item5/); side a today's Ada
default with items 1 + 2, i.e. Triton tf32x3; side b CUTLASS + `LINEAR_DIV`, `lds`: also `SEGMENT_ROWS`):

| Side b | SrTiO3 ab | SrTiO3 ba | NbCl5 ab | NbCl5 ba |
|---|---:|---:|---:|---:|
| CUTLASS + `LINEAR_DIV` | **0.9364** (16.13 → 15.10 ms) | **0.9335** | **0.8912** (3.34 → 2.97 ms) | **0.8887** |
| CUTLASS + `LINEAR_DIV` + `SEGMENT_ROWS` | 0.9421 | 0.9374 | 0.8986 | 0.8999 |

`SEGMENT_ROWS` is slower than without it in all four cells: off.

**Gate** (`jobs/ada.sh gate`, murphy, [ada-murphy-gate/](ada-murphy-gate/); OMAT-0 medium, NaCl and NbCl5 3×3×3,
`force_stress_gate.py` 3× float32 rule): passes. Production / float32 yardstick ratio: forces 0.87 / 0.98, stress
1.11 / 1.26, energy 0.33 / 2.53 (energy within its 0.01 meV/atom floor). No rule-13 revisions.

**Attribution** (leeloo, [ada-leeloo-item5/](ada-leeloo-item5/), same recipe; SrTiO3 / NbCl5, µs per step):

| Site | Triton backend (`718caab6`) | CUTLASS + `LINEAR_DIV` |
|---|---|---|
| GPU kernel time | 15,195 / 2,851 | **14,741 / 2,765** |
| Time we don't own | 2,273 / 342 | **2,042 / 324** |
| atom GEMMs: Triton `tf32x3_mm` | 1,631 / 545 | → CUTLASS `mm_into` 671 / 291 + `remap_div_into` (LINEAR_DIV, the division in the epilogue) 750 / 252 |
| species products `species_mm_rows` + grad (Triton) | 999 / 266 | → CUTLASS `species_into` 907 / 211 |
| `generic/modules.py:530` cat + div (`add_cat_div_view`) / its backward reduction | 207 / 34, 182 / 24 | **0 / 0, 0 / 0** |
| `tf32x3_gemm.py:171` clone / slice_backward | 220 / 35 | **26 / 5** |
| `generic/modules.py:529` (LINEAR_DIV's backward: per-segment slice / divide / clone) | – | 357 / 56 |

What the conductor listed: the join (cat) and `add_cat_div_view` and `red_fused_add_div_mul_neg_sum` go away; the
tf32x3:171 clone almost does (220 → 26). The division's backward survives as `:529` (357 µs), so the Triton
strided store / load was not built (the Ada no longer runs these products on Triton). The existing
`LINEAR_DIV_SPLIT` (h200-fastpath: the divided gradient as contiguous blocks in one kernel) targets exactly
`:529`; its A/B is below.

**Adopted on the Ada** (decision 66, rounding change, gate passes): `tf32x3_gemm.CUTLASS_GPUS` and
`cutlass.tf32x3.FAMILY_GPUS` (the RTX 4000 Ada by model: no other sm_89 GPU such as the L4 was measured) and the
Ada in `linear_div.LINEAR_DIV_GPUS`. `CUTLASS_CAPABILITIES = CUTLASS_GPUS = ()` still puts Triton everywhere (A/B).
The 5060 Ti (sm_120) has no CUTLASS cubin and is unchanged. Tests: `test_gpu_models.py`
(`test_the_ada_takes_the_cutlass_backend_with_linear_div`), `test_step_layout_cleanup_cuda.py` (the family sets,
the Ada's default backend).

LINEAR_DIV_SPLIT on top (leeloo, [ada-leeloo-split/](ada-leeloo-split/), both sides CUTLASS + LINEAR_DIV):
b/a SrTiO3 1.0126 / 1.0058, NbCl5 1.0026 / 1.0046: slower, stays off on the Ada. `:529` stays (357 µs on
SrTiO3); a better kernel for it would be a separate item (an epilogue that also writes the gradient blocks).

## Item 6: product 1 generated (`SC_GENERATED` / `SC_FACTORED`): stopped at the lab bound

Product 1 of OMAT-0 medium (128x0e output, one output segment, 13 weight segments, 94 monomials) through
`tools/sc0_lab.py --product 1` (new: `--product`, `--sources-only`; `sc0_bound.capture` takes the product), Ada
leeloo, SrTiO3 10×10×10 capture, L2 flushed. Records: [ada-leeloo-item6/](ada-leeloo-item6/) (`jobs/item6.sh`).

**Census** (CPU, NVRTC of the production env, t215 `census_nvrtc.py`, sm_80 / sm_89 / sm_90; `census.json`): no
spills or local memory anywhere. Forward (division fused) 400 / 400 / 408 SASS instructions, 103 FFMA, 40 / 40 /
32 registers; backward 640 / 640 / 656, 268 FFMA, 64 registers. The factored form is no smaller here (forward 336,
backward 648; its 4-wide backward 92 registers).

**Lab** (µs, kernel time): cuEq forward 177.8, generated forward 176.9 (fused division 178.2); the pair cuEq 434
(forward 139.1 + backward 294.6, inside the proxy) against generated 463 (176 + 286, with the division and its
density gradient). Errors against cuEq 1.9e-6 max abs, 2.5e-7 rms relative.

- Against the step before item 5, the proxy with the Triton division kernels (1,093 µs) against the pair (463)
  would bound about 630 µs (lab, L2 flushed).
- Against today's Ada (item 5): the forward division sits in the GEMM epilogue, so the contraction alone is the
  lab's cuEq-against-generated difference, about 10 µs (0.07 % of 14.7 ms). That is under the 2 % bar: stopped,
  no MD A/B, no gate. The only cost a fused division could still take is LINEAR_DIV's backward at `:529`
  (357 µs, 2.4 %); see LINEAR_DIV_SPLIT above for the existing kernel that targets it.

