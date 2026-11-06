# H200 ideas 37 and 38: kernel-lab measurements (6 November 2026)

Two lab measurements of the w′ backward (`oeq_projected_backward_wprime`) from the H200 ideas plan, revision 15
(ideas 37 and 38, after the Codex review `plans/codex-ideas-20261106.md`). Lab only; nothing goes into production.

- **Branch / commit:** `fastmace/wp-ideas-37-38`, lab source at `dbb91337` (`tools/lab_ideas3738.py`, hooked into
  `tools/omat0_kernel_lab.py`; the lab's launch marker takes an optional `cohort=`, threads per edge).
- **Run:** Modal H200 (driver 610.57.04, CUDA UMD 13.3), profile `fourth`, app `ap-OuKysQ24cWwcgUChKRsK6W`,
  `tools/modal/hopper_sweep.sh` with `ALIAS='NVIDIA H200' TESTS=0 GATE=0 LOUP=0 AB= FORWARD_VARIANTS=` and
  `WPRIME_VARIANTS='base maxreg72 maxreg80 FF-dout-row0 harmonic harmonic-rot'`; wall 337 s, metered $0.20.
  Raw output: [ideas-37-38/h200-r1](ideas-37-38/h200-r1/) (`lab-wprime/lab.json`, `lab-wprime.log`, `run.log`).
- **Workload:** OMAT-0 medium, SrTiO3 10×10×10 (5,000 atoms), captured after 300 MD steps at 300 K;
  373,760 padded edges, 128 threads per block, 15 round-robin rounds, 256 MB L2 flush before each launch, medians.
  The plan's H200 render (FACTORED, TABLE_W, HORNER, SCALAR_PATHS; BUTTERFLY off) has SHA-256 `5565cb33…`, the same
  source as round 9 and Codex's `base.cu`.
- **CPU check:** NVRTC 13.3 (`cuda.bindings` in WSL `mace_cueq_oeq`), sm_90, `cuobjdump` from the same venv:
  [ideas-37-38/cpu-sm90](ideas-37-38/cpu-sm90/) (`sass_check.py`, `sass-summary.json`).

## Results (H200)

| Variant | Kind | Registers | Stack / spill stores | Median µs (min–max) | / base | / maxreg72 |
| --- | --- | --- | --- | --- | --- | --- |
| base | plan kernel | 96 | 0 / 0 | 1,634.9 (1,633.9–1,675.8) | 1.000 | 0.928 |
| maxreg72 | control, real addresses | 72 | 64 B / 104 B | 1,762.2 (1,757.5–1,773.2) | 1.078 | 1.000 |
| maxreg80 | second control, no spills | 80 | 0 / 0 | 2,055.0 (2,049.6–2,060.8) | 1.257 | 1.166 |
| FF-dout-row0 | falsifier (idea 38) | 72 | 0 / 0 | 1,445.9 (1,443.8–1,451.9) | **0.884** | **0.820** |
| harmonic | idea 37 | 64 | 0 / 0, 9,232 B shared | 3,051.6 (3,049.3–3,060.0) | **1.867** | |
| harmonic-rot | idea 37, rotated warps | 64 | 0 / 0, 9,232 B shared | 3,051.4 (3,048.9–3,060.5) | **1.866** | |

maxreg72 and maxreg80 repeat the conductor's idea-36 caps (1.078 and 1.255 then; 1.078 and 1.257 here).

## Idea 38, step 1: dout locality

**Variant.** `FF-dout-row0` replaces `int64_t(receiver)` by `int64_t(0)` in every `dout<n>[int64_t(receiver) * D + …]`
load: 40 expressions in this render. All edges then read receiver 0's 20 KB of dout, which stays in L1/L2. The
values are wrong (timing only). The source is byte-identical to Codex's `dout_fixed_bound.cu` (SHA-256 `1ad76d8d…`).

**SASS evidence (sm_90).**

- The old `FF-dout-one-row` (which set the unused `output_base = 0`) builds to SASS identical to base's. This
  confirms Codex's correction A.
- `FF-dout-row0` builds to different SASS from base and from maxreg72. It has 72 registers, no stack, 1,422
  instructions (base: 1,432) and the same 97 LDG and 4 REDG.
- The control `maxreg72` (the real addresses at `--maxrregcount=72`) has 72 registers but spills: 64 B stack, 26 STL
  and 19 LDL (ptxas on the H200: 104 bytes spill stores). Their 5,284 SASS lines differ.

**Reading.**

- The falsifier is 11.6 % faster than base (−189 µs) and 18.0 % faster than the control.
- The control is not a perfect match, because its spills penalise it. Two facts still rule out register count and
  occupancy as the cause:
  - The base-relative gain (11.6 %) is already above the line, and base has more registers than the falsifier.
  - Fewer registers alone make this kernel slower: maxreg80, with no spills, takes 1.257.
- So the gain is in the dout addresses: locality, plus slightly simpler address arithmetic (10 fewer static
  instructions).
- This is an upper bound with perfect reuse. It is not what a staging scheme would reach, and it does not include
  any staging, reduction or descriptor costs.

**Verdict: go.** The gain against the control is 18.0 %, well above the 5 % stop line (11.6 % against base).
Earlier statements that "dout locality is worth nothing" are now refuted, not just unsupported. Next is the plan's
CPU check: the share of contributing edges that lie in receiver tiles of at least eight edges, with a stop below
one half. Then comes the 32-channel striped design. Its own ceiling is ≈ 190 µs, before staging costs.

## Idea 38, continued: receiver-staged dout (STAGE_DOUT)

### CPU check: receiver-tile lengths

[tile_census.py](ideas-37-38/tile_census.py) emulates FastMACE's skin list on SrTiO3 10×10×10 (cutoff 6 Å, skin
0.5 Å; output in [tile_census.txt](ideas-37-38/tile_census.txt)):

- pairs within cutoff + skin are sorted by receiver (`sort_receivers` is on for the hybrid graph engine), then
  compacted stably so the pairs within the cutoff come first;
- the list is cut to the captured 373,760 edges.

With 0.07 Å Gaussian disorder (≈ 300 K):

- 368,902 contributing edges;
- runs of equal receiver have 62–86 contributing edges (mean 73.8);
- with 16-edge tiles, 98.9 % of the contributing edges lie in tiles of at least 8 edges (32-edge tiles: 99.9 %);
- 0.068 dout rows are read per edge (16-edge tiles) or 0.038 (32-edge tiles), against one per edge today.

The ideal lattice gives 98.4 %. The plan's stop line (below one half) is far away.

### The form the tiles support

The runs are about 74 edges long, so the receiver's whole dout row can be staged instead of 32-channel stripes.
That is 5,120 floats, 20 KB, per block of up to 16–32 edges.

- The whole row keeps the per-edge body unchanged: one warp per edge, four channel passes.
- So no dsh/dr reduction across stripes is needed, and the outputs stay bitwise.
- The cost is 20 KB of shared memory per 128-thread block. At 72 registers that allows seven blocks per SM, which
  uses 147 KB of the H200's 228 KB.

Two lab forms (`tools/lab_ideas3738.py`):

- **`stage-t<T>` (descriptor tiles):**
  - one block per tile of at most T consecutive edges of one receiver; the tile starts are built from the receivers
    on the host side of the lab and are not timed;
  - the block copies the receiver's row with float4 loads, then one barrier;
  - the four warps take the tile's edges in turn.
- **`stage-c<C>` (chunks, what production uses):**
  - block b owns edges [bC, (b + 1)C) and walks the runs of equal receiver inside them;
  - one `__ballot_sync` over the next 32 receivers finds each run's end;
  - per run: copy the row, barrier, the warps run the run's edges, barrier.
  - It needs no descriptors and no data-dependent grid, so it can be captured in a CUDA graph. It is also correct
    for any edge order; unsorted edges just make the runs short.

sm_90 builds of all forms (CPU, [sass-summary.json](ideas-37-38/cpu-sm90/sass-summary.json)):

- 72 registers, no stack, 21,504 B static shared memory;
- the 40 dout LDGs become 40 LDS (64 LDG instead of 97).

### H200 lab

Runs r2 at `4123f9f3` (app `ap-p5tocgQG2krVpM5OQUo2nP`) and r3 at `0cca39cf` (app `ap-0THiqFWkpbA9FAiicKlJfb`). Same workload and lab settings as above.

| Variant | r2 µs | r2 / base | r3 µs | r3 / base | dsh | dr | dx |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | 1,621.2 | 1.000 | 1,643.7 | 1.000 | | | |
| stage-t8 | 1,496.9 | 0.923 | | | bitwise | bitwise | 2.5e-7 |
| stage-t16 | 1,432.0 | 0.883 | | | bitwise | bitwise | 3.1e-7 |
| stage-t32 | 1,419.4 | 0.875 | 1,441.0 | 0.877 | bitwise | bitwise | 3.1–4.4e-7 |
| stage-c16 | | | 1,458.6 | 0.887 | bitwise | bitwise | 3.1e-7 |
| **stage-c32** | | | **1,440.7** | **0.877** | bitwise | bitwise | 3.7e-7 |
| FF-dout-row0 (bound) | 1,450.2 | 0.895 | 1,462.5 | 0.890 | – | – | – |

- The dx differences are the atomics' order, as for the bitwise-exact maxreg variants.
- The staged kernels beat the "perfect locality" bound by about 1.5 %. The bound still reads its dout through
  global loads, while staging turns 40 LDG per channel pass into LDS. The chunk form is as fast as the descriptor
  tiles at 32 edges.
- **Lab verdict:** 0.877 ≤ 0.95, so it is ported.

### Production: `wprime_backward.STAGE_DOUT`

Branch `fastmace/wp-ideas-37-38`, `0107847c` (kernel) and `0cca39cf` (job script).

- **The switch:**
  - `STAGE_DOUT = None`, `STAGE_DOUT_GPUS = GPUModels(())` (empty: off everywhere);
  - `stage_dout_default(device)`;
  - `STAGE_DOUT_CHUNK = 32`.
- **Where:** a `stage_dout` render option in the Jinja template, the chunk form. It covers the block-layout message
  (`BLOCK_MESSAGE`, the H200 default) and the dense row.
- **Fallback:** `stage_dout_supported` requires float32, 32-lane cohorts, a chunk of 1–32 edges and a dout row
  within 48 KB; otherwise the kernel is unstaged. The copy uses float4 when the row is 16-byte aligned and its
  length is a multiple of 4, else scalar loads. The launch is `ceil(edges / 32)` blocks.
- **Layering:** a `test_layering` allow-list entry.
- **GPU test** (`tests/gpu/test_wprime_backward_cuda.py`):
  - cases `stage-dout-{ir_mul,mul_ir}-{32,5,32-table-w}`, staged against unstaged through `WPrime.convolution`,
    with and without graph capture;
  - case `stage-dout-blocks`: the block-layout kernels called directly;
  - dsh and dL/dr must be bitwise, dx within the atomic tolerance.
  - **H200:** the whole file passes (r3, `gputest.log`, 28 s).
  - **RTX 5060 Ti:** the file's first, pre-existing check fails, on origin/main too (REDUX's fixed-point dsh against
    the 1e-6 tolerance). With REDUX off, every STAGE_DOUT case is bitwise.

**MD-step A/B** (r3; `tools/ab_rewrites.py`, SrTiO3 10×10×10, OMAT-0 medium, 1 fs, skin 0.5, 6 blocks of 256
steps after 200 warm-up; b = `STAGE_DOUT=True`):

| Order | a (ms) | b (ms) | b/a |
| --- | --- | --- | --- |
| ab | 5.5034 | 5.3313 | **0.9687** |
| ba | 5.4899 | 5.3239 | **0.9698** |

That is about −170 µs per MD step, consistent with the kernel's −200 µs. No force/stress gate was run: dsh and dL/dr
are bitwise and dx differs only in the atomics' order, which already varies between runs of the base.

**Verdict:** STAGE_DOUT is ready for a decision-66 H200 default (add "NVIDIA H200" to `STAGE_DOUT_GPUS`). It stays
off here as asked. The A100-SXM4 and other GPUs are unmeasured.

## Idea 37: the w′ split by harmonic degree

**Integration.** Feasible, about an hour of work. The lab transform rebuilds Codex's prototype from the plan's
render with Codex's own split and wrapper code (`compile_research.py` degree bodies, `compile_combined.py` wrapper).
On this render, `harmonic-rot` reproduces `combined/harmonic_cta_rotated.cu` byte for byte (SHA-256 `0232182c…`).
`harmonic` is the unrotated `harmonic_cta.cu` plus the lab marker.

- **Path degrees:** 0, 1, 1, 0, 2, 2, 1, 3, 3, 2.
- **Launch:** one 128-thread block per edge (grid = 373,760), so the lab marker carries `cohort=128`. Nothing else
  changed.
- **sm_90 build:** 64 registers, no stack, 9,232 B static shared memory, one `BAR.SYNC`, 1 REDG site in the final
  loop, 2,133 / 2,138 static instructions.

**Correctness (lab comparison with base, same arguments):**

| Output | Result |
| --- | --- |
| dsh | bitwise |
| dr | max\|d\|/max\|a\| = 5.4e-7 |
| dx | 1.87e-7, the same as the bitwise-exact maxreg variants show from atomic ordering alone |

The prototype is correct to rounding class. The rotation changes nothing (1.866 against 1.867).

**Verdict: stop.** 1.867 of base (3,052 against 1,635 µs, +1.42 ms) is far above the 0.95 line. The occupancy gain
(64 registers, eight blocks of four warps per SM) does not pay for the costs it brings:

- every edge's setup is done four times (132 against 97 static LDG);
- x and dout rows are read by several warps;
- the block waits at the barrier for its slowest degree;
- the dx partials go through shared memory.

Combined with idea 36's caps (more warps per SM are slower), this closes the "more warps for the w′" direction for
this kernel.

## Budget

Three Modal H200 jobs (profile `fourth`):

| Run | Wall time | Metered |
| --- | --- | --- |
| r1 | 337 s | $0.20 |
| r2 | 334 s | $0.47 |
| r3 | 477 s | $0.92 |

The profile's month stood at $14.55 afterwards.
