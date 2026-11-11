# What to try next for inference speed and capacity (11 November 2026, evening)

Source: an owner request (11 November 2026) to review three outside ideas and compile everything open into one
queue. State read: main `b951e804`; `fastmace/review-retries-20261111` (`fd9a74ad`, pushed, not merged);
`fastmace/tiled-speed` (`0c2cccec`); the outside note `D:\Margraf\inference-ideas-20261111\ideas.md` with its two CPU
probes; this session's files (`plans/ideas-20261111*.md`, `plans/notes/rejected-ledger-20261111/`,
`results/binned-screen-5060ti-20261111/`). **Sizes without a measurement behind them are my estimates.**

Profile used for every share below (`results/step-limiters-20261109/`, OMAT-0 medium, SrTiO3 10×10×10): kernel time
of one force call 3,258 µs on the H200 and 6,512 µs on the A100. The lab bar (twice the 1 % bound) is 68 µs on the
H200 and 130 µs on the A100.

## The short version

1. **Merge two results:** `BINNED_SCREEN` on the RTX 5060 Ti (measured today) and `tf32x3.HYBRID` on the A100
   (measured today by the review-retries lane).
2. **Three packed jobs decide three more defaults:** `HYBRID` on the H200, `BINNED_SCREEN` on the Ada, larger
   receiver caps for the A100's uncaptured tiled calls.
3. **The largest unbuilt target is the first interaction** (13 % of the A100 call, 10 % of the H200's): ideas 18
   and 19.
4. **The three outside ideas:** two are sound and worth about 1 % each, to be screened by an offline compile; the
   texture idea cannot beat a plain float table and needs only one cheap test before it is dropped.
5. **Closed today, not to be retried:** half-precision far edges, half-systems on two streams, the formulation
   search, the device row pool.

## A. Ready to merge

| Item | Evidence | What the merge does |
| --- | --- | --- |
| `BINNED_SCREEN` on the RTX 5060 Ti | MD step 0.9812 / 0.9769 at 20,480 atoms, 0.9614 at 40,000; same list bit for bit; rebuild 2,131 → 73 ms at 214,375 atoms (`results/binned-screen-5060ti-20261111/README.md`) | Add `"NVIDIA GeForce RTX 5060 Ti"` to `pair_search.BINNED_SCREEN_GPUS`; decision 66 (work removal) |
| `tf32x3.HYBRID` on the A100 family | Lab 0.873 of 3×TF32, error at or below 3×TF32's on every shape; MD step 0.9896 / 0.9877 (SrTiO3), 0.9832 / 0.9846 (NbCl5); gate passes (`results/hybrid-gemm-20261111/README.md` on the branch) | Merge `fastmace/review-retries-20261111` after the leto suite |

## B. One packed job each

| # | Item | GPU and resource | Expected | Stop rule |
| --- | --- | --- | --- | --- |
| B1 | `HYBRID` on the H200 (the sm_90 cubin exists, nothing measured) | Modal H200: A/B both orders and the gate in one pack | About 2 % if the A100's lab ratio carries over (GEMMs are 560 µs, 17 %) | Not faster in both orders, or the gate fails |
| B2 | `BINNED_SCREEN` on the RTX 4000 Ada | leto or murphy after the idle check: A/B at 16×16×16 and 20×20×20 | As on the 5060 Ti; 10–18 % at the Ada's limits (estimate) | Slower in either order |
| B3 | Receiver caps 32,768 and 65,536 for the A100's host-state and wavefront tiled calls | Alex: `tiled_ab.py --b-receivers` | Unknown; these calls are not captured in CUDA graphs, so fewer tiles means less host time (7.04 µs/atom today) | Within noise |
| B4 | Shuffled against ASE-ordered atoms at 24×24×24 | Local 5060 Ti, one run | No gain for benchmark crystals; shows whether arbitrary inputs need a spatial sort at setup | Shuffled within 3 % |

Pending in other lanes, nothing to launch: the A100 397.5M-atom rerun with the heap trim (Alex 4981047), and the
A100 A/B of the coordinate fix (`f8ea447e`).

## C. Kernel work, falsifier first

Ranked by estimated size. Each starts on free hardware.

| # | Item | Helps | Estimated size | First test | Stop rule |
| --- | --- | --- | --- | --- | --- |
| C1 | First interaction's forward by slot, sums in registers (idea 18) | A100 most, H200 | A100 up to 4.6 %, H200 up to 1.9 % | Today's kernel without its slot loads and stores (wrong numbers, timing only): 5060 Ti, then Alex A100 | A100 falsifier saves under 130 µs |
| C2 | First radial table packed like `PACKED8` (idea 19) | both tabled kernels | about 2 % | One load per column instead of six (falsifier) | Under the lab bar on both GPUs |
| C3 | One spline decode for two edges on the same piece (outside idea 2) | H200 | 1.3–1.6 % | Offline compile of the paired body for sm_90 | Over 80 registers, any spill, or under 27 instructions saved per edge and channel |
| C4 | Joint rescaling of inputs, harmonics and outputs (outside idea 1) | A100 and H200 | 1.1–1.6 % | Offline census of w′ with the transform, the staging-only form first | Under the lab bar after the boundary transforms |
| C5 | w′ accumulators started from their first product (idea 20) | H200 | at most 1 % | The same census pass as C4 | No drop in instructions on sm_90 |
| C6 | All three products in one FP16 pass (1.5 tensor instructions per step against `HYBRID`'s 2) | A100, H200 | at most a quarter of the remaining GEMM time | `hybrid_lab.py` with a row-scaled variant, after B1 | Slower than `HYBRID`, or error above 3×TF32's |

## D. The capacity path (tiled and wavefront)

From `plans/ideas-20261111-tiled.md` and `-tiled-round2.md`. Built since: idea 1 (`TILE_HEAP_TRIM`), CUDA graphs for
the device-state tiles (default on the A100), idea 11 (coordinates, by size, records in). Rejected: idea 3.

| # | Item | Gain | First test |
| --- | --- | --- | --- |
| D1 | CUDA-graph tile bodies for host state, wavefront and aliased device state (idea 9) | Unknown; the device-state call gained 43 % on the A100 | Nsight Systems idle share of one A100 wavefront call |
| D2 | 512 state columns instead of 640 (idea 10) | State −20 % (Ada pool 48.8 → 39 GB) | CPU test against today's sweeps |
| D3 | Kept lists as bitmasks over the search stencil (idea 12) | 312 → about 55 B/atom on the host | CPU: bytes and rebuild time at 1M atoms |
| D4 | Harmonics and edge force inside the edge kernels (idea 13) | Up to 10.7 of 68.2 kB/atom; two elementwise kernels go | Which of the three edge tensors are live at the peak |
| D5 | Ideas 14–17 and 4, 6, 7, 8 of the first tiled list | See the two files | – |

## Review of the three outside ideas

Note: `D:\Margraf\inference-ideas-20261111\ideas.md` (outside the checkout). I checked each against the kernel
sources and the records; I did not rerun the probes.

### 1. Joint rescaling of the angular coordinates: sound, about 1 %

- **The baseline is right.** I counted the captured A100 w′ source: 86 terms, 50 with a unit coefficient, 36 with
  a multiply, 25 distinct (output, coefficient) products. That is the probe's 25. Each path is already normalised
  to its most frequent magnitude.
- **Where the scales can go for free.** Harmonics at edge staging and dout at `STAGE_DOUT`'s copy cost nothing per
  channel. Scaling x needs a scaled copy of `[N, 512]`, the same coordinates in the spline forward, and one unscale of
  dx per atom. The probe gives 18 with the dout scale alone and 9 with all three; it has no count for dout and
  harmonics without x, which is the form to compile first.
- **Size.** 16 fewer multiplies among 340 FMA-pipe instructions per channel step is at most 1.6 % of the A100 call;
  among 497 instructions it is 1.1 % of the H200's. The forward adds something only where `GEOM` is off.
- **Risk.** The scales are not powers of two, so this is a rounding change and needs the gate. It sits under the
  lab bar unless the forward contributes.

### 2. One decode for two edges on the same spline piece: sound, H200 only

- **The numbers hold.** `PACKED8` decodes at about 14 instructions per weight
  (`results/packed-table-lab-20261109/README.md`), ten weights per channel, so about 140 of the forward's roughly 340
  instructions per edge and channel. The probe removes 19–25 % of decodes at 0.05–0.08 Å displacement and 9 % at
  0.30 Å.
- **Size.** 27–35 instructions per edge and channel, 8–10 % of a forward that is 82 % issue-bound on the H200:
  41–53 µs. On the A100 the forward is FMA-bound and decodes are shifts and masks, so little.
- **Risks.** The forward already sits at its 80-register cap, and the paired body needs a second edge's inputs. The
  gain depends on the structure: little for liquids and molecular crystals. The proposed compile screen is the
  right first test.

### 3. Texture interpolation with a correction: dominated, one test then drop

- **The correction is valid** for a quadratic per piece, if each piece has its own two texels and the kernel knows
  the hardware's rounded coordinate exactly.
- **But it saves no arithmetic.** The texture unit's interpolation is one multiply-add per weight, and the
  correction puts two back: the same two as today's Horner step. What is left is a fetch in place of a load and a
  decode.
- **That is an unpacked float table.** Value and derivatives in float32 for two texels per piece are 3–4 times
  `PACKED8`'s bytes, and `PACKED8` won on three GPUs because of bytes.
- **The one test:** a fetch-rate microbenchmark (float32 RGBA, linear filter). The forward evaluates 4.8e8 weights
  per call, so the texture units would need well over 1e12 fetches per second to matter. If they do not, close it.

## Closed today

| Idea | Why | Record |
| --- | --- | --- |
| Half-precision far edges (`HALF_FAR`) | Float16 weight gradients in dr fail NaCl at 5.5 Å (F 7.1×, S 13×); no safe radius | `results/half-far-wprime-20261111/README.md` on the review branch |
| Two half-systems on two streams | Out of phase 1.016 of the full graph; the edge kernels fill the register file | `results/half-systems-offset-20261111/README.md` |
| Formulation search | Today's form is rank 2 of 27; best mix about 1 % | `results/a100-colab-ideas-20261111/README.md` |
| Wavefront row pool on the device | Tie with the host pool | `results/tiled-speed-20261111/README.md` |
| 43 older rejections | None needs reopening | `plans/notes/rejected-ledger-20261111/review-old-and-resources.md` |

Limits that any new w′ variant on the A100 has to respect: about 85 registers and 27 KB of shared memory per
block, or it drops from 6 to 5 blocks per SM.

## Housekeeping

- **Ledger statuses that no longer match:** `omat0-slope-vector` (the default `SPLINE_VECTOR`), `omat0-stream-w`,
  `omat0-stream-store-w`, `omat0-eos-chunks` (rejected in the index, pending in their entries), and the 27 items in
  `plans/notes/rejected-ledger-20261111/agent-notes.md`.
- **Uncommitted in the main working tree:** `plans/ideas-20261111.md`, `plans/ideas-20261111-round3.md`, this file,
  `plans/notes/rejected-ledger-20261111/`, `plans/notes/conductor-prompt-20261111-next-steps.md`,
  `results/binned-screen-5060ti-20261111/`.
- **Outside the checkout:** `D:\Margraf\inference-ideas-20261111\` (the outside note and its two probes).
