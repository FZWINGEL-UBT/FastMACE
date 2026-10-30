# C1: fusing the symmetric contraction with a neighbouring GEMM (falsifier)

Idea C1 of [plans/capacity-ideas-20261030.md](../../plans/capacity-ideas-20261030.md), approved by the owner as a
falsifier only. The idea: produce the contracted product in registers and consume it in a GEMM, so the `[N, 8192]`
tensor is never written to memory.

## Verdict: stop

- **Speed.** The reachable gain is under 3 % of the step. The contraction kernels are 5.9 % of the step at 9,216
  atoms. Only the forward fusion is buildable, and its ceiling is 3.4 % before the contraction's own arithmetic is
  paid for in the GEMM.
- **Memory.** The `[N, 8192]` tensor is a peak term:
  - it is kept for the backward, one per interaction;
  - it is 65.5 kB per atom, alive at the force call's peak;
  - it is 37.7 % of the peak with `recompute_node_segments` on.

  Fusion removes it only if the backward rebuilds it, with the `linear_2` GEMM again. CAP-T08's combined region
  did exactly that, and the compiled peak did not move.
- **Which tensor.** The idea names two different tensors:
  - the contraction's *output* goes into the following GEMM (`products.*.linear`). It is 4–16 MB at 1,944 atoms,
    and `products.0.linear` is already folded away (OPT-T08 C3). Its ceiling is under 0.5 % (OPT3-T02).
  - the `[N, 8192]` tensor of the 1.58 GiB figure is the contraction's *input*, the output of the *preceding* GEMM
    (`linear_2`).

  This report measures the input, because only it is large.
- **Side finding, worth a task of its own.** The Triton `segment_gemm` products are DRAM-bound because of their
  tile order. They read their A operand 8.5 times. Running the column tiles of a row tile next to each other makes
  them 16–17 % faster, with bitwise equal results. These GEMMs are about a third of the force call, so that is
  about 5 % of the step at 9,216 and at 51,840 atoms ([below](#side-finding-segment_gemm-tile-order)).

## Setup

- **Machine:** Fabi-PC, WSL2, RTX 5060 Ti 16 GB (sm_120, driver 610.88), `mace_cueq_oeq` (torch 2.14.1+cu130,
  cuEquivariance 0.12.0).
- **Code:** FastMACE main `15d465cd`, which includes CAP-T08. The check without the tool ran on main `68b14c12`.
- **Model and plan:** MH-1 `mh1-r2scan`, hybrid backend, float32, the default plan; NbCl5.
- **GPU access:** local reservations, with the idle check before every job.
- **Tools:**
  - `tools/step_breakdown.py`: main pass at 1 fs and 1000 K, plus the ncu pass (one eager force call; DRAM bytes
    per kernel);
  - `tools/bench/saved_tensors.py`;
  - `results/c1-falsifier-20261030/gemm_order_probe.py` (new).
- **Records:** `results/c1-falsifier-20261030/`.

**Cells.**
- The kernel profile is at 9,216 atoms (NbCl5 4×4×8, 89.4 ms per step).
- The profile could not run at 21,600 or 51,840 atoms. `step_breakdown`'s eager force calls run on top of the
  captured graphs' memory and ran out of memory.
- The kernels on this path are per-atom and memory- or GEMM-bound, so their shares carry over to larger cells. The
  probe below confirms that for the GEMMs at 51,840 atoms.

## 1. Traffic and kernel time of the `[N, 8192]` tensor (9,216 atoms)

The tensor `Y` is the contraction's input, `[atoms, 16, 512]` float32, 302 MB per interaction at 9,216 atoms.

**Who touches it, per force call** (ncu, one eager call):

| Kernel | Interaction a | Interaction b | What it does with `Y` |
| --- | --- | --- | --- |
| `joined_products` (`linear_2`, 4 launches each) | 7.06 ms, writes 295 MB | 6.92 ms, writes 302 MB | writes `Y` |
| contraction forward (`segmented_polynomial_algo1_*`) | 0.95 ms, reads 319 MB | 0.78 ms, reads 309 MB | reads `Y` |
| contraction backward (`segmented_polynomial_bwd_algo4_2`) | 1.66 ms, R 327 / W 293 MB | 1.88 ms, R 398 / W 288 MB | reads `Y` and the output gradient; writes `dY` |
| `joined_products_input_grads` (4 launches each) | 6.89 ms, reads 2.38 GB | 6.99 ms, reads 2.41 GB | reads `dY`, 7.8 times over |

**Shares of the 89.4 ms step:**

- The contraction kernels: 1.73 ms forward and 3.54 ms backward, 5.27 ms in all (**5.9 %**). OPT3-T02 measured
  5.4 % at 1,944 atoms.
- The compulsory traffic of `Y` and `dY`: one write and one read each, per interaction and direction. That is
  2.42 GB per call, 5.4 ms at the card's 448 GB/s (6.0 %). The 8.5-fold re-reads by the GEMMs come on top; they
  are the tile-order finding below.

**What a fusion could reach:**

| Fusion | Removes | Ceiling | Buildable |
| --- | --- | ---: | --- |
| Forward: contraction as the `linear_2` GEMM's epilogue | the contraction forward (1.73 ms) and the write of `Y` (0.6 GB, about 1.35 ms at peak bandwidth) | **3.4 %** | yes, but see below |
| Backward: contraction backward as the prologue of the input-gradient GEMM | the contraction backward (3.54 ms) and the write and first read of `dY` | 4.8 % more | no |
| The idea as written: contraction into the following GEMM | the contraction's 4–16 MB output | < 0.5 % | yes (OPT3-T02) |

**Why the forward ceiling is not reachable:**

- The epilogue has to do the contraction's arithmetic. Interaction a's forward runs at 64 % of SM speed-of-light,
  so it would not hide under a GEMM that is itself at 60 % of SM throughput.
- The epilogue needs all 16 components of an atom for its channel tile: four GEMMs (l = 0..3) accumulated into one
  tile, 16× the accumulator of today's tile. That is OPT3-T02's "not buildable as an epilogue", now with the joined
  products in one launch.
- Realistically this is under 2 %.

**Why the backward fusion is not buildable:**

- The input-gradient GEMM's K dimension is the 512 channels of `dY`. A CTA computing `dY` in its prologue needs
  the contraction backward for all 512 channels of its rows.
- So it would either recompute the contraction backward once per column tile (8 times) or hold a full-K tile.
- MD needs no weight gradients (`linear_2`'s weights are constants), so that is the only consumer of `dY`. But it
  is the expensive one.

## 2. Memory: share of saved state and of the peak

`tools/bench/saved_tensors.py`, compiled default plan, `recompute_conv_weights` on, NbCl5 5×5×12 (21,600 atoms):

| `recompute_node_segments` | Force-call peak (above the call's start) | Saved per-atom state | The two `[N, 8192]` tensors | Their share of the peak |
| --- | ---: | ---: | ---: | ---: |
| off | 4.79 GiB | 3.51 GiB | 1.32 GiB, both alive at the peak | 27.5 % |
| on | 3.50 GiB | 1.96 GiB | 1.32 GiB, both alive at the peak | **37.7 %** |

They are the contraction inputs `Y` of both interactions. With the record on (separate regions) they are the
product blocks' checkpoint inputs. That is 65.5 kB per atom: 3.16 GiB at 51,840 atoms.

**A peak term, but not one fusion removes.**

- The contraction backward needs `Y`. A fused forward that does not write `Y` forces the backward to rebuild it with
  the `linear_2` GEMM: 14 ms per call for both interactions at 9,216 atoms (+16 %), or 7 ms (+8 %) for
  interaction 0's alone.
- Interaction 0's `Y` is the one held across interaction 1's backward.
- CAP-T08 measured this route ([capacity-T08-chunks.md](capacity-T08-chunks.md), one region per interaction).
  Rebuilding `Y` in the backward cut the saved state by 35 kB per atom, but the compiled peak stayed at 3.5 GiB:
  the recomputation's working set set it.
- So the memory side of C1 is the pending atom-tiling idea: a tiled backward that rebuilds `Y` in tiles and never
  holds it whole. Fusing the forward does not achieve it.

## 3. What is different from the earlier attempts

| Earlier | Finding then | Now |
| --- | --- | --- |
| OPT3-T02 (`docs/design/opt3-T02-symmetric-contraction.md`), 1,944 atoms | Contraction 5.4 % of the step, its kernels at 72–96 % of memory speed-of-light; fused input ≈ 2.7 % ceiling, "not buildable as an epilogue"; output side < 0.5 % | The same at 9,216 atoms: 5.9 %, forward-fusion ceiling 3.4 %. The joined products now run as one launch per l, but the epilogue still needs four of them in one tile. |
| OPT-T09 rank 6 (`sc-fused-input`) | ceiling 1 / 3 / 2 % | Unchanged. |
| OPT-T08 audit (`lin-t08-paper`) | `products.*.linear` into the contraction and `linear_2` into the contraction dropped on paper (more arithmetic, nonlinear on both sides) | Unchanged; `products.0.linear` is folded (C3). |
| Combo kernels, megakernel, cuBLASDx | no speed (1.02–1.06× slower or ≤ 2 %) | Not repeated. |
| New since then | — | **Memory:** with CAP-T08 on, `Y` is 37.7 % of the peak, the largest single saved term. **Speed:** the GEMMs around `Y` are DRAM-bound on re-reads (below). |

## Side finding: `segment_gemm` tile order

**The defect.** `fastmace.kernels.triton.segment_gemm` launches a `(row tiles, column tiles)` grid. Program id 0,
the row tile, runs fastest, so the eight 64-column tiles of a row tile run a whole pass apart.

- A (`[rows, 512]` float32, up to 132 MB per l at 9,216 atoms) is larger than L2, so it is read from DRAM once per
  column tile.
- ncu: the l = 1..3 products read 8.5 times A and run at 91–92 % of memory throughput and 60 % of SM throughput
  (10.6–11.1 TFLOPS). The l = 0 product's A fits in L2; it runs at 13.5 TFLOPS.

**The probe** (`gemm_order_probe.py`) runs the same kernel with a one-dimensional grid, column index fastest. It
keeps the same tiles and the same K order, in alternating blocks in one process:

| l | Rows (9,216 atoms) | Built | Column-fastest | Ratio | Bitwise equal |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 9,216 | 336 µs | 358 µs | 1.064 | yes |
| 1 | 27,648 | 1,329 µs | 1,068 µs | 0.804 | yes |
| 2 | 46,080 | 2,204 µs | 1,823 µs | 0.827 | yes |
| 3 | 64,512 | 3,081 µs | 2,579 µs | 0.837 | yes |
| sum | | 6,950 µs | 5,828 µs | **0.838** | |

At 51,840 atoms the sum goes from 39.7 to 33.0 ms (0.832), and every l is faster, l = 0 included (0.811).

**What it is worth.**

- Per force call at 9,216 atoms, `joined_products` and `joined_products_input_grads` are 28 ms of the 89 ms step.
  0.84 of that is about 4.5 ms less, **about 5 % of the step**, at both sizes.
- It is exact: the same tiles and accumulation order. It is general: it depends on no shape table, only on A being
  larger than L2. Small products, whose A fits in L2, can keep the current order; at 9,216 atoms l = 0 was 6 %
  slower the new way.
- It needs the full force-call A/B before anyone relies on it. I did not change the library.

## The 51,840-atom core dump

**What happened.** `saved_tensors.py` aborted with a core dump at NbCl5 6×6×20 (exit 134).

- The two `Simulation` steps ran.
- The tool's own instrumented force call (`saved_tensors.py:137`) runs outside the captured graphs while the graph
  engine still holds its memory. It ran out of memory ("Failed to create GPU mapping").
- That left the CUDA context sticky, and a tensor destructor at interpreter exit threw, so the process aborted.

**Library check, without the tool.** Main `68b14c12`, `capacity_scan.py point` at the same cell: ok, 200 steps,
594.5 ms per step, 9.34 GiB reserved at steady state. **The cause is the tool, not the library.**

**Tool limits.** `step_breakdown.py`'s eager profiler calls fail the same way at 21,600 and 51,840 atoms. Both
tools need their extra force call before the engine captures its graphs, or after it releases them, to profile
cells near the limit.
