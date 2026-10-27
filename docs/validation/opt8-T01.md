# OPT8-T01: radial-table output straight into the streaming convolution (27 October 2026)

Plan: `plans/opt-8.md`. Probe: `tools/bench/opt8_fused_conv.py` (timing, ceilings, exactness) and
`tools/bench/opt8_fused_kernels.py` (the generated fused kernels). Operands: `tools/bench/opt8_record_operands.py`.
Runs, sweeps, Nsight Compute output and run scripts: `results/opt8-T01-fused-conv-20261027/`.

**Verdict: the stop rule is not met.** The fused forward and backward are exact. Together they are
**17.9 % faster** than today's chain on NbCl5 (3.34 against 4.07 ms) and **17.5 %** faster on NaNbO3 (5.18
against 6.27 ms). The rule asked for 25 % on both cells. Per the plan, no record is built (OPT8-T02 and T03 are
not started).

## Setup

- **Machine:** local RTX 5060 Ti (sm_120, 36 SMs, 32 MiB L2), driver 610.88, WSL2.
  - Environment: `~/environments/mace_cueq_oeq` (torch 2.14.1+cu130).
  - Runs used `tools/ci/local_gpu.py run --quiet` reservations, with the desktop running.
- **FastMACE:** branch `fastmace/opt8-T01` from main `770e3898`. No library change.
- **Operands:** `mh1-r2scan` float32 with the default plan, so `radial_tables` and `chunked_conv_backward` are on.
  - Cells are 3×3×3, rattled by ±0.05 Å (seed 20261027).
  - One eager force call per cell, recorded at `folded_streaming_convolution`: `a [E, 65]` (the radial
    tables' output), the folded `W [1280, 65]`, `x`, `sh`, senders, receivers, CSR rows, the message and `dout`,
    plus the plan's `chunk_edges` (1,536) and `tile_rule` (`waves`).
  - Edges: NaCl 6,912; NbCl5 52,416 (27 per atom); NaNbO3 84,072 (78 per atom).
- **Chain (the baseline):** exactly today's path.
  - Forward: `torch.mm(a, W.T)` (cuBLAS, IEEE FP32) into `w`, then the vendored streaming forward.
  - Backward: `streaming_conv_folded_backward`: the chunked edge-owned backward, with the Triton input gradient
    on the side stream.
- **Timing:**
  - every variant is one CUDA graph of 20 calls;
  - every call first overwrites twice the L2; that flush is timed alone and subtracted;
  - variants alternate block by block in one process: 15 blocks × 3 rounds (fresh graphs each round);
  - each round reports its medians.

## Sobek's reassociation: rejected on arithmetic

The reassociation of arXiv 2607.18074 sums per receiver before the channel products, so no per-edge weight
tensor exists. Here the weight is a 65-wide linear map of `a`, so the per-receiver sum
`Σ_e a[e,k] TP(x_s, sh_e)` holds 65 × 5,120 values. That costs 333k FMA per edge. Building the edge's
`w` row and using it costs 83k FMA plus 86 angular terms per channel, about a quarter of that. Only the tile
approach was built.

## The prototype (plain FP32 SIMT, generated from the streaming schedule)

**Slices.** A block owns one slice: 32 channels × one half of the 10 paths, so 8 slices for MH-1.
- It keeps that slice of `W` in shared memory: 5 × 65 × 33 floats, 43 KB. The row stride of 33 makes both access
  patterns (lanes = channels, lanes = k) free of bank conflicts.
- The two halves have equal path counts, and their angular terms are balanced greedily (43 and 43). This is a
  general rule, not a table.

**Work units.** One warp handles a chunk of 2 receivers at a time and takes the next chunk from a per-slice
atomic counter.
- Edge batches of 8 run across receiver boundaries, so there is no padding per receiver.
- Lane = channel.

**Forward.**
- The batch's `a` rows go to warp-private shared memory, transposed: four edges are one float4 broadcast.
- `w[5][8]` is built by a k-loop that reads each `W` value once for all 8 edges.
- Then, per edge, the angular terms accumulate into the receiver's outputs in registers. They are flushed with
  plain stores when the receiver changes.

**Backward.**
- `w` is rebuilt the same way. Per edge, the kernel computes `dw`, the sender-gradient atomics (as in the
  streaming backward) and the edge gradient. The edge gradient is transpose-reduced over the lanes and added
  atomically per slice.
- `da = dw W`: `dw` goes through warp-private shared memory and is read back as float4 broadcasts, with
  lanes = k (two passes of 32). Column 64 is a lane-local partial, transpose-reduced at the end. The result is
  added atomically per slice.

**Configuration.** The fastest, from 33 configuration runs in 7 sweeps (`sweep*.jsonl`, `sweep_configs.txt`):
- `b=8,nw=12` forward;
- `bb=8,bw=12` backward;
- `rc=2`, with 168 registers and no spills.

It is fixed for all three cells.

**Exactness against the chain** (relative to the reference's largest absolute value):

| | message | `dx` | `dsh` | `da` |
| --- | ---: | ---: | ---: | ---: |
| all three cells | 0 (bitwise) | ≤ 8.0e-7 | ≤ 2.6e-7 | ≤ 3.9e-6 |

The message stays bitwise equal even when the output starts as NaN. `dx`, `dsh` and `da` differ only in the
order of float32 sums: lane reductions and atomics. `dx` was already nondeterministic.

## Results (`final.jsonl`, ms per force call of the subgraph, medians of the three rounds)

| Cell | chain fwd / bwd / total | fused fwd / bwd / total | fused / chain | gain |
| --- | --- | --- | ---: | ---: |
| NaCl | 0.205 / 0.328 / 0.533 | 0.132 / 0.302 / 0.434 | 0.80–0.82 | 0.10 ms |
| NbCl5 | 2.020 / 2.050 / 4.070 | 1.047 / 2.289 / 3.336 | **0.818–0.822** | 0.73 ms |
| NaNbO3 | 3.117 / 3.155 / 6.277 | 1.608 / 3.570 / 5.180 | **0.825–0.827** | 1.10 ms |

All rounds lie within ±0.4 % of these. The fused **forward** halves the chain's forward (0.52 NbCl5, 0.52 NaNbO3).
The fused **backward** is slower than the chain's: NbCl5 1.12×, NaNbO3 1.13×.
- The chain's backward already keeps `dw` in L2 (`chunked_conv_backward`), so fusing it saves only the `w` read.
- That saving does not pay for rebuilding `w`.

**Stop rule (≥ 25 % on NaNbO3 and NbCl5): not met** (17.5 % and 17.9 %).

## Why it stops there (Nsight Compute, `mix1.txt`, `mix2.txt`, `stall2.txt`, NaNbO3)

**Instruction counts.**
- The three products (the forward `w`, the backward's rebuilt `w`, and `da`) need 657 M warp-level FFMA in
  total; FP32 SIMT cannot remove them.
- The forward executes 420 M instructions and the backward 887 M (an intermediate configuration; the final one is
  similar).
  - Of these, 545 M (backward) and 257 M (forward) are FFMA.
  - The remainder is shared-memory loads, the angular terms and the lane reductions.

**Issue rate.** Both kernels are issue bound.
- Warps are about 27 % "not selected" (another warp issued) and 10–13 % in dispatch stalls.
- The kernels run at an IPC of 2.6–2.9 of 4 per SM.
- 12 warps per SM are the most that 168 registers allow; 14 or more spill and run slower.

**Ceiling (estimate).** Even at a perfect issue rate, the extra product of the backward keeps the fused total near 0.6–0.7 of
the chain. The measured 0.82 is what this SIMT design reaches. The FP32 tensor-core alternative (FP16×3) does not
apply on this GPU: its FP16-with-FP32-accumulate rate is about 2× FP32, below the split count of 3. On data-centre
GPUs (ratio ≫ 3) the balance differs; that was not measured.

**The ceiling runs of part (a) are not bounds.** They are recorded in every run (`stub_*`, `gemm_l2`,
`igrad_l2`, `bound_*` in `final.jsonl`).
- cuBLAS and the Triton input gradient, run on 1,536-edge chunks into an L2-resident buffer, are slower than the
  prototype's own products: NaNbO3 `gemm_l2` 2.10 ms, against a fused forward of 1.61 ms that includes its
  product.
- So `max(stub, gemm_l2)` (0.94–0.98 of the chain) overstates the floor.
- The stubs themselves show the convolution without `w`/`dw` traffic: NaNbO3 forward 0.33 ms, backward 0.86 ms.

## Step-level view (inference, no full-step A/B)

At the opt-6 step times (NaNbO3 15.3 ms, NbCl5 19.2 ms), the measured subgraph gains would be about 7 % and 4 %
of a step.
- This is not the 25 % subgraph gain the plan required.
- The generated kernels would still need the record work: the CPU and GPU tests, the gate,
  NVE drift, both A/B orders, and the MH-1 family (POLAR-1, OMOL-0).

**Owner decision:** whether a 4–7 % step gain is worth building despite the missed stop rule. The plan says stop,
so this probe stops here.

## Not done

- OPT8-T02 and T03; no record, no FastMACE library code.
- sm_89 (workstations) and data-centre GPUs were not measured.
- Chunks by edges instead of receivers: rc=3 against rc=2 on NaNbO3 showed a tail effect (0.90 against 0.83). A
  general edge-count chunk rule could remove it, but would not close the gap to 0.75.

## T01b: fused forward that also stores `w`, today's backward unchanged (owner, 27 October 2026)

**Variant.** T01's fused forward with one addition: it writes the `w` rows it builds to `w [E, 1280]`. The chunked
backward (`streaming_conv_folded_backward`) then runs unchanged on that `w`.
- The variant drops the cuBLAS GEMM and the streaming forward's `w` read, but keeps the `w` write.
- Kernel configuration: `b=8,nw=12,rc=2`, as in T01, plus `store_w=1` (the stores are coalesced per path).
- Method as in T01: same operands, the same process, CUDA graphs of 20 calls with the L2 flush subtracted, 15
  alternating blocks × 3 rounds.
- `chain_fb` and `fusedw_fb` are forward and backward together in one graph.
- Run: `tools/bench/opt8_fused_conv.py --variants fusedw`; record `t01b.jsonl`.

**Exactness**, on every cell, against the chain:
- **bitwise equal:** `w` (cuBLAS and the k-loop sum in the same order) and the message, both checked from
  NaN-filled outputs; also `dsh` and `da` (the same backward on the same `w`);
- **`dx`:** 1.4–2.0e-7 relative, the order of its existing atomics.

ms per force call, medians per round:

| Cell | chain fwd | fused fwd + `w` store | fwd ratio | chain fwd + bwd | fused fwd + `w` + today's bwd | **fwd + bwd ratio** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NaCl | 0.201–0.206 | 0.159–0.168 | 0.79–0.83 | 0.510–0.518 | 0.488–0.493 | 0.952–0.961 |
| NbCl5 | 2.016–2.019 | 1.329–1.342 | 0.66 | 4.054–4.086 | 3.395–3.402 | **0.831–0.839** |
| NaNbO3 | 3.126–3.129 | 2.039–2.045 | 0.65 | 6.316–6.328 | 5.217–5.232 | **0.826–0.827** |

**Stop rule** (forward + backward ≤ 0.90 of the chain on NbCl5 and NaNbO3): **met**, in every round.
- The gain is 0.66 ms per force call on NbCl5 and 1.10 ms on NaNbO3; NaCl gains 0.02 ms.
- It equals full fusion on NaNbO3 (T01: 1.10 ms) and comes within 0.07 ms on NbCl5 (T01: 0.73 ms).
- The backward stays today's kernels, bitwise.
- The `w` write costs the fused forward 0.29 ms on NbCl5 and 0.43 ms on NaNbO3.

Step-level estimate (inference, no full-step A/B): about 3–4 % of an NbCl5 step and 7 % of an NaNbO3 step at the
opt-6 step times.

No record and no default were built (owner instruction). A record would replace the forward of
`folded_streaming_convolution` (opt-in), keep its chunked backward, and then need:
- the CPU tests and a GPU exactness test against the chain;
- the gate, NVE drift, and the A/B in both build orders;
- the MH-1 family (POLAR-1, OMOL-0);
- a check on sm_89.
