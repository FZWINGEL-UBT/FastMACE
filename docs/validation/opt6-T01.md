# OPT6-T01: L2-resident edge chunks for the folded radial layer and the streaming convolution (26 October 2026)

Plan: `plans/opt-6.md`. Probe: `tools/bench/opt6_l2_chunked_conv.py`; runs and logs:
`results/opt6-T01-l2-chunks-20261026/` (`run.sh`, `local-a` to `local-d-*`).

**Verdict: exact; NbCl5 passes the falsifier, NaNbO3 misses it narrowly, NaCl gains nothing.**
- Chunking the **backward** wins on both large cells. The edge-owned streaming backward writes `dw` one
  chunk at a time into a 1,536-edge scratch (7.9 MB per slot, two slots), and the input gradient reads it
  from L2 on a side stream. This saves **0.61 ms on NbCl5 and 0.93 ms on NaNbO3** per force call.
  - This part needs no receiver alignment: fixed edge ranges over the padded edges would do, so its shapes
    are static under CUDA graphs.
  - It also removes the `[E, 1280]` `dw` tensor (268 MB on NbCl5, 430 MB on NaNbO3).
- Chunking the **forward** wins only where receivers have few edges. The receiver-streaming forward runs one
  thread per (receiver, channel), so a chunk that fits in L2 holds few receivers and leaves the GPU
  under-filled:
  - a 4,096-edge chunk is 152 receivers on NbCl5 (27 edges per atom), but 52 on NaNbO3 (78 edges per atom);
  - with the GEMMs on a side stream, NbCl5's forward gains a further 0.51 ms (1.90 → 1.39 ms);
  - NaNbO3's forward loses at every chunk size.
- Stop rule: forward + backward ≥ 1.0 ms on NaNbO3 and ≥ 0.6 ms on NbCl5.
  - NbCl5: best combination **−1.06 ms**, a pass.
  - NaNbO3: **−0.93 ms**, a narrow miss.
  - Against opt-4's untimed step baseline (`ccab528`: NbCl5 21.9 ms, NaNbO3 18.4 ms), this is about 4.8 % and
    5.0 % of the step. That estimate is an inference: no full-step A/B has been run.

## Setup

- **Machine:** local reference PC, NVIDIA GeForce RTX 5060 Ti (sm_120, 36 SMs, 32 MiB L2), WSL2.
  - Environment: `~/environments/mace_cueq_oeq` (torch 2.14.1+cu130, Triton 3.8.0).
  - Reservation: `tools/ci/local_gpu.py run --quiet`.
  - The desktop ran during the runs, as always on this PC.
- **FastMACE:** branch `fastmace/opt6-T01` from `main` `b422e67f`. Library code is unchanged; only the probe
  was added. The probe records `fastmace_commit: null`, because git does not resolve a Windows worktree from
  WSL.
- **Model and plan:** `mh1-r2scan`, float32, the default plan (hybrid, `streaming_conv` on).
  - Cells are 3×3×3, rattled by ±0.05 Å (seed 20261026).
  - Real edge counts: NaCl 6,912, NbCl5 52,414, NaNbO3 84,048.
- **Operands:**
  - Real, from one force call (recorded at the streaming kernels' launch): the second interaction's node
    features, spherical harmonics, senders, receivers, CSR rows and output gradient.
  - Random, with the production shapes: `a [E, 65]` and `W [1280, 65]`. Timing does not depend on their
    values.
  - The padded rows of `a` are zero, as `[h c, c]` is with `c = 0`.
- **Kernels:**
  - every variant uses the same `torch.mm` (cuBLAS, IEEE FP32), the vendored streaming forward and backward,
    and the `folded_linear` Triton input gradient with the `waves` tile rule;
  - the chunked forward is the vendored forward with a receiver range (`oeq_projected_forward_range`, made by
    text substitution in the probe only);
  - the chunked backward is the unchanged kernel, called with edge-pointer offsets.
- **Timing:**
  - each variant is captured as one CUDA graph of 20 calls;
  - every call first overwrites a buffer of twice the L2 size; that flush is timed on its own and
    subtracted;
  - the graphs alternate block by block in one process: 9 blocks for `local-a` to `local-c`, 15 for
    `local-d`; medians are reported.
- **Exactness:** message, `dsh` and `da` are **bitwise equal** to the unchunked path in every variant and
  size.
  - `dx` differs by at most 2.6e-7 relative. That is the order of the existing sender-gradient atomics,
    already nondeterministic in production.

## Results (ms per force call of this subgraph, flush subtracted)

`local-d` (15 blocks; base, `mixed1536` and `combo1536` ranges do not overlap):

| Cell | Base fwd / bwd / total | Best backward-only (`mixed`, 1,536 edges) | Best combination | Gain |
| --- | --- | --- | --- | ---: |
| NaCl | 0.174 / 0.343 / 0.505 | 0.183 / 0.323 / 0.490 | – | 0.015 (noise) |
| NbCl5 | 1.899 / 2.685 / 4.567 | 1.871 / 2.108 / 3.960 (−0.61) | `combo1536`: forward pipelined on 4,096-edge chunks, 1.391 / 2.123 / 3.511 | **1.057** |
| NaNbO3 | 2.894 / 4.204 / 7.081 | 2.882 / 3.315 / 6.154 | backward-only | **0.928** |

Backward chunk size (`mixed`, `local-c` and `local-d`): the optimum is sharp.
- 1,536 edges is best on both cells.
- At 2,048 the gain halves. With w streaming in from DRAM, the L2 holds far less reusable data than its
  32 MiB.
- At 768 and below, launches dominate. 217 chunks of 256 edges make NbCl5's backward 7.5 ms.

Forward variants (`local-a` and `local-b`):
- `store`: chunked forward on one stream. NbCl5 4,096 edges 1.45 against 1.88 ms; NaNbO3 always slower,
  4.9 ms at 2,048.
- `recompute`: no stored `w`, the GEMM again in the backward. Slower everywhere: the extra GEMM costs more
  than the traffic it saves.
- `pipe`: GEMMs on a side stream. NbCl5 4,096 edges 1.38 ms; NaNbO3 3.04–6.55 against 2.89 ms.

## What a rewrite would need (not built)

1. **Backward-only, exact.** The candidate to build first.
   - One autograd function over the folded last layer and the streaming convolution, so that `dw` is never
     returned.
   - Backward: fixed edge chunks over the padded edge count; per chunk, the edge-owned backward into a
     two-slot scratch, then the Triton input gradient on a side stream.
   - Chunk size: about 1,536 edges on this GPU. A general rule must come from the L2 size and Q (here
     2 × 1,536 × 5,120 B ≈ L2 / 2); checking that on sm_89 is still to do.
2. **Forward, only for low-degree cells.** Needs either receiver-aligned chunks known on the device (the
   kernel would take its receiver range from device memory, since boundaries move every step) or fixed edge
   ranges with a `+=` at the receivers that straddle a boundary.
   - It gains only where edges per atom are low (NbCl5), so it would need a degree rule. Lower priority.
3. Before adoption: the force gate, a same-process full-step A/B on NaCl, NbCl5 and NaNbO3 in both orders,
   and the same on an sm_89 GPU when workstations are allowed again.
