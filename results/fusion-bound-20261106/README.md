# Receiver-local fusion bound (6 November 2026)

**Question:** would a kernel that runs the last interaction's forward and backward per receiver tile save time?
After interaction 2 everything up to the energy is node-local, so a receiver's backward seed (`dout`) is known as soon
as its message is complete. Such a kernel would skip the message store and keep `dout`, and the edges' sh (and at
best the sender features x), on chip for the backward.

**Workload:** OMAT-0 medium, float32 hybrid, Benchmark_loup SrTiO3 10×10×10 (5,000 atoms), the plan's own kernels
(`tools/omat0_kernel_lab.py`, commit `1296069d`). Timing-only bounds: the w′ kernel reads the named inputs from a 4 KB
shared array instead of global memory (values garbage), the forward drops its message store (`F-no-out`).

| Variant (ratio to base) | H200 (Modal, profile `third`, about $0.24) | RTX 5060 Ti (local, 7 rounds, desktop not idle) |
| --- | ---: | ---: |
| forward base | 1,670 µs | 6,223 µs |
| forward F-no-out (no message store) | 0.994 | 1.015 |
| w′ base | 1,566 µs | 6,599 µs |
| w′ F-smem-dout | **0.847** | **0.825** |
| w′ F-smem-sh | 1.159 | 0.987 |
| w′ F-smem-x | 1.252 | 1.005 |
| w′ F-smem-dout-sh | 0.878 | 0.866 |
| w′ F-smem-all | 0.863 | 0.923 |

**Reading:**
- The fusion as such gains nothing. The forward's message store is free (0.994), and sh or x from shared memory are
  no faster than their cached global reads; on the H200 they are slower.
- The one real cost is `dout`. Every edge re-reads its receiver's full output gradient (5,120 floats, 20 KB), so a
  call moves about 7.6 GB of `dout` through L1/L2 for 373,760 edges. If those reads were free shared-memory reads,
  w′ would drop by 15 % (H200, about 240 µs, about 4 % of the 5.5 ms step) to 17.5 % (RTX 5060 Ti).
- That lever does not need a fused forward. It needs a w′ that stages each receiver's `dout` once per block of
  receiver-sorted edges, while keeping one edge per warp. The rejected receiver-owned backward
  ([conv-receiver-backward](../../docs/optimisation-ledger.md#conv-receiver-backward)) lost by serialising a receiver's
  edges in one block, so a staging design must keep the edge parallelism. The staging itself (20 KB per receiver, the
  block's shared memory and occupancy) is not in this bound.
- Next falsifier, if pursued: a w′ variant with real staging (cooperative copy of the block's receivers' `dout` into
  shared memory, then the same per-warp edge loop); stop below 5 % of w′ on the H200.
