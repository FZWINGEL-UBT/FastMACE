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
- The conductor's FF-dout-one-row falsifier (ideas round 2, H200, `2472c0e0`: every edge reads receiver 0's row,
  fully cache-resident) gave 1.001, so the bytes cost nothing; F-smem-dout's gain is load latency or issue (and
  possibly its 80 instead of 96 registers on the H200), which only a real staging could show.

## Staging measured (commit `32aca599`): stopped

Exact lab variants: the block's (4 warps, 4 receiver-sorted edges) first and last receivers' `dout` rows copied to a
static shared array once, `__syncthreads`, then one generic pointer per `dout` block chosen per warp (shared row if the
warp's receiver was staged, global row otherwise). `F-stage-dout-nofill` reads the same layout with real per-receiver
offsets but skips the copy and the barrier.

| Variant | H200 (`h200-stage/`, about $0.32) | RTX 5060 Ti (`rtx5060ti-stage/`, 9 rounds) |
| --- | ---: | ---: |
| w′ base | 1,565 µs | 6,591 µs |
| F-smem-dout (4 KB array) | 0.848 | 0.821 |
| stage-dout (2 rows, 40 KB) | **1.986** (dsh, dr bitwise) | 2.323 |
| stage-dout-r0 (1 row, 20 KB) | **1.935** (dsh, dr bitwise) | 1.464 |
| F-stage-dout-nofill | **1.131** | 1.373 |

A first version selected the source per read (three-way ternary) and spilled 40–248 bytes on the RTX 5060 Ti
(2.479 / 1.415 / 1.605); the numbers above use the per-warp pointers. The 20–40 KB of shared memory per 4-warp block
cuts the resident blocks, and the copy plus barrier exposes the fill latency on every 4 edges. Even without the fill
the realistic layout is slower than base. **Stop rule (< 5 % of w′ on the H200) not met: dout staging is rejected.**
A design with larger blocks would need more warps per block than the 96-register cap allows on the H200 (about 21
resident warps per SM).
