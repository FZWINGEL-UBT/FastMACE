# Group-ordered wprime_backward falsifier (31 October 2026)

**Verdict: not GO. Recommend stop.**

- The group order makes NbCl5 slower on both GPUs: 1.29× on the RTX 4000 Ada, 1.43× on the RTX 5060 Ti.
- It helps NaNbO3 only on the Ada (0.80×). On the RTX 5060 Ti NaNbO3 is 1.02×.
- The sort itself costs 118–130 µs on the RTX 5060 Ti, against the 50 µs limit.
- The strict STOP rule (≥ 0.92× everywhere) is not met, because Ada NaNbO3 reaches 0.797×.

Item 1 of the ninth list, plans/capacity-ideas-20261030.md. No library code was changed.

## Premise check: the built kernel is edge-owned

`fastmace/kernels/oeq_streaming/wprime_backward.py`, `oeq_projected_backward_wprime`:

- **Ownership:** one 32-lane group per edge, `e = blockIdx.x * groups_per_block + group`.
- **Per-edge arrays:** `sh`, `weights`, `lengths`, `pairs`, `dsh[e]` and `dr[e]`.
- **Atomics:** the sender gradients go into `dx` atomically.
- **No `row_ptr`.**

So the kernel runs correctly in any edge order, as the item assumed. The probe confirms it: with the order permuted,
`dsh` and `dr` are bitwise equal, and `dx` is equal up to the atomic order (|Δ| ≤ 1.2e-7 of 0.23).

## Method

`results/wprime-group-order-20261031/wprime_group_probe.py`:

- **Plan and run:** the default float32 plan (MinimalPlanner, hybrid), NVT 1 fs at 1000 K, 300 equilibration steps.
- **Capture:** one eager call of the engine's force field. Its `WPrimeBackward` launches are recorded with real
  operands. There is one launch per call: interaction 0, the one with a radial table.
- **Replay:** each launch in a CUDA graph of 20 launches (each zeroing `dx` first, as the op does), 15 alternating
  blocks, median.
  - (a) The edges as captured, in receiver order.
  - (b) The per-edge arrays permuted by the group key g(e) = pair × pieces + piece, stable, so edges keep receiver
    order within a group. Unsloped edges go last. The outputs are inverse-permuted.
- **Not timed:** the permutation's gathers. A built version would store the edges in group order.
- **Also timed:** `torch.argsort` of the key, stable and unstable.
- MH-1 r2SCAN head; FastMACE a0c06074 (main 5c975d8d plus the probe).

## Results

E edges; G′ groups used; Q = 1,280 weights per edge; 10 Chebyshev terms. In receiver order the group changes on
91 % of consecutive edge pairs (NbCl5: 51,523 of 56,320; NaNbO3: 74,919 of 87,040).

| GPU | Cell | E | Groups | (a) receiver order (µs) | (b) group order (µs) | b/a | argsort stable / unstable (µs) |
|---|---|---|---|---|---|---|---|
| RTX 4000 Ada (leto) | NbCl5 3×3×3, 1,944 atoms | 56,320 | 67 | 1,976.6 | 2,546.7 | **1.288** | 42.9 / 41.9 |
| | NaNbO3 3×3×3, 1,080 atoms | 87,040 | 141 | 3,114.8 | 2,482.2 | **0.797** | 49.2 / 42.7 |
| RTX 5060 Ti (local) | NbCl5 3×3×3 | 56,320 | 67 | 1,750.1 | 2,501.1 | **1.429** | 124.0 / 129.6 |
| | NaNbO3 3×3×3 | 87,040 | 141 | 2,693.5 | 2,749.7 | **1.021** | 117.6 / 121.6 |

**Against the decision rule:**
- **GO** (NaNbO3 ≤ 0.80 and NbCl5 ≤ 0.90 on one GPU, sort ≤ 50 µs): not met on either GPU. NbCl5 is slower on
  both. On the RTX 5060 Ti the sort alone is 2.4× the limit.
- **STOP** (≥ 0.92 everywhere): not met only because of Ada NaNbO3 at 0.797.

**Why the order hurts.**
- In receiver order, neighbouring edges share the receiver's `dout` row and often the sender's `x` row, and those
  gathers dominate the kernel.
- In group order those reads scatter over all atoms.
- All the tables fit in L2 on both GPUs: NbCl5's 67 groups are 3.4 MB, NaNbO3's 141 are 7.2 MB (10 × 1,280 floats
  per group). So the group order can only win L1 and load-latency reuse within a block, not DRAM traffic.
- That small gain outweighs the scattered gathers only for NaNbO3 on the Ada. Why there and not on sm_120 was not
  profiled.
- OPT9-T01's isolated 2.46 → 0.46 ms (Ada, NaNbO3) timed the table term alone, without the node gathers that the
  fused kernel also does.

**Recommendation.**
- Stop. A gain on one structure and one GPU, a loss on the other structure on both GPUs, and a sort over budget on
  sm_120 are not a general rule (SPEC §1.1 goal 2).
- A per-structure switch would be a structure-keyed choice, which decision 66 does not allow.
- **Variant not measured:** receiver-major order with the group as a tie-break within each receiver. It would not
  change the kernel's work, because edges are already receiver-sorted.

## Records

`results/wprime-group-order-20261031/`:
- `ada-leto/wprime_group_probe.json` and `run.log`: leto; NAS `mace_cueq_oeq`; driver 595.99.02.
- `local-5060ti/wprime_group_probe.json` and `run.log`: local WSL2; `mace_cueq_oeq`; driver 610.88.
