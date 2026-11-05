# segment_gemm tile order (30 October 2026)

**Verdict:** a row tile's column tiles now run next to each other when the product's A is larger than the L2. The
results are bitwise equal. The change applies on the RTX 5060 Ti and the RTX 4000 Ada (decision 66) and makes the
step 5–8 % faster at 8,640–9,216 atoms. Cells whose A fits in the L2 keep the old order, and so does every other
GPU, the A100 included.

Branch `fastmace/segment-gemm-order`. The idea came from the C1 falsifier
([c1-contraction-fusion.md](c1-contraction-fusion.md)). At NbCl5 9,216 atoms, MH-1's `linear_2` products read A
8.5 times from DRAM.

## The change

- `segment_gemm` launched a grid of (row tiles, column tiles), row tile fastest. The column tiles of one row tile
  therefore ran a whole pass apart, and an A larger than the L2 was read from DRAM once per column tile.
- With `GemmChoice.column_fastest`, the grid is 1-D with the column tile fastest. The tiles and the K order are the
  same, so the results are bitwise equal.
- `tile_rule.columns_together(m, k, element_bytes, l2_bytes, gpu)` turns it on when:
  - the GPU is in `COLUMN_ORDER_GPUS` (exact device names, decision 66), and
  - `m * k * element_bytes` is larger than the device's L2 (`L2_cache_size`).
- It covers both kernels of `joined_products`: the products and the input gradients. `gated_products`' input
  gradients go through the same `_choices`.
- `tile_rule.TILE_ORDER` ("auto", "rows", "columns") is a string constant for the A/B sides only, after the
  `RECOMPUTE_KERNEL` precedent.

## Correctness

- **Bitwise GPU tests**, `tests/gpu/test_segment_gemm_cuda.py`:
  - `test_column_tiles_together_equal_the_row_order_bitwise`: `_gemm` in both orders, with edge tiles and a NaN
    prefill.
  - `test_joined_products_order_columns_by_the_l2_rule`: the joined output and the input gradients are equal in
    both orders.
  - With `test_tf32x3_gemm_cuda.py`, 13 passed on each GPU: RTX 5060 Ti, RTX 4000 Ada and A100-SXM4-80GB.
- **Kernel probe** (`results/segment-gemm-order-20261030/gemm_order_probe.py`): bitwise equal for every product on
  the three GPUs.
- **Force/stress gate** (default plan; NaCl, NbCl5, NaNbO3 at 3×3×3; RTX 5060 Ti, bad26503): passed. Its cells
  are below the L2 threshold, so it ran the old order. The bitwise tests carry the new one.
- **CPU:** `tests/cpu/test_tile_rule.py` covers the L2 rule, its boundary, the GPU list and the overrides. At
  2e8fa317 it passed with `test_ab_rewrites_tool.py` and `test_layering.py` (42 tests) on leto with the GPU hidden.

## Kernel probe: MH-1 `linear_2` products, float32 IEEE, K = N = 512

New order's time over the old one's, at the same Triton tiles.

| GPU (L2) | 9,216 atoms, l = 0 | 9,216, l = 1..3 | 9,216, sum | 51,840, sum |
|---|---|---|---|---|
| RTX 5060 Ti (32 MiB) | 1.064 (19 MB: the rule keeps rows) | 0.80–0.84 | 0.838 | 0.832 |
| RTX 4000 Ada (40 MiB) | 0.992 (rows kept) | 0.747–0.762 | 0.767 | 0.787 |
| A100-SXM4-80GB (40 MiB) | cuBLAS | 1.002–1.003 | 1.003 | 1.003 |

The A100's HBM2e serves the 8.5 reads without a loss, so the order gains nothing there.

## Whole step: same-process A/Bs, both build orders

Setup:
- `tools/ab_rewrites.py --prepare`, MH-1 hybrid float32, NVT 300 K.
- Side a: `TILE_ORDER="rows"`, the order before. Side b: the default.
- The 3×3×3 cells ran 12 blocks of 300 steps. The large cells ran 100 warm-up steps, then 8 blocks of 100.
- Figures are b/a, build order ab / ba. Every run was valid.

| Cell | RTX 5060 Ti | RTX 4000 Ada (leto) | A100-SXM4-80GB (Modal) |
|---|---|---|---|
| NbCl5 4×4×8, 9,216 atoms | **0.940 / 0.938** | **0.925 / 0.923** | 1.0017 / 1.0015 |
| NaNbO3 6×6×6, 8,640 atoms | **0.959 / 0.955** | **0.945 / 0.944** | 1.0017 / 0.9990 |
| NbCl5 3×3×3, 1,944 atoms | 0.998 / 1.004 | 1.000 / 0.999 | 1.0002 / 0.9998 |
| NaNbO3 3×3×3, 1,080 atoms | 1.000 / 0.998 | 1.001 / 1.000 | 1.0016 / 0.9989 |
| NaCl 3×3×3, 216 atoms | 0.998 / 1.004 | 0.998 / 1.004 | 0.994 / 1.002 |

- **Small cells:** at 3×3×3 every A fits in the L2, so both sides ran the same kernels. These rows are the
  noise floor.
- **The A100's loss** at NbCl5 9,216 is consistent but small (0.15 %, kernel 0.3 %). Decision 66 keeps the
  default off there: the list names only the GPUs that measured a speed-up.

The measured ratios are in the `COLUMN_ORDER_GPUS` comment in `tile_rule.py`. The column order is a default only on
`NVIDIA GeForce RTX 5060 Ti` and `NVIDIA RTX 4000 Ada Generation`. A GPU not in the list, such as the A100-SXM4-40GB
or the L4, keeps the old order until its own A/B.

## The A/B tool bug found on the way

The first round of these A/Bs gave b/a 1.000 everywhere: the tool had compared the new order with itself.

- `--a-constant` was set only around a side's build. Graphs captured again in warm-up read the module's value at
  that moment, so both sides ran the new order. `order_trace.py`, a trace of `segment_gemm._gemm`, showed it.
- The fix is merged as d16d41c6: `ConstantsSide` sets each side's constants before every step.
- The audit of earlier constant A/Bs is in [ab-constants-audit-20261030.md](ab-constants-audit-20261030.md).
- All the A/Bs above ran with the fixed tool. The trace of the fixed run shows each side capturing in its own
  order.

## Records

All under `results/segment-gemm-order-20261030/`:

| Folder | GPU | Machine and environment | Source | What |
|---|---|---|---|---|
| `local-5060ti/` | RTX 5060 Ti, driver 610.88 | local WSL2, `mace_cueq_oeq` | 701feeaa | A/Bs |
| `local-5060ti/` | RTX 5060 Ti | as above | bad26503 | GPU tests and gate |
| `ada-leto/` | RTX 4000 Ada, driver 595.99.02 | leto, NAS `mace_cueq_oeq` | 701feeaa | GPU tests, probes, trace, A/Bs |
| `modal-a100/` | A100-SXM4-80GB, driver 610.57.04 | Modal image | 0d164b27 | GPU tests, probes, A/Bs; $1.47 |

- **Model:** MH-1 r2SCAN head (`mace-mh-1.model`).
- **Same kernel code:** 701feeaa and 0d164b27 differ from the final branch only by tools and scripts. The GPU list
  (2e8fa317) leaves the RTX 5060 Ti and the Ada on the measured rule.
- **Scripts:** `local.sh`, `ada.sh`, `tools/modal/sgo_a100.sh`, `gemm_order_probe.py` and `order_trace.py`.
- **Baseline:** none needed. Both sides run FastMACE in one process, side a in the order main had before.
