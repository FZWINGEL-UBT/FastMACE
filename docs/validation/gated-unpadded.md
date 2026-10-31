# gate_epilogue's input-gradient kernel without row padding (31 October 2026)

**Verdict: default on the RTX 5060 Ti and the RTX 4000 Ada (decision 66).**
- **Whole step:** 0.947–0.951 times on the Ada at NbCl5 21,600 atoms, and 0.964 on its memory-rescue (region) path
  at 43,200. On the RTX 5060 Ti, 0.956–0.959 at 14,400.
- **Exactness:** rounding class, deterministic; the force/stress gate passed.
- **Scope:** launches of at least 96 atoms per SM.

[step-roofline.md](step-roofline.md) found `gated_input_grads`, the kernel of `gate_epilogue`'s backward, at 30 % of
FP32 peak against `segment_gemm`'s 50 % for the same product shape. A tile alone could not close the gap. The
remaining cause was structural: the kernel stacks each atom's D = 2l + 1 rows padded to a power of two, so 1, 3 and 1
of every 4, 8 and 8 rows of `tl.dot` are masked (D = 3, 5, 7).

## The kernel

`kernels/triton/gated_products.py`, `_unpadded_kernel` and `_fused_unpadded`:
- **Packing.** One tile of 64 rows holds `64 // D` whole atoms (21, 12 or 9) packed atom-major: row r is component
  `r % D` of atom `r // D`. At most D − 1 rows are masked, and no atom crosses a tile.
- **The products.** One K loop of `tl.dot` per tile, 128 columns, 32 K, 4 warps, 3 stages: the best of 16 tiles on
  three GPUs. 128-row tiles exceed the shared memory.
- **The gate and the gated rows** are read per row through the row's atom. `dgated = acc × gate` is stored
  contiguously (row `atom × D + m`).
- **`dgate`**, the sum over an atom's D rows, is a second `tl.dot`: the 0/1 selection matrix `S[a, r] = (r // D == a)`
  times `acc × gated`. Products by 0 and 1 are exact, and the D terms are summed in `tl.dot`'s order. It stays in
  registers, needs no atomics and is deterministic.
- **Spills.** On the RTX A4000 (sm_86) the tile spills registers (30–314 spills). The speed-ups below include that.

**Selection** (`gated_products.unpadded`): the unpadded kernel runs for a launch of at least `UNPADDED_ATOMS_PER_SM`
(96) atoms per SM, on a GPU in `UNPADDED_GPUS`. The launch is a call's atoms in the whole-system path, or a tile's
in the tiled path. `UNPADDED` = "on" / "off" is an A/B side's override on every GPU. The padded kernel keeps
everything else: smaller launches, the other GPUs, the A100.

## Kernel measurements

`results/gated-unpadded-20261101/unpadded.py`:
- the three gated segments of one call, MH-1's widths (K = N = 512), synthetic operands;
- every variant checked against today's kernel: 0.9–2.0e-7 relative;
- interleaved blocks, medians.

The table gives the unpadded kernel's time over the padded one's.

| GPU | 1,944 atoms | 21,600 atoms | 25,920 atoms (one region chunk at 103,680) |
|---|---:|---:|---:|
| RTX 4000 Ada (leeloo, 48 SMs) | 1.013 | 0.768 | 0.809 |
| RTX A4000 (trillian, 48 SMs) | 1.128 | 0.794 | 0.805 |
| RTX 5060 Ti (local, 36 SMs) | 0.910 | 0.812 | 0.811 |

**Crossover** (14 atom counts from 256 to 21,600, `results/gated-unpadded-20261101/sweep-*`): the unpadded kernel
first wins at about 21 atoms per SM on the 5060 Ti (1,024 atoms), 54 on the Ada (2,592) and 96 on the A4000
(4,608). One rule for all three: 96 atoms per SM, which is 4,608 atoms on 48 SMs and 3,456 on 36. Every GPU wins
there; the Ada and the 5060 Ti give up their smaller gains between their own crossover and the rule's.

## Whole step

`tools/ab_rewrites.py --prepare`: same-process A/B with a = `UNPADDED="off"` and b = `"on"`, in both build orders
(ABAB and ABBA). Default float32 plan.

| GPU | Cell | Blocks | b/a, order ab | b/a, order ba |
|---|---|---|---:|---:|
| RTX 4000 Ada (leeloo) | NbCl5 21,600 atoms | 12 × 100 steps | 0.9471 (216.7 → 205.3 ms) | 0.9513 (215.9 → 205.4 ms) |
| RTX 4000 Ada (leeloo) | NbCl5 43,200 atoms, every rescue record (region path) | 10 × 50 steps | 0.9643 (582.0 → 561.2 ms) | 0.9641 (582.2 → 561.3 ms) |
| RTX 5060 Ti (local) | NbCl5 14,400 atoms | 12 × 100 steps | 0.9587 (127.3 → 122.1 ms) | 0.9560 (127.5 → 121.9 ms) |

- **103,680 atoms** does not fit two A/B sides in 20 GB (out of memory in the second side's build). The region path
  at 43,200 atoms stands in: there, too, every launch is a region chunk (10,800 atoms). The kernel measurement at
  25,920 atoms (0.809) is the chunk size at 103,680.
- **On the 5060 Ti** two 21,600-atom sides do not fit 16 GB. Under WSL2 the second side's memory spills to the host,
  which gave 1.131 against 0.816 depending on the build order. At 14,400 atoms both fit.
- **The RTX A4000** keeps `gate_epilogue` off by default (decision 66), so it does not run this kernel.
  **The A100** is not measured. Both keep the padded kernel.

## Tiled path

One data point (leeloo, RTX 4000 Ada):
- NbCl5 41×41×123 (14,886,936 atoms), the tiled session's setup: tiles of R = 4,096 receivers, tile lists, the state
  on the host (`tools/bench/tiled_capacity.py --tile-list --host-state`), two calls;
- two processes, each from a source copy with `UNPADDED` "off" or "on", so the cross-process noise of about 3 %
  applies.

| Call | Padded | Unpadded | Ratio |
|---|---:|---:|---:|
| First (warms up) | 422.5 s | 411.2 s | 0.973 |
| Second | 411.4 s | 402.3 s | 0.978 |

The tiled call runs this kernel once per tile and interaction, on the tile's nodes (thousands to tens of thousands of
atoms, above the crossover). It saves about 9 s per call here. The largest force agrees to the seventh digit (0.0896501
eV/Å on both sides), and the edge check passed on both.

## Correctness

- **Force/stress gate** (`tools/force_stress_gate.py`, leeloo, a source copy with `UNPADDED = "on"`): passed for
  NaCl, NbCl5 and NaNbO3.
- **CPU tests:** `test_gate_epilogue.py`'s selection test (crossover, GPU list, A/B overrides, not a CUDA device). On
  trillian, the targeted files passed: 428 passed, 20 skipped.
- **GPU tests** (`tests/gpu/test_gated_products_cuda.py`, trillian):
  - the unpadded kernel against the chain within 1e-5 relative, deterministic, with a partial atom tile, on both
    sides of the override;
  - one launch against `drows` for D = 3, 5, 7.
  10 of 11 passed. The failure is the existing padded kernel's `test_stacked_tile_has_no_register_spills[3]`, which
  misses its 1e-5 absolute tolerance by one element (1.3e-5) on the A4000. It is independent of this change.

## Records

`results/gated-unpadded-20261101/`:
- `unpadded.py`, the prototype and benchmark;
- each GPU's kernel and crossover records;
- the A/B records and logs;
- the gate summary.
