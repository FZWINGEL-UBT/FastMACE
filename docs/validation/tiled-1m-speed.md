# Tiled call at one million atoms: where the time goes (3 November 2026)

Owner priority from 3 November 2026: no bigger cells on one GPU; make the tiled call as fast as possible at about
1M atoms. This page measures where a 1M call spends its time and which levers are left.

**Setup.**
- **Cell and model:** SrTiO3 58×58×58 (975,560 atoms, 72,191,440 edges in the cutoff, 74 per atom), OMAT-0
  medium, float32, `--tile-list --species-index`, compiled mode, steady third call.
- **GPU:** murphy, RTX 4000 Ada 20 GB, environment `mace_cueq_oeq`.
- **Source:** `b64cf7ba` (profiles) and `95a6f879`/`e7a1a06a` (the new bench options `--sweep-times` and
  `--sweep-profile`, otherwise the same library code) on branch `fastmace/tiled-band-lists`.
- **Records:** `results/tiled-1m-20261103/`.

## Result

The tiled call at 1M runs at about 15 µs per atom (14.5–14.7 s per call), with the GPU busy 97.8 % of the time.
Almost all of that is the model's own per-edge and per-receiver work. Tile size does not change it, and launch or
host overhead is about 2 %. The one tiled-only cost of note is sweep 1, about 10 % of the call.

## Whole-call profile

`--profile-last`, R = 16,384 (68 tiles, the default budget's choice with the state on the device).

| Kernel class | Share |
| --- | --- |
| Convolution (`oeq_projected_backward` 19 %, `oeq_projected_forward` 9 %) | 31 % |
| GEMMs (float32 SIMT) | 20 % |
| Pointwise | 14 % |
| cuEquivariance symmetric contraction | 13 % |
| Copies | 10 % |
| Index | 6 % |

The host spends its time in `cudaMemcpyAsync` waiting for the GPU (11.1 s of self time). So capturing the bodies in
CUDA graphs can remove at most the roughly 2 % of the span when the GPU is idle.

The local RTX 5060 Ti profile (`rtx5060ti-local/`) was taken at R = 8,192, 138 tiles, with both state placements.

## Per sweep

`--sweep-times`: CUDA events around each tile's sweep body, fill and scatter, on the last call.

| Part | State on the device | State on the host (aliased) |
| --- | --- | --- |
| Sweep 1 (interaction 0 forward) | 1.46 s | 1.57 s |
| Sweep 2 (interaction 1 forward and backward) | 8.71 s | 8.85 s |
| Sweep 3 (interaction 0 with autograd) | 3.24 s | 3.34 s |
| Fill and scatter, all sweeps | 0.23 s | 0.40 s |
| Call | 14.5 s | 14.6 s |

`--sweep-profile`: torch.profiler around tiles 4–6 of each sweep. Per tile at R = 16,384, sweep 2 takes about
140 ms:

| Sweep 2 kernel | Share |
| --- | --- |
| `oeq_projected_backward` | 31 % |
| cuEquivariance `kernel` (symmetric contraction) | 15 % |
| `oeq_projected_forward` | 15 % |
| `ampere_sgemm_64x64_tn` | 14 % |
| Other GEMMs | 7 % |
| Slice-backward fusions (halo rows) | about 3 % |

Sweeps 1 and 3 take about 23 and 51 ms per tile. They are mostly GEMMs, the contraction, and copies and
concatenations around the reassociated first interaction.

## Tile size

`--sweep-times --no-budget`: the receiver cap fixed.

| Cap | State | Tiles | Largest halo | GPU peak | Compile call | Steady call |
| --- | --- | --- | --- | --- | --- | --- |
| 16,384 | device | 68 | 8,920 | 12.4 GiB | 76 s | 14.6 s |
| 24,576 | device | 54 | 11,959 | 16.2 GiB | 282 s | 14.6 s |
| 32,768 | device | out of memory | | | | |
| 32,768 | host | 54 | 14,054 | 15.0 GiB | 195 s | 14.7 s |
| 49,152 | host | 27 | 15,134 | 17.2 GiB | 109 s | 15.6 s |

Larger tiles have a smaller halo fraction but are not faster. The sweep times stay the same within 1 %, and at
49,152 they get slightly worse. The halo rows' node work is not where the time goes. The default budget's
R = 16,384 is the right choice at 1M, and it also has the shortest compile.

## What this leaves

- **Sweep-2 conv on the real rows only:** worth about the 3 % of slice-backward fusions. Parked as small.
- **Sweep 3 without interaction 0's forward:** at most the forward part of sweep 3 (about sweep 1's 1.5 s, 10 %).
  Its backward needs the forward's saved activations, though: per-edge tensors, estimated (not measured) at about 30 GB at 1M, which would
  have to round-trip over PCIe each call. Storing only u does not remove the recompute. Not pursued.
- **CUDA graphs for the bodies:** at most the 2 % idle span.
- **The remaining 85 %:** the same OEq convolution, contraction and GEMM kernels as the whole-system engine, which
  runs at 12.5–14.1 µs per atom on this GPU up to about 69k atoms. They have been screened at length
  (`docs/optimisation-ledger.md`: 3×TF32 and 3×FP16 GEMMs, own convolutions, receiver backward and others). Gains
  there are general FastMACE kernel work, not tiled-path work.
