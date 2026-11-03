# Tiled speed ideas, 2 November 2026

A/Bs of the tiled call after the wavefront (owner: "Try out all of your ideas"). SrTiO3, MACE-OMAT-0 medium,
hybrid backend, float32 forces, `tools/bench/tiled_capacity.py` with `--tile-list --host-state --alias-state
--species-index --no-budget`, one process per run, mode `auto` (eager first call, then compiled). Times are the
steady compiled calls (the last one or two of each run). The report is `docs/validation/tiled-ideas-20261102.md`.

| Directory | GPU, provider | Environment | Source |
| --- | --- | --- | --- |
| `rtx4000ada-leto/` | NVIDIA RTX 4000 Ada Generation, leto | NAS `mace_cueq_oeq`, `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` for the lanes runs | per run, `idle_check.txt` |
| `rtx5060ti-local/` | NVIDIA GeForce RTX 5060 Ti, local PC (WSL) | `~/environments/mace_cueq_oeq` | `dde2a457` (worktree) |
| `rtxpro6000-modal/` | NVIDIA RTX PRO 6000 Blackwell Server Edition, Modal | FastMACE Modal image (`modal.json`) | `009dea0c` |

## Lanes (5M atoms, R = 8,192, leto)

| Variant | Lanes off | Lanes on |
| --- | --- | --- |
| three sweeps | 94.4 s | 89.9 s (−4.8 %) |
| wavefront | 98.4 s | 96.1 s (−2.3 %) |
| R = 16,384, three sweeps | 89.4 s | "auto" off ("a second tile workspace does not fit", peak 7.9 of 20 GiB) |

One steady call per variant: not significant yet.

## Streaming backward chunk size (1.08M atoms, wavefront, R = 8,192)

| GPU | ×1 | ×4 | ×8 |
| --- | --- | --- | --- |
| RTX 5060 Ti (1,536 edges at ×1) | 23.4, 25.4, 24.7, 24.7 s | 21.9, 22.5, 21.9, 21.6 s (−10.5 %) | 22.0, 23.2 s |
| RTX 4000 Ada (2,048 edges at ×1) | 19.4, 19.5, 19.8, 19.8 s | 19.0, 19.1, 18.9, 19.0 s (−3.2 %) | 18.9, 19.0 s |
| RTX PRO 6000 | does not chunk at tile sizes (128 MB L2) | 8.01, 8.18 s (no change) | |

## Prefetch (`TileConfig.prefetch`)

| GPU, size | off | on |
| --- | --- | --- |
| RTX PRO 6000, 1.08M, wavefront | 8.76, 9.18, 8.43, 8.45 s | 8.16, 8.16, 8.03, 8.11 s (−3.8 % against the second off run) |
| RTX 4000 Ada, 5M, wavefront | 97.7, 97.6 s | 91.6, 91.5 s (−6.2 %) |

Both on sources before the wavefront groups (`f78292be`): the wavefront then changed sweep at every tile, so only
the graphs were prefetched, not the host rows.

## Search pass

0.7–0.8 s of a 22 s call (1.08M, 5060 Ti), 1.9 s of a 98 s call (5M, Ada): about 2–3.5 %. Not built further.

## Wavefront group, halo trim and boundary skin (5M SrTiO3, RTX 4000 Ada on leto, 2–3 November 2026)

Wavefront on, R = 8,192, lanes off. Calls: eager, compiled with compilation, steady.

| Variant (source) | Calls | Row pool | Result |
| --- | --- | --- | --- |
| `wavefront_group` 1 (`91f646b7`) | 157.8/111.1/88.0 s; 156.6/107.5/87.8 s | 36.6 % | |
| `wavefront_group` 16 (`91f646b7`), default | 153.2/102.2/82.6 s; 153.5/102.2/82.5 s | 37.5 % | −6.1 % steady |
| Before the trims (`91f646b7`) | 82.4, 82.4 s steady | 37.5 % | |
| Halo trim (`c988f5ff`) | 82.5, 82.4 s | 37.5 % | no change; kept |
| Boundary skin (`c988f5ff`) | 84.4, 84.5 s | 32.4 % | +2.4 % for 3 % less host; not adopted |
| Default budget, lanes auto, wavefront auto (`c988f5ff`) | 110.8 s at R = 16,384, wavefront off | 0 | see the receiver-cap A/B |

## Band cache of the kept lists and the single-stream backward (5M SrTiO3, 2–3 November 2026)

Three arms, a/b order reversed. Steady compiled call; `list_cache_hits` 2,208 = 2 × 1,104 tiles with the band cache.

| GPU (source) | Base | Band cache | Band cache + single-stream backward |
| --- | --- | --- | --- |
| RTX 4000 Ada, leto (`c446d26f`) | 81.8, 81.8 s | 80.5, 80.5 s (−1.6 %) | 79.7, 79.6 s (−2.6 %); GPU peak 4.40 → 4.36 GiB |
| RTX 5060 Ti, local (`c446d26f`) | 97.8, 92.9 s | 95.6, 97.5 s | 94.0, 91.9 s (noisy, desktop not idle) |

Both are kept. The single-stream backward (one L2-sized slot on the current stream once a slot is at least half the
L2) is the default on these two GPUs by device name (`SINGLE_STREAM_DEVICES`, decision 66). Records:
`rtx4000ada-leto/leto-band-ab-*`, `rtx5060ti-local/{base,band,single}-*`.

## Receiver cap at 5M (SrTiO3, RTX 4000 Ada on leto, source `f5b158a1`, 3 November 2026)

Host state, aliased, lanes off, wavefront auto (off: the host has room). Order r8 r16 budget budget r16 r8.

| Cap | Tiles | GPU peak | Steady call |
| --- | --- | --- | --- |
| 8,192 (`--no-budget`) | 1,024 | 4.39 GiB | 78.5, 78.5 s |
| 16,384 (`--no-budget`) | 343 | 7.89 GiB | 79.4, 79.2 s |
| Default budget from 4,096 (goes to 16,384) | 343 | 7.89 GiB | 79.4, 79.1 s |

The cap changes the steady call by about 1 %, as at 1M (`docs/validation/tiled-1m-speed.md`). The 110.8 s of the
"lanes auto" arm above came from that arm's other settings and source (`c988f5ff`), not from R = 16,384. The total
energy differs by 0.46 eV (1e-8 relative) between the two tilings: float32 summation order. Records:
`rtx4000ada-leto/leto-cap-ab-*`.
