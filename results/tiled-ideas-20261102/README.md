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
