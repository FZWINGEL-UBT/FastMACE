# Chunked conv backward on the A100 (3 November 2026)

Modal, NVIDIA A100-SXM4-80GB, driver 610.57.04, source 5c6cf6a1, `tools/modal/a100_chunk_size.sh`; estimated GPU cost
$0.83. MACE-OMAT-0 medium, SrTiO3 10×10×10 (5,000 atoms), float32, skin 0.5 Å, 1 fs, both build orders.

Side a is the A100 default then (radial tables on, chunked backward off); side b adds the chunked backward. The study
is cited from `docs/validation/generic-wprime.md`.

| Side b | Step a (ms) | Step b (ms) | b/a, ab | b/a, ba |
|---|---|---|---:|---:|
| chunk 0.25 of the L2 (the rule) | 22.78 / 22.77 | 29.79 / 29.89 | 1.308 | 1.313 |
| chunk 1 × the L2 | 22.76 / 22.76 | 23.13 / 23.16 | 1.016 | 1.018 |
| chunk 4 × the L2 | 22.77 / 22.78 | 22.58 / 22.59 | 0.992 | 0.992 |
| unchunked inside the op | 22.79 / 22.79 | 22.51 / 22.48 | 0.988 | 0.987 |

Note on the files: the two 0.25 runs both wrote `ab-share0.json` (the output name lost `.25`), so that file is the ba
order; the ab order survives only in `ab-share0.25-ab.log` and `run.log`.
