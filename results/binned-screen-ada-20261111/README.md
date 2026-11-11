# BINNED_SCREEN on the RTX 4000 Ada (11 November 2026)

Source: the conductor's B2 in `plans/next-steps-20261111.md`. The cell-list neighbour screen was on for the H200, the
A100 family and (since [binned-screen-5060ti](../binned-screen-5060ti-20261111/README.md)) the RTX 5060 Ti, and "not
measured" on the RTX 4000 Ada.

- **Machine:** murphy, NVIDIA RTX 4000 Ada Generation (20 GB), driver 595.99.02, `mace_cueq_oeq`, hybrid backend,
  float32, OMAT-0 medium (`omat0-medium`).
- **Source:** FastMACE main `f31d4f50`, a `git archive` snapshot on the shared home.
- **Sides:** a = `fastmace.neighbors.pair_search:BINNED_SCREEN=False` (today's Ada default), b = `True`.
- **Setup (as on the 5060 Ti):** `tools/ab_rewrites.py`, NVT 300 K, 1 fs, skin 0.5 Å, warm-up 200 steps, 6
  alternating blocks of 128 steps per side; build order ab (pattern abab) and ba (abba).
- **Jobs:** [jobs/ada_binned.sh](jobs/ada_binned.sh). Each run waits for two quiet minutes (no GPU process, no pytest,
  no other user's busy CPU), samples the same check every 15 s while it runs, and is repeated when anything ran
  beside it. Records: [records/](records/).

## Result

**The binned screen is faster in every run, and the gain grows with the cell.** The list is the same bit for bit, so
by decision 66 one not-slower A/B in each order decides it: the Ada joins `BINNED_SCREEN_GPUS`.

| SrTiO3 cell | Atoms | Order | a (ms/step) | b (ms/step) | b/a | Pair ratios |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 16×16×16 | 20,480 | ab | 76.30 | 75.12 | **0.9846** | 0.98–0.99 |
| 16×16×16 | 20,480 | ba | 76.76 | 75.31 | **0.9810** | 0.98–0.99 |
| 20×20×20 | 40,000 | ab | 153.78 | 148.51 | **0.9657** | 0.96–0.98 |
| 20×20×20 | 40,000 | ba | 154.32 | 148.18 | **0.9602** | – |

- **Block spread:** at most 1.9 % per side.
- **20×20×20 ba:** the first try overlapped another session's pytest and was repeated. Both tries read 0.9602.
- **Earlier attempts:** two 16×16×16 ab runs read 0.9844 and 0.9838, but overlapped another session's CPU suite. They
  are not in the table.

## The rebuild itself

`tools/bench/rebuild_parts.py`, SrTiO3, cutoff 6 Å, median of 3 ([records/parts.json](records/parts.json)):

| Cell | Atoms | Screen, tile → binned (ms) | `build_gpu`, tile → binned (ms) |
| --- | ---: | --- | --- |
| 16×16×16 | 20,480 | 18.2 → 0.67 | 21.9 → 4.4 |
| 20×20×20 | 40,000 | 68.9 → 0.94 | 76.6 → 8.6 |

These agree with the Ada's earlier rebuild timing (22.3 → 5.1 ms at 20,480 atoms,
[ideas queue 2](../ideas-queue2-20261106/README.md)) and with the MD gain:

- **16×16×16:** about 17.5 ms saved per rebuild, at one rebuild per ~15 steps, is about 1.2 ms per step. Measured:
  1.2–1.5 ms.
- **20×20×20:** about 68 ms saved per rebuild is about 4.5 ms per step. Measured: 5.3–6.1 ms.

## Change

`pair_search.BINNED_SCREEN_GPUS` gains `"NVIDIA RTX 4000 Ada Generation"`. As everywhere, it applies from
`BINNED_MIN_ATOMS` = 10,000 atoms. The change also updates the `tests/cpu/test_gpu_models.py` assertion, the
defaults-by-gpu rows, and the ledger row `nl-binned`.
