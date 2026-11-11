# BINNED_SCREEN on the RTX 5060 Ti (11 November 2026)

Source: an owner request (11 November 2026, "Yes, run the BINNED_SCREEN A/B on the 5060 Ti"), after the review in
`plans/notes/rejected-ledger-20261111/review-old-and-resources.md` found the cell-list neighbour screen off on the
RTX 5060 Ti and the RTX 4000 Ada (`docs/current/defaults-by-gpu.md`: "not measured").

- **Machine:** local PC, WSL2, NVIDIA GeForce RTX 5060 Ti 16 GB, driver 621.14, `~/environments/mace_cueq_oeq`,
  hybrid backend, float32, OMAT-0 medium (`omat0-medium`).
- **Source:** FastMACE main `b951e804`, a `git archive` snapshot outside every checkout.
- **Sides:** a = `fastmace.neighbors.pair_search:BINNED_SCREEN=False` (today's 5060 Ti default), b = `True`.
- **Runs:** two jobs inside `tools/ci/local_gpu.py run` reservations (quiet), scripts in [jobs/](jobs/), records in
  [records/](records/). Watchdog: card above 14,500 MiB, a new soft-lockup line, or the step's time limit.

## Result

**The binned screen is faster in every run, and the gain grows with the cell.** It gives the same neighbour list bit
for bit (`tests/gpu/test_neighbors_gpu.py`), so by decision 66 one not-slower A/B decides it.

| SrTiO3 cell | Atoms | Order | a (ms/step) | b (ms/step) | b/a | Block spread a / b |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 16×16×16 | 20,480 | ab (second job) | 66.12 | 64.88 | **0.9812** | 4 % / 3 % |
| 16×16×16 | 20,480 | ba | 66.11 | 64.58 | **0.9769** | 1 % / 1 % |
| 20×20×20 | 40,000 | ab | 132.97 | 127.84 | **0.9614** | 3 % / 4 % |
| 20×20×20 | 40,000 | ba | 140.60 | 130.27 | 0.9265 (noisy) | 14 % / 15 % |
| 16×16×16 | 20,480 | ab (first job) | 70.98 | 70.69 | 0.9958 (noisy) | 18 % / 19 % |

`tools/ab_rewrites.py`, NVT 300 K, 1 fs, skin 0.5 Å, warm-up 200 steps, 6 alternating blocks of 128 steps per side.

- **Clean runs:** the pair ratios lie within 0.973–0.988 at 16×16×16 and 0.958–0.969 at 20×20×20.
- **Noisy runs:** two runs drifted by 14–19 % between blocks on both sides. The tool marked them valid. The first
  was the first process on a cold compile cache; during the second this session ran file searches on the PC. Both
  medians still favour b, and in the 20×20×20 "ba" run all six pair ratios are below 1 (0.832–0.983). They are kept
  in the table but should not be quoted.

## The rebuild itself

`tools/bench/rebuild_parts.py`, SrTiO3, cutoff 6 Å, median of 3 (`records/out/parts.json`):

| Cell | Atoms | Screen, tile → binned (ms) | `build_gpu`, tile → binned (ms) |
| --- | ---: | --- | --- |
| 10×10×10 | 5,000 | 1.36 → 0.73 | 3.5 → 5.1 |
| 16×16×16 | 20,480 | 18.3 → 0.96 | 24.8 → 8.2 |
| 20×20×20 | 40,000 | 69.6 → 1.64 | 80.8 → 13.3 |
| 24×24×24 | 69,120 | 206 → 2.26 | 228 → 26.6 |
| 30×30×30 | 135,000 | 821 → 3.11 | 856 → 37.9 |
| 35×35×35 | 214,375 | 2,070 → 4.22 | 2,131 → 72.9 |

The tile screen grows with the square of the atom count, as on the Ada (201 ms at 69,120 atoms there). At 5,000
atoms the binned rebuild is slower, which is why the switch starts at `BINNED_MIN_ATOMS` = 10,000.

## The A/B agrees with the rebuild timing

Every side rebuilt 52 times in 768 timed steps at 16×16×16 and 55 times at 20×20×20 (one rebuild per 14–15 steps).

| Cell | Rebuild saving × rebuilds per step | Predicted | Measured (clean runs) |
| --- | --- | ---: | ---: |
| 16×16×16 | 16.6 ms × 52 / 768 | 1.1 ms/step (1.7 %) | 1.2–1.5 ms/step (1.9–2.3 %) |
| 20×20×20 | 67.5 ms × 55 / 768 | 4.8 ms/step (3.6 %) | 5.1 ms/step (3.9 %) |

With the binned screen the step costs 3.17 µs/atom at 20,480 atoms and 3.20 at 40,000; with the tile screen 3.23
and 3.32.

**Larger cells (my estimate, not measured):** at the same rebuild rate the saving is about 14 ms per step at
69,120 atoms, 59 ms at 135,000 and 147 ms at 214,375 (the card's SrTiO3 limit). Against today's per-atom step time
that is about 6 % at 69,120 atoms and 10–14 % from 135,000 atoms up.

## Not done

- **No default was changed.** Adding `"NVIDIA GeForce RTX 5060 Ti"` to `pair_search.BINNED_SCREEN_GPUS` is the
  conductor's merge.
- **The RTX 4000 Ada** still needs its own A/B (workstation idle check first). Its rebuild timing is on file
  (`results/ideas-queue2-20261106/README.md`).
- **Cells above 40,000 atoms** were not run as A/Bs: two engines in one process would pass 12 GiB on this card.

## Notes on the jobs

- **Memory:** the card peaked at 7.4 GiB (16×16×16, two engines) and 10.7 GiB (20×20×20), with 2.7 GiB held by
  Windows. No watchdog fired.
- **First job:** its guard skipped 20×20×20. `watch()` overwrote the global holding the card's idle memory, so the
  prediction used a base of 0 MiB (14,517 MiB instead of 11,897). The second job fixes this (`jobs/job2.sh`).
