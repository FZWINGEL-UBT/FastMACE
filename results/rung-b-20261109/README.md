# Rung (B) and a recompute step that saves memory (memory profile item 2, 9–10 November 2026)

The conductor's item 2 after the memory profile ([memory-profile-20261109](../memory-profile-20261109/README.md),
[memory-slack-20261109](../memory-slack-20261109/README.md)): first make rule 12's estimate for rescue rung (B)
honest, then give (B)'s path a block-layout gradient, or a recompute step that really saves memory. The default plan
below the limit must stay the same (checked with the plan text), and then the limits are measured again.

Source: branch `fastmace/rung-b-blocks` from `a7a8211e`. The commits:
- `58ad5c99`: rung B never beside w′; the layer region inside the first-table rung.
- `cd56800c`: the layer region as its own rung; the probe measures the next cheaper stage.
- `18fa5e7f`, `34db2467`: tests, golden plan texts.
- `997e344b`: main `5249d8d8` merged in.
- `1522aef2`: the probe fits an offset from two replica sizes (after the conductor's review).

The full CPU suite result is under "Checks" below.
GPUs: leto's NVIDIA RTX 4000 Ada Generation (20 GB, NAS `mace_cueq_oeq`, idle check before every job) and the local
RTX 5060 Ti (WSL2, inside a `tools/ci/local_gpu.py` reservation). OMAT-0 medium, float32, hybrid backend.

## Summary

- **Rung (B) (`generic_recompute_conv_weights`) is never taken beside w′ (`generic_wprime_backward`).** With w′ on,
  (B) stands w′ aside. Inductor then builds the dense message gradient `[N, 5120]` beside the four 3×TF32 GEMM blocks
  (20.5 kB per atom more), and (B) saves nothing per edge. The memory fallback took (B) after an out-of-memory error
  at SrTiO3 39³, which made the shortage worse. Changes:
  - `planner.memory.RESCUE_NOT_BESIDE` names that pair. Rule 12 (`minimal.eligible`) and the rebuild
    (`fallback.memory_rescue`) skip (B) while w′ is on.
  - (B)'s memory term beside w′ (`generic_recompute_conv_weights_beside_wprime`) is now honest: +the message
    gradient per node, 0 per edge.
- **New rung `generic_recompute_first_layer` (`RewriteConfig.recompute_first_layer`), after the first table's rung.**
  The first interaction, its division and product 0 run under one non-reentrant checkpoint. The backward keeps the
  layer's inputs instead of product 0's input: the divided message `[N, 2048]`, 8.2 kB per atom, which is live at the
  force backward's peak. It rebuilds the layer after the second interaction's backward.
  - Rule 12 term: −`target` floats per node.
  - Float32 only; rounding (the sender gradients' atomic order); not deterministic.
  - An explicit "on" needs `recompute_first_table` on a generic model.
- **The memory probe also measures the next cheaper stage** (`memory_probe.cheaper_after`), when the chosen stage's
  measured scale says that stage might fit. Rule 12 overstates the plain plan about 2× (SrTiO3 32³: 24.4 GiB
  estimated, 12.0 GiB measured). Without this, the new rung would be taken at 32³, far below the limit.
- **The probe fits an offset from two replica sizes** (`memory_probe.fitted_offset`, `fitted_scale`;
  `REPLICA_LARGE_ATOMS` = 45,000–60,000). One replica's measured-over-estimated ratio includes a reserve that does
  not grow with the cell, so carried to the whole cell it overstated large cells. On SrTiO3 the ratio is 0.70 at
  16,875 atoms and 0.52 at 233,280. On `cd56800c` the default therefore took the layer rung at 34³–36³, about 7 %
  slower than `a7a8211e`. Now the first stage is also measured on a 53,240-atom replica. The line through the two
  peaks gives an offset (0.45 GB on SrTiO3) and a slope (0.50). Each stage's scale is then the one that puts the
  run's own cell on that line. The extra child costs about 50 s, only when the probe runs at all, that is, near the
  limit.
- **Default plan unchanged below the old limit:** see "Default plan against `a7a8211e`" below. At SrTiO3 32³ the
  plan text matches `a7a8211e` line for line, and at 33³–36³ both trees take the same rung (the first table's) at
  the same step time.
- **Speed cost of the new rung:** b/a **1.0923 / 1.0917** (ab / ba). The A/B ran on SrTiO3 10×10×10 on the Ada, with
  both sides on the first table's rung and the region off (a) or on (b).
- **Limits:** see the table below. On the Ada, SrTiO3 rises from 274,360 to 296,595 atoms and NaCl from 343,000 to 405,224. The 5060 Ti limits rise
  from 163,840 to 214,375 atoms (SrTiO3) and from 195,112 to 287,496 atoms (NaCl).

## Memory at SrTiO3 36³ (Ada, MD profile, `58ad5c99`)

| Case | Peak reserved | kB/atom | Graph pool live peak |
|---|---:|---:|---:|
| First table's rung only (region off) | 15.82 GiB | 72.8 | 9.99 GiB |
| **With the layer region** | **14.81 GiB** | **68.2** | **8.45 GiB** |

At the graph pool's peak (36³), what is live: the GEMM gradient blocks (20.5 kB/atom), the first interaction's divided
message (8.2, now recomputed), `edge_sh3` (4.7) and the w′ blocks (6.8). The target in the brief was 20 kB per atom
less. This step gives 4.6 kB per atom less reserved (8.4 kB per atom at 32³). The rest is in the GEMM gradient blocks,
which a recompute step cannot drop: they are the gradient itself.

## Forced plans (Ada, `58ad5c99`, `capacity_scan.py point`, 100 steps)

| SrTiO3 | Plain (no rung) | First table + region | Cost |
|---|---:|---:|---:|
| 32³ (163,840) | 909.7 ms, 12.04 GiB | 982.7 ms, 10.72 GiB | +8.0 % |
| 36³ (233,280) | 1,410.9 ms, 16.85 GiB | 1,516.2 ms, 14.96 GiB | +7.5 % |

On `cd56800c`, SrTiO3 36³ with the default plan took both rungs (1,514.6 ms, 14.97 GiB). The one-size probe
corrected the first table's stage to 23.5 GiB, against a 17.44 GiB limit; that stage measures 15.96 GiB. `1522aef2`
fixes this (next section). The plain plan (16.85 GiB) does not fit under decision 51's limit (90 % of the free
memory) with the probe's 5 % headroom even when measured exactly. So the first table's rung is the right stage at
36³, as in `a7a8211e`.

## Default plan against `a7a8211e` (Ada, `capacity_scan.py point`, 100 steps, the default plan)

Chain 4 ran the new tree first, then `a7a8211e`. Chain 5 ran in the reverse order: `a7a8211e` first, then the new
tree.

| SrTiO3 | Rung taken (both trees) | Chain 4: new / old ms | new/old | Chain 5: old / new ms | new/old | Peak (both) |
|---|---|---:|---:|---:|---:|---:|
| 34³ (196,520) | first table | 1,161.11 / 1,160.92 | 1.0002 | – | – | 13.53 / 13.54 GiB |
| 35³ (214,375) | first table | 1,296.94 / 1,296.58 | 1.0003 | 1,307.86 / 1,298.20 | 0.9926 | 14.72 / 14.73 GiB |
| 36³ (233,280) | first table | 1,415.63 / 1,415.43 | 1.0001 | 1,413.94 / 1,413.86 | 0.9999 | 15.96 / 15.96 GiB |
| 37³ (253,265) | new: first layer; old: first table | 1,704.24 (new) | – | 1,594.08 (old) | 1.069 | 16.21 / 17.31 GiB |

These are separate processes, so they are not an in-process A/B. Where both trees take the same stage, the code path
is the same, and so is the plan text (32³). At 33³ the new tree also takes the first table's rung (1,043.35 ms,
12.42 GiB).

For each new-tree stage, the corrected estimate (with the 5 % headroom) and the measured peak:

| SrTiO3 | First-table stage corrected | Layer stage corrected | Measured (chosen stage) |
|---|---:|---:|---:|
| 33³ | 12.59 GiB | 11.94 GiB | 12.42 GiB |
| 34³ | 13.71 GiB | 12.99 GiB | 13.53 GiB |
| 35³ | 14.89 GiB | 14.11 GiB | 14.72 GiB |
| 36³ | 16.15 GiB | 15.30 GiB | 15.96 GiB |
| 37³ | 17.47 GiB (above 17.44) | 16.55 GiB | 16.21 GiB (layer rung) |

The corrected estimates lie 1.2 % above the measured peaks. Without the headroom they would be about 3.5 % below:
the peak grows a little faster than linearly with the cell. Decision 51's 10 % margin to the free memory covers this.
At 37³, `a7a8211e` takes the first table's rung and measures 17.31 GiB. That is just under decision 51's 17.44 GiB
limit; the old probe had corrected the stage to 23.78 GiB and kept it only because it had no deeper rung. The new
tree's fit corrects the table stage to 17.47 GiB, 0.2 % over the limit, so it takes the layer rung there: 6.9 %
slower (1,704 against 1,594 ms) and 1.1 GiB less. The switch to the layer rung therefore comes one size earlier
than an exact estimate would put it. Above that, from 38³ up, the old tree ran past the limit (18.68 GiB at 38³).

## Capacity (`capacity_scan.py point`, 20 warm-up and 200 timed steps, alone on the GPU)

| GPU | Crystal | Before (`a7a8211e`) | Now (`cd56800c`; checked again on `1522aef2`) | First failing size |
|---|---|---:|---:|---|
| RTX 4000 Ada (leto) | SrTiO3 | 274,360 (38³) | **296,595** (39³, 2,042.8 ms, 18.86 GiB) | 320,000 (40³): build, out of memory |
| RTX 4000 Ada (leto) | NaCl | 343,000 (35³) | **405,224** (37³, 2,060.9 ms, 18.99 GiB) | 438,976 (38³): build, out of memory (3.35 GiB request) |
| RTX 5060 Ti (local) | SrTiO3 | 163,840 (32³) | **214,375** (35³, 1,525.9 ms, 14.23 GiB) | 233,280 (36³): capture, out of memory (WSL "Failed to create GPU mapping") |
| RTX 5060 Ti (local) | NaCl | 195,112 (29³) | **287,496** (33³, 1,443.4 ms, 14.50 GiB) | 314,432 (34³): capture, out of memory |

On `1522aef2` (the two-size probe), all four limits hold, each with the layer rung: Ada SrTiO3 39³ 2,042.7 ms,
18.86 GiB; Ada NaCl 37³ 2,063.0 ms, 18.99 GiB; 5060 Ti SrTiO3 35³ 1,534.3 ms, 14.23 GiB; 5060 Ti NaCl 33³
1,444.4 ms, 14.50 GiB. The next sizes on the 5060 Ti (SrTiO3 36³, NaCl 34³) again run out of memory in the build.

Most of the 5060 Ti gain comes from the earlier fixes on this line (`SORT_KEYS_ONLY`, the shape carrier) together
with these rungs. No separate measurement was made for each.

## Next limiter: the backward GEMM gradients (bound only, 10 November 2026)

The conductor's follow-up was to stream or chunk the backward's 3×TF32 GEMM gradients, the four blocks of linear_1's
input gradient (20.5 kB/atom), after a bound. `tools/bench/pool_bound.py` replays the graph pool's allocation trace
from the SrTiO3 36³ snapshot (r3 of the memory profile) and reports the live peak with selected allocations scaled
down:

| Graph pool, SrTiO3 36³ | Live peak | kB/atom |
|---|---:|---:|
| As profiled (no layer region) | 9.99 GiB | 46.0 |
| The layer region's saved `[N, 2048]` dropped (the plan at the limit) | 8.45 GiB | 38.9 |
| … and the GEMM blocks at 1/8 | 8.45 GiB | 38.9 |
| … and the outer-sum gradient `[3N, 1024]` at 1/8 | 8.17 GiB | 37.6 |
| … and the species GEMM outputs of the product backward at 1/8 | 7.43 GiB | 34.2 |

With the layer rung on, the peak is in the product backward: the outer-sum gradient (2.67 GiB) and four species GEMM
outputs (2.67 GiB together). The GEMM blocks are no longer live there, so chunking them alone gains nothing at the
limit. Chunking all three backward sites gains 1.0 GiB live (4.4 kB/atom). REJECTED at the bound.

What is left at the limit is spread out. The neighbour list holds 1.6–2.0 GiB, of which `edge_index` in int64 is
0.37 GiB (0.8 kB/atom to gain with int32). The rest are single tensors of a few kB/atom. The one large item is
placement loss: the graph pool reserves 12.98 GiB for a 9.99 GiB live peak (13.8 kB/atom, as the memory-slack study
found). The build's eager pass loses 3.41 GiB more in the default pool, but releases it before the capture.

**Bound for a planned capture arena** (`pool_bound.py`, offline, the same trace). The captured pool's 195
allocations, with their lifetimes known, are packed at the lowest offset free over each lifetime. The best of three
orders is shown:

| Graph pool, SrTiO3 36³ | Live peak | Caching allocator | Planned arena | Gain |
|---|---:|---:|---:|---:|
| As profiled | 9.99 GiB | 12.98 GiB | 10.44 GiB (by lifetime) | 2.54 GiB, 10.9 kB/atom |
| The layer region's output dropped | 8.45 GiB | (about 11.5, not traced) | 9.15 GiB (by size × lifetime) | about 2.4 GiB |

An arena that serves the capture's requests at planned offsets would recover about 80 % of the placement loss. That is
about 11 kB of the 68 kB per atom reserved at the limit, roughly +15–20 % atoms. It needs a deterministic request
sequence between a recording capture and the real one, and a pluggable allocator or memory pool that serves offsets
in one buffer. Stage 0 (`results/capture-arena-20261110`) found that a `MemPool` shim can place only whole
segments (42–45 % of the loss); stopped there.

**Cheap candidate: `edge_index` in int32.** The skin list keeps `edge_index` as int64 `[2, capacity]` (0.37 GiB at
SrTiO3 36³), while its candidates are already int32. Storing it as int32 saves 0.8 kB/atom. Each consumer that
indexes with it (`index_add`, gathers, the custom kernels' sender and receiver arguments) must accept int32 or cast
in place. A work-removal switch: it needs its own not-slower A/B in both orders (decision 66). Not started.

## Checks

- **Force/stress gate** (`tools/force_stress_gate.py`, 3× float32 limit, decision 66). It ran on the Ada on
  `1522aef2`, with the default plan and `recompute_first_table`, `recompute_first_layer` forced on (the plan text
  confirms the rung is on). Ratios to the float32 yardstick:
  - SrTiO3 3×3×3: energy 1.3e-3, forces 0.81, stress 0.42. Pass.
  - NaCl: energy 1.2e-4, forces 0.93, stress 0.96. Pass.
- **Targeted CPU tests** on `1522aef2` (leto): 657 passed, 11 skipped.
- **Full CPU suite** on `9b20ddd4` (main `5249d8d8` merged in; leto, `-n 8`, `not acceptance`, idle check before):
  5,573 passed, 203 skipped, 0 failed. Ruff's findings are all in lines that predate this branch (no line it adds is
  over 120 characters).

## Files

- `jobs/`: the leto and WSL job scripts (`chain*.sh`, `points*.sh`, `abregion.sh`, `gate3.sh`, `golden.sh`,
  `tests.sh`, `launch_local3.sh`, `launch_local4.sh`, `local35.sh`). The point JSONs, gate output and profiles
  stay on leto under `/mnt/homes/zwingel/fm-memprofile/runs/rungb{1..6}`, and in WSL under
  `~/fm-ada-capacity/rungb/`.
