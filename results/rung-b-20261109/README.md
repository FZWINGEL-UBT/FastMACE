# Rung (B) and a recompute step that saves memory (memory profile item 2, 9–10 November 2026)

The conductor's item 2 after the memory profile ([memory-profile-20261109](../memory-profile-20261109/README.md),
[memory-slack-20261109](../memory-slack-20261109/README.md)): first make rule 12's estimate for rescue rung (B)
honest, then give (B)'s path a block-layout gradient, or a recompute step that really saves memory. The default plan
below the limit must stay the same (checked with the plan text), and then the limits are measured again.

Source: branch `fastmace/rung-b-blocks` from `a7a8211e`. The commits: `58ad5c99` (rung B never beside w′; the layer region
inside the first-table rung), `cd56800c` (the layer region as its own rung; the probe change), `18fa5e7f` and `34db2467` (tests, golden plan texts). Full CPU suite on leto (`eb3ccb37`): 5,548 passed, 7 golden plan texts failed only on the "+N more" off count, rewritten in `34db2467` (30 passed); ruff findings only in pre-existing files.
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
- **Default plan below the limit unchanged:** at SrTiO3 32³ the plan text on `cd56800c` matches `a7a8211e` line for
  line (17 rewrites on, `recompute_first_table` the only rung). The probe measured `generic_recompute_first_layer` and
  then the first table's stage, and chose the latter. 915 ms/step, 11.42 GiB.
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

On `cd56800c`, SrTiO3 36³ with the default plan takes both rungs (1,514.6 ms, 14.97 GiB), although the plain plan
fits (16.85 GiB). The probe corrects rule 12's estimate to 22.65 GiB, against a 17.44 GiB limit. The probe's replica
(16,875 atoms) overstates the scale at this size (0.70 measured, 0.46 actual). So between about 34³ and the limit,
the default pays about 7 % for memory it would not need. `a7a8211e` also took a rescue rung there (the first table's).
This is a follow-up for the probe, not for this rung.

## Capacity (`capacity_scan.py point`, 20 warm-up and 200 timed steps, alone on the GPU)

| GPU | Crystal | Before (`a7a8211e`) | Now (`cd56800c`) | First failing size |
|---|---|---:|---:|---|
| RTX 4000 Ada (leto) | SrTiO3 | 274,360 (38³) | **296,595** (39³, 2,042.8 ms, 18.86 GiB) | 320,000 (40³): build, out of memory |
| RTX 4000 Ada (leto) | NaCl | 343,000 (35³) | **405,224** (37³, 2,060.9 ms, 18.99 GiB) | 438,976 (38³): build, out of memory (3.35 GiB request) |
| RTX 5060 Ti (local) | SrTiO3 | 163,840 (32³) | **214,375** (35³, 1,525.9 ms, 14.23 GiB) | 233,280 (36³): capture, out of memory (WSL "Failed to create GPU mapping") |
| RTX 5060 Ti (local) | NaCl | 195,112 (29³) | **287,496** (33³, 1,443.4 ms, 14.50 GiB) | 314,432 (34³): capture, out of memory |

Most of the 5060 Ti gain comes from the earlier fixes on this line (`SORT_KEYS_ONLY`, the shape carrier) together
with these rungs. No separate measurement was made for each.

## Files

- `jobs/`: the leto and WSL job scripts (`chain.sh`, `chain2.sh`, `chain3.sh`, `points*.sh`, `abregion.sh`,
  `tests.sh`, `launch_local3.sh`, `local35.sh`). The point JSONs and the profiles stay on leto under
  `/mnt/homes/zwingel/fm-memprofile/runs/rungb{1,2,3}`, and in WSL under `~/fm-ada-capacity/rungb/`.
