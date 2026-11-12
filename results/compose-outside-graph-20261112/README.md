# compose-outside-graph: COMPOSE_FAST's weights composed once, outside the captured MD step (12 Nov 2026)

Brief (conductor, 12 November 2026): remove the per-step constant work that the torch 2.15 attribution found in the
captured step ([torch-215-trial §6a](../torch-215-trial-20261112/README.md), branch `fastmace/torch-215-trial`
`f4103042`): COMPOSE_SKIP's `refresh_composed` → `_compose_weights` ran inside the CUDA graph and replayed four float64
CUTLASS d884 GEMMs every step (Ada: 301 µs, 1.8 %, SrTiO3 10³; 331 µs with copies, 9.8 %, NbCl5). Branch
`fastmace/compose-outside-graph`, based on main `236fa60c`.

| | SrTiO3 10³ (5,000 atoms) | NbCl5 3³ (1,944 atoms) |
|---|---|---|
| Ada MD step b/a, ab / ba | **0.9873 / 0.9823** (17.11 → 16.89 ms) | **0.9273 / 0.9341** (3.774 → 3.500 ms) |
| Ada compose kernels per step (2.15 attribution) | 301 µs → **0** | 278 µs CUTLASS + copies → **0** |
| A100 MD step b/a | pending (Alex job via tiled-speed, awaiting the owner's OK) | pending |
| Exactness | composed weights bitwise equal; same arithmetic, run once instead of per replay | same |

## Why the composition was captured

`ForceField._bind_streaming_count` binds each reassociated interaction's species table before every force call
(`forces.py:1036` on `236fa60c`), and `FirstInteractionReassociation.bind` recomposes when the table's
`_version` differs from its cached entry. The graph engine's `_capture` takes a registry snapshot, runs the warm-ups
and **restores the snapshot** (`registry.restore`: `copy_` of every captured buffer, the table included) just before
`torch.cuda.graph`. The restore writes the same values but moves the table's version, so the bind inside the capture
saw a "changed" table and recomposed in place: the four float64 GEMMs (and on NbCl5 the copies of `composed` and the
in-place refresh) became graph nodes and were replayed every step, for inputs that are constant in MD.

## The change (`fastmace/rewrites/mh1/modules.py`, `fastmace/model/forces.py`)

1. **No composition inside a capture for a known table.** `bind` passes `adopt=torch.cuda.is_current_stream_capturing()`
   to `refresh_composed`; a cached entry whose parameters are unchanged then takes the table's new version without
   composing (proof counter `CAPTURE_ADOPTS()`). Outside a capture a moved version still recomposes in place (same
   values, same addresses). A new table, a nested-rotation change or a changed parameter still composes, also inside
   a capture (correct, just not free).
2. **The weights are derived state beside their table.** `ForceField._derived` registers each table's composed
   weights (`species_feats.interactions.<i>.composed_<k>`, depends on charges, refreshed by the table's own refresh,
   which recomposes in place). So every snapshot restore — a capture's warm-up and a rejected transaction — brings the
   table and its weights back together, and a replayed step reads consistent weights without composing. Before, the
   replayed composition was what kept them consistent after a rejected charge transaction.
3. **New invalidation: parameter versions.** The cache entry records the versions of every tensor the composition reads
   besides the table (the radial last layer, the interaction's linear and skip, the nested rotation). A reload or an
   optimiser step writes them in place and recomposes; before, a parameter change was never noticed outside a capture.

No new switch. Same arithmetic (`compose_skip.composed`, float64, cast once), same buffers: the change is bitwise.

### Invalidating events, checked

| Event | What recomposes |
|---|---|
| Capture / recapture (edge size, pool reset, friction event) | nothing: the restore brings the same table and weights back; the bind adopts |
| Charge / total-charge event (OMOL-0 joint embedding: the table depends on it) | the table's derived refresh, in place (eager in a transaction; inside an attempt graph it is captured, as it must be) |
| Rejected transaction | the registry restores table and weights together |
| Species event | nothing: the table is the identity basis of the superset (SPEC §8.5), species-independent |
| Model or weights reload in place, optimiser step / finetune on a routed model | parameter versions move → recompose on the next bind |
| A new model, dtype or device | a new model object and new tables → its own cache entries |
| nested_rotation set or cleared (`precision._route_nested`) | cache cleared, recomposed |
| Tiled mode (a table per slot, `species_tables`) | a new table → composed at its first bind, outside the slot capture; slot tables are never rewritten |
| Several force fields sharing a model, `follow` copies | one cache entry per table (`id(table)`); each force field registers its own |
| Inference tensors (no version counter) | version `None`, never written in place |

## Tests

- `tests/cpu/test_compose_outside_graph.py` (new): a bind inside a (simulated) capture keeps the cached weights and the
  forward stays bitwise the forward's own composition; outside a capture a new version recomposes in place; a changed
  radial, skip or linear weight recomposes also inside a capture; a new table composes; the engine `Registry`
  snapshot / refresh / restore round trip brings the weights back bitwise and the next captured bind adopts them.
- Targeted run on leeloo (`42b87382` + the counter fix, CUDA hidden): test_compose_outside_graph, test_compose_skip,
  test_composed_cache, test_layering, test_superset, test_events, test_registry, test_ab_rewrites_tool: 182 passed,
  1 skipped.
- Full `tests/cpu`: 5,738 passed, 206 skipped, 0 failed (see "Suite").

## GPU smoke (`gpu_smoke.py`, Ada, OMAT-0 medium float32, production plan, 300 steps)

Three engines in one process, built and stepped in turn: `a` with `main`'s bind (`tools/bench/reference_kernels.py`
`compose_in_graph_bind`), `b` and `b2` with the library's. Calls of `compose_skip.composed` inside / outside a capture:

| | a: in capture (steps) | b: in capture | b: eager (once per capture, after the restore) | CAPTURE_ADOPTS b | weights a = b |
|---|---:|---:|---:|---:|---|
| NbCl5 | 8 | **0** | 4 | 2 | bitwise |
| SrTiO3 4³ (320 atoms) | 4 | **0** | 4 | 1 | bitwise |

Forces are not bitwise between any two builds of this plan, the control included (scatter atomics): build forces
a vs b 9.5e-7 / 1.5e-6 eV/Å, b vs b2 7.2e-7 / 1.6e-6; after 300 steps a vs b 9.3e-5 / 3.2e-5, b vs b2 1.1e-4 /
2.8e-5. The composed weights — the only values the change touches — are bitwise equal, so the force path reads
identical inputs; CPU tests show the forward bitwise.

## Ada MD-step A/B (leeloo, RTX 4000 Ada, `jobs/ada.sh`, `ab-ab.json` / `ab-ba.json`)

`tools/bench/ab_swap.py --swap fastmace.rewrites.mh1.modules:FirstInteractionReassociation.bind=compose_in_graph_bind`
(side a composes inside the capture as `main`), OMAT-0 medium float32, the Ada production plan, 1 fs, skin 0.5,
warm-up 200, 6 × 256 steps, proof `CAPTURE_ADOPTS` (b: 1 / 2 per build, a: 0; swap trace: 0 calls on the wrong side).
Idle check before each job (no GPU process, no process above 50 % CPU, LC_ALL=C); the full CPU suite ran on leto,
not here.

| | order ab | order ba | pair ratios |
|---|---|---|---|
| SrTiO3 10³ | 17.111 → 16.893 ms, **0.9873** | 17.307 → 17.001 ms, **0.9823** | 0.9786–0.9884 |
| NbCl5 3³ | 3.774 → 3.500 ms, **0.9273** | 3.814 → 3.562 ms, **0.9341** | 0.9261–0.9362 |

−218 to −306 µs on SrTiO3 and −252 to −274 µs on NbCl5, against the attribution's bound of 301 / 331 µs.

## Ada re-profile (torch 2.15 attribution env, cupti-python, PYSTACKS=1; `attribution-*.json`)

Same tooling as torch-215-trial §6a (`/mnt/homes/zwingel/t215/jobs/profile_point.py`, `attribution.py`), 5 annotated
steps after 200 + 200. Before = §6a's records (`b7c19b19`); after = this branch.

| µs per step | SrTiO3 before → after | NbCl5 before → after |
|---|---|---|
| ours: CUTLASS (the d884 compose GEMMs; nothing else is CUTLASS on the Ada plan) | 301 → **0** | 278 → **0** |
| copy | 141 → 37 | 61 → 20 |
| sites in `compose_skip.py` / `_compose_weights` / `refresh_composed` | present → **none** | 33 mentions → **none** |
| graph events per step | 232 → 195 | 166 → 129 |
| GPU kernel time | 16,878 → 16,728 | 3,372 → 3,182 |
| point (steady, 2.15 env) | 17.10 → 17.01 ms | 3.81 → 3.54 ms |

The other classes moved by tens of µs between the two code bases (`b7c19b19` → `236fa60c` + this change).

## A100

Pending: `jobs/alex_compose.sh` (smoke + A/B both orders, SrTiO3 10³ + NbCl5, caches and code on node-local
$TMPDIR). The conductor's ssh is refused; the job was handed to the tiled-speed session, whose permission check asks
the owner before it stages and submits a job for another session. On the A100 float64 runs on tensor cores at full
rate, so the gain should be far smaller than on the Ada (where float64 is 1/64 rate).

## H200

Not run (the Modal account is at its cap; no `--owner-override`). The change is bitwise and removes work; the
conductor decides on the H200.

## Suite

Full `tests/cpu` on `381ac910` (main `236fa60c` + this branch's code; later commits add records and docs only), leto,
idle check (no GPU process), CUDA hidden, 12 Nov 17:52 CET: **5,738 passed, 206 skipped, 0 failed** (13 min 49 s).

## Files

- `gpu_smoke.py`: the smoke; `jobs/ada.sh` (smoke, A/B, profile), `jobs/alex_compose.sh` (A100).
- `ada-leeloo/`: run.log, smoke-*.json, ab-*.json (+ logs, swap traces), attribution-*.json, points/, profile/
  pystacks (Chrome traces stay on the NAS: /mnt/homes/zwingel/fm-compose/ada-2/profile).
