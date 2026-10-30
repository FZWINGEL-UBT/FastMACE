# Setup fixes F1, F2, F3 and F5

Fixes from [setup-across-compositions.md](setup-across-compositions.md), "Fixes, ranked by expected seconds saved".
F4 (the self-check cache) waits for the owner and is not part of this branch.

## Verdict

- **A later composition in the same process now builds in 10.6–11.1 s, down from 15.6–16.2 s** (warm cache).
  That is what the setup report predicted for F2 and F3 (about 10.5 s).
- **A new composition on a cold cache, after the first in the process, now builds in 11–15 s when it shares its
  species count, down from 23–26 s.** The report predicted about 21 s. NbCl5 after NaCl, a different species
  count, takes 26.2 s, down from 34.9 s.
- **A new atom count in a fresh process now hits the FX cache:** NbCl5 3×3×4 on NbCl5 3×3×3's cache takes 18.7 s,
  down from 30.4 s.
- **A fresh process on a warm cache, or a cold cache, is unchanged** (17.6–17.9 s and 40–43 s). F2 saves about 3 s
  there, and F5 costs 2.0 s (see "F5's first-build cost"). I kept F5 on: it saves 8.6 s whenever the atom count is
  new.
- **Step time: unchanged.** b/a 0.995–1.005 in both build orders, at 1,944 and 9,216 atoms.
- **Forces, energy and stress pass the gate** on NaCl, NbCl5 and NaNbO3.

## The fixes

| Fix | Commit | What changed |
| --- | --- | --- |
| F3 | `a495f5c5` | The converted-model templates belong to the `LoadedModel` (`LoadedModel.template_cache()`), so every `ForceField` of one `fm.load` result shares them. Each plan still gets its private copy, and the mutation guard still applies. |
| F1 | `953b649d` | The OEq kernel registry ids come from each kernel's source, rule and device (`content_id`), not from a counter of objects; Inductor saw them as graph constants. The radial-table piece count is padded to a bucket of 16 pieces (`fastmace.rewrites.radial.PIECE_BUCKET`) by repeating the last piece; the padded pieces are never selected, and values and gradients are bitwise the same. |
| F2 | `c20b4881` | A compiled OEq `ScaleShiftMACE` is called with forces and stress off, and MACE's own `get_outputs` takes the gradient after the call. The compiled frame holds no `torch.autograd.grad`, so there is no graph break and no Dynamo restart. |
| F5 | `0d62d0c7` | The edge tensors' capacity buffers (their `_base`) are marked dynamic with the views, so their storage size no longer enters the FX cache key. The per-atom inputs are marked dynamic on the atom dimension from the first build. |

The measurements below ran at `0d62d0c7`, which had two module-level switches for the ablation and the A/B:
`fastmace.model.forces.SPLIT_FORCE_FRAME` and `fastmace.model.compile.MARK_ATOMS_DYNAMIC`. The layering test forbids
boolean module switches, so a later commit removed both. F2 and F5 are now unconditional, and the code paths are
otherwise unchanged. `PIECE_BUCKET` is an integer constant and stays.

Nothing here is a per-structure or shape-keyed table: the piece bucket is a fixed rounding, and the ids are content
hashes.

## Setup time

`tools/bench/setup_compositions.py`, MH-1 hybrid float32, 3×3×3 cells, seconds from `Simulation` to the first
step. "Before" is `c20ddf31`, "after" is `0d62d0c7`.

- **Cold:** a fresh process with an empty cache.
- **Fresh-warm:** a fresh process on its own cold run's cache.
- **Same-process:** NaCl, NbCl5, TaCl5 and NbCl5 at 0.95 strain in one process, on a fresh cache ("cold chain") or on
  that cache again ("warm chain").

| Case | Before | After |
| --- | ---: | ---: |
| Cold, NaCl (54 atoms) | 42.8 | 39.8 |
| Cold, NbCl5 (972 atoms) | 40.7 | 40.5 |
| Fresh-warm, NaCl | 17.8 | 17.9 |
| Fresh-warm, NbCl5 | 17.6 | 17.7 |
| Fresh process, NbCl5 3×3×4 (1,296 atoms) on NbCl5 3×3×3's cache | 30.4 | **18.7** |
| Cold chain: NaCl / NbCl5 / TaCl5 / NbCl5 0.95 | 41.1 / 34.9 / 25.7 / 23.2 | 39.5 / **26.2 / 14.7 / 11.2** |
| Warm chain: NaCl / NbCl5 / TaCl5 / NbCl5 0.95 | 18.0 / 16.0 / 15.8 / 16.2 | 17.6 / **10.6 / 11.1 / 11.0** |

Every number is one run.

### Each fix on its own

The same chains on the after-source with one switch off (`with_constants.py` sets the module constant before the tool
runs; each configuration has its own cache). F3 has no switch, so its share comes from its stages. F1's kernel ids
have no switch either; probe v1 below shows their miss.

| Chain | After | F2 off | F1 radial padding off | F5 off |
| --- | --- | --- | --- | --- |
| Cold: NaCl / NbCl5 / TaCl5 / NbCl5 0.95 | 39.5 / 26.2 / 14.7 / 11.2 | 42.0 / 28.6 / 16.9 / 13.6 | 40.0 / 27.8 / **20.9** / 11.2 | 38.1 / 27.6 / 14.1 / 11.0 |
| Warm: NaCl / NbCl5 / TaCl5 / NbCl5 0.95 | 17.6 / 10.6 / 11.1 / 11.0 | **21.1 / 13.4 / 13.3 / 13.4** | 17.7 / 10.6 / 10.9 / 11.0 | **15.7** / 10.7 / 11.0 / 10.7 |
| Fresh process, NbCl5 3×3×4 on 3×3×3's cache | 18.7 | – | – | **27.3** |

- **F3: about 3.1 s per composition after the first.** The execution-model stage goes from 3.1–3.6 s to 0.5 s (the
  private copy), and the checkpoint load and conversion (2.6–3.0 s) are gone.
- **F2: 2.2–3.5 s per build, every build.** It is above its stop rule (2 s).
  - The probe's restart attempt (6.9 s) became 2.1 + 5.0 s of tracing, so most of the saving is the restart's second
    trace.
  - Forces and stress: bitwise MACE's own on the CPU test model (`tests/cpu/test_split_force_frame.py`), and the GPU
    gate passes.
- **F1, radial padding: 6.2 s for TaCl5 after NbCl5 on a cold cache** (20.9 against 14.7 s). TaCl5 has 53 pieces
  where NbCl5 has 52. With the bucket, both have 64 and share Inductor's code. On a warm cache it changes nothing.
- **F5: 8.6 s for a new atom count across processes** (27.3 against 18.7 s), and it costs about 2 s on a process's
  first build (below). In the same process, Dynamo's automatic dynamic shapes already covered a new atom count, so the
  chains do not show the saving.
- **F1's and F5's edge part** (the kernel ids and the capacity buffers) have no switch. Probe v1 shows the misses they
  remove.

### F5's first-build cost

Fresh processes on warm caches, alternating F5 on and off, three runs each, on the after-source:

| Structure | F5 on | F5 off | Of which first compiled force call |
| --- | --- | --- | --- |
| NaCl | 18.5 / 17.9 / 18.2 | 16.4 / 16.3 / 15.9 | 10.2–10.6 against 8.2–8.6 |
| NbCl5 | 18.3 / 18.0 / 17.6 | (26.8) / 16.0 / 16.1 | 9.9–10.1 against 8.0–8.2 |

- **F5 costs 2.0 s on a process's first build,** all of it in the first compiled force call. The likely cause is
  that Dynamo traces with a symbolic atom count, which it does not cache.
- **The 26.8 s run with F5 off shows what F5 removes.** That cache had compiled NbCl5 only after NaCl in one process,
  with a dynamic atom count, so the first fresh process with a static 972 missed it.
- **Net:** F5 trades +2.0 s on every fresh process for −8.6 s whenever the atom count is new to the cache. In a fresh
  process F2's saving covers it, which is why the fresh-warm rows above are unchanged.
- **Recommendation: keep F5 on.** Runs of one structure at another supercell or strain are the case this work was
  for, and a miss costs four times what the marking costs. If a later change makes the symbolic trace cheaper, the
  2 s goes away.

## Cache probes

`TORCH_LOGS="recompiles,+torch._inductor.codecache"` on one process that builds NbCl5, NbCl5 again, NbCl5 at 0.95,
TaCl5 and NbCl5 2×2×3, with the FX cache's hash details.

**Probe v1** (F3 only, `a495f5c5`). The FX cache missed on:

- every new composition, because of the kernel registry ids (graph constants that counted objects);
- TaCl5, because of its piece count (53 against 52 pieces, a static shape);
- NbCl5 at 0.95, because of the storage size of the edge views' capacity buffers;
- a new atom count, because the atom count was static in a process's first compile.

**Probe v2** (all four fixes, `0d62d0c7`):

| Build | Graph breaks | Restart | FX cache | Setup now | Probe v1 |
| --- | --- | --- | --- | ---: | ---: |
| NbCl5 again | 0 | none | hits | 18.0 s | 18.3 s |
| NbCl5 at 0.95 | 0 | none | hits | 17.8 s | 21.5 s |
| TaCl5 | 0 | none | hits | 21.3 s | 27.6 s |
| NbCl5 2×2×3 (432 atoms) | 0 | none | misses | | |

The 2×2×3 cell misses on purpose. At 432 atoms a size-regime rule chooses another program (`[s, 5, 512]` against
`[5·s, 512]`), so it is different code, not a cache-key artefact. That fits F1's stop rule, which covers compositions
with the same species count and different atom counts in one regime: NbCl5 3×3×4 hits.

## Step time

`tools/ab_rewrites.py --prepare`, same process, MH-1 hybrid float32 NVT at 300 K.

- **Side a:** the three switches off while it is built and its first step's stages are captured:
  - `fastmace.model.forces:SPLIT_FORCE_FRAME=False`;
  - `fastmace.rewrites.radial:PIECE_BUCKET=1`;
  - `fastmace.model.compile:MARK_ATOMS_DYNAMIC=False`.
- **Side b:** the defaults.

F1's kernel ids, F3 and F5's capacity-buffer marking cannot be switched off. They change neither the program nor
its kernels. Median step in ms, ratios b/a.

| Cell | Blocks | Order ab: a / b, b/a | Order ba: a / b, b/a |
| --- | --- | --- | --- |
| NbCl5 3×3×3, 1,944 atoms | 12 × 300 steps | 17.61 / 17.70, **1.005** | 17.75 / 17.66, **0.995** |
| NbCl5 4×4×8, 9,216 atoms | 8 × 100 steps | 88.21 / 88.33, **1.001** | 88.35 / 88.70, **1.004** |

Every run was valid: nothing compiled while timed, and no other process used the GPU.

## Gate

`tools/force_stress_gate.py`, default plan, MH-1 hybrid float32 against MACE's own float64 model; production error
divided by ASE float32's error:

| Structure | Energy | Forces | Stress | Passed |
| --- | ---: | ---: | ---: | --- |
| NaCl | 0.16 | 0.96 | 0.39 | yes |
| NbCl5 | 0.08 | 1.05 | 0.37 | yes |
| NaNbO3 | 0.50 | 1.02 | 0.66 | yes |

## For the conductor

- **SPEC §7.1** still says the `ForceField` keeps the templates. After F3 the `LoadedModel` keeps them, and every
  `ForceField` of one `fm.load` result shares them. The section needs an amendment and a §19 entry.
- `docs/design/opt-T06-template-cache.md` has a "Superseded by F3" note, and `docs/reference/engine.md` describes the
  shared cache.
- **Leto slip, reported earlier:** one targeted CPU run on leto started while the busy check printed 1. Since then
  every launch waits for 0.

## Checks

- CPU on leto (GPU hidden, NAS basetemp), targeted files:
  - at `c20b4881`: 365 passed, 29 skipped;
  - at `0d62d0c7`: 515 passed, 29 skipped.
- New tests:
  - `tests/cpu/test_template_cache.py`: the shared templates;
  - `tests/cpu/test_radial_tables.py::test_padded_pieces_change_no_bit`;
  - `tests/cpu/test_kernels_oeq_streaming.py::test_kernel_ids_follow_the_content`;
  - `tests/cpu/test_split_force_frame.py`.
- Ruff is clean on the changed files.

## Setup

- **Machine:** Fabi-PC, WSL2, RTX 5060 Ti 16 GB (driver 610.88), `mace_cueq_oeq` (torch 2.14.1+cu130).
- **GPU access:** `tools/ci/local_gpu.py run` reservations. The A/B and gate jobs ran the idle check before each job.
- **Sources:** pinned copies of FastMACE `c20ddf31` (before) and `0d62d0c7` (after). There is no baseline, since
  both sides are FastMACE.
- **Model:** MH-1 (`mh1-r2scan`), hybrid backend, float32.
- **Records:**
  - `results/setup-fixes-20261030/`: the run script `local.sh`, the measurement scripts, the setup summaries, the
    A/B records and the gate summary;
  - the full `setup.json` files stay in `~/fastmace-capacity/setupfix/` (WSL).
