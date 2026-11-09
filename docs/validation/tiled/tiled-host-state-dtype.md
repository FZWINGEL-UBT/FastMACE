# Tiled capacity path: a 16-bit host state and the host budget (`TileConfig.state_dtype`)

**The question (conductor's capacity brief, 9 November 2026).** Under the wavefront, the host memory of a tiled call
is the row pool (x, sc and their aliased adjoints, 2,560 bytes per pooled atom in float32), the alias window and the
kept tile lists. Can a bfloat16 or float16 host state raise the atom ceiling without costing accuracy or speed? And
should the kept lists (73.1 GB of the 149M record's 179.8 GB host peak) stay when the host is short?

**Result.**
- Neither 16-bit format passes the 3× float32 force/stress gate, so neither is a default.
- `TileConfig.state_dtype="auto"` (the new default) is a host-budget decision:
  - it uses the call's dtype while the state fits;
  - otherwise it drops the kept lists first (exact, about 14 % slower calls);
  - it stores the state in float16 only if the state still does not fit.
- On a host with room, nothing changes.

## The change

`fastmace/model/tiled.py`:

- **`TileConfig.state_dtype`.** Accepts None, `"auto"`, `"float16"` or `"bfloat16"`. A 16-bit value needs
  `alias_state=True`; `"auto"` is ignored without the aliased host state.
- **`_State(store=…)`.** The host tensors use `store`:
  - each value is rounded once where it is written (`put`, and `_close_now` for the window's dE/dx);
  - each value is read back exactly into the call's dtype (`rows`, `fetch`);
  - the tiles compute in the call's dtype;
  - the alias window, which sums dE/dx over an atom's tiles, stays in the call's dtype.
- **`_auto_store`**, at the call's first allocation:
  - **The estimate:** full = rows · width · itemsize, plus a quarter of that for the window; half = rows · width · 2,
    plus the same window.
  - **The room:** `_host_available()` less `list_reuse_floor`.
  - **The choice:**
    - the call's dtype if full fits;
    - the call's dtype after `_drop_kept` if full fits once the kept lists' bytes are freed;
    - otherwise float16, and the kept lists go too if even half does not fit.
- **`_host_available()`.** Now the minimum of `MemAvailable` and the room the process's memory cgroups leave
  (`_cgroup_room`, v1 and v2, over the cgroup and its ancestors: limit less usage less inactive file cache).
  - Inside a Slurm job, `/proc/meminfo` shows the whole node (1.6 TB available on Alex node a0731), not the job's
    limit (231.5 GB for one A100-80, cgroup v2).
- **The call's report** has `state_dtype` and `state_dtype_reason`.
- **A fix on the way:** the wavefront with `neighbours="global"` failed on a missing `_list_cache` (initialised in
  `__init__` now, with a test).

## Accuracy

**3× float32 gate** (`tools/force_stress_gate.py --tiled-config`; host state aliased, wavefront, compiled, R = 48;
5060 Ti, job 25). Each row is max |production − reference| against max |ASE float32 − reference|, and their ratio:

| Store | SrTiO3 forces | NaCl forces | NbCl5 forces | Energy ratio | Stress ratio | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| float32 | 2.5e-5 / 3.0e-5 (0.8) | 3.3e-6 / 4.7e-6 (0.7) | 7.8e-5 / 8.2e-5 (1.0) | 1.0–1.5 | 0.5–0.7 | pass |
| float16 | 1.6e-3 / 3.0e-5 (51) | 1.4e-4 / 4.7e-6 (30) | 9.9e-4 / 8.2e-5 (12) | 15–47 | 8–29 | fail |
| bfloat16 | 9.4e-3 / 3.0e-5 (310) | 1.2e-3 / 4.7e-6 (256) | 8.8e-3 / 7.8e-5 (112) | 146–746 | 50–353 | fail |

- The tiles round x (the node features, read by every later layer) once per write.
- float16 has 11 significant bits, bfloat16 has 8: the errors grow about as 2^3.
- float16 has the range for these features (no overflow in any gate structure).

**NVE drift** (`tools/bench/tiled_nve.py`): SrTiO3 20×20×20, 40,000 atoms, 600 K, 1,000 velocity-Verlet steps in
float64; tiled, host state aliased, wavefront.

| Store | Drift (meV/atom/ps) | Spread (meV/atom) |
| --- | --- | --- |
| float32 | 0.0023 | 0.0027 |
| float16 | 0.0022 | 0.0028 |
| bfloat16 | 0.00007 | 0.0143 |

- float16 conserves energy as well as float32: its rounding is a fixed, smooth perturbation of the potential, so it
  causes no drift.
- bfloat16's spread is five times larger.
- So float16 is usable for MD where the host is short. It fails the force gate, so it is not a default.

## Speed and host memory

**A/B, both orders** (`tools/bench/tiled_ab.py`, `config:state_dtype=…` sides; SrTiO3 975,560 atoms, wavefront,
5060 Ti, jobs 24–25; the ratio is the variant's time over float32's):

| Variant | Order AB | Order BA |
| --- | --- | --- |
| float16 store | 0.983 | 0.977 |
| bfloat16 store | 0.979 | 0.981 |
| no kept lists (searched every call) | 1.137 | 1.150 |

- The 16-bit store is about 2 % faster (half the host traffic), but that does not outweigh the gate.
- Dropping the kept lists costs 14–15 %, so `"auto"` drops them only when the state does not fit otherwise.
- At 975,560 atoms the host RSS goes from 7.25 GB (float32) to 6.33 GB (float16).

**The host by owner** at 10,001,880 atoms (SrTiO3 126³; `tools/bench/tiled_host_profile.py` with
`HOSTPROF_CENSUS=1`; jobs 27–31). Wavefront, host state aliased, species index, an eager then a compiled call.
Bytes per atom:

| Owner | float32, kept lists | float16, no kept lists |
| --- | ---: | ---: |
| fixed (Python, torch, CUDA, the model; 2.2 GB) | 233 | 233 |
| hidden anonymous memory not in any array (CUDA, compile, allocator; 3.6 GB + 60 B/atom at 1.25–10M) | (not censused) | about 420 |
| row pool x (30 % of the atoms after call 2) | 616 | 308 |
| row pool sc | 154 | 77 |
| alias window (float32, 4.5 % of the atoms) | 92 | 92 |
| kept tile lists (dropped at the 20 GiB floor in call 2 on this 62 GB PC) | 313 | 0 |
| per-atom int32 maps (`_row_of`, `_slot_of`), atomic numbers (int64, twice), others | 33 | 33 |
| **process peak** | **2,144** | **1,147** |
| build peak (MACE's batch without its list, the planner's build) | 569 | 550 |

- **Fixed versus per-atom.** The fixed and hidden rows are almost constant in N (3.6 GB hidden at 1.25M atoms, 3.8 GB
  at 5M, 4.2 GB at 10M). At 10M they look large per atom; at 149M they are about 40 B/atom together.
- **Growth with N.** The row pool's share falls with N (74–91 % of the atoms at 1M, 30 % at 10M, 12.3 % at 149M);
  the window's share falls too.
- **What limits at large N:**
  - the build's transient, about 340 B/atom (MACE's input batch on the host: float64 positions, the uint8 one-hot of
    89 elements, the batch vector);
  - the call: the row pool, window, maps and hidden memory, about 290 B/atom at 149M in float16 and 450 in float32.
- **malloc settings** (job 30): `MALLOC_MMAP_THRESHOLD_=1048576` lowers the process peak by 7 % (1,147 → 1,068 B/atom)
  and `MALLOC_ARENA_MAX=2` by 5 %, with call times within 1 %. This is an environment setting for capacity runs, not a
  code change.

**The items the brief named:**
- **Float64 integrator arrays.** The tiled path has none: `SimulationState` keeps positions, velocities and forces
  on the device. The host's float64 is the kept lists' reference positions (24 B/atom), which go with the lists.
- **Tile-list cache.** About 3 B per edge (int16 ranks with a dictionary, one-byte unit codes, narrow counts).
  - It is a superset within cutoff + skin, so it holds about 2.2× the cutoff's edges.
  - Delta-encoding the ranks would save at most a third of it. The lists now go whole when the host is short, which
    saves all of it, so compaction is not worth the code (ledger row).
- **Build host peak.** 6.5–6.9 kB/atom at 20M before MACE's batch skipped its neighbour list; about 0.55 kB/atom
  now (340 B/atom of it per atom).

## Tests

`tests/cpu/test_tiled.py`:
- **The 16-bit stores.** `test_state_dtype_stores_the_host_state_in_16_bits_and_rounds_only_it`, for each store,
  with the aliased state and with the wavefront: only the stored values are rounded; the result is within the
  format's rounding of float64's.
- **"auto".** `test_state_dtype_auto_drops_the_kept_lists_before_it_rounds_the_state`: with room, the call's dtype;
  with less room, the kept lists go and the call is bitwise the same; with less still, float16.
- **The cgroup room.** `test_host_room_is_the_memory_cgroups_where_they_limit_it` covers v2 (a job's limit one level
  above its step), v1, and no limit.
- **The neighbours fix.** `test_wavefront_runs_on_the_force_fields_own_list`.
- **The precondition.** `test_state_dtype_needs_the_aliased_host_state`.

## Runs

- **Records:** `results/tiled-capacity-20261109/`, holding the job scripts, A/B JSONs, gate metadata, host profiles and
  the Alex scan.
- **Tools:**
  - `tools/bench/tiled_nve.py`;
  - `tools/bench/tiled_host_profile.py`;
  - `tools/bench/tiled_capacity.py --state-dtype --list-reuse`;
  - `tools/bench/tiled_ab.py config:FIELD=VALUE`;
  - `tools/force_stress_gate.py --tiled-config`.
