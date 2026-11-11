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

## Capacity on one A100-80 (Alex, 9–10 November 2026)

**Result: 397,535,000 atoms** (SrTiO3 430³) in one Slurm job with one A100-SXM4-80GB on node a0934. Alex gives a
one-GPU job a 231.5 GB memory cgroup. Source `7e4fbf4c`; environment `fm-a100omat/env` (torch 2.14.1+cu130); OMAT-0
medium, float32 calls. Settings:
- `tools/bench/tiled_capacity.py --mode auto --tile-list --host-state --alias-state --species-index --wavefront on
  --calls 2 --state-dtype auto`;
- the first call eager (it measures its memory), the second compiled.

| Atoms | Build | Eager call | Compiled call | µs/atom (compiled) | GPU peak | Host peak | Host state, kept lists |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 148,955,000 | 371 s | 3,719 s (R 8,192) | 1,810 s (R 16,384) | 12.15 | 15.6 GiB | 117.3 GB | float32, kept |
| 320,000,000 | 519 s | 6,970 s | 3,585 s | 11.20 | 32.8 GiB | 207.9 GB | float32; lists dropped in call 2 |
| **397,535,000** | 660 s | 11,468 s | **3,913 s** | **9.84** | 40.6 GiB | 229.1 GB | **float16** in call 2; lists dropped |

- **Validity.** Every point is valid, with edges exactly 74 per atom. 397.5M atoms have 29,417,590,000 edges.
- **Energy per atom.** −7.989547 eV at 149M and 320M (float32 state). At 397.5M it is −7.989009 eV, 0.54 meV/atom
  above: the float16 state's rounding, about four times the shift measured at 975,560 atoms.
- **Largest force component.** 2.2 meV/Å at 397.5M, against 1.3 meV/Å in float32 (perfect crystal, so the exact
  value is 0).
- **What limits.** The host: 229.1 GB of the 231.5 GB cgroup at 397.5M.
  - `"auto"` worked as designed. At 320M the kept lists went in the second call. At 397.5M they went in the first
    (the list floor), and the second call stored float16.
  - SrTiO3 470³ (519M atoms) would need about 300 GB at the same bytes per atom.
  - That point ended with a segmentation fault 10 minutes in, before the build. Its faulthandler stack dump fired at
    the same moment and printed only one ASE frame. With the host projection above, the point was not retried.
- **GPU memory** is about 100 B/atom plus the tiles' workspace, so an A100-80 would hold about 700M atoms on the
  device.

**Two device leaks and two slow build steps, found on the way (fixed in `ddd26d16`, `7e4fbf4c`).** Neither the
workstations nor the 5060 Ti run these A100-only rewrites, so neither leak showed there.
- **`segment_rows._indices`** (the A100's `SEGMENT_ROWS`): one int32 index tensor per row count. Tiles meet a new row
  count almost every tile.
  - Now there is one tensor per layout, sized to the most rows seen; fewer rows read its prefix.
- **`refresh_composed`** (compose_skip, also A100 only): the composed weights of every tile's species table, about
  0.55 MB per tile. That was 3.9 GiB after a 5M-atom call (census of job 4973583), and at 149M the GPU ran out of
  memory 30 minutes into the first call (17 to 81 GB).
  - Now entries go once their table is gone; a 5M-atom call stays flat at 0.92 GiB.
- **The build's whole-structure scans:**
  - a Python set over `atoms.numbers` in six places (now `np.unique`);
  - `torch.unique(dim=0)` on the 13 GB one-hot, once per module in the A100's species-product self-checks (now the
    unit rows of the columns present, checked in chunks).
  - The 149M build went from 2,018 s and a 79 GB host peak to 371 s and 43 GB. 519M had run out of host memory in
    this build (226 GB).

**Records:** `results/tiled-capacity-20261109/alex/` holds:
- `cap-4973640` (the scan), with its `run.log`, per-minute `mem.log` and the call records;
- `cap-4972536` (the 149M run that ran out of GPU memory);
- `leak-4973583` and `leak-4973621` (the census before and after the fix);
- job scripts in `jobs/`.

## Capacity on one RTX 4000 Ada (murphy, 10 November 2026)

**Result: 179,685,000 atoms** (SrTiO3 330³) on murphy's RTX 4000 Ada (20 GB, 251 GiB host memory), up from the
148,955,000-atom record of 2 November (compiled call 2,947 s, 19.8 µs/atom). Source `d036568b`; NAS
`mace_cueq_oeq`; the same `tiled_capacity.py` settings as on the A100. The scan (`jobs/ws_scan.sh`) went upward from
163.8M and stopped at the first failure. Before each point it checked that the host had none of our other jobs and
passed `run.sh`'s idle check (no compute process on the GPU, no other user's busy process).

| Atoms | Build | Eager call | Compiled call | µs/atom (compiled) | GPU peak | Host peak | Host state, kept lists |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 163,840,000 | 230 s | 2,354 s | 1,622 s | 9.90 | 17.1 GiB | 124.5 GB | float32, kept |
| **179,685,000** | 218 s | 2,479 s | **1,773 s** | **9.87** | 18.7 GiB | 129.0 GB | float32, kept |
| 196,520,000 | 236 s | out of GPU memory | – | – | – | – | – |

- **Validity.** Both points are valid, with 74 edges per atom (13,296,690,000 at 179.7M). The energy is
  −7.989549 eV/atom; the largest force component is 1.3 meV/Å.
- **What limits.** The device. At 196.5M the first tiled call asked for 2.26 GiB with 1.39 GiB free: before the
  tiles, about 10.7 GiB held the per-atom device arrays (about 60 B/atom), and the tiles' workspace (R = 16,384) took
  the rest. The host used 129 of 251 GiB, so host memory and `state_dtype` "auto" never came into play.
- **Speed.** The compiled call costs 9.9 µs/atom, half the 2 November record's 19.8 µs/atom. Two things changed:
  the tiled speed switches from the November rounds (`TILE_BLOCK_ROWS` is on everywhere; the `TILED_SPEED_GPUS`
  switches are not on the Ada), and the receiver budget's R = 16,384 for the compiled call.

**Records:** `results/tiled-capacity-20261109/murphy/` (each point's `run.log`, idle check and record, and
`ws-scan-murphy.log`).

**Recheck with the speed switches on** (`4a1ade7e`, `TILED_SPEED_GPUS` holds the Ada; murphy, same settings): 179,685,000
atoms still fit. Build 221 s, eager call 2,472 s, compiled call **1,676 s (9.33 µs/atom**, was 9.87, −5.5 %), GPU peak
18.7 GiB, host peak 136.1 GB (was 129.0 GB; the host worker and the side-stream copies), valid with 74 edges per
atom and the largest force component 1.3 meV/Å. The capacity call gains less than the 975,560-atom A/B (−13 % with
the state on the device, −23 % on the wavefront): at this size the host state's link traffic and the neighbour
search (329 s) are fixed costs the switches do not touch. Records:
`results/tiled-speed-20261110/murphy/murphy-cap-4a1ade7e-330/`.

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
- **The A100 fixes.**
  - `tests/cpu/test_segment_rows_indices.py`: one cached index tensor per layout, prefixes exact.
  - `tests/cpu/test_composed_cache.py`: dead tables' entries go; the live one keeps its addresses.
  - `tests/cpu/test_capacity_build_scans.py`: `distinct_numbers`, and the one-hot's distinct rows equal to
    `torch.unique`'s.
- **Full CPU suite** on `d036568b` (murphy, NAS `mace_cueq_latest`, `-n 8 -m "not acceptance"`): 5,513 passed, 863
  skipped.

## Runs

- **Records:** `results/tiled-capacity-20261109/`, holding the job scripts, A/B JSONs, gate metadata, host profiles and
  the Alex scan.
- **Tools:**
  - `tools/bench/tiled_nve.py`;
  - `tools/bench/tiled_host_profile.py`;
  - `tools/bench/tiled_capacity.py --state-dtype --list-reuse`;
  - `tools/bench/tiled_ab.py config:FIELD=VALUE`;
  - `tools/force_stress_gate.py --tiled-config`.
