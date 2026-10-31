# Tiled capacity path: species tables at the call's row count

The conductor's option (a) after [tiled-t3.md](tiled-t3.md). With per-tile neighbour lists, the force field's
construction limited the tiled path: `reassociate_first` evaluates the species table on an identity basis with
as many rows as the structure, for a moment about 5.9 kB per atom. Now the table has the row count of the model
call that reads it.

## What changed

- **`MACEForceField(call_rows=None)`.**
  - It is the row count of the model's calls, at which the reassociated first interaction's species tables are
    evaluated.
  - It reaches the rewrites as `Context.inputs["call_rows"]`, in both the mh1 and the generic reassociation.
  - The default is the atom count, so the whole-system path is unchanged: its tables are still bitwise the
    features of the whole-system call (3-T02).
- **`MACEForceField.species_tables(rows)`.** The tables evaluated in a batch of `rows` rows, in the order of the
  field's own tables.
- **`TiledForceField`** binds its own tables.
  - A padded slot keeps tables evaluated at its R + H node rows (made with the slot), bound together with the
    slot's streaming count.
  - An eager tile gets tables evaluated at its own R + H rows in sweeps 1 and 3.
  - Each tile's interaction 0 then reads features bitwise those its own embedding gives the atoms.
  - The field's own call binds the field's tables again.
- **`planner.build_evaluation(..., call_rows=None)`** passes the value through.
  - `tools/bench/tiled_capacity.py` builds with `call_rows=--receivers`, except with `--compare eager`, which
    calls the field on the whole system.
- **A side effect:** the species refresh closure kept the N-row identity basis alive. It now keeps `call_rows`
  rows, so the memory before the call falls too (1.54 GiB at 3.24M atoms, against 2.42 GiB at 2.99M in T3).

## Tests

Three new CPU tests in `tests/cpu/test_tiled.py`:

- A field built with `call_rows=7` evaluates the identity basis on 7 rows only, and its tables are bitwise
  `species_tables(7)` of a default field; non-positive values are refused.
- Every tile's interaction 0 reads the table of its own row count, bitwise `species_tables(R + H)`. Eager mode
  sees several row counts; padded mode sees its slots'.
- The tiled call is bitwise the same whichever `call_rows` the field was built with. The field's own call after a
  tiled call is bitwise what it was before.

murphy, GPU hidden, `-n 8`: `test_tiled.py`, `test_layering.py`, `test_forces.py`, `test_rewrites_mh1.py`,
`test_rewrites_generic.py`, `test_rewrites_shared_fields.py`, `test_superset.py`, `test_planner.py` and
`test_acceptance_tools.py`.

- First run: 388 passed, 53 skipped, 1 failed. The failing test assumed several padded slots; this small
  structure has one.
- After that fix, `test_tiled.py` alone: 35 passed.

## GPU check (murphy, RTX 4000 Ada 20 GB)

The run setup:

- `mace_cueq_oeq`, MH-1 float32 hybrid, model `mh1-r2scan`, eager tiles of R = 4,096 receivers, tile lists;
- FastMACE d550891b;
- runs `results/tiled-rows-20261031/scan.sh`.

**NbCl5 4×4×8, 9,216 atoms, rattled by 0.05 Å, against T3's records:**

| | ΔE (total) | Δ max \|F\| |
| --- | ---: | ---: |
| global list | 5.5e-6 eV | 4.8e-7 eV/Å |
| tile lists | 6.0e-6 eV | 7.2e-7 eV/Å |

The differences are float32 rounding: the tables are now evaluated at the tile's rows instead of N.

**Capacity.** Every completed point passed the capacity check: edges per atom equal to ASE's (deviation 0) and a
nonzero largest force.

| Structure | Atoms | Before | Build peak | Call total peak | Call | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| NbCl5 | 3,240,000 | 1.54 GiB | 6.80 GiB | 17.81 GiB | 85 s | valid |
| NbCl5 | 3,375,000 | 1.60 GiB | 7.08 GiB | 18.50 GiB | 99 s | valid, **limit** |
| NbCl5 | 3,796,416 | | 7.95 GiB | | | OOM in the call (the global state) |
| NaCl | 3,241,792 | 1.54 GiB | 7.50 GiB | 17.95 GiB | 93 s | valid |
| NaCl | 3,375,000 | 1.60 GiB | 7.80 GiB | 18.63 GiB | 99 s | valid, **limit** |
| NaCl | 3,511,808 | | 8.11 GiB | | | OOM in the call |

**The valid limits on 20 GB rise from 2.99M to 3.38M atoms for both NbCl5 and NaCl.**

- The build peak falls from 16.49 GiB to 7.08 GiB at about 3M atoms: about 2.2 kB per atom, now below the call's
  own peak.
- The call's total peak is about 5.9 kB per atom, mostly the global state: x, sc and their adjoints, 4.75 kB per
  atom.
- The 3.80M NbCl5 point fails allocating that state in sweep 1 (`TiledForceField._state`).

## What limits next

The global state in GPU memory: host staging, T3 part 2. The build is next after that, at about 2.2 kB per
atom. With staging, the build would limit at about 8M atoms on 20 GB.
