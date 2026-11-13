# Resident cuts against main (leto, 13 November 2026)

Branch `fastmace/first-call-diet`. Three exact cuts from the census (`../census-leto/`):

1. **Sender groups** (`5d52be61`): the neighbour list keeps the force field's own uint8 sender groups instead of an
   int64 copy (`skin.py`; the order keys widen the gathered groups): 8 B/atom.
2. **Block order** (`5d52be61`): the tiled call's bind drops the unbound whole-system BLOCK_ORDER order
   (`forces.py` `_bind_streaming_count(block_order=False)`): 4 B/atom where it was built (the A100; the Ada builds none).
3. **Host rows with kept lists** (`e253ce92`): with `--host-positions` the kept lists' float64 distance test gathers each
   tile's rows on the host instead of a whole-system float64 copy on the device (`tiled.py` `_source`): 24 B/atom in
   any call that keeps lists, so the 397.5M first call's.

`tiled.py` `_g.pos` (the third suspect) is a view of `field.positions`: nothing to cut.

## Exactness

- CPU (leeloo, `e253ce92`, `tests/cpu/test_tiled.py test_tiled_supercell.py test_neighbors.py test_neighbors_lean.py
  test_superset.py test_forces.py test_layering.py`): 446 passed, 107 skipped (FASTMACE_BASELINE not set), 0 failed.
  New: the released field's list shares its uint8 groups with the same edges (`test_released_species_maps_are_uint8_with_the_same_values`);
  the tiled bind drops the block order; host rows with kept lists bitwise the device rows over calls that search and
  calls that reuse (eager and padded, energy, forces and stress).
- GPU bitwise digests (`tiled_capacity.py --digest`, new) are not usable: the same tree twice gives different digests
  (80-a1 against 80-a2, the atomics' run-to-run order), so the GPU check is the gate.
- **Gate** (`gate/`, leto, `e253ce92`, `tools/force_stress_gate.py`, tiled R 48 compiled, OMAT-0 medium): passed on
  SrTiO3 3x3x3, NaCl and NbCl5 (energy, forces, stress). Job script `../jobs/ws_gate_cuts.sh`.

## Device bytes at 160^3 (20,480,000 atoms; the go/no-go's call, list reuse on; `160-a`, `160-b`, `160-c`)

Tree A is `7b4ac08e` with main's `forces.py`, `skin.py`, `tile_list.py`; B is `7b4ac08e` (cuts 1-2); C is `e253ce92`
(1-3). B/atom; job scripts `../jobs/ws_cuts_ab.sh`, `../jobs/ws_cuts_c.sh`.

| Tree | Before call 0 | Call 0 (eager) peak | Call 1 (compiled) peak | Energy/atom | Seconds (call 0 / 1) |
| --- | ---: | ---: | ---: | --- | ---: |
| A (main) | 48.8 | 144.3 | 177.8 | -7.9895478292 | 219.7 / 194.3 |
| B | 40.8 | 136.3 | 169.8 | -7.9895478293 | 219.3 / 197.1 |
| C | 40.8 | **112.3** | **145.8** | -7.9895478292 | 222.0 / 206.4 |

32 B/atom less in both calls (8 + 24). The seconds are one unpaired run each (cross-process noise about 3 %): C's
compiled call +6 % against A is to be confirmed by a same-process A/B before anything is said about speed; the host
gathers replace the device copy and TILE_FILTER32 is off with host rows. At 20M the size-independent tables and the
tile workspace (about 50-60 B/atom here) hide the per-atom totals; the projection to 397.5M is in the plan.

The 80^3 runs (rattle 0.05, strain 0.01; A, B twice each) show the same 8 B/atom (236.1 against 228.1 before call 0)
and are "valid": false by design (the perfect crystal's edge count does not hold rattled).
