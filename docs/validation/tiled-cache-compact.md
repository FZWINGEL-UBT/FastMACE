# Tiled path: the compact list cache, MD at 70.96M with the cache on, and the MD proof plot

The 70.96M NVE gate run of [tiled-md-host.md](tiled-md-host.md) had the list cache off. The reason was host memory:
the cache took 271 bytes per atom (19 GB), on top of a 220 GiB host peak, next to the host integrator. This note
covers four things:
- the cache is now about half that size;
- MD at 70.96M runs with the cache on;
- the device's `local_of` index is now int32;
- the owner's plot of the run.

## The compact cache (bitwise)

`TiledForceField._source` keeps each tile's edges on the host for the call's later sweeps:

- **Senders:** int32, as before.
- **Unit shifts:** when every shift of the tile lies in {-1, 0, 1}³, each is one byte, an index into the 27 shifts
  (`_pack_units` and `_unpack_units`). That holds whenever every cell length exceeds the search radius. Otherwise
  the shifts are int16, as before; a tile whose shifts exceed int16 is not cached.
- **Receiver counts:** the narrowest of uint8, int16 and int32 that holds them (`_narrow_counts`).

The edges rebuilt from the cache are those of the search: the same integer values and the same shifts. Tests:

- `test_tile_lists_built_once_per_call_are_bitwise_the_rebuilt_ones` now also asserts at most 5 bytes per edge plus
  1 per atom.
- `test_list_cache_unit_shifts_one_byte_codes_or_int16_and_back_exactly` checks both encodings and the narrow counts.

## int32 `local_of`

`local_of` is the call's whole-system map from atoms to their local index in a tile. It is now int32, and the local
edge index is cast back to int64. That saves 4 bytes per atom on the device, 0.28 GB at 70.96M.

CPU tests on trillian (`test_tiled.py`, `test_layering.py`, `test_neighbors.py`): 163 passed, 91 skipped.

## MD at 70.96M with the cache on (leto)

**Setup.** `tiled_md.py --mode auto --integrator host --steps 4` at efb1ebbb, run on leto. Run:
`results/tiled-md-proof-20261101/runs/leto-nve-proof-NbCl5-69x69x207`.

- **Mode.** "auto", the library default: the first call is eager and measures its memory, then the field compiles
  with the budget at R = 16,384.
- **Host guard.** 3,330 bytes per atom, keeping 20 GB free. 264 GB were available and the guard required 236 GB.
- **New in `tiled_md.py`.** Each call records the device's peak and reserved bytes, the host's resident and peak
  bytes (VmHWM, reset before each call), the call's time, its mode and R.

| Step | Call | Wall time | GPU peak (of 19.54 GiB) | Host peak (of 251 GiB) | List cache |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | eager, R = 4,096 (the first call) | 1,758 s | 14.22 GiB | 210.6 GiB | 9.39 GB |
| 1 | compiled, R = 16,384 | 1,080 s | 18.29 GiB | 215.9 GiB | 9.53 GB |
| 2 | compiled | 1,053 s | 18.29 GiB | 216.0 GiB | 9.61 GB |
| 3 | compiled | 1,112 s | 18.32 GiB | 216.3 GiB | 9.63 GB |
| 4 | compiled | 1,094 s | 18.32 GiB | 216.3 GiB | 9.63 GB |

- **The cache:** 9.63 GB, or **136 bytes per atom** against 271 before. Its size grows by 2.5 % over the run with
  the edges, 1.865e9 to 1.912e9.
- **The MD step with the cache on:** 1,053-1,112 s. The run with the cache off took 1,194-1,279 s; a repeated call on
  fixed positions takes 1,034 s. So the cache saves about 10 %, as at 9.26M (0.915). The rest of the gap to the
  repeated call is the edges' growth in MD and the slot changes (3 by step 3), plus the host integrator; their
  shares were not measured separately.
- **Memory:**
  - The GPU peak in MD is 18.32 GiB, against 17.73 GiB for repeated calls. Reserved memory is 19.26 GiB of 19.54
    GiB. The difference is the device copy of the positions for each call, plus the slots' slightly larger sizes
    in MD.
  - The host peak is 216 GiB of 251 GiB.
- **Drift:** 0.028 meV/atom over the 4 steps.

## The MD proof plot

Files: `results/tiled-md-proof-20261101/plot.py`, `md_proof.png` and `md_proof.csv` (one row per point, with its
source file).

**Panels:**
- the energies per atom, with the drift;
- the temperature;
- the wall time per step, with the eager first call marked;
- the GPU peak and the host peak per step;
- the 884,736-atom 200-step NVE drift, eager and compiled, as context for a longer run.

**The two 70.96M runs:**
- this leto run, which has every quantity;
- the murphy run of [tiled-md-host.md](tiled-md-host.md) (3baf655f, cache off), which has the energies and wall
  times. It did not record memory or the call's mode; its first call was compiled, and its first call's time was
  not kept.

**Temperature** is 2 E_kin / (3 N k_B).
