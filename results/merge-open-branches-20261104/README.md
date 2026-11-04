# Merging the open branches (4 November 2026)

The owner asked for the branches not yet on main to be merged: `tile-grid-fit`, `bf16x3-screen`, `tiled-modal-scan`,
`tiled-slot-growth` and `reassoc-layout`. Production behaviour is unchanged:

- `TileConfig.grid_fit` is merged off (the branch had it on): one unrepeated 20M pair is not an A/B in both orders.
- `reassoc-layout` was rejected as 5 % slower on MH-1 (`docs/validation/tiled-1m-speed.md`), so `edge_outer_sum`'s
  `inner=` layout is merged and the MH-1 caller stays on the plain layout.
- `FASTMACE_SPLIT_PRECISION` keeps its default `tf32x3`.
- `tiled-slot-growth`: main already has the code in a later form; the merge adds the run records.

Check on the RTX 5060 Ti (WSL2, `mace_cueq_oeq`): MH-1 `matpes_r2scan`, NaNbO3, 1 fs, skin 0.5 Å, main and the merge
alternating in separate processes with separate compile caches (`--blocks 6`, side a of each run, ms):

| Source | Runs | Mean |
|---|---|---:|
| main 43a6891b | 13.57, 13.60, 13.51, 13.47 | 13.54 |
| merge | 13.53, 13.46, 13.52, 13.53 | 13.51 |

No slowdown. These are separate processes, so a difference below about 1 % would not show. Compile caches must not be
shared across this merge in the downgrade direction: main's source fails on graphs cached by the merge, which pass
`edge_outer_sum`'s new argument.
