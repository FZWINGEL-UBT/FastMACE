# Tiled call at 1M atoms (3 November 2026)

SrTiO3 58×58×58 (975,560 atoms), OMAT-0 medium, float32. Report: `docs/validation/tiled-1m-speed.md`.

- `rtx4000ada-murphy/murphy-1m-profile-{device,host}`: `--profile-last` (source `b64cf7ba`).
- `rtx4000ada-murphy/murphy-1m-sweeps-{device,host}`: `--sweep-times` (`95a6f879`).
- `rtx4000ada-murphy/murphy-1m-sweep-profile`: `--sweep-profile` (`e7a1a06a`).
- `rtx4000ada-murphy/murphy-1m-cap-*`: receiver cap 16,384 to 49,152 with `--no-budget --sweep-times`
  (`e7a1a06a`). `dev-r32` ran out of GPU memory.
- `rtx4000ada-murphy/murphy-trims-{base,trim}-{a,b}`: the trims A/B, main `7e3e8cd5` against `aa5dbbfc`
  (`trims_ab.sh`), and `murphy-trims-gputests/log.txt`, the GPU tests of `aa5dbbfc`.
- `rtx5060ti-local/profile-1m-{device,host}`: `--profile-last` on the local RTX 5060 Ti (WSL), R = 8,192
  (`f5b158a1`).

Each run directory holds the bench record (`tiled-*.json`) and, for the workstation runs, the idle check.
