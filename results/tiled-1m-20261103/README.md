# Tiled call at 1M atoms (3 November 2026)

SrTiO3 58×58×58 (975,560 atoms), OMAT-0 medium, float32. Report: `docs/validation/tiled-1m-speed.md`.

- `rtx4000ada-murphy/murphy-1m-profile-{device,host}`: `--profile-last` (source `b64cf7ba`).
- `rtx4000ada-murphy/murphy-1m-sweeps-{device,host}`: `--sweep-times` (`95a6f879`).
- `rtx4000ada-murphy/murphy-1m-sweep-profile`: `--sweep-profile` (`e7a1a06a`).
- `rtx4000ada-murphy/murphy-1m-cap-*`: receiver cap 16,384 to 49,152 with `--no-budget --sweep-times`
  (`e7a1a06a`). `dev-r32` ran out of GPU memory.
- `rtx4000ada-murphy/murphy-trims-{base,trim}-{a,b}`: the trims A/B, main `7e3e8cd5` against `aa5dbbfc`
  (`trims_ab.sh`), and `murphy-trims-gputests/log.txt`, the GPU tests of `aa5dbbfc`.
- `rtx4000ada-murphy/murphy-reassoc-{base,new}-{a,b}`, `murphy-reassoc-sweep-profile`, `murphy-reassoc-gputests`:
  the rejected reassociation layout, main `a7159afd` against `e2903496` (`reassoc_ab.sh`);
  `rtx5060ti-local/reassoc-local/`: its whole-system MH-1 step A/B (`local_ab.sh`).
- `rtx4000ada-leto/leto-graphs-{off,on}-{a,b}` (`graphs_ab.sh`), `rtx-a4000-trillian/trillian-graphs-*` (R = 16,384, all
  out of memory; `-r8-`: R = 8,192, the graphs side out of memory), `rtx5060ti-local/graphs-{214k,40k}/`: the opt-in
  CUDA graphs for compiled tiles, `TileConfig.compiled_graphs` (`fe7b0a7b`), without and with.
- `rtx4000ada-leto/leto-md1m-{slots,branch,branch2}`: 60-step 1M MD (`tiled_md.py`) of main `7430fb51`, the branch
  with edge headroom `5a2dd70f` (rejected) and without it `3a83c308` (`md_after_gate.sh`);
  `leto-search-parts` and `leto-search-stencil`: a list rebuild's parts and the stencil A/B (`search_parts.py`,
  `search_stencil.py`).
- `rtx5060ti-local/profile-1m-{device,host}`: `--profile-last` on the local RTX 5060 Ti (WSL), R = 8,192
  (`f5b158a1`).

Each run directory holds the bench record (`tiled-*.json`) and, for the workstation runs, the idle check.
