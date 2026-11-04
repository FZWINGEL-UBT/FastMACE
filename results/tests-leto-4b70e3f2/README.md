# CPU suite on leto for fastmace/omat0-strided-tables at 4b70e3f2 (4 November 2026)

- Machine: leto, CPU only (`CUDA_VISIBLE_DEVICES=''`); environment `/mnt/homes/zwingel/environments/mace_cueq_latest`;
  the branch copied with `git archive 4b70e3f2` to `/mnt/homes/zwingel/fastmace-branches/omat0-strided-4b70e3f2`.
- Command: `run.sh` (16 xdist workers, `--dist=load --maxschedchunk=1`, NAS basetemp, `safe.directory` for
  `test_colab_driver.py` by environment variables only).
- Result: 5,127 passed, 202 skipped, 0 failed in 28 min 46 s (`summary.txt`, `pytest-cpu-tail.txt`).
- Ruff: `run.sh` linted the whole tree (315 findings, almost all in `results/` and `tools/` scripts). The project
  scope, `ruff check fastmace tests`, at 4b70e3f2 gives 3 long lines, all present on main before the merge.
- `idle_check.txt`: the GPU and process state at launch (GPU idle; only another user's idle session).
