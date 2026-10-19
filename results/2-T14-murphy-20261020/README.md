# 2-T14 raw results (murphy, 20 October 2026)

Raw files of plan task 2-T14, "Find the flaky test". The report is
[docs/validation/2-T14-flaky-test.md](../../docs/validation/2-T14-flaky-test.md). Everything
ran on murphy's CPUs in the NAS environment `mace_cueq_latest`, from `PYTHONPATH` copies under
`/mnt/homes/zwingel/fastmace-2-T14/` (deleted after the merge).

| Path | What it is | Report section |
| --- | --- | --- |
| `batches.log` | One line per run: start, exit status, pytest summary; batch starts and aborts | Batch 1, Confirmation runs |
| `logs/run-01.log` … `run-04.log` | Full pytest output (`-rf`) of batch 1: `main` ed39645, default order, cold caches; 03 and 04 show the failure | Batch 1 |
| `logs/run-c1.log` … `run-c6.log` | Confirmation runs: c1–c5 at f364992 (4cdcfd1 + 1b-T27), c6 at 4cdcfd1 | Confirmation runs |
| `junit/run-*.xml` | The junit files of the same ten runs | Batch 1, Confirmation runs |
| `aborted/run-01.log` … `run-04.log` | First start of batch 1: stopped within 2 s because the basetemp parent was missing (a slip in the runner) | — |
| `aborted/run-c1-stale-base.log` | First start of c1, stopped after 80 s because `main` moved (2-T01 merged) | — |
| `run_batch.sh` | The runner: one detached process, runs one after another, `nice -n 10`, log and junit per run | Setup |
| `probe.sh` | Runs given tests in one process (`-p no:xdist`) with the same environment; output in `probe/<tag>.log` | Cause |
| `plugins/t14_order.py` | Seeded order plugin (`--t14-seed`), prepared, not needed | Setup |
| `plugins/*.py` (others) | Diagnosis scripts: `polar_probe.py` (the polar test's deviation after optional pre-steps from `pre.py`), `state_probe.py`, `snap_probe.py`, `weights_probe.py`, `trace_probe.py` and `trace_cmp.py`, `core_probe.py`, `desc_probe.py`, `fill_probe.py`, `hash_probe.py` | Cause, Fix |
| `probe/modules.txt`, `bisect.txt`, `after-test_*.log` | Module bisection: each CPU test module followed by the polar test in one process; five modules fail it | Cause |
| `probe/pg*.{txt,log}` | Test bisection inside `test_provenance_git.py`: only `test_a_simulation_builds_the_block_once_across_ten_runs` fails it | Cause |
| `probe/alone*.log`, `module*.log` | The polar test alone and with its own module: passes | Cause |
| `probe/a.txt`, `b.txt`, `c.txt`, `s-*.txt`, `t-*.txt`, `u-*.txt`, `v-*.txt`, `w-*.txt`, `x-*.txt` | `polar_probe.py` outputs: patch restore and dynamo reset (a–c), Simulation stages (s), float32 vs float64 loads (t), conversion and import variants (u), clearing functools caches per package (v), model families (w), `PYTORCH_JIT=0` (x, e3nn refuses it) | Cause |
| `probe/m-*.txt` | `sys.modules` after the gate references and after the build, and the difference | Cause (imports ruled out) |
| `probe/casc-*.log`, `casc2-*.log` | The `test_loading` conv-fusion cascade before and after the fixture fix | Tests |
| `probe/fixed-*.log` | The first tolerance change (later reverted) on the trigger pairs | — |
| `probe/red*.log`, `green*.log` | The new tests before the library fix (failing) and after it, with neighbouring modules | Tests |

Other processes: no other user's process ran on murphy during the runs. Processes of our own
account (trial suites of the conductor and other workers) show in the `ps` lines at the start
and end of runs 01, 02 and c2–c6. That is allowed for untimed runs; no run here is timed.

No tokens or other users' data are in these files. Before the commit they were checked for
token patterns, other users' paths, hostnames, IP addresses and e-mail addresses. Nothing was
dropped: the only hits were module names (`secrets`, `pygments.token`), and the only home paths
are `/mnt/homes/zwingel`.
