# 2-T14e raw results (murphy, 20 October 2026)

Raw files of plan task 2-T14e, which measured reference work in the GPU tiers. The report is
[docs/validation/2-T14e-reference-reuse.md](../../docs/validation/2-T14e-reference-reuse.md).
murphy RTX 4000 Ada, shared with processes of our own account (untimed), FastMACE `main` e24d48e.

| Path | What it is |
| --- | --- |
| `full.log`, `full.junit.xml` | The full tier, `-m "not acceptance" tests/gpu`, with `--durations=0`: 244 passed, 83 min |
| `full.events.jsonl` | The timing hook's events of that run: one JSON object per reference call or process (pid, test, kind, seconds, details) |
| `full.attribution.json` | `scripts/attribute.py`'s per-test attribution of the full tier |
| `acc.log`, `acc.junit.xml`, `acc.events.jsonl`, `acc.attribution.json` | The same for the acceptance subset (`-m acceptance` of `test_engine_gpu.py`, `test_forces_gpu.py`, `test_selfcheck_gpu.py`; 7 passed, 72 min) |
| `scripts/timing_hook_sitecustomize.py` | The timing hook, used as `sitecustomize.py` at the root of a scratch copy (never in the repository). This copy also logs atoms fingerprints for reference evaluations and the load arguments, added after the runs; the runs above logged the same kinds and times without them |
| `scripts/run_full.sh` | The runner: environment, own caches, GPU process list before and after |
| `scripts/attribute.py` | Adds the events up per test against the junit times (`python attribute.py <events> <junit>`) |

No tokens or other users' data are in these files. The only home paths are `/mnt/homes/zwingel`,
and the only processes listed are of the `zwingel` account.
