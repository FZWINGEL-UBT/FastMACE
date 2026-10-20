# 2-T14e: the shortened ASE-equivalence tests on the local RTX 5060 Ti (20 October 2026)

The changed `tests/gpu/test_ase_equivalence_gpu.py` (Langevin: 4,000 steps, limits 1e-6 K and
1e-7 meV/atom; BAOAB: the energy limit 1e-7 meV/atom), MH-1 plain cuEq float64, run under the local GPU
reservation (`local_gpu.py run`, 16:28–17:03 UTC), on FastMACE 7ce7f3b plus the uncommitted change
(the commit that follows it), WSL `mace_cueq_latest`. Report: `docs/validation/2-T14e-reference-reuse.md`.

| Path | What it is |
| --- | --- |
| `runs.log` | Start and end of both tests (the git lines are empty: WSL cannot read this worktree's Windows git path) |
| `logs/langevin.log`, `logs/langevin.junit.xml` | The Langevin test: 1 passed, 988 s |
| `logs/baoab.log`, `logs/baoab.junit.xml` | The BAOAB test: 1 passed, 1115 s |
| `results/1b-T09-langevin-ase-equivalence.json`, `results/1b-T09-baoab-ase-equivalence.json` | The tests' own summaries (`FASTMACE_RESULTS`) |
| `local_ase.sh` | The runner |
