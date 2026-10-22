# 3-T14 smoke, local RTX 5060 Ti, 22 October 2026

- **Machine:** the local PC (WSL2), NVIDIA GeForce RTX 5060 Ti, driver 610.88, under a
  `tools/ci/local_gpu.py run` reservation.
- **Environment:** `~/environments/mace_cueq_oeq` (Python 3.10.12, no `tomli`: the protocol went in as
  YAML, `smoke.yaml`), `PYTHONPATH` = the 3-T14 worktree.
- **FastMACE:** commit `e55952a` (main a031b96 with 3-T11's move-energy gate, plus the YAML config
  reader). Baseline: none (no A/B comparison here).
- **Run:** `results/3-T14-protocol/smoke.toml`: `swap_anneal_NaNbOCl4` with a 5.5 ps schedule
  (0 → 1000 K in 0.5 ps, then 150 K/ps to 250 K, 5,500 steps at 1 fs). MH-1 `matpes_r2scan`,
  hybrid float32, NaNbOCl4 Ma 2026 3×3×3 (756 atoms), seed drawn and recorded in `metadata.json`.
  Two pieces: `--steps 3000`, then `--resume`. The trajectories and restart files stay in
  `~/fastmace-t14/smoke-20261022a/`.

| Stage | Result |
| --- | --- |
| Relaxation | LBFGS, fixed cell: −10.248026 → −10.250177 eV/atom in 19 steps (48 s), fmax 0.0082 eV/Å |
| Move-energy gate | passed: swap ΔE error 0.0014 (first piece) and 0.0023 (resume) k_B T_min, limit 0.01 at 250 K, 13 configurations; MC energies float32 |
| MD, piece 1 (steps 1–3,000) | 8.78 ms/step (MD only), setup 94 s |
| MD, piece 2 (steps 3,001–5,500) | 15.1 ms/step (MD only; includes the resumed process's graph captures, which run inside the loop); setup 83 s |
| MC | swap O–Cl every 100 steps, 50 attempts, `exchange`; attached over steps 124–5,500; 2,700 attempted, 0 impossible, **0 accepted**; 4.35 ms per attempt (device time) |
| Restart | at step 3,000; the resumed run continued the schedule and the attached move set |
| Quench | 21 frames (10 heating, 10 cooling, the final one), 0 not converged; heating −10.250179 ± 0.000001 eV/atom, cooling −10.24791 ± 0.00173 eV/atom, final −10.247667 eV/atom |
| Engine | graph, pipelined, 109 neighbour rebuilds, 1 re-capture, 0 overflows redone |

The physical outcome is recorded, not judged: no O↔Cl swap was accepted between 250 and 1000 K in this
short run, and the cooling quenches end about 2–4 meV/atom above the relaxed start.

The run's exit status was 2 although every stage completed (`protocol.json` status `complete`): `run.sh`
was edited while bash was still reading it, and bash read the changed tail as the usage branch.
