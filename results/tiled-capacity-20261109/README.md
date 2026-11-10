# Tiled capacity: host state dtype, host budget, A100 ceiling (9–10 November 2026)

Record: [docs/validation/tiled/tiled-host-state-dtype.md](../../docs/validation/tiled/tiled-host-state-dtype.md).

| Folder | Jobs | What |
| --- | --- | --- |
| `ab/` | 24, 25 (5060 Ti) | `tiled_ab.py`: float16 and bfloat16 store against float32, no kept lists; both orders |
| `gate/` | 24, 25 (5060 Ti) | `force_stress_gate.py --tiled-config state_dtype=…`: metadata and summary lines |
| `nve/` | 25 (5060 Ti) | `tiled_nve.py`: 40,000 atoms, 1,000 steps, each store |
| `hostprof/` | 26–31, 33, 34 (5060 Ti) | `tiled_host_profile.py`: host by owner, censuses, malloc settings, 1.25M–30M atoms |
| `alex/` | Slurm 4972536, 4973583, 4973621, 4973640 | A100-80 scan (149M, 320M, 397.5M atoms), the leak census before and after the fix |
| `jobs/` | – | the job scripts and the summary helpers (`hostsumm.py`, `censsumm.py`) |
