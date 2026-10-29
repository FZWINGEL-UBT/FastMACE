# Decision 66: `grouped_launch` for MH-1 on the RTX 5060 Ti (29 October 2026)

Owner decision 66 (SPEC §18): per-GPU defaults where same-process whole-step A/Bs in both build orders measured
a speed-up, by exact device name. `grouped_launch` (owner decision 57) had been default for the generic MACE
family only, because MH-1 lost 0.5 % on sm_89 (OPT-T03b) although it gained on sm_120 in optimisation round 1
(NaCl 0.977 / 0.982). That round-1 number predates `segment_gemm`, so it was measured again on current main.

## Measurement

- `tools/ab_rewrites.py --b grouped_launch=on`, MH-1 `mh1-r2scan` float32 default plan, 300 K, 12 × 300 steps at
  0.5 fs, same process; build orders ab (ABAB) and ba (ABBA).
- Local RTX 5060 Ti (sm_120), `~/environments/mace_cueq_oeq` (torch 2.14.1+cu130), FastMACE `30ff3b7c` (main),
  FIFO GPU reservation (`tools/ci/local_gpu.py run --quiet`). Records: `results/decision66-grouped-launch-20261029/`.

| Cell | ab: a / b ms, b/a | ba: b / a ms, b/a |
| --- | --- | --- |
| NaCl | 2.657 / 2.579, **0.971** | 2.602 / 2.669, **0.975** |
| NbCl5 | 18.546 / 18.590, 1.002 | 19.004 / 19.034, 0.998 |
| NaNbO3 | 14.534 / 14.544, 1.001 | 14.675 / 14.688, 0.999 |

All runs valid (no compile while timed, no other GPU process). NaCl gains 2.5–2.9 % in both orders; the large cells
are within noise in both directions (the shared shape rule groups only small products, so they run the stock
products).

## Rule

On for MH-1 (`planner.fixed._mh1`) on `DECISION_66_GROUPED_LAUNCH_GPUS = ("NVIDIA GeForce RTX 5060 Ti",)` in
float32 plans that are not reproducible; off elsewhere with the reason "MH-1 not on a measured GPU, decision 66
(<GPU>)". The RTX 4000 Ada stays off (OPT-T03b: 1.005 / 1.003); other GPUs are not measured. POLAR-1, OMOL-0 and
MACE-OFF are unchanged (off). An explicit `"on"` or `"off"` still overrides.
