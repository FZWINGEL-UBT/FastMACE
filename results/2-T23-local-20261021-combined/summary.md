| Process | Structure | 1b (ms) | 2 (ms) | 2/1b | pair range | floor | plan ok | valid | pass |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| regression2-ab-2-first | NaCl | 3.588 | 3.444 | 0.9596 | 0.9450–0.9729 | 1.10 % | True | True | yes |
| regression2-ab-2-first | NbCl5 | 27.213 | 23.162 | 0.8511 | 0.8480–0.8525 | 0.40 % | True | True | yes |
| regression2-ab-abab | NaCl | 3.577 | 3.449 | 0.9643 | 0.9329–0.9752 | 1.10 % | True | True | yes |
| regression2-ab-abab | NbCl5 | 27.178 | 23.213 | 0.8541 | 0.8464–0.8551 | 0.40 % | True | True | yes |
| regression2-ab-abba | NaCl | 3.568 | 3.424 | 0.9597 | 0.9426–0.9770 | 1.10 % | True | True | yes |
| regression2-ab-abba | NbCl5 | 27.232 | 23.238 | 0.8533 | 0.8476–0.8549 | 0.40 % | True | True | yes |

| Control | Side | Structure | load (s) | setup (s) | median (ms) |
| --- | --- | --- | ---: | ---: | ---: |
| regression2-control-1b | 1b | NaCl | 0.1 | 18.1 | 3.515 |
| regression2-control-1b | 1b | NbCl5 | 0.1 | 17.4 | 26.949 |
| regression2-control-2 | 2 | NaCl | 0.1 | 20.2 | 3.313 |
| regression2-control-2 | 2 | NbCl5 | 0.1 | 18.7 | 23.044 |

| Forces | Structure | Side | max abs (eV/Å) | pass |
| --- | --- | --- | ---: | --- |
| regression2-ab-2-first | NaCl | 2 | 2.28e-07 | True |
| regression2-ab-2-first | NaCl | 1b | 2.23e-07 | True |
| regression2-ab-2-first | NbCl5 | 2 | 1.84e-06 | True |
| regression2-ab-2-first | NbCl5 | 1b | 1.68e-06 | True |
| regression2-ab-2-first | NaCl | between versions | 2.44e-07 | None |
| regression2-ab-2-first | NbCl5 | between versions | 1.62e-06 | None |
| regression2-ab-abab | NaCl | 1b | 2.57e-07 | True |
| regression2-ab-abab | NaCl | 2 | 2.19e-07 | True |
| regression2-ab-abab | NbCl5 | 1b | 1.43e-06 | True |
| regression2-ab-abab | NbCl5 | 2 | 1.40e-06 | True |
| regression2-ab-abab | NaCl | between versions | 2.55e-07 | None |
| regression2-ab-abab | NbCl5 | between versions | 1.39e-06 | None |
| regression2-ab-abba | NaCl | 1b | 3.12e-07 | True |
| regression2-ab-abba | NaCl | 2 | 2.84e-07 | True |
| regression2-ab-abba | NbCl5 | 1b | 1.51e-06 | True |
| regression2-ab-abba | NbCl5 | 2 | 1.49e-06 | True |
| regression2-ab-abba | NaCl | between versions | 2.50e-07 | None |
| regression2-ab-abba | NbCl5 | between versions | 1.48e-06 | None |

Trajectories, cueq: {"backend": "cueq", "release": {"side": "1b", "commit": "673a8a14cdb6ef869c7b186d82201ff182669cf5"}, "current": {"side": "2", "commit": "a3eaf89a9136da4afb323538f0d587d1c620881d"}, "plan_checks": {"20261019": {"equal": true, "differences": {}, "allowed": ["gated_rows", "segment_gemm"], "unexpected": [], "as_expected": true}}, "plans_equal": {"20261019": true}, "positions_max_A": {"20261019": 1.3500311979441904e-13}, "repeat_positions_max_A": {"20261019": 4.3520742565306136e-13}, "limit_A": 1e-12, "passed": true}

Trajectories, hybrid: {"backend": "hybrid", "release": {"side": "1b", "commit": "673a8a14cdb6ef869c7b186d82201ff182669cf5"}, "current": {"side": "2", "commit": "a3eaf89a9136da4afb323538f0d587d1c620881d"}, "plan_checks": {"20261001": {"equal": false, "differences": {"gated_rows": [null, true], "segment_gemm": [null, true]}, "allowed": ["gated_rows", "segment_gemm"], "unexpected": [], "as_expected": true}, "20261101": {"equal": false, "differences": {"gated_rows": [null, true], "segment_gemm": [null, true]}, "allowed": ["gated_rows", "segment_gemm"], "unexpected": [], "as_expected": true}}, "plans_equal": {"20261001": false, "20261101": false}, "between_seeds": {"release": {"temperature_K": 14.550037846606244, "energy_meV_per_atom": 1.8393516220429618, "bins_meV_per_atom_max": 11.711948447756804, "bins_meV_per_atom_rms": 4.999389165762534}, "current": {"temperature_K": 14.550058825585268, "energy_meV_per_atom": 1.8393536055532067, "bins_meV_per_atom_max": 11.711952862916597, "bins_meV_per_atom_rms": 4.999391949275708}}, "across_versions": {"20261001": {"temperature_K": 2.6099729268480587e-05, "energy_meV_per_atom": 3.4316824937263846e-06, "bins_meV_per_atom_max": 4.4151588163288125e-05, "bins_meV_per_atom_rms": 1.7234285763007815e-05}, "20261101": {"temperature_K": 5.1207502451688924e-06, "energy_meV_per_atom": 1.448172248785795e-06, "bins_meV_per_atom_max": 4.85667479566132e-05, "bins_meV_per_atom_rms": 1.8494827488923557e-05}}, "limits": {"temperature_K": 29.10007569321249, "energy_meV_per_atom": 3.6787032440859235, "bins_meV_per_atom_rms": 9.998778331525068}, "passed": true}

| Gate job | Model | E ratio | F ratio | S ratio | E max (meV/atom) | status | pass |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| gate-matpes-pbe | matpes-pbe | 1.09 | 1.11 | 1.56 | 0.0003 | complete | yes |
| gate-matpes-r2scan | matpes-r2scan | 0.95 | 1.15 | 1.64 | 0.0002 | complete | yes |
| gate-mh1-mp | mh1-mp | 0.18 | 1.27 | 0.87 | 0.0002 | complete | yes |
| gate-mh1-omat | mh1-omat | 0.17 | 1.11 | 0.88 | 0.0002 | complete | yes |
| gate-mh1-r2scan | mh1-r2scan | 0.20 | 1.20 | 0.93 | 0.0002 | complete | yes |
| gate-mp0-medium | mp0-medium | 2.52 | 1.04 | 1.23 | 0.0003 | complete | yes |
| gate-mp0b3-medium | mp0b3-medium | 1.28 | 1.02 | 1.18 | 0.0002 | complete | yes |
| gate-mpa0-medium | mpa0-medium | 1.97 | 1.12 | 1.26 | 0.0005 | complete | yes |
| gate-omat0-medium | omat0-medium | 1.48 | 1.13 | 1.24 | 0.0003 | complete | yes |
| gate-omol0-xl | omol0-xl | – | – | – | – | failed | NO |
| gate-omol0-xl-NaNbO3-model-limited | omol0-xl | – | – | – | – | failed | NO |

acceptance-tier: 31 passed, 2 failed, 0 errors, 0 skipped, 117.1 min; failed: ['tests.gpu.test_fallback_gpu::test_a_failed_candidates_memory_is_freed_before_the_next_build', 'tests.gpu.test_subprocess_worker_gpu::test_a_killed_parent_releases_the_workers_gpu_memory']

round-tests: 9 passed, 0 failed, 0 errors, 0 skipped, 6.3 min; failed: []
