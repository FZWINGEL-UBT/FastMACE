| Process | Structure | 1b (ms) | 2 (ms) | 2/1b | pair range | floor | plan equal | valid | pass |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| regression2-ab-2-first | NaCl | 3.608 | 3.570 | 0.9893 | 0.9849–1.0074 | 1.10 % | True | True | yes |
| regression2-ab-2-first | NbCl5 | 27.240 | 27.346 | 1.0039 | 0.9969–1.0093 | 0.40 % | True | True | yes |
| regression2-ab-abab | NaCl | 3.562 | 3.577 | 1.0043 | 0.9931–1.0155 | 1.10 % | True | True | yes |
| regression2-ab-abab | NbCl5 | 27.326 | 27.288 | 0.9986 | 0.9968–1.0004 | 0.40 % | True | True | yes |
| regression2-ab-abba | NaCl | 3.616 | 3.620 | 1.0011 | 0.9886–1.0086 | 1.10 % | True | True | yes |
| regression2-ab-abba | NbCl5 | 27.279 | 27.128 | 0.9945 | 0.9880–0.9963 | 0.40 % | True | True | NO |

| Control | Side | Structure | load (s) | setup (s) | median (ms) |
| --- | --- | --- | ---: | ---: | ---: |
| regression2-control-1b | 1b | NaCl | 0.1 | 18.5 | 3.486 |
| regression2-control-1b | 1b | NbCl5 | 0.1 | 17.3 | 27.137 |
| regression2-control-2 | 2 | NaCl | 0.1 | 17.9 | 3.471 |
| regression2-control-2 | 2 | NbCl5 | 0.1 | 17.9 | 26.947 |

| Forces | Structure | Side | max abs (eV/Å) | pass |
| --- | --- | --- | ---: | --- |
| regression2-ab-2-first | NaCl | 2 | 2.78e-07 | True |
| regression2-ab-2-first | NaCl | 1b | 2.63e-07 | True |
| regression2-ab-2-first | NbCl5 | 2 | 1.43e-06 | True |
| regression2-ab-2-first | NbCl5 | 1b | 1.56e-06 | True |
| regression2-ab-2-first | NaCl | between versions | 2.48e-07 | None |
| regression2-ab-2-first | NbCl5 | between versions | 1.43e-06 | None |
| regression2-ab-abab | NaCl | 1b | 2.83e-07 | True |
| regression2-ab-abab | NaCl | 2 | 2.28e-07 | True |
| regression2-ab-abab | NbCl5 | 1b | 1.62e-06 | True |
| regression2-ab-abab | NbCl5 | 2 | 1.87e-06 | True |
| regression2-ab-abab | NaCl | between versions | 2.51e-07 | None |
| regression2-ab-abab | NbCl5 | between versions | 1.48e-06 | None |
| regression2-ab-abba | NaCl | 1b | 2.60e-07 | True |
| regression2-ab-abba | NaCl | 2 | 2.44e-07 | True |
| regression2-ab-abba | NbCl5 | 1b | 2.00e-06 | True |
| regression2-ab-abba | NbCl5 | 2 | 1.44e-06 | True |
| regression2-ab-abba | NaCl | between versions | 2.64e-07 | None |
| regression2-ab-abba | NbCl5 | between versions | 1.40e-06 | None |

Trajectories, cueq: {"backend": "cueq", "release": {"side": "1b", "commit": "673a8a14cdb6ef869c7b186d82201ff182669cf5"}, "current": {"side": "2", "commit": "d885b1227ca13f1bd45f6b4f52b56d9ad599dd2f"}, "plans_equal": {"20261019": true}, "positions_max_A": {"20261019": 4.3520742565306136e-13}, "repeat_positions_max_A": {"20261019": 3.2862601528904634e-13}, "limit_A": 1e-12, "passed": true}

Trajectories, hybrid: {"backend": "hybrid", "release": {"side": "1b", "commit": "673a8a14cdb6ef869c7b186d82201ff182669cf5"}, "current": {"side": "2", "commit": "d885b1227ca13f1bd45f6b4f52b56d9ad599dd2f"}, "plans_equal": {"20261001": true, "20261101": true}, "between_seeds": {"release": {"temperature_K": 14.550055883790947, "energy_meV_per_atom": 1.8393524796875838, "bins_meV_per_atom_max": 11.711903192378692, "bins_meV_per_atom_rms": 4.999374981509425}, "current": {"temperature_K": 14.550047317383758, "energy_meV_per_atom": 1.8393526695392737, "bins_meV_per_atom_max": 11.711898777219787, "bins_meV_per_atom_rms": 4.999373891019979}}, "across_versions": {"20261001": {"temperature_K": 7.845305276532599e-07, "energy_meV_per_atom": 9.382183918660303e-08, "bins_meV_per_atom_max": 3.1395224198149663e-06, "bins_meV_per_atom_rms": 1.8834209896469098e-06}, "20261101": {"temperature_K": 9.350937716590124e-06, "energy_meV_per_atom": 9.602985073797754e-08, "bins_meV_per_atom_max": 7.5084152229010215e-06, "bins_meV_per_atom_rms": 4.351303320217341e-06}}, "limits": {"temperature_K": 29.100111767581893, "energy_meV_per_atom": 3.6787049593751675, "bins_meV_per_atom_rms": 9.99874996301885}, "passed": true}

acceptance-tier: 30 passed, 0 failed, 0 errors, 0 skipped, 178.8 min; failed: []

full-tier: 244 passed, 1 failed, 0 errors, 2 skipped, 37.4 min; failed: ['tests.gpu.test_reference_gpu::test_polar1s_reference_equals_1a_t19']
