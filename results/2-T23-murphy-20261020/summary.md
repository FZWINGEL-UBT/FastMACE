| Process | Structure | 1b (ms) | 2 (ms) | 2/1b | pair range | floor | plan equal | valid | pass |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| regression2-ab-2-first | NaCl | 3.628 | 3.631 | 1.0010 | 0.9927–1.0019 | 0.16 % | True | True | yes |
| regression2-ab-2-first | NbCl5 | 31.459 | 31.458 | 1.0000 | 0.9977–1.0001 | 0.16 % | True | True | yes |
| regression2-ab-abab | NaCl | 3.597 | 3.603 | 1.0016 | 1.0010–1.0169 | 0.16 % | True | True | NO |
| regression2-ab-abab | NbCl5 | 31.556 | 31.575 | 1.0006 | 1.0001–1.0025 | 0.16 % | True | True | yes |
| regression2-ab-abba | NaCl | 3.629 | 3.631 | 1.0004 | 0.9973–1.0036 | 0.16 % | True | True | yes |
| regression2-ab-abba | NbCl5 | 31.327 | 31.332 | 1.0001 | 0.9968–1.0021 | 0.16 % | True | True | yes |

| Control | Side | Structure | load (s) | setup (s) | median (ms) |
| --- | --- | --- | ---: | ---: | ---: |
| regression2-control-1b | 1b | NaCl | 0.2 | 19.7 | 3.499 |
| regression2-control-1b | 1b | NbCl5 | 0.2 | 19.0 | 30.863 |
| regression2-control-2 | 2 | NaCl | 0.2 | 19.7 | 3.485 |
| regression2-control-2 | 2 | NbCl5 | 0.2 | 19.4 | 30.751 |

| Forces | Structure | Side | max abs (eV/Å) | pass |
| --- | --- | --- | ---: | --- |
| regression2-ab-2-first | NaCl | 2 | 2.83e-07 | True |
| regression2-ab-2-first | NaCl | 1b | 3.36e-07 | True |
| regression2-ab-2-first | NbCl5 | 2 | 1.49e-06 | True |
| regression2-ab-2-first | NbCl5 | 1b | 1.41e-06 | True |
| regression2-ab-2-first | NaCl | between versions | 2.84e-07 | None |
| regression2-ab-2-first | NbCl5 | between versions | 1.39e-06 | None |
| regression2-ab-abab | NaCl | 1b | 2.34e-07 | True |
| regression2-ab-abab | NaCl | 2 | 2.52e-07 | True |
| regression2-ab-abab | NbCl5 | 1b | 1.31e-06 | True |
| regression2-ab-abab | NbCl5 | 2 | 1.49e-06 | True |
| regression2-ab-abab | NaCl | between versions | 2.08e-07 | None |
| regression2-ab-abab | NbCl5 | between versions | 1.37e-06 | None |
| regression2-ab-abba | NaCl | 1b | 2.69e-07 | True |
| regression2-ab-abba | NaCl | 2 | 2.88e-07 | True |
| regression2-ab-abba | NbCl5 | 1b | 1.35e-06 | True |
| regression2-ab-abba | NbCl5 | 2 | 1.19e-06 | True |
| regression2-ab-abba | NaCl | between versions | 2.77e-07 | None |
| regression2-ab-abba | NbCl5 | between versions | 1.37e-06 | None |

Trajectories, cueq: {"backend": "cueq", "release": {"side": "1b", "commit": "673a8a14cdb6ef869c7b186d82201ff182669cf5"}, "current": {"side": "2", "commit": "37c4014d1c568613dd801fecb20d54ff7a11fcc5"}, "plans_equal": {"20261019": true}, "positions_max_A": {"20261019": 3.2862601528904634e-13}, "repeat_positions_max_A": {"20261019": 3.863576125695545e-14}, "limit_A": 1e-12, "passed": true}

Trajectories, hybrid: {"backend": "hybrid", "release": {"side": "1b", "commit": "673a8a14cdb6ef869c7b186d82201ff182669cf5"}, "current": {"side": "2", "commit": "37c4014d1c568613dd801fecb20d54ff7a11fcc5"}, "plans_equal": {"20261001": true, "20261101": true}, "between_seeds": {"release": {"temperature_K": 14.55004009597178, "energy_meV_per_atom": 1.8393500038875743, "bins_meV_per_atom_max": 11.711956174284666, "bins_meV_per_atom_rms": 4.999391448688633}, "current": {"temperature_K": 14.550029404945406, "energy_meV_per_atom": 1.8393479740170804, "bins_meV_per_atom_max": 11.7119053999577, "bins_meV_per_atom_rms": 4.999374712718406}}, "across_versions": {"20261001": {"temperature_K": 1.3953514724107663e-05, "energy_meV_per_atom": 1.9647465876460046e-06, "bins_meV_per_atom_max": 3.421748129284197e-05, "bins_meV_per_atom_rms": 1.334608289874357e-05}, "20261101": {"temperature_K": 3.262488348809711e-06, "energy_meV_per_atom": 6.512390626767228e-08, "bins_meV_per_atom_max": 1.655684567225535e-05, "bins_meV_per_atom_rms": 7.09612453673617e-06}}, "limits": {"temperature_K": 29.10008019194356, "energy_meV_per_atom": 3.6787000077751486, "bins_meV_per_atom_rms": 9.998782897377266}, "passed": true}
