## cases
('mh1-r2scan', 'NaCl', '3x3x3') natoms 216 baseline ratios {'energy_meV_per_atom': 0.987, 'force_eV_per_A': 1.03, 'stress_eV_per_A3': 1.006} repeatable True yard F 3.48e-06 peakGiB [1.37, 1.28, 1.31] rows 288
('mh1-r2scan', 'NbCl5', '3x3x3') natoms 1944 baseline ratios {'energy_meV_per_atom': 1.009, 'force_eV_per_A': 0.997, 'stress_eV_per_A3': 0.965} repeatable True yard F 8.85e-05 peakGiB [11.43, 10.44, 10.68] rows 288
('mh1-r2scan', 'NaNbO3', '3x3x3') natoms 1080 baseline ratios {'energy_meV_per_atom': 1.006, 'force_eV_per_A': 1.0, 'stress_eV_per_A3': 1.033} repeatable True yard F 1.26e-04 peakGiB [8.8, 8.96] rows 168
('omat0-medium', 'NaCl', '3x3x3') natoms 216 baseline ratios {'energy_meV_per_atom': 0.849, 'force_eV_per_A': 0.93, 'stress_eV_per_A3': 0.974} repeatable True yard F 4.27e-06 peakGiB [0.75, 0.74, 0.82] rows 240
('omat0-medium', 'NbCl5', '3x3x3') natoms 1944 baseline ratios {'energy_meV_per_atom': 0.971, 'force_eV_per_A': 0.993, 'stress_eV_per_A3': 0.97} repeatable True yard F 8.91e-05 peakGiB [5.26, 4.85, 5.82] rows 240
('mh1-r2scan', 'NaNbO3', '2x2x2') natoms 320 baseline ratios {'energy_meV_per_atom': 1.028, 'force_eV_per_A': 0.995, 'stress_eV_per_A3': 0.923} repeatable True yard F 4.96e-05 peakGiB [3.12] rows 120

## mh1-r2scan: worst force ratio over NaCl, NbCl5, NaNbO3, NaNbO3; * = gate passes on all
| stage | out-bf16 | out-fp16 | grad-bf16 | grad-fp16 | saved-bf16 | saved-fp16 | tf32-fwd | tf32-bwd | autocast-bf16 | autocast-fp16 | cast-bf16 | cast-fp16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| radial-basis | 7.97e+03 | 1.05e+03 | 557 | 73.8 | 1.42e+03 | 171 | = | = | = | = | 3.08e+04 | 3.27e+03 |
| spherical-harmonics | 207 | 27.1 | 20.7 | 3.19 | 90 | 14.4 | = | = | = | = | 566 | 75.3 |
| node-embedding | 130 | 5.44 | = | = | = | = | 21.3 | **1.06** | = | = | 101 | 22.3 |
| zbl | **1.03** | **1.03** | 3.56 | **1.03** | 6.41 | **1.24** | = | = | = | = | 315 | inf |
| i0-radial-mlp | 176 | 23 | 266 | 30.7 | 619 | 79.1 | 599 | 44.7 | 5.55e+03 | 626 | 6.24e+03 | 645 |
| i0-species-embeddings | 47 | 30.7 | = | = | = | = | 29.1 | **1.06** | = | = | 277 | 8.24 |
| i0-linears | 305 | 36.6 | 219 | 28.3 | 22.4 | 3.02 | 51.2 | 23.5 | = | = | 708 | 91.4 |
| i0-conv-tp | 146 | 21.9 | 199 | 23.8 | 41.3 | 4.02 | = | = | 335 | 45.7 | 335 | 45.7 |
| i0-scatter | 364 | 42.9 | 238 | 24.9 | = | = | = | = | = | = | 1.26e+03 | 222 |
| i0-gate | 153 | 19.4 | 97.4 | 11.5 | 54.9 | 6.1 | = | = | 27.7 | 3.61 | 366 | 39.5 |
| i0-skip-residual | 44.4 | 5.42 | = | = | = | = | 10.8 | **1.06** | = | = | 60.8 | 28.9 |
| i0-symmetric-contraction | 241 | 27.7 | 143 | 13.8 | 112 | 14.4 | 3.45 | **1.25** | 429 | 70.5 | 427 | 74 |
| i0-product-linear | 81.1 | 8.18 | 109 | 14.5 | 25.8 | 1.94 | 23.1 | 16.4 | = | = | 377 | 48.8 |
| i0-readout | 1.03 | 1.03 | 43.8 | 9.99 | 45.6 | 4.78 | 1.03 | 34 | = | = | 71.2 | 27.3 |
| i1-radial-mlp | 117 | 16.9 | 128 | 18.2 | 472 | 70.4 | 476 | 25 | 3.32e+03 | 480 | 3.32e+03 | 464 |
| i1-species-embeddings | 183 | 3.48 | = | = | = | = | 14.7 | **1.06** | = | = | 42.3 | 20.4 |
| i1-linears | 276 | 27.6 | 166 | 19.9 | 34.2 | 5.92 | 35.2 | 17.3 | = | = | 522 | 64.6 |
| i1-conv-tp | 140 | 17.5 | 150 | 16.8 | 89.5 | 12.1 | 2.13 | 1.95 | 307 | 44.9 | 343 | 47 |
| i1-scatter | 220 | 28.2 | 111 | 14.4 | = | = | = | = | = | = | 824 | 94.9 |
| i1-gate | 119 | 12.1 | 77.6 | 8.06 | 56.6 | 7.83 | = | = | 44.3 | 5.67 | 230 | 25.2 |
| i1-skip-residual | 134 | 15.8 | 117 | 17.1 | 6.85 | 2.96 | 27.3 | 6.16 | = | = | 314 | 42.3 |
| i1-symmetric-contraction | 18 | 2.26 | 67.8 | 7.72 | 64.5 | 9.19 | 4.04 | 1.5 | 129 | 17.1 | 151 | 20.8 |
| i1-product-linear | 16.9 | 2.46 | 56.7 | 8.18 | 8.65 | 1.87 | *2.6* | 5.46 | = | = | 97.9 | 10.7 |
| i1-readout | 1.03 | 1.03 | 44 | 5.26 | 90.9 | 11.4 | 3.07 | 24.3 | = | = | err | err |

passing (stage, mode, worst ratios E/F/S):
  node-embedding tf32-bwd {'energy_meV_per_atom': 1.01, 'force_eV_per_A': 1.06, 'stress_eV_per_A3': 1.07} ratio<=2
  zbl out-bf16 {'energy_meV_per_atom': 1.02, 'force_eV_per_A': 1.03, 'stress_eV_per_A3': 1.03} ratio<=2
  zbl out-fp16 {'energy_meV_per_atom': 1.01, 'force_eV_per_A': 1.03, 'stress_eV_per_A3': 1.03} ratio<=2
  zbl grad-fp16 {'energy_meV_per_atom': 1.01, 'force_eV_per_A': 1.03, 'stress_eV_per_A3': 1.09} ratio<=2
  zbl saved-fp16 {'energy_meV_per_atom': 1.01, 'force_eV_per_A': 1.24, 'stress_eV_per_A3': 1.84} ratio<=2
  i0-species-embeddings tf32-bwd {'energy_meV_per_atom': 1.01, 'force_eV_per_A': 1.06, 'stress_eV_per_A3': 1.07} ratio<=2
  i0-skip-residual tf32-bwd {'energy_meV_per_atom': 1.01, 'force_eV_per_A': 1.06, 'stress_eV_per_A3': 1.07} ratio<=2
  i0-symmetric-contraction tf32-bwd {'energy_meV_per_atom': 1.01, 'force_eV_per_A': 1.25, 'stress_eV_per_A3': 0.89} ratio<=2
  i1-species-embeddings tf32-bwd {'energy_meV_per_atom': 1.03, 'force_eV_per_A': 1.06, 'stress_eV_per_A3': 1.02} ratio<=2
  i1-product-linear tf32-fwd {'energy_meV_per_atom': 6.35, 'force_eV_per_A': 2.6, 'stress_eV_per_A3': 1.2} floor only
ERR i1-readout cast-bf16 RuntimeError: The following operation failed in the TorchScript interpreter.
Traceback of TorchScript, serialized code (
ERR i1-readout cast-bf16 RuntimeError: The following operation failed in the TorchScript interpreter.
Traceback of TorchScript, serialized code (
ERR i1-readout cast-bf16 RuntimeError: The following operation failed in the TorchScript interpreter.
Traceback of TorchScript, serialized code (
ERR i1-readout cast-fp16 RuntimeError: The following operation failed in the TorchScript interpreter.
Traceback of TorchScript, serialized code (
ERR i1-readout cast-fp16 RuntimeError: The following operation failed in the TorchScript interpreter.
Traceback of TorchScript, serialized code (
ERR i1-readout cast-fp16 RuntimeError: The following operation failed in the TorchScript interpreter.
Traceback of TorchScript, serialized code (

## omat0-medium: worst force ratio over NaCl, NbCl5; * = gate passes on all
| stage | out-bf16 | out-fp16 | grad-bf16 | grad-fp16 | saved-bf16 | saved-fp16 | tf32-fwd | tf32-bwd | autocast-bf16 | autocast-fp16 | cast-bf16 | cast-fp16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| radial-basis | 5.82e+03 | 715 | 387 | 58.7 | 1.71e+03 | 215 | = | err | = | = | 2.58e+11 | 1.33e+05 |
| spherical-harmonics | 169 | 22.1 | 24.7 | 3.42 | 129 | 18 | = | = | = | = | 682 | 91 |
| node-embedding | 48.5 | 4.05 | = | = | = | = | 19.2 | **1.03** | = | = | 81.2 | 16.4 |
| zbl | **0.993** | **0.993** | **1.2** | **0.998** | *2.18* | **1.11** | = | = | = | = | 257 | 47.4 |
| i0-radial-mlp | 491 | 66.6 | 368 | 36.4 | 274 | 35.5 | 360 | 65.8 | 3.18e+03 | 453 | 3.05e+03 | 436 |
| i0-linears | 284 | 35.7 | 199 | 26.5 | 14.4 | 5.36 | 141 | 30.3 | = | = | 1.57e+03 | 194 |
| i0-conv-tp | 445 | 54.3 | 213 | 26.2 | 53.2 | 8.83 | = | = | 837 | 96.2 | 837 | 96.2 |
| i0-scatter | 748 | 102 | 253 | 31.5 | = | = | = | = | = | = | 3.53e+03 | 526 |
| i0-skip-residual | 362 | 46.9 | 226 | 26.4 | 190 | 2.84 | 32.2 | 13.4 | 539 | 59.5 | 581 | 76.4 |
| i0-symmetric-contraction | 306 | 28.8 | 108 | 12.6 | 112 | 14.2 | 32.3 | 3.92 | 639 | 85.7 | 529 | 96.3 |
| i0-product-linear | 186 | 26.9 | 121 | 15.6 | 15.6 | 2.26 | 33.8 | 10.4 | = | = | 518 | 62.1 |
| i0-readout | 0.993 | 0.993 | 10.8 | 3.15 | 16.4 | 5.38 | = | = | = | = | 40.8 | 32.5 |
| i1-radial-mlp | 180 | 23.2 | 574 | 74.6 | 524 | 54.4 | 318 | 82.2 | 4.19e+03 | 400 | 3.78e+03 | 407 |
| i1-linears | 446 | 46.7 | 143 | 19.3 | 17.9 | 6.03 | 43 | 26.7 | = | = | 605 | 98.8 |
| i1-conv-tp | 160 | 20.5 | 206 | 24.6 | 130 | 15.9 | 2.93 | 1.61 | 425 | 57.4 | 433 | 57.6 |
| i1-scatter | 288 | 30.4 | 142 | 19.9 | = | = | = | = | = | = | 996 | 116 |
| i1-skip-residual | 69 | 11 | 54.2 | 6.37 | 57.7 | 1.23 | 11.2 | 4.81 | 220 | 21.1 | 220 | 25.8 |
| i1-symmetric-contraction | 197 | 25.9 | 197 | 20.4 | 185 | 18.9 | 4 | 1.36 | 542 | 59.9 | 588 | 67.1 |
| i1-product-linear | 68.4 | 6.92 | 194 | 21.6 | 9.11 | 2.94 | 20.8 | 10.4 | = | = | 345 | 41.2 |
| i1-readout | 0.993 | 0.993 | 76.8 | 11 | 459 | 57.1 | 16.3 | 46.2 | = | = | 977 | 106 |

passing (stage, mode, worst ratios E/F/S):
  node-embedding tf32-bwd {'energy_meV_per_atom': 1.01, 'force_eV_per_A': 1.03, 'stress_eV_per_A3': 1.03} ratio<=2
  zbl out-bf16 {'energy_meV_per_atom': 1.02, 'force_eV_per_A': 0.99, 'stress_eV_per_A3': 0.97} ratio<=2
  zbl out-fp16 {'energy_meV_per_atom': 0.98, 'force_eV_per_A': 0.99, 'stress_eV_per_A3': 0.97} ratio<=2
  zbl grad-bf16 {'energy_meV_per_atom': 0.97, 'force_eV_per_A': 1.2, 'stress_eV_per_A3': 1.12} ratio<=2
  zbl grad-fp16 {'energy_meV_per_atom': 0.97, 'force_eV_per_A': 1.0, 'stress_eV_per_A3': 1.03} ratio<=2
  zbl saved-bf16 {'energy_meV_per_atom': 0.97, 'force_eV_per_A': 2.18, 'stress_eV_per_A3': 1.09} floor only
  zbl saved-fp16 {'energy_meV_per_atom': 0.97, 'force_eV_per_A': 1.11, 'stress_eV_per_A3': 1.33} ratio<=2
ERR radial-basis tf32-bwd AttributeError: 'NoneType' object has no attribute 'requires_grad'
ERR radial-basis tf32-bwd AttributeError: 'NoneType' object has no attribute 'requires_grad'
