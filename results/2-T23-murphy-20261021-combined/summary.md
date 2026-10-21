| Process | Structure | 1b (ms) | 2 (ms) | 2/1b | pair range | floor | plan ok | valid | pass |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| regression2-ab-2-first | NaCl | 3.635 | 3.486 | 0.9591 | 0.9563–0.9592 | 0.16 % | True | True | yes |
| regression2-ab-2-first | NbCl5 | 31.436 | 27.177 | 0.8645 | 0.8595–0.8647 | 0.16 % | True | True | yes |
| regression2-ab-abab | NaCl | 3.573 | 3.436 | 0.9616 | 0.9597–0.9675 | 0.16 % | True | True | yes |
| regression2-ab-abab | NbCl5 | 31.386 | 27.154 | 0.8651 | 0.8635–0.8654 | 0.16 % | True | True | yes |
| regression2-ab-abba | NaCl | 3.635 | 3.501 | 0.9632 | 0.9541–0.9657 | 0.16 % | True | True | yes |
| regression2-ab-abba | NbCl5 | 31.450 | 27.162 | 0.8637 | 0.8609–0.8656 | 0.16 % | True | True | yes |
| regression2-nacl-abab-1b-first | NaCl | 3.630 | 3.496 | 0.9633 | 0.9613–0.9668 | 0.16 % | True | True | yes |
| regression2-nacl-abab-2-first | NaCl | 3.628 | 3.479 | 0.9588 | 0.9571–0.9629 | 0.16 % | True | True | yes |
| regression2-nacl-abba-1b-first | NaCl | 3.623 | 3.490 | 0.9632 | 0.9614–0.9656 | 0.16 % | True | True | yes |

| Control | Side | Structure | load (s) | setup (s) | median (ms) |
| --- | --- | --- | ---: | ---: | ---: |
| regression2-control-1b | 1b | NaCl | 0.2 | 19.6 | 3.509 |
| regression2-control-1b | 1b | NbCl5 | 0.2 | 18.9 | 30.804 |
| regression2-control-2 | 2 | NaCl | 0.2 | 23.2 | 3.366 |
| regression2-control-2 | 2 | NbCl5 | 0.2 | 21.8 | 26.602 |

| Gate job | Model | E ratio | F ratio | S ratio | E max (meV/atom) | status | pass |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| gate-mp0-large | mp0-large | 1.54 | 1.10 | 1.22 | 0.0002 | complete | yes |
| gate-off23-medium | off23-medium | – | – | – | – | failed | NO |
| gate-off23-small | off23-small | 1.32 | 1.13 | 1.00 | 0.0003 | complete | yes |
| gate-polar1-l | polar1-l | 0.79 | 1.12 | 1.00 | 0.0039 | complete | yes |
| gate-polar1-l-NbCl5x4 | polar1-l | 1.43 | 0.94 | 1.12 | 0.0018 | complete | yes |
| gate-polar1-m | polar1-m | 1.36 | 1.47 | 1.48 | 0.4583 | complete | yes |
| gate-polar1-m-NbCl5x4 | polar1-m | 0.87 | 0.70 | 0.62 | 0.0004 | complete | yes |
| gate-polar1-s | polar1-s | 1.10 | 1.03 | 1.65 | 0.0004 | complete | yes |
| gate-polar1-s-NbCl5x4 | polar1-s | 1.04 | 1.94 | 1.50 | 0.0012 | complete | yes |

acceptance-tier: 33 passed, 0 failed, 0 errors, 0 skipped, 127.0 min; failed: []
