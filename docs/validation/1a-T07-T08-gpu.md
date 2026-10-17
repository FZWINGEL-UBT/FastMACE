# GPU confirmation of the merged MH-1 and generic rewrites (1a-T07, 1a-T08, 1a-T07b)

18 October 2026. The conductor promised this confirmation in `plans/phase-1a.md`
("Conductor decisions after T07/T08"). With it, T07 and T08 are confirmed on the GPU.

## What ran

| Run | Machine | Tree | Result |
| --- | --- | --- | --- |
| Full GPU suite, one process, `-p no:randomly` | murphy, RTX 4000 Ada, driver 595.58.03, NAS `mace_cueq_latest` | main `944de8e` (T07+T08 merged) | 173 passed, 9 failed, 2 skipped |
| Same | murphy | main `8258ebb` (review fixes) | 173 passed, 9 failed, 2 skipped |
| Same | RTX 5060 Ti (WSL), `mace_cueq_latest` | main `8258ebb` | 173 passed, 9 failed, 2 skipped |
| Same | RTX 5060 Ti (WSL) | trial merge of `8258ebb` + `fastmace/1a-T07b` + `fastmace/1a-T12` (merged as the next main) | **187 passed, 2 skipped** |

Environment for all runs: `FASTMACE_MODELS` (real checkpoints), `FASTMACE_BASELINE`
(`research-engine-baseline-20261017`), and `FASTMACE_HYBRID_PYTHON` pointing at
`mace_cueq_oeq` for the hybrid child processes. The 2 skips are OEq-only kernel tests that
need openequivariance in the parent interpreter.

## The 9 failures and their cause (1a-T07b)

These were the MH-1 package on cuEq GPU models and the linear GPU tests. They failed only
when the whole GPU suite ran in one process, and passed alone. The cause was a real leak:
`loading.load(backend="cueq")` switched MACE's fused cuEq convolution on and left it on, so
every later plain cuEq conversion in the process was fused, and deep copies of fused models
lose MACE's adapter. The trigger was the module-scoped POLAR-1-S fixture in
`test_polar_gpu.py`, which runs before the per-test restore fixture takes its snapshot. The
fix in `_patches.cueq_conv_fusion` (a context manager) and `loading.load` applies the rule
inside each load only. It would have affected T11's force fields as well.

## Agreement numbers

**Generic rewrites with T07's real modules** (murphy, `8258ebb`, from
`tests/gpu/test_rewrites_generic_gpu.py`, rewritten against unrewritten, max relative error):

| Model | Backend | Precision | Energy | Forces | Stress |
| --- | --- | --- | --- | --- | --- |
| mp0-medium | cueq | float64 | 0 | 2.9e-15 | 9.6e-16 |
| mp0-medium | hybrid | float64 | 1.3e-16 | 1.9e-15 | 4.8e-15 |
| mp0-medium | hybrid | float32 | 0 | 8.1e-7 | 1.4e-6 |
| off23-small | cueq | float64 | 0 | 6.9e-16 | — |
| off23-small | hybrid | float64 | 0 | 5.2e-16 | — |
| off23-small | hybrid | float32 | 0 | 2.6e-7 | — |

These replace the stand-in numbers of `1a-T08-generic-rewrites.md` and agree with them.
MACE-OFF's first block is correctly left without reassociation.

**MH-1 package** (`tests/gpu/test_rewrites_mh1_gpu.py`, included in the 187 passed): the
stacked MH-1 rewrites on tiny cuEq models in float32/float64 and on MH-1 (`matpes_r2scan`)
plain cuEq float64 on NaCl 2×2×2 pass at the §6.4 thresholds. T07 reported 1.3e-16 relative
on energy and 1.4e-15 on forces for that model, and 7.1e-7 on forces for the hybrid with
streaming.
