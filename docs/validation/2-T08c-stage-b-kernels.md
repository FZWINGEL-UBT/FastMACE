# 2-T08c: stage-B kernel count against the research engine on murphy

Plan task 2-T08c (conductor, 20 October 2026). The question: why does
`tests/gpu/test_engine_gpu.py::test_mh1_plain_cueq_float64_against_the_research_engine`
fail on murphy? Phase 2 acceptance needs it green on both machines.

Status: **analysed from recorded data only.** The kernel diff on murphy's GPU
is on hold: the owner wants no big accuracy runs on murphy until the
optimisation round is in. It runs with the post-optimisation acceptance if it
is still needed. No library code was changed.

## What the test compares

- The baseline is not a recorded number. `baseline_run` starts a fresh
  interpreter with the research engine (`FASTMACE_BASELINE`, the 1a-T18
  export of tag `research-engine-baseline-20261017`) on the same GPU,
  environment and structure, right after FastMACE's child.
- Each child counts the nodes of every captured graph (`CountingGraph`) and
  records the kernel names.
- The assertion compares `kernel_sequence` of the two name lists, not the
  raw counts. That function maps cuEquivariance's run-time variants of each
  segmented-polynomial kernel (algorithm, split into launches) to one name.
  The counts are only reported.

## Recorded results (MH-1 plain cuEq float64, NaCl 3×3×3, edge bucket 7168)

| Run | Machine | Library | Stage B: FastMACE / baseline | Sequences | GPU | Compile caches |
| --- | --- | --- | ---: | --- | --- | --- |
| 1a-T13, 18 Oct | leto | 1a | 236 / 234 | equal | — | — |
| 1a-T13, 18 Oct | local RTX 5060 Ti | 1a | 240 / 238 | equal | — | — |
| 1b-T03b, 19 Oct | murphy | ≈9cff042 | 235 / 236 | equal | timed-style run | shared, warm (`fastmace-cache/`) |
| 2-T08b validation, 20 Oct | murphy | 14a5e20 (`main`) | 248 / 236 | **differ** | shared with 2 sibling processes | fresh per-task directory |
| 2-T08b validation, 20 Oct | murphy | a3ecb2f (2-T08b) | 249 / 237 | **differ** | shared with 2 sibling processes | per-task directory |
| 2-T23 acceptance, 20 Oct | local RTX 5060 Ti | 85360cc | 238 / 239 | equal | reserved | acceptance cache |
| 2-T23 acceptance, 20 Oct | murphy | 4b35b60 | 235 / 235 | equal | idle (manifest: no compute process) | shared, warm (`fastmace-cache/acceptance-murphy`) |

Sources:
- `results/1a-T13-engine-20261018/` and `results/1b-T03b-murphy/reports/`;
- the 2-T08b runs: `docs/validation/2-T08-isolation.md` (their logs were
  deleted with the task copy);
- 2-T23's run folders: `~/fastmace-acceptance/2-T23-local-20261020` and
  `/mnt/homes/zwingel/fastmace-acceptance/2-T23-murphy-20261020`, with
  `acceptance-tier/results/1a-T13-mh1-cueq-float64-nvt.json` and the
  manifest.

## Findings

1. **The failure is conditional, not fixed.** 4b35b60 contains 2-T08b
   (a3ecb2f), which gave 249. Between a3ecb2f and 4b35b60 nothing on this
   test's path changes what it builds:
   - `GraphEngine.prepare()` and a capture-error tag, neither of which the
     test calls;
   - `torch.are_deterministic_algorithms_enabled()` in the compile-cache key,
     False here;
   - `deterministic=` fields on rewrite records;
   - a reproducibility claim in the process record.

   So the same build gave 249 in one run and 235 in another, both on murphy.
2. **It shifts FastMACE only, by the same amount both times:** +12 to +13
   nodes in stage B, while the baseline, run in the same conditions right
   after, stayed at 236–237.
3. **The environment is not the cause.**
   - Every file holding a source that FastMACE hash-checks before patching
     is byte-identical in murphy's NAS `mace_cueq_latest` and the local
     environment: `mace/cli/convert_e3nn_cueq.py`, `mace/calculators/mace.py`,
     `mace/modules/{utils,blocks,irreps_tools,models,wrapper_ops}.py` and
     `torch/_inductor/fx_passes/post_grad.py` (sha256, 20 October).
   - So no hash gate (2-T04) drops a patch on murphy.
   - mace-torch 0.3.16, torch 2.14.1+cu130 and triton 3.8.0 match, and the
     three cuEquivariance packages have identical dist-info records.
4. **Ruled out by reading the code:**
   - the conv-fusion default (`cueq_conv_fusion=True`, fused on any CUDA
     device, unchanged);
   - 8b51c8c's device scope (it acts only on an unhashed converter);
   - the addmm rule (it tests `is_cuda`, not the architecture);
   - the allocator setting;
   - the planner's, gate's and reproducible mode's device-dependent code
     (the test builds `MACEForceField` and `GraphEngine` directly);
   - 2-T13's dynamic edges (off by default).
5. **The conditions that differ between pass and fail on murphy:**
   - the passing runs had an idle GPU and warm shared compile caches;
   - the failing runs shared the GPU with two sibling processes and used
     fresh per-task cache directories;
   - 1b-T03b also set `LD_LIBRARY_PATH` to the cu13 and `cuequivariance_ops`
     libraries, while the failing runs and 2-T23's did not, so that one is
     unlikely to decide it.

## Candidates, and the run that separates them

- **(a) A shared GPU changes a run-time kernel choice:**
  - cuEquivariance's choice of variant or split per segmented polynomial, if
    its variant names fall outside `kernel_sequence`'s pattern;
  - or cuBLAS's algorithm or split-K under memory pressure.
- **(b) A cold Inductor/Triton cache compiles a different stage-B graph than
  a warm one.**

The probe `tools/validation/stage_b_kernel_diff.py` runs both children once,
in checked mode. It first prints the package versions and every hash-gate
problem (`--environment-only`), then for every captured graph the counts and
a `difflib` diff of the normalised kernel names. Run on current `main` on
murphy:
1. idle GPU, fresh cache directories;
2. the same directories again (warm);
3. `--load-gib 4`, beside a GPU load of our own.

The names then decide it:
- cuBLAS split-K or xmma kernels, or segmented-polynomial names outside the
  pattern: (a);
- `triton_*` kernels that only the cold run has: (b).

If only run 3 fails, the fix belongs in the test: run this comparison only on
an idle GPU, as the timing jobs already require, or extend `kernel_sequence`
to the variant names the diff shows. If run 1 fails, the test needs a warm
cache or a normalisation of the cold compile.

## Until then

The test passes on both machines in the phase 2 acceptance tier when run as
2-T23 runs it: an idle GPU and the acceptance compile cache.
