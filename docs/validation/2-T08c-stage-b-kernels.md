# 2-T08c: stage-B kernel count against the research engine on murphy

Plan task 2-T08c (conductor, 20 October 2026). The question: why does
`tests/gpu/test_engine_gpu.py::test_mh1_plain_cueq_float64_against_the_research_engine`
fail on murphy? Phase 2 acceptance needs it green on both machines.

**Cause: cuEquivariance's run-time kernel choice on a shared GPU, which the
test's normalisation did not cover. There is no difference between FastMACE
and the research engine.**
- Under a GPU load, cuEquivariance picks a transposed-operand variant for one
  forward segmented polynomial (`segmented_polynomial_tr0_1__algo…`).
- That variant first groups the edge indices, adding 12 graph nodes.
- `kernel_sequence` normalised neither the `tr…__` name nor the index
  grouping.
- Either engine can pick the variant. The test failed when one did and the
  other did not, or when both did with different algorithm suffixes.
- Fix (test only, in `tests/gpu/test_engine_gpu.py`): `kernel_sequence`
  normalises the transposed variant and leaves out its index grouping. No
  library code was changed.

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

## The diagnostic on murphy (20 October 2026, 17:55–18:03 UTC)

Conditions:
- murphy, NVIDIA RTX 4000 Ada Generation (sm_89);
- NAS `mace_cueq_latest` (mace-torch 0.3.16, torch 2.14.1+cu130,
  cuequivariance 0.12.0);
- library 85360cc, research engine `FASTMACE_BASELINE=/mnt/homes/zwingel/fastmace-baseline/1a-T18`
  (tag `research-engine-baseline-20261017`, d2faa9fd);
- probe `tools/validation/stage_b_kernel_diff.py`, checked mode, 100 steps;
- the GPU had no compute process before each run (`nvidia-smi`);
- no hash-gate problem (`patch_problems` empty).

| Run | Condition | Stage B: FastMACE / baseline | Normalised length | Equal (old normalisation) |
| --- | --- | ---: | ---: | --- |
| i | idle GPU, fresh compile caches | 234 / 236 | 234 / 234 | yes |
| ii | idle GPU, the same caches again (warm) | 236 / 235 | 234 / 234 | yes |
| iii | warm caches, beside a GPU load of ours (`--load-gib 4`: 4 GiB held, continuous 4096² matmuls) | 249 / 248 | 246 / 246 | **no** |

What run iii added, in both engines, compared with run i:
- one forward polynomial became `segmented_polynomial_tr0_1__algo2_2`
  (FastMACE) or `..._algo1_4` (baseline);
- before it, 3 `CU_GRAPH_NODE_TYPE_MEM_ALLOC` nodes, 4
  `kernelcatcher::equivariance::index_grouping` kernels
  (`segmentOffsetsKernel`, `gatherStridedLanesKernel`, `iotaWithinLaneKernel`,
  `computeIndptrBatchedKernel`) and 2 cub `DeviceSegmentedRadixSortKernel`
  launches; after it, 3 `MEM_FREE` nodes. That makes the 12 extra nodes;
- the other polynomials switched among `algo…` and `split…_of_…` variants,
  which the old normalisation already handled.

With the old normalisation the only remaining difference in run iii was that
one name (`tr0_1__algo2_2` vs `tr0_1__algo1_4`). The earlier failures (248/236
and 249/237) were the same shift in FastMACE alone: the GPU was shared with
sibling processes, and FastMACE's process chose the transposed variant while
the baseline's did not.

Cold vs warm compile caches change nothing (runs i and ii). `LD_LIBRARY_PATH`
was not set in any of the three runs.

The new normalisation, checked against every pairing of the six recorded
stage-B graphs (runs i, ii and iii; FastMACE and baseline; 15 pairs), finds
all of them equal. The old one fails exactly the 9 pairs that include a run-iii
graph.

Not covered: the test's other assertions under a load, above all the 1e-12 Å
trajectory bound (a different variant changes the summation order). The
acceptance tier runs this test on an idle GPU (2-T23's idle check), where it
passes on both machines.

Raw data: `results/2-T08c-murphy-20261020/diag-{i,ii,iii}.json` (versions, hash gates, both
engines' captured graphs with every node name, and the comparison).
