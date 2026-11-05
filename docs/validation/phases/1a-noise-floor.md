# Phase 1a: noise floor of the A/B step-time criterion

Phase 1a's acceptance needs FastMACE's MD step time within 3 % of the
frozen research engine, measured in alternating A/B blocks in one process
(SPEC §12.2 Performance row, §12.5). FastMACE has no engine yet (plan task
1a-T13), so plan task 1a-T18 measured the research engine against itself
with the A/B tool. The resulting spread is the noise floor of the 3 %
criterion.

## Result

The ratio of two identical engines in one process stays within **0.32 %**
of 1 for the SPEC protocol (4 blocks of 300 steps per side), and within
**0.16 %** for 12 blocks per side, on all three crystals. This is about ten
times smaller than the 3 % cross-process noise quoted in SPEC §12.2. The
3 % criterion is therefore limited by real differences between the engines,
not by measurement noise. One process was enough: the two engines did not
interfere with each other, so the fallback to alternating processes was not
needed.

| Crystal (3×3×3) | Atoms | Step time (ms) | Block spread (max−min)/median | Block stdev | B/A, 12 blocks | B/A, 4-block windows | A median across processes |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| NaCl | 216 | 3.565–3.636 | 6.8–7.8 % | 2.0–2.4 % | 0.9985–0.9995 | 0.9968–0.9997 | 1.81 % |
| NaNbO3 | 1080 | 25.99–26.13 | 0.9–1.7 % | 0.29–0.53 % | 1.0007–1.0016 | 1.0007–1.0027 | 0.39 % |
| NbCl5 | 1944 | 31.43–31.46 | 3.4–3.8 % | 0.88–1.03 % | 0.9992–0.9997 | 0.9991–1.0008 | 0.09 % |

Columns: step times are the per-side medians of all three processes;
the spread and stdev are block to block within one side of one process;
"B/A, 12 blocks" is the ratio of the side medians per process (three
processes); "4-block windows" splits each process's 12 block pairs into
three consecutive windows of 4 pairs, the protocol of SPEC §12.5 (nine
estimates per crystal); the last column is the spread of side A's median
over the three processes.

Earlier figures on the same machine: 3.78/25.9/32.4 ms per step for
NaCl/NaNbO3/NbCl5 (ResearchModule `studies/hybrid_oeq/murphy.md`, 15
October 2026, an earlier revision with its own protocol).

## What the numbers show

- **Blocks drift, ratios do not.** NaCl's first three blocks take
  3.43–3.44 ms and the later ones 3.52–3.67 ms. The systems start at 0 K
  and heat up to 300 K during the run; more neighbour-list rebuilds follow
  (155 per side on NaCl). This drift dominates NaCl's 7 % block spread. Both
  sides follow the same trajectory and drift together, so the per-pair
  ratios stay between 0.996 and 1.003. Alternation removes the drift; a
  block-by-block comparison of separate runs would not.
- **ABAB and ABBA agree.** Process 2 mirrored every second pair (ABBA).
  Its ratios lie inside the range of the two ABAB processes, so block order
  causes no measurable bias here.
- **A small consistent offset on NaNbO3.** The second engine is
  0.07–0.16 % slower on NaNbO3 in all three processes, with both patterns.
  So it is tied to the engine, not the order: the second engine is built
  second and has its own memory placement. At 0.2 % it is far below 3 %.
  T19 should still swap which engine is built first in one of its repeats.
- **Across processes** side A's median moves by up to 1.8 % (NaCl), 0.4 %
  (NaNbO3) and 0.1 % (NbCl5). With only three processes this is only a
  rough estimate. It agrees with the SPEC's warning that small effects need
  one process.

## Interference between two engines in one process

The two sides each load their own calculator, so they never share a
compiled model (the research engine caches compiled models by
`id(model)`, `gpu_langevin._COMPILED_MODELS`). That is the situation of a
FastMACE-against-baseline run. Before each timed block the tool calls
`MACEForces.bind_streaming_count()`, so the stepped field owns the
streaming interactions' edge counter. Checks, recorded per crystal in the
JSON:

- **Same trajectory.** Both sides start from the same structure with seed 1.
  After 300 + 12 × 300 steps their temperatures differ by at most 3.1e-4 K
  and their potential energies by at most 2.8e-5 eV/atom (NbCl5), the
  divergence that FP32 round-off produces in a chaotic trajectory (the
  20-step smoke run differed by 1.8e-7 K and 0 eV). Both sides made the same
  number of neighbour rebuilds (NaCl 155, NaNbO3 207, NbCl5 150) and used
  the same edge sizes. A side that read the other's edge counter would get
  wrong forces and diverge at once.
- **No extra compilation.** Dynamo's counters are the same in every
  process (6, 10 and 12 unique graphs after NaCl, NaNbO3 and NbCl5, that is
  two per side and crystal; no recompiles counter).
- **Nothing else on the GPU.** `nvidia-smi` listed no other compute
  process before each run or after each crystal.

Points for the FastMACE-against-baseline run (T19):

- The research engine's `upstream_mace` patches MACE process-wide
  (single-graph strain, cuEq convolution fusion, first-derivative forces).
  FastMACE's `_patches` does similar things. For NVT forces these patches
  are exact rewrites, but T19 must check that each engine still gets its own
  path when both are loaded in one process. The test: FastMACE's forces in
  the A/B process must equal those of a FastMACE-only process.
- Both engines stay alive during the timing, so `peak_memory_bytes` in the
  JSON is the peak of the process (both engines together), not of one side.
  Measure memory per engine in a single-engine process (`--sides baseline`).
- Setup time: the side built first includes the per-process OEq JIT and
  the first compilation (17–43 s against 10–28 s for the second side). Setup
  times from an A/B process are not a fair comparison of the engines.

## Configuration

| Item | Value |
| --- | --- |
| Machine | murphy, NVIDIA RTX 4000 Ada Generation 20 GB, driver 595.58.03 |
| Environment | NAS `mace_cueq_oeq` (torch 2.14.1+cu130), launched through the export's `studies/hybrid_oeq/scripts/oeq_no_nvcc.sh` |
| Backend, precision | cuEq/OEq hybrid, float32, TF32 off, `torch.compile` default |
| Model | `mh1-r2scan` (MH-1, head `matpes_r2scan`), `mace-mh-1.model`, SHA-256 `a522eb7f…be47` |
| Engine on both sides | research engine `GPULangevin` with `main.py`'s gpu default for MH-1 (every optimisation it supports, pipelined; `species_skip` off as not supported), NVT 300 K, ASE-Langevin settings of `main.py` |
| Crystals | NaCl, NaNbO3, NbCl5 r²SCAN benchmark CIFs, 3×3×3 |
| Protocol | 300 warm-up steps per side, then 12 timed blocks of 300 steps per side, alternating |
| Processes | p1 ABAB, p2 ABBA, p3 ABAB, one after another (17:25–17:58 UTC, 17 October 2026) |
| FastMACE commit | `4d04b5c` (branch `fastmace/1a-T18`) |
| Baseline | ResearchModule tag `research-engine-baseline-20261017`, commit `d2faa9fd`, export `/mnt/homes/zwingel/fastmace-baseline/1a-T18` |

The export at `fastmace-baseline/20261017` holds only `langevin/`. The A/B
tool also needs the launcher script, the benchmark scripts
(`common.py`, `model_store.py`) and the benchmark structures. A fuller
export of the same tag was therefore written to `fastmace-baseline/1a-T18`:
`Experiments/Langevin_Thermostat/langevin`,
`Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts`,
`Experiments/Benchmark_MACE_GPU_OPTIMIZATIONS/scripts` and
`Structures/Benchmark/R2SCAN`, with `COMMIT`.

## Reproducing

From a FastMACE checkout on the NAS (`$FM`), with the export `$B`:

```bash
ENV=/mnt/homes/zwingel/environments/mace_cueq_oeq
PYTHONPATH=$FM PYTHONDONTWRITEBYTECODE=1 bash $B/Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh \
  $ENV /mnt/homes/zwingel/fastmace-cache/<name>/oeq-cache \
  $ENV/bin/python $FM/tools/ab_step_time.py --output <fresh directory> --baseline $B \
  --checkpoint /mnt/homes/zwingel/ResearchModule/Models/mace-mh-1.model --model mh1-r2scan \
  --sides baseline baseline --structures NaCl NaNbO3 NbCl5 --blocks 12 --pattern abab
```

Run it detached (`setsid nohup … &`) on an idle GPU. One process takes
about 11 minutes.

## Data

`results/1a-T18-noise-floor-20261017/`: `nf-p1-abab/`, `nf-p2-abba/` and
`nf-p3-abab/metadata.json`, one per process, with every block time,
observables, engine info, dynamo counters and provenance;
`summary.json` has the derived statistics of the table above.
