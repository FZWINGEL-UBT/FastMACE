# 1a-T05 model loading on leto (17 October 2026)

| Item | Value |
| --- | --- |
| Machine, GPU | leto, NVIDIA RTX 4000 Ada Generation 20 GB, driver 595.58.03, idle before each run |
| Environments | NAS `mace_cueq_latest` (plain cuEq) and `mace_cueq_oeq` (hybrid): torch 2.14.1+cu130, MACE 0.3.16, cuEquivariance 0.12.0, OEq 0.7.0 |
| FastMACE | branch `fastmace/1a-T05`, copied to `/mnt/homes/zwingel/fastmace-work/1a-T05` |
| Baseline | ResearchModule tag `research-engine-baseline-20261017` (d2faa9fd), exported to `/mnt/homes/zwingel/fastmace-baseline/1a-T05` |
| Checkpoints | `FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models` (set at run time) |
| Model | MH-1 `mace-mh-1.model` (SHA-256 a522eb7f…), head `matpes_r2scan`, label `mh1-r2scan` |
| Structure | NaCl rock salt a = 5.64 Å, cubic, 2×2×2 (64 atoms), rattled σ = 0.05 Å, seed 17 |

`forces_probe.py ROUTE BACKEND OUT`: `fastmace` is `fastmace.model.loading.load`;
`baseline` is `common.load_calculator` with `upstream_mace.use_cueq_conv_fusion`
and `require_fused_cueq`, as `workflow.load_md_calculator` calls them. Plain cuEq
runs float64 with the fused convolution; the hybrid float32. The FastMACE hybrid
ran without `oeq_no_nvcc.sh` and without `LD_LIBRARY_PATH` (in-process OEq
runtime, which needed the `CXX` wrapper on leto); the baseline hybrid ran under
the export's `oeq_no_nvcc.sh`. Each hybrid process rebuilt the OEq host extension
(load 24–27 s).

| Comparison (`comparison.json`) | max \|ΔF\| eV/Å | ΔE eV |
| --- | ---: | ---: |
| plain cuEq float64: FastMACE vs baseline | 4.8e-16 | 0 |
| hybrid float32: baseline run 1 vs run 2 (scatter) | 3.3e-7 | 0 |
| hybrid float32: FastMACE run 1 vs run 2 | 2.7e-7 | 0 |
| hybrid float32: FastMACE vs baseline (4 pairs) | 2.4e-7 – 3.0e-7 | 3.1e-5 (1 ulp at 440 eV) |
| hybrid float32 FastMACE vs plain cuEq float64 baseline | 2.4e-6 | 6.8e-5 |

FastMACE gives the baseline's forces: within rounding for plain cuEq, and within
the run-to-run scatter of the hybrid (OEq and cuEq reduce with atomics).

`labels_probe.py` loaded every label of the table once (plain cuEq float64):
`labels.json` records file, SHA-256 (all equal to the label table, trust
`label`), licence and the licence warning (shown once per checkpoint, so only
`mh1-omat` for the three MH-1 labels), heads, model class, the fused-cuEq guard
summary and OMOL-0's `embedding_specs`.

Also checked (no JSON): a tiny PolarMACE converts to the hybrid only with
`hybrid_keeps_last_layer_irreps` (stock MACE 0.3.16: `RuntimeError: einsum():
subscript a has size 7 for operand 1 which does not broadcast …`), and its hybrid
float64 forces match its e3nn forces within 1e-11 eV/Å; a hybrid load after
`import mace.calculators` is refused with the reason. GPU tests
(`tests/gpu/test_loading_gpu.py`, with `FASTMACE_HYBRID_PYTHON` = the
`mace_cueq_oeq` Python): 22 passed.
