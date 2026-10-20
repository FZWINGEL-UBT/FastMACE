# OPT-T06: converted-model template cache

**Status: implemented and merged, 21 October 2026; engine-level check on murphy done (run `20261021c`).** Plan `plans/opt-1.md`, OPT-T06, branch `fastmace/opt-T06`. Design:
[docs/design/opt-T06-template-cache.md](../design/opt-T06-template-cache.md) (approved by the conductor, option B).

## What changed

- `fastmace/model/templates.py` (new): `TemplateCache`, `applies` (CUDA and cuEq or hybrid only),
  `conversion_key`, `private` (a `compile.private_copy` in a shallow calculator shell), `fingerprint` (each
  tensor's identity, storage and in-place version counter).
- `ForceField.execution_model`:
  - the first plan of a conversion key gets the fresh load as before, and a private copy of it is kept as the
    template;
  - every later plan gets a private copy of the template. That includes the gate's reduced-cell model and
    fallback candidates.
- New methods: `ForceField.release_templates()` (returns the bytes freed) and `template_count()`.
- `EngineConfig.model_template_cache: bool = True`, the opt-out (switch-map row in `docs/design/switch-map.md`).
- `TemplateChangedError` in `fastmace/errors.py`, raised when a stored template was changed.
- SPEC §7.1 has a paragraph on execution copies and templates. The §19 revision entry is the conductor's.

**Key:** checkpoint SHA-256, path, label, head, model type, every option passed to `loading.load`, the versions of
mace-torch, torch, cuequivariance(-torch, -ops-torch), openequivariance and e3nn, and the two process patches that
act on a conversion: `cueq_conv_fusion` (MACE's converter) and `oeq_runtime` (the OEq build).

**Fixed during the run:** the first version keyed on every process patch in force. The engine patches the first
build installs (`capture_safe_mask_head`, `first_derivative_only`, `unfuse_full_addmm`) then made every later
plan miss: murphy's second `Simulation` kept two templates (run `opt-T06-murphy-20261021b`). Those patches act
when MACE's functions are called, not on the converted modules, so the key now holds only the two conversion
patches (commit 7e5a58d, with a test per engine patch).

## Measurements

murphy (RTX 4000 Ada 20 GB, driver 595.58.03), untimed (GPU checked empty before each process, load average up to 7
from the host), NAS `mace_cueq_oeq`. MH-1 `matpes_r2scan`, one backend per process. Script:
`results/opt-T06-probes/gpu_copy_vs_load.py`, runner `run_murphy.sh`. Data: `results/opt-T06-murphy-20261021b/`.

### Execution model per plan (s)

| Backend, dtype | first plan (load + template copy) | later plan, cache on (copy) | later plan, cache off (reload) |
| --- | ---: | ---: | ---: |
| hybrid float32 | 5.41 | **0.78** | 3.94 |
| hybrid float64 | 4.27 | **0.80** | 2.88 |
| plain cuEq float32 | 4.85 | **0.56** | 3.87 |
| plain cuEq float64 | 3.73 | **0.34** | 2.58 |

- A later plan saves 2.2–3.3 s.
- The first plan includes the process's first load (imports, first CUDA work) and the template copy, so it
  is not comparable with the reload column. The copy itself costs what the later-plan column shows.
- Template memory: 25.8 MB (float32) and 51.5 MB (float64). `release_templates()` gave back what it reported:
  `torch.cuda.memory_allocated` fell by 25.79 and 51.56 MB.

### Numerics: copy against reload

GPU kernels here are not run-to-run deterministic: the same plan's model evaluated twice is not bitwise equal.
So the copy is judged against that noise, on rattled NaCl 2×2×2, energy (eV) and largest force difference (eV/Å):

| Backend, dtype | same model twice (noise) | template copy vs reload | reload vs first plan |
| --- | --- | --- | --- |
| hybrid float32 | 0, 2.9e-7 | 3.1e-5, 3.1e-7 | 9.2e-5, 2.8e-7 |
| hybrid float64 | 5.7e-14, 7.8e-16 | 5.7e-14, 5.8e-16 | 5.7e-14, 6.9e-16 |
| plain cuEq float32 | 6.1e-5, 2.4e-7 | 0, 2.4e-7 | 0, 2.4e-7 |
| plain cuEq float64 | 0, 5.6e-16 | 0, 5.8e-16 | 0, 6.3e-16 |

The copy is as far from a reload as a reload is from the first plan's model, and as a model is from itself on a
rerun. On the CPU (e3nn float64, `results/opt-T06-cpu-20261021`), where the kernels are deterministic, a copy of
the template is bitwise a fresh reload, and bitwise a copy of either.

The engine-level columns of run `20261021b` compare two reloads (the miss above). The run after the fix is below.

### Engine level after the key fix

Run `opt-T06-murphy-20261021c` (murphy, 21 October 2026, 22:11–22:20 UTC, GPU empty, source 615464b, the merged
code; data in `results/opt-T06-murphy-20261021c/`). Two `Simulation` objects in a row per force field: with the
cache on (the second one is built from a template copy), and with it off (the second one from a reload). The table
gives each column's largest difference. Initial forces are in eV/Å, positions after 10 steps in Å.

| Backend, dtype | templates kept (on, off) | first plans, both loads: forces, positions | second plans, copy vs reload: forces, positions |
| --- | --- | --- | --- |
| hybrid float32 | 1, 0 | 2.5e-7, 4.1e-10 | 2.4e-7, 5.5e-10 |
| hybrid float64 | 1, 0 | 5.0e-16, 1.4e-17 | 6.8e-16, 0 (bitwise) |
| plain cuEq float32 | 1, 0 | 2.7e-7, 4.8e-10 | 2.1e-7, 5.0e-10 |
| plain cuEq float64 | 1, 0 | 7.4e-16, 2.1e-17 | 6.4e-16, 1.2e-17 |

- After the key fix, the second `Simulation` hits the cache: one template is kept.
- A plan built from a template copy differs from one built from a reload by no more than two plans built from
  two loads differ from each other, after 10 MD steps as well. That difference is the run-to-run noise of the GPU
  kernels.
- The execution-model times of this run agree with run `20261021b`: copy 0.32–0.83 s, reload 2.52–3.79 s.

## Tests

- `tests/cpu/test_template_cache.py`, 32 tests:
  - the rule (CUDA cuEq and hybrid only) and the opt-out;
  - first plan the load, later plans copies;
  - no shared storage, module, calculator or `results` between plans or with the template;
  - a rewritten plan does not leak into the next one (the fallback case);
  - every key field, including the conversion patches; engine patches still hit;
  - the LRU bound of 2; `release_templates`; lifetime with the `ForceField` (weakref);
  - the mutation guard, for a change in place and for a replaced tensor;
  - the same `loading.load` options as before; no module-level cache.
- `tests/gpu/test_template_cache_gpu.py`, tiny model:
  - fast tier: copy against reload on the model level for hybrid and plain cuEq in float32 and float64, and the
    memory release. 4 passed on murphy (run `20261021b`);
  - slow tier: a second `Simulation`, 10 steps, hybrid and plain cuEq float32. Failed in run `20261021b` on the
    template count (the miss, fixed).
  - run `20261021c`: **6 passed** (fast and slow tier, 152 s).
- CPU suite subset: config, contracts, research surface, planners, fallback (4 files), isolation, selfcheck,
  simulation, metadata and provenance. 471 + 198 passed, one switch-map row added for the new field.

## Not done here

- No timed setup measurement. Per later plan, the saving is the difference of the two right-hand columns above.
- The local GPU was off limits.
