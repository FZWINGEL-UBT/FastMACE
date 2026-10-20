# The one-graph strain per model (plan task 2-T08b)

Spec: §8.7 "MACE patches" and "Instance isolation", §6.3 (hash gate).
Finding 1 of [isolation.md](isolation.md): `single_graph_strain` is a
process-wide patch of `mace.modules.utils.get_symmetric_displacement`. A
failing self-check (`Registry._selfcheck_failure`) or a reference
calculator whose record has the stock strain off
(`_patches.set_single_graph_strain(False)`, `reference/calculators.py`)
removes it for every force field built earlier.

## How MACE reaches the function

`MACE.forward`, `ScaleShiftMACE.forward` and `PolarMACE.forward` call
`prepare_graph(...)`, a global of their module (`mace.modules.models`,
`mace.modules.extensions`) bound to `mace.modules.utils.prepare_graph`.
`prepare_graph` calls `get_symmetric_displacement` as a global of
`mace.modules.utils`, looked up at call time. That lookup is what the
process patch changes.

## Design: rebind the globals, per forward function

Nothing is edited in MACE's source and nothing is installed in MACE's
modules.

1. **`strain_prepare_graph(one_graph)`** (in `rewrites/engine.py`): MACE's
   `prepare_graph` as a new function object with the same code and a copy
   of `mace.modules.utils`' globals. In that copy, `get_symmetric_displacement`
   is the one-graph function (`one_graph=True`) or MACE's original
   (`_patches._original`, `one_graph=False`). The two variants are built
   once and cached by `one_graph`.
2. **`strain_forward(function, one_graph)`**: a forward function (MACE's
   class forward, the float64-energy forward of `model.energy`, or one of
   polar's rewritten forwards) as a new function with the same code and
   globals in which `prepare_graph` is the variant from step 1. It refuses
   a forward that does not read `prepare_graph` as a global
   (`"prepare_graph" not in code.co_names`).
3. **Every force field binds its strain path explicitly, on or off.** A
   force field built with the record off is then also immune to a process
   patch installed by a reference calculator.
   - Private copy (rewrites, compiled or polar): at the end of
     `MACEForceField`'s `prepare`, after the float64 forward is installed,
     `model.forward = MethodType(strain_forward(current forward, on), model)`.
   - The eager source model, shared and never written: `_model_call =
     partial(strain_forward(class forward or float64 forward, on), source)`.
     This is how the float64 forward is already called.
   - PolarMACE: the forward is a class-level dispatcher (`polar.install`),
     so the choice goes into the per-instance `PolarHolder` (a
     `single_graph_strain` field), and the dispatcher's cache key
     `(PR 1762, float64 energy)` gains it. This is one narrow edit in
     `longrange/polar.py`.
4. **The record becomes a model record.** `apply` marks the model (an
   attribute read by step 3) and returns no `PatchRecord`. Its self-check
   (random inputs, MACE's function against the one-graph function) is
   unchanged. `_selfcheck_failure` then never restores anything for a
   registry record, and the restore branch goes away.
5. **Hash gate.** It now pins `mace.modules.utils:prepare_graph` next to
   `get_symmetric_displacement`, because the variant reuses its code. If
   either hash fails, the record is unqualified and off, with the reason
   logged, as today. `_patches.set_single_graph_strain` stays only for the
   reference calculators, whose records describe MACE's own calculator with
   or without the stock patch. It no longer reaches any `MACEForceField`.

## Dynamo, capture, compile cache

- A forward function bound to the instance is what the float64 energy
  already uses under `torch.compile`. Dynamo guards on the function object
  and its globals, and every instance keeps its own: nothing process-wide
  changes after a build, so no recompilation follows from another
  instance.
- Capture is unchanged: the graph records the same kernels.
  `single_graph_symmetric_displacement` is already the function traced
  today.
- The compile-cache key already contains the enabled records, so on and
  off never share a compiled model.

## What changes, what does not

- **Numbers:** none. A default plan (strain on) runs the same function on
  the same inputs; a plan with it off runs MACE's function. CPU float64
  stays bitwise against `main`.
- **Provenance:** `single_graph_strain` leaves `_patches.applied()` (the
  process patches) and stays in the force field's `applied` list and the
  plan's rewrite block. `rewrites/versions.py` ("installed"/"not
  installed") and `fixed.py` notes are reworded only.
- **Tests:**
  - `test_failing_strain_selfcheck_undoes_the_patch_of_earlier_force_fields`
    flips: B's failure leaves A's path and stress bitwise unchanged;
  - new: A built off is immune to a reference calculator's patch;
  - new: on and off force fields interleaved, each bitwise equal to fresh;
  - new: a forward without a global `prepare_graph` is refused;
  - the 1b reference-order tests stay unchanged.

## Files

- `rewrites/engine.py`: `strain_prepare_graph`, `strain_forward`, the
  record's `apply`; `single_graph_strain` moves from `PROCESS_RECORDS`'
  install path to a model record.
- `rewrites/registry.py`: `_selfcheck_failure` without `restore`.
- `_patches.py`: the `prepare_graph` pin in `SOURCE_PINS`; the setter's
  docstring says it is for the reference calculators.
- `model/forces.py`, after 2-T13: the binding of step 3.
- `longrange/polar.py`: the holder field and the dispatcher's key. This
  file is not in the plan's list, so I name it here first.
- Tests: `tests/cpu/test_isolation.py`, `tests/cpu/test_rewrites_engine.py`.
