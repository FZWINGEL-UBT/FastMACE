# 3-T03: mutation events and the derived-state registry, validation

Plan task 3-T03 (SPEC §8.5, §8.1 "Graph-safe events", §12.2 "Mutation events"), 22 October 2026.
Branch `fastmace/3-T03`. Tests run on the branch after it was synced with main (with 3-T02 and 3-T08).
No research-engine baseline is involved: every comparison is with a fresh FastMACE construction from
the committed state.

## GPU: fast tier

- **Run:** `tests/gpu/test_events_gpu.py`, 12 tests, 12 passed in 8 s. `tools/test_budget.py`: within
  budget (0.1 min).
- **Setup:** local PC (WSL2), RTX 5060 Ti, environment `mace_cueq_latest` (torch 2.14.1+cu130), through
  `tools/ci/local_gpu.py`. In process, with `test_engine_gpu.py`'s deterministic Lennard-Jones force
  field, float64, captured CUDA graphs, and the GPU neighbour list.
- **What was compared**, all bit for bit:
  - **A refresh captured in a CUDA graph and replayed after a new mass write.** The buffers were
    Langevin `c3_unit`, `c4_unit` and `c5_unit`, and BAOAB `inv_sqrt_masses` and `force_eps`. They
    equal a fresh construction's.
  - **The first three captured steps after an accepted mass or friction event equal a fresh
    construction.** Cases: checked and pipelined Langevin; BAOAB hydrostatic (checked) and full cell
    (pipelined). After a friction event the captured stages are dropped and recaptured, so the new
    `c1`/`c2` are in the graphs.
  - **A rejected mass-and-friction event, which used random numbers in its hook, restores the run.**
    The steps that follow equal those of a twin run without the event.

## CPU (WSL, `fastmace_cpu`, float64, `-n 2`)

- **Run:** `tests/cpu/test_events.py`, with the neighbouring files (contracts 3, superset, simulation,
  subprocess runner and worker, restart and restart v3, canonical restart, registry, integrators,
  engine, interfaces, minimal planner, forces): 554 passed, 36 skipped. The skips are for a missing
  `FASTMACE_BASELINE`, the polar module and the shallow tag. Ruff is clean.
- **Bit for bit against a fresh construction**, with `events.check_against`, the CI registry check:
  - species (Na ↔ Cl on tiny MH-1), mass, friction and total-charge events, accepted and rejected;
  - Lennard-Jones in the checked, pipelined and speculative modes, and BAOAB hydrostatic and full cell
    (`force_eps`).
- **Negative controls:** dropping Langevin's `c3_unit` registration, or skipping BAOAB's friction
  constants, fails the check.
- **Bookkeeping:** a transaction writes only registered tensors (`unregistered_mutations`). A ramp
  changes the target every step without an event.
- **Subprocess runner:** `update` through the persistent worker and through fresh children gives the
  in-process state bit for bit.

## Not done here

- **The hybrid (cuEq + OEq) "within the gate" case and a reproducible-mode MACE run on the GPU.** The
  fast tier covers the event path with a deterministic force field. MACE-specific buffers are 3-T02's
  and are covered on the CPU in float64.
