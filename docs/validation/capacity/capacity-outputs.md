# Capacity screen A: the compiled model leaves out `node_feats`

Conductor, 30 October 2026: check whether model outputs that no evaluation reads, such as MACE's concatenated
`node_feats`, stay alive through the force backward; if so, compile a model that returns only what is needed. MH-1
hybrid float32 on the local RTX 5060 Ti, on `main` at `f3c635b9`.

## Verdict

**Exact, and proposed as the default.** `compile_model` gives its private copy of the model a `forward` that drops
the outputs in `fastmace.model.compile.UNUSED_OUTPUTS` (`("node_feats",)`) from the output dict.

- **Bitwise:** MD with `reproducible=True` over 300 NaCl steps with rebuilds gives the same positions and momenta,
  in float32 and in float64.
- **Memory:** the peak reserved memory drops by 0.43 GiB at 51,840 NbCl5 atoms and by 0.08 GiB at 9,216, about
  8.9 kB per atom. The graph pool shrinks by the same amount.
- **New limits:** NbCl5 51,840 → 54,432 atoms (6×6×21, +5.0 %); NaCl 49,248 → 51,984 atoms (18×19×19, +5.6 %). One grid step each.
- **Step time:** no change. In the same-process A/B, in both build orders, b/a is 0.997–0.998 at 1,944 atoms and
  1.000–1.004 at 9,216.
- **The force/stress gate passed** on NaCl, NbCl5 and NaNbO3 with the default plan.

## Why `node_feats` was alive

MACE's `ScaleShiftMACE.forward` returns, beside energy, forces and stress, `node_feats`: `torch.cat` of every
interaction's node features. FastMACE reads only energy, forces, stress and virials. But the compiled graph returns
`node_feats`, so Inductor has to materialise the concatenation as a graph output. The output dict then holds it until
the force call returns, across the whole force backward, which is where the peak is.

It is output memory, not a saved activation: the backward never reads it. So the gain is independent of
recompute. The combined recompute region now in the works (`recompute_node_segments`) removes saved activations;
it does not remove this tensor, and the two gains add.

## Change

- `fastmace/model/compile.py`: `UNUSED_OUTPUTS` and `drop_unused_outputs(model, names=None)`. `compile_model`
  calls it on its private copy right after freezing the parameters, before `torch.compile`. Only that copy's
  instance `forward` changes, so the user's model and its class are untouched. A forward that does not return a
  dict is passed through.
- `fastmace/model/forces.py`: `UNUSED_OUTPUTS` is in the compiled-model cache key, so a model compiled with one
  setting is never reused with another.
- The tuple is a constant, not a switch: the A/B sets it to `()` with `tools/ab_rewrites.py --a-constant`, as with
  the other measured constants.

## Measurements

MH-1 float32, default plan, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, local RTX 5060 Ti (sm_120, WSL2,
driver 610.88), torch 2.14.1+cu130, environment `mace_cueq_oeq` (hybrid backend). FastMACE `483bb295` (branch
`fastmace/cap-outputs`). Every job ran in its own local GPU reservation with a fresh compiler cache.

### Ledger (`tools/bench/memory_ledger.py`)

Peak reserved over the ledger's run (build, 20 warm-up steps, 20 steady steps, release and one re-capture), and the
graph pool at the end:

| NbCl5 | `main` peak | branch peak | `main` pool | branch pool |
| --- | ---: | ---: | ---: | ---: |
| 4×4×8, 9,216 atoms | 4.100 GiB | 4.021 GiB | 3.631 GiB | 3.553 GiB |
| 6×6×20, 51,840 atoms | 14.285 GiB | 13.855 GiB | 13.365 GiB | 12.936 GiB |

`main` is `results/cap-ledger-20261030` (runs p2 and p8 at main's library code). The branch's ledger matches the
screen's prototype exactly, which wrapped the model at the same place (`memory_ledger.py --output-wrapper`).

### Limits (`tools/bench/capacity_scan.py point`)

20 warm-up and 200 timed steps, one fresh process per point:

| Structure (float32) | atoms | status | ms/step | peak reserved | on `main` (capacity-nlist.md) |
| --- | ---: | --- | ---: | ---: | --- |
| NbCl5 6×6×21 | 54,432 | **ok** | 786 | 14.54 GiB | fails at build |
| NbCl5 6×6×22 | 57,024 | out of memory in the warm-up | | 14.85 GiB | |
| NaCl 18×19×19 | 51,984 | **ok** | 563 | 14.04 GiB | fails in the warm-up |
| NaCl 19×19×19 | 54,872 | marginal: 200 steps ran, but at 1,176 ms/step, and the extra rebuild after the run ran out of memory (the process aborted) | 1,176 | 14.78 GiB | |

NaCl at 54,872 atoms is not counted: twice the step time for 5.6 % more atoms is the card at its limit, and the
next rebuild fails.

### Same-process A/B (`tools/ab_rewrites.py --prepare`)

Side a is `UNUSED_OUTPUTS=()` (every output returned, as on `main`), side b the default. ms per step; the bracket
is the range of the block ratios:

| NbCl5 | order | a | b | b/a |
| --- | --- | ---: | ---: | --- |
| 3×3×3, 1,944 atoms, 24 blocks of 300 steps | ab (abab) | 17.91 | 17.87 | 0.997 (0.992–1.007) |
| 3×3×3 | ba (abba) | 17.80 | 17.76 | 0.998 (0.986–1.016) |
| 4×4×8, 9,216 atoms, 16 blocks of 100 steps | ab (abab) | 88.25 | 88.24 | 1.000 (0.995–1.011) |
| 4×4×8 | ba (abba) | 88.35 | 88.73 | 1.004 (0.993–1.011) |

No recompilation on either side. Setup times (13–37 s per side) show no consistent difference. In each process
the side built first compiles, so its setup takes longer. Built second, b took 13.0 s (1,944 atoms) and 20.0 s
(9,216 atoms); a took 13.3 s and 14.8 s.

### Tests

- `tests/cpu/test_unused_outputs.py` (3 tests). The CPU suite on leto: 360 passed, 16 skipped
  (`FASTMACE_BASELINE` not set), including `test_layering.py`.
- `tests/gpu/test_unused_outputs_gpu.py`: 2 passed (float32 and float64, bitwise).
- `tools/force_stress_gate.py`: passed (`results/cap-outputs-20261030/gate-summary.txt`).

## Records

`results/cap-outputs-20261030/`: `local.sh` (the jobs), `jobs.log` (runs o1–o6), `gate-summary.txt`, `gate-metadata.json`, `ab/`, `ledger/` and
`points/`.
