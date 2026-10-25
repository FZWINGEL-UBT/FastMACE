# FastMACE examples

Short, runnable scripts for the common jobs. Every script takes

- `--checkpoint PATH`: the MH-1 checkpoint (`mace-mh-1.model`); it is checked against the label table's
  SHA-256. Without it, FastMACE looks in `$FASTMACE_MODELS` and its cache.
- `--steps N`: how many MD (or optimiser) steps to run, small for a smoke run;
- `--output DIR`: where to write (default `~/tmp/fastmace-examples/<script>/`);
- `--seed`, and `--structure FILE` where the script reads one.

`python examples/<script> --help` lists the rest. Run them with a FastMACE install, or from a checkout with
`PYTHONPATH=<checkout>` and an environment that has the hybrid backend (see `docs/quickstart.md`).

| Script | What it shows |
| --- | --- |
| `01_nvt_anneal_nacl.py` | NVT simulated anneal of NaCl 3×3×3 with MH-1 through `fm.ForceField` + `fm.Simulation` and an `fm.protocols` ramp schedule (0 → 1000 K → 250 K); `--steps` pauses the schedule early |
| `02_npt_d3_melt_quench_nbcl5.py` | NPT (Langevin-BAOAB, 0 GPa) + D3(BJ) short melt-quench of NbCl5 through `fm.md`, then a GPU LBFGS quench of the last frame with the same D3 record |
| `03_swap_mc_anneal.py` | The shipped `swap_anneal_NaNbOCl4_short` protocol (O ↔ Cl swap MC during an anneal; relax, MD, quench) through the protocol runner, as `fastmace run` does; continue with `fastmace run --resume DIR` |
| `04_gpu_relaxations.py` | `fm.optimize.LBFGS` and `fm.optimize.FIRE` on the graph engine, fixed cell and with `FrechetCell` (variable cell) |
| `05_port_ase_script.py` | Porting an ASE script: `fm.calc` for single points, `fastmace.ase.Langevin(..., model=)` in place of ASE's `Langevin` |

Smoke commands (WSL, hybrid environment, structures from the R2SCAN benchmark set):

```bash
export PYTHONPATH=$PWD M=~/models/mace-mh-1.model S=~/fastmace-baseline/1a-T18/Structures/Benchmark/R2SCAN
PY=~/environments/mace_cueq_oeq/bin/python
$PY examples/01_nvt_anneal_nacl.py --structure $S/NaCl.cif --checkpoint $M --steps 200
$PY examples/02_npt_d3_melt_quench_nbcl5.py --structure $S/NbCl5.cif --checkpoint $M --steps 50 --relax-steps 20
$PY examples/03_swap_mc_anneal.py --checkpoint $M --steps 700 --repeat 1 1 1
$PY examples/04_gpu_relaxations.py --structure $S/NaCl.cif --checkpoint $M --steps 50
$PY examples/05_port_ase_script.py --structure $S/NaCl.cif --checkpoint $M --steps 100
```

The step counts of the schedules are physical: at 1 fs, `01` is 270,000 steps and `02`/`03` 27,000. Short
`--steps` runs are smoke runs, not results. `tests/cpu/test_examples.py` checks that each script imports and
prints `--help`.
