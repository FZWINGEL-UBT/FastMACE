# FastMACE

FastMACE runs long molecular-dynamics trajectories with pretrained MACE
models (MH-1, MP-0, OMAT-0, MatPES, MACE-POLAR-1, OMOL-0, MACE-OFF) on one
NVIDIA GPU. It owns the whole MD loop: CUDA graphs over each step, exact
model rewrites, a cuEquivariance/OpenEquivariance hybrid backend, and a
planner that picks the fastest engine that passes its checks on your
structure. Results are exact to MACE within float32 rounding.

Measured by us on an RTX 5060 Ti, float32, 216 to 1,944 atom cells: 3.8 to
10 times faster per step than TorchSim 0.6.2, and 1.1 to 2 times faster
than Symmetrix-XL in LAMMPS; 3 to 22 times faster than MACE's ASE
calculator. MH-1 on 216-atom NaCl: 2.90 ms per 1 fs step. The margin is
largest on small cells; we claim nothing beyond the measured sizes and GPUs
([TorchSim comparison](https://github.com/FZWINGEL-UBT/FastMACE/blob/main/docs/validation/torchsim-comparison-20261025.md),
[positioning](https://github.com/FZWINGEL-UBT/FastMACE/blob/main/docs/design/2026-10-positioning.md)).

## One run

After [installing](install.md):

```python
import fastmace as fm

run = fm.md("NaCl.cif", model="mh1", head="matpes_r2scan", ensemble="NVT",
            temperature_K=300, steps=1_000_000, repeat=(3, 3, 3), output="run/", seed=1)
print(run.summary)
```

This is 1,000,000 Langevin steps of 0.5 fs at 300 K on a 3×3×3 NaCl
supercell with MH-1's r2SCAN head. [FastMACE in five minutes](quickstart.md)
walks through it.

## What you can do

Each section shows the shortest call. The guide explains it, and the
reference page lists every option. The snippets continue the one above
(`import fastmace as fm`).

### NVT, NPT and NVE molecular dynamics

`ensemble=` picks Langevin NVT, Langevin-BAOAB NPT or velocity-Verlet NVE.

```python
run = fm.md("NbCl5.cif", model="mh1", head="matpes_r2scan", ensemble="NPT",
            temperature_K=300, pressure_GPa=0.0, cell_mode="hydrostatic",
            steps=200_000, repeat=(3, 3, 3), output="npt/", seed=1)
```

NVE takes no temperature. It starts from the momenta stored on the
structure.

Guide: [Running MD](guide/md.md), [NPT and D3](guide/npt-d3.md).
Reference: [Integrators and optimisers](reference/integrators.md).

### Other thermostats and barostats

Pass any `fm.integrators` record: `CSVR` (ASE's `Bussi`),
`NoseHooverChain`, `MTK` (NPT with Nosé-Hoover chains) or `Berendsen`.
Times are in fs.

```python
run = fm.md("NaCl.cif", model="mh1", head="matpes_r2scan", steps=100_000, repeat=(3, 3, 3),
            integrator=fm.integrators.NoseHooverChain(timestep_fs=1.0, temperature_K=300,
                                                      tdamp_fs=100.0),
            output="nhc/", seed=1)
```

Guide: [Running MD](guide/md.md#choosing-an-integrator).
Reference: [Integrators and optimisers](reference/integrators.md).

### Temperature schedules and anneals

`fm.protocols.ramp` and `fm.protocols.hold` build a schedule; `+` joins
segments. With a schedule, `steps=` is optional: the run covers the whole
schedule.

```python
anneal = (fm.protocols.ramp(0, 1000, ps=20) + fm.protocols.hold(1000, ps=10)
          + fm.protocols.ramp(1000, 250, rate_K_per_ps=-3))
run = fm.md("NaCl.cif", model="mh1", head="matpes_r2scan", schedule=anneal,
            timestep_fs=1.0, repeat=(3, 3, 3), output="anneal/", seed=1)
```

Guide: [Running MD](guide/md.md#steps-and-schedules).
Reference: [Models, D3, PolarMACE and schedules](reference/models.md).
Example: [`examples/01_nvt_anneal_nacl.py`](https://github.com/FZWINGEL-UBT/FastMACE/blob/main/examples/01_nvt_anneal_nacl.py).

### Melt-quench

Heat and cool at constant pressure, then relax frames of the run to their
minima on the GPU.

```python
melt_quench = fm.protocols.ramp(0, 1000, ps=2) + fm.protocols.ramp(1000, 250, rate_K_per_ps=-30)
run = fm.md("NbCl5.cif", model="mh1", head="matpes_r2scan", ensemble="NPT",
            schedule=melt_quench, timestep_fs=1.0, dispersion="d3bj", output="mq/", seed=1)
```

```bash
fastmace quench mq/ --into gpu --optimizer gpu-LBFGS
```

Guide: [Relaxations and quenches](guide/relax.md).
Example: [`examples/02_npt_d3_melt_quench_nbcl5.py`](https://github.com/FZWINGEL-UBT/FastMACE/blob/main/examples/02_npt_d3_melt_quench_nbcl5.py).

### GPU relaxations

LBFGS and FIRE run on the graph engine with ASE 3.27's defaults.
`FrechetCell` relaxes the cell too (ASE's `FrechetCellFilter`, pressure in
GPa).

```python
from ase.io import read

atoms = read("NaCl.cif")
model = fm.load("mh1", head="matpes_r2scan")
opt = fm.optimize.FIRE(cell_filter=fm.optimize.FrechetCell(scalar_pressure_GPa=0.0))
sim = fm.Simulation(atoms, model, optimizer=opt, seed=0)
relaxed, summary = sim.relax(fmax=0.01)    # eV/Å
```

Guide: [Relaxations and quenches](guide/relax.md).
Reference: [Integrators and optimisers](reference/integrators.md).
Example: [`examples/04_gpu_relaxations.py`](https://github.com/FZWINGEL-UBT/FastMACE/blob/main/examples/04_gpu_relaxations.py).

### Monte Carlo species swaps and transmutations (experimental)

MC moves run between MD steps. A species change is an event inside the
captured graph, so a swap costs no re-capture.

```python
run = fm.md("NaNbOCl4.cif", model="mh1", head="matpes_r2scan",
            temperature_K=600, steps=200_000,
            mc=fm.mc.Swap(pairs=[("O", "Cl")], every=100, attempts=50),
            output="swap/", seed=1)
```

`fm.mc.Transmute` changes species semi-grand-canonically, and
`fm.mc.Displace` moves single atoms. MC moves are marked experimental in
the run report until their qualification study is recorded.

Guide: [Monte Carlo and swaps](guide/mc.md).
Reference: [Monte Carlo](reference/mc.md).
Example: [`examples/03_swap_mc_anneal.py`](https://github.com/FZWINGEL-UBT/FastMACE/blob/main/examples/03_swap_mc_anneal.py).

### D3(BJ) dispersion

FastMACE's own GPU D3(BJ) runs inside the captured step, in MD,
relaxations and quenches. `"d3bj"` takes the parameter set of the head's
training functional; `fm.D3(...)` sets it explicitly.

```python
run = fm.md("NbCl5.cif", model="mh1", head="matpes_r2scan", ensemble="NPT",
            temperature_K=300, steps=200_000, dispersion="d3bj", output="npt/", seed=1)
```

Guide: [NPT and D3](guide/npt-d3.md#d3bj-dispersion).
Reference: [Models, D3, PolarMACE and schedules](reference/models.md).

### PolarMACE and charged systems

MACE-POLAR-1 adds long-range electrostatics (install the `[polar]` extra).
It runs at charge 0 and without D3 (PolarMACE with D3 is refused).
OMOL-0 takes a total charge and a spin multiplicity.

```python
run = fm.md("NaCl.cif", model="polar1-m", temperature_K=300, steps=100_000,
            output="polar/", seed=1)

run = fm.md("box.extxyz", model="omol0-xl", charge=1, spin_multiplicity=2,
            temperature_K=300, steps=100_000, output="ion/", seed=1)
```

The structure must be periodic in all three directions, also for a
molecule.

Guide: [Models and licences](guide/models.md).
Reference: [Models, D3, PolarMACE and schedules](reference/models.md).

### Port an ASE script

`fm.calc` is an ASE calculator for tools that need only energies, forces
and stress. The `fastmace.ase` classes take ASE's arguments and run the MD
loop on the graph engine at full speed.

```python
from ase import units
from ase.io import read
from fastmace.ase import Langevin       # was: from ase.md.langevin import Langevin

atoms = read("NaCl.cif").repeat((3, 3, 3))
dyn = Langevin(atoms, 0.5 * units.fs, temperature_K=300, friction=0.002 / units.fs,
               model="mh1", head="matpes_r2scan", trajectory="run.traj",
               logfile="run.log", loginterval=20)
dyn.run(1_000_000)
```

Guide: [Port your ASE script](guide/ase.md).
Reference: [API: ASE front end](api/ase.md).
Example: [`examples/05_port_ase_script.py`](https://github.com/FZWINGEL-UBT/FastMACE/blob/main/examples/05_port_ase_script.py).

### Restarts and resume

A restart file holds the complete state: positions, momenta, cell,
thermostat and barostat state, RNG state, exchanged species, schedule
position, MC counters and the plan.

```python
sim = fm.Simulation.resume("run/")
sim.run()                       # the rest of the schedule
```

Guide: [Output, restart and resume](guide/restart.md).
Reference: [Output files](reference/output.md).

### The plan and the gate

Every run starts with a plan: tier, backend, precision, rewrites and
neighbour-list sizes, each with its reason. The optional gate checks the
planned engine on your structure against a float64 reference before the
first step. If it fails, FastMACE falls back to the closest plan that
passes and records what it did.

```python
run = fm.md("NaCl.cif", model="mh1", head="matpes_r2scan", temperature_K=300,
            steps=100_000, check="gate", output="run/", seed=1)
print(run.plan)
print(run.gate)
```

Guide: [The planner](guide/planner.md).
Reference: [Planner, gate and fallback](reference/planner.md).

### The command line

```bash
fastmace doctor                          # check the installation
fastmace models                          # model labels, heads, licences
fastmace md --model mh1 --head matpes_r2scan --structure NaCl.cif \
  --temperature 300 --steps 100000 --repeat 3 3 3 --output run/
fastmace init > run.toml                 # every fm.md argument, commented
fastmace run run.toml                    # schedules, MC, explicit integrators
fastmace relax NaCl.cif --model mh1 --head matpes_r2scan --fmax 0.01
fastmace quench run/ --into gpu
```

Guide: [FastMACE in five minutes](quickstart.md).
Reference: [Command line and TOML](reference/cli.md).

## Use something else when

- **You have many small independent structures** (screening, batched
  relaxations): use a batched engine such as
  [TorchSim](https://github.com/TorchSim/torch-sim) or
  [NVIDIA ALCHEMI](https://github.com/NVIDIA/nvalchemi-toolkit). FastMACE
  runs one structure at a time.
- **Your cell has more than about 100,000 atoms, or you need several GPUs**:
  use MACE's [LAMMPS ML-IAP](https://mace-docs.readthedocs.io/en/latest/guide/lammps_mliap.html)
  interface or [Symmetrix-XL](https://github.com/amphilagus/symmetrix-xl),
  which decompose the domain and use less memory per atom.
- **Your model is not MACE 0.3.x** (MatterSim, UMA, SevenNet, ORB, NequIP
  ...): TorchSim or ALCHEMI run many model families.
- **You run on AMD GPUs or CPUs**: Symmetrix runs there; FastMACE is
  NVIDIA only.
- **You need LAMMPS's ecosystem** (PLUMED, enhanced sampling, its many
  fixes): use LAMMPS.

## How the docs are organised

!!! info "Where to look"
    - **Getting started:** [Install](install.md) and
      [FastMACE in five minutes](quickstart.md), the first run three ways.
    - **User guide:** one page per job, from [Running MD](guide/md.md) to
      [Relaxations and quenches](guide/relax.md), and the [FAQ](faq.md).
    - **Reference:** [every option](reference/index.md), command-line flag
      and TOML key with its default, and the
      [API reference](api/index.md), generated from the code.
    - **Project:** [benchmarks](benchmarks.md),
      [rewrite screens](rewrites.md), [licences](licences.md) and
      [development](development.md).

The specification, design notes and validation records are kept in the
[repository](https://github.com/FZWINGEL-UBT/FastMACE) and are not part of
this site.
