# FastMACE in five minutes

This page takes you from installation to a running MD simulation (SPEC §15).
FastMACE runs MD with pretrained MACE models on an NVIDIA GPU.

## 1. Install

```bash
pip install "fastmace[full]"
```

`[full]` is the union of `[oeq]` (OpenEquivariance, the fastest backend),
`[polar]` (graph-longrange, for PolarMACE) and `[d3]` (torch-dftd). Install
the PyTorch build that matches your CUDA first. The cuEquivariance ops wheel
(`cuequivariance-ops-torch-cu12` or `-cu13`) depends on your CUDA build too,
so it is not installed automatically.

The base install (`pip install fastmace`) also works. It runs on the plain
cuEquivariance backend, and the run report tells you so.

## 2. Check the installation

```bash
fastmace doctor
```

`doctor` checks the GPU, driver, torch, MACE, the cuEquivariance ops and
their fused kernels, OpenEquivariance and its build path, Triton, NVRTC,
graph-longrange, torch-dftd, the cache root, the lock mode on NFS and the
model files. For each item that fails it names the command or document that
fixes it. It changes nothing. It exits 0 when everything is ready and 1
otherwise. On a CPU-only machine it exits 1, because the GPU items fail.

`fastmace models` lists the model labels with their heads, file names,
SHA-256 and licences.

## 3. Run MD from the command line

```bash
fastmace md --model mh1 --head matpes_r2scan --structure NaCl.cif \
  --ensemble NVT --temperature 300 --steps 1000000 --repeat 3 3 3 --output run/
```

This is a Langevin NVT run at 300 K with MH-1's r2SCAN head on a 3×3×3
supercell. The summary report shows the plan, the gate status and the speed.
`run/` holds the trajectory, the thermo table, the restart files and
`metadata.json`, which records the plan, gate and provenance.

| Flag | `fm.md` argument | Default |
| --- | --- | --- |
| `--model` | `model`: label, path or URL | required |
| `--head` | `head` | required for multi-head models such as MH-1 |
| `--charge`, `--spin-multiplicity` | `charge`, `spin_multiplicity` | the model's |
| `--sha256`, `--trusted` | `sha256`, `trusted`, for model files outside the label table | |
| `--structure` | the structure, any format ASE reads | required |
| `--repeat NA NB NC` | `repeat` | `1 1 1`: the cell is never expanded silently |
| `--ensemble` | `ensemble`: `NVT` (Langevin) or `NPT` (LangevinBAOAB) | `NVT` |
| `--temperature` | `temperature_K` (K) | required |
| `--pressure`, `--cell-mode` | `pressure_GPa` (GPa), `cell_mode` (`hydrostatic`, `full`), NPT only | `0`, `hydrostatic` |
| `--timestep` | `timestep_fs` (fs) | `0.5` |
| `--steps` | `steps` | required |
| `--dispersion`, `--d3-functional` | `dispersion`: `none`, or `d3bj` expanded from the head's functional (or the functional given) | `none` |
| `--seed` | `seed` | drawn and recorded |
| `--check` | `check`: `skip`, `gate` or `allow_unavailable` | `skip` |
| `--output` | `output` directory | `./<structure stem>.fastmace/` |

The Langevin friction is `fm.md`'s default, 0.02 per ASE time unit (about
0.002/fs), as in `main.py`.

Some requests are refused before anything runs, with the reason: NVE
(phase 4) and D3 (phase 3b). For D3, the refusal shows the expanded record.
MC moves run through `fastmace run` (below), and they are experimental until
their qualification study is recorded. An error prints its message without a traceback and exits
1. `--debug` also prints the traceback. A wrong combination of arguments
exits 2.

The gate checks the model on your structure against references, and it is
opt-in. `--check gate` runs it before the first step.

## 4. Runs with a schedule or an explicit integrator: `fastmace init` and `fastmace run`

Some runs have no `md` flag: a temperature schedule, MC moves, an explicit
integrator, a full D3 record or the output intervals. For these, write a TOML
file:

```bash
fastmace init > run.toml
fastmace run run.toml
```

`run.toml` holds every `fm.md` argument under its own name. The values that
are set are those of the example run above. Every other argument is a
commented line with what it does and an example. With `md`'s flags, `init`
sets those flags instead:

```bash
fastmace init --model mh1 --head matpes_r2scan --structure NaCl.cif --temperature 300 --steps 1000 > run.toml
```

Then `fastmace run run.toml` is the same run as `fastmace md` with the same
flags. `fastmace run -` reads the file from standard input. A key that is not
an `fm.md` argument is refused, naming it. Paths are relative to the directory
you start `fastmace run` in.

The records are tables at the end of the file:

```toml
schedule = "anneal"            # a profile: slow, fast, ultrafast, anneal

# or segments:
[[schedule]]
kind = "ramp"
start_K = 0.0
end_K = 1000.0
ps = 20.0

[[schedule]]
kind = "hold"
temperature_K = 1000.0
ps = 10.0

[integrator]                   # fm.integrators.Langevin or LangevinBAOAB
type = "LangevinBAOAB"
temperature_K = 300.0
cell_mode = "full"

[dispersion]                   # a full fm.D3 record
functional = "r2scan"

[output]                       # fm.Output
directory = "run/"
thermo_every = 20
```

Use either `schedule = "anneal"` or `[[schedule]]` segments, not both.

## 5. The same run from Python: `fm.md`

```python
import fastmace as fm

run = fm.md("NaCl.cif", model="mh1", head="matpes_r2scan", ensemble="NVT",
            temperature_K=300, timestep_fs=0.5, steps=1_000_000, repeat=(3, 3, 3),
            output="run/", seed=1)
print(run.summary)                 # the report
run.plan, run.gate                 # the plan and gate records
sim = run.simulation               # the Simulation, to run on or restart
```

`fm.md`, `fastmace md` and `fastmace run` build the same `Simulation`. For
the same inputs they give the same plan, gate and trajectory.

## For CI: the install check

The CPU CI checks that a fresh environment resolves `[full]` and that
`doctor` runs. These commands do that, from the repository root, with CPU
torch:

```bash
python -m venv /tmp/fastmace-full
/tmp/fastmace-full/bin/python -m pip install --upgrade pip
/tmp/fastmace-full/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/fastmace-full/bin/python -m pip install ".[full]"
/tmp/fastmace-full/bin/python -m pip check
/tmp/fastmace-full/bin/fastmace doctor; test $? -le 1
```

Without a GPU, `doctor` exits 1, so the check accepts 0 or 1. Exit code 2 or
a traceback fails the check.
