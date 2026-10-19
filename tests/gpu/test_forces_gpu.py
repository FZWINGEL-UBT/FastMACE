"""MACEForceField on CUDA (plan task 1a-T11): adapters on copies, CUDA graphs, the streaming counter, MH-1.

- Plain cuEquivariance tiny models converted on the GPU with MACE's fused convolution: a private
  copy re-applies MACE's adapter (a plain deep copy loses it), the rewritten and compiled force
  fields agree with the eager unrewritten one, and a captured CUDA graph of the force call
  replays the eager result.
- The hybrid (OEq convolutions) in a fresh interpreter (``FASTMACE_HYBRID_PYTHON``): a private
  copy re-applies the OEq adapters, and two force fields from one compiled model with streaming
  convolutions step independently with no rebinding, eagerly and as captured graphs. This is the
  regression test of the research engine's ``bind_streaming_count`` failure: the same model run
  with the other field's count bound gives wrong forces, which the test also shows.
- MH-1 (``FASTMACE_MODELS``, head ``matpes_r2scan``) on NaCl 2x2x2 against the research engine's
  ``MACEForces``, each in its own process: plain cuEq float64 (eager without rewrites, and
  compiled with the production rewrites) and the hybrid in float32 (compiled, production rewrites
  with streaming). The research engine runs from the export named by ``FASTMACE_BASELINE``,
  which must hold ``common.py`` and the OEq launcher (the 1a-T18 export).
"""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from fastmace.config import EngineConfig, NeighbourConfig
from fastmace.model import compile as fc
from fastmace.model.forces import MACEForceField
from fastmace.neighbors.buckets import edge_bucket, round_up_edges
from fastmace.rewrites import registry as reg

pytest.importorskip("cuequivariance_torch")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODELS = os.environ.get("FASTMACE_MODELS")
BASELINE = os.environ.get("FASTMACE_BASELINE")
HYBRID_PYTHON = os.environ.get("FASTMACE_HYBRID_PYTHON") or sys.executable
Z_VALUES = [8, 11, 17, 26]
MH1_PACKAGE = ("factor_species", "fold_cutoff", "density_first", "joint_radial_input", "fused_folded_backward",
               "reassociate_first", "segmented_first_sum")
LINEAR = ("block_linear", "grouped_linear", "merged_residual")
SORTED = NeighbourConfig(skin=0.5, sort_receivers="on", gpu_neighbour_list="on")
UNSORTED = NeighbourConfig(skin=0.5, sort_receivers="off", gpu_neighbour_list="on")


def _hybrid_has(module):
    if HYBRID_PYTHON == sys.executable:
        return importlib.util.find_spec(module) is not None
    code = f"import importlib.util, sys; sys.exit(importlib.util.find_spec({module!r}) is None)"
    return subprocess.run([HYBRID_PYTHON, "-c", code], capture_output=True, timeout=120, check=False).returncode == 0


needs_oeq = pytest.mark.skipif(not _hybrid_has("openequivariance"),
                               reason="OpenEquivariance is not installed (hybrid environment needed)")


def structure(rattle=0.1, seed=8, oxygen=True, a=5.64):
    from _support.structures import nacl

    atoms = nacl(repeat=2, rattle=0, seed=seed)
    atoms.set_cell(atoms.cell.array * a / 5.64, scale_atoms=True)
    atoms.rattle(rattle, seed=seed)
    if oxygen:
        atoms.numbers[3] = 8
    return atoms


def only(*names):
    return {name: "on" if name in names else "off" for name in reg.default_registry().names()}


def relative(got, expected):
    return float((got - expected).abs().max()) / float(expected.abs().max())


def tiny_cueq_calculator(dtype):
    """A tiny MH-1-class calculator converted on the GPU with MACE's fused convolution (device "cuda")."""
    from mace.cli.convert_e3nn_cueq import run

    from _support.tiny_models import FAMILIES, tiny_mace_calculator

    calc = tiny_mace_calculator(Z_VALUES, **FAMILIES["mh1"])
    model = run(calc.models[0].to("cuda", dtype), device="cuda", return_model=True)
    calc.models = [model]
    calc.device = torch.device("cuda")
    return calc


def evaluate(field, positions, padded=True):
    field.compact(positions)
    count = int(field.edge_count)
    field.use_edges(round_up_edges(count, field.edge_capacity, edge_bucket(count, 0.25, 256)) if padded else count)
    out = field(positions)
    return [value.detach().clone() for value in out if value is not None]


def captured(field, positions):
    """Capture the force call in a CUDA graph (warm-up on a side stream first) and replay it."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            field(positions)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = field(positions)
    graph.replay()
    torch.cuda.synchronize()
    return graph, out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_private_copy_keeps_the_fused_cueq_adapter(dtype):
    calc = tiny_cueq_calculator(dtype)
    model = calc.models[0]
    fused = [name for name, module in model.named_modules() if "original_forward" in module.__dict__
             and type(module).__module__.startswith("cuequivariance_torch")]
    assert fused, "MACE did not fuse the convolution (device 'cuda')"
    private = fc.private_copy(model)
    modules = dict(private.named_modules())
    assert all("forward" in modules[name].__dict__ for name in fused)
    atoms = structure()
    plain = MACEForceField(calc, atoms, neighbours=UNSORTED)
    copied = MACEForceField(calc, atoms, neighbours=UNSORTED, rewrites=only("factor_species"))
    positions = torch.tensor(atoms.positions, device="cuda")
    threshold = reg.threshold_for(dtype)
    for a, b in zip(evaluate(copied, positions), evaluate(plain, positions)):
        assert relative(a, b) <= threshold


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("compile_mode", ["none", "default"])
def test_rewritten_force_field_agrees_and_replays_in_a_graph(dtype, compile_mode):
    calc = tiny_cueq_calculator(dtype)
    atoms = structure()
    fc.install_engine_patches()  # capture-safe readout mask (the engine's job; undone by conftest)
    plain = MACEForceField(calc, atoms, neighbours=UNSORTED, compute_stress=True)
    field = MACEForceField(calc, atoms, neighbours=SORTED, compute_stress=True,
                           rewrites=only(*MH1_PACKAGE, *LINEAR), engine=EngineConfig(compile_mode=compile_mode),
                           compile_cache=fc.CompiledModelCache())
    assert [entry.name for entry in field.applied] == [*MH1_PACKAGE, *LINEAR]
    threshold = reg.threshold_for(dtype)
    positions = torch.tensor(atoms.positions + 0.03, device="cuda")
    expected = evaluate(plain, positions)
    got = evaluate(field, positions)
    errors = [relative(a, b) for a, b in zip(got, expected)]
    _, out = captured(field, positions)
    replayed = [out.energy, out.forces, out.stress]
    errors += [relative(a, b) for a, b in zip(replayed, got)]
    print(json.dumps({"dtype": str(dtype), "compile_mode": compile_mode, "errors": errors}))
    assert max(errors) <= threshold


# --------------------------------------------------------------------------- fresh hybrid interpreters


def run_fresh(code, timeout=2400, python=HYBRID_PYTHON, prefix=(), extra_env=None):
    """Run code in a fresh interpreter; returns the JSON its last stdout line prints."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "tests")]),
               PYTHONDONTWRITEBYTECODE="1")
    env.update(extra_env or {})
    if python != sys.executable:
        env.pop("LD_LIBRARY_PATH", None)  # set for this environment; the in-process OEq runtime needs none
    result = subprocess.run([*prefix, python, "-c", textwrap.dedent(code)], cwd=REPOSITORY_ROOT, env=env,
                            capture_output=True, text=True, timeout=timeout, check=False)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


STREAMING_CHILD = """
import json
import fastmace
from fastmace.model import loading

loaded = loading.load({path!r}, backend="hybrid", dtype={dtype!r}, device="cuda:0", trusted=True)
import torch
from fastmace.config import EngineConfig, NeighbourConfig
from fastmace.model import compile as fc
from fastmace.model.forces import MACEForceField
from fastmace.neighbors.buckets import edge_bucket, round_up_edges
from fastmace.rewrites import mh1
from fastmace.rewrites import registry as reg
from _support.structures import nacl

def structure(rattle, seed, a):
    atoms = nacl(repeat=2)
    atoms.set_cell(atoms.cell.array * a / 5.64, scale_atoms=True)
    atoms.rattle(rattle, seed=seed)
    atoms.numbers[3] = 8
    return atoms

def relative(a, b):
    return float((a - b).abs().max()) / float(b.abs().max())

def padded(field, positions):
    field.compact(positions)
    count = int(field.edge_count)
    field.use_edges(round_up_edges(count, field.edge_capacity, edge_bucket(count, 0.25, 256)))
    return count

def values(out):
    return [out.energy.detach().clone(), out.forces.detach().clone()]

def errors(got, expected):
    return max(relative(a, b) for a, b in zip(got, expected))

# The OEq adapters on a private copy.
source = loaded.model
private = fc.private_copy(source)
restored = sum(1 for _, m in private.named_modules() if type(m).__module__.startswith("openequivariance")
               and "original_forward" in m.__dict__)

names = reg.default_registry().names()
package = ("factor_species", "fold_cutoff", "density_first", "joint_radial_input", "fused_folded_backward",
           "streaming_conv", "reassociate_first", "segmented_first_sum", "block_linear", "grouped_linear",
           "merged_residual")
requests = {{name: "on" if name in package else "off" for name in names}}
sorted_list = NeighbourConfig(skin=0.5, sort_receivers="on", gpu_neighbour_list="on")
cache = fc.CompiledModelCache()
engine = EngineConfig(compile_mode={compile_mode!r})
atoms_a, atoms_b = structure(0.1, 8, 5.64), structure(0.1, 3, 6.20)  # B expanded: fewer edges than A
first = MACEForceField(loaded, atoms_a, rewrites=requests, engine=engine, neighbours=sorted_list, compile_cache=cache)
second = MACEForceField(loaded, atoms_b, rewrites=requests, engine=engine, neighbours=sorted_list, compile_cache=cache)
plain_a = MACEForceField(loaded, atoms_a, neighbours=sorted_list)
plain_b = MACEForceField(loaded, atoms_b, neighbours=sorted_list)
x_a = torch.tensor(atoms_a.positions, device="cuda")
x_b = torch.tensor(atoms_b.positions, device="cuda")
count_a, count_b = padded(first, x_a), padded(second, x_b)
for field, x in ((plain_a, x_a), (plain_b, x_b)):
    padded(field, x)
ref_a, ref_b = values(plain_a(x_a)), values(plain_b(x_b))

result = {{"restored": restored, "shared": first.model is second.model, "count_a": count_a, "count_b": count_b,
          "streaming": first.info()["streaming_interactions"], "edges_a": int(first.data["edge_index"].shape[1])}}
# Eager, interleaved, no rebinding by anyone.
result["eager"] = max(errors(values(first(x_a)), ref_a), errors(values(second(x_b)), ref_b),
                      errors(values(first(x_a)), ref_a))

# Graphs captured after both fields exist, replayed in turn.
fc.install_engine_patches(oeq=True)
graphs = {{}}
for name, field, x in (("a", first, x_a), ("b", second, x_b)):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            field(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = field(x)
    graphs[name] = (graph, out)
replays = []
for name, ref in (("a", ref_a), ("b", ref_b), ("a", ref_a)):
    graph, out = graphs[name]
    graph.replay()
    torch.cuda.synchronize()
    replays.append(errors(values(out), ref))
result["graphs"] = max(replays)

# The research engine's failure: the shared model reading the other field's count.
mh1.bind_streaming_count(first.model, second.edge_count)
with torch.no_grad():
    first.positions.copy_(x_a)
out = first.model(first.data, training=first.training, compute_force=True, compute_stress=False)
result["wrong_count"] = errors([out["energy"].detach(), out["forces"].detach()], ref_a)
print(json.dumps(result))
"""


@needs_oeq
@pytest.mark.parametrize("compile_mode", ["none", "default"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_two_force_fields_from_one_model_step_independently(tmp_path, dtype, compile_mode):
    """The bind_streaming_count failure as a regression test: each field reads its own edge count."""
    from _support.tiny_models import FAMILIES, tiny_mace_model

    path = tmp_path / "tiny-mh1.model"
    torch.save(tiny_mace_model(Z_VALUES, **FAMILIES["mh1"]), path)
    out = run_fresh(STREAMING_CHILD.format(path=str(path), dtype=dtype, compile_mode=compile_mode))
    print(json.dumps(out, indent=1))
    threshold = reg.THRESHOLDS[dtype]
    assert out["restored"] > 0 and out["streaming"] == 2
    assert out["shared"] is (compile_mode != "none")
    # B has fewer edges than A, so A's model reading B's count drops some of A's real edges.
    assert out["count_b"] < out["count_a"] < out["edges_a"]  # and A's padded edges are present
    assert out["eager"] <= threshold and out["graphs"] <= threshold
    # What bind_streaming_count got wrong: the shared model reading B's (smaller) count leaves out
    # some of A's edges. Measured on murphy: ~1e-3 relative, against rounding of 3e-5 (float32)
    # and 5e-14 (float64).
    assert out["wrong_count"] > max(5 * threshold, 20 * out["eager"])


# --------------------------------------------------------------------------- MH-1 against the research engine


def labels_mh1_file():
    from fastmace.model import labels

    return labels.LABELS["mh1-r2scan"].checkpoint.file_name


def _has_mh1():
    if not MODELS:
        return False
    from fastmace.model import labels

    return (Path(MODELS) / labels.LABELS["mh1-r2scan"].checkpoint.file_name).is_file()


def _baseline_root():
    if not BASELINE:
        return None
    root = Path(BASELINE)
    needed = (root / "Experiments/Benchmark_MACE_GPU_OPTIMIZATIONS/scripts/common.py",
              root / "Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh")
    return root if all(path.is_file() for path in needed) else None


needs_mh1_baseline = pytest.mark.skipif(
    not (_has_mh1() and _baseline_root()),
    reason="needs FASTMACE_MODELS with MH-1 and FASTMACE_BASELINE with common.py and the OEq launcher (1a-T18 export)")

# One shared evaluation protocol: NaCl 2x2x2 rattled, two position sets below skin / 2 (no rebuild),
# exact and padded edges, energy, forces and stress.
EVALUATIONS = """
import numpy as np
from ase.build import bulk

def structure():
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    atoms.rattle(0.1, seed=8)
    return atoms

def bucket(count, capacity):
    step = 256 * max(1, int(np.ceil(0.05 * count / 256)))
    return min(capacity, step * max(1, int(np.ceil(count / step))))

def evaluations(compact, count, capacity, use_edges, call, positions0, torch):
    rows = []
    for shift in (0.0, 0.03):
        x = torch.tensor(positions0 + shift * np.random.default_rng(9).standard_normal(positions0.shape),
                         dtype=torch.float64, device="cuda")
        compact(x)
        n = int(count())
        for padded in (False, True):
            use_edges(bucket(n, capacity()) if padded else n)
            energy, forces, stress = call(x)[:3]
            rows.append({"shift": shift, "padded": padded, "edges": n, "energy": float(energy.detach().sum()),
                         "forces": forces.detach().double().cpu().tolist(),
                         "stress": stress.detach().double().cpu().reshape(3, 3).tolist()})
    return rows
"""

BASELINE_CHILD = """
import json, sys
sys.path[:0] = [{engine!r}, {scripts!r}]
import common
import torch
from langevin import gpu_langevin, upstream_mace
upstream_mace.use_cueq_conv_fusion(True)
upstream_mace.use_single_graph_strain(True)
calc = common.load_calculator("mh1-r2scan", dtype={dtype!r}, enable_cueq=True, enable_oeq={hybrid})
upstream_mace.require_fused_cueq(calc.models[0])
for name, value in {switches!r}.items():
    setattr(gpu_langevin, name, value)
atoms = structure()
field = gpu_langevin.MACEForces(calc, atoms, skin=0.3, compile_mode={compile_mode}, factor_species={factor},
                                sort_by_receiver={sort}, compute_stress=True)
rows = evaluations(field.neighbours.compact, lambda: field.neighbours.count, lambda: field.neighbours.capacity,
                   field.use_edges, field, atoms.positions, torch)
print(json.dumps({{"rows": rows, "torch": torch.__version__}}))
"""

FASTMACE_CHILD = """
import json
import fastmace
from fastmace.model import loading
loaded = loading.load("mh1-r2scan", backend={backend!r}, dtype={dtype!r}, device="cuda:0")
import torch
from fastmace.config import EngineConfig, NeighbourConfig
from fastmace.model import compile as fc
from fastmace.model.forces import MACEForceField
from fastmace.rewrites import registry as reg
names = reg.default_registry().names()
atoms = structure()
field = MACEForceField(loaded, atoms, rewrites={{name: "on" if name in {rewrites!r} else "off" for name in names}},
                       engine=EngineConfig(compile_mode={compile_mode!r}),
                       neighbours=NeighbourConfig(skin=0.3, sort_receivers={sort!r}, gpu_neighbour_list="on"),
                       compute_stress=True, compile_cache=fc.CompiledModelCache(),
                       float64_energy=False)  # 1a's energy sum, as the research engine (1b-T03)
rows = evaluations(field.compact, lambda: field.edge_count, lambda: field.edge_capacity, field.use_edges, field,
                   atoms.positions, torch)
print(json.dumps({{"rows": rows, "info": {{k: v for k, v in field.info().items() if k != "decisions"}},
                  "torch": torch.__version__}}))
"""

PRODUCTION = (*MH1_PACKAGE, *LINEAR, "single_graph_strain")
FIRST = ("reassociate_first", "segmented_first_sum")
FIRST_SWITCHES = ("REASSOCIATE_FIRST_INTERACTION", "SEGMENTED_FIRST_SUM")
BASELINE_SWITCHES = {"FOLD_RADIAL_CUTOFF": True, "DENSITY_BEFORE_LINEAR": True, "JOINT_RADIAL_INPUT": True,
                     "FUSED_FOLDED_BACKWARD": True, "REASSOCIATE_FIRST_INTERACTION": True,
                     "SEGMENTED_FIRST_SUM": True, "BLOCK_LINEAR": True, "GROUPED_BLOCK_LINEAR": True,
                     "MERGED_RESIDUAL_LINEAR": True}
MH1_CASES = {
    # name: (backend, dtype, compile mode, FastMACE rewrites, research-engine switches, receiver sorting)
    "cueq-float64-eager-plain": ("cueq", "float64", "none", ("single_graph_strain",), {}, False),
    # Without reassociation: the research engine's probe demands cuEq's path coefficients exactly
    # equal and refuses plain cuEq ("First convolution coefficients differ within a path"; 1a-T07).
    "cueq-float64-compiled-production": (
        "cueq", "float64", "default", tuple(n for n in PRODUCTION if n not in FIRST),
        {k: v for k, v in BASELINE_SWITCHES.items() if k not in FIRST_SWITCHES}, False),
    "hybrid-float32-compiled-production": ("hybrid", "float32", "default", (*PRODUCTION, "streaming_conv"),
                                           {**BASELINE_SWITCHES, "STREAMING_CONVOLUTION": True}, True),
}


def _compare(ours, theirs):
    report = {"energy_eV": [], "energy_error_eV": [], "force_error_eV_per_A": [], "stress_error_eV_per_A3": [],
              "edges": []}
    for a, b in zip(ours["rows"], theirs["rows"], strict=True):
        assert (a["shift"], a["padded"]) == (b["shift"], b["padded"])
        report["edges"].append([a["edges"], b["edges"]])
        report["energy_eV"].append(b["energy"])
        report["energy_error_eV"].append(abs(a["energy"] - b["energy"]))
        report["force_error_eV_per_A"].append(float(np.abs(np.subtract(a["forces"], b["forces"])).max()))
        report["stress_error_eV_per_A3"].append(float(np.abs(np.subtract(a["stress"], b["stress"])).max()))
    return report


@needs_mh1_baseline
@pytest.mark.parametrize("case", list(MH1_CASES))
def test_mh1_agrees_with_maceforces(case):
    backend, dtype, compile_mode, rewrites, switches, sort = MH1_CASES[case]
    root = _baseline_root()
    hybrid = backend == "hybrid"
    python = HYBRID_PYTHON if hybrid else sys.executable
    theirs_code = EVALUATIONS + BASELINE_CHILD.format(
        engine=str(root / "Experiments/Langevin_Thermostat"),
        scripts=str(root / "Experiments/Benchmark_MACE_GPU_OPTIMIZATIONS/scripts"), dtype=dtype, hybrid=hybrid,
        switches={**switches, "GPU_NEIGHBOUR_LIST": True}, compile_mode=None if compile_mode == "none" else
        repr(compile_mode), factor="factor_species" in rewrites, sort=sort)
    prefix = ()
    if hybrid:  # the research engine's hybrid runs through its launcher (SPEC §3.2)
        cache = Path(os.environ.get("FASTMACE_CACHE", REPOSITORY_ROOT / ".cache")) / "oeq-baseline-1a-T11"
        prefix = ("bash", str(root / "Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh"),
                  str(Path(python).parents[1]), str(cache))
    # The research engine's model store looks for the NAS path unless told where MH-1 is (1a-T13 Q2).
    checkpoint = Path(MODELS) / labels_mh1_file()
    theirs = run_fresh(theirs_code, python=python, prefix=prefix,
                       extra_env={"MACE_BENCH_CHECKPOINT": os.environ.get("MACE_BENCH_CHECKPOINT", str(checkpoint))})
    ours = run_fresh(EVALUATIONS + FASTMACE_CHILD.format(backend=backend, dtype=dtype, compile_mode=compile_mode,
                                           rewrites=rewrites, sort="on" if sort else "off"), python=python)
    report = _compare(ours, theirs)
    report.update(case=case, info=ours["info"], torch=ours["torch"])
    print(json.dumps(report, indent=1))
    assert all(a == b for a, b in report["edges"])
    if dtype == "float64":
        assert max(report["force_error_eV_per_A"]) <= 1e-10 and max(report["energy_error_eV"]) <= 1e-9
        assert max(report["stress_error_eV_per_A3"]) <= 1e-12
    else:  # float32: the hybrid's run-to-run scatter is about 3e-7 eV/A (1a-T05)
        assert max(report["force_error_eV_per_A"]) <= 1e-5 and max(report["energy_error_eV"]) <= 1e-3
        assert max(report["stress_error_eV_per_A3"]) <= 1e-6
