"""The CUDA-graph engine on the GPU (plan task 1a-T13).

In this process, with a deterministic torch Lennard-Jones force field (no MACE, seconds):
- captured replay equals eager steps (same padded edge sizes) bitwise over 200 steps with rebuilds;
- capture warm-ups leave the whole runtime state as it was, at the same addresses;
- asynchronous output (``snapshot``) is bit-identical to synchronous output (``sync``);
- pipelined and speculative steps equal checked steps.

MH-1 (``FASTMACE_MODELS``, head ``matpes_r2scan``) on NaCl 3x3x3, NVT 300 K, in fresh processes, against
the research engine's ``workflow.GPUEngine`` (``GPULangevin``, ``GPULangevinBAOAB``) from the export named
by ``FASTMACE_BASELINE`` (``common.py`` and the OEq launcher: the 1a-T18 export), run in its own process:
- plain cuEq float64: replay equals eager over 200 steps, pipelined and speculative equal checked,
  async output, warm-ups, and the graph node counts of AC, B and C equal the baseline's; positions
  over 500 steps equal the baseline's (checked and pipelined), and one NPT case (BAOAB, hydrostatic);
- the hybrid in float32 (``FASTMACE_HYBRID_PYTHON``): the same comparisons, statistically, and the
  500-step trajectory within the baseline's own run-to-run scatter (two baseline processes).

The process constraints refuse a mixed process: a compiled plain-cuEq engine after a compiled hybrid
one, in a fresh hybrid interpreter with a tiny model.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.build import bulk

from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig
from fastmace.engine import constraints
from fastmace.engine.graph import GraphEngine
from fastmace.integrators.baoab import LangevinBAOAB
from fastmace.integrators.langevin import Langevin
from fastmace.model.forces import ForceOutput
from fastmace.neighbors.skin import SkinNeighbourList, det_3x3
from fastmace.state import SimulationState

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODELS = os.environ.get("FASTMACE_MODELS")
BASELINE = os.environ.get("FASTMACE_BASELINE")
HYBRID_PYTHON = os.environ.get("FASTMACE_HYBRID_PYTHON") or sys.executable
SIGMA, EPSILON, RC = 3.4, 0.0104, 7.0
# In this (pytest) process CUDA was initialised long ago; the allocator constraint is checked in the
# fresh processes below.
IN_PROCESS = dict(expandable_segments="off", compile_mode="none")


class DeterministicLJ:
    """ASE's non-smooth Lennard-Jones with analytic forces summed by a one-hot product (no atomics).

    Deterministic for a given padded edge count, so eager and replayed steps can be compared bitwise.
    """

    def __init__(self, atoms, neighbours, compute_stress=False):
        self.nl = SkinNeighbourList(atoms.cell.array, RC, "cuda", torch.float64, neighbours,
                                    sort_receivers=False, gpu_build=True, variable_cell=compute_stress)
        self._compute_stress = compute_stress
        self.num_edges = 0
        self.atom_ids = torch.arange(len(atoms), device="cuda")
        positions = torch.tensor(atoms.get_positions(), device="cuda")
        self.rebuild(positions)
        self.compact(positions)
        self.use_edges(int(self.nl.count))

    device = torch.device("cuda", 0)
    dtype = torch.float64
    padding_exact = True
    compile_mode = "none"

    @property
    def compute_stress(self):
        return self._compute_stress

    @property
    def edge_count(self):
        return self.nl.count

    @property
    def edge_capacity(self):
        return self.nl.capacity

    def rebuild(self, positions):
        return self.nl.build(positions)

    def use_edges(self, num_edges):
        self.num_edges = num_edges

    def set_cell(self, cell):
        self.nl.set_cell(cell)

    def needs_rebuild(self, positions, fraction=1.0):
        return self.nl.needs_rebuild(positions, fraction)

    def compact(self, positions):
        self.nl.compact(positions)

    def buffers(self):
        return {f"neighbours.{name}": value for name, value in self.nl.buffers().items()}

    def info(self):
        return {"streaming_interactions": 0, "sort_receivers": False}

    def __call__(self, positions):
        sender, receiver = self.nl.edge_index[:, :self.num_edges]
        vectors = positions[receiver] - positions[sender] + self.nl.shifts[:self.num_edges]
        r2 = (vectors * vectors).sum(dim=1)
        inside = r2 < RC * RC
        r2 = torch.where(inside, r2, torch.full_like(r2, RC * RC))
        s6 = (SIGMA * SIGMA / r2) ** 3
        shift = 4 * EPSILON * ((SIGMA / RC) ** 12 - (SIGMA / RC) ** 6)
        pair = torch.where(inside, 4 * EPSILON * (s6 * s6 - s6) - shift, torch.zeros_like(r2))
        # -dE_pair/dr / r, times the vector: force on the receiver from this edge (half of the pair)
        scale = torch.where(inside, 24 * EPSILON * (2 * s6 * s6 - s6) / r2, torch.zeros_like(r2))
        edge_forces = 0.5 * scale[:, None] * vectors
        onehot_receiver = (receiver[None, :] == self.atom_ids[:, None]).to(torch.float64)
        onehot_sender = (sender[None, :] == self.atom_ids[:, None]).to(torch.float64)
        forces = onehot_receiver @ edge_forces - onehot_sender @ edge_forces
        energy = (0.5 * pair.sum()).reshape(1)
        stress = None
        if self._compute_stress:
            virial = -(vectors[:, :, None] * edge_forces[:, None, :]).sum(dim=0)
            stress = virial / det_3x3(self.nl.cell_t)
        return ForceOutput(energy, forces, stress)


def argon(seed=3):
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((3, 3, 3))
    atoms.rattle(0.05, seed=seed)
    return atoms


def build(mode="checked", capture=True, npt=False, neighbours=None, **engine):
    """A test helper standing in for the planner (T15)."""
    atoms = argon()
    neighbours = neighbours or NeighbourConfig(skin=0.3, sort_receivers="off", gpu_neighbour_list="on",
                                               edge_bucket_fraction=0.0, edge_multiple=64)
    field = DeterministicLJ(atoms, neighbours, compute_stress=npt)
    state = SimulationState.from_atoms(atoms, "cuda", 7, compute_stress=npt)
    if npt:
        dyn = LangevinBAOAB(state, IntegratorConfig(ensemble="npt", timestep_fs=2.0, friction=0.02, noise_batch=7,
                                                    pressure_GPa=0.5, barostat_mass_factor=0.01))
    else:
        dyn = Langevin(state, IntegratorConfig(timestep_fs=2.0, friction=0.02, noise_batch=7))
    options = dict(device="cuda:0", pipelined="on" if mode == "pipelined" else "off",
                   speculative_graph=mode == "speculative", **IN_PROCESS)
    config = EngineConfig(**{**options, **engine})
    return GraphEngine(config, field, dyn, state, capture=capture, pad_edges=True,
                       constraints=constraints.ProcessRecord())


def trajectory(engine, steps=200, every=50):
    rows = []
    for step in range(1, steps + 1):
        engine.integrator.set_temperature(150.0 * step / steps + 30.0)
        engine.step()
        if step % every == 0:
            rows.append(engine.sync())
    return rows


def assert_bitwise(rows, expected):
    for got, want in zip(rows, expected, strict=True):
        for key in want:
            assert np.array_equal(got[key], want[key]), f"{key}: max |difference| {np.abs(got[key] - want[key]).max()}"


@pytest.mark.parametrize("npt", [False, True])
def test_replay_equals_eager_bitwise(npt):
    eager = build(capture=False, npt=npt)
    expected = trajectory(eager)
    engine = build(npt=npt)
    rows = trajectory(engine)
    assert engine.info()["neighbour_rebuilds"] >= 1 and eager.info()["neighbour_rebuilds"] >= 1
    assert {"AC", "C"} <= set(engine.info()["graphs"])
    assert_bitwise(rows, expected)


@pytest.mark.parametrize("mode", ["pipelined", "speculative"])
def test_pipelined_and_speculative_equal_checked(mode):
    """Pipelined and speculative steps can pad to other sizes than checked ones, so the sums differ in rounding."""
    if mode == "speculative":
        expected = trajectory(build(mode="pipelined"))
    else:
        expected = trajectory(build())
    engine = build(mode=mode)
    rows = trajectory(engine)
    info = engine.info()
    assert info["neighbour_rebuilds"] >= 1
    for got, want in zip(rows, expected, strict=True):
        for key in want:
            np.testing.assert_allclose(got[key], want[key], rtol=0, atol=1e-12, err_msg=key)


@pytest.mark.parametrize("mode", ["checked", "pipelined", "speculative"])
def test_async_output_is_bit_identical(mode):
    engine = build(mode=mode)
    engine.integrator.set_temperature(120.0)
    for step in range(1, 61):
        engine.step()
        if step % 15 == 0:
            snapshot = engine.snapshot()
            assert snapshot.event is not None and snapshot.step == step
            values = snapshot.wait()
            assert all(value.is_pinned() for value in values.values())
            synced = engine.sync()
            for key, value in values.items():
                assert np.array_equal(value.numpy(), synced[key]), key


@pytest.mark.parametrize("mode", ["checked", "pipelined", "speculative", "npt"])
def test_capture_warmups_leave_the_registered_state(mode):
    engine = build(mode="pipelined" if mode == "npt" else mode, npt=mode == "npt")
    engine.integrator.set_temperature(100.0)
    engine.run(5)  # the stages of the first steps are captured
    engine.flush()
    engine.integrator.prepare_step(engine.state)
    edges = engine.force_field.num_edges
    keys = {"checked": ["AC", "C", ("B", edges)], "pipelined": ["ACP", "C", ("B", edges)],
            "speculative": [("ACB", edges + 64, 0), "C", ("B", edges + 64)], "npt": ["ACP", "C", ("B", edges)]}[mode]
    for key in keys:
        engine._graphs.pop(key, None)
        before = {name: (tensor.data_ptr(), tensor.clone()) for name, tensor in engine.registry.tensors().items()}
        engine._capture(key)
        torch.cuda.synchronize()
        after = engine.registry.tensors()
        assert set(after) == set(before)
        for name, (pointer, value) in before.items():
            assert after[name].data_ptr() == pointer, (key, name)
            assert torch.equal(after[name], value), (key, name)


def test_engine_refuses_auto_and_a_device_mismatch():
    with pytest.raises(ValueError, match="auto"):
        build(pipelined="auto")
    with pytest.raises(ValueError, match="device"):
        build(device="cpu")


# --------------------------------------------------------------------------- MH-1 in fresh processes


def _has_oeq():
    if HYBRID_PYTHON == sys.executable:
        return importlib.util.find_spec("openequivariance") is not None
    code = "import importlib.util, sys; sys.exit(importlib.util.find_spec('openequivariance') is None)"
    return subprocess.run([HYBRID_PYTHON, "-c", code], capture_output=True, timeout=120, check=False).returncode == 0


def _mh1_path():
    if not MODELS:
        return None
    from fastmace.model import labels

    path = Path(MODELS) / labels.LABELS["mh1-r2scan"].checkpoint.file_name
    return path if path.is_file() else None


def _baseline_root():
    if not BASELINE:
        return None
    root = Path(BASELINE)
    needed = (root / "Experiments/Benchmark_MACE_GPU_OPTIMIZATIONS/scripts/common.py",
              root / "Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh")
    return root if all(path.is_file() for path in needed) else None


needs_oeq = pytest.mark.skipif(not _has_oeq(), reason="OpenEquivariance is not installed (hybrid environment needed)")
needs_mh1_baseline = pytest.mark.skipif(
    not (_mh1_path() and _baseline_root()),
    reason="needs FASTMACE_MODELS with MH-1 and FASTMACE_BASELINE with common.py and the OEq launcher (1a-T18 export)")


def run_fresh(code, timeout=3600, python=sys.executable, prefix=(), extra_env=None):
    """Run code in a fresh interpreter; returns the JSON its last stdout line prints."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "tests")]),
               PYTHONDONTWRITEBYTECODE="1", **(extra_env or {}))
    if python != sys.executable:
        env.pop("LD_LIBRARY_PATH", None)  # set for this environment; the in-process OEq runtime needs none
    result = subprocess.run([*prefix, python, "-c", textwrap.dedent(code)], cwd=REPOSITORY_ROOT, env=env,
                            capture_output=True, text=True, timeout=timeout, check=False)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


# Shared by both sides: the structure, the node-counting graph class and the run protocol.
COMMON = """
import gc, json, sys, time
import numpy as np
SPEC = {spec!r}

def structure():
    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
    MaxwellBoltzmannDistribution(atoms, temperature_K=SPEC["temperature"], rng=np.random.default_rng(11))
    return atoms

def node_kinds(raw):
    # node types of a captured graph, kernels by their function name
    from cuda.bindings import driver
    graph = driver.CUgraph(raw)
    _, _, count = driver.cuGraphGetNodes(graph, 0)
    _, nodes, count = driver.cuGraphGetNodes(graph, count)
    kinds = []
    for node in nodes:
        _, kind = driver.cuGraphNodeGetType(node)
        name = kind.name
        if kind == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            _, params = driver.cuGraphKernelNodeGetParams(node)
            _, function = driver.cuFuncGetName(params.func)
            name = function.decode() if isinstance(function, bytes) else str(function)
        kinds.append(name)
    return kinds

def count_graph_nodes(torch):

    class CountingGraph(torch.cuda.CUDAGraph):
        # keep_graph=True keeps the captured cudaGraph_t, whose nodes are counted after the capture.
        def __new__(cls, keep_graph=False):
            return super().__new__(cls, True)

        def __init__(self, keep_graph=False):
            super().__init__(True)

        def capture_end(self):
            super().capture_end()
            self.node_kinds = node_kinds(self.raw_cuda_graph())
            self.node_count = len(self.node_kinds)

    torch.cuda.CUDAGraph = CountingGraph

def protocol(torch, step, positions, graphs, flush, snapshot=None):
    rows, async_equal = [], True
    start = time.perf_counter()
    for n in range(1, SPEC["steps"] + 1):
        if n == SPEC["steps"] - SPEC["timed"]:
            flush(); torch.cuda.synchronize(); start = time.perf_counter()
        step()
        if n % SPEC["every"] == 0:
            if snapshot is not None:
                async_equal &= snapshot()
            flush()
            rows.append(positions().detach().cpu().tolist())
    flush(); torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / min(SPEC["timed"], SPEC["steps"]) * 1e3
    nodes = {{str(key): getattr(graph, "node_count", None) for key, graph in graphs().items()}}
    kinds = {{str(key): getattr(graph, "node_kinds", None) for key, graph in graphs().items()}}
    return {{"positions": rows, "ms_per_step": elapsed, "nodes": nodes, "kinds": kinds, "async_equal": async_equal}}
"""

FASTMACE_CHILD = """
import fastmace
from fastmace.model import compile as fc
allocator = fc.configure_allocator()  # before the first CUDA allocation
import torch
count_graph_nodes(torch)
from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig
from fastmace.engine.graph import GraphEngine
from fastmace.integrators.baoab import LangevinBAOAB
from fastmace.integrators.langevin import Langevin
from fastmace.model import loading
from fastmace.model.forces import MACEForceField
from fastmace.rewrites import registry as reg
from fastmace.state import SimulationState

loaded = loading.load("mh1-r2scan", backend=SPEC["backend"], dtype=SPEC["dtype"], device="cuda:0")
names = reg.default_registry().names()
rewrites = {{name: "on" if name in SPEC["rewrites"] else "off" for name in names}}
cache = fc.CompiledModelCache()
npt = SPEC["ensemble"] == "npt"

def build(mode):
    atoms = structure()
    field = MACEForceField(loaded, atoms, rewrites=rewrites, engine=EngineConfig(compile_mode="default"),
                           neighbours=NeighbourConfig(skin=0.3, sort_receivers=SPEC["sort"], gpu_neighbour_list="on"),
                           compute_stress=npt, compile_cache=cache, float64_energy=SPEC["float64_energy"])
    state = SimulationState.from_atoms(atoms, "cuda:0", SPEC["seed"], compute_stress=npt)
    if npt:
        dyn = LangevinBAOAB(state, IntegratorConfig(ensemble="npt", timestep_fs=0.5, friction=0.02),
                            temperature_K=SPEC["temperature"])
    else:
        dyn = Langevin(state, IntegratorConfig(timestep_fs=0.5, friction=0.02), temperature_K=SPEC["temperature"])
    config = EngineConfig(compile_mode="default", pipelined="on" if mode == "pipelined" else "off",
                          speculative_graph=mode == "speculative")
    return GraphEngine(config, field, dyn, state, capture=mode != "eager", pad_edges=True)

def snapshot_check(engine):
    def check():
        values = engine.snapshot().wait()
        synced = engine.sync()
        return all(np.array_equal(value.numpy(), synced[key]) for key, value in values.items())
    return check

def warmup_check(engine, keys):
    # capture warm-ups leave every runtime tensor as it was, at the same address
    engine.integrator.prepare_step(engine.state)
    bad = []
    for key in keys:
        # detach(): the force field's positions are a leaf that requires grad. A plain clone made here, on
        # the default stream and kept alive across the capture, creates the leaf's AccumulateGrad node
        # on that stream, and the hybrid's eager autograd.grad then refuses the capture ("stale
        # reference to the default stream from warmup"; 1b-T03b). The registry's own snapshot detaches.
        before = {{name: (t.data_ptr(), t.detach().clone()) for name, t in engine.registry.tensors().items()}}
        engine._capture(key)
        torch.cuda.synchronize()
        after = engine.registry.tensors()
        bad += [f"{{key}}:{{name}}" for name, (pointer, value) in before.items()
                if after[name].data_ptr() != pointer or not torch.equal(after[name], value)]
    return bad

result = {{"allocator": allocator, "modes": {{}}, "warmups": {{}}}}
for mode, steps in SPEC["modes"]:
    SPEC["steps"] = steps
    engine = build(mode)
    out = protocol(torch, engine.step, lambda: engine.state.positions, lambda: engine._graphs, engine.flush,
                   snapshot_check(engine) if mode != "eager" else None)
    out["info"] = {{k: v for k, v in engine.info().items() if k != "capture_memory"}}
    out["force_field"] = {{k: v for k, v in engine.force_field.info().items() if k != "decisions"}}
    result["modes"][mode] = out
    if SPEC["warmups"] and mode in ("checked", "pipelined", "speculative"):
        edges = max(engine.edge_sizes_used)
        keys = {{"checked": ["AC", "C", ("B", edges)], "pipelined": ["ACP", ("B", edges)],
                "speculative": [("ACB", edges, 0)]}}[mode]
        result["warmups"][mode] = warmup_check(engine, keys)
    del engine, out
    gc.collect(); torch.cuda.empty_cache()
result["torch"] = torch.__version__
print(json.dumps(result))
"""

BASELINE_CHILD = """
import os
sys.path[:0] = [{engine!r}, {scripts!r}]
from langevin import gpu_langevin
allocator = gpu_langevin.configure_allocator()  # before the first CUDA allocation
import common
import torch
count_graph_nodes(torch)
from langevin import upstream_mace, workflow
upstream_mace.use_single_graph_strain(True)
upstream_mace.use_cueq_conv_fusion(True)
hybrid = SPEC["backend"] == "hybrid"
calc = common.load_calculator("mh1-r2scan", dtype=SPEC["dtype"], enable_cueq=True, enable_oeq=hybrid)
upstream_mace.require_fused_cueq(calc.models[0])
workflow.configure_md_calculator(calc, compute_stress=False)
result = {{"allocator": allocator, "modes": {{}}}}
for mode, steps in SPEC["modes"]:
    SPEC["steps"] = steps
    engine = workflow.GPUEngine(structure(), calc, timestep_fs=0.5, friction=0.02, seed=SPEC["seed"], skin=0.3,
                                compile_mode="default", gpu_neighbour_list=True, sort_receivers=SPEC["sort"] == "on",
                                pipelined=mode == "pipelined", speculative_graph=mode == "speculative",
                                ensemble=SPEC["ensemble"], **SPEC["baseline_options"])
    dyn = engine.dyn
    out = protocol(torch, lambda: engine.step(SPEC["temperature"]), lambda: dyn.x, lambda: dyn._graphs,
                   dyn.flush_speculative)
    out["edge_sizes_used"] = sorted(dyn.edge_sizes_used)
    out["rebuilds"] = dyn.neighbours.rebuilds
    result["modes"][mode] = out
    del engine, dyn, out
    gc.collect(); torch.cuda.empty_cache()
result["torch"] = torch.__version__
print(json.dumps(result))
"""

MH1_PACKAGE = ("factor_species", "fold_cutoff", "density_first", "joint_radial_input", "fused_folded_backward",
               "reassociate_first", "segmented_first_sum")
LINEAR = ("block_linear", "grouped_linear", "merged_residual")
FIRST = ("reassociate_first", "segmented_first_sum")
# Production rewrites as main.py runs them (1a-T11's MH-1 cases): plain cuEq without reassociation (the
# probe refuses plain cuEq), the hybrid with it and with streaming convolutions on receiver-sorted edges.
CUEQ_REWRITES = tuple(name for name in (*MH1_PACKAGE, *LINEAR, "single_graph_strain") if name not in FIRST)
HYBRID_REWRITES = (*MH1_PACKAGE, *LINEAR, "single_graph_strain", "streaming_conv")


def _baseline_options(rewrites):
    """GPUEngine keywords for the same rewrites (workflow.GPUEngine sets the research engine's switches)."""
    options = {name: name in rewrites for name in (*MH1_PACKAGE, *LINEAR, "streaming_conv")}
    options["species_skip"] = False
    return options


def spec(backend, ensemble="nvt", modes=(), warmups=False, float64_energy=False):
    """The run of both children. ``float64_energy`` is False against the research engine, which sums the
    energy in the model's precision (1b-T03): its float64 reduction adds kernels to stage B."""
    rewrites = HYBRID_REWRITES if backend == "hybrid" else CUEQ_REWRITES
    return {"backend": backend, "dtype": "float32" if backend == "hybrid" else "float64", "ensemble": ensemble,
            "float64_energy": float64_energy,
            "rewrites": rewrites, "sort": "on" if backend == "hybrid" else "off", "seed": 20261018,
            "temperature": 300.0, "every": 50, "timed": 100, "modes": list(modes), "warmups": warmups,
            "baseline_options": _baseline_options(rewrites)}


def fastmace_run(run_spec, python=sys.executable):
    return run_fresh(COMMON.format(spec=run_spec) + FASTMACE_CHILD.format(), python=python)


def baseline_run(run_spec, python=sys.executable):
    root = _baseline_root()
    code = COMMON.format(spec=run_spec) + BASELINE_CHILD.format(
        engine=str(root / "Experiments/Langevin_Thermostat"),
        scripts=str(root / "Experiments/Benchmark_MACE_GPU_OPTIMIZATIONS/scripts"))
    prefix = ()
    if run_spec["backend"] == "hybrid":  # the research engine's hybrid runs through its launcher (SPEC §3.2)
        cache = Path(os.environ.get("FASTMACE_CACHE", REPOSITORY_ROOT / ".cache")) / "oeq-baseline-1a-T13"
        prefix = ("bash", str(root / "Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh"),
                  str(Path(python).parents[1]), str(cache))
    return run_fresh(code, python=python, prefix=prefix, extra_env={"MACE_BENCH_CHECKPOINT": str(_mh1_path())})


def deviation(rows, reference, upto=None):
    """Largest |position difference| (Å) over the recorded steps (the first ``upto`` records)."""
    rows, reference = rows[:upto], reference[:upto]
    assert len(rows) == len(reference)
    return float(max(np.abs(np.subtract(a, b)).max() for a, b in zip(rows, reference)))


CUEQ_KERNEL = re.compile(r"segmented_polynomial(_bwd)?_(?:split(\d+)_of_\d+_)?algo\d+_\d+")


def kernel_sequence(kinds):
    """A graph's node sequence with cuEquivariance's run-time algorithm choice taken out.

    cuEq picks a variant of each segmented-polynomial kernel when it runs (``algo1_4``, ``algo2_2``,
    or one split into two launches), and the choice differs between processes and engines: the research
    engine's B graph of MH-1 NaCl had 242 nodes in one run and 239 in the next. Each such launch (or
    split pair) counts as one node of its operation here; everything else must match by name.
    """
    out = []
    for name in kinds:
        match = CUEQ_KERNEL.fullmatch(name)
        if match:
            if match.group(2) not in (None, "1"):
                continue
            name = "segmented_polynomial" + (match.group(1) or "")
        out.append(name)
    return out


def same_kernels(ours, theirs):
    """Per graph key: (our node count, theirs, equal kernel sequences after kernel_sequence)."""
    out = {key: (len(kinds), len(theirs[key]) if key in theirs else None,
                 key in theirs and kernel_sequence(kinds) == kernel_sequence(theirs[key]))
           for key, kinds in ours.items()}
    out.update({key: (None, len(kinds), False) for key, kinds in theirs.items() if key not in ours})
    return out


def report(name, payload):
    print(f"\n{name}: " + json.dumps(payload, indent=1, default=str))
    directory = os.environ.get("FASTMACE_RESULTS")
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / f"{name}.json").write_text(json.dumps(payload, indent=1, default=str) + "\n")


# Plain cuEq float64 is not bitwise reproducible between engines: cuEq's run-time choice of kernel
# variants (kernel_sequence) changes the summation order. A 500-step trajectory then moves by about
# 1e-13 A between two runs of the same engine (measured below, "repeat").
FLOAT64_LIMIT = 1e-12


@pytest.mark.slow
@pytest.mark.acceptance
@needs_mh1_baseline
def test_mh1_plain_cueq_float64_against_the_research_engine():
    ours = fastmace_run(spec("cueq", modes=[("eager", 200), ("checked", 500), ("repeat", 500), ("pipelined", 500),
                                            ("speculative", 200)], warmups=True))
    theirs = baseline_run(spec("cueq", modes=[("checked", 500), ("repeat", 500), ("pipelined", 500)]))
    modes, base = ours["modes"], theirs["modes"]
    checked, pipelined = modes["checked"], modes["pipelined"]
    summary = {
        "replay_vs_eager_200": deviation(checked["positions"], modes["eager"]["positions"], 4),
        "fastmace_repeat_500": deviation(modes["repeat"]["positions"], checked["positions"]),
        "baseline_repeat_500": deviation(base["repeat"]["positions"], base["checked"]["positions"]),
        "pipelined_vs_checked_500": deviation(pipelined["positions"], checked["positions"]),
        "speculative_vs_pipelined_200": deviation(modes["speculative"]["positions"], pipelined["positions"], 4),
        "baseline_checked_500": deviation(checked["positions"], base["checked"]["positions"]),
        "baseline_pipelined_500": deviation(pipelined["positions"], base["pipelined"]["positions"]),
        "kernels": {mode: same_kernels(modes[mode]["kinds"], base[mode]["kinds"]) for mode in ("checked", "pipelined")},
        "ms_per_step": {mode: {"fastmace": modes[mode]["ms_per_step"], "baseline": base[mode]["ms_per_step"]}
                        for mode in ("checked", "pipelined")},
        "info": {mode: out["info"] for mode, out in modes.items()},
        "baseline_edge_sizes": {mode: out["edge_sizes_used"] for mode, out in base.items()},
        "baseline_builds": {mode: out["rebuilds"] for mode, out in base.items()},
        "warmups": ours["warmups"], "allocator": [ours["allocator"], theirs["allocator"]], "torch": ours["torch"],
    }
    report("1a-T13-mh1-cueq-float64-nvt", summary)
    assert checked["info"]["neighbour_rebuilds"] >= 1
    for name in ("replay_vs_eager_200", "fastmace_repeat_500", "pipelined_vs_checked_500",
                 "speculative_vs_pipelined_200", "baseline_checked_500", "baseline_pipelined_500"):
        assert summary[name] <= FLOAT64_LIMIT, (name, summary[name])
    assert all(out["async_equal"] for out in modes.values())
    assert not any(ours["warmups"].values()), ours["warmups"]
    for mode in ("checked", "pipelined"):
        assert modes[mode]["info"]["edge_sizes_used"] == base[mode]["edge_sizes_used"]
        # The baseline counts the force field's first build too.
        assert modes[mode]["info"]["neighbour_rebuilds"] + 1 == base[mode]["rebuilds"]
        assert all(equal for _, _, equal in summary["kernels"][mode].values()), summary["kernels"][mode]


@pytest.mark.slow
@pytest.mark.acceptance
@needs_mh1_baseline
def test_mh1_plain_cueq_float64_npt_against_the_research_engine():
    options = spec("cueq", ensemble="npt", modes=[("checked", 500), ("pipelined", 500)])
    ours = fastmace_run(options)
    theirs = baseline_run(options)
    modes, base = ours["modes"], theirs["modes"]
    summary = {mode: {"baseline_500": deviation(modes[mode]["positions"], base[mode]["positions"]),
                      "kernels": same_kernels(modes[mode]["kinds"], base[mode]["kinds"]),
                      "ms_per_step": [modes[mode]["ms_per_step"], base[mode]["ms_per_step"]],
                      "builds": [modes[mode]["info"]["neighbour_rebuilds"] + 1, base[mode]["rebuilds"]],
                      "edge_sizes": [modes[mode]["info"]["edge_sizes_used"], base[mode]["edge_sizes_used"]]}
               for mode in ("checked", "pipelined")}
    summary["pipelined_vs_checked_500"] = deviation(modes["pipelined"]["positions"], modes["checked"]["positions"])
    report("1a-T13-mh1-cueq-float64-npt", summary)
    assert all(out["async_equal"] for out in modes.values())
    assert summary["pipelined_vs_checked_500"] <= FLOAT64_LIMIT
    for mode in ("checked", "pipelined"):
        assert summary[mode]["baseline_500"] <= FLOAT64_LIMIT, (mode, summary[mode])
        assert all(equal for _, _, equal in summary[mode]["kernels"].values()), summary[mode]["kernels"]


@pytest.mark.slow
@pytest.mark.acceptance
@needs_oeq
@needs_mh1_baseline
def test_mh1_hybrid_float32_against_the_research_engine():
    ours = fastmace_run(spec("hybrid", modes=[("eager", 200), ("checked", 500), ("repeat", 500), ("pipelined", 500),
                                              ("speculative", 200)], warmups=True), python=HYBRID_PYTHON)
    first = baseline_run(spec("hybrid", modes=[("checked", 500)]), python=HYBRID_PYTHON)
    second = baseline_run(spec("hybrid", modes=[("checked", 500)]), python=HYBRID_PYTHON)
    modes = ours["modes"]
    reference = first["modes"]["checked"]["positions"]
    scatter = deviation(second["modes"]["checked"]["positions"], reference)
    summary = {
        "replay_vs_eager_200": deviation(modes["checked"]["positions"], modes["eager"]["positions"], 4),
        "fastmace_repeat_500": deviation(modes["repeat"]["positions"], modes["checked"]["positions"]),
        "pipelined_vs_checked_500": deviation(modes["pipelined"]["positions"], modes["checked"]["positions"]),
        "speculative_vs_pipelined_200": deviation(modes["speculative"]["positions"], modes["pipelined"]["positions"],
                                                  4),
        "baseline_scatter_500": scatter,
        "baseline_checked_500": deviation(modes["checked"]["positions"], reference),
        "baseline_scatter_per_record": [deviation([a], [b]) for a, b in
                                        zip(second["modes"]["checked"]["positions"], reference)],
        "fastmace_per_record": [deviation([a], [b]) for a, b in zip(modes["checked"]["positions"], reference)],
        "kernels": same_kernels(modes["checked"]["kinds"], first["modes"]["checked"]["kinds"]),
        "ms_per_step": {"fastmace": {mode: out["ms_per_step"] for mode, out in modes.items()},
                        "baseline": [first["modes"]["checked"]["ms_per_step"],
                                     second["modes"]["checked"]["ms_per_step"]]},
        "info": {mode: out["info"] for mode, out in modes.items()},
        "baseline_builds": [first["modes"]["checked"]["rebuilds"], second["modes"]["checked"]["rebuilds"]],
        "streaming": modes["checked"]["force_field"]["streaming_interactions"], "warmups": ours["warmups"],
    }
    report("1a-T13-mh1-hybrid-float32-nvt", summary)
    assert summary["streaming"] > 0 and modes["checked"]["info"]["neighbour_rebuilds"] >= 1
    assert all(out["async_equal"] for out in modes.values())
    assert not any(ours["warmups"].values()), ours["warmups"]
    # Statistical: float32 rounding differs between eager and compiled-graph kernels and between runs;
    # trajectories from one seed stay within the run-to-run scatter of the research engine.
    limit = max(3 * scatter, 1e-6)
    for name in ("replay_vs_eager_200", "fastmace_repeat_500", "pipelined_vs_checked_500",
                 "speculative_vs_pipelined_200", "baseline_checked_500"):
        assert summary[name] <= limit, (name, summary[name], limit)
    assert all(equal for _, _, equal in summary["kernels"].values()), summary["kernels"]


MIXED_CHILD = """
import json
import fastmace
from fastmace.model import compile as fc
fc.configure_allocator()
from fastmace.model import loading
hybrid = loading.load({path!r}, backend="hybrid", dtype="float64", device="cuda:0", trusted=True)
plain = loading.load({path!r}, backend="cueq", dtype="float64", device="cuda:0", trusted=True)
import torch
from ase.build import bulk
from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig
from fastmace.engine import constraints
from fastmace.engine.graph import GraphEngine
from fastmace.integrators.langevin import Langevin
from fastmace.model.forces import MACEForceField
from fastmace.state import SimulationState

def engine(model, compile_mode):
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    atoms.numbers[3] = 8
    atoms.rattle(0.05, seed=1)
    config = EngineConfig(compile_mode=compile_mode, pipelined="on")
    field = MACEForceField(model, atoms, engine=config, neighbours=NeighbourConfig(sort_receivers="on",
                           gpu_neighbour_list="on"))
    state = SimulationState.from_atoms(atoms, "cuda:0", 1)
    return GraphEngine(config, field, Langevin(state, IntegratorConfig()), state)

first = engine(hybrid, "default")
first.run(3)
result = {{"family": constraints.PROCESS.family, "eager_plain": None, "compiled_plain": None}}
second = engine(plain, "none")  # uncompiled: no compile policy involved, allowed
second.run(3)
result["eager_plain"] = "ok"
try:
    engine(plain, "default")
except constraints.ProcessConstraintError as error:
    result["compiled_plain"] = str(error)
result["engines"] = constraints.PROCESS.engines
print(json.dumps(result))
"""


@needs_oeq
def test_process_constraints_refuse_a_mixed_process(tmp_path):
    from _support.tiny_models import FAMILIES, tiny_mace_model

    path = tmp_path / "tiny-mh1.model"
    torch.save(tiny_mace_model([8, 11, 17, 26], **FAMILIES["mh1"]), path)
    out = run_fresh(MIXED_CHILD.format(path=str(path)), python=HYBRID_PYTHON)
    print(json.dumps(out, indent=1))
    assert out["family"] == constraints.OEQ_FAMILY
    assert out["eager_plain"] == "ok" and out["engines"] == 2
    assert out["compiled_plain"] and "process" in out["compiled_plain"]


# --------------------------------------------------------------------------- no compile at a later edge bucket

BUCKET_CHILD = """
import json, time
import numpy as np
import fastmace
from fastmace.config import EngineConfig, IntegratorConfig
from fastmace.planner import fixed

planner = fixed.FixedPlanner(engine=EngineConfig(backend={backend!r}))
hardware = planner.hardware()
loaded = planner.load("mh1-r2scan")
import torch
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from torch._dynamo.utils import counters

atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
MaxwellBoltzmannDistribution(atoms, temperature_K=900.0, rng=np.random.default_rng(11))
plan = planner.plan(loaded, atoms, IntegratorConfig(timestep_fs=0.5, friction=0.02, seed=20261018), hardware)
engine = planner.build(plan, loaded, atoms, temperature_K=900.0)

def compiled():
    return {{"unique_graphs": counters["stats"]["unique_graphs"], "frames_ok": counters["frames"]["ok"]}}

for _ in range({warmup}):
    engine.step()
engine.flush()
if {foreign_mask_head}:
    # What the research engine's use_capture_safe_mask_head does before each of its captures in a
    # shared process: another capture-safe mask_head in mace.modules.blocks.
    from mace.modules import blocks
    from fastmace.rewrites import engine as engine_rewrites

    def foreign_mask_head(x, heads, num_heads):
        return engine_rewrites.capture_safe_mask_head(x, heads, num_heads)
    blocks.mask_head = foreign_mask_head
after_warmup, sizes_after_warmup = compiled(), sorted(engine.edge_sizes_used)
events, seen = [], set(engine.edge_sizes_used)
for n in range({warmup} + 1, {steps} + 1):
    start = time.perf_counter()
    engine.step()
    engine.flush()
    torch.cuda.synchronize()
    if set(engine.edge_sizes_used) != seen:
        events.append({{"step": n, "new_sizes": sorted(set(engine.edge_sizes_used) - seen), "after": compiled(),
                        "step_s": time.perf_counter() - start}})
        seen = set(engine.edge_sizes_used)
print(json.dumps({{"after_warmup": after_warmup, "end": compiled(), "sizes_after_warmup": sizes_after_warmup,
                  "sizes": sorted(engine.edge_sizes_used), "events": events, "edge_bucket": engine.edge_bucket,
                  "initial_edges": engine.initial_edges}}))
"""


def _bucket_crossing(backend, python, directory):
    # Fresh compiler caches: dynamo's automatic-dynamic profile persists in the Inductor cache
    # directory, and a warm one would hide a compile that a cold process does.
    caches = {"TORCHINDUCTOR_CACHE_DIR": str(directory / "inductor"), "TRITON_CACHE_DIR": str(directory / "triton"),
              "FASTMACE_CACHE": str(directory / "fastmace")}
    out = run_fresh(BUCKET_CHILD.format(backend=backend, warmup=20, steps=200, foreign_mask_head=True),
                    python=python, extra_env=caches)
    print(json.dumps(out, indent=1))
    # At 900 K the edge count of NaCl 3x3x3 grows from 6912 (padded to 7168 in warm-up) into the
    # next bucket (7680) within 200 steps; a new padded size needs a capture, but no compile:
    # warm-up saw the exact initial count and a padded size, so the edge dimension is dynamic.
    # The compiled model guards on mace.modules.blocks.mask_head, which another engine in the
    # process (the research engine in T19's A/B) replaces; the capture puts FastMACE's back first.
    assert out["events"], "the run never reached a new edge bucket after warm-up"
    assert out["end"] == out["after_warmup"], out


@pytest.mark.slow
@pytest.mark.skipif(not _mh1_path(), reason="needs FASTMACE_MODELS with MH-1")
def test_no_compile_at_a_later_edge_bucket_plain_cueq(tmp_path):
    _bucket_crossing("cueq", sys.executable, tmp_path)


@pytest.mark.slow
@needs_oeq
@pytest.mark.skipif(not _mh1_path(), reason="needs FASTMACE_MODELS with MH-1")
def test_no_compile_at_a_later_edge_bucket_hybrid(tmp_path):
    _bucket_crossing("hybrid", HYBRID_PYTHON, tmp_path)
