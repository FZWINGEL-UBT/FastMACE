"""The tiled call's speed switches (results/tiled-speed-20261108) on CUDA with MACE-OMAT-0 medium.

``rewrites.generic.modules.TILE_BLOCK_ROWS`` (the w' convolution's block-layout message kept for a tile's receivers),
``model.tiled.TILE_VIRIAL`` (the strain gradient from the edge vectors' gradient) and ``model.tiled.
TILE_PREFETCH_DEVICE`` (the next tile's graph built ahead with the state on the device): each switch on against off
on the same tiled force field, and the tiled call against the whole-system call of the same plan, on SrTiO3 6x6x6
(1,080 atoms, rattled and strained) in tiles of 256 receivers, eager and padded. The forces are summed with atomics
on the GPU, and so are the forward's sums: the sides agree to float32 rounding, not bitwise, energy included. Needs
``FASTMACE_MODELS`` (the checkpoint) and the hybrid backend's kernels; the w' path is on only where decision 66 lists
the GPU for OMAT-0 medium (otherwise the test skips). Runs in a fresh interpreter (the OEq runtime is prepared before
MACE is imported).
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fastmace.kernels import streaming_available

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODELS = os.environ.get("FASTMACE_MODELS")

pytestmark = [pytest.mark.skipif(not streaming_available(), reason=streaming_available().reason),
              pytest.mark.skipif(not MODELS or not Path(MODELS, "mace-omat-0-medium.model").exists(),
                                 reason="needs FASTMACE_MODELS with mace-omat-0-medium.model")]

CHILD = r"""
import fastmace  # before torch: soft file locks (SPEC §14)
import dataclasses, json, sys

import numpy as np
import torch

import fastmace as fm
from fastmace.model import tiled as tiled_module
from fastmace.model.tiled import TileConfig, TiledForceField
from fastmace.planner import minimal
from fastmace.rewrites.generic import modules as generic

sys.path.insert(0, "tools/bench")
import tiled_capacity

atoms = tiled_capacity.supercell("SrTiO3", (6, 6, 6), 0.05, 0.01)
user = fm.ForceField(fm.load("omat0-medium"), atoms, precision="float32")
planner = minimal.default_planner(user)
plan = planner.plan_evaluation(user, atoms)
eager = dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, compile_mode="none", dynamic_edges=False))
field = planner.build_evaluation(eager, user, atoms)
applied = [getattr(record, "name", str(record)) for record in getattr(field, "applied", ())]
report = {"applied": applied}
if "generic_wprime_backward" not in applied:
    print(json.dumps(dict(report, skip="generic_wprime_backward is off on this GPU")))
    raise SystemExit(0)
whole = field.evaluate(atoms, stress=True)


def run(mode, switches):
    saved = {}
    for (module, name), value in switches.items():
        saved[(module, name)] = getattr(module, name)
        setattr(module, name, value)
    try:
        tiled = TiledForceField(field, TileConfig(256, mode=mode))
        tiled.evaluate(atoms, stress=True)  # the first call in a process may round differently
        energy, forces, stress = tiled.evaluate(atoms, stress=True)
        tiles = tiled.last["tiles"]
    finally:
        for (module, name), value in saved.items():
            setattr(module, name, value)
    return float(energy), np.asarray(forces, dtype=np.float64), np.asarray(stress, dtype=np.float64), tiles


def relative(got, expected):
    return float(np.abs(got - expected).max() / np.abs(expected).max())


cases = {
    "block_rows": (generic, "TILE_BLOCK_ROWS"),
    "virial": (tiled_module, "TILE_VIRIAL"),
    "prefetch": (tiled_module, "TILE_PREFETCH_DEVICE"),
}
w_energy, w_forces, w_stress = float(whole[0]), np.asarray(whole[1], np.float64), np.asarray(whole[2], np.float64)
for mode in ("eager", "padded"):
    for name, (module, constant) in cases.items():
        if name == "prefetch" and mode == "eager":
            continue  # prefetch builds padded tiles' graphs only
        off = run(mode, {(module, constant): False})
        on = run(mode, {(module, constant): True})
        report[f"{mode}-{name}"] = dict(
            tiles=on[3], energy_bitwise=on[0] == off[0], energy_rel=abs(on[0] - off[0]) / abs(off[0]),
            forces=relative(on[1], off[1]), stress=relative(on[2], off[2]),
            whole_energy_rel=abs(on[0] - w_energy) / abs(w_energy), whole_forces=relative(on[1], w_forces),
            whole_stress=relative(on[2], w_stress))
print(json.dumps(report))
"""


def test_tiled_speed_switches_agree_with_the_switch_off_and_the_whole_system():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), os.environ.get("PYTHONPATH", "")]),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(CHILD)], cwd=REPOSITORY_ROOT, env=env,
                            capture_output=True, text=True, timeout=1800, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    print(json.dumps(report, indent=1))
    if "skip" in report:
        pytest.skip(report["skip"])
    cases = {key: value for key, value in report.items() if key != "applied"}
    assert set(cases) == {"eager-block_rows", "eager-virial", "padded-block_rows", "padded-virial",
                          "padded-prefetch"}
    for name, case in cases.items():
        assert case["tiles"] > 1, name  # halos on every tile
        # The forward's sums are float32 atomics too, so even an exact switch moves the energy by rounding (1.9e-9
        # relative seen for the virial, which leaves the forward as it was); TILE_BLOCK_ROWS also multiplies
        # linear_1 on the block layout, as the whole-system call does.
        assert case["energy_rel"] <= 1e-8, (name, case)
        assert case["forces"] <= 1e-5, (name, case)  # atomics' order (float32)
        assert case["stress"] <= 1e-5, (name, case)  # the virial: another summation order
        assert case["whole_energy_rel"] <= 1e-6, (name, case)
        assert case["whole_forces"] <= 1e-4, (name, case)
        assert case["whole_stress"] <= 1e-4, (name, case)


@pytest.mark.parametrize("reach", [60.0, 900.0], ids=["small-cell", "large-coordinates"])
def test_filter32_keeps_exactly_the_float64_tests_edges_on_cuda(reach):
    """``model.tiled.TILE_FILTER32`` on CUDA: the float32 test with the float64 one in its band keeps the same edges
    and shifts as the float64 test (CUDA's own float64 kernels), with a quarter of the pairs within 1e-7 A of the
    cutoff."""
    import torch

    from fastmace.model.tiled import _filter32, _filter32_band
    from fastmace.neighbors.skin import times_3x3

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(11)
    cutoff, n, e = 6.0, 200_000, 4_000_000
    cell = torch.tensor([[23.3, 0.0, 0.0], [0.4, 22.1, 0.0], [0.2, -0.3, 21.7]], dtype=torch.float64, device=device)
    positions64 = ((torch.rand(n, 3, dtype=torch.float64, generator=generator) * 2 - 1) * (reach - 25)).to(device)
    sender = torch.randint(0, n, (e,), generator=generator).to(device)
    receiver = torch.randint(0, n // 2, (e,), generator=generator).to(device)
    codes = torch.randint(0, 27, (e,), generator=generator).to(device)
    index = torch.arange(27, device=device)
    table = torch.stack([index // 9, index // 3 % 3, index % 3], 1).double() - 1
    shifts27 = times_3x3(table, cell)
    unit = table[codes]
    near = torch.arange(n // 2, device=device)
    sender[near] = near + n // 2
    direction = torch.nn.functional.normalize(torch.randn(n // 2, 3, dtype=torch.float64, generator=generator), dim=1)
    length = cutoff + (torch.rand(n // 2, dtype=torch.float64, generator=generator) * 2 - 1) * 1e-7
    positions64[sender[near]] = (positions64[receiver[near]] + shifts27[codes[near]]
                                 - direction.to(device) * length.to(device).unsqueeze(1))
    band = _filter32_band(cutoff, float(positions64.abs().max()) + float(shifts27.abs().max()))
    got = _filter32(positions64.float(), positions64, shifts27, band, cutoff, sender, receiver, unit, codes,
                    torch.float32)
    vectors = positions64[receiver] - positions64[sender] + times_3x3(unit, cell)
    keep = (vectors * vectors).sum(dim=1) < cutoff ** 2
    expected = sender[keep], receiver[keep], unit[keep], times_3x3(unit, cell)[keep].to(torch.float32)
    assert int(((vectors.norm(dim=1) - cutoff).abs() < 1e-6).sum()) > 50_000
    for a, b in zip(got, expected):
        assert torch.equal(a, b)
