"""``atom_chunks`` (plans/capacity.md CAP-T08): the node-local segments on atom chunks with recomputation.

On the CPU in float64:

- ``modules.chunked_forward`` against the unchunked forward of a node-local module, values and input gradients;
  arguments passed by keyword, a ``None`` and a non-per-atom tensor; too few atoms run unchunked;
- a deep copy rebinds the chunked forward and the forward before it to the copy;
- the record on a tiny MH-1-class cuEquivariance model (naive kernels): it chunks every ``gated_rows`` chain and
  product block, passes its self-check, and keeps the model's energy and forces to rounding on 512 atoms.
"""

import copy
import importlib.util

import pytest
import torch

from fastmace.config import RewriteConfig
from fastmace.rewrites import chunks, fusion, linear, mh1
from fastmace.rewrites import registry as reg
from fastmace.rewrites.chunks import modules as cm

pytest.importorskip("mace")
CUEQ = pytest.mark.skipif(importlib.util.find_spec("cuequivariance_torch") is None,
                          reason="cuequivariance_torch is not installed")

from test_rewrites_mh1 import PACKAGE, energy_and_forces, relative, tiny  # noqa: E402

LINEAR = ("block_linear", "grouped_linear", "merged_residual")


class NodeLocal(torch.nn.Module):
    """A node-local stand-in: per-atom rows in, per-atom rows out, a skip term and a shared table."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 5, dtype=torch.float64)

    def forward(self, x, sc=None, table=None):
        out = torch.nn.functional.silu(self.linear(x)) * table.sum()
        return out if sc is None else out + sc


def _inputs(rows, generator):
    x = torch.randn((rows, 6), generator=generator, dtype=torch.float64).requires_grad_()
    sc = torch.randn((rows, 5), generator=generator, dtype=torch.float64).requires_grad_()
    table = torch.randn((3,), generator=generator, dtype=torch.float64).requires_grad_()
    return x, sc, table


@pytest.mark.parametrize("count", [2, 3, 5])
def test_the_chunked_forward_equals_the_unchunked_one(count):
    generator = torch.Generator().manual_seed(count)
    module = NodeLocal()
    reference = copy.deepcopy(module)
    cm.chunk(module, count)
    rows = count * cm.MIN_ATOMS_PER_CHUNK + 7
    x, sc, table = _inputs(rows, generator)
    grad = torch.randn((rows, 5), generator=generator, dtype=torch.float64)
    results = []
    for function in (reference, module):
        out = function(x, sc=sc, table=table)
        results.append((out.detach(), *torch.autograd.grad(out, (x, sc, table), grad)))
    for a, b in zip(*results):
        torch.testing.assert_close(b, a, rtol=1e-13, atol=1e-13)
    out = module(x, None, table=table)  # positional and keyword arguments, a None
    torch.testing.assert_close(out, reference(x, None, table=table), rtol=1e-13, atol=1e-13)


def test_too_few_atoms_run_unchunked():
    generator = torch.Generator().manual_seed(3)
    module = cm.chunk(NodeLocal(), 4)
    x, sc, table = _inputs(4 * cm.MIN_ATOMS_PER_CHUNK - 1, generator)
    calls = []
    inner = module.atom_chunked_forward
    module.atom_chunked_forward = lambda *args, **kwargs: calls.append(1) or inner(*args, **kwargs)
    module(x, sc, table)
    assert calls == [1]  # one call: the whole input


def test_a_deep_copy_rebinds_the_chunked_forward():
    module = cm.chunk(NodeLocal(), 2)
    twin = copy.deepcopy(module)
    assert twin.forward.__self__ is twin and twin.atom_chunked_forward.__self__ is twin
    assert twin.atom_chunk_count == 2 and cm.unchunked(twin).__self__ is twin


def test_the_record_is_rounding_deterministic_off_and_rescued_after_recompute():
    from fastmace.planner import memory

    record = chunks.RECORDS["atom_chunks"]
    assert (record.exactness, record.default, record.deterministic) == ("rounding", "off", True)
    assert record.requires == ("gated_rows",)
    assert memory.MEMORY_RESCUE == ("recompute_conv_weights", "atom_chunks")
    assert memory.ATOM_CHUNK_COUNTS == chunks.CHUNK_COUNTS
    assert "fastmace.rewrites.chunks" in reg.RECORD_SOURCES
    with pytest.raises(ValueError):
        RewriteConfig(atom_chunk_count=0)


def test_the_saving_grows_with_the_chunk_count():
    from fastmace.model import capabilities
    from fastmace.rewrites import memory as terms

    model, _ = tiny("mh1")
    dims = capabilities.model_dims(model)
    savings = [terms.atom_chunks_saving(dims, 4, count) for count in chunks.CHUNK_COUNTS]
    assert all(0 < a < b for a, b in zip(savings, savings[1:]))
    assert terms.atom_chunks(dims, 4) == (0, -terms.atom_chunks_saving(dims, 4, chunks.DEFAULT_CHUNKS))


# --------------------------------------------------------------------------- the tiny MH-1 model

def _registry():
    out = reg.Registry()
    for family in (mh1, linear, fusion, chunks):
        for record in family.RECORDS.values():
            out.register(record)
    return out


def _rewritten(atom_chunks, count=None, repeat=None):
    from _support.structures import nacl
    from _support.tiny_models import FAMILIES, tiny_mace_calculator

    model, batch = tiny("mh1", backend="cueq")
    if repeat is not None:
        from test_rewrites_mh1 import Z_VALUES

        calc = tiny_mace_calculator(Z_VALUES, **FAMILIES["mh1"])
        batch = calc._atoms_to_batch(nacl(repeat=repeat, rattle=0.1, seed=8)).to_dict()
    regs = _registry()
    on = {*PACKAGE, *LINEAR, "gated_rows"}
    requests = {name: "on" if name in on else "off" for name in regs.names()} | {"atom_chunks": atom_chunks}
    out = copy.deepcopy(model)
    resolution = regs.resolve(out, requests, strict=False)
    context = reg.Context(config=RewriteConfig(atom_chunk_count=count), inputs=reg.structure_inputs(batch))
    applied = regs.apply(out, resolution, context, original=model)
    return model, out, batch, resolution, applied


@CUEQ
def test_the_record_chunks_every_segment_and_passes_its_selfcheck():
    original, model, _, resolution, applied = _rewritten("on", count=3)
    assert resolution.decision("atom_chunks").enabled
    entry = next(entry for entry in applied if entry.name == "atom_chunks")
    assert entry.result.passed and entry.result.max_relative_error <= 1e-12
    names = [name for name, _ in chunks.segments(model)]
    assert names == [*(f"interactions.{i}.gated_rows" for i in range(len(model.interactions))),
                     *(f"products.{i}" for i in range(len(model.products)))]
    assert all(module.atom_chunk_count == 3 for _, module in chunks.segments(model))
    product = model.products[1]
    product.forward = lambda *args, **kwargs: cm.chunked_forward(product, *args, **kwargs).roll(1, 0)  # rows mixed
    assert not chunks.RECORDS["atom_chunks"].selfcheck(original, model, reg.Context()).passed


@CUEQ
def test_the_default_count_without_a_plan():
    _, model, _, _, _ = _rewritten("on")
    assert {module.atom_chunk_count for _, module in chunks.segments(model)} == {chunks.DEFAULT_CHUNKS}


@CUEQ
def test_energy_and_forces_keep_to_rounding_on_512_atoms():
    _, without, batch, _, _ = _rewritten("off", repeat=4)
    _, chunked, _, _, _ = _rewritten("on", count=2, repeat=4)
    assert batch["positions"].shape[0] == 512 >= 2 * cm.MIN_ATOMS_PER_CHUNK
    energy_and_forces(without, batch)  # throwaway evaluation (docs/development.md)
    energy, forces = energy_and_forces(without, batch)
    new_energy, new_forces = energy_and_forces(chunked, batch)
    assert relative(new_energy, energy) <= 1e-12 and relative(new_forces, forces) <= 1e-11
