# ruff: noqa: I001  (fastmace before torch)
"""Which autograd nodes keep large tensors for the backward in MACE-OMAT-0 medium with the generic rewrites (eager,
CUDA, float32): the energy's graph walked from its grad_fn, every saved tensor of at least ``atoms`` rows listed with
its node, shape and bytes per atom.
    python saved_probe.py [REPEAT]      (SrTiO3, default 6x6x6)
"""

import sys
from collections import defaultdict

import fastmace  # noqa: F401  (before torch: the OEq runtime and file locks)
import torch
from ase import Atoms

from fastmace.model import loading
from fastmace.rewrites import generic, radial
from fastmace.rewrites import registry as reg

repeat = tuple(int(n) for n in (sys.argv[1] if len(sys.argv) > 1 else "6x6x6").split("x"))
atoms = Atoms("SrTiO3", scaled_positions=[(0, 0, 0), (.5, .5, .5), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)],
              cell=[3.905] * 3, pbc=True).repeat(repeat)
atoms.rattle(0.02, seed=1)
loaded = loading.load("omat0-medium", backend="hybrid", dtype="float32", device="cuda:0")
model = loaded.model
for parameter in model.parameters():
    parameter.requires_grad_(False)
batch = loaded.calculator._atoms_to_batch(atoms).to("cuda:0")
order = torch.sort(batch.edge_index[1], stable=True).indices
batch.edge_index, batch.shifts, batch.unit_shifts = batch.edge_index[:, order], batch.shifts[order], \
    batch.unit_shifts[order]
data = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in batch.to_dict().items()}
registry = reg.Registry([*generic.RECORDS.values(), radial.RECORDS["generic_radial_tables"]])
asked = {name: "on" for name in ("generic_reassociate", "generic_fused_radial_backward", "generic_streaming",
                                 "generic_chunked_conv_backward", "generic_recompute_conv_weights",
                                 "generic_radial_tables")}
resolution = registry.resolve(model, asked, strict=False)
applied = registry.apply(model, resolution, reg.Context(inputs=reg.structure_inputs(data)), check=False)
print("applied", [entry.name for entry in applied])
edges = int(data["edge_index"].shape[1])
for interaction in model.interactions:
    if getattr(interaction, "streaming_kernels", None) is not None:
        interaction.streaming_count = torch.tensor(edges, device="cuda:0")
data["positions"].requires_grad_(True)
out = model(dict(data), training=False, compute_force=False, compute_stress=False)
nodes, seen, stack = defaultdict(list), set(), [out["energy"].sum().grad_fn]
n = len(atoms)
while stack:
    node = stack.pop()
    if node is None or node in seen:
        continue
    seen.add(node)
    for name in dir(node):
        if not name.startswith("_saved_"):
            continue
        try:
            value = getattr(node, name)
        except Exception:  # noqa: BLE001
            continue
        for tensor in (value if isinstance(value, (list, tuple)) else [value]):
            if torch.is_tensor(tensor) and tensor.dim() and tensor.shape[0] >= n:
                nodes[(type(node).__name__, name, tuple(tensor.shape), str(tensor.dtype))].append(tensor.data_ptr())
    stack.extend(next_fn for next_fn, _ in node.next_functions)
print(f"{n} atoms, {edges} edges")
for (node, name, shape, dtype), pointers in sorted(nodes.items(), key=lambda item: -item[0][2][0] * (
        item[0][2][1] if len(item[0][2]) > 1 else 1)):
    size = 1
    for d in shape:
        size *= d
    print(f"{node:45s} {name:22s} {str(shape):22s} {dtype:14s} x{len(set(pointers))}  "
          f"{size * 4 / n / 1000:8.1f} kB/atom")
