"""The atom-chunked forward of ``atom_chunks`` (plans/capacity.md CAP-T08; ``fastmace.rewrites.chunks``)."""

from __future__ import annotations

import types

import torch

# Fewer atoms than this many per chunk run unchunked: a chunk of a few atoms gains nothing and costs launches.
MIN_ATOMS_PER_CHUNK = 64


def chunked_forward(self, *args, **kwargs):
    """``self.atom_chunked_forward`` (the module's forward before the record) on ``self.atom_chunk_count`` atom
    chunks, each under non-reentrant ``torch.utils.checkpoint``: the backward keeps only the chunks' inputs and
    rebuilds one chunk's intermediates at a time. Every tensor argument whose first dimension is the atom count
    (that of the first tensor argument) is split with ``tensor_split``; the others pass whole. The module must be
    node-local (each output row reads only its own atom's rows) and return one per-atom tensor."""
    from torch.utils.checkpoint import checkpoint

    inner, chunks = self.atom_chunked_forward, self.atom_chunk_count
    first = next(value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor))
    atoms = first.shape[0]
    if atoms < chunks * MIN_ATOMS_PER_CHUNK:
        return inner(*args, **kwargs)

    def split(value):
        if isinstance(value, torch.Tensor) and value.dim() and value.shape[0] == atoms:
            return torch.tensor_split(value, chunks)
        return (value,) * chunks

    parts_args = [split(value) for value in args]
    parts_kwargs = {name: split(value) for name, value in kwargs.items()}
    outputs = [checkpoint(inner, *(part[i] for part in parts_args), use_reentrant=False, preserve_rng_state=False,
                          **{name: part[i] for name, part in parts_kwargs.items()})
               for i in range(chunks)]
    return torch.cat(outputs)


def chunk(module, chunks):
    """Give ``module`` the chunked forward with ``chunks`` chunks; its forward before is kept as
    ``atom_chunked_forward`` (a bound method, so a deep copy rebinds both to the copy)."""
    module.atom_chunked_forward = module.forward
    module.atom_chunk_count = int(chunks)
    module.forward = types.MethodType(chunked_forward, module)
    return module


def unchunked(module):
    """``module``'s forward before ``chunk`` (the self-check's reference)."""
    return getattr(module, "atom_chunked_forward", module.forward)


__all__ = ["MIN_ATOMS_PER_CHUNK", "chunk", "chunked_forward", "unchunked"]
