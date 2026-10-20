"""Converted-model templates: a plan's execution model without a reload (opt-1 OPT-T06).

Every plan converts its own copy of the pristine model (decision 24), and the conversion costs
3.2–3.7 s on the GPU (2-T19), once per plan, again for the gate's reduced cell, and once per
fallback candidate. A ``TemplateCache`` keeps the converted model of the first plan of each
conversion key (the template) and gives every later plan a private copy of it
(``compile.private_copy``, 0.5–0.7 s), so no module, tensor or calculator state is shared between
plans (SPEC §8.7). The template itself never reaches a plan. Design and measurements:
``docs/design/opt-T06-template-cache.md``.

- **Where:** CUDA with the cuEq or hybrid backend only (``applies``). On the CPU a deepcopy of a
  cuEq model rebuilds its naive kernels, as long as the reload; e3nn has no conversion to save.
- **Key:** checkpoint hash, head, backend, dtype, device, the conversion options, the process
  patches that act on the conversion (``CONVERSION_PATCHES``) and the package versions
  (``conversion_key``).
- **Owner:** each ``ForceField`` holds its own cache; nothing is module-level. At most
  ``MAX_TEMPLATES`` templates, least recently used out; ``release`` frees them.
- **Guard:** a template that changed since it was stored (a tensor replaced or changed in place)
  raises ``TemplateChangedError`` instead of being copied.

Self-checks are not cached: every plan runs all of them on its own copy (owner, checkpoint 2).
"""

from __future__ import annotations

import copy
import dataclasses
import sys
from collections import OrderedDict

from fastmace.errors import TemplateChangedError

MAX_TEMPLATES = 2
BACKENDS = ("cueq", "hybrid")
# The process patches that change what a conversion builds: MACE's cuEq converter (conv fusion) and the
# OEq build the converted modules use. The engine patches a build installs (mask_head, compute_forces,
# the addmm rule, strain, polar long-range) act when MACE's functions are called, not on the modules, so
# they stay out of the key; otherwise the first build would make every later plan miss.
CONVERSION_PATCHES = ("cueq_conv_fusion", "oeq_runtime")
_VERSIONED_MODULES = ("torch", "cuequivariance", "cuequivariance_torch", "cuequivariance_ops_torch",
                      "openequivariance", "e3nn")


def applies(backend, device) -> bool:
    """Whether converted models of ``backend`` on ``device`` are kept as templates: CUDA, cuEq or hybrid."""
    return backend in BACKENDS and str(device).split(":", 1)[0] == "cuda"


def _versions():
    from fastmace.rewrites.registry import installed_mace_version

    found = tuple((name, str(getattr(sys.modules.get(name), "__version__", None))) for name in _VERSIONED_MODULES)
    return (("mace-torch", installed_mace_version()),) + found


def conversion_key(loaded, options) -> tuple:
    """Everything the conversion of ``loaded`` (the pristine model) with ``options`` (loading.load's) depends on."""
    from fastmace import _patches

    return (loaded.sha256, str(loaded.path), loaded.label, loaded.head, loaded.model_type,
            tuple(sorted((name, repr(value)) for name, value in options.items())),
            tuple(repr(record) for record in _patches.applied() if record.name in CONVERSION_PATCHES), _versions())


def _tensors(model):
    return [(name, tensor) for name, tensor in model.named_parameters()] + \
           [(name, tensor) for name, tensor in model.named_buffers()]


def fingerprint(model) -> tuple:
    """Cheap identity of a model's tensors: object, storage and in-place version counter of each."""
    return tuple((name, id(tensor), tensor.data_ptr(), tensor._version, str(tensor.dtype), str(tensor.device),
                  tuple(tensor.shape)) for name, tensor in _tensors(model))


def nbytes(model) -> int:
    return sum(tensor.numel() * tensor.element_size() for _, tensor in _tensors(model))


def private(loaded):
    """A copy of ``loaded`` for one plan: its own model (``compile.private_copy``) in a shallow calculator shell."""
    from fastmace.model import compile as compiling

    shell = copy.copy(loaded.calculator)
    shell.models = [compiling.private_copy(loaded.model)]
    shell.results = {}
    if hasattr(shell, "atoms"):
        shell.atoms = None
    return dataclasses.replace(loaded, calculator=shell)


class TemplateCache:
    """The templates of one ``ForceField``: conversion key -> (template, fingerprint), at most ``limit``."""

    def __init__(self, limit: int = MAX_TEMPLATES):
        self.limit = int(limit)
        self._entries = OrderedDict()

    def __len__(self):
        return len(self._entries)

    def keys(self):
        return tuple(self._entries)

    def template(self, key):
        return self._entries[key][0]

    def execution_model(self, key, load):
        """The plan's converted model: on a miss ``load()`` (kept as a template copy), on a hit a private copy."""
        entry = self._entries.get(key)
        if entry is None:
            loaded = load()
            # Option B of the design: the template is copied at the first plan, so every later plan,
            # the gate's reduced cell and every fallback candidate are copies.
            template = private(loaded)
            self._entries[key] = (template, fingerprint(template.model))
            while len(self._entries) > self.limit:
                self._entries.popitem(last=False)
            return loaded
        template, stored = entry
        if fingerprint(template.model) != stored:
            raise TemplateChangedError(
                "the cached converted model changed after it was stored",
                why="a template is never handed to a plan, so something changed it in place or replaced a tensor")
        self._entries.move_to_end(key)
        return private(template)

    def release(self) -> int:
        """Drop every template; returns the bytes their models held."""
        freed = sum(nbytes(template.model) for template, _ in self._entries.values())
        self._entries.clear()
        return freed


__all__ = ["BACKENDS", "CONVERSION_PATCHES", "MAX_TEMPLATES", "TemplateCache", "TemplateChangedError", "applies",
           "conversion_key", "fingerprint", "nbytes", "private"]
