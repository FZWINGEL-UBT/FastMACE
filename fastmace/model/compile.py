"""Private model copies, compilation and the compiled-model cache (SPEC §3.2, §8.1; plan task 1a-T11).

Copied from the research engine's ``gpu_langevin.py`` (``_restore_oeq_wrappers``,
``_restore_cueq_conv_fusion``, ``_simplify_traceable``, ``compile_model``, ``_COMPILED_MODELS``,
``configure_allocator``) and refactored:

- **Copies keep MACE's convolution adapters** (``private_copy``). OpenEquivariance modules
  serialise only their C++ description, and cuEquivariance's fused convolution
  (``cuet.SegmentedPolynomial`` with MACE PR 1641's adapter) pickles through ``__reduce__`` with
  its constructor arguments only. A deep copy therefore loses MACE's four-argument adapters, so
  every copy re-applies both from the source model.
- **The compiled-model cache is an object** (``CompiledModelCache``), owned by whoever creates
  force fields and passed in; there is no module dict. Two force fields given the same cache and
  key share one compiled model.
- **Options are arguments.** ``compile_model`` takes an ``EngineConfig`` for the process-wide
  compile settings (installed through ``fastmace._patches``) and a ``prepare`` callback that
  applies the rewrites to the private copy before it is simplified and compiled.
- **cuEquivariance on the CPU** (conductor decision after 1a-T06): when
  ``cuequivariance_ops_torch`` imports, MACE's cuEq conversion builds some products with the
  CUDA-only ``uniform_1d`` kernel even for ``device="cpu"``. ``naive_cueq()`` converts with cuEq's
  ``naive`` method instead; ``cpu_cueq_problems`` finds a CPU model that was not.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import os
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

# Expandable segments grow in place instead of adding segments (3.53 -> 3.92 GiB peak on MH-1), at
# the same speed. Tested with CUDA graphs on torch 2.14 only.
ALLOCATOR_SETTINGS = "expandable_segments:True"
ALLOCATOR_MIN_TORCH = (2, 14)


# --------------------------------------------------------------------------- adapters on copies


def restore_oeq_wrappers(source, private):
    """Restore MACE's Python wrappers lost by OEq module serialisation; returns the number restored.

    OpenEquivariance modules intentionally serialise only their C++ input description. A deepcopy
    therefore recreates the bare operator and drops MACE's four-argument adapter (plus its optional
    layout transforms). Rebind each adapter to the independent module in ``private`` before that
    copy is moved to its execution device. The source model is never mutated.
    """
    private_modules = dict(private.named_modules())
    oeq_sources = [(name, module) for name, module in source.named_modules()
                   if type(module).__module__.startswith("openequivariance")]
    if not oeq_sources:
        return 0
    wrapper_ops = importlib.import_module("mace.modules.wrapper_ops")
    restored = 0
    for name, source_module in oeq_sources:
        private_module = private_modules.get(name)
        if private_module is None:
            raise RuntimeError(f"OEq module {name or '<root>'} is missing from the private copy")
        if not hasattr(source_module, "original_forward"):
            raise RuntimeError(f"OEq module {name or '<root>'} is not wrapped by MACE")
        # A fresh OEq copy should be bare, but remove any stale instance-level wrapper state
        # before applying MACE's adapter a second time.
        for attribute in ("forward", "original_forward", "layout_transpose_in", "layout_transpose_out"):
            private_module.__dict__.pop(attribute, None)
            private_module._modules.pop(attribute, None)
        has_layout = hasattr(source_module, "layout_transpose_in") or hasattr(source_module, "layout_transpose_out")
        if has_layout:
            wrap = getattr(wrapper_ops, "with_oeq_conv_fusion", None)
            if not callable(wrap):
                raise RuntimeError("installed MACE lacks with_oeq_conv_fusion for OEq")
            transpose_in = getattr(source_module, "layout_transpose_in", None)
            transpose_out = getattr(source_module, "layout_transpose_out", None)
            # Separate module trees: sharing either transform would let later device or state
            # changes of the copy mutate the source model.
            transpose_in = copy.deepcopy(transpose_in) if transpose_in is not None else None
            transpose_out = copy.deepcopy(transpose_out) if transpose_out is not None else None
            wrap(private_module, transpose_in=transpose_in, transpose_out=transpose_out)
        else:
            wrap = getattr(wrapper_ops, "with_oeq_scatter_sum", None)
            if not callable(wrap):
                raise RuntimeError("installed MACE lacks with_oeq_scatter_sum for standalone OEq")
            wrap(private_module)
        restored += 1
    return restored


def restore_cueq_conv_fusion(source, private):
    """Re-wrap the fused cuEquivariance convolutions that a deepcopy left bare; returns the number re-wrapped.

    A plain cuEquivariance model converted with conv fusion (MACE PR 1641) holds each convolution
    as a ``cuet.SegmentedPolynomial`` with MACE's four-argument adapter as an instance-level
    forward (``wrapper_ops.with_cueq_conv_fusion``). SegmentedPolynomial pickles through
    ``__reduce__`` (its constructor arguments only), so its deepcopy is the bare polynomial: no
    adapter, no ``weight_numel``, and the interaction's conv-fusion branch would call it with the
    wrong arguments. Apply MACE's adapter again to each copy in ``private``; the source model is
    never mutated.
    """
    sources = [(name, module) for name, module in source.named_modules()
               if type(module).__module__.startswith("cuequivariance_torch") and "original_forward" in module.__dict__]
    if not sources:
        return 0
    wrap = getattr(importlib.import_module("mace.modules.wrapper_ops"), "with_cueq_conv_fusion", None)
    if not callable(wrap):
        raise RuntimeError("installed MACE lacks with_cueq_conv_fusion for the fused cuEquivariance convolution")
    private_modules = dict(private.named_modules())
    restored = 0
    for name, source_module in sources:
        private_module = private_modules.get(name)
        if private_module is None:
            raise RuntimeError(f"fused cuEq convolution {name} is missing from the private copy")
        if "forward" in private_module.__dict__:
            continue  # the copy kept its adapter
        wrap(private_module)
        if private_module.weight_numel != source_module.weight_numel:
            raise RuntimeError(f"re-wrapped fused cuEq convolution {name}: weight_numel "
                               f"{private_module.weight_numel} != {source_module.weight_numel}")
        restored += 1
    return restored


def _device_of(model):
    for tensor in model.parameters():
        return tensor.device
    for tensor in model.buffers():
        return tensor.device
    return None


def private_copy(model):
    """A deep copy of ``model`` on its device, with MACE's OEq and fused-cuEq adapters re-applied.

    Moving the copy back to the model's device is needed: a deepcopy can leave buffers on the CPU
    (with cuEquivariance 0.12 the scalar coefficients of its naive fallback come back there,
    which eager mode tolerates but torch.compile rejects); otherwise it is a no-op. An
    instance-level ``forward`` that closes over the original model is dropped, so the copy runs
    its own class forward.
    """
    private = copy.deepcopy(model)
    restore_oeq_wrappers(model, private)
    restore_cueq_conv_fusion(model, private)
    device = _device_of(model)
    if device is not None:
        private = private.to(device)
    private.__dict__.pop("forward", None)
    return private


# --------------------------------------------------------------------------- cuEquivariance on the CPU


@contextlib.contextmanager
def naive_cueq():
    """While MACE converts a model to cuEquivariance, build every product with cuEq's naive (CPU) method.

    Without it, an importable ``cuequivariance_ops_torch`` makes the conversion pick the CUDA-only
    ``uniform_1d`` kernel even for ``device="cpu"``, and the first forward pass fails without a GPU.
    Process-local and undone on exit; the package is not modified. The conversion also runs between
    fresh cuEquivariance basis caches (``loading._fresh_cueq_basis_cache``, 2-T14).
    """
    from cuequivariance_torch.primitives import segmented_polynomial

    from fastmace.model import loading

    previous = segmented_polynomial.HAS_CUE_OPS
    segmented_polynomial.HAS_CUE_OPS = False
    try:
        with loading._fresh_cueq_basis_cache():
            yield
    finally:
        segmented_polynomial.HAS_CUE_OPS = previous


def cpu_cueq_problems(model):
    """Why a CPU cuEquivariance model cannot run without a GPU ([] = it can, or it is not on the CPU).

    Every ``SegmentedPolynomial`` of a CPU model must use cuEq's ``naive`` method; any other
    method (``uniform_1d``) is a CUDA-only kernel.
    """
    device = _device_of(model)
    if device is None or device.type != "cpu":
        return []
    problems = []
    for name, module in model.named_modules():
        if not type(module).__module__.startswith("cuequivariance_torch"):
            continue
        if type(module).__name__ != "SegmentedPolynomial":
            continue
        method = getattr(module, "method", None)
        if method not in (None, "naive"):
            problems.append(f"{name} uses cuEquivariance's CUDA-only {method!r} kernel instead of 'naive'")
    return problems


# --------------------------------------------------------------------------- compilation


def install_engine_patches(engine=None, *, oeq=False):
    """The process-wide patches compiled models need, before tracing; returns their PatchRecords.

    ``capture_safe_mask_head`` always (it must be in place before tracing),
    ``first_derivative_only`` when asked for and not on OEq models (which run with
    training=False), ``unfuse_full_addmm`` when asked for (read by Inductor when the graph is
    compiled).
    """
    from fastmace import _patches
    from fastmace.config import EngineConfig

    engine = engine or EngineConfig()
    records = [_patches.set_capture_safe_mask_head(True)]
    if engine.first_derivative_only and not oeq:
        records.append(_patches.set_first_derivative_only(True))
    if engine.unfuse_full_addmm:
        records.append(_patches.set_unfuse_full_addmm(True))
    return tuple(record for record in records if record is not None)


def simplify_traceable(model, simplify):
    """mace.tools.compile.simplify, leaving blocks that torch.fx cannot trace to dynamo.

    simplify replaces every registered block by its torch.fx symbolic trace and raises if one
    fails. Some of PolarMACE's blocks use cuEquivariance's naive backend
    (SegmentedPolynomialNaive), whose Python shape checks symbolic tracing cannot follow. Here
    each block is traced exactly as simplify does; only a block whose trace raises stays as it is
    (dynamo then traces it, which needs ``longrange.polar.allow_irrep_guards``). Models whose
    blocks all trace (MH-1, MP-0 and the density family; MH-1's hybrid model has naive modules
    too and they trace) are simplified exactly as by simplify. The untraced names are recorded
    on ``model._untraced_blocks``.
    """
    from torch.fx import symbolic_trace

    try:
        from mace.tools.compile import _SIMPLIFY_REGISTRY
    except ImportError:  # a MACE without the registry (or a test stand-in): its own simplify
        return simplify(model)
    registered = tuple(_SIMPLIFY_REGISTRY)
    untraced = []

    def walk(module, prefix=""):
        for name, child in module.named_children():
            if isinstance(child, registered):
                try:
                    setattr(module, name, symbolic_trace(child))
                except Exception:  # noqa: BLE001 - torch.fx raises several types (TraceError, ...)
                    untraced.append(prefix + name)
            else:
                walk(child, prefix + name + ".")

    walk(model)
    if untraced:
        from fastmace.longrange import polar

        polar.allow_irrep_guards()
    model._untraced_blocks = untraced
    return model


def compile_model(model, mode, *, accelerated, oeq=False, prepare: Callable[[Any], Any] | None = None,
                  engine=None):
    """torch.compile a private copy of a MACE model exactly as MACECalculator prepares one.

    ``accelerated`` (cuEquivariance / OpenEquivariance models): simplify a private copy
    (``private_copy``), as the calculator does. ``oeq=True`` follows MACE's OEq path, where autograd
    stays outside the compiled graph. ``prepare(private)`` applies the rewrites to the copy before
    it is simplified (symbolic tracing inlines the replaced modules). Plain e3nn models are rebuilt
    from their configuration with e3nn's TorchScript code generation switched off
    (``prepare(extract_model)``): compiled code cannot differentiate through those TorchScript
    parts, and skipping the rebuild silently gives zero forces. ``engine`` (an ``EngineConfig``)
    gives the process-wide settings: the patches of ``install_engine_patches`` and dynamo's
    cache-size limit, raised because each edge-size change and each model counts against dynamo's
    per-function recompile limit and dynamo silently falls back to eager execution beyond it.
    """
    import torch
    from mace.tools.compile import configure_autograd_for_compile, disable_e3nn_codegen, simplify
    from mace.tools.compile import prepare as prepare_for_compile
    from mace.tools.scripts_utils import extract_model

    from fastmace.config import EngineConfig

    engine = engine or EngineConfig()
    if oeq and not accelerated:
        raise ValueError("OpenEquivariance compilation requires an accelerated model")
    install_engine_patches(engine, oeq=oeq)
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit,
                                                engine.dynamo_cache_size_limit)
    if accelerated:
        private = private_copy(model)
        # This mirrors MACECalculator: OEq operators are opaque to AOTAutograd, while
        # cuEquivariance's autograd.grad is traced into the graph.
        configure_autograd_for_compile(allow_autograd=not oeq)
        if prepare is not None:
            prepare(private)
        with disable_e3nn_codegen():
            private = simplify_traceable(private, simplify)
    else:
        private = prepare_for_compile(extract_model)(model=model, map_location=_device_of(model))
        if prepare is not None:
            prepare(private)
    for parameter in private.parameters():
        parameter.requires_grad_(False)
    return torch.compile(private, mode=mode, fullgraph=False)


# --------------------------------------------------------------------------- the cache


@dataclass(frozen=True)
class CompiledEntry:
    """One compiled model: its source (kept alive, so its id cannot be reused), the model, the rewrites applied."""

    source: Any
    model: Any
    applied: tuple = ()


class CompiledModelCache:
    """Compiled models by key, owned by whoever creates force fields (SPEC §3.2: no module-level cache).

    The force field's key is the source model's identity, the compile mode and route, the
    rewrites applied and, for structure-dependent rewrites, the structure's species. An entry is
    returned only for the very source model it was compiled from.
    """

    def __init__(self):
        self._entries = {}

    def get(self, key: Hashable, source) -> CompiledEntry | None:
        entry = self._entries.get(key)
        return entry if entry is not None and entry.source is source else None

    def put(self, key: Hashable, source, model, applied=()) -> CompiledEntry:
        entry = self._entries[key] = CompiledEntry(source, model, tuple(applied))
        return entry

    def keys(self):
        return tuple(self._entries)

    def clear(self):
        self._entries.clear()

    def __len__(self):
        return len(self._entries)


# --------------------------------------------------------------------------- allocator


def configure_allocator(enabled=True, environ=None, torch_version=None):
    """Set ``PYTORCH_CUDA_ALLOC_CONF`` to ``ALLOCATOR_SETTINGS`` unless already set; returns the value in effect.

    Must run before the first CUDA allocation of the process (the allocator reads the variable
    when it initialises); the engine's process constraints (plan task 1a-T13) check that. Older
    torch versions are left unchanged, and so is the variable when ``enabled`` is False.
    """
    environ = os.environ if environ is None else environ
    if torch_version is None:
        import torch

        torch_version = torch.__version__
    version = tuple(int(part) for part in torch_version.split("+")[0].split(".")[:2])
    if enabled and environ.get("PYTORCH_CUDA_ALLOC_CONF") is None and version >= ALLOCATOR_MIN_TORCH:
        environ["PYTORCH_CUDA_ALLOC_CONF"] = ALLOCATOR_SETTINGS
    return environ.get("PYTORCH_CUDA_ALLOC_CONF")


__all__ = [
    "ALLOCATOR_MIN_TORCH", "ALLOCATOR_SETTINGS", "CompiledEntry", "CompiledModelCache", "compile_model",
    "configure_allocator", "cpu_cueq_problems", "install_engine_patches", "naive_cueq", "private_copy",
    "restore_cueq_conv_fusion", "restore_oeq_wrappers", "simplify_traceable",
]
