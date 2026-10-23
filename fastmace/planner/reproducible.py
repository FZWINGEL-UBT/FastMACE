"""``reproducible=True`` (SPEC §9.3 rule 5 and "reproducible=True", §8.7 "Reproducible processes").

Plan task 2-T16. A reproducible plan is bitwise within the scope of §9.3: same hardware model,
driver, software, model, plan, seeds and restart cadence. It is offered for one (backend, model
family): plain cuEq with the unfused convolution (owner decision 13) and the MH-1 architecture.

- **The plan.** Plain cuEq, ``cueq_conv_fusion`` off; hybrid-only rewrites are off on plain cuEq
  anyway (``factor_species`` and everything built on it, the streaming convolution, the block
  linears), and with them ``edge_outer_sum`` and its segmented forward. An explicit hybrid backend
  or fused convolution is refused (rule 5).
- **The process** (owner decision 14). ``CUBLAS_WORKSPACE_CONFIG``, ``TORCHINDUCTOR_DETERMINISTIC``
  and ``CUEQUIVARIANCE_OPS_U1D_ALGOS`` are set together with
  ``torch.use_deterministic_algorithms(True)`` and Inductor's deterministic mode, before CUDA
  starts (``prepare_process``). A process is reproducible or not; a plan of the other kind runs in
  a child through the subprocess runner, which passes the variables on.
- **cuEquivariance's run-time kernel choice.** ``cuequivariance_ops`` 0.12 JIT-compiles each
  ``uniform_1d`` problem (the unfused ChannelWiseTensorProduct and the SymmetricContraction) in
  several (schedules, warps per schedule) variants, times them on the GPU and keeps the fastest.
  Variants sum in different orders, so the choice made a plain-cuEq run differ from itself by up
  to 3.2e-13 Å over 500 steps (SPEC §18). ``CUEQUIVARIANCE_OPS_U1D_ALGOS`` replaces the candidate
  list (default (1,1) (1,2) (1,4) (2,2) (4,2) (6,4)); one pair leaves one candidate. The variable
  is an **undocumented internal** of ``libcue_ops.so``, read from its disassembly, so the family is
  qualified only on the ``cuequivariance_ops`` version it was checked on (``PINNED_CUEQ_OPS``), the
  build checks that two evaluations are bitwise equal (``check_repeatable``), and the plan records
  and reports the reliance.
- **Inductor** benchmarks Triton configurations on the device unless its deterministic mode is on;
  under the deterministic flag its scatter lowerings fall back to the deterministic ATen kernels.
- **FastMACE's kernels** declare their determinism in their rewrite records; a record in use whose
  ``deterministic`` is not True (``None`` = undeclared) makes rule 5 refuse.

A family that fails its build-time check raises ``ReproducibilityError`` naming the blocking
kernels; rule 5 refusals are ``PlanRefused``. This module imports neither torch nor MACE at
import time.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from fastmace.errors import ProcessConstraintError, Reason, ReproducibilityError

TASK = "plan task 2-T16"
# What an allow-list entry covers: a cuEquivariance kernel, a torch operator under
# torch.use_deterministic_algorithms(True), or one of FastMACE's own kernels.
KINDS = ("cueq-kernel", "torch-op", "fastmace-kernel")

FAMILY = "mh1"
# FastMACE's own tiers run the qualified kernels; the reference tier is MACE's calculator (2-T03).
QUALIFIED_TIERS = ("graph", "eager")
BACKEND = "cueq"
# The architecture qualified: MH-1's model class and interaction class, no joint embeddings.
FAMILY_MODEL_CLASS = "ScaleShiftMACE"
FAMILY_INTERACTION = "RealAgnosticResidualNonLinearInteractionBlock"

# The cuEquivariance ops package (installed under a CUDA-specific distribution name) and the one
# version whose undocumented candidate-list variable was checked (2-T16, murphy and the RTX 5060 Ti).
CUEQ_OPS = "cuequivariance_ops"
CUEQ_OPS_DISTRIBUTIONS = ("cuequivariance_ops_cu13", "cuequivariance_ops_cu12", "cuequivariance_ops")
PINNED_CUEQ_OPS = "0.12.0"
U1D_ALGOS = "1,1"  # one (schedules, warps per schedule) candidate for every uniform_1d problem
UNDOCUMENTED = (f"CUEQUIVARIANCE_OPS_U1D_ALGOS={U1D_ALGOS}, an undocumented internal of cuequivariance_ops "
                f"{PINNED_CUEQ_OPS} (libcue_ops.so), pins cuEq's run-time uniform_1d kernel choice")

# Set together, before CUDA starts (SPEC §8.7), and passed to the subprocess runner's children.
PROCESS_ENVIRONMENT: Mapping[str, str] = MappingProxyType({
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "TORCHINDUCTOR_DETERMINISTIC": "1",
    "CUEQUIVARIANCE_OPS_U1D_ALGOS": U1D_ALGOS,
})


@dataclass(frozen=True)
class AllowListEntry:
    """One component of a reproducible plan and whether it is deterministic.

    - ``component``: readable name ("ChannelWiseTensorProduct (unfused, uniform_1d)", "scatter_add_");
    - ``kind``: one of ``KINDS``;
    - ``deterministic``: True, False, or None = undeclared (refused like False);
    - ``reason``: the evidence or the source of the claim;
    - ``condition``: what it holds under ("cuequivariance_ops 0.x pinned"), empty if unconditional.
    """

    component: str
    kind: str
    deterministic: bool | None
    reason: str
    condition: str = ""

    def __post_init__(self):
        if not isinstance(self.component, str) or not self.component:
            raise ValueError(f"AllowListEntry.component must be a non-empty string, got {self.component!r}")
        if self.kind not in KINDS:
            raise ValueError(f"AllowListEntry.kind {self.kind!r} is not one of {KINDS}")
        if self.deterministic is not None and not isinstance(self.deterministic, bool):
            raise ValueError(f"AllowListEntry.deterministic must be True, False or None, got {self.deterministic!r}")


@dataclass(frozen=True)
class Qualification:
    """The reproducibility verdict for one (backend, model family): ``qualified``, the ``blocking``
    components when not, and the allow-list ``entries`` it was judged on."""

    backend: str
    family: str
    qualified: bool
    blocking: tuple[str, ...] = ()
    entries: tuple[AllowListEntry, ...] = ()
    reasons: tuple[Reason, ...] = ()  # rule 5 refusals, one per blocking component

    def __post_init__(self):
        object.__setattr__(self, "blocking", tuple(str(name) for name in self.blocking))
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if self.qualified and self.blocking:
            raise ValueError(f"a qualified family has no blocking components, got {self.blocking}")


_PINNED = f"cuequivariance_ops {PINNED_CUEQ_OPS} with CUEQUIVARIANCE_OPS_U1D_ALGOS={U1D_ALGOS}"
_FLAG = "torch.use_deterministic_algorithms(True)"

# The allow-list of plain cuEq MH-1 with the unfused convolution (besides the rewrite records in use,
# which ``qualify`` adds from their declarations).
ALLOW_LIST: tuple[AllowListEntry, ...] = (
    AllowListEntry("ChannelWiseTensorProduct (unfused, uniform_1d)", "cueq-kernel", True,
                   "per-edge (batched) outputs, no atomics; one candidate kernel left, so the summation order is "
                   "fixed; repeated and interrupted runs bitwise equal (2-T16)", _PINNED),
    AllowListEntry("SymmetricContraction (uniform_1d)", "cueq-kernel", True,
                   "per-node outputs, species-indexed weights as an input, no atomics; one candidate kernel left "
                   "(2-T16)", _PINNED),
    AllowListEntry("Linear and FullyConnectedTensorProduct (naive)", "cueq-kernel", True,
                   "MACE builds them with method='naive': torch einsum and matmul, cuBLAS"),
    AllowListEntry("scatter_add_ / index_add / index_select backward", "torch-op", True,
                   "the unfused convolution's messages are summed by MACE's scatter_add_; deterministic CUDA "
                   "implementations under the flag (torch 2.14 documentation)", _FLAG),
    AllowListEntry("cuBLAS GEMM", "torch-op", True, "deterministic with a fixed workspace configuration",
                   "CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA starts"),
    AllowListEntry("Inductor-generated kernels", "torch-op", True,
                   "no on-device benchmarking of Triton configurations; scatter lowerings fall back to ATen under "
                   "the flag", "TORCHINDUCTOR_DETERMINISTIC=1 and torch._inductor.config.deterministic"),
    AllowListEntry("GPU neighbour list", "fastmace-kernel", True,
                   "sorts the atomic pair-search output on a unique key (SPEC §9.3)"),
)
# Not qualified: the components a reproducible plan never uses, named in refusals.
FUSED_CONVOLUTION = AllowListEntry(
    "ChannelWiseTensorProduct (fused convolution)", "cueq-kernel", False,
    "scatters to the receivers inside the cuEq kernel, which the deterministic flag does not govern (owner "
    "decision 13)")
OEQ_STREAMING = AllowListEntry(
    "OpenEquivariance convolutions (hybrid)", "fastmace-kernel", False, "sums with atomics (SPEC §3.1)")


# --------------------------------------------------------------------------- facts

def family_of(traits) -> str:
    """``FAMILY`` for the MH-1 architecture, else a short description of the model."""
    interactions = tuple(traits.interaction_classes)
    if (traits.model_class == FAMILY_MODEL_CLASS and interactions and not traits.polar
            and all(name == FAMILY_INTERACTION for name in interactions) and not traits.joint_embeddings):
        return FAMILY
    extra = (" (PolarMACE)" if traits.polar else
             f" with joint embeddings {[name for name, _ in traits.joint_embeddings]}" if traits.joint_embeddings
             else "")
    return f"{traits.model_class} of {', '.join(interactions) or 'no interactions'}{extra}"


def installed_versions() -> dict[str, str | None]:
    """{``CUEQ_OPS``: the installed cuequivariance_ops version, or None}."""
    from importlib import metadata

    for distribution in CUEQ_OPS_DISTRIBUTIONS:
        try:
            return {CUEQ_OPS: metadata.version(distribution)}
        except metadata.PackageNotFoundError:
            continue
    return {CUEQ_OPS: None}


def record_determinism(registry=None) -> dict[str, bool | None]:
    """{record name: ``deterministic``} of the registry's records and the engine's process records."""
    from fastmace.rewrites import engine as engine_records
    from fastmace.rewrites import registry as reg

    registry = registry or reg.default_registry()
    out = {rewrite.name: rewrite.deterministic for rewrite in registry}
    for name, rewrite in engine_records.PROCESS_RECORDS.items():
        out.setdefault(name, rewrite.deterministic)
    return out


def records_in_use(plan, registry=None) -> tuple[str, ...]:
    """The rewrite records a plan may apply: every registry record whose switch is not off, and the
    engine's process records the graph engine installs (conservative: support is not known yet)."""
    from fastmace.rewrites import engine as engine_records
    from fastmace.rewrites import registry as reg

    registry = registry or reg.default_registry()
    requested = registry.requested(plan.rewrites)
    names = [name for name in registry.order() if requested[name] != "off"]
    engine = plan.engine
    switches = {"capture_safe_mask_head": True, "first_derivative_only": engine.first_derivative_only != "off",
                "unfuse_full_addmm": engine.unfuse_full_addmm != "off"}
    names += [name for name in engine_records.PROCESS_RECORDS if switches.get(name, False) and name not in names]
    return tuple(names)


# --------------------------------------------------------------------------- rule 5

def explicit_conflicts(engine) -> list[Reason]:
    """Rule 5 refusals of an explicit request a reproducible plan cannot honour."""
    out = []
    if engine.backend == "hybrid":
        out.append(Reason(f"reproducible=True needs the plain cuEq backend, not EngineConfig.backend='hybrid': "
                          f"{OEQ_STREAMING.component} {OEQ_STREAMING.reason} (SPEC §9.3 rule 5)", rule=5,
                          code="reproducible-backend"))
    if engine.cueq_conv_fusion == "on":
        out.append(Reason(f"reproducible=True needs the unfused convolution, not EngineConfig.cueq_conv_fusion='on': "
                          f"{FUSED_CONVOLUTION.component} {FUSED_CONVOLUTION.reason} (SPEC §9.3 rule 5)", rule=5,
                          code="reproducible-conv-fusion"))
    return out


def qualify(plan: Any, traits: Any = None, *, versions: Mapping[str, str | None] | None = None,
            registry=None) -> Qualification:
    """The ``Qualification`` of ``plan``'s backend and model family (``traits``) against ``ALLOW_LIST``.

    ``versions`` (default ``installed_versions()``) gives the cuequivariance_ops version, which must
    be ``PINNED_CUEQ_OPS`` on a CUDA device; ``registry`` the rewrite records (default registry).
    """
    versions = installed_versions() if versions is None else versions
    family = family_of(traits) if traits is not None else FAMILY
    engine = plan.engine
    cuda = str(engine.device).startswith("cuda")
    reasons, blocking = [], []
    entries = list(ALLOW_LIST)

    def block(component, text, code):
        blocking.append(component)
        reasons.append(Reason(f"reproducible=True: {text} (SPEC §9.3 rule 5)", rule=5, code=code))

    tier = getattr(plan, "tier", "graph")
    if tier not in QUALIFIED_TIERS:
        why = "runs MACE's own calculator, which" if tier == "reference" else "(Inductor's generated kernels)"
        block(f"the {tier} tier", f"the {tier} tier {why} is not qualified; "
              f"reproducible plans run on the {' or '.join(QUALIFIED_TIERS)} tier", "reproducible-tier")
    if family != FAMILY:
        block(f"model family {family}", f"only the MH-1 family is qualified with plain cuEq; this model is {family}",
              "reproducible-family-not-qualified")
    if engine.backend != BACKEND:
        block(OEQ_STREAMING.component, f"the {engine.backend} backend: {OEQ_STREAMING.reason}", "reproducible-backend")
    if engine.cueq_conv_fusion == "on":
        block(FUSED_CONVOLUTION.component, FUSED_CONVOLUTION.reason, "reproducible-conv-fusion")
    version = versions.get(CUEQ_OPS)
    if cuda and version != PINNED_CUEQ_OPS:
        block("cuEquivariance uniform_1d",
              f"cuEquivariance's uniform_1d kernels pick a variant at run time by timing; the pin "
              f"(CUEQUIVARIANCE_OPS_U1D_ALGOS, undocumented) is qualified for cuequivariance_ops {PINNED_CUEQ_OPS} "
              f"only, installed: {version or 'none'}", "reproducible-cueq-version")
    declared = record_determinism(registry)
    for name in records_in_use(plan, registry):
        value = declared.get(name)
        entries.append(AllowListEntry(f"rewrite {name}", "fastmace-kernel", value, "its rewrite record's declaration"))
        if value is not True:
            state = "undeclared (deterministic=None)" if value is None else "not deterministic"
            block(f"rewrite {name}", f"rewrite {name} is {state}; switch it off or declare it",
                  "reproducible-kernel-not-deterministic")
    return Qualification(backend=engine.backend, family=family, qualified=not blocking, blocking=tuple(blocking),
                         entries=tuple(entries), reasons=tuple(reasons))


def decisions(qualification: Qualification, versions: Mapping[str, str | None]) -> tuple[tuple[str, str, Any], ...]:
    """The plan's records of a qualified reproducible plan: the verdict and the process environment."""
    version = versions.get(CUEQ_OPS)
    pinned = (f"; relies on {UNDOCUMENTED}" if version == PINNED_CUEQ_OPS else
              f"; cuequivariance_ops {version or 'none'} (no uniform_1d kernel in use)")
    verdict = Reason(f"plain cuEq {qualification.family}, unfused convolution, deterministic flags; "
                     f"{len(qualification.entries)} components on the allow-list" + pinned,
                     rule=5, code="reproducible-qualified")
    environment = " ".join(f"{name}={value}" for name, value in PROCESS_ENVIRONMENT.items())
    return (("reproducible", f"qualified ({qualification.backend}, {qualification.family})", verdict),
            ("reproducible.environment", environment,
             Reason("set together with torch.use_deterministic_algorithms(True) before CUDA starts (SPEC §8.7, "
                    "owner decision 14); passed to the subprocess runner's children", rule=5,
                    code="reproducible-environment")))


# --------------------------------------------------------------------------- the process

def prepare_process(record, *, environ: MutableMapping[str, str] | None = None, torch_module: Any = None,
                    owner: str = "a reproducible plan") -> None:
    """Make this process reproducible (SPEC §8.7, owner decision 14): the variables of
    ``PROCESS_ENVIRONMENT``, ``torch.use_deterministic_algorithms(True)`` and Inductor's deterministic
    mode, all before CUDA starts. ``record`` (``constraints.ProcessRecord``) then says so.

    Raises ``ProcessConstraintError`` after CUDA was initialised, or in a process already claimed
    as not reproducible. Idempotent.
    """
    if record.reproducible is True:
        return
    if record.reproducible is False:
        raise ProcessConstraintError(
            f"this process is not reproducible ({record.reproducible_owner}); a process is reproducible or not "
            "(SPEC §8.7, owner decision 14): run the reproducible plan in a child through the subprocess runner")
    environ = os.environ if environ is None else environ
    if torch_module is None:
        import torch as torch_module
    if torch_module.cuda.is_initialized():
        raise ProcessConstraintError(
            "reproducible=True after CUDA was initialised in this process: CUBLAS_WORKSPACE_CONFIG and the "
            "deterministic flags must be set before (SPEC §8.7); plan before any CUDA use, or let the subprocess "
            "runner run the plan in a child")
    environ.update(PROCESS_ENVIRONMENT)
    torch_module.use_deterministic_algorithms(True)
    inductor_config = getattr(getattr(torch_module, "_inductor", None), "config", None)
    if inductor_config is None:  # the real torch imports its Inductor config on demand
        import torch._inductor.config as inductor_config
    inductor_config.deterministic = True
    record.claim_reproducible(True, owner)


def check_repeatable(evaluate: Callable[[], Any], *, kernels: tuple[str, ...] = ("cuEquivariance uniform_1d",),
                     ) -> None:
    """Evaluate twice and require bitwise-equal outputs (the build-time check of the pinned cuEq
    variable); ``ReproducibilityError`` naming ``kernels`` otherwise."""
    first = [_bytes(value) for value in evaluate()]
    second = [_bytes(value) for value in evaluate()]
    if first != second:
        raise ReproducibilityError(
            f"reproducible=True: two evaluations of the same positions differ bitwise ({', '.join(kernels)})",
            "a kernel picked another variant or summed atomically, so this family is not reproducible on this "
            "machine and software (SPEC §9.3)", blocking=kernels)


def _bytes(value):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value)).tobytes()


__all__ = [
    "ALLOW_LIST", "AllowListEntry", "BACKEND", "CUEQ_OPS", "FAMILY", "FUSED_CONVOLUTION", "KINDS", "OEQ_STREAMING",
    "PINNED_CUEQ_OPS", "PROCESS_ENVIRONMENT", "QUALIFIED_TIERS", "Qualification", "U1D_ALGOS", "UNDOCUMENTED",
    "check_repeatable", "decisions", "explicit_conflicts", "family_of", "installed_versions", "prepare_process",
    "qualify", "record_determinism", "records_in_use",
]
