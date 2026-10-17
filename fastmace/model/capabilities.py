"""Model traits: what a loaded MACE model is, and which rewrites it can take (SPEC §4.3, §9.2).

``model_traits(model)`` reads a loaded model once and returns a frozen ``ModelTraits`` record:
interaction classes, ``apply_cutoff``, joint embeddings, PolarMACE, heads, edge-padding
exactness, species-only first inputs, the per-element E0 classes and the rewrite-support
verdicts with their reasons. The planner (§9) decides from this record; nothing here changes the
model.

``rewrite_support``, ``padding_is_exact`` and ``first_inputs_species_only`` keep the research
engine's verdicts and reason strings, so the two-stage ``auto`` resolution (1a-T15,
docs/design/switch-map.md) can reproduce ``adapt_options_to_model`` exactly.

This module never imports MACE; it only looks at attributes and class names, so it works on
stand-ins as well. ``torch`` and ``ase`` are imported inside the functions that need them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NamedTuple

# Interaction classes whose radial networks read species embeddings (MH-1's class): the
# MH-1 rewrites (factor_species and everything built on it) apply to them.
FACTORED_INTERACTIONS = ("RealAgnosticResidualNonLinearInteractionBlock",)

# The other benchmark architectures (MACE-MP-0 and the density family), which take the GENERIC_*
# rewrites (adopted 15 October 2026): class name -> variant. The source hashes of their
# forward methods belong to the rewrites (Rewrite.source_hashes), not to the traits.
GENERIC_INTERACTIONS: Mapping[str, str] = MappingProxyType({
    "RealAgnosticResidualInteractionBlock": "residual",
    "RealAgnosticDensityInteractionBlock": "density",
    "RealAgnosticDensityResidualInteractionBlock": "density_residual",
})

# E0 classes (SPEC §4.3 "Energy scale").
ALL_ELECTRON = "all_electron"  # total energy of the isolated atom with all electrons
PSEUDOPOTENTIAL = "pseudopotential"  # valence energy with an effective core potential (def2 from Rb on)
SMALL = "small"  # a few eV: an atom reference relative to a pseudised atom (PAW style)
UNKNOWN = "unknown"  # fits none of the above
E0_KINDS = (ALL_ELECTRON, PSEUDOPOTENTIAL, SMALL, UNKNOWN)

HARTREE_EV = 27.211386245988  # CODATA 2018
# |E0| at or above this, without being all-electron, is a valence (effective-core) energy. The
# lightest def2 effective-core element, Rb, has a valence energy near -24 Ha (about -650 eV);
# the largest "few eV" E0 seen (MH-1 matpes_r2scan Ta) is -40.5 eV.
PSEUDOPOTENTIAL_MIN_EV = 200.0
# E0 / all-electron estimate inside this window counts as all-electron. The estimate is within
# about 15% for Z >= 3 and about half the exact value for H and He (-13.6 eV against -7.3 eV).
ALL_ELECTRON_RATIO = (0.5, 3.0)


class RewriteSupport(NamedTuple):
    """Whether a model can take one rewrite, and why (``"ok"`` when it can)."""

    supported: bool
    reason: str


@dataclass(frozen=True)
class E0Class:
    """One element's E0 in one head, with its class and the reason for it."""

    head: str
    z: int
    symbol: str
    value_ev: float
    kind: str
    reason: str


@dataclass(frozen=True)
class ModelTraits:
    """What the planner needs to know about a loaded model (SPEC §9.2 "Model traits").

    ``rewrite_support`` is a read-only mapping (so the record is frozen but not hashable).
    """

    model_class: str
    interaction_classes: tuple[str, ...]
    apply_cutoff: bool | None  # MACE's radial_embedding.apply_cutoff; None without a radial embedding
    radial_cutoff_passed: bool  # the interactions receive the cutoff separately (apply_cutoff=False, MH-1)
    factored: bool  # an interaction of FACTORED_INTERACTIONS
    generic: bool  # an interaction of GENERIC_INTERACTIONS
    joint_embedding: bool
    joint_embeddings: tuple[tuple[str, str], ...]  # (name, "graph" | "node"), sorted by name
    polar: bool  # PolarMACE (MACE-POLAR-1)
    heads: tuple[str, ...]
    atomic_numbers: tuple[int, ...]
    r_max: float | None
    padding_is_exact: bool
    first_inputs_species_only: bool
    e0: tuple[E0Class, ...]  # per head (in ``heads`` order), then per element (``atomic_numbers`` order)
    e0_dtype: str | None  # dtype of the stored E0 table (POLAR-1 float32, OMOL-0 and MH-1 float64)
    rewrite_support: Mapping[str, RewriteSupport]

    def _head(self, head):
        if head is None:
            if len(self.heads) != 1:
                raise ValueError(f"the model has heads {list(self.heads)}; name the head")
            return self.heads[0]
        if head not in self.heads:
            raise KeyError(f"head {head!r} not in {list(self.heads)}")
        return head

    def e0_for(self, head: str | None = None) -> tuple[E0Class, ...]:
        """The E0 classes of one head (the only head when ``head`` is None)."""
        head = self._head(head)
        return tuple(entry for entry in self.e0 if entry.head == head)

    def e0_kinds(self, head: str | None = None) -> dict[int, str]:
        """{atomic number: E0 class} of one head."""
        return {entry.z: entry.kind for entry in self.e0_for(head)}

    def large_e0(self, head: str | None = None) -> bool:
        """Does any E0 of this head have more than a few eV (all-electron, effective core or unknown)?

        This is what matters for float32 energies: the float32 problem depends on the magnitude
        of the total, not on the labelling (§4.3), so mixed tables count as well.
        """
        return any(entry.kind != SMALL for entry in self.e0_for(head))


# --------------------------------------------------------------------------- rewrite verdicts


def radial_cutoff_passed(model: Any) -> bool:
    """True if the interactions receive the edge cutoff separately (MACE apply_cutoff=False, as MH-1)."""
    radial = getattr(model, "radial_embedding", None)
    return radial is not None and getattr(radial, "apply_cutoff", True) is False


def first_inputs_species_only(model: Any) -> bool:
    """Are the first interaction's node features a function of the species alone (per structure)?

    True for a linear node embedding of the one-hot species, optionally plus a
    joint embedding whose features are all per graph (MACE-OMOL-0: total charge
    and spin). The engine evaluates one structure (one graph), so a per-graph
    term is the same for every atom and atoms of one species share their
    features exactly; per-node joint features (e.g. atomic charges) do not.
    """
    if type(getattr(model, "node_embedding", None)).__name__ != "LinearNodeEmbeddingBlock":
        return False
    if not hasattr(model, "joint_embedding"):
        return True
    specs = getattr(model, "embedding_specs", None) or {}
    return bool(specs) and all(spec.get("per") == "graph" for spec in specs.values())


def _interaction_classes(model: Any) -> list[str]:
    return [type(interaction).__name__ for interaction in getattr(model, "interactions", ())]


def rewrite_support(model: Any) -> dict[str, RewriteSupport]:
    """Which MH-1-style model rewrites a MACE model can take: {option: (supported, reason)}.

    factor_species: an interaction of FACTORED_INTERACTIONS (species
    embeddings concatenated into the radial input; MH-1's class).
    fold_cutoff (and everything built on it: density first, joint radial
    input, fused folded backward, streaming convolution): additionally the
    cutoff passed to the interactions (apply_cutoff=False). The interaction
    source is still hash-checked when the rewrite is applied.
    reassociate_first: additionally species-only first-interaction inputs.
    The other benchmark architectures (GENERIC_INTERACTIONS: MACE-MP-0 and the
    density family) take streaming_conv and reassociate_first through the
    GENERIC_* rewrites instead (adopted 15 October 2026).

    A model counts as generic when *any* of its interactions is in GENERIC_INTERACTIONS
    (MACE-OFF: a RealAgnosticInteractionBlock first, residual blocks after it).
    """
    interactions = _interaction_classes(model)
    factorable = any(name in FACTORED_INTERACTIONS for name in interactions)
    support = {"factor_species": RewriteSupport(factorable, "ok" if factorable else
                                                f"interactions {sorted(set(interactions))} are not "
                                                f"{FACTORED_INTERACTIONS}")}
    cutoff = radial_cutoff_passed(model)
    support["fold_cutoff"] = RewriteSupport(
        factorable and cutoff,
        "ok" if factorable and cutoff else
        support["factor_species"].reason if not factorable else
        "the model applies its cutoff inside the radial embedding (apply_cutoff=True)")
    species_only = first_inputs_species_only(model)
    fold = support["fold_cutoff"].supported
    support["reassociate_first"] = RewriteSupport(
        fold and species_only,
        "ok" if fold and species_only else
        support["fold_cutoff"].reason if not fold else
        "first-interaction inputs are not species-only")
    generic = any(name in GENERIC_INTERACTIONS for name in interactions)
    support["streaming_conv"] = RewriteSupport(fold or generic,
                                               "ok" if fold or generic else support["fold_cutoff"].reason)
    if generic:
        support["reassociate_first"] = RewriteSupport(
            species_only, "ok (GENERIC_REASSOCIATE)" if species_only else
            "first-interaction inputs are not species-only")
    # Second round for the other architectures (adopted 15 October 2026): the species-restricted
    # skip connection (GENERIC_SPECIES_SKIP) and the fused radial input gradient
    # (GENERIC_FUSED_RADIAL_BACKWARD, reached through fused_folded_backward).
    support["species_skip"] = RewriteSupport(generic, "ok" if generic else
                                             "no MACE-MP-0 or density-family interaction")
    support["fused_folded_backward"] = RewriteSupport(fold or generic,
                                                      "ok" if fold or generic else support["fold_cutoff"].reason)
    return support


def padding_is_exact(model: Any) -> bool:
    """Do edges beyond the cutoff contribute exactly zero (so the CUDA-graph edge padding is exact)?

    Always with the cutoff passed to the interactions (apply_cutoff=False, MH-1).
    With apply_cutoff=True the radial features are zero beyond the cutoff, but a
    radial network with biases, or one that also reads species embeddings,
    still gives far edges nonzero weights (seen with a NonLinear-interaction test
    model, 15 October 2026); bias-free networks on the radial features alone
    (MACE-MP-style e3nn FullyConnectedNet) stay exact.
    """
    if radial_cutoff_passed(model) or getattr(model, "radial_embedding", None) is None:
        return True  # (no radial embedding: not a MACE model, e.g. test stand-ins; nothing to check)
    import torch

    for interaction in getattr(model, "interactions", ()):
        if hasattr(interaction, "source_embedding") or hasattr(interaction, "target_embedding"):
            return False
        for name in ("conv_tp_weights", "density_fn"):
            network = getattr(interaction, name, None)
            if network is None:
                continue
            for module in network.modules():
                bias = getattr(module, "bias", None)
                if isinstance(module, torch.nn.Linear) and bias is not None:
                    return False
    return True


def is_polar(model: Any) -> bool:
    """PolarMACE (MACE-POLAR-1), by class name as the research engine checks it."""
    return type(model).__name__ == "PolarMACE"


# --------------------------------------------------------------------------- E0 classes


def all_electron_estimate_ev(z: int) -> float:
    """Total energy of the neutral atom with all electrons, in eV (Thomas-Fermi plus Scott term).

    -0.7687 Z^(7/3) + Z^2/2 hartree; within about 15% of the exact value for Z >= 3 (Na
    -3,983 against -4,415 eV, Cl -11,612 against -12,522 eV, I -1.82e5 eV) and half of it for
    H. Good enough to tell an all-electron E0 from a valence-only one, which is 1-2 orders of
    magnitude smaller from Rb on.
    """
    return (-0.7687 * z ** (7.0 / 3.0) + 0.5 * z * z) * HARTREE_EV


def _symbol(z: int) -> str:
    from ase.data import chemical_symbols

    return chemical_symbols[z] if 0 < z < len(chemical_symbols) else f"Z{z}"


def classify_e0(z: int, value_ev: float, head: str = "Default") -> E0Class:
    """Classify one element's E0 from its value (SPEC §4.3 "Energy scale").

    - all-electron: within a factor ``ALL_ELECTRON_RATIO`` of the all-electron estimate of the
      atom (Cl -12,522, Na -4,412, O -2,048, Br -70,042 eV);
    - pseudopotential: not all-electron, but at least ``PSEUDOPOTENTIAL_MIN_EV`` below zero, a
      valence energy with an effective core potential (def2 from Rb on: Nb and Ta near
      -1,545 eV, I -8,102.52 eV);
    - small: |E0| below ``PSEUDOPOTENTIAL_MIN_EV`` (a few eV, as MH-1 matpes_r2scan Cl -4.8 eV);
    - unknown: anything else (far below the all-electron energy, or large and positive).
    """
    z, value = int(z), float(value_ev)
    symbol = _symbol(z)
    estimate = all_electron_estimate_ev(z)
    ratio = value / estimate if estimate < 0 else math.nan
    low, high = ALL_ELECTRON_RATIO
    if low <= ratio <= high:
        kind = ALL_ELECTRON
        reason = (f"{symbol} E0 {value:.6g} eV is {ratio:.2f} x the all-electron estimate {estimate:.6g} eV "
                  "of the atom: an all-electron reference")
    elif abs(value) < PSEUDOPOTENTIAL_MIN_EV:
        kind = SMALL
        reason = (f"{symbol} E0 {value:.6g} eV is below {PSEUDOPOTENTIAL_MIN_EV:g} eV in magnitude: "
                  "a small (relative) atom reference")
    elif value < 0 and ratio < low:
        kind = PSEUDOPOTENTIAL
        reason = (f"{symbol} E0 {value:.6g} eV is only {ratio:.3f} x the all-electron estimate {estimate:.6g} eV "
                  "but more than a few eV: a valence energy with an effective core potential (pseudopotential)")
    else:
        kind = UNKNOWN
        reason = (f"{symbol} E0 {value:.6g} eV fits no reference (all-electron estimate {estimate:.6g} eV, "
                  f"ratio {ratio:.3g})")
    return E0Class(head=head, z=z, symbol=symbol, value_ev=value, kind=kind, reason=reason)


def _e0_table(model: Any, heads: tuple[str, ...], atomic_numbers: tuple[int, ...]):
    """(E0 classes, dtype name) from MACE's atomic_energies_fn, or ((), None) without one."""
    energies = getattr(getattr(model, "atomic_energies_fn", None), "atomic_energies", None)
    if energies is None or not atomic_numbers:
        return (), None
    dtype = str(energies.dtype).removeprefix("torch.")
    rows = energies.detach().cpu().double().reshape(-1, len(atomic_numbers)).tolist()
    if len(rows) == 1 and len(heads) > 1:  # one E0 row shared by every head
        rows = rows * len(heads)
    if len(rows) != len(heads):
        raise ValueError(f"E0 table has {len(rows)} rows for heads {list(heads)}")
    return tuple(classify_e0(z, value, head) for head, row in zip(heads, rows)
                 for z, value in zip(atomic_numbers, row)), dtype


# --------------------------------------------------------------------------- the record


def _ints(values) -> tuple[int, ...]:
    if values is None:
        return ()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return tuple(int(v) for v in values)


def model_traits(model: Any) -> ModelTraits:
    """Read the traits of a loaded MACE model (a calculator gives those of its first model)."""
    if not hasattr(model, "interactions") and hasattr(model, "models"):
        model = model.models[0]
    interactions = tuple(_interaction_classes(model))
    radial = getattr(model, "radial_embedding", None)
    apply_cutoff = None if radial is None else bool(getattr(radial, "apply_cutoff", True))
    joint = hasattr(model, "joint_embedding")
    specs = (getattr(model, "embedding_specs", None) or {}) if joint else {}
    heads = tuple(str(h) for h in (getattr(model, "heads", None) or ("Default",)))
    atomic_numbers = _ints(getattr(model, "atomic_numbers", None))
    r_max = getattr(model, "r_max", None)
    e0, e0_dtype = _e0_table(model, heads, atomic_numbers)
    return ModelTraits(
        model_class=type(model).__name__,
        interaction_classes=interactions,
        apply_cutoff=apply_cutoff,
        radial_cutoff_passed=radial_cutoff_passed(model),
        factored=any(name in FACTORED_INTERACTIONS for name in interactions),
        generic=any(name in GENERIC_INTERACTIONS for name in interactions),
        joint_embedding=joint,
        joint_embeddings=tuple(sorted((str(name), str(spec.get("per"))) for name, spec in specs.items())),
        polar=is_polar(model),
        heads=heads,
        atomic_numbers=atomic_numbers,
        r_max=None if r_max is None else float(r_max),
        padding_is_exact=padding_is_exact(model),
        first_inputs_species_only=first_inputs_species_only(model),
        e0=e0,
        e0_dtype=e0_dtype,
        rewrite_support=MappingProxyType(rewrite_support(model)),
    )
