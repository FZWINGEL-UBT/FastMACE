"""Versioned reference records (SPEC §12.1; plan task 1b-T13).

"Correct" is defined by a record per MACE version and model family. It has the four kinds of
entries of §12.1, and a fifth list for the numerical treatment of the yardstick:

- ``stock``: the reference is MACE's ``MACECalculator`` of that version, cuEquivariance, float64
  (the float32 calculator is the yardstick of the ratio rule). It states 1a-T19's references
  exactly: MACE's own cuEq conversion with the convolution unfused (MACE 0.3.16's rule on
  ``cuda:0``) and MACE PR 1498's one-graph strain, the stress formula of the production path;
- ``upstream_fixes``: defects upstream fixed after the pinned version, each with its independent
  validation. Today one: the PolarMACE stress (MACE issue 1642, PR 1718), validated by finite
  differences of the reference energy on NaCl 3x3x3. It is required for every PolarMACE record
  on MACE 0.3.16, and retired by the record of a MACE that contains it;
- ``optional_physics``: opt-in model changes, applied to the reference too and labelling the
  results non-stock. PR 1762 (Fukui regularisation, dipole cap) is accepted; D3 is declared and
  refused until phase 3b;
- ``integrator_corrections``: today the full-cell BAOAB drag, with 1b-T09's qualification
  (``fastmace.reference.integrators.QUALIFICATION``);
- ``numerical``: float64 accumulation of the yardstick's energy (§12.1 "Energy accumulation in
  the references"), a treatment of the reference, not of the model.

The record id is ``fastmace.reference/<mace version>/<family>/<digest>``, the digest a SHA-256 of
the canonical JSON of every field, so any change of any entry changes it. It enters the
force-field identity (§10.6, ``ForceFieldIdentity.reference_record_id``) and provenance (§13).
Fallbacks keep the record: a polar model is refused with a record that lacks the stress fix.
``content_digest()`` is the same digest without the MACE version: a resume that only changes the
version compares it (2-T09, decision 23).

**Unqualified versions** (SPEC §18 decision 2; plan task 2-T04). A MACE other than 0.3.16 gets an
"unqualified stock" record: stock equivalence only, with the stock entries the installed sources
allow. PR 1498's one-graph strain is stock only when its pinned source hashes, and the yardstick's
float64 accumulation only when the float64 energy forward hashes (without it the yardstick, and so
the gate, is unavailable). PolarMACE is refused: its stress fix needs the patched forward. The
qualified records never depend on the installed sources.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field

from fastmace.errors import FastMACEError
from fastmace.rewrites import versions

SCHEMA = 1
ID_PREFIX = "fastmace.reference"

# MACE versions with a qualified record. Another version gets the unqualified stock record.
QUALIFIED_MACE_VERSIONS = versions.QUALIFIED_MACE_VERSIONS

# Model families of the label table (``fastmace.model.labels.FAMILIES``), plus "unlabelled" for a
# checkpoint outside it (a PolarMACE outside it is "polar").
FAMILIES = ("mh1", "omol", "polar", "residual", "density", "off", "unlabelled")

STOCK = {
    "calculator": "mace.calculators.MACECalculator",
    "backend": "cueq",
    "conversion": "MACE's own e3nn -> cuEquivariance conversion",
    "cueq_conv_fusion": False,
    "single_graph_strain": True,
    "dtypes": {"reference": "float64", "yardstick": "float32"},
    "source": "1a-T19's gate references (run 1a-T19-gate-leto-20261018)",
}

POLAR_STRESS_FIX = {
    "name": "polar_kspace_stress",
    "issue": "MACE issue 1642",
    "fixed_upstream": "MACE PR 1718, after 0.3.16",
    "defect": "PolarMACE's stress misses the k-space energy's dependence on the cell",
    "implementation": "fastmace.longrange.polar: enable() before the first evaluation, the class-level forward "
                      "active (no instance forward stored)",
    "validation": {
        "method": "central finite differences of the reference energy",
        "structure": "NaCl 3x3x3",
        "strain": 1e-4,
        "tolerance_eV_per_A3": 1e-6,
        "result": "corrected reference <= 8.1e-9 eV/A^3, MACE 0.3.16's own stress 2e-5 eV/A^3 off (1a-T19)",
        "probe": "ResearchModule studies/other_models/probes/polar_stress_probe.py",
    },
    "open_findings": [
        "NbCl5 3x3x3: the corrected float64 POLAR-1-S stress differs from finite differences by 4.4e-4 eV/A^3 "
        "(about 3%); validated on NaCl 3x3x3 only, to be resolved in phase 3b before PolarMACE NPT is recommended",
    ],
    "retire_when": "the pinned MACE contains PR 1718; the record then states plain stock equivalence",
}

# Every optional physics change the policy knows, and whether a reference can take it.
OPTIONAL_PHYSICS = {
    "pr1762": {"status": "accepted", "upstream": "MACE PR 1762 (open)",
               "changes": "regularised Fukui normalisation and charge-dipole cap (PolarConfig)"},
    "d3": {"status": "refused until phase 3b",
           "reference": "MACE plus torch-dftd with the parameters of SPEC §11.2"},
}

FLOAT64_ACCUMULATION = {
    "name": "float64_energy_accumulation",
    "applies_to": "yardstick",
    "what": "the float32 yardstick's energy terms reduced into float64, with the checkpoint's float64 E0 table "
            "(fastmace.model.energy; PolarMACE: fastmace.longrange.polar, per-k and self terms)",
    "why": "stock float32 totals are off by up to 102 meV/atom with all-electron E0; with float64 accumulation on "
           "both sides the ratio rule measures the forward pass, not the E0 sum",
    "forces_and_stress": "unchanged: MACE's autograd of its own float32 scalar",
    "polar_note": "the yardstick runs the corrected long-range path, so the electrostatic energy is reduced per k "
                  "into float64; MACE's own long-range code would give only the float32 total, upcast",
}


class ReferencePolicyError(FastMACEError):
    """No qualified reference for this request, or a record that would drop a correction (SPEC §12.1)."""

    default_next_step = ("Use a model family and MACE version with a qualified reference record "
                         "(fastmace.reference.records.QUALIFIED_MACE_VERSIONS), and keep every correction.")


def _canonical(value):
    """``value`` as plain JSON data (tuples become lists), so equal records compare and hash equal."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


@dataclass(frozen=True, eq=True)
class ReferenceRecord:
    """One versioned reference record (SPEC §12.1). Entries are JSON mappings; ``id`` is stable."""

    schema: int
    mace_version: str
    family: str
    stock: dict = field(default_factory=dict)
    upstream_fixes: tuple = ()
    optional_physics: tuple = ()
    integrator_corrections: tuple = ()
    numerical: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "stock", _canonical(dict(self.stock)))
        for name in ("upstream_fixes", "optional_physics", "integrator_corrections", "numerical"):
            entries = tuple(_canonical(dict(entry)) for entry in getattr(self, name))
            object.__setattr__(self, name, entries)
        if (self.family == "polar" and self.mace_version in QUALIFIED_MACE_VERSIONS
                and not self.has_fix(POLAR_STRESS_FIX["name"])):
            raise ReferencePolicyError(
                f"a PolarMACE record for MACE {self.mace_version} must carry the stress fix (MACE issue 1642)",
                why="MACE's own PolarMACE stress misses the k-space cell dependence; a reference without the fix "
                    "would fail a correct production path and pass a defective one.")

    __hash__ = None  # entries are mappings

    def has_fix(self, name):
        return any(entry.get("name") == name for entry in self.upstream_fixes)

    @property
    def is_stock(self):
        """No optional physics: the model is the released one (upstream fixes correct defects only)."""
        return not self.optional_physics

    def _payload(self):
        return {f.name: (list(value) if isinstance(value := getattr(self, f.name), tuple) else value)
                for f in dataclasses.fields(self)}

    @property
    def digest(self):
        text = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()

    def content_digest(self):
        """SHA-256 of every entry except the MACE version: equal when only the version differs."""
        payload = self._payload()
        payload.pop("mace_version")
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()

    @property
    def qualified(self):
        """Whether the record's MACE version is a qualified one (else stock equivalence only)."""
        return self.mace_version in QUALIFIED_MACE_VERSIONS

    @property
    def id(self):
        return f"{ID_PREFIX}/{self.mace_version}/{self.family}/{self.digest[:16]}"

    def to_dict(self):
        """The record for provenance (§13): every field and the id."""
        return {**self._payload(), "id": self.id}

    @classmethod
    def from_dict(cls, data):
        values = {f.name: data[f.name] for f in dataclasses.fields(cls)}
        record = cls(**values)
        if "id" in data and data["id"] != record.id:
            raise ReferencePolicyError(f"reference record id {data['id']} does not match its entries ({record.id})")
        return record


def mace_version():
    """The installed MACE's version (``mace-torch``)."""
    from importlib import metadata

    return metadata.version("mace-torch")


def family_of(model, label_family=None):
    """The record family of ``model``: its label's family, "polar" for PolarMACE, else "unlabelled"."""
    from fastmace.longrange import polar

    is_polar = polar.is_polar(model)
    if label_family is None:
        return "polar" if is_polar else "unlabelled"
    if is_polar != (label_family == "polar"):
        raise ReferencePolicyError(f"the model is {'' if is_polar else 'not '}PolarMACE, but its label's family is "
                                   f"{label_family!r}")
    return label_family


def _pr1762_options(polar_config):
    from fastmace.longrange import polar

    if polar_config is None or not polar.stabilised(polar_config):
        return None
    return {"fukui_regularisation": float(polar_config.fukui_regularisation),
            "dipole_cap": None if polar_config.dipole_cap is None else list(polar_config.dipole_cap)}


def record_for(mace_version, family, *, polar=None, d3=None):
    """The reference record for (MACE version, model family) and the requested physics options.

    ``polar``: a ``PolarConfig`` (its PR 1762 options become an optional-physics entry);
    ``d3``: a ``D3`` record, refused until phase 3b. A version outside ``QUALIFIED_MACE_VERSIONS``
    gets the unqualified stock record (module docstring). Raises ``ReferencePolicyError`` for
    PolarMACE on such a version, an unknown family, or options the family cannot take.
    """
    from fastmace.reference.integrators import QUALIFICATION

    qualified = mace_version in QUALIFIED_MACE_VERSIONS
    if not qualified and family == "polar":
        raise ReferencePolicyError(
            f"no reference record for PolarMACE on MACE {mace_version}: only 0.3.16 is qualified",
            why="PolarMACE's stress fix (MACE issue 1642, SPEC §12.1) needs the hashed forward of the qualified "
                "version; an unqualified version gets stock equivalence only (SPEC §18 decision 2).",
            next_step="Install MACE 0.3.16 for PolarMACE models.")
    if family not in FAMILIES:
        raise ReferencePolicyError(f"unknown model family {family!r} for a reference record",
                                   why=f"the families are {', '.join(FAMILIES)}.")
    if d3 is not None:
        raise ReferencePolicyError(
            "a D3 reference is refused until phase 3b",
            why="the D3 reference (MACE plus torch-dftd, SPEC §11.2) and its kernel arrive in phase 3b.",
            next_step="Run without dispersion, or wait for phase 3b.")
    fixes = (POLAR_STRESS_FIX,) if family == "polar" else ()
    optional = ()
    options = _pr1762_options(polar)
    if options is not None:
        if family != "polar":
            raise ReferencePolicyError(f"the PR 1762 options apply to PolarMACE only, not to family {family!r}")
        optional = ({"name": "pr1762", **OPTIONAL_PHYSICS["pr1762"], "options": options},)
    correction = {"name": "baoab_full_cell_drag", "id": QUALIFICATION["id"], "qualification": QUALIFICATION}
    stock, numerical = STOCK, (FLOAT64_ACCUMULATION,)
    if not qualified:
        from fastmace import _patches

        stock = {**STOCK, "single_graph_strain": not _patches.patch_problems("single_graph_strain")}
        if versions.treatment("float64_energy").qualified is False:
            numerical = ()
    return ReferenceRecord(schema=SCHEMA, mace_version=mace_version, family=family, stock=stock,
                           upstream_fixes=fixes, optional_physics=optional, integrator_corrections=(correction,),
                           numerical=numerical)


def require_record_for_model(record, model):
    """Refuse ``record`` for ``model`` when it would drop a correction the model needs (§12.1, §9.4)."""
    from fastmace.longrange import polar

    if polar.is_polar(model) and record.mace_version in QUALIFIED_MACE_VERSIONS \
            and not record.has_fix(POLAR_STRESS_FIX["name"]):
        raise ReferencePolicyError(
            f"record {record.id} has no PolarMACE stress fix (MACE issue 1642) for a PolarMACE model",
            why="fallbacks keep the record: a reference may never drop a correction.",
            next_step="Use the polar record (record_for(version, 'polar')).")
    return record


__all__ = [
    "FAMILIES", "FLOAT64_ACCUMULATION", "ID_PREFIX", "OPTIONAL_PHYSICS", "POLAR_STRESS_FIX", "QUALIFIED_MACE_VERSIONS",
    "ReferencePolicyError", "ReferenceRecord", "SCHEMA", "STOCK", "family_of", "mace_version", "record_for",
    "require_record_for_model",
]
