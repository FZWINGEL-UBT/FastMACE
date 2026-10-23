"""Versioned reference records (SPEC §12.1; plan task 1b-T13).

"Correct" is defined by a record per MACE version and model family. It has the four kinds of
entries of §12.1, and a fifth list for the numerical treatment of the yardstick:

- ``stock``: the reference is MACE's ``MACECalculator`` of that version, cuEquivariance, float64
  (the float32 calculator is the yardstick of the ratio rule). It states 1a-T19's references
  exactly: MACE's own cuEq conversion with the convolution unfused (MACE 0.3.16's rule on
  ``cuda:0``) and MACE PR 1498's one-graph strain, the stress formula of the production path;
- ``upstream_fixes``: defects upstream fixed after the pinned version, each with its independent
  validation. Today one: the PolarMACE stress (MACE issue 1642, PR 1718), validated by finite
  differences of the reference energy with the k-set frozen, on NaCl and NbCl5 3x3x3. It is
  required for every PolarMACE record on MACE 0.3.16, and retired by the record of a MACE that
  contains it;
- ``optional_physics``: opt-in model changes, applied to the reference too and labelling the
  results non-stock. PR 1762 (Fukui regularisation, dipole cap) and D3 (MACE plus torch-dftd on the
  ``fm.D3`` record's numbers, 3b-T04) are accepted;
- ``integrator_corrections``: today the full-cell BAOAB drag, with 1b-T09's qualification
  (``fastmace.reference.integrators.QUALIFICATION``);
- ``numerical``: float64 accumulation of the yardstick's energy (§12.1 "Energy accumulation in
  the references"), a treatment of the reference, not of the model; for PolarMACE also the gate's
  stress floor of 5e-7 eV/A^3 (§12.3, decision 49), a treatment of the comparison.

A qualified record's entries are fixed: a change is a new record, and the old id goes into
``SUPERSEDED`` with the reason (4-T12: the PolarMACE record of 1b, by the stress floor and the
restated issue 1642 validation).

The record id is ``fastmace.reference/<mace version>/<family>/<digest>``, the digest a SHA-256 of
the canonical JSON of every field, so any change of any entry changes it. It enters the
force-field identity (§10.6, ``ForceFieldIdentity.reference_record_id``) and provenance (§13).
Fallbacks keep the record: a polar model is refused with a record that lacks the stress fix.
``content_digest()`` is the same digest without the MACE version: a resume that only changes the
version compares it (2-T09, decision 23).

**Unqualified versions** (SPEC §18 decision 2; plan task 2-T04). A MACE other than 0.3.16 gets an
"unqualified stock" record: stock equivalence only, with the stock entries the installed sources
allow. PR 1498's one-graph strain is stock only when its pinned source hashes, and the yardstick's
float64 accumulation only when the float64 energy forward hashes (otherwise the
``float32_sum_unqualified`` entry: MACE's own float32 sum, a looser energy bound that the gate
record and the report name). PolarMACE is refused: its stress fix needs the patched forward. The
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

# Restated in 4-T11/4-T12's record version (the 1b text is in tests/cpu/test_versioned_registry.py): the
# self-test holds the unstrained cell's k-set since 4-T11, so it validates the stress on any structure, and
# 3b-T08 resolved the NbCl5 finding (the finite-difference reference stepped where a k-vector crossed the hard
# k-space cutoff; the stress itself is exact).
POLAR_STRESS_FIX = {
    "name": "polar_kspace_stress",
    "issue": "MACE issue 1642",
    "fixed_upstream": "MACE PR 1718, after 0.3.16",
    "defect": "PolarMACE's stress misses the k-space energy's dependence on the cell",
    "implementation": "fastmace.longrange.polar: enable() before the first evaluation, the class-level forward "
                      "active (no instance forward stored)",
    "validation": {
        "method": "central finite differences of the reference energy, with the unstrained cell's included k-set "
                  "held fixed (fastmace.longrange.polar.frozen_kset, 4-T11)",
        "structures": ["NaCl 3x3x3", "NbCl5 3x3x3"],
        "strain": 1e-4,
        "tolerance_eV_per_A3": 1e-6,
        "result": "corrected float64 reference against finite differences: NaCl 3x3x3 6.9e-9 eV/A^3 (1a-T19: <= "
                  "8.1e-9), NbCl5 3x3x3 (1944 atoms, the gate's hydrostatic-0 configuration) 8.3e-9 (4-T11); MACE "
                  "0.3.16's own stress 2e-5 eV/A^3 off on NaCl (1a-T19)",
        "probe": "ReferenceCalculator.check_stress_fix (fastmace.reference.calculators), on the run's own structure; "
                 "first ResearchModule studies/other_models/probes/polar_stress_probe.py",
    },
    "resolved_findings": [
        "NbCl5 3x3x3: 4.4e-4 eV/A^3 against live finite differences was a step of the energy where a Bragg "
        "k-vector crosses the hard k-space cutoff, not a stress defect; 8.3e-9 with the k-set frozen (3b-T08, 4-T11)",
    ],
    "retire_when": "the pinned MACE contains PR 1718; the record then states plain stock equivalence",
}

# Every optional physics change the policy knows, and whether a reference can take it.
OPTIONAL_PHYSICS = {
    "pr1762": {"status": "accepted", "upstream": "MACE PR 1762 (open)",
               "changes": "regularised Fukui normalisation and charge-dipole cap (PolarConfig)"},
    "d3": {"status": "accepted",
           "reference": "MACE plus torch-dftd 0.5.1 with the parameters of SPEC §11.2 (the fm.D3 record), summed "
                        "as ASE's SumCalculator does; torch-dftd in float64 for the reference and the yardstick, so "
                        "the yardstick measures MACE's float32 alone",
           "changes": "adds the D3(BJ) dispersion energy, forces and stress"},
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

# The yardstick of an unqualified MACE whose float64 energy forward does not hash (2-T04): MACE's
# own float32 sum. The energy bound of the ratio rule is then the looser one; production runs at
# float64 there (the precision promotion), so its own energy error stays small.
FLOAT32_SUM_UNQUALIFIED = {
    "name": "float32_sum_unqualified",
    "applies_to": "yardstick",
    "what": "the float32 yardstick's energy is MACE's own float32 sum: the float64 energy forward does not hash on "
            "this MACE",
    "consequence": "the energy bound of the ratio rule is looser (up to the float32 E0 sum's error); forces and "
                   "stress are unaffected",
}
ENERGY_BOUND_UNQUALIFIED = "looser: unqualified MACE, float32 yardstick energy sum (SPEC §18 decision 2)"

# A treatment of the gate's comparison, not of the model (SPEC §12.3, decision 49): PolarMACE's float32
# stress sits at the general 1e-7 floor on rounding alone, so its record raises the stress floor. The
# energy and force floors stay the general ones. ``gate_floors`` reads the entries that apply to the gate.
POLAR_GATE_FLOOR = {
    "name": "polar_float32_stress_floor",
    "applies_to": "gate",
    "floors": {"stress_eV_per_A3": 5e-7},
    "what": "the gate's floor for the largest stress component is 5e-7 eV/A^3 (0.08 MPa) for PolarMACE; the energy "
            "and force floors are SPEC §12.3's",
    "why": "PolarMACE's float32 stress is 0.8-2.7e-7 eV/A^3 from float64 on every path (3b-T08), the order of the "
           "general 1e-7 floor, so the ratio rule failed some configurations on rounding alone (1b-T23: 1.2e-7 on "
           "NaCl 3x3x3)",
    "decision": "SPEC §18 decision 49 (owner, 23 October 2026; 4-T12)",
}

# Qualified record ids that this FastMACE no longer builds, with the reason. Restart files and metadata
# carry the ids, so a resume names the reason when it refuses one (``fastmace.io.restart.check_identity``).
SUPERSEDED = {
    "fastmace.reference/0.3.16/polar/8e216bde0c3515e9":
        "the PolarMACE record of phases 1b-3b, superseded in 4-T12 by "
        "fastmace.reference/0.3.16/polar/1b78dc6383d5150e, which adds the gate's PolarMACE stress floor "
        "(polar_float32_stress_floor, SPEC §18 decision 49) and restates the issue 1642 validation (NaCl and "
        "NbCl5 with the k-set frozen, the NbCl5 finding resolved by 3b-T08 and 4-T11)",
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
    ``d3``: a ``D3`` record, an optional-physics entry with every number (``d3.identity_entry``); refused
    for PolarMACE (SPEC §18 decision 41). A version outside ``QUALIFIED_MACE_VERSIONS``
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
    if d3 is not None and family == "polar":
        raise ReferencePolicyError(
            "a D3 reference for PolarMACE is refused in phase 3b",
            why="PolarMACE with D3 is not qualified (SPEC §18 decision 41).",
            next_step="Run PolarMACE without dispersion.")
    fixes = (POLAR_STRESS_FIX,) if family == "polar" else ()
    optional = ()
    options = _pr1762_options(polar)
    if options is not None:
        if family != "polar":
            raise ReferencePolicyError(f"the PR 1762 options apply to PolarMACE only, not to family {family!r}")
        optional = ({"name": "pr1762", **OPTIONAL_PHYSICS["pr1762"], "options": options},)
    if d3 is not None:
        from fastmace.longrange import d3 as dispersion

        optional += ({"name": "d3", **OPTIONAL_PHYSICS["d3"], "parameters": dispersion.identity_entry(d3)},)
    correction = {"name": "baoab_full_cell_drag", "id": QUALIFICATION["id"], "qualification": QUALIFICATION}
    stock, numerical = STOCK, (FLOAT64_ACCUMULATION,) + ((POLAR_GATE_FLOOR,) if family == "polar" else ())
    if not qualified:
        from fastmace import _patches

        stock = {**STOCK, "single_graph_strain": not _patches.patch_problems("single_graph_strain")}
        if versions.treatment("float64_energy").qualified is False:
            numerical = (FLOAT32_SUM_UNQUALIFIED,)
    return ReferenceRecord(schema=SCHEMA, mace_version=mace_version, family=family, stock=stock,
                           upstream_fixes=fixes, optional_physics=optional, integrator_corrections=(correction,),
                           numerical=numerical)


def gate_floors(record):
    """The gate floors ``record`` raises, ``{quantity: floor}`` (empty for every family but PolarMACE)."""
    floors = {}
    for entry in record.numerical:
        if entry.get("applies_to") == "gate":
            floors.update(entry["floors"])
    return floors


def superseded(record_id):
    """Why a qualified record id is no longer built, or ``None`` for an id that is not superseded."""
    return SUPERSEDED.get(str(record_id))


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
    "ENERGY_BOUND_UNQUALIFIED", "FAMILIES", "FLOAT32_SUM_UNQUALIFIED",
    "FLOAT64_ACCUMULATION", "ID_PREFIX", "OPTIONAL_PHYSICS", "POLAR_GATE_FLOOR", "POLAR_STRESS_FIX",
    "QUALIFIED_MACE_VERSIONS", "ReferencePolicyError", "ReferenceRecord", "SCHEMA", "STOCK", "SUPERSEDED", "family_of",
    "gate_floors", "mace_version", "record_for", "require_record_for_model", "superseded",
]
