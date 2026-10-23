"""``FastMACECalculator`` and ``fm.calc``: the ASE calculator, compiled tier (SPEC §7.4, §7.7 level 0).

ASE calls a calculator once per step, so this front end gets the compiled tier at most (§8.2): a
compiled model, exact edge counts, no CUDA graphs. Full speed needs FastMACE to own the MD loop
(``fm.md``, the ``fastmace.ase`` integrators). The calculator builds nothing itself: at the first
calculation, and again whenever the species or the atom count change, it makes a one-off
evaluation plan through the minimal planner (``MinimalPlanner.plan_evaluation``: compiled, with
stress, no workload) and gets that plan's ``MACEForceField`` from ``build_evaluation``. Each plan's
superset is its structure's species (owner decision D3), so new species get a fresh plan; the
plans share one compiled-model cache. Evaluation plans compile with a dynamic edge dimension
(``EngineConfig.dynamic_edges``, 2-T13), so relaxations, phonopy and NEB, whose edge counts change
between calls, pay one compile.
"""

from __future__ import annotations

from typing import Any

from ase.calculators.calculator import Calculator, all_changes

NOTE = "compatibility tier: compiled, no CUDA graphs; use `fm.md` or `fastmace.ase` integrators for full speed"

# fm.calc's keywords that go to fm.load; the rest go to FastMACECalculator.
_LOAD_OPTIONS = ("sha256", "trusted", "download", "charge", "spin_multiplicity")
_CALCULATOR_OPTIONS = ("rewrites", "precision", "dispersion")


class FastMACECalculator(Calculator):
    """``FastMACECalculator(model, *, head=None, rewrites="auto", precision="float32", dispersion=None)``:
    energy, free_energy, forces and Voigt stress in ASE units; full 3D PBC only (§7.6).

    - ``model``: an ``fm.load`` result or a label of the model table (a checkpoint path or URL
      needs ``fm.load`` or ``fm.calc`` with ``sha256=``/``trusted=``, since the trust decision
      needs one); ``head`` selects a head of a label's checkpoint, and must match a loaded
      model's head;
    - ``rewrites``: ``"auto"`` or a ``RewriteConfig``; ``precision``: the force precision,
      ``"float32"`` or ``"float64"``. Energies are accumulated in float64 (§7.3);
    - ``dispersion``: ``None``, ``"d3bj"`` or an ``fm.D3``, as ``fm.ForceField`` takes it (MACE plus D3,
      SPEC §11.2; 3b-T04). ``dispersion`` is then the complete record. With D3 there are no per-atom
      ``energies`` (D3's energy is not split by atom here).

    ``energies`` (per atom) is listed only for models whose energy terms are all atom-resolved;
    PolarMACE's k-space electrostatics are not (§7.3). Non-periodic or mixed-PBC atoms raise
    ``PlanRefused`` (rule 1); gas-phase molecules need MACE's ``MACECalculator`` (§4.3).

    ``plan`` is the current evaluation plan (``None`` before the first calculation) and ``notes``
    what the calculator recorded, the compatibility-tier note first; the note is printed once.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, model: Any, *, head: str | None = None, rewrites: Any = "auto", precision: str = "float32",
                 dispersion: Any = None):
        from fastmace.model.forcefield import PRECISIONS, ForceField

        super().__init__()
        if isinstance(model, str):
            if not _is_label(model):
                raise TypeError(f"{model!r} is not a label of the model table; load a checkpoint path or URL with "
                                "fm.load(path, sha256=... or trusted=True) or fm.calc(path, sha256=... or "
                                "trusted=True) first, the trust decision needs it")
            from fastmace.model import loading

            model = loading.load_pristine(model, head)
        elif head is not None and head != getattr(model, "head", None):
            raise ValueError(f"head={head!r} does not match the loaded model's head {getattr(model, 'head', None)!r}; "
                             "pass head= to fm.load instead")
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
        # Checks them and expands "d3bj"; each plan gets its own force field.
        self.dispersion = ForceField(model, precision=precision, rewrites=rewrites, dispersion=dispersion).dispersion
        self.model, self.precision, self.rewrites = model, precision, rewrites
        self.implemented_properties = list(type(self).implemented_properties)
        if _atom_resolved(model.model) and self.dispersion is None:
            self.implemented_properties.append("energies")
        self.plan = None
        self.notes: list[str] = []
        self._key = None
        self._evaluator = None
        self._compile_cache = None

    def _evaluator_for(self, atoms):
        """The ``MACEForceField`` for ``atoms``; plans again when the species or the atom count change."""
        key = (tuple(int(z) for z in atoms.numbers), tuple(bool(flag) for flag in atoms.pbc))
        if key == self._key:
            return self._evaluator
        from fastmace.model.compile import CompiledModelCache
        from fastmace.model.forcefield import ForceField
        from fastmace.planner import minimal

        force_field = ForceField(self.model, atoms, precision=self.precision, rewrites=self.rewrites,
                                 dispersion=self.dispersion)
        planner = minimal.default_planner(force_field)
        plan = planner.plan_evaluation(force_field, atoms)  # rule 1 (3D PBC) and the others refuse here
        if self._compile_cache is None:
            self._compile_cache = CompiledModelCache()
        self._key, self._evaluator = None, None  # free the previous copy before building the next
        evaluator = planner.build_evaluation(plan, force_field, atoms, compile_cache=self._compile_cache)
        self.plan, self._key, self._evaluator = plan, key, evaluator
        if NOTE not in self.notes:
            self.notes.insert(0, NOTE)
            print(NOTE)
        return evaluator

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        from ase.stress import full_3x3_to_voigt_6_stress

        Calculator.calculate(self, atoms, properties, system_changes)
        atoms = self.atoms
        evaluator = self._evaluator_for(atoms)
        energy, forces, stress = evaluator.evaluate(atoms, stress=True)
        self.results = {"energy": energy, "free_energy": energy, "forces": forces,
                        "stress": full_3x3_to_voigt_6_stress(stress)}
        if "energies" in properties and "energies" in self.implemented_properties:
            self.results["energies"] = evaluator.per_atom_energies(atoms).energies


def calc(model: Any, *, head: str | None = None, **options: Any) -> FastMACECalculator:
    """``atoms.calc = fm.calc("mh1", head="matpes_r2scan")``: a ``FastMACECalculator`` for ``model``
    (label, path or URL, loaded as ``fm.load`` does); prints the compatibility-tier note once.

    ``options``: ``fm.load``'s ``sha256``, ``trusted``, ``download``, ``charge`` and
    ``spin_multiplicity``, and the calculator's ``rewrites``, ``precision`` and ``dispersion``.
    """
    unknown = sorted(set(options) - set(_LOAD_OPTIONS) - set(_CALCULATOR_OPTIONS))
    if unknown:
        raise TypeError(f"fm.calc got unexpected keyword arguments {', '.join(unknown)}; it takes "
                        f"{', '.join(_LOAD_OPTIONS + _CALCULATOR_OPTIONS)}")
    load = {name: options[name] for name in _LOAD_OPTIONS if name in options}
    if isinstance(model, str):
        from fastmace.model import loading

        model = loading.load_pristine(model, head, **load)
        head = None
    elif load:
        raise TypeError(f"{', '.join(sorted(load))} apply to loading; model is already loaded")
    return FastMACECalculator(model, head=head, **{name: options[name] for name in _CALCULATOR_OPTIONS
                                                     if name in options})


def _is_label(name):
    from fastmace.model import labels

    try:
        return labels.find(name) is not None
    except ValueError:  # an alias of a multi-head checkpoint without head=: fm.load says which heads
        return True


def _atom_resolved(model):
    """True when every energy term of ``model`` is atom-resolved (§7.3)."""
    from fastmace.errors import FastMACEError
    from fastmace.model import energy

    try:
        return not energy.terms(model)["unresolved"]
    except FastMACEError:  # no float64 reduction for this class: the planner refuses it with the reason
        return False


__all__ = ("NOTE", "FastMACECalculator", "calc")
