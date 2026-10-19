"""The public, plan-free force field ``fm.ForceField`` (SPEC §7.1, §7.6 "ForceField and plans").

Plan task 1b-T01 fixed the signature; plan task 1b-T12 implements it. SPEC §5's ``forces.py`` holds
phase 1a's ``MACEForceField``, the execution copy a plan builds; this module holds the object
the user holds. It keeps the pristine loaded model, the force precision, the species superset and
the force-field identity (§10.6, ``state.ForceFieldIdentity``), and builds nothing itself: every
plan, including the one-off plan of ``evaluate``, gets its own execution copy through the planner
(``fastmace.planner.minimal``).
"""

from __future__ import annotations

from typing import Any, Literal

PRECISIONS = ("float32", "float64")


class ForceField:
    """``ForceField(model, atoms=None, *, precision="float32", rewrites="auto", species=None)``.

    - ``model``: an ``fm.load`` result (a ``LoadedModel``, normally the pristine e3nn float64 CPU
      model) or a label, which is loaded with ``fm.load``. A path needs ``fm.load`` first, because
      the trust decision needs one (§7.6);
    - ``atoms``: optional; its species are the default superset (§7.1 passes it). Without it the
      superset is the species of the first structure the force field is used with;
    - ``precision``: the force precision, ``"float32"`` or ``"float64"`` (§7.3);
    - ``rewrites``: ``"auto"`` or a ``RewriteConfig``; resolved per plan;
    - ``species``: the declared superset (§8.5), symbols or atomic numbers, compared as a set; a
      species without an embedding raises ``ModelElementError``. In 1b it must equal the
      structure's species (owner decision D3; the planner refuses a larger one).
    """

    def __init__(self, model: Any, atoms: Any = None, *, precision: Literal["float32", "float64"] = "float32",
                 rewrites: Any = "auto", species: Any = None):
        from fastmace.config import RewriteConfig

        if isinstance(model, str):
            from fastmace.model import loading

            if not _is_label(model):
                raise TypeError(f"{model!r} is not a label of the model table; load a checkpoint path or URL with "
                                "fm.load(path, sha256=... or trusted=True) first, the trust decision needs it")
            model = loading.load_pristine(model)
        if not hasattr(model, "calculator") or not hasattr(model, "sha256"):
            raise TypeError(f"ForceField takes an fm.load result or a label, got {type(model).__name__}; use fm.load")
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
        if not (rewrites == "auto" or isinstance(rewrites, RewriteConfig)):
            raise ValueError(f"rewrites must be 'auto' or a RewriteConfig, got {rewrites!r}")
        self.model = model
        self.precision = precision
        self.rewrites = rewrites
        self._traits = None
        self._evaluations = {}
        self.species = None
        if species is not None:
            from fastmace.planner.minimal import species_numbers

            self._fix_superset(species_numbers(species))
        elif atoms is not None:
            self._fix_superset(tuple(sorted({int(z) for z in atoms.numbers})))

    # ------------------------------------------------------------------ facts the planner reads

    @property
    def traits(self):
        """The model traits (``capabilities.ModelTraits``) of the pristine model."""
        if self._traits is None:
            from fastmace.model.capabilities import model_traits

            self._traits = model_traits(self.model.model)
        return self._traits

    def _fix_superset(self, numbers):
        from fastmace.errors import ModelElementError
        from fastmace.planner.minimal import _symbols

        known = set(self.traits.atomic_numbers)
        unknown = sorted(set(numbers) - known) if known else []
        if unknown:
            raise ModelElementError(
                f"species {_symbols(unknown)} have no embedding in the model (it has {_symbols(known)})",
                why="the model was not trained on them, so it has no input for them (SPEC §8.5)")
        self.species = tuple(numbers)

    def superset_for(self, atoms) -> tuple[int, ...]:
        """The superset, fixed from ``atoms`` at first use if none was declared (atomic numbers, sorted)."""
        if self.species is None:
            self._fix_superset(tuple(sorted({int(z) for z in atoms.numbers})))
        return self.species

    @property
    def identity(self):
        """The ``state.ForceFieldIdentity`` (SPEC §10.6); ``None`` until the superset is known."""
        if self.species is None:
            return None
        import fastmace
        from fastmace.provenance import _mace_version
        from fastmace.state import ForceFieldIdentity

        model = self.model
        return ForceFieldIdentity(
            checkpoint_sha256=model.sha256, head=model.head or None, species=self.species,
            force_precision=self.precision,
            physics={"charge": model.charge, "spin_multiplicity": model.spin_multiplicity},
            mace_version=_mace_version() or "unknown", fastmace_version=fastmace.__version__)

    # ------------------------------------------------------------------ for the planner

    def execution_model(self, plan):
        """The model converted to ``plan``'s backend, precision and device (a ``LoadedModel``).

        Called by the planner only, once per plan. The checkpoint is loaded again, its hash checked
        against the pristine model's, so the pristine model is never changed and no two plans share
        a model (the per-model state that SPEC §8.7's instance isolation is about stays per plan).
        """
        from fastmace.model import loading

        engine = plan.engine
        fusion = engine.cueq_conv_fusion == "on" if engine.backend == "cueq" else None
        allow = "allow_naive_cueq" in plan.allowances
        model = self.model
        return loading.load(
            model.label or str(model.path), None if model.label else model.head, checkpoint=str(model.path),
            backend=engine.backend, dtype=plan.precision.dtype, device=engine.device, cueq_conv_fusion=fusion,
            sha256=model.sha256, allow_naive_cueq=allow, charge=model.charge, spin_multiplicity=model.spin_multiplicity)

    # ------------------------------------------------------------------ one-off evaluation

    def evaluation_plan(self, atoms):
        """The one-off evaluation plan ``evaluate`` uses for structures like ``atoms`` (planned once)."""
        return self._evaluator(atoms)[0]

    def _evaluator(self, atoms):
        key = tuple(int(z) for z in atoms.numbers)
        if key not in self._evaluations:
            from fastmace.planner import minimal

            planner = minimal.default_planner(self)
            plan = planner.plan_evaluation(self, atoms)
            self._evaluations[key] = (plan, planner.build_evaluation(plan, self, atoms))
        return self._evaluations[key]

    def evaluate(self, atoms: Any, stress: bool = False) -> tuple[float, Any, Any]:
        """``(energy, forces, stress)`` of ``atoms`` as host copies, through a one-off minimal plan;
        ``stress`` is ``None`` unless asked for. The energy is accumulated in float64 (§7.3)."""
        if not all(atoms.pbc):
            from fastmace.errors import PlanRefused, Reason

            raise PlanRefused.from_reasons([Reason(
                f"periodic boundaries in all three directions (3D PBC) are required, got pbc={tuple(atoms.pbc)} "
                "(SPEC §4.4)", rule=1, code="non-3d-pbc")])
        return self._evaluator(atoms)[1].evaluate(atoms, stress=stress)


def _is_label(name):
    from fastmace.model import labels

    try:
        return labels.find(name) is not None
    except ValueError:  # an alias of a multi-head checkpoint without head=: fm.load says which heads
        return True
