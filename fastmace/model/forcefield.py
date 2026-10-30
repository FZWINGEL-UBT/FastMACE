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
      species without an embedding raises ``ModelElementError``. It may be larger than the
      structure's species (phase 3, rule 8): every species-indexed table is built for it, so an atom
      can become any superset species in place (3-T02);
    - ``dispersion``: ``None``, ``"d3bj"`` or an ``fm.D3`` (SPEC §7.7, §11.2). ``"d3bj"`` expands
      from the model's training functional (``fastmace.longrange.d3.expand``: ``ValueError`` when it
      is unknown or already contains dispersion); an explicit record whose functional differs from
      the training one is kept with a warning in ``dispersion_warnings``. The complete record is
      ``dispersion`` and part of the identity. Refused (``PlanRefused``): the ATM term
      (``d3-three-body``) and PolarMACE (``d3-polar``, decision 41). Plans carry the record and put
      the D3 term into their force field (``fastmace.model.dispersed``, 3b-T04).
    """

    def __init__(self, model: Any, atoms: Any = None, *, precision: Literal["float32", "float64"] = "float32",
                 rewrites: Any = "auto", species: Any = None, dispersion: Any = None, polar: Any = None):
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
        self.dispersion, self.dispersion_warnings = _dispersion(dispersion, model)
        self.polar, self.polar_warnings = _polar(polar, model)
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

    def effective_precision(self, float64_energy=True):
        """``(precision, reason)``: the force precision plans run at, and why it differs from ``precision``.

        On a MACE whose float64 energy forward does not hash, a float32 force field runs at float64
        (``fastmace.rewrites.versions.effective_precision``; plan task 2-T04); ``reason`` is then a
        ``Reason``, else None.
        """
        from fastmace.longrange import polar
        from fastmace.rewrites import versions

        return versions.effective_precision(self.precision, polar_model=polar.is_polar(self.model.model),
                                            float64_energy=float64_energy)

    @property
    def identity(self):
        """The ``state.ForceFieldIdentity`` (SPEC §10.6); ``None`` until the superset is known.

        Its force precision is the effective one (``effective_precision``), so a resume on the same
        MACE re-plans the same way and the identity matches. Known limitation (2-T04): it assumes
        ``float64_energy=True``. A plan with ``PrecisionConfig(float64_energy=False)``, which is for A/B
        against the research engine only, on a MACE whose float64 energy forward does not hash runs
        at float32 but records a float64 identity.
        """
        if self.species is None:
            return None
        import fastmace
        from fastmace.provenance import _mace_version
        from fastmace.state import ForceFieldIdentity

        model = self.model
        physics = {"charge": model.charge, "spin_multiplicity": model.spin_multiplicity}
        if self.dispersion is not None:  # absent without D3, so D3-free identities keep their digests
            from fastmace.longrange import d3

            physics["dispersion"] = d3.identity_entry(self.dispersion)
        options = _pr1762_options(self.polar)
        if options is not None:  # absent without MACE PR 1762's options, so other identities keep their digests
            physics["polar"] = options
        return ForceFieldIdentity(
            checkpoint_sha256=model.sha256, head=model.head or None, species=self.species,
            force_precision=self.effective_precision()[0], physics=physics,
            mace_version=_mace_version() or "unknown", fastmace_version=fastmace.__version__)

    # ------------------------------------------------------------------ for the planner

    def execution_model(self, plan):
        """The model converted to ``plan``'s backend, precision and device (a ``LoadedModel``).

        Called by the planner only, once per plan. The checkpoint is loaded again, its hash checked
        against the pristine model's, so the pristine model is never changed and no two plans share
        a model (the per-model state that SPEC §8.7's instance isolation is about stays per plan).
        On CUDA with cuEq or hybrid, a later plan of the same conversion gets a private copy of the
        first plan's converted model instead (``fastmace.model.templates``; opt-1 OPT-T06), unless
        ``EngineConfig.model_template_cache`` is False. The templates belong to the loaded model, so a
        later force field of the same ``fm.load`` result (another composition) copies them too.
        """
        from fastmace.model import loading, templates

        engine = plan.engine
        fusion = engine.cueq_conv_fusion == "on" if engine.backend == "cueq" else None
        allow = "allow_naive_cueq" in plan.allowances
        model = self.model
        source, head = model.label or str(model.path), None if model.label else model.head
        options = dict(checkpoint=str(model.path), backend=engine.backend, dtype=plan.precision.dtype,
                       device=engine.device, cueq_conv_fusion=fusion, sha256=model.sha256, allow_naive_cueq=allow,
                       charge=model.charge, spin_multiplicity=model.spin_multiplicity)

        def load():
            return loading.load(source, head, **options)

        # CUDA cuEq and hybrid: later plans copy the first plan's converted model instead of converting
        # again (opt-1 OPT-T06); each plan still gets its own model, and the template is never handed out.
        if not (getattr(engine, "model_template_cache", True) and templates.applies(engine.backend, engine.device)):
            return load()
        return model.template_cache().execution_model(templates.conversion_key(model, options), load)

    def template_count(self) -> int:
        """How many converted models are kept as templates for this force field's model (opt-1 OPT-T06). The
        templates belong to the loaded model, so force fields of the same ``fm.load`` result share them."""
        cache = self.model.template_cache(create=False)
        return 0 if cache is None else len(cache)

    def release_templates(self) -> int:
        """Drop the converted models kept as templates; returns the bytes their tensors held.

        The next plan of each conversion converts the checkpoint again. Use it to give the memory back,
        for example before an out-of-memory retry (opt-1 OPT-T06). The templates belong to the loaded
        model: this drops them for every force field made from it.
        """
        cache = self.model.template_cache(create=False)
        return 0 if cache is None else cache.release()

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


# MACE PR 1762 through ForceField(polar=): wired through the planner, the gate's reference and restarts, but no
# PolarMACE run with these options has been validated (owner, 28 October 2026: "add it but flag it as not tested").
POLAR_NOT_TESTED = ("PolarMACE with MACE PR 1762 options (fukui_regularisation={fukui:g}, dipole_cap={cap}): "
                    "not tested; these options change the model and no FastMACE run with them has been validated")


def _polar(polar, model):
    """``(PolarConfig, warnings)`` of ``ForceField(polar=...)``: the default without one; a non-default record is
    refused for a model that is not PolarMACE, and MACE PR 1762's options add the not-tested warning."""
    from fastmace.config import PolarConfig
    from fastmace.longrange import polar as longrange

    if polar is None:
        return PolarConfig(), ()
    if not isinstance(polar, PolarConfig):
        raise TypeError(f"polar must be a fastmace.config.PolarConfig or None, got {type(polar).__name__}")
    if polar != PolarConfig() and not longrange.is_polar(model.model):
        raise ValueError(f"polar={polar!r} applies to PolarMACE (MACE-POLAR-1) only, not {type(model.model).__name__}")
    if not longrange.stabilised(polar):
        return polar, ()
    return polar, (POLAR_NOT_TESTED.format(fukui=polar.fukui_regularisation, cap=polar.dipole_cap),)


def _pr1762_options(polar):
    """MACE PR 1762's options of ``polar`` as plain values, or ``None`` when it asks for none."""
    from fastmace.longrange import polar as longrange

    if not longrange.stabilised(polar):
        return None
    return {"fukui_regularisation": float(polar.fukui_regularisation),
            "dipole_cap": None if polar.dipole_cap is None else list(polar.dipole_cap)}


def _dispersion(dispersion, model):
    """``(record, warnings)`` of ``ForceField(dispersion=...)``; refuses what no D3 term implements."""
    if dispersion is None:
        return None, ()
    from fastmace.errors import PlanRefused, Reason
    from fastmace.longrange import d3, polar

    record, warnings = d3.expand(dispersion, functional=model.functional,
                                 model=model.label or str(getattr(model, "path", "the model")))
    reasons = []
    if record.three_body:
        reasons.append(Reason(f"the D3 three-body (ATM) term is not implemented; got {d3.describe(record)} "
                              "(SPEC §11.2)", code="d3-three-body"))
    if polar.is_polar(model.model):
        reasons.append(Reason("PolarMACE with D3 is refused in phase 3b: PolarMACE NPT is not qualified "
                              "(SPEC §18 decision 41)", code="d3-polar"))
    if reasons:
        raise PlanRefused.from_reasons(reasons)
    return record, tuple(warnings)


def _is_label(name):
    from fastmace.model import labels

    try:
        return labels.find(name) is not None
    except ValueError:  # an alias of a multi-head checkpoint without head=: fm.load says which heads
        return True
