"""The one-screen run report (SPEC §7.7 "The report"; setup stages of §12.5).

``fm.md``, the ``fastmace.ase`` integrators and the CLI keep one report, in ``summary`` and in
``metadata.json`` under ``"report"`` (written by ``provenance.write_metadata``). It is built in
two stages:

1. at setup, everything known then: header, physics, plan, gate, setup time, outputs, licence and
   warnings, with ``speed: pending`` (``Report.setup_text()``);
2. once the steady-state step time is measured over a fixed window after warm-up, the speed line
   is filled in and printed once (``record_speed``); at the end of the run the whole-run mean
   replaces it (``finish``). ``Simulation`` (1b-T17) owns the step timing and feeds it in.

Speeds are measured, never predicted (§9.1 forbids per-GPU tables): the report has no expected
figure and no comparison between tiers. The licence line is read from the label table
(``fastmace.model.labels``) for every model, never written by hand. When the run did not get the
plan it would get on a complete installation, or was refused, the report says why and what would
change it, matching the planner's stable reason codes (``docs/design/plan-1b.md``), never text.

This module imports neither torch nor MACE.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

from fastmace.model import labels as label_table
from fastmace.planner.fixed import REWRITE_FIELDS

INDENT = 9  # "licence: " is the longest field name
FALLBACK_PREFIX = "fallback: "  # planner.fallback.PREFIX: the reason of a fallback revision (2-T15)
DEFAULT_WARMUP_STEPS = 1000
DEFAULT_WINDOW_STEPS = 1000
SETUP_STAGES = ("load", "rewrites", "compilation", "capture", "gate")  # SPEC §12.5; others are kept as given
GATE_OPT_IN = 'not run (opt-in: check="gate")'  # owner decision D1

# Refusals (PlanRefused reasons) by code: why, and what would change it. SPEC §7.7, §9.3.
REFUSALS = {
    "non-3d-pbc": "not 3D-periodic: tiers 1-3 need 3D PBC (§4.4); put the structure in a periodic box, or use "
                  "MACE's own calculator (the reference tier is phase 2+, allow_reference is refused in 1b)",
    "constraints": "constraints are refused until the integrators support them (phase 4, §4.4); remove them",
    "variable-cell-without-barostat": "a variable cell needs a barostat integrator; use LangevinBAOAB (NPT)",
    "reference-tier-not-in-1b": "the reference tier (allow_reference=True) is not in 1b (owner decision D6); "
                                "use MACE's own calculator",
    "species-not-in-superset": "a species is outside the ForceField's superset (rule 2); declare it in "
                               "fm.ForceField(species=...)",
    "polar-plain-compiled-cueq": "PolarMACE with compiled plain cuEq gives zero forces (rule 3, §4.3); use the "
                                 "hybrid backend (OEq; `fastmace doctor` names a missing piece)",
    "inexact-padding": "padded edges would change this model's energies (rule 4); pass mode='compiled', or no "
                       "mode, for the compiled tier with exact edge counts",
    "reproducible-backend": "reproducible=True needs plain cuEq (rule 5): the hybrid's OEq kernels sum atomically; "
                            "leave the backend on auto",
    "reproducible-conv-fusion": "reproducible=True needs the unfused convolution (rule 5, owner decision 13); leave "
                                "cueq_conv_fusion on auto",
    "reproducible-family-not-qualified": "reproducible=True is qualified for plain cuEq MH-1 only (rule 5, §9.3); "
                                         "drop reproducible=True for statistical reproducibility",
    "reproducible-cueq-version": "reproducible=True: cuEquivariance's run-time kernel choice is pinned for "
                                 "cuequivariance_ops 0.12.0 only (rule 5); install it, or drop reproducible=True",
    "reproducible-tier": "reproducible=True runs on the graph or eager tier, not the compiled or reference tier "
                         "(rule 5); drop that mode or reproducible=True",
    "reproducible-kernel-not-deterministic": "reproducible=True: a rewrite in use is not declared deterministic "
                                             "(rule 5); switch it off, or drop reproducible=True",
    "naive-cueq": "cuEquivariance's fused kernels do not load (rule 6); install cuequivariance-ops-torch "
                  "(`fastmace doctor`), or pass allow_naive_cueq=True to run slowly, recorded",
    "atom-count-changes": "a changing atom count is not in 1.0 (rule 7, §8.6)",
    "superset-beyond-structure": "a species superset beyond the structure's species comes in phase 3 (§8.5, "
                                 "owner decision D3); pass species equal to the structure's",
    "mode-compiled": "MD on the compiled tier was refused (1b; the tier exists since 3-T06)",
    "mode-eager": "MD on the eager tier is not in 1b; 1b runs MD on the graph tier",
    "mode-reference": "MD on the reference tier is not in 1b (owner decision D6); use MACE's own calculator",
    "hybrid-unavailable": "backend='hybrid' was requested but cannot run here; `fastmace doctor` names the "
                          "missing piece, or leave the backend on auto",
    "hybrid-needs-cuda": "backend='hybrid' needs a CUDA device; leave the backend on auto",
    "conv-fusion-needs-cueq": "cueq_conv_fusion needs the plain cuEq backend; leave it on auto",
    "inplace-block-linear-not-carried-over": "inplace_block_linear was not carried over (§6.2); leave it off",
    "speculative-graph-npt": "speculative_graph does not support NPT; leave it on auto",
    "pipelined-excludes-speculative": "pipelined and speculative_graph exclude each other; switch one off",
    "rewrite-requirement-missing": "a rewrite requested 'on' needs another rewrite that is off; leave it on auto",
    "rewrite-unsupported-by-model": "a rewrite requested 'on' is not supported by this model; leave it on auto",
    "reassociation-plain-cueq": "reassociate_first is refused on plain cuEq; leave it on auto",
    "streaming-needs-sorted-receivers": "streaming_conv needs sort_receivers on; leave both on auto",
    "rewrite-needs-hybrid": "a rewrite requested 'on' needs the hybrid backend; leave it on auto",
    "unknown-workload-constraint": "a rewrite declares a workload constraint the planner does not know; report "
                                   "this as a FastMACE bug",
}

# Downgrades: decision codes that mean the run is slower than on a complete installation. SPEC §7.7.
DOWNGRADES = {
    "backend-cueq-cpu": "plain cuEq: CPU device, the hybrid backend needs CUDA",
    "backend-cueq-oeq-unavailable": "plain cuEq, not hybrid: OpenEquivariance cannot build; `fastmace doctor` "
                                    "names the missing piece",
    "backend-cueq-mace-imported-first": "plain cuEq, not hybrid: MACE was imported before FastMACE could prepare "
                                        "OpenEquivariance; import fastmace first",
    "backend-cueq-process-family": "plain cuEq, not hybrid: this process already compiled the other backend "
                                   "family (§8.7); run in a fresh process",
    "backend-cueq-requested": "plain cuEq as requested (backend='cueq'); auto would choose hybrid where it builds",
    "host-callbacks-graph-sync": "host callbacks: graph tier with a host sync at each callback (rule 9, decision "
                                 "38); the compiled tier was 1.3-4.7x slower even with a callback every step",
    "mode-eager-tier": "eager tier: no CUDA graphs, no compilation (mode='eager', SPEC §8.3); correct but slower",
    # The compiled tier (3-T06; SPEC §8.2): no CUDA graphs, so every step is launched from the host.
    "mode-compiled-tier": "compiled tier: no CUDA graphs (mode='compiled', SPEC §8.2); correct but slower",
    "mode-compiled-rule-4": "compiled tier: no CUDA graphs, because padded edges would change this model's "
                            "energies (rule 4, SPEC §8.2); exact edge counts, correct but slower",
    "precision-promoted-unqualified-energy": "float64 forces, promoted from float32: this MACE is not the "
                                             "qualified 0.3.16 and its float64 energy forward does not hash, so a "
                                             "float64 model sums the energy (SPEC §18 decision 2)",
    "allow-naive-cueq": "naive cuEq allowed (allow_naive_cueq=True): cuEquivariance's fused kernels do not load, "
                        "so this run is slow; install cuequivariance-ops-torch (`fastmace doctor`)",
    "backend-cueq-reproducible": "plain cuEq with the unfused convolution, not hybrid: reproducible=True (rule 5, "
                                 "owner decision 13), slower than the default plan",
    "reproducible-qualified": "reproducible=True relies on CUEQUIVARIANCE_OPS_U1D_ALGOS, an undocumented internal "
                              "of cuequivariance_ops 0.12.0, to pin cuEq's run-time kernel choice (checked at "
                              "build: two evaluations bitwise equal)",
}

# SPEC §4.3 model-level known issues, as data (conductor answer for 1b): labels, the species the
# structure must contain (all of them), an extra condition ("" for none), and the warning.
KNOWN_ISSUES = (
    (("polar1-s",), ("Na", "Nb", "O"), "",
     "POLAR-1-S has a Fukui charge runaway on NaNbO3 (SPEC §4.3)"),
    (("polar1-s", "polar1-m", "polar1-l"), ("Na", "Ta", "O"), "",
     "POLAR-1 gives huge forces and pressures on NaTaO3 (SPEC §4.3)"),
    (("omol0-xl",), ("Na", "Cl"), "",
     "OMOL-0 is outside its training domain on dense inorganic solids and collapses in NaCl MD (SPEC §4.3)"),
    (("omol0-xl",), ("O",), "inorganic",
     "OMOL-0 is outside its training domain on dense oxides (SPEC §4.3)"),
    (("off23-small", "off23-medium", "off23-large", "off23b-medium"), (), "npt",
     "MACE-OFF23 relaxes molecular crystals to 5.5-9.2% below the experimental volume; OFF24-medium is closest "
     "(SPEC §4.3)"),
    (("off23-large",), (), "large",
     "MACE-OFF23-large on about 2,000 atoms or more needs more than 16 GB in some engines (SPEC §4.3)"),
)
_LARGE_ATOMS = 2000
_ORGANIC = frozenset(("H", "C"))


def _number(value):
    """``1,000`` for integers; three significant digits otherwise."""
    return f"{value:,}" if isinstance(value, int) else f"{value:.3g}"


def _ms(seconds):
    return f"{seconds * 1e3:.3g} ms/step"


def _seconds(value):
    return f"{value:.0f} s" if value >= 10 else f"{value:.2g} s"


def _positive(value, what):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{what} must be a positive finite number of seconds, got {value!r}")
    return float(value)


def describe_move(move):
    """One MC move record in words: ``swap Na-Cl every 100, 50 attempts (exchange)``."""
    from ase.data import chemical_symbols

    def symbol(species):
        return chemical_symbols[species] if isinstance(species, int) else str(species)

    kind = getattr(move, "move", type(move).__name__.lower())
    if kind == "swap":
        what = "swap " + ", ".join(f"{symbol(a)}-{symbol(b)}" for a, b in move.pairs)
    elif kind == "transmute":
        what = "transmute " + "/".join(symbol(s) for s in move.species)
    elif kind == "displace":
        what = f"displace δ {move.delta_A:g} Å" + (" (adaptive)" if move.adaptive else "")
    else:
        what = str(kind)
    text = f"{what} every {_number(move.every)}, {_number(move.attempts)} attempts"
    velocities = getattr(move, "velocities", None)
    return text + (f" ({velocities})" if velocities else "")


# The header's name of each integrator scheme (IntegratorConfig.scheme); MTK and Berendsen add the cell mode.
SCHEME_NAMES = {"langevin": "NVT Langevin", "baoab": "NPT LangevinBAOAB", "nve": "NVE VelocityVerlet",
                "csvr": "NVT CSVR (Bussi)", "nhc": "NVT Nose-Hoover chain", "mtk": "NPT MTK",
                "berendsen": "Berendsen"}


def _scheme_name(config):
    """The header's integrator: ``SCHEME_NAMES`` by the config's scheme, with the MTK cell mode and the
    Berendsen ensemble."""
    scheme = getattr(config, "scheme", None)
    name = SCHEME_NAMES.get(scheme, f"{str(config.ensemble).upper()} {scheme}")
    if scheme == "mtk":
        name = f"{name} {config.cell_mode}"
    elif scheme == "berendsen":
        name = f"{str(config.ensemble).upper()} {name}"
    return name


def _scheme_record(config):
    """The registry record of the config's scheme, ``None`` if it is not registered."""
    from fastmace.integrators import registry

    try:
        return registry.registry().integrator(config.scheme)
    except KeyError:
        return None


def _mapping(record):
    """A gate or D3 record as a dict: a mapping, ``to_metadata()``/``to_dict()``, a dataclass, or attributes."""
    if record is None or isinstance(record, Mapping):
        return None if record is None else dict(record)
    for method in ("to_metadata", "to_dict"):
        if callable(getattr(record, method, None)):
            return dict(getattr(record, method)())
    if dataclasses.is_dataclass(record) and not isinstance(record, type):
        return dataclasses.asdict(record)
    return {name: getattr(record, name) for name in dir(record) if not name.startswith("_")
            and not callable(getattr(record, name))}


def _code(reason):
    return getattr(reason, "code", None)


def licence(model=None, sha256=None):
    """(licence, line) from the label table: by checkpoint SHA-256 first, then by label or alias."""
    entry = None
    if sha256:
        entry = next((c for c in label_table.CHECKPOINTS.values() if c.sha256 == sha256), None)
    if entry is None and model is not None:
        name = str(model)
        if name in label_table.LABELS:
            entry = label_table.LABELS[name].checkpoint
        elif name in label_table.CHECKPOINTS or name in label_table.ALIASES:
            entry = label_table.CHECKPOINTS[label_table.ALIASES.get(name, name)]
    if entry is None:
        digest = f" (SHA-256 {sha256[:12]}…)" if sha256 else ""
        return None, f"not a label of the FastMACE table{digest}; check the model's release"
    text = {label_table.ASL: "ASL (non-commercial)"}.get(entry.licence, entry.licence)
    return entry.licence, f"{text}, from the label table (§4.3)"


def _label_of(model, head, sha256):
    """The label-table label for ``model`` (a label, or a checkpoint found by SHA-256 and head), or None."""
    if model is not None and str(model) in label_table.LABELS:
        return str(model)
    for name, entry in label_table.LABELS.items():
        if sha256 and entry.checkpoint.sha256 == sha256 and entry.head in (head, None):
            return name
    return None


def _floors_text(floors):
    """"floors §12.3", plus each floor a reference record raised (4-T12: PolarMACE's stress, decision 49)."""
    from fastmace.planner.selfcheck import FLOORS

    raised = [f"{_GATE_NAMES[key]} {value:g} {unit}" for key, unit in _GATE_UNITS
              if (value := (floors or {}).get(key, FLOORS[key])) != FLOORS[key]]
    return "floors §12.3" + (f", {', '.join(raised)} from the reference record" if raised else "")


_GATE_UNITS = (("energy_meV_per_atom", "meV/atom"), ("force_eV_per_A", "eV/Å"), ("stress_eV_per_A3", "eV/Å³"))
_GATE_NAMES = {"energy_meV_per_atom": "energy", "force_eV_per_A": "force", "stress_eV_per_A3": "stress"}


def gate_line(gate) -> str:
    """The report's gate line from a gate record (mapping); the not-run text for none or ``skipped``.

    After the first step of the gate's resource ladder (§12.3 "Resources", 2-T05) it names where the
    references ran: the CPU after an out-of-memory, or the reduced cell (``passed-reduced``).
    """
    gate = _mapping(gate)
    status = None if gate is None else gate.get("status")
    if status in (None, "skipped"):
        return GATE_OPT_IN
    if status == "unavailable":
        reason = gate.get("reason") or "no reference ran"
        return f'unavailable ({reason}); run allowed by check="allow_unavailable"'
    numbers = []
    for key, unit in (("energy_meV_per_atom", "meV/atom"), ("force_eV_per_A", "eV/Å"),
                      ("stress_eV_per_A3", "eV/Å³")):
        if gate.get(key) is not None:
            numbers.append(f"{gate[key]:.2g} {unit}")
    where = None
    steps = gate.get("steps") or ()
    if gate.get("reduced_cell"):
        where = (f"reduced cell {'×'.join(str(m) for m in gate['reduced_cell'])} of the primitive cell on "
                 f"{gate.get('device')}, replay checked on the production cell")
    elif len(steps) > 1 and gate.get("device"):
        where = f"reference on {gate['device']} after {steps[0].get('outcome')}"
    parts = [part for part in (where, gate.get("copies"), ", ".join(numbers)) if part]
    detail = "; ".join(parts + [_floors_text(gate.get("floors"))])
    yardstick = (gate.get("references") or {}).get("yardstick") or {}
    if str(yardstick.get("energy_bound", "")).startswith("looser"):  # 2-T04: unqualified MACE
        detail = f"{detail}; energy bound {yardstick['energy_bound']}"
    if gate.get("reason") and status not in ("passed", "passed-reduced"):
        detail = f"{gate['reason']}; {detail}"
    return f"{status} ({detail})"



def move_energy_line(gate) -> str | None:
    """The report's MC energy line from a gate record's move-energy part (3-T11; SPEC §12.3, §10.4, decision 35):
    the MC energy precision and why, the worst ΔE error per move type in k_B T_min against 0.01. ``None``
    without an MC plan."""
    gate = _mapping(gate)
    part = None if gate is None else gate.get("move_energy")
    if not part:
        return None
    precision, status = part.get("precision"), part.get("status")
    limit = f"limit {part.get('fraction', 0.01):g} k_B T_min at T_min {part.get('min_mc_temperature_K', 0):g} K"
    if status == "unavailable":
        return f"{precision}; move-energy gate unavailable ({part.get('reason')})"
    moves = part.get("moves") or {}
    worst = ", ".join(f"{move} {entry['worst_k_B_T_min']:.2g}" for move, entry in moves.items())
    text = f"{precision}; move-energy gate {status} (worst ΔE error in k_B T_min: {worst or 'none'}; {limit}"
    text += f"; {part.get('configurations', 0)} configurations"
    if part.get("not_sampleable"):
        text += f"; {len(part['not_sampleable'])} not sampleable"
    text += ")"
    tried = part.get("tried") or ()
    if len(tried) > 1:
        steps = ", ".join(f"{entry['precision']} {entry['status']}" for entry in tried)
        text += f"; {steps} (MD forces keep their precision)"
    return text


def revision_lines(plan) -> list[str]:
    """One line per fallback revision of ``plan``'s chain, oldest first (SPEC §9.4, §9.5; 2-T15):
    ``plan revision 1 (fallback): rewrites X, Y off, reason …``. Empty when nothing fell back."""
    if plan is None:
        return []
    lines = []
    for revision in (*plan.chain(), plan):
        reason = str(revision.reason)
        if reason.startswith(FALLBACK_PREFIX):
            change, _, why = reason[len(FALLBACK_PREFIX):].partition(": ")
            lines.append(f"plan revision {revision.revision} (fallback): {change}" + (f", reason {why}" if why else ""))
    return lines

class Report:
    """The run report of one run (SPEC §7.7). Build it at setup; feed it the speed; read ``text()``.

    Keyword arguments (everything the run knows at setup):

    - ``model`` (label or display name), ``head``, ``sha256``: the header and the licence line;
    - ``structure`` (name), ``n_atoms``, ``repeat``, ``pbc``, ``species`` (symbols, for §4.3 warnings);
    - ``plan``: the ``Plan`` (§9.5); or ``refusal``: the ``PlanRefused`` when there is no plan, with
      ``integrator`` (an ``IntegratorConfig``) for the header;
    - ``gate``: the gate record (mapping or record with ``to_metadata()``; keys ``status``, ``reason``,
      ``copies``, ``energy_meV_per_atom``, ``force_eV_per_A``, ``stress_eV_per_A3``); ``None`` means
      not requested (D1);
    - ``setup_s``: ``{stage: seconds}``, §12.5's stage names or finer ones, in order;
    - ``outputs``: the output files; ``dispersion``: ``None`` or the ``fm.D3`` record;
    - ``charge_spin``: ``None`` when the model does not read them, else ``(total charge, spin multiplicity)``;
    - ``mc``: the MC record of the run (``Simulation`` keeps it, 3-T15), or ``None`` without MC: ``moves``
      (the records attached last), ``ranges`` (``[first, last]`` step ranges with a move set attached),
      ``blocks``, ``device_ms`` and ``timed_attempts`` (device time and attempts of the blocks after the first
      one of each attach, whose time includes the captures; ``None`` and 0 on the CPU) and ``counters``
      (``{move: {attempted, impossible, accepted, ...}}``);
    - ``cell_heights_A`` and ``cutoff_A``: the small-cell warning;
    - ``warnings``: extra warning lines (more through ``add_warning``);
    - ``warmup_steps``, ``window_steps``, ``start_step``: the speed window, steps counted from
      ``start_step`` (the step the run starts at, nonzero after a resume).
    """

    def __init__(self, *, model, head=None, sha256=None, structure, n_atoms, repeat=(1, 1, 1),
                 pbc=(True, True, True), species=(), plan=None, refusal=None, integrator=None, gate=None,
                 setup_s=None, outputs=(), dispersion=None, charge_spin=None, mc=None, cell_heights_A=None,
                 cutoff_A=None, warnings=(), warmup_steps=DEFAULT_WARMUP_STEPS, window_steps=DEFAULT_WINDOW_STEPS,
                 start_step=0, version=None):
        if (plan is None) == (refusal is None):
            raise ValueError("Report needs exactly one of plan= and refusal=")
        for name, value in (("warmup_steps", warmup_steps), ("window_steps", window_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "warmup_steps" else 1):
                raise ValueError(f"{name} must be an integer ≥ {0 if name == 'warmup_steps' else 1}, got {value!r}")
        if version is None:
            from fastmace import __version__ as version
        self.model, self.head, self.sha256 = model, head, sha256
        self.structure, self.n_atoms, self.repeat, self.pbc = structure, int(n_atoms), tuple(repeat), tuple(pbc)
        self.species = frozenset(species)
        self.plan, self.refusal = plan, refusal
        self.integrator = plan.integrator if plan is not None and plan.integrator is not None else integrator
        self.gate = _mapping(gate)
        self.setup_s = dict(setup_s or {})
        self.outputs = tuple(str(path) for path in outputs)
        self.dispersion = _mapping(dispersion)
        self.charge_spin, self.mc = charge_spin, mc
        self.cell_heights_A, self.cutoff_A = cell_heights_A, cutoff_A
        self.extra_warnings = list(warnings)
        self.start_step = int(start_step)
        self.window = (self.start_step + warmup_steps, self.start_step + warmup_steps + window_steps)
        self.version = version
        self.licence, self.licence_line = licence(model, sha256)
        self.speed = {"status": "pending"}
        self.measured = None
        self.runtime = None  # the runtime monitor's line (planner.adapt.report_line; 5-T06), or None

    # ------------------------------------------------------------------ the lines

    def _header(self):
        model = f"{self.model} (head {self.head})" if self.head else f"{self.model}"
        repeat = "×".join(str(n) for n in self.repeat)
        pbc = "3D PBC" if all(self.pbc) else "PBC " + "".join("T" if p else "F" for p in self.pbc)
        run = "integrator unknown"
        if self.integrator is not None:
            run = f"{_scheme_name(self.integrator)} {self.integrator.timestep_fs:g} fs"
        structure = f"{self.structure}, {self.n_atoms} atoms ({repeat}), {pbc}, {run}"
        return f"FastMACE {self.version}  |  {model}  |  {structure}"

    def _physics(self):
        if self.dispersion:
            d3 = self.dispersion
            parts = [f"D3({str(d3.get('damping', 'bj')).upper()}) functional {d3.get('functional')} "
                     f"(s6 {d3.get('s6')}, s8 {d3.get('s8')}, a1 {d3.get('a1')}, a2 {d3.get('a2')}, cutoff "
                     f"{d3.get('pair_cutoff_bohr')} bohr), non-stock (MACE plus D3)"]
        else:
            parts = ["no dispersion"]
        if self.charge_spin is None:
            parts.append("charge/spin not used by this model")
        else:
            charge, spin = self.charge_spin
            parts.append(f"total charge {charge}, spin multiplicity {spin}")
        return "; ".join(parts)

    def _plan(self):
        if self.plan is None:
            return "refused (reasons below; run.plan has none)"
        plan = self.plan
        backend = plan.engine.backend if plan.engine is not None else "?"
        if backend == "hybrid":
            backend = "hybrid cuEq/OEq"
        elif backend == "cueq":
            fused = plan.engine.cueq_conv_fusion
            backend = "plain cuEq" + {"on": " (fused convolution)", "off": " (unfused convolution)"}.get(fused, "")
        try:
            precision = plan.decision("force_precision")[1]
        except KeyError:
            precision = plan.precision.dtype if plan.precision is not None else "?"
        choices = {key: choice for key, choice, _ in plan.decisions}
        states = [choices.get(f"RewriteConfig.{name}") for name in REWRITE_FIELDS]
        on, off = states.count("on"), states.count("off")
        return f"{plan.tier}, {backend}, {precision}, {on} rewrites on, {off} off (reasons in run.plan)"

    def _gate(self):
        return gate_line(self.gate)

    def _mc(self):
        """The MC line: the moves, acceptance per move type and the measured cost per attempt (§7.7)."""
        mc = self.mc
        counters = mc.get("counters") or {}
        parts = []
        for move in mc.get("moves", ()):
            counts = counters.get(move.move, {})
            text = f"{describe_move(move)}; accepted {counts.get('accepted', 0)}/{counts.get('attempted', 0)}"
            parts.append(text)
        timed = int(mc.get("timed_attempts") or 0)
        blocks = int(mc.get("blocks") or 0)
        if mc.get("device_ms") is not None and timed:
            parts.append(f"{mc['device_ms'] / timed:.3g} ms/attempt over {_number(blocks)} blocks (device time; "
                         "each attach's first block, with its captures, left out)")
        else:
            parts.append(f"{_number(blocks)} blocks")
        ranges = mc.get("ranges") or ()
        if ranges:
            parts.append("attached over steps " + ", ".join(f"{_number(a)}–{_number(b)}" for a, b in ranges))
        return "; ".join(parts) if parts else "attached"

    def _setup(self):
        if not self.setup_s:
            return "not timed"
        stages = ", ".join(f"{stage} {_seconds(seconds)}" for stage, seconds in self.setup_s.items())
        return f"{_seconds(sum(self.setup_s.values()))} ({stages})"

    def _speed(self):
        if self.speed["status"] == "pending":
            first, last = self.window
            return f"pending (measured over steps {_number(first)}–{_number(last)} of this run)"
        speed = self.speed
        span = f"steps {_number(speed['first_step'])}–{_number(speed['last_step'])}"
        md_only = "; MD only; MC blocks: see mc" if self.mc else ""  # the blocks' cost is the mc line's
        if speed["status"] == "measured":
            return f"{_ms(speed['seconds_per_step'])}, measured over {span} of this run{md_only}"
        return f"{_ms(speed['seconds_per_step'])}, whole-run mean over {span}{md_only}"

    def _why(self):
        """Reasons the run is not on the plan of a complete installation, or was refused."""
        lines = []
        if self.refusal is not None:
            for reason in getattr(self.refusal, "reasons", (str(self.refusal),)):
                code = _code(reason)
                text = REFUSALS.get(code)
                lines.append(("refused", text if text is not None else f"{code}: {reason}"))
            return lines
        for _key, _choice, reason in self.plan.decisions:
            code = _code(reason)
            if code in DOWNGRADES:
                lines.append(("why", DOWNGRADES[code]))
        return lines

    def _experimental(self):
        lines = []
        npt = self.integrator is not None and self.integrator.ensemble == "npt"
        if npt:
            lines.append("NPT barostat: experimental: not yet qualified (§10.2)")
            if self.plan is not None and self.plan.polar is not None:
                lines.append("PolarMACE NPT stress: experimental: validated against finite differences on "
                             "NaCl 3×3×3 only (phase 3b)")
        record = None if self.integrator is None else _scheme_record(self.integrator)
        if record is not None and record.non_canonical:
            lines.append(f"{_scheme_name(self.integrator)}: non-canonical: equilibration only (§10.2)")
        if self.mc:
            lines.append("MC: experimental: not yet qualified (§10.4)")
        if self.dispersion:
            lines.append("D3: experimental: not yet qualified (§11.2)")
        return lines

    def _known_issues(self):
        label = _label_of(self.model, self.head, self.sha256)
        npt = self.integrator is not None and self.integrator.ensemble == "npt"
        lines = []
        for names, needed, condition, text in KNOWN_ISSUES:
            if label not in names or not self.species >= set(needed):
                continue
            if condition == "npt" and not npt:
                continue
            if condition == "large" and self.n_atoms < _LARGE_ATOMS:
                continue
            if condition == "inorganic" and self.species & _ORGANIC:
                continue
            lines.append(text)
        return lines

    def _small_cell(self):
        if self.cell_heights_A is None or self.cutoff_A is None:
            return []
        height = min(self.cell_heights_A)
        if height >= 2 * self.cutoff_A:
            return []
        return [f"small cell: height {height:.3g} Å < 2 × cutoff ({2 * self.cutoff_A:.3g} Å), atoms see their own "
                "periodic images; consider a larger repeat"]

    def warnings(self):
        """Every warning line of this run, in order: experimental, model issues, cell, extra."""
        return self._experimental() + self._known_issues() + self._small_cell() + list(self.extra_warnings)

    def add_warning(self, text):
        """Add a warning line (for example D4's observer note from ``fastmace.ase``)."""
        self.extra_warnings.append(str(text))

    def _lines(self):
        fields = [("", self._header()), ("physics", self._physics()), ("plan", self._plan())]
        fields += [(None, line) for line in revision_lines(self.plan)]
        fields += self._why()
        fields += [("gate", self._gate())]
        mc_energy = move_energy_line(self.gate)
        fields += [("mc ΔE", mc_energy)] if mc_energy else []
        fields += [("mc", self._mc())] if self.mc else []
        fields += [("setup", self._setup()), ("speed", self._speed())]
        fields += [("runtime", self.runtime)] if self.runtime else []  # re-captures or overflows (5-T06)
        fields += [("output", ", ".join(self.outputs) or "none"), ("licence", self.licence_line)]
        warnings = self.warnings() or ["none"]
        fields += [("warn", warnings[0])] + [(None, line) for line in warnings[1:]]
        out = []
        for name, text in fields:
            if name == "":
                out.append(text)
            elif name is None:
                out.append(" " * INDENT + text)
            else:
                out.append(f"{name + ':':<{INDENT}}{text}")
        return out

    def setup_text(self):
        """The report as printed at setup (``speed: pending`` until ``record_speed``)."""
        return self.text()

    def text(self):
        """The report as it stands now: the setup lines with the latest speed line."""
        return "\n".join(self._lines())

    def __str__(self):
        return self.text()

    # ------------------------------------------------------------------ speed

    def record_speed(self, seconds_per_step, first_step, last_step):
        """The steady-state step time over steps ``first_step``–``last_step``; returns the one line to print."""
        seconds = _positive(seconds_per_step, "seconds_per_step")
        self.measured = {"seconds_per_step": seconds, "first_step": int(first_step), "last_step": int(last_step)}
        self.speed = {"status": "measured", **self.measured}
        return f"{'speed:':<{INDENT}}{self._speed()}"

    def finish(self, mean, *, steps):
        """End of the run: the whole-run mean (seconds per step over ``steps`` steps) replaces the speed line."""
        seconds = _positive(mean, "mean")
        first = self.start_step + 1
        self.speed = {"status": "whole-run mean", "seconds_per_step": seconds, "first_step": first,
                      "last_step": self.start_step + int(steps)}
        return f"{'speed:':<{INDENT}}{self._speed()}"

    # ------------------------------------------------------------------ metadata

    def to_metadata(self):
        """The ``"report"`` block of ``metadata.json``: JSON-safe, written by ``provenance.write_metadata``."""
        why = self._why()
        return {
            "text": self.text(),
            "speed": dict(self.speed),
            "measured_window": None if self.measured is None else dict(self.measured),
            "speed_window": {"warmup_steps": self.window[0] - self.start_step,
                             "window_steps": self.window[1] - self.window[0], "start_step": self.start_step},
            "setup_s": {stage: float(seconds) for stage, seconds in self.setup_s.items()},
            "setup_total_s": float(sum(self.setup_s.values())),
            "licence": {"licence": self.licence, "source": "label table" if self.licence else None},
            "gate_status": "skipped" if self.gate is None else self.gate.get("status", "skipped"),
            "move_energy_status": None if not (self.gate or {}).get("move_energy") else
            self.gate["move_energy"].get("status"),
            "mc_energy_precision": None if self.gate is None else self.gate.get("mc_energy_precision"),
            "reasons": [{"kind": kind, "text": text} for kind, text in why],
            "warnings": self.warnings(),
            "mc": None if not self.mc else {
                "moves": [describe_move(move) for move in self.mc.get("moves", ())],
                "ranges": [list(pair) for pair in self.mc.get("ranges", ())],
                "blocks": int(self.mc.get("blocks") or 0), "device_ms": self.mc.get("device_ms"),
                "timed_attempts": int(self.mc.get("timed_attempts") or 0)},
        }
