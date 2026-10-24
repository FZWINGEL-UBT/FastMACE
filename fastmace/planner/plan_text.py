"""The human-readable plan, ``str(plan)``, by sections (SPEC §9.5; plan task 5-T05).

The layout is docs/design/planner-5.md's ("``str(plan)`` layout"): header, hard rules, rewrites, sizes,
kernels, memory (rule 12), workload (rules 8-11), warnings (rule 15) and history, at most ``MAX_LINES``
lines of at most ``WIDTH`` characters. A section with nothing to say is one line ("none") or, for the hard
rules and the warnings, left out. ``Plan.describe(full=True)`` keeps the list of every decision; provenance
stores both texts.

The text is read from the plan's resolved configs and its decisions, never from a table of plans: a
decision's rule is its reason's ``rule`` or, for a code the rule table knows, ``rules.rule_of(code)``. A
section's lines are capped, and what does not fit is counted ("+n more"), so a long reason or many rewrites
never push the plan past its length. The memory line (5-T03), the calibration line (5-T09) and the
revision chain's switches (5-T07) read fields those tasks fill; until then they say "not estimated" and
"none".

This module holds no state and imports nothing from torch or MACE.
"""

from __future__ import annotations

import re
from typing import Any

# The longest line and the most lines of the default text (docs/design/planner-5.md).
WIDTH = 120
MAX_LINES = 30
INDENT = "  "
# Wrapped continuation lines of a section.
CONTINUATION = INDENT * 3
# Hard-rule codes that record a rule did not apply, so they did not shape the plan.
_NOT_APPLIED = frozenset({"no-allow-reference", "fused-cueq"})
# Spec citations, left to the full text.
_SPEC = re.compile(r"\s*\((?:SPEC|see SPEC)[^()]*\)")
_TRAILING_PARENTHESIS = re.compile(r"\s*\([^()]*\)$")


def text(plan: Any) -> str:
    """The sectioned text of ``plan`` (``Plan.describe()``)."""
    lines = [*_header(plan), *_hard_rules(plan), *_rewrites(plan), *_sizes(plan), _kernels(plan), _memory(plan),
             *_workload(plan), *_warnings(plan), *_history(plan)]
    return "\n".join(_fit(line) for line in lines[:MAX_LINES])


# --------------------------------------------------------------------------- helpers

def _fit(line: str, width: int = WIDTH) -> str:
    return line if len(line) <= width else line[:width - 1].rstrip() + "…"


def _short(reason: Any) -> str:
    """A reason without its spec citations."""
    return _SPEC.sub("", str(reason)).strip()


def _rule(reason: Any) -> int | None:
    from fastmace.planner import rules

    number = getattr(reason, "rule", None)
    if number is not None:
        return number
    code = getattr(reason, "code", None)
    return None if code is None else rules.rule_of(code)


def _decision(plan: Any, key: str):
    for entry in plan.decisions:
        if entry[0] == key:
            return entry
    return None


def _name(key: str) -> str:
    """A decision key without its config prefix (``EngineConfig.backend`` -> ``backend``)."""
    return key.rsplit(".", 1)[-1]


def _number(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _tokens(groups: list[list[str]], inner: str = ",", outer: str = ";") -> list[str]:
    """The items of ``groups`` as words to wrap: ``inner`` after each but a group's last, ``outer`` between groups."""
    out = []
    for index, group in enumerate(groups):
        last = index == len(groups) - 1
        out += [item + ("" if last and position == len(group) - 1 else outer if position == len(group) - 1
                        else inner) for position, item in enumerate(group)]
    return out


def _wrap(head: str, words: list[str], *, max_lines: int = 2) -> list[str]:
    """``head`` and ``words`` (punctuated by ``_tokens``) over at most ``max_lines`` lines, breaking between
    words; the words that do not fit are counted."""
    def joined(prefix, line):
        return prefix + " ".join(line)

    lines: list[list[str]] = [[]]
    for word in words:
        prefix = INDENT + head if len(lines) == 1 else CONTINUATION
        if lines[-1] and len(joined(prefix, [*lines[-1], word])) > WIDTH:
            lines.append([])
        lines[-1].append(word)
    prefixes = [INDENT + head] + [CONTINUATION] * (len(lines) - 1)
    if len(lines) <= max_lines:
        return [joined(prefix, line) for prefix, line in zip(prefixes, lines)]
    last, hidden = list(lines[max_lines - 1]), sum(len(line) for line in lines[max_lines:])
    while len(last) > 1 and len(joined(prefixes[max_lines - 1], last)) + len(_more(hidden)) > WIDTH:
        last, hidden = last[:-1], hidden + 1
    tail = joined(prefixes[max_lines - 1], last).rstrip(",;") + _more(hidden)
    return [*(joined(prefix, line) for prefix, line in zip(prefixes, lines[:max_lines - 1])), tail]


def _more(count: int) -> str:
    return f" (+{count} more; describe(full=True) lists them)"


def _capped(lines: list[str], max_lines: int) -> list[str]:
    if len(lines) <= max_lines:
        return lines
    tail = _more(len(lines) - max_lines + 1)
    return [*lines[:max_lines - 1], _fit(lines[max_lines - 1], WIDTH - len(tail)) + tail]


# --------------------------------------------------------------------------- 1. header

def _header(plan: Any) -> list[str]:
    entry = _decision(plan, "mode") or _decision(plan, "tier")
    if entry is None:
        why = _short(plan.reason)
    else:
        number = _rule(entry[2])
        why = (f"rule {number}: " if number is not None else "") + _short(entry[2])
    first = f"Plan revision {plan.revision}, tier {plan.tier} ({why})"
    if plan.revision and plan.step is not None:
        suffix = f", from step {plan.step}"
        first = _fit(first, WIDTH - len(suffix)) + suffix
    parts = []
    if plan.engine is not None:
        backend = plan.engine.backend
        if backend == "cueq":
            backend += f" (fused convolution {plan.engine.cueq_conv_fusion})"
        parts.append(f"backend {backend}")
    if plan.precision is not None:
        parts.append(f"force {plan.precision.dtype}" + (", TF32" if plan.precision.tf32 else ""))
        moves = () if plan.workload is None else plan.workload.mc_moves
        if moves:
            mc = _decision(plan, "PrecisionConfig.mc_energy")
            pending = mc is not None and str(getattr(mc[2], "code", "")).endswith("pending")
            parts.append(f"MC energy {plan.precision.mc_energy}" + (" (pending the move-energy gate)"
                                                                     if pending else ""))
        else:
            parts.append("MC energy none (no MC)")
    parts.append(f"runner {plan.runner}")
    return [first, INDENT + " | ".join(parts)]


# --------------------------------------------------------------------------- 2. hard rules

def _hard_rules(plan: Any) -> list[str]:
    from fastmace.planner import rules

    hard = set(rules.numbers("hard"))
    shaped: dict[int, list[str]] = {}
    for key, choice, reason in plan.decisions:
        number = _rule(reason)
        if number in hard and getattr(reason, "code", None) not in _NOT_APPLIED:
            shaped.setdefault(number, []).append(f"{_name(key)} {choice}")
    groups = [[f"rule {number}: {shaped[number][0]}", *shaped[number][1:]] for number in sorted(shaped)]
    if plan.allowances:
        groups.append([f"allowed: {plan.allowances[0]}", *plan.allowances[1:]])
    return _wrap("hard rules: ", _tokens(groups), max_lines=3) if groups else []


# --------------------------------------------------------------------------- 3. rewrites

def _off_label(reason: Any) -> str:
    """Why a rewrite is off, short: its code, else the last clause of its reason."""
    code = getattr(reason, "code", None)
    if code:
        return code
    clause = _short(reason).split("; ")[-1].split(": ")[-1]
    clause = _TRAILING_PARENTHESIS.sub("", clause)
    return clause.removeprefix("off, ").removeprefix("off: ") or "off"


def _rewrites(plan: Any) -> list[str]:
    entries = [(_name(key), choice, reason) for key, choice, reason in plan.decisions
               if key.startswith("RewriteConfig.") and choice in ("on", "off")]
    if not entries:
        return [INDENT + "rewrites: none"]
    on = [name for name, choice, _ in entries if choice == "on"]
    groups: dict[str, list[str]] = {}
    for name, choice, reason in entries:
        if choice == "off":
            groups.setdefault(_off_label(reason), []).append(name)
    lines = _wrap(f"rewrites on ({len(on)}): ", _tokens([on or ["none"]]), max_lines=4)
    if groups:
        lines += _wrap("rewrites off: ", _tokens([[*names[:-1], f"{names[-1]} ({label})"]
                                                   for label, names in groups.items()]), max_lines=3)
    return lines


# --------------------------------------------------------------------------- 4. sizes

def _sizes(plan: Any) -> list[str]:
    neighbours, engine = plan.neighbours, plan.engine
    first = []
    if neighbours is not None:
        first += [f"skin {_number(neighbours.skin)} A",
                  f"headroom {_number(neighbours.capacity_headroom)} ({neighbours.headroom})",
                  f"edge bucket +{_number(neighbours.edge_bucket_fraction * 100)} %, multiple "
                  f"{neighbours.edge_multiple}"]
    if engine is not None:
        capture = _decision(plan, "capture")
        if capture is not None and capture[1] == "off":
            first.append("no CUDA graphs")
        else:
            first.append(f"re-capture budget {engine.recapture_budget}, variant cap {engine.graph_variant_cap}")
    second = []
    d3 = _decision(plan, "NeighbourConfig.d3_skin")
    if plan.dispersion is not None or (d3 is not None and d3[1] != "unused"):
        second.append(f"D3 skin {_number(neighbours.d3_skin)} A" if neighbours is not None else "D3 list")
    else:
        second.append("D3: no list")
    if plan.polar is not None:
        second.append(f"k-space headroom {_number(plan.polar.kspace_headroom)}")
    else:
        second.append("k-space headroom n/a")
    return [INDENT + "sizes: " + (", ".join(first) or "none"), INDENT + "sizes: " + "; ".join(second)]


# --------------------------------------------------------------------------- 5. kernels

def _kernels(plan: Any) -> str:
    software = plan.software
    if software is None:
        return INDENT + "kernels: not recorded"
    names = (("OEq", software.oeq), ("Triton", software.triton), ("NVRTC", software.nvrtc),
             ("fused cuEq", software.fused_cueq))
    return INDENT + "kernels: " + ", ".join(f"{name} {'yes' if trait[0] else 'no'}" for name, trait in names)


# --------------------------------------------------------------------------- 6. memory (rule 12)

_SPLIT = re.compile(r"\((?:build|run); run [0-9.]+, build [0-9.]+\)")
_FREE = re.compile(r"of ([0-9.]+) GiB free")


def _memory(plan: Any) -> str:
    """Rule 12's line: the prediction, which of build and run it is, the decision against the free memory and
    the rewrites the decision turned off (5-T03)."""
    predicted = plan.predicted_memory_bytes
    if predicted is None:
        parts = ["not estimated"]
    else:
        head = f"{predicted / 2**30:.2f} GiB predicted"
        decision = _decision(plan, "memory")
        if decision is not None and decision[1] != "not estimated":
            reason = str(decision[2])
            split, free = _SPLIT.search(reason), _FREE.search(reason)
            head += (f" {split.group(0)}" if split else "") + f", {decision[1]}"
            if free is not None:
                head += f" ({free.group(1)} GiB free)"
        parts = [head]
        dropped = [_name(key) for key, _, reason in plan.decisions
                   if getattr(reason, "code", None) == "rule-12-memory-drop"]
        if dropped:  # the first one turned off and its dependants
            parts.append("turned off " + ", ".join(dropped[:2]) + (f" +{len(dropped) - 2} more"
                                                                    if len(dropped) > 2 else ""))
    hardware = _decision(plan, "hardware")
    if hardware is not None:
        parts.append(f"GPU {hardware[1]}")
    d3 = _decision(plan, "d3 list memory")
    if d3 is not None:
        parts.append(f"D3 list {d3[1]} ({_short(d3[2])})")
    return INDENT + "memory (rule 12): " + "; ".join(parts)


# --------------------------------------------------------------------------- 7. workload (rules 8-11)

def _workload(plan: Any) -> list[str]:
    from fastmace.planner import rules

    parts = []
    if plan.optimizer is not None:
        parts.append(f"relaxation {plan.optimizer.scheme}"
                     + (f" with the {plan.optimizer.cell_filter} cell filter" if plan.optimizer.cell_filter else ""))
    else:
        integrator = _decision(plan, "integrator")
        if integrator is not None:
            parts.append(f"{integrator[1]} {_TRAILING_PARENTHESIS.sub('', _short(integrator[2]))}")
    structure = _decision(plan, "workload")
    if structure is not None:
        parts.append(_short(structure[2]))
    workload = plan.workload
    if workload is not None:
        parts.append(f"up to {_number(workload.max_temperature_K)} K")
        if workload.mc_moves:
            parts.append(f"MC {', '.join(workload.mc_moves)}" + (
                "" if workload.min_mc_temperature_K is None else
                f" from {_number(workload.min_mc_temperature_K)} K"))
        if workload.species_changes:
            parts.append("species changes")
        if workload.host_callbacks_every is not None:
            parts.append(f"host callbacks every {workload.host_callbacks_every} steps")
        if workload.reproducible:
            parts.append("reproducible")
    lines = _wrap("workload: ", _tokens([[part] for part in parts or ["none (an evaluation)"]]), max_lines=2)
    if plan.dispersion is not None:
        from fastmace.longrange import d3

        lines.append(f"{INDENT}dispersion: {d3.describe(plan.dispersion)}, non-stock (SPEC §12.1)")
    preferences = set(rules.numbers("preference")) - {12}
    items = []
    for key, choice, reason in plan.decisions:
        number = _rule(reason)
        if number not in preferences:
            continue
        entry = rules.overridden(reason)
        if entry is not None:
            items.append(f"{_name(key)} {choice} (override; rule {entry[0]} would give {entry[1]})")
        else:
            items.append(f"{_name(key)} {choice} ({number})")
    if items:
        lines += _wrap("rules 8-11: ", _tokens([items]), max_lines=2)
    return lines


# --------------------------------------------------------------------------- 8. warnings (rule 15)

def _warnings(plan: Any) -> list[str]:
    lines = [f"{INDENT}warning (rule 15): {_short(reason)}" for key, choice, reason in plan.decisions
             if _rule(reason) == 15]
    return _capped(lines, 3)


# --------------------------------------------------------------------------- 9. history

def _history(plan: Any) -> list[str]:
    if plan.fallbacks:
        items = []
        for tried in plan.fallbacks:
            outcome = (f"activated as revision {tried.revision}" if tried.outcome == "activated"
                       else tried.outcome + (f" at {tried.stage}" if tried.stage else ""))
            items.append(f"{tried.change}: {outcome}")
        lines = _wrap(f"fallbacks tried ({len(items)}): ", _tokens([[item] for item in items]), max_lines=2)
    else:
        lines = [INDENT + "fallbacks: none"]
    lines.append(INDENT + "calibration: " + _calibration(plan.calibration))
    if plan.selfchecks:
        failed = [check.name for check in plan.selfchecks if check.passed is False]
        lines.append(f"{INDENT}self-checks: {len(plan.selfchecks)} records"
                     + (f", failed: {', '.join(failed)}" if failed else ", none failed"))
    chain = (*plan.chain(), plan)
    items = [f"{entry.revision} " + ("" if entry.step is None else f"at step {entry.step}: ") + _short(entry.reason)
             for entry in chain]
    lines += _wrap("revisions: ", _tokens([[item] for item in items]), max_lines=2)
    return lines


def _calibration(record: Any) -> str:
    if record is None:
        return "none"

    def name(change):
        return change or "rules' plan"

    def mean(values):
        return sum(values) / len(values) if values else float("nan")

    times = ", ".join(f"{name(c.change)} {mean(c.step_ms):.2f} ms" for c in record.candidates)
    chosen = f"chose {name(record.chosen)}" if record.changed_plan else "kept the rules' plan"
    margin = "" if record.margin is None else f", margin {_number(record.margin * 100)} %"
    return f"{chosen}; {times} ({record.steps} steps{margin}, setup {_number(round(record.setup_s, 1))} s)"


__all__ = ("CONTINUATION", "INDENT", "MAX_LINES", "WIDTH", "text")
