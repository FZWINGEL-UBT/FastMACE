"""Planner reactions to checks (SPEC §9.3 rules 13-15; plan task 2-T04: rule 13).

Rule 13: a rewrite whose self-check fails is disabled together with every rewrite that requires
it, the reasons are recorded, and the run is re-planned. ``Registry.apply`` raises
``SelfCheckError`` carrying the registry's ``Reaction``; ``selfcheck_revision`` turns it into the
next plan revision (SPEC §9.5), which the planner builds again from the pristine model (decision
24), never by undoing rewrites in place. The fallback (2-T15) calls it under
``fallback="auto"``; with ``fallback="refuse"`` the error propagates with the record.

A record is switched through its ``RewriteConfig`` field, so a field that several records share
(the MH-1 and ``GENERIC_*`` variants) goes off as a whole; the variants that do not fit the model
were off anyway.
"""

from __future__ import annotations

import dataclasses

from fastmace.errors import Reason

FAILED_CODE = "selfcheck-failed"
DEPENDANT_CODE = "selfcheck-dependant"


def _reaction(failure):
    reaction = getattr(failure, "reaction", failure)
    if not all(hasattr(reaction, name) for name in ("rewrite", "disabled", "fields", "reason")):
        raise TypeError(f"expected a SelfCheckError from Registry.apply or a Reaction, got {type(failure).__name__}")
    return reaction


def selfcheck_revision(plan, failure, *, step=0):
    """The plan revision of rule 13 for ``failure`` (a ``SelfCheckError`` or a registry ``Reaction``).

    The failed record's field and those of the records requiring it are "off" (``False`` for a
    bool field) in ``plan.rewrites``, their decisions say why (codes ``FAILED_CODE`` and
    ``DEPENDANT_CODE``, rule 13), and the revision's reason names them. ``step`` is 0: phase 2
    reacts before the first step.
    """
    reaction = _reaction(failure)
    if plan.rewrites is None:
        raise ValueError("the plan has no rewrites to react on")
    failed_field = reaction.fields[0]
    changes = {}
    for field in reaction.fields:
        changes[field] = False if isinstance(getattr(plan.rewrites, field), bool) else "off"
    rewrites = dataclasses.replace(plan.rewrites, **changes)
    reasons = {}
    for field in reaction.fields:
        if field == failed_field:
            reasons[field] = Reason(f"self-check of {reaction.rewrite} failed: {reaction.reason} (SPEC §9.3 rule 13)",
                                    rule=13, code=FAILED_CODE)
        else:
            reasons[field] = Reason(f"requires {reaction.rewrite}, whose self-check failed (SPEC §9.3 rule 13)",
                                    rule=13, code=DEPENDANT_CODE)
    decisions = []
    for key, choice, why in plan.decisions:
        field = key.split(".", 1)[1] if key.startswith("RewriteConfig.") else None
        if field in reasons:
            decisions.append((key, "off", reasons.pop(field)))
        else:
            decisions.append((key, choice, why))
    decisions += [(f"RewriteConfig.{field}", "off", why) for field, why in reasons.items()]
    text = (f"rule 13: rewrite {reaction.rewrite} failed its self-check ({reaction.reason}); "
            f"{', '.join(reaction.disabled)} off")
    return plan.revise(text, step, rewrites=rewrites, decisions=tuple(decisions))


__all__ = ["DEPENDANT_CODE", "FAILED_CODE", "selfcheck_revision"]
