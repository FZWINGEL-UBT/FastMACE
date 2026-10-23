"""Rule 15 of SPEC §9.3: known model-structure warnings (§4.3), as data (plan task 5-T04, conductor's choice 6).

The planner adds one decision ``warn.model_structure.<id>`` per matching issue, with rule 15's code; the run
report reads them from the plan (``fastmace.report``). A known issue warns, it never refuses.

This module imports neither torch nor MACE.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastmace.model import labels as label_table

# The code of every rule 15 decision (planner/rules.py).
MODEL_STRUCTURE_CODE = "model-structure-warning"
# The decision key of one issue is this prefix and the issue's id.
DECISION_PREFIX = "warn.model_structure."

# SPEC §4.3 model-level known issues, as data (conductor answer for 1b; moved from fastmace.report by 5-T04):
# labels, the species the structure must contain (all of them), an extra condition ("" for none), and the warning.
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
# The stable id of each row of KNOWN_ISSUES, in the same order (the decision key's suffix).
ISSUE_IDS = ("polar1-s-nanbo3", "polar1-natao3", "omol0-nacl", "omol0-oxides", "off23-volume", "off23-large-memory")
_LARGE_ATOMS = 2000
_ORGANIC = frozenset(("H", "C"))


@dataclass(frozen=True)
class Issue:
    """One matching row of ``KNOWN_ISSUES``: its ``id`` (``ISSUE_IDS``) and its warning ``text``."""

    id: str
    text: str

    @property
    def key(self) -> str:
        """The plan decision's key."""
        return DECISION_PREFIX + self.id


def label_of(model, head, sha256):
    """The label-table label for ``model`` (a label, or a checkpoint found by SHA-256 and head), or None."""
    if model is not None and str(model) in label_table.LABELS:
        return str(model)
    for name, entry in label_table.LABELS.items():
        if sha256 and entry.checkpoint.sha256 == sha256 and entry.head in (head, None):
            return name
    return None


def matching(label, species, *, variable_cell: bool, n_atoms: int) -> tuple[Issue, ...]:
    """The known issues of model ``label`` on a structure of ``species`` (symbols), in ``KNOWN_ISSUES`` order.

    ``variable_cell``: an NPT integrator or a cell-filtered optimiser (the "npt" condition); ``n_atoms``: the
    structure's size (the "large" condition). A ``label`` outside the table (``None``) has no known issue.
    """
    species = frozenset(species)
    out = []
    for issue_id, (names, needed, condition, text) in zip(ISSUE_IDS, KNOWN_ISSUES):
        if label not in names or not species >= set(needed):
            continue
        if condition == "npt" and not variable_cell:
            continue
        if condition == "large" and n_atoms < _LARGE_ATOMS:
            continue
        if condition == "inorganic" and species & _ORGANIC:
            continue
        out.append(Issue(issue_id, text))
    return tuple(out)


def decide(book, label, species, *, variable_cell: bool, n_atoms: int) -> None:
    """Rule 15: one ``warn`` decision per matching issue into the planner's ``book``; never a refusal."""
    from fastmace.errors import Reason

    for issue in matching(label, species, variable_cell=variable_cell, n_atoms=n_atoms):
        book.decide(issue.key, "warn", Reason(issue.text, rule=15, code=MODEL_STRUCTURE_CODE))


def decisions(plan) -> tuple[tuple[str, str, object], ...]:
    """The rule 15 decisions of ``plan`` (``(key, choice, reason)``), in the plan's order."""
    return tuple(entry for entry in getattr(plan, "decisions", ()) if entry[0].startswith(DECISION_PREFIX))


__all__ = ["DECISION_PREFIX", "ISSUE_IDS", "Issue", "KNOWN_ISSUES", "MODEL_STRUCTURE_CODE", "decide", "decisions",
           "label_of", "matching"]
