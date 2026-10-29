"""Layering checks (SPEC §5, AGENTS.md "Code rules").

A static walk over the AST of every module in ``fastmace/``; nothing is imported, so
the test needs neither torch nor a GPU. The rules are tables (``IMPORT_RULES``,
``FORBIDDEN_IMPORT_ROOTS``, ``FORBIDDEN_SOURCE_STRINGS``, ``SWITCH_ALLOW_LIST``) so
later phases extend them by adding rows. Each rule has a negative fixture below: a
temporary package that breaks it, which the checker must reject.
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "fastmace"


@dataclass(frozen=True)
class ImportRule:
    """Modules under ``scope`` must not import ``forbidden``, except ``allowed``.

    Names are dotted module prefixes: ``fastmace.engine`` covers the package and
    everything below it. An import ``from M import a`` counts as importing ``M.a``,
    so ``from fastmace import engine`` is caught as well as ``import fastmace.engine``.
    """

    name: str
    scope: tuple[str, ...]
    forbidden: tuple[str, ...]
    allowed: tuple[str, ...] = ()


IMPORT_RULES = (
    ImportRule(
        name="model and rewrites never import engine",
        scope=("fastmace.model", "fastmace.rewrites"),
        forbidden=("fastmace.engine",),
    ),
    ImportRule(
        name="integrators see only SimulationState and a ForceField",
        scope=("fastmace.integrators",),
        forbidden=("fastmace.engine", "fastmace.model"),
        # The ForceField interface (SPEC §5); everything else in fastmace.model is internal.
        allowed=("fastmace.model.forces",),
    ),
)

# Absolute imports whose first component is one of these are the research engine's
# modules (AGENTS.md: no imports from Experiments/, langevin, common.py, model_store.py).
FORBIDDEN_IMPORT_ROOTS = ("langevin", "common", "model_store", "Experiments")

# Substrings that must not appear anywhere in a module's source (NAS paths).
FORBIDDEN_SOURCE_STRINGS = ("/mnt/homes",)

# Modules allowed to bind module-level UPPER_CASE names to bool or None.
SWITCH_EXEMPT_FILES = ("fastmace/config.py",)

# Module-level UPPER_CASE names bound to bool or None outside config.py. Every entry is
# a constant that nothing changes at run time, with the reason it is not a config field:
# {"fastmace/<file>.py": {"NAME": "reason"}}.
SWITCH_ALLOW_LIST: dict[str, dict[str, str]] = {
    # fp32 profile screens (28-29 October 2026): A/B switches read by the screen tools, off by default; neither
    # prototype was adopted, so neither is a config field (conductor, batch-2 trial merge).
    "fastmace/neighbors/skin.py": {"ASYNC_EARLY_BUILD": "async early rebuild screen; no gain on sm_120 or sm_89"},
    "fastmace/rewrites/species/__init__.py": {"TAIL_HEAD": "readout head-slice prototype; mixed A/B, not adopted"},
}

UPPER_CASE = re.compile(r"^_*[A-Z][A-Z0-9_]*$")


# --------------------------------------------------------------------------- scanner


@dataclass(frozen=True)
class Module:
    path: str  # repository-relative, forward slashes
    name: str  # dotted module name
    package: str  # package that relative imports resolve against
    tree: ast.Module
    source: str


def load_modules(root: Path, package: str = PACKAGE) -> list[Module]:
    modules = []
    for file in sorted((root / package).rglob("*.py")):
        rel = file.relative_to(root)
        parts = list(rel.with_suffix("").parts)
        is_init = parts[-1] == "__init__"
        if is_init:
            parts.pop()
        name = ".".join(parts)
        source = file.read_text(encoding="utf-8")
        modules.append(
            Module(
                path=rel.as_posix(),
                name=name,
                package=name if is_init else name.rpartition(".")[0],
                tree=ast.parse(source, filename=str(file)),
                source=source,
            )
        )
    return modules


def imported_targets(module: Module) -> list[tuple[int, str, bool]]:
    """(line, dotted target, is_absolute) for every import in the module, at any depth."""
    targets = []
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Import):
            targets.extend((node.lineno, alias.name, True) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = module.package.split(".")
                if node.level - 1 > len(base) - 1:
                    raise AssertionError(
                        f"{module.path}:{node.lineno}: relative import beyond the top-level package"
                    )
                base = base[: len(base) - (node.level - 1)]
                if node.module:
                    base.append(node.module)
                origin = ".".join(base)
            else:
                origin = node.module
            for alias in node.names:
                target = origin if alias.name == "*" else f"{origin}.{alias.name}"
                targets.append((node.lineno, target, not node.level))
    return targets


def _under(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def check_import_rules(modules: list[Module], rules=IMPORT_RULES) -> list[str]:
    violations = []
    for module in modules:
        for rule in rules:
            if not _under(module.name, rule.scope):
                continue
            for line, target, _ in imported_targets(module):
                if _under(target, rule.forbidden) and not _under(target, rule.allowed):
                    violations.append(f"{module.path}:{line}: imports {target} ({rule.name})")
    return violations


def check_research_engine_imports(
    modules: list[Module], roots=FORBIDDEN_IMPORT_ROOTS, strings=FORBIDDEN_SOURCE_STRINGS
) -> list[str]:
    violations = []
    for module in modules:
        for line, target, absolute in imported_targets(module):
            if absolute and target.split(".")[0] in roots:
                violations.append(f"{module.path}:{line}: imports research-engine module {target}")
        for node in ast.walk(module.tree):
            # importlib.import_module("langevin...") and __import__("langevin...")
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", getattr(node.func, "id", None))
                in ("import_module", "__import__")
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and node.args[0].value.split(".")[0] in roots
            ):
                violations.append(
                    f"{module.path}:{node.lineno}: dynamically imports {node.args[0].value}"
                )
        for lineno, text in enumerate(module.source.splitlines(), start=1):
            for s in strings:
                if s in text:
                    violations.append(f"{module.path}:{lineno}: contains {s!r}")
    return violations


def check_no_global_statement(modules: list[Module]) -> list[str]:
    return [
        f"{module.path}:{node.lineno}: global {', '.join(node.names)}"
        for module in modules
        for node in ast.walk(module.tree)
        if isinstance(node, ast.Global)
    ]


def _module_level_statements(body):
    """Statements executed at import time: the module body, descending into if/try/with/for
    blocks but not into functions or classes."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody"):
            yield from _module_level_statements(getattr(stmt, field, []))
        for handler in getattr(stmt, "handlers", []):
            yield from _module_level_statements(handler.body)
        for case in getattr(stmt, "cases", []):  # match statements
            yield from _module_level_statements(case.body)


def _target_names(target):
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _target_names(element)


def _is_bool_or_none(value) -> bool:
    if isinstance(value, ast.Constant):
        return value.value is None or isinstance(value.value, bool)
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not):
        return True  # `FOO = not BAR` is a bool switch too
    if isinstance(value, (ast.Tuple, ast.List)):
        return any(_is_bool_or_none(element) for element in value.elts)
    return False


def module_level_switches(module: Module) -> list[tuple[int, str]]:
    found = []
    for stmt in _module_level_statements(module.tree.body):
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        else:
            continue
        if not _is_bool_or_none(value):
            continue
        for target in targets:
            found.extend(
                (stmt.lineno, name) for name in _target_names(target) if UPPER_CASE.match(name)
            )
    return found


def check_no_module_switches(
    modules: list[Module], exempt=SWITCH_EXEMPT_FILES, allow=SWITCH_ALLOW_LIST
) -> list[str]:
    violations = []
    for module in modules:
        if module.path in exempt:
            continue
        allowed = allow.get(module.path, {})
        for line, name in module_level_switches(module):
            if name not in allowed:
                violations.append(
                    f"{module.path}:{line}: module-level switch {name} bound to bool/None"
                    " (put it in a fastmace.config dataclass)"
                )
    return violations


CHECKS = {
    "import_rules": check_import_rules,
    "research_engine_imports": check_research_engine_imports,
    "no_global_statement": check_no_global_statement,
    "no_module_switches": check_no_module_switches,
}


# --------------------------------------------------------------------------- the package


@pytest.fixture(scope="module")
def package_modules():
    modules = load_modules(REPO_ROOT)
    assert any(m.name == "fastmace" for m in modules), "fastmace/__init__.py not found"
    return modules


@pytest.mark.parametrize("check", CHECKS)
def test_fastmace_obeys(check, package_modules):
    violations = CHECKS[check](package_modules)
    assert not violations, "\n".join(violations)


def test_switch_allow_list_is_current(package_modules):
    """Every allow-list entry names an existing module-level bool/None binding and a reason."""
    by_path = {m.path: m for m in package_modules}
    for path, names in SWITCH_ALLOW_LIST.items():
        assert path in by_path, f"allow-list names missing file {path}"
        bound = {name for _, name in module_level_switches(by_path[path])}
        for name, reason in names.items():
            assert name in bound, f"allow-list entry {path}:{name} is stale"
            assert reason.strip(), f"allow-list entry {path}:{name} has no reason"


# --------------------------------------------------------------------------- negative fixtures


def make_package(tmp_path: Path, files: dict[str, str]) -> list[Module]:
    """A temporary ``fastmace`` package with the given files (paths relative to it)."""
    for rel in ("__init__.py", *files):
        path = tmp_path / PACKAGE / rel
        for parent in reversed(path.relative_to(tmp_path).parents[:-1]):
            init = tmp_path / parent / "__init__.py"
            init.parent.mkdir(parents=True, exist_ok=True)
            init.touch()
    for rel, source in files.items():
        (tmp_path / PACKAGE / rel).write_text(textwrap.dedent(source), encoding="utf-8")
    return load_modules(tmp_path)


# (fixture id, check, files of the broken package, substring every violation must contain)
NEGATIVE_FIXTURES = [
    ("model_imports_engine_absolute", "import_rules",
     {"model/forces.py": "import fastmace.engine.graph\n"}, "fastmace.engine.graph"),
    ("model_imports_engine_from_root", "import_rules",
     {"model/forces.py": "from fastmace import engine\n"}, "fastmace.engine"),
    ("model_imports_engine_relative", "import_rules",
     {"model/forces.py": "from ..engine import graph\n"}, "fastmace.engine.graph"),
    ("model_init_imports_engine_relative", "import_rules",
     {"model/__init__.py": "from ..engine.graph import GraphEngine\n"}, "fastmace.engine.graph"),
    ("rewrites_imports_engine_in_function", "import_rules",
     {"rewrites/registry.py": "def f():\n    from fastmace.engine import eager\n"},
     "fastmace.engine.eager"),
    ("integrators_import_engine", "import_rules",
     {"integrators/nve.py": "from ..engine.graph import GraphEngine\n"}, "fastmace.engine"),
    ("integrators_import_model_internals", "import_rules",
     {"integrators/nve.py": "from fastmace.model.loading import load\n"}, "fastmace.model.loading"),
    ("integrators_import_model_package_name", "import_rules",
     {"integrators/nve.py": "from ..model import loading\n"}, "fastmace.model.loading"),
    ("imports_langevin", "research_engine_imports",
     {"engine/graph.py": "import langevin.gpu_langevin as gl\n"}, "langevin"),
    ("imports_common", "research_engine_imports",
     {"model/loading.py": "from common import load_calculator\n"}, "common"),
    ("imports_model_store", "research_engine_imports",
     {"io/restart.py": "import model_store\n"}, "model_store"),
    ("imports_experiments", "research_engine_imports",
     {"bench/__init__.py": "from Experiments.Langevin_Thermostat import langevin\n"},
     "Experiments"),
    ("imports_langevin_dynamically", "research_engine_imports",
     {"cli/__init__.py": "import importlib\nm = importlib.import_module('langevin.common')\n"},
     "langevin"),
    ("nas_path", "research_engine_imports",
     {"model/loading.py": "MODELS = '/mnt/homes/zwingel/models'\n"}, "/mnt/homes"),
    ("global_statement", "no_global_statement",
     {"engine/graph.py": "_POOL = {}\ndef reset():\n    global _POOL\n    _POOL = {}\n"},
     "global _POOL"),
    ("switch_false", "no_module_switches",
     {"engine/graph.py": "FOLD_RADIAL_CUTOFF = False\n"}, "FOLD_RADIAL_CUTOFF"),
    ("switch_true_annotated", "no_module_switches",
     {"engine/graph.py": "FIRST_DERIVATIVE_ONLY: bool = True\n"}, "FIRST_DERIVATIVE_ONLY"),
    ("switch_none", "no_module_switches",
     {"engine/graph.py": "PREFERRED_BLAS = None\n"}, "PREFERRED_BLAS"),
    ("switch_private_chained", "no_module_switches",
     {"rewrites/registry.py": "_A = _B = False\n"}, "_"),
    ("switch_in_try_block", "no_module_switches",
     {"kernels/__init__.py": "try:\n    import triton\n    HAS_TRITON = True\n"
                             "except ImportError:\n    HAS_TRITON = False\n"}, "HAS_TRITON"),
]


@pytest.mark.parametrize(
    "check, files, expected",
    [pytest.param(c, f, e, id=i) for i, c, f, e in NEGATIVE_FIXTURES],
)
def test_negative_fixture_fails(tmp_path, check, files, expected):
    violations = CHECKS[check](make_package(tmp_path, files))
    assert violations, f"{check} accepted a package that breaks it"
    assert all(expected in v for v in violations), violations


def test_every_check_has_a_negative_fixture():
    assert {c for _, c, _, _ in NEGATIVE_FIXTURES} == set(CHECKS)


# Things the checks must accept, so that the negative fixtures fail for the right reason.
POSITIVE_FIXTURES = [
    ("integrators_import_forcefield", "import_rules",
     {"integrators/nve.py": "from ..model.forces import ForceField\nfrom ..model import forces\n"
                            "from fastmace.state import SimulationState\n"}),
    ("engine_imports_model", "import_rules",
     {"engine/graph.py": "from ..model.forces import ForceField\nfrom ..rewrites import registry\n"}),
    ("relative_common_inside_fastmace", "research_engine_imports",
     {"io/restart.py": "from . import common\nfrom .common import x\n"}),
    ("lowercase_and_non_bool_constants", "no_module_switches",
     {"neighbors/buckets.py": "EDGE_MULTIPLE = 256\nEDGE_BUCKET_FRACTION = 0.05\n"
                              "__all__ = ['x']\n_cache = None\ndebug = False\n"
                              "def f():\n    LOCAL = False\n"
                              "class C:\n    FLAG = False\n"}),
    ("config_is_exempt", "no_module_switches",
     {"config.py": "DEFAULT_GRAPHS = True\n"}),
]


@pytest.mark.parametrize(
    "check, files", [pytest.param(c, f, id=i) for i, c, f in POSITIVE_FIXTURES]
)
def test_positive_fixture_passes(tmp_path, check, files):
    violations = CHECKS[check](make_package(tmp_path, files))
    assert not violations, "\n".join(violations)


def test_allow_list_admits_named_constant(tmp_path):
    modules = make_package(tmp_path, {"engine/graph.py": "SUPPORTS_X = False\nOTHER = False\n"})
    allow = {"fastmace/engine/graph.py": {"SUPPORTS_X": "fixed fact about the platform"}}
    violations = check_no_module_switches(modules, allow=allow)
    assert len(violations) == 1 and "OTHER" in violations[0]
