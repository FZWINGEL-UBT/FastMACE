"""An earlier FastMACE release beside this one in one process (plan task 1b-T23, acceptance item 5).

Step-time differences of 0.16 % are visible only in one process (cross-process noise is about 3 %,
SPEC §12.5), so 1b is timed against the 1a release in alternating blocks of one process. Both are the
package ``fastmace``, so the earlier release is exported from its git tag under another name:

- ``export_renamed`` writes ``<destination>/<name>/`` from ``git archive <ref> fastmace``, with every
  standalone word ``fastmace`` in its Python files replaced by ``name``: imports, module-name strings,
  the ``fastmace::`` namespace of its torch custom operators (two registrations of one name would
  fail) and its Inductor cache key. Environment variables (``FASTMACE_CACHE``) and other identifiers
  keep their names; non-Python files (CUDA sources) are copied unchanged. ``SOURCE.json`` records the
  tag, its commit and a digest of the written tree, which ``load_renamed`` checks.
- ``ProcessPatches`` handles what both versions install process-wide. Each patches MACE and Inductor
  in place (``fastmace._patches``: ``mask_head``, ``compute_forces``, the one-graph strain, the
  unfused ``addmm`` rule and Inductor's single custom-pass slot, the cuEq conversion), and each refuses
  or misreads the other's replacement: a version whose qualification sees the other's function hashes
  the wrong source. ``ProcessPatches`` generalises ``tools/ab_step_time.py``'s ``CustomPassSlot``: it
  records MACE's own objects once, gives a side those before its first load, and afterwards its own,
  before every load, build, warm-up and timed block (``fastmace.bench.ab.run_ab``'s ``before``), outside
  the clock. A compiled model guards on the functions it traced, so a side that ran with the other's
  patches would recompile; the A/B tool checks that nothing compiled while timed.
- ``share_record`` gives the earlier release this process's OEq runtime record: there is one OEq host
  extension per process, prepared by this release before MACE was imported, and the earlier
  release's loader would otherwise refuse the hybrid because MACE is already imported.

PolarMACE (its k-space patch on the class), plain cuEq (its conversion patch is per load) and NPT are
not taken in one process by ``tools/ab_fastmace.py``; those comparisons run in alternating processes.

The export uses the standard library only and runs without torch, so it can be prepared on a CPU
machine and copied to the workstations: ``python -m fastmace.bench.versions export --repository . --ref
fastmace-1a --destination DIR``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import keyword
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

PACKAGE = "fastmace"
SOURCE_FILE = "SOURCE.json"
DEFAULT_NAME = "fastmace_1a"
_WORD = re.compile(r"(?<![A-Za-z0-9_])fastmace(?![A-Za-z0-9_])")

# Every (module, attribute) that fastmace._patches replaces in place, in 1a and in 1b (the test compares
# this list with both versions' _patches.py). Absent modules or attributes are skipped.
PATCH_TARGETS = (
    ("mace.modules.blocks", "mask_head"),
    ("mace.modules.utils", "compute_forces"),
    ("mace.modules.utils", "compute_forces_virials"),
    ("mace.modules.utils", "get_symmetric_displacement"),
    ("mace.modules.models", "get_symmetric_displacement"),
    ("torch._inductor.fx_passes.post_grad", "should_prefer_unfused_addmm"),
    ("torch._inductor.config", "post_grad_custom_pre_pass"),
    ("mace.cli.convert_e3nn_cueq", "run"),
    ("mace.calculators.mace", "run_e3nn_to_cueq"),
)
_MISSING = object()


class VersionError(RuntimeError):
    """A renamed release is missing, changed, or not the commit it should be."""


# ------------------------------------------------------------------ export


def _check_name(name: str) -> None:
    if name == PACKAGE or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"the renamed package needs an identifier other than {PACKAGE!r}, not {name!r}")


def rename_source(text: str, name: str) -> str:
    """``text`` with every standalone word ``fastmace`` replaced by ``name`` (see the module docstring)."""
    _check_name(name)
    return _WORD.sub(name, text)


def tree_digest(root) -> str:
    """SHA-256 over the relative paths and contents of every file under ``root`` except ``SOURCE.json``."""
    root = Path(root)
    digest = hashlib.sha256()
    files = {p.relative_to(root).as_posix(): p for p in root.rglob("*")
             if p.is_file() and p.name != SOURCE_FILE and "__pycache__" not in p.parts}
    for relative in sorted(files):  # by POSIX path: the same order on every platform (Windows sorts Paths casefolded)
        digest.update(relative.encode() + b"\0")
        digest.update(files[relative].read_bytes() + b"\0")
    return digest.hexdigest()


def _git(repository, *args) -> bytes:
    # safe.directory=*: on NFS (the NAS home) the repository's owner uid differs from the login's, and git
    # refuses a "dubious ownership" repository otherwise. We only read from it.
    done = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(repository), *args], capture_output=True,
                          check=False)
    if done.returncode != 0:
        raise VersionError(f"git {' '.join(args)} failed in {repository}: "
                           f"{done.stderr.decode(errors='replace').strip()}")
    return done.stdout


def export_renamed(repository, ref: str, destination, name: str = DEFAULT_NAME) -> dict:
    """Write ``<destination>/<name>/`` from ``ref``'s ``fastmace`` package (renamed); returns the SOURCE record."""
    _check_name(name)
    destination = Path(destination)
    target = destination / name
    if target.exists():
        raise FileExistsError(f"{target} exists; remove it or choose another destination")
    commit = _git(repository, "rev-parse", f"{ref}^{{commit}}").decode().strip()
    archive = _git(repository, "-c", "core.autocrlf=false", "archive", "--format=tar", commit, PACKAGE)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as scratch:
        staging = Path(scratch) / name
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts
                if member.issym() or member.islnk() or ".." in parts or parts[0] != PACKAGE:
                    raise VersionError(f"unexpected archive member {member.name}")
                if not member.isfile():
                    continue
                data = tar.extractfile(member).read()
                path = staging.joinpath(*parts[1:])
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix == ".py":
                    data = rename_source(data.decode("utf-8"), name).encode("utf-8")
                path.write_bytes(data)
        record = dict(name=name, ref=ref, commit=commit, renamed=f"{PACKAGE} -> {name} (standalone words, .py files)",
                      tree_sha256=tree_digest(staging))
        (staging / SOURCE_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        shutil.move(str(staging), str(target))
    return record


def check_renamed(destination, name: str = DEFAULT_NAME, expected_commit: str | None = None) -> dict:
    """The SOURCE record of ``<destination>/<name>``, after checking its tree digest and (a prefix of) its commit."""
    package = Path(destination) / name
    source = package / SOURCE_FILE
    if not source.is_file():
        raise VersionError(f"{package} has no {SOURCE_FILE}: export it with export_renamed")
    record = json.loads(source.read_text(encoding="utf-8"))
    if record.get("name") != name:
        raise VersionError(f"{source} names package {record.get('name')!r}, not {name!r}")
    if tree_digest(package) != record.get("tree_sha256"):
        raise VersionError(f"{package}: the tree differs from the one exported (tree digest); export it again")
    if expected_commit and not str(record.get("commit", "")).startswith(expected_commit):
        raise VersionError(f"{package} is commit {record.get('commit')}, not {expected_commit}")
    return record


def load_renamed(destination, name: str = DEFAULT_NAME, expected_commit: str | None = None):
    """``(module, SOURCE record)``: the renamed release imported from ``destination`` and nowhere else."""
    record = check_renamed(destination, name, expected_commit)
    root = str(Path(destination).resolve())
    if name in sys.modules:
        module = sys.modules[name]
    else:
        while root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
        module = importlib.import_module(name)
    if Path(module.__file__).resolve().parent != (Path(root) / name).resolve():
        raise VersionError(f"{name} was imported from {module.__file__}, not from {root}")
    return module, record


# ------------------------------------------------------------------ process-wide patches


class ProcessPatches:
    """Gives each side its own process-wide patches (see the module docstring).

    Construct it after this release prepared its OEq runtime and before any side loads a model: it
    imports the target modules and records their objects as MACE's own.
    """

    def __init__(self, targets: Iterable[tuple[str, str]] = PATCH_TARGETS):
        self.targets = []
        self.pristine = {}
        for module_name, attribute in targets:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            key = f"{module_name}:{attribute}"
            self.targets.append((key, module, attribute))
            self.pristine[key] = getattr(module, attribute, _MISSING)
        self.sides = {}
        self.current = None

    def _put(self, values) -> None:
        for key, module, attribute in self.targets:
            value = values[key]
            if value is _MISSING:
                if hasattr(module, attribute):
                    delattr(module, attribute)
            else:
                setattr(module, attribute, value)

    def _read(self) -> dict:
        return {key: getattr(module, attribute, _MISSING) for key, module, attribute in self.targets}

    def enter(self, side: str) -> None:
        """Put ``side``'s patches in place: what it left last time, MACE's own before its first load."""
        self._put(self.sides.get(side, self.pristine))
        self.current = side

    def leave(self, side: str) -> list[str]:
        """Record what ``side`` left in place; returns the targets it changed since it last left (empty when
        the side had not left before)."""
        now = self._read()
        before = self.sides.get(side)
        self.sides[side] = now
        if before is None:
            return []
        return [key for key in now if now[key] is not before[key]]

    def restore_pristine(self) -> None:
        self._put(self.pristine)
        self.current = None

    def as_dict(self) -> dict:
        def name(value):
            if value is _MISSING:
                return None
            if hasattr(value, "uuid"):
                return f"custom pass {value.uuid()}"
            module = getattr(value, "__module__", None)
            qualified = getattr(value, "__qualname__", None)
            return f"{module}.{qualified}" if module and qualified else repr(value)

        return dict(targets=[key for key, _, _ in self.targets],
                    pristine={key: name(value) for key, value in self.pristine.items()},
                    sides={side: {key: name(value) for key, value in values.items()}
                           for side, values in self.sides.items()})


def share_record(source_patches, target_patches, name: str = "oeq_runtime") -> dict | None:
    """Give ``target_patches`` (the earlier release's ``_patches``) ``source_patches``' record ``name``.

    Returns the shared record as a dict, or None when the source has none. Raises when the target
    holds a different one.
    """
    record = source_patches._RECORDS.get(name)
    if record is None:
        return None
    existing = target_patches._RECORDS.get(name)
    shared = target_patches.PatchRecord(record.name, record.value, tuple(record.targets),
                                        f"{record.detail} (shared from {source_patches.__name__})")
    if existing is not None and (existing.value, tuple(existing.targets)) != (shared.value, shared.targets):
        raise VersionError(f"{target_patches.__name__} already holds another {name} record")
    target_patches._RECORDS[name] = existing or shared
    if name in source_patches._KEEP_ALIVE:
        target_patches._KEEP_ALIVE.setdefault(name, source_patches._KEEP_ALIVE[name])
    return dict(name=shared.name, value=shared.value, targets=list(shared.targets), detail=shared.detail)


# ------------------------------------------------------------------ command line


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m fastmace.bench.versions", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Write a renamed copy of a tagged release")
    export.add_argument("--repository", default=".", type=Path)
    export.add_argument("--ref", default="fastmace-1a")
    export.add_argument("--destination", required=True, type=Path)
    export.add_argument("--name", default=DEFAULT_NAME)
    check = sub.add_parser("check", help="Check a renamed copy")
    check.add_argument("--destination", required=True, type=Path)
    check.add_argument("--name", default=DEFAULT_NAME)
    check.add_argument("--expected-commit")
    args = p.parse_args(argv)
    if args.command == "export":
        record = export_renamed(args.repository, args.ref, args.destination, args.name)
    else:
        record = check_renamed(args.destination, args.name, args.expected_commit)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
