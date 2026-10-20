"""2-T14e timing hook (scratch copy only, never committed): reference work per test, in every process.

Active when T14E_LOG names a JSON-lines file. Imports nothing heavy at start-up: it wraps the targets
below when they are imported (a meta-path finder), so FastMACE's "MACE imported first" checks and the
soft-lock start-up hook see an unchanged process.

Events (one JSON object per line): pid, ppid, test (PYTEST_CURRENT_TEST, inherited by child processes),
kind, seconds, and details:
- mace_calculate: MACECalculator.calculate (MACE's own calculator), with dtype, atoms, a model and a
  positions fingerprint, and the caller ("fastmace.reference", "fastmace.planner.selfcheck", "test", ...);
- gate_references: fastmace.planner.selfcheck.references;
- process: the wall time of every Python process, with flags (research engine imported, MACE calls);
"""

import atexit
import json
import os
import sys
import time

_LOG = os.environ.get("T14E_LOG")


def _emit(kind, seconds, **detail):
    record = {"pid": os.getpid(), "ppid": os.getppid(), "test": os.environ.get("PYTEST_CURRENT_TEST", ""),
              "kind": kind, "seconds": round(seconds, 6), **detail}
    try:
        with open(_LOG, "a") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        pass


if _LOG:
    import importlib.abc
    import importlib.machinery
    import threading

    _START = time.time()
    _FLAGS = {"research_engine": False, "mace_calls": 0, "mace_seconds": 0.0}
    _DEPTH = threading.local()
    _FINGERPRINTS = {}

    def _caller():
        frame = sys._getframe(2)
        seen = []
        for _ in range(40):
            if frame is None:
                break
            name = frame.f_globals.get("__name__", "")
            if name.startswith(("fastmace.reference", "fastmace.planner.selfcheck", "fastmace.ase",
                                "fastmace.model", "ase.md", "ase.optimize", "langevin")):
                seen.append(name)
            frame = frame.f_back
        return seen[-1] if seen else "test"

    def _model_fingerprint(model):
        key = id(model)
        if key not in _FINGERPRINTS:
            import hashlib
            digest = hashlib.sha256()
            for name, value in list(model.state_dict().items())[:8]:
                digest.update(name.encode())
                digest.update(value.detach().reshape(-1)[:64].to("cpu").double().numpy().tobytes())
            _FINGERPRINTS[key] = digest.hexdigest()[:16]
        return _FINGERPRINTS[key]

    def _patch_mace(module):
        cls = module.MACECalculator
        original = cls.calculate

        def calculate(self, atoms=None, *args, **kwargs):
            depth = getattr(_DEPTH, "value", 0)
            _DEPTH.value = depth + 1
            started = time.perf_counter()
            try:
                return original(self, atoms, *args, **kwargs)
            finally:
                seconds = time.perf_counter() - started
                _DEPTH.value = depth
                _FLAGS["mace_calls"] += 1
                _FLAGS["mace_seconds"] += seconds
                try:
                    import hashlib
                    positions = hashlib.sha256(atoms.positions.tobytes()).hexdigest()[:12] if atoms is not None else ""
                    model = _model_fingerprint(self.models[0])
                except Exception:  # noqa: BLE001
                    positions, model = "", ""
                _emit("mace_calculate", seconds, dtype=str(getattr(self, "default_dtype", "")),
                      atoms=len(atoms) if atoms is not None else 0, model=model, positions=positions,
                      caller=_caller(), nested=depth)

        cls.calculate = calculate

    def _patch_selfcheck(module):
        original = module.references

        def references(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                _emit("gate_references", time.perf_counter() - started, check=str(kwargs.get("check", "")))

        module.references = references

    def _flag_research_engine(module):
        _FLAGS["research_engine"] = True

    def _timed(kind, function, **fixed):
        def wrapper(*args, **kwargs):
            depth = getattr(_DEPTH, "value", 0)
            _DEPTH.value = depth + 1
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                _DEPTH.value = depth
                _emit(kind, time.perf_counter() - started, caller=_caller(), nested=depth, **fixed)
        return wrapper

    def _atoms_key(atoms):
        import hashlib
        digest = hashlib.sha256()
        for part in (atoms.numbers, atoms.positions, atoms.cell.array):
            digest.update(part.tobytes())
        return digest.hexdigest()[:16]

    def _patch_reference(module):
        load = module.load

        def timed_load(source, atoms, *args, **kwargs):
            depth = getattr(_DEPTH, "value", 0)
            _DEPTH.value = depth + 1
            started = time.perf_counter()
            try:
                return load(source, atoms, *args, **kwargs)
            finally:
                _DEPTH.value = depth
                _emit("reference_load", time.perf_counter() - started, caller=_caller(), nested=depth,
                      source=str(source)[-60:], dtype=str(kwargs.get("dtype", "float64")), head=str(kwargs.get("head")))

        module.load = timed_load
        cls = module.ReferenceCalculator
        evaluate = cls.evaluate

        def timed_evaluate(self, atoms, *args, **kwargs):
            depth = getattr(_DEPTH, "value", 0)
            _DEPTH.value = depth + 1
            started = time.perf_counter()
            try:
                return evaluate(self, atoms, *args, **kwargs)
            finally:
                _DEPTH.value = depth
                try:
                    key = _atoms_key(atoms)
                except Exception:  # noqa: BLE001
                    key = ""
                _emit("reference_evaluate", time.perf_counter() - started, caller=_caller(), nested=depth,
                      atoms_key=key, natoms=len(atoms))

        cls.evaluate = timed_evaluate

    _TARGETS = {"mace.calculators.mace": _patch_mace, "fastmace.planner.selfcheck": _patch_selfcheck,
                "fastmace.reference.calculators": _patch_reference,
                "langevin.gpu_langevin": _flag_research_engine}

    class _Finder(importlib.abc.MetaPathFinder):
        """Finds the target with the other finders and wraps its loader's exec_module in place.

        The loader object itself is kept (runpy's ``python -m`` needs its get_code)."""

        def find_spec(self, name, path, target=None):
            hook = _TARGETS.get(name)
            if hook is None:
                return None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                spec = getattr(finder, "find_spec", lambda *a: None)(name, path, target)
                if spec is not None and spec.loader is not None:
                    inner = spec.loader.exec_module

                    def exec_module(module, inner=inner, hook=hook):
                        inner(module)
                        try:
                            hook(module)
                        except Exception as error:  # noqa: BLE001
                            _emit("hook_error", 0.0, module=module.__name__, error=repr(error))

                    spec.loader.exec_module = exec_module
                    return spec
            return None

    sys.meta_path.insert(0, _Finder())

    @atexit.register
    def _process_end():
        _emit("process", time.time() - _START, argv0=(sys.argv[0] if sys.argv else ""),
              python=sys.executable, **_FLAGS)
