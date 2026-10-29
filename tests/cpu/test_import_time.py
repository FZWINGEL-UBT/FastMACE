"""Import-time behaviour (SPEC §14 "Caches", §13): soft file locks, the CUDA driver cache on NFS, the cache root.

Import order is the thing under test, so every test runs its code in a fresh interpreter.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Variables that change what `import fastmace` does; removed from every child's environment.
CONTROLLED = (
    "FASTMACE_SOFT_FILE_LOCKS",
    "FASTMACE_CACHE",
    "LANGEVIN_SOFT_FILE_LOCKS",
    "NFS_SOFT_LOCKS_HOOK",
    "CUDA_CACHE_PATH",
    "CUDA_CACHE_DISABLE",
)


def run(code, soft=None, hook=False, **extra):
    """Run code in a fresh interpreter; return (exit code, last stdout line, stderr).

    hook=False switches off the `nfs-soft-locks` start-up hook that our research environments
    carry (NFS_SOFT_LOCKS_HOOK=0), so FastMACE's own switch is tested on its own.
    """
    env = {k: v for k, v in os.environ.items() if k not in CONTROLLED}
    pythonpath = [str(REPOSITORY_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env.update(PYTHONPATH=os.pathsep.join(pythonpath), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    if not hook:
        env["NFS_SOFT_LOCKS_HOOK"] = "0"
    if soft is not None:
        env["FASTMACE_SOFT_FILE_LOCKS"] = soft
    env.update(extra)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPOSITORY_ROOT,
        env=env,
        check=False,  # the caller inspects the exit code
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = result.stdout.strip().splitlines()
    return result.returncode, (lines[-1] if lines else ""), result.stderr


def run_json(code, **kwargs):
    code_, line, err = run(code, **kwargs)
    assert code_ == 0, err
    return json.loads(line)


# --- Soft locks for PyTorch (ported from langevin/file_locks.py's tests) ---------------------------


def test_default_makes_pytorch_locks_soft():
    result = run_json("""
        import json, fastmace, torch, filelock
        from torch.utils._filelock import FileLock
        from torch.utils import cpp_extension
        info = fastmace.runtime_info()
        # torch < 2.14 locks extension builds with its own FileBaton (an O_EXCL lock file, already soft).
        extension_lock = getattr(cpp_extension, "FileLock", filelock.SoftFileLock)
        print(json.dumps(dict(mode=info.file_locks, late=info.file_lock_switch_late,
                              torch_soft=issubclass(FileLock, filelock.SoftFileLock),
                              extension_soft=issubclass(extension_lock, filelock.SoftFileLock),
                              async_soft=filelock.AsyncFileLock is filelock.AsyncSoftFileLock)))
    """)
    assert result == {"mode": "soft", "late": False, "torch_soft": True, "extension_soft": True, "async_soft": True}


def test_soft_lock_excludes_and_removes_its_file():
    with tempfile.TemporaryDirectory() as temp:
        result = run_json(f"""
            import json, os, fastmace, filelock
            from torch.utils._filelock import FileLock
            path = os.path.join({temp!r}, "build", "lock")
            first = FileLock(path)
            first.acquire()
            held = os.path.exists(path)
            try:
                FileLock(path, timeout=0).acquire()
                excluded = False
            except filelock.Timeout:
                excluded = True
            first.release()
            print(json.dumps(dict(held=held, excluded=excluded, removed=not os.path.exists(path))))
        """)
    assert result == {"held": True, "excluded": True, "removed": True}


def test_stale_lock_of_a_dead_process_is_broken():
    with tempfile.TemporaryDirectory() as temp:
        path = os.path.join(temp, "lock")
        code, _, err = run(f"""
            import os, fastmace
            from torch.utils._filelock import FileLock
            lock = FileLock({path!r})  # keep a reference: a collected lock releases itself
            lock.acquire()
            os._exit(0)  # dies holding the lock: the marker stays
        """)
        assert code == 0, err
        assert os.path.exists(path)
        result = run_json(f"""
            import json, fastmace
            from torch.utils._filelock import FileLock
            with FileLock({path!r}, timeout=10):
                print(json.dumps("acquired"))
        """)
    assert result == "acquired"


def test_zero_keeps_kernel_locks():
    result = run_json("""
        import json, fastmace, filelock
        from torch.utils._filelock import FileLock
        info = fastmace.runtime_info()
        print(json.dumps(dict(mode=info.file_locks, source=info.file_lock_source,
                              kernel=not issubclass(FileLock, filelock.SoftFileLock))))
    """, soft="0")
    assert result == {"mode": "kernel", "source": "opted out (FASTMACE_SOFT_FILE_LOCKS=0)", "kernel": True}


def test_one_insists_on_soft_locks():
    result = run_json("""
        import json, fastmace, torch
        print(json.dumps(fastmace.runtime_info().file_locks))
    """, soft="1")
    assert result == "soft"


def test_invalid_value_is_refused():
    code, _, err = run("import fastmace", soft="yes")
    assert code != 0
    assert "FASTMACE_SOFT_FILE_LOCKS" in err


# --- Import after torch ----------------------------------------------------------------------------


def test_torch_imported_first_works_and_reports_the_late_switch():
    result = run_json("""
        import json, torch, fastmace
        info = fastmace.runtime_info()
        print(json.dumps(dict(mode=info.file_locks, late=info.file_lock_switch_late, source=info.file_lock_source)))
    """)
    assert result["mode"] == "kernel"  # unset: too late to switch, kept and recorded
    assert result["late"] is True
    assert "torch was imported before fastmace" in result["source"]


def test_torch_imported_first_with_insist_fails():
    code, _, err = run("import torch, fastmace", soft="1")
    assert code != 0
    assert "before torch is imported" in err


def test_runtime_info_is_frozen():
    result = run_json("""
        import dataclasses, json, fastmace
        info = fastmace.runtime_info()
        try:
            info.file_locks = "kernel"
            frozen = False
        except dataclasses.FrozenInstanceError:
            frozen = True
        print(json.dumps(dict(frozen=frozen, record=dataclasses.asdict(info))))
    """)
    assert result["frozen"] is True
    assert set(result["record"]) >= {"file_locks", "file_lock_source", "file_lock_switch_late", "cuda_driver_cache"}


# --- Coexistence with the nfs-soft-locks start-up hook ------------------------------------------------

# What environment/nfs_soft_locks does at interpreter start (ResearchModule), reproduced so the test
# needs no installed hook: soft filelock classes and a module `nfs_soft_locks` with its STATE.
FAKE_HOOK = """
import sys, types, filelock
filelock.FileLock = filelock.SoftFileLock
filelock.AsyncFileLock = filelock.AsyncSoftFileLock
hook = types.ModuleType("nfs_soft_locks")
hook.STATE = {"filelock": "soft", "cuda_cache": "untouched", "error": None}
sys.modules["nfs_soft_locks"] = hook
"""


@pytest.mark.parametrize("torch_first", [False, True])
def test_hook_already_active_is_a_no_op(torch_first):
    order = "torch, fastmace" if torch_first else "fastmace, torch"
    result = run_json(FAKE_HOOK + f"""
import json, os
classes = (filelock.FileLock, filelock.AsyncFileLock)
environ = dict(os.environ)
import {order}
from torch.utils._filelock import FileLock
info = fastmace.runtime_info()
print(json.dumps(dict(unchanged=(filelock.FileLock, filelock.AsyncFileLock) == classes,
                      environ=dict(os.environ) == environ, torch_soft=issubclass(FileLock, filelock.SoftFileLock),
                      mode=info.file_locks, late=info.file_lock_switch_late, source=info.file_lock_source)))
""")
    assert result == {"unchanged": True, "environ": True, "torch_soft": True, "mode": "soft", "late": False,
                      "source": "start-up hook (nfs_soft_locks)"}


def test_opt_out_restores_kernel_locks_after_the_hook():
    result = run_json(FAKE_HOOK + """
import json, fastmace, torch
from torch.utils._filelock import FileLock
info = fastmace.runtime_info()
print(json.dumps(dict(mode=info.file_locks, source=info.file_lock_source,
                      torch_kernel=not issubclass(FileLock, filelock.SoftFileLock),
                      async_kernel=not issubclass(filelock.AsyncFileLock, filelock.AsyncSoftFileLock))))
""", soft="0")
    assert result == {"mode": "kernel", "source": "opted out (FASTMACE_SOFT_FILE_LOCKS=0)",
                      "torch_kernel": True, "async_kernel": True}


def test_opt_out_after_torch_keeps_the_hooks_soft_locks_and_says_so():
    result = run_json(FAKE_HOOK + """
import json, torch, fastmace
info = fastmace.runtime_info()
print(json.dumps(dict(mode=info.file_locks, source=info.file_lock_source,
                      unchanged=filelock.FileLock is filelock.SoftFileLock)))
""", soft="0")
    assert result["mode"] == "soft"
    assert result["unchanged"] is True
    assert "FASTMACE_SOFT_FILE_LOCKS=0" in result["source"]
    assert "soft since torch was imported" in result["source"]


def hook_installed():
    code, line, _ = run("import importlib.util, json; "
                        "print(json.dumps(importlib.util.find_spec('nfs_soft_locks') is not None))", hook=True)
    return code == 0 and json.loads(line)


@pytest.mark.parametrize("torch_first", [False, True])
def test_opt_out_with_the_installed_hook(torch_first):
    if not hook_installed():
        pytest.skip("nfs-soft-locks start-up hook not installed in this environment")
    order = "torch, fastmace" if torch_first else "fastmace, torch"
    result = run_json(f"""
        import json, filelock, nfs_soft_locks
        hook = nfs_soft_locks.STATE["filelock"]
        import {order}
        print(json.dumps(dict(hook=hook, mode=fastmace.runtime_info().file_locks)))
    """, soft="0", hook=True)
    assert result == {"hook": "soft", "mode": "soft" if torch_first else "kernel"}


def test_installed_hook_is_a_no_op():
    """The real hook, where the environment has it (our research environments do; CI does not)."""
    if not hook_installed():
        pytest.skip("nfs-soft-locks start-up hook not installed in this environment")
    result = run_json("""
        import json, filelock, nfs_soft_locks
        before = filelock.FileLock
        import torch, fastmace
        info = fastmace.runtime_info()
        print(json.dumps(dict(hook=nfs_soft_locks.STATE["filelock"], unchanged=filelock.FileLock is before,
                              mode=info.file_locks, late=info.file_lock_switch_late, source=info.file_lock_source)))
    """, hook=True)
    assert result == {"hook": "soft", "unchanged": True, "mode": "soft", "late": False,
                      "source": "start-up hook (nfs_soft_locks)"}


# --- The CUDA driver cache on NFS ----------------------------------------------------------------------


def mounts_table(temp):
    """A fake /proc/self/mounts: NFS at <temp>/nas, a local disk mounted inside it, a space in a name."""
    mounts = Path(temp) / "mounts"
    mounts.write_text(
        "/dev/sda1 / ext4 rw 0 0\n"
        f"server:/homes {temp}/nas nfs4 rw,relatime 0 0\n"
        f"/dev/sdb1 {temp}/nas/scratch ext4 rw 0 0\n"
        f"server:/old {temp}/old\\040share nfs rw 0 0\n"
        "garbage\n"
    )
    return str(mounts)


def test_nfs_detection_from_the_mounts_table():
    with tempfile.TemporaryDirectory() as temp:
        result = run_json(f"""
            import json
            from fastmace._file_locks import on_nfs
            t, m = {temp!r}, {mounts_table(temp)!r}
            print(json.dumps([on_nfs(t + "/nas/cache/cuda-driver-cache", m), on_nfs(t + "/nas", m),
                              on_nfs(t + "/local/cache", m), on_nfs(t + "/nasty", m),
                              on_nfs(t + "/nas/scratch/cache", m), on_nfs(t + "/old share/cache", m),
                              on_nfs(t + "/nas/x", t + "/missing-mounts-file")]))
        """)
    assert result == [True, True, False, False, False, True, False]


def test_nfs_detection_never_calls_stat_on_the_filesystem():
    with tempfile.TemporaryDirectory() as temp:
        result = run_json(f"""
            import json, os, subprocess
            def refuse(*args, **kwargs):
                raise AssertionError("no statvfs or subprocess")
            os.statvfs = refuse
            subprocess.run = subprocess.Popen = refuse
            from fastmace._file_locks import on_nfs
            print(json.dumps(on_nfs({temp!r} + "/nas/cache", {mounts_table(temp)!r})))
        """)
    assert result is True


def test_cuda_driver_cache_rule():
    with tempfile.TemporaryDirectory() as temp:
        result = run_json(f"""
            import json
            from fastmace._file_locks import configure_cuda_driver_cache
            t, m = {temp!r}, {mounts_table(temp)!r}
            cases = dict(
                nfs=dict(CUDA_CACHE_PATH=t + "/nas/cuda"),
                local=dict(CUDA_CACHE_PATH=t + "/local/cuda"),
                user_zero=dict(CUDA_CACHE_PATH=t + "/nas/cuda", CUDA_CACHE_DISABLE="0"),
                user_one=dict(CUDA_CACHE_PATH=t + "/local/cuda", CUDA_CACHE_DISABLE="1"),
                default_on_nfs=dict(HOME=t + "/nas/home"),
                default_local=dict(HOME=t + "/local/home"),
            )
            out = {{}}
            for name, environ in cases.items():
                status = configure_cuda_driver_cache(environ, mounts_file=m)
                out[name] = [environ.get("CUDA_CACHE_DISABLE"), status]
            print(json.dumps(out))
        """)
    assert result["nfs"][0] == "1" and result["nfs"][1].startswith("disabled")
    assert result["local"] == [None, "enabled"]
    assert result["user_zero"][0] == "0" and "set by the user" in result["user_zero"][1]
    assert result["user_one"][0] == "1" and "set by the user" in result["user_one"][1]
    # Without CUDA_CACHE_PATH the driver uses ~/.nv/ComputeCache, which is on NFS where the home is.
    assert result["default_on_nfs"][0] == "1"
    assert result["default_local"] == [None, "enabled"]


def test_import_applies_the_cuda_rule_unless_opted_out():
    # The real mounts table of this machine: point the cache at a path and compare with on_nfs.
    code = """
        import json, os, fastmace
        from fastmace._file_locks import on_nfs
        print(json.dumps(dict(nfs=on_nfs(os.environ["CUDA_CACHE_PATH"]), disable=os.environ.get("CUDA_CACHE_DISABLE"),
                              status=fastmace.runtime_info().cuda_driver_cache)))
    """
    with tempfile.TemporaryDirectory() as temp:
        result = run_json(code, CUDA_CACHE_PATH=temp)
        opted_out = run_json(code, soft="0", CUDA_CACHE_PATH=temp)
    assert result["disable"] == ("1" if result["nfs"] else None)
    assert opted_out["disable"] is None
    assert "FASTMACE_SOFT_FILE_LOCKS=0" in opted_out["status"]


# --- Cache root ----------------------------------------------------------------------------------------


def test_cache_root_from_environment_and_subdirectories():
    with tempfile.TemporaryDirectory() as temp:
        result = run_json("""
            import json, os
            from fastmace import _cache
            root = _cache.cache_root()
            paths = {kind: str(_cache.cache_dir(kind, create=False)) for kind in _cache.CACHE_KINDS}
            print(json.dumps(dict(root=str(root), kinds=sorted(_cache.CACHE_KINDS), paths=paths,
                                  created=os.path.exists(paths["triton"]))))
        """, FASTMACE_CACHE=os.path.join(temp, "fm"))
        assert Path(result["root"]) == Path(temp, "fm").resolve()
    assert result["kinds"] == ["compile", "inductor", "models", "oeq", "radial-tables", "triton"]  # radial-table-cache
    assert all(Path(p).parent == Path(result["root"]) for p in result["paths"].values())
    assert result["created"] is False


def test_cache_dir_is_created_on_request():
    with tempfile.TemporaryDirectory() as temp:
        result = run_json("""
            import json
            from fastmace import _cache
            path = _cache.cache_dir("inductor")
            print(json.dumps([str(path), path.is_dir()]))
        """, FASTMACE_CACHE=temp)
    assert result[1] is True and Path(result[0]).name == "inductor"


def test_unknown_cache_kind_is_refused():
    code, _, err = run("from fastmace import _cache; _cache.cache_dir('pip', create=False)")
    assert code != 0 and "pip" in err


def test_cache_root_is_never_inside_sys_prefix():
    result = run_json("""
        import json, os, sys
        from fastmace import _cache
        from fastmace.errors import FastMACEError
        out = {}
        for name, environ in dict(
            explicit=dict(FASTMACE_CACHE=os.path.join(sys.prefix, "cache")),
            prefix_itself=dict(FASTMACE_CACHE=sys.prefix),
            default_home=dict(HOME=os.path.join(sys.prefix, "home"), XDG_CACHE_HOME="",
                              LOCALAPPDATA=os.path.join(sys.prefix, "home")),
            default_xdg=dict(XDG_CACHE_HOME=os.path.join(sys.prefix, "xdg"),
                             LOCALAPPDATA=os.path.join(sys.prefix, "xdg")),
        ).items():
            try:
                out[name] = str(_cache.cache_root(environ))
            except FastMACEError as error:
                out[name] = "refused: " + str(error)
        print(json.dumps(out))
    """)
    for name, value in result.items():
        assert value.startswith("refused: "), (name, value)
        assert "sys.prefix" in value


def test_default_cache_root_is_the_user_cache_directory():
    with tempfile.TemporaryDirectory() as temp:
        result = run_json(f"""
            import json
            from fastmace import _cache
            print(json.dumps([str(_cache.cache_root(dict(HOME={temp!r}))),
                              str(_cache.cache_root(dict(HOME={temp!r}, XDG_CACHE_HOME={temp!r} + "/xdg")))]))
        """)
    if sys.platform.startswith("linux"):
        assert [Path(p) for p in result] == [Path(temp, ".cache", "fastmace").resolve(),
                                             Path(temp, "xdg", "fastmace").resolve()]


def test_import_sets_no_cache_variables_and_creates_nothing():
    with tempfile.TemporaryDirectory() as temp:
        root = os.path.join(temp, "fm")
        result = run_json("""
            import json, os
            before = dict(os.environ)
            import fastmace
            after = dict(os.environ)
            print(json.dumps(sorted(k for k in after if before.get(k) != after[k])))
        """, FASTMACE_CACHE=root)
        assert not os.path.exists(root)
    assert set(result) <= {"CUDA_CACHE_DISABLE"}


# --- Lazy public names (plan task 1b-T01) --------------------------------------------------------------

PUBLIC_NAMES = ("load", "ForceField", "Simulation", "Workload", "Output", "integrators", "protocols", "md", "calc",
                "D3")
# The records a user builds before any model: resolving them must not import torch or MACE either.
LIGHT_NAMES = ("Simulation", "Workload", "Output", "integrators", "protocols", "md", "D3")


def test_import_fastmace_imports_neither_torch_nor_mace():
    result = run_json("""
        import json, sys
        import fastmace
        print(json.dumps(sorted(m for m in sys.modules if m.split(".")[0] in ("torch", "mace", "ase"))))
    """)
    assert result == []


def test_public_names_are_listed_and_lazy():
    result = run_json(f"""
        import json, sys
        import fastmace
        listed = sorted(set({PUBLIC_NAMES!r}) - set(fastmace.__all__))
        in_dir = sorted(set({PUBLIC_NAMES!r}) - set(dir(fastmace)))
        loaded_before = sorted(n for n in {PUBLIC_NAMES!r} if n in vars(fastmace))
        light = {{name: type(getattr(fastmace, name)).__name__ for name in {LIGHT_NAMES!r}}}
        heavy_after_light = sorted(m for m in sys.modules if m.split(".")[0] in ("torch", "mace"))
        print(json.dumps(dict(listed=listed, in_dir=in_dir, loaded_before=loaded_before, light=light,
                              heavy_after_light=heavy_after_light)))
    """)
    assert result["listed"] == [] and result["in_dir"] == []
    assert result["loaded_before"] == []
    assert result["light"]["integrators"] == "module" and result["light"]["protocols"] == "module"
    assert result["heavy_after_light"] == []


def test_every_public_name_resolves_to_its_owner():
    result = run_json(f"""
        import json
        import fastmace
        out = {{}}
        for name in {PUBLIC_NAMES!r}:
            value = getattr(fastmace, name)
            out[name] = getattr(value, "__module__", None) or value.__name__
        print(json.dumps(out))
    """)
    assert result == {
        "load": "fastmace.model.loading",
        "ForceField": "fastmace.model.forcefield",
        "Simulation": "fastmace.simulation",
        "Workload": "fastmace.planner.workload",
        "Output": "fastmace.io.output",
        "integrators": "fastmace.integrators",
        "protocols": "fastmace.protocols",
        "md": "fastmace.quick",
        "calc": "fastmace.ase.calculator",
        "D3": "fastmace.longrange.d3",
    }


def test_unknown_attribute_is_an_attribute_error():
    code, _, err = run("import fastmace; fastmace.no_such_name")
    assert code != 0 and "AttributeError" in err and "no_such_name" in err
