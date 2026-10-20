"""Pre-steps for polar_probe: stages of test_a_simulation_builds_the_block_once_across_ten_runs."""
import os, sys, tempfile
import torch


def pre(stage):
    from _support.structures import nacl
    from _support.tiny_models import FAMILIES, tiny_mace_model
    from fastmace import integrators
    from fastmace.config import EngineConfig
    from fastmace.engine import constraints
    from fastmace.io.output import Output
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.simulation import Simulation
    dtype = torch.get_default_dtype()
    tmp = tempfile.mkdtemp(dir=os.environ["TMPDIR"])
    family = os.environ.get("T14_FAMILY", "mh1")
    model = tiny_mace_model([11, 17], **FAMILIES[family])
    path = os.path.join(tmp, "tiny.model")
    torch.save(model, path)
    if stage == "software":
        from fastmace import provenance
        provenance._software("cpu")
        return torch.set_default_dtype(dtype)
    if stage == "detect":
        minimal.detect_software("cpu")
        return torch.set_default_dtype(dtype)
    if stage == "save":
        return torch.set_default_dtype(dtype)
    if stage.startswith("mff"):
        import test_forces as tf
        from fastmace.model.forces import MACEForceField
        loading._LICENCES_SHOWN.add(str(path))
        dt = "float32" if "32" in stage else "float64"
        cq = loading.load(str(path), backend="cueq", device="cpu", dtype=dt, trusted=True, allow_naive_cueq=True)
        atoms = nacl(repeat=1, rattle=0.05, seed=3)
        options = {"rewrites": tf.only()} if "off" in stage else {}
        if "e64" in stage:
            options["float64_energy"] = True
        if "load" in stage:
            return torch.set_default_dtype(dtype)
        if stage == "mff-nocheck":
            options["check"] = False
        MACEForceField(cq, atoms, **options)
        return torch.set_default_dtype(dtype)
    if stage == "e3nn32-load":
        loading._LICENCES_SHOWN.add(str(path))
        loading.load(str(path), backend="e3nn", device="cpu", dtype="float32", trusted=True)
        return torch.set_default_dtype(dtype)
    if stage in ("convert32", "convert32-plain"):
        from mace.cli.convert_e3nn_cueq import run
        from fastmace.model import compile as fc
        m32 = model.float()
        torch.set_default_dtype(torch.float32)
        ctx = fc.naive_cueq() if stage == "convert32" else __import__("contextlib").nullcontext()
        with ctx:
            run(m32, device="cpu", return_model=True)
        return torch.set_default_dtype(dtype)
    if stage.startswith("clear-"):  # float32 conversion, then clear the functools caches of one package
        from mace.cli.convert_e3nn_cueq import run
        import gc
        m32 = model.float()
        torch.set_default_dtype(torch.float32)
        run(m32, device="cpu", return_model=True)
        torch.set_default_dtype(dtype)
        package = stage[len("clear-"):]
        cleared = []
        for obj in gc.get_objects():
            if callable(obj) and hasattr(obj, "cache_clear") and hasattr(obj, "cache_info"):
                module = getattr(obj, "__module__", "") or ""
                if module.split(".")[0] == package and obj.cache_info().currsize:
                    obj.cache_clear()
                    cleared.append(f"{module}.{getattr(obj, '__qualname__', obj)}")
        print("CLEARED", cleared)
        return
    if stage == "dtype32-cueq-import":
        torch.set_default_dtype(torch.float32)
        import cuequivariance_torch  # noqa: F401
        import cuequivariance as cue
        return torch.set_default_dtype(dtype)
    if stage in ("cueq", "cueq-naive"):
        loading._LICENCES_SHOWN.add(str(path))
        loading.load(str(path), backend="cueq", device="cpu", trusted=True, allow_naive_cueq=True)
        return torch.set_default_dtype(dtype)
    if stage == "convert":  # MACE's own conversion, as polar_calculators does it
        from mace.cli.convert_e3nn_cueq import run
        from fastmace.model import compile as fc
        with fc.naive_cueq():
            run(model, device="cpu", return_model=True)
        return torch.set_default_dtype(dtype)
    if stage == "convert-plain":
        from mace.cli.convert_e3nn_cueq import run
        run(model, device="cpu", return_model=True)
        return torch.set_default_dtype(dtype)
    loaded = loading.load_pristine(str(path), trusted=True)
    if stage == "load":
        return torch.set_default_dtype(dtype)
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    ff = ForceField(loaded, atoms)
    if stage == "ff":
        return torch.set_default_dtype(dtype)
    planner = minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                     process=constraints.ProcessRecord())
    minimal.default_planner = lambda force_field: planner
    if stage in ("build-field", "build-noeval"):
        import fastmace.engine.graph as graph
        from fastmace.model import forces
        real = forces.MACEForceField

        def recording(model, atoms_, **kw):
            print("BUILD KW:", {k: (v if k != "rewrites" else {n: s for n, s in v.items()} if isinstance(v, dict) else v)
                                for k, v in kw.items() if k not in ("registry", "compile_cache")})
            print("MODEL:", type(model).__name__, getattr(model, "backend", None), getattr(model, "dtype", None))
            return real(model, atoms_, **kw)
        forces.MACEForceField = recording
        if stage == "build-field":
            graph.GraphEngine = lambda *a, **k: (_ for _ in ()).throw(SystemExit("stop"))
        else:  # the field's own forward does nothing: does building the field's model change things?
            from fastmace.model import forces
            forces.MACEForceField.__call__ = lambda *a, **k: (_ for _ in ()).throw(SystemExit("stop"))
        try:
            Simulation(atoms, ff, seed=1, output=Output(os.path.join(tmp, "run")),
                       integrator=integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01))
        except SystemExit:
            pass
        return torch.set_default_dtype(dtype)
    if stage in ("plan", "refs", "build"):
        from fastmace import simulation as simmod
        from fastmace.planner import selfcheck
        orig_refs, orig_build = selfcheck.references, minimal.MinimalPlanner.build
        if stage == "plan":
            selfcheck.references = lambda *a, **k: (_ for _ in ()).throw(SystemExit("stop"))
        elif stage == "refs":
            minimal.MinimalPlanner.build = lambda *a, **k: (_ for _ in ()).throw(SystemExit("stop"))
        try:
            Simulation(atoms, ff, seed=1, output=Output(os.path.join(tmp, "run")),
                       integrator=integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01))
        except SystemExit:
            pass
        selfcheck.references, minimal.MinimalPlanner.build = orig_refs, orig_build
        return torch.set_default_dtype(dtype)
    sim = Simulation(atoms, ff, seed=1, output=Output(os.path.join(tmp, "run")),
                     integrator=integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01))
    if stage == "sim":
        return torch.set_default_dtype(dtype)
    sim.run(steps=1)
    torch.set_default_dtype(dtype)
