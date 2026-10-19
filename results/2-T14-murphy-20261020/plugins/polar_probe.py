"""2-T14 probe: the body of test_polar_force_field_enables_its_own_copy, printing the force deviation.
usage: python polar_probe.py [noprof] [warm=K] [repeat=R]"""
import sys
import fastmace  # noqa: F401
import numpy as np
import torch
sys.path[:0] = ["tests", "tests/cpu"]
args = dict(a.split("=") if "=" in a else (a, "1") for a in sys.argv[1:])
if "ops" in args:
    import cuequivariance_ops_torch  # noqa: F401
if "noprof" in args:
    torch._C._jit_set_profiling_executor(False)
import test_forces as tf

def deviation():
    dtype = torch.get_default_dtype()
    calc, cueq = tf.polar_calculators()
    atoms = tf.structure(oxygen=False)
    field = tf.MACEForceField(cueq, atoms, neighbours=tf.NEIGHBOURS, rewrites=tf.only())
    positions = tf.moved(atoms, 0.05)
    tf.reference(calc, atoms, positions)
    energy, forces = tf.evaluate(field, positions, padded=True)
    e, f = tf.reference(calc, atoms, positions)
    torch.set_default_dtype(dtype)
    if "save" in args:
        np.savez(args["save"], field=forces.numpy(), ref=f)
    d = np.abs(forces.numpy() - f)
    excess = (d - (1e-12 + 1e-12 * np.abs(f))).max()
    return d.max(), excess, abs(float(energy[0]) - e) / abs(e)

if "pre" in args:
    import pre as _pre; _pre.pre(args["pre"])
    if "mods" in args:
        open(args["mods"], "w").write("\n".join(sorted(sys.modules)))
    if "imports" in args:
        import importlib
        for name in open(args["imports"]).read().split():
            try:
                importlib.import_module(name)
            except Exception as error:  # noqa: BLE001
                pass
    from fastmace import _patches
    print("patches after pre:", [r.name for r in _patches.applied()])
    if "restore" in args:
        for rec in reversed(_patches.applied()):
            if rec.name not in ("oeq_runtime", "polar_longrange"):
                _patches.restore(rec.name)
        print("patches after restore:", [r.name for r in _patches.applied()])
    if "dynreset" in args:
        torch._dynamo.reset()
for k in range(int(args.get("warm", "0"))):  # other polar evaluations first, as earlier tests would
    calc = tf.calculator("polar", [11, 17]); tf.reference(calc, tf.structure(oxygen=False), tf.moved(tf.structure(oxygen=False), 0.02, seed=k))
for r in range(int(args.get("repeat", "1"))):
    m, x, er = deviation()
    print(f"{' '.join(sys.argv[1:]) or 'plain'} rep {r}: max |dF| {m:.3e}  excess over tol {x:+.3e}  rel dE {er:.1e}")
