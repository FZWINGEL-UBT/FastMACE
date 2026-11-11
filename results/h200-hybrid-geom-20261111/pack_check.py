"""CPU check of the H200 HYBRID/GEOM pack (tools/modal/h200_hybrid_geom.sh) before its one Modal launch.

1. tools/ab_rewrites.py and tools/force_stress_gate.py parse the pack's exact arguments, and both constants resolve
   (constant_overrides imports the module and checks the type).
2. GEOM: geom_source accepts the STAGE_EDGES forward (captured A100 source, with and without the PREFETCH_AHEAD block,
   the H200 having none) and the result compiles for sm_90 with NVRTC: registers, spills.
3. HYBRID: tf32x3_hyb_sm90.cubin is sm_90 and holds the same entries as the shipped tf32x3 sm_90 cubin.
4. The slim tarball given as argv[1] holds the script, the cubins and the structure.

    python results/h200-hybrid-geom-20261111/pack_check.py SLIM_TAR > pack_check.json
"""
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "bench"),
                str(ROOT / "results" / "census-c345-20261111")]
HYBRID = "fastmace.kernels.cutlass.tf32x3:HYBRID=True"
GEOM = "fastmace.kernels.oeq_streaming.spline_forward:GEOM=True"
ST = "results/loup-a100-20261103/SrTiO3.cif"
result = {}

# 1. arguments
import ab_rewrites  # noqa: E402
import force_stress_gate  # noqa: E402

for order, pattern in (("ab", "abab"), ("ba", "abba")):
    args = ab_rewrites.parse_args([
        "--output", "/tmp/x", "--checkpoint", "/tmp/m", "--model", "omat0-medium", "--structures", f"{ST}@10x10x10",
        "NbCl5@3x3x3", "--timestep-fs", "1", "--a-neighbours", "skin=0.5", "--b-neighbours", "skin=0.5", "--warmup",
        "200", "--block", "256", "--blocks", "6", "--order", order, "--pattern", pattern,
        "--pack", f"hybrid|{HYBRID};geom|{GEOM}"])
entries = args.pack if isinstance(args.pack, list) else ab_rewrites.pack_entries(args.pack)  # parse_args parses it
result["pack"] = [(name, sorted(f"{m}:{n}={v}" for (m, n), v in consts.items())) for name, _, consts in entries]
result["structures"] = [force_stress_gate.structure_spec(s, (3, 3, 3), "/nonexistent")[0:3:2]
                        for s in (f"{ST}@10x10x10",)]
parse = getattr(force_stress_gate, "parse_args", None)
gate_argv = ["--output", "/tmp/g", "--engine", "fastmace", "--model", "omat0-medium", "--checkpoint", "/tmp/m",
             "--structures-root", "/tmp/s", "--structures", f"{ST}@3x3x3", "NaCl", "NbCl5"]
if parse is not None:
    for extra in ([], ["--production-constant", HYBRID], ["--production-constant", GEOM]):
        g = parse(gate_argv + extra)
        assert list(g.production_constant) == extra[1:]
    result["gate_args"] = "parsed"
else:
    result["gate_args"] = "no parse_args: checked through the job file path only"
result["constants"] = {f"{m}:{n}": v for (m, n), v in ab_rewrites.constant_overrides([HYBRID, GEOM]).items()}

# 2. GEOM on the H200's forward form, compiled for sm_90
from census_c345 import analyse, forward_parts  # noqa: E402

from fastmace.kernels.oeq_streaming.spline_forward import geom_source  # noqa: E402

base = (ROOT / "results" / "a100-colab-ideas-20261111" / "screen1" / "ghoist" / "sources" / "real-0.cu").read_text()
pre, post, _, _, body = forward_parts(base, prefetch=False)
h200 = "\n".join(pre + ["        for (int k = 0; k < count; ++k) {"] + body + ["        }"] + post)
result["geom"] = {}
for label, src in (("a100_form", base), ("h200_form", h200)):
    g, ng = geom_source(src)
    result["geom"][label] = {"shared_g_per_edge": ng, "base": analyse(src, "90"), "geom": analyse(g, "90")}
    for side in ("base", "geom"):
        result["geom"][label][side] = {k: result["geom"][label][side][k]
                                       for k in ("registers", "local", "shared", "spill_ops", "loop_len")}

# 3. HYBRID sm_90 cubin
cutlass = ROOT / "fastmace" / "kernels" / "cutlass"


def entries_of(path):
    from cu_census import tool
    text = subprocess.run([tool("cuobjdump"), "-elf", str(path)], capture_output=True, text=True, check=True).stdout
    arch = sorted(set(re.findall(r"arch = (sm_\d+)", text)))
    return arch, sorted(set(re.findall(r"\.text\.(\w+)", text)))


hyb_arch, hyb = entries_of(cutlass / "tf32x3_hyb_sm90.cubin")
ref_arch, ref = entries_of(cutlass / "tf32x3_sm90.cubin")
result["hybrid_cubin"] = {"arch": hyb_arch, "entries": len(hyb), "same_entries_as_3xtf32": hyb == ref}

# 4. slim tarball
if len(sys.argv) > 1:
    names = set(tarfile.open(sys.argv[1]).getnames())
    need = ["tools/modal/h200_hybrid_geom.sh", "fastmace/kernels/cutlass/tf32x3_hyb_sm90.cubin",
            "fastmace/kernels/cutlass/tf32x3_sm90.cubin", ST, "tools/ab_rewrites.py", "tools/force_stress_gate.py",
            ".source_commit"]
    result["tarball"] = {n: any(x == n or x.endswith("/" + n) for x in names) for n in need}
print(json.dumps(result, indent=1, default=str))
