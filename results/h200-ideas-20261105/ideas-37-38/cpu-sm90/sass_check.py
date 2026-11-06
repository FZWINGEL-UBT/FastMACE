"""CPU-only sm_90 build of the idea 37/38 lab variants: registers, stack, shared memory, SASS identity and counts.
Run in WSL: ~/environments/mace_cueq_oeq/bin/python -I -B sass_check.py RENDER OUTDIR"""
import collections
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
import lab_ideas3738 as lab  # noqa: E402
from cuda.bindings import nvrtc  # noqa: E402

CUOBJDUMP = "/home/fzwin/environments/mace_cueq_oeq/lib/python3.10/site-packages/triton/backends/nvidia/bin/cuobjdump"


def check(result):
    if int(result[0]) != 0:
        raise RuntimeError(result)
    return result[1] if len(result) > 1 else None


def build(source, extra):
    program = check(nvrtc.nvrtcCreateProgram(source.encode(), b"oeq_streaming.cu", 0, [], []))
    options = [b"--gpu-architecture=sm_90", b"--std=c++17", b"-default-device"] + [o.encode() for o in extra]
    result = nvrtc.nvrtcCompileProgram(program, len(options), options)
    if int(result[0]) != 0:
        size = check(nvrtc.nvrtcGetProgramLogSize(program))
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(program, log)
        raise RuntimeError(log.decode(errors="replace")[-3000:])
    size = check(nvrtc.nvrtcGetCUBINSize(program))
    cubin = b" " * size
    check(nvrtc.nvrtcGetCUBIN(program, cubin))
    return cubin


render = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
table = lab.ideas3738_variants()
variants = {"base": (render, ()), "maxreg72": (render, ("--maxrregcount=72",)),
            "old-FF-dout-one-row": (render.replace("const int64_t output_base = int64_t(receiver) * OUTPUT_DIM;",
                                                   "const int64_t output_base = 0;"), ())}
for name, (kind, transform, extra, *_) in table.items():
    variants[name] = (transform(render), extra)
results = {}
for name, (source, extra) in variants.items():
    cubin = build(source, extra)
    (out / f"{name}.cubin").write_bytes(cubin)
    sass = subprocess.check_output([CUOBJDUMP, "-sass", str(out / f"{name}.cubin")], text=True)
    res = subprocess.check_output([CUOBJDUMP, "-res-usage", str(out / f"{name}.cubin")], text=True)
    (out / f"{name}.sass").write_text(sass)
    inst = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9.]+)\s", sass)
    ops = collections.Counter(x.split(".")[0] for x in inst)
    m = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", res)
    body = "\n".join(line.split("/*", 2)[-1] if False else re.sub(r"/\*[0-9a-f]{4,}\*/", "", line)
                     for line in sass.splitlines() if "Function" not in line and ".target" not in line)
    results[name] = dict(extra=list(extra), reg=int(m[1]), stack=int(m[2]), shared=int(m[3]), local=int(m[4]),
                         instructions=len(inst), LDG=ops["LDG"], STL=ops["STL"], LDL=ops["LDL"], LDS=ops["LDS"],
                         STS=ops["STS"], REDG=ops["REDG"], BAR=ops["BAR"], SHFL=ops["SHFL"],
                         source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                         sass_sha256=hashlib.sha256(body.encode()).hexdigest())
    print(name, json.dumps(results[name]), flush=True)
for a, b in (("base", "old-FF-dout-one-row"), ("base", "FF-dout-row0"), ("maxreg72", "FF-dout-row0")):
    print(f"SASS identical {a} vs {b}: {results[a]['sass_sha256'] == results[b]['sass_sha256']}")
# dout addressing in the falsifier: how many LDGs use a receiver-dependent address. Count the IMAD.WIDE/LEA chains is
# fragile; instead diff the instruction streams.
sa = (out / "maxreg72.sass").read_text().splitlines()
sb = (out / "FF-dout-row0.sass").read_text().splitlines()
import difflib  # noqa: E402
diff = [d for d in difflib.unified_diff([re.sub(r"/\*[0-9a-f]{4,}\*/", "", x) for x in sa],
                                        [re.sub(r"/\*[0-9a-f]{4,}\*/", "", x) for x in sb], lineterm="", n=0)
        if d[:1] in "+-" and d[:3] not in ("+++", "---")]
print("differing SASS lines maxreg72 vs FF-dout-row0:", len(diff))
(out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
