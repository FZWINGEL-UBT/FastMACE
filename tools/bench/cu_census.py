"""Offline SASS census of NVRTC sources (no GPU needed): registers, spills and instruction counts per opcode.

    python tools/bench/cu_census.py --arch 80 A.cu B.cu ...

Compiles each source with NVRTC (cuda-bindings) for sm_<arch>, disassembles with cuobjdump (the CUDA toolkit's, or
Triton's bundled copy) and prints, per file: registers, local bytes, total SASS instructions, and the counts of FFMA,
FMUL, FADD, LDS, LDG, the half2 and conversion opcodes, and the branch-free inner-loop length (the longest basic
block) with its own opcode counts.
"""
import argparse
import collections
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def tool(name):
    found = shutil.which(name)
    if found:
        return found
    for pattern in (f"{sys.prefix}/lib/python3*/site-packages/triton/backends/nvidia/bin/{name}",
                    f"{sys.prefix}/lib/python3*/site-packages/nvidia/cuda_nvcc/bin/{name}",
                    f"{sys.prefix}/lib/python3*/site-packages/nvidia/cu13/bin/{name}"):
        hits = glob.glob(pattern)
        if hits:
            return hits[0]
    raise SystemExit(f"{name} not found")


def compile_cubin(source, arch):
    from cuda.bindings import nvrtc
    program = nvrtc.nvrtcCreateProgram(source.encode(), b"census.cu", 0, [], [])[1]
    options = [f"--gpu-architecture=sm_{arch}".encode(), b"--std=c++17", b"-default-device", b"-w"]
    # the CUDA headers (cuda_fp16.h ...) for sources that include them: the pip toolkit's, else /usr/local/cuda's
    for include in (*glob.glob(f"{sys.prefix}/lib/python3*/site-packages/nvidia/cu13/include"), "/usr/local/cuda/include"):
        if os.path.isdir(include):
            options.append(f"-I{include}".encode())
            break
    result = nvrtc.nvrtcCompileProgram(program, len(options), options)
    if int(result[0]) != 0:
        size = nvrtc.nvrtcGetProgramLogSize(program)[1]
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(program, log)
        raise SystemExit(log.decode(errors="replace")[-3000:])
    size = nvrtc.nvrtcGetCUBINSize(program)[1]
    cubin = b" " * size
    nvrtc.nvrtcGetCUBIN(program, cubin)
    return cubin


def census(path, arch):
    cubin = compile_cubin(Path(path).read_text(), arch)
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "k.cubin")
        Path(f).write_bytes(cubin)
        sass = subprocess.run([tool("cuobjdump"), "-sass", f], capture_output=True, text=True).stdout
        res = subprocess.run([tool("cuobjdump"), "-res-usage", f], capture_output=True, text=True).stdout
    ops = collections.Counter()
    blocks, current, block_ops, current_ops = [], 0, [], collections.Counter()
    for line in sass.splitlines():
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
        if not m:
            if re.match(r"\s*\.L_x_\d+:", line):
                blocks.append(current)
                block_ops.append(current_ops)
                current, current_ops = 0, collections.Counter()
            continue
        op = m.group(2).split(".")[0]
        ops[op] += 1
        current += 1
        current_ops[op] += 1
        if op in ("BRA", "EXIT", "BSYNC", "BAR"):
            blocks.append(current)
            block_ops.append(current_ops)
            current, current_ops = 0, collections.Counter()
    blocks.append(current)
    block_ops.append(current_ops)
    regs = re.search(r"REG:(\d+)", res)
    local = re.search(r"LOCAL:(\d+)", res)
    shared = re.search(r"SHARED:(\d+)", res)
    keys = ("FFMA", "FMUL", "FADD", "LDS", "LDG", "STS", "LOP3", "IMAD", "SHF", "BAR", "F2F", "I2F", "HFMA2", "HMUL2",
            "HADD2", "PRMT", "RED", "F2FP")
    longest = max(range(len(blocks)), key=blocks.__getitem__)
    return dict(file=str(path), arch=arch, registers=int(regs.group(1)) if regs else None,
                local=int(local.group(1)) if local else None, shared=int(shared.group(1)) if shared else None,
                total=sum(ops.values()), longest_block=blocks[longest], **{k: ops.get(k, 0) for k in keys},
                longest_block_ops=dict(block_ops[longest].most_common()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="80")
    p.add_argument("files", nargs="+")
    args = p.parse_args()
    for f in args.files:
        print(census(f, args.arch), flush=True)


if __name__ == "__main__":
    main()
