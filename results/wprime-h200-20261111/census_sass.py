"""Offline SASS census of w' sources for sm_90 (H200 / H100) and sm_80: per loop, opcode counts by category.

    python census_sass.py [--arch 90] [--json OUT] [--sass-dir DIR] A.cu B.cu ...

Compiles with the production options (``runtime.compile_cubin``: NVRTC, --std=c++17 -default-device), disassembles
with cuobjdump and finds the loops (a branch to an earlier label). For the w' kernel the innermost loop that holds
the FFMAs is the channel pass (CHANNEL_DIM / 32 = 4 trips per edge); the loop around it is the edge loop. Per edge
and lane: channel body x trips + the edge loop's own instructions (setup, reduction, stores), predicated-off
instructions included (they still issue).

Categories: fp (FFMA, FMUL, FADD, FMNMX, FSEL? no: FSEL is a move), int (IMAD*, IADD3, LEA, SHF, LOP3, ISETP, ...),
mem (LDG, LDS, STS, STG, RED, ATOM*), shfl, sync (WARPSYNC, BAR, BSSY, BSYNC, ...), move (MOV, SEL, FSEL, PRMT,
S2R/S2UR, ...), conv (F2I, I2F, F2F, ...), branch (BRA, EXIT, ...), uniform (U* opcodes: the uniform datapath),
other.
"""
import argparse
import collections
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "bench"))

FP = {"FFMA", "FMUL", "FADD", "FMNMX", "FSETP", "FCHK", "MUFU", "FSET"}
INT = {"IMAD", "IADD3", "LEA", "SHF", "LOP3", "ISETP", "IABS", "IMNMX", "POPC", "FLO", "IADD", "ISCADD", "LOP",
       "VIADD", "VIMNMX", "IMUL", "PLOP3", "P2R", "R2P", "BMSK", "SGXT", "LEPC", "IDP", "BREV"}
MEM = {"LDG", "LDS", "STS", "STG", "RED", "ATOM", "ATOMS", "ATOMG", "LD", "ST", "LDL", "STL", "LDC", "ULDC", "LDGSTS",
       "LDSM", "REDUX"}
SHFL = {"SHFL"}
SYNC = {"WARPSYNC", "BAR", "BSSY", "BSYNC", "NANOSLEEP", "DEPBAR", "MEMBAR", "ERRBAR", "CCTL", "ENDCOLLECTIVE",
        "VOTE", "VOTEU", "MATCH", "ELECT"}
MOVE = {"MOV", "SEL", "FSEL", "PRMT", "S2R", "S2UR", "CS2R", "R2UR", "UMOV", "SHFL_"}
CONV = {"F2I", "I2F", "F2F", "I2I", "F2FP", "FRND", "I2FP", "F2IP"}
BRANCH = {"BRA", "EXIT", "RET", "CALL", "BRX", "JMP", "YIELD", "BPT", "NOP", "WARPGROUP", "ACQBULK"}


def category(op):
    if op in FP:
        return "fp"
    if op in MEM:
        return "mem"
    if op in SHFL:
        return "shfl"
    if op in SYNC:
        return "sync"
    if op in CONV:
        return "conv"
    if op in BRANCH:
        return "branch"
    if op in MOVE or op == "UMOV":
        return "move"
    if op.startswith("U"):
        return "uniform"
    if op in INT:
        return "int"
    return "other"


def disassemble(source, arch, options=()):
    from cu_census import tool
    from fastmace.kernels.oeq_streaming.runtime import compile_cubin

    cubin = compile_cubin(source, (arch // 10, arch % 10), tuple(options))
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "k.cubin"
        f.write_bytes(cubin)
        sass = subprocess.run([tool("cuobjdump"), "-sass", str(f)], capture_output=True, text=True).stdout
        res = subprocess.run([tool("cuobjdump"), "-res-usage", str(f)], capture_output=True, text=True).stdout
    return sass, res


def parse(sass):
    """[(address, predicate, opcode, text)], {label: address}"""
    rows, labels, pending = [], {}, []
    for line in sass.splitlines():
        lab = re.match(r"\s*\.(L_x_\d+):", line)
        if lab:
            pending.append(lab.group(1))
            continue
        m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)(.*?);", line)
        if not m:
            continue
        address = int(m.group(1), 16)
        for name in pending:
            labels[name] = address
        pending = []
        rows.append((address, (m.group(2) or "").strip(), m.group(3).split(".")[0], line.strip()))
    return rows, labels


def loops(rows, labels):
    """[(start, end)] address ranges of backward branches (start = target label, end = the branch)."""
    out = []
    for address, _, op, text in rows:
        if op == "BRA":
            target = re.search(r"BRA[.\w]*\s+(?:[^,;]+,\s*)?0x([0-9a-f]+)\s*;", text)
            if target and int(target.group(1), 16) <= address:
                out.append((int(target.group(1), 16), address))
    return sorted(set(out))


def count(rows, lo, hi, exclude=()):
    ops = collections.Counter()
    for address, _, op, _ in rows:
        if lo <= address <= hi and not any(a <= address <= b for a, b in exclude):
            ops[op] += 1
    cats = collections.Counter()
    for op, n in ops.items():
        cats[category(op)] += n
    return ops, cats


def census(path, arch=90, options=(), sass_dir=None, trips=4):
    sass, res = disassemble(Path(path).read_text(), arch, options)
    if sass_dir:
        Path(sass_dir).mkdir(parents=True, exist_ok=True)
        (Path(sass_dir) / (Path(path).stem + f".sm{arch}.sass")).write_text(sass)
    rows, labels = parse(sass)
    found = loops(rows, labels)
    regs = re.search(r"REG:(\d+)", res)
    local = re.search(r"LOCAL:(\d+)", res)
    shared = re.search(r"SHARED:(\d+)", res)
    # the channel pass: the loop with the most FFMAs; the edge loop: the innermost loop around it
    stats = []
    for lo, hi in found:
        ops, cats = count(rows, lo, hi)
        stats.append(dict(start=lo, end=hi, total=sum(ops.values()), ffma=ops.get("FFMA", 0)))
    top = max(s["ffma"] for s in stats)
    channel = min((s for s in stats if s["ffma"] >= 0.8 * top), key=lambda s: s["end"] - s["start"])
    around = [s for s in stats if s["start"] <= channel["start"] and s["end"] >= channel["end"] and s is not channel]
    # the edge loop: the enclosing loop that ends first (later ones take in the out-of-line BRA.DIV paths)
    edge = min(around, key=lambda s: s["end"]) if around else None
    ch_ops, ch_cats = count(rows, channel["start"], channel["end"])
    result = dict(file=Path(path).name, arch=arch, options=list(options),
                  registers=int(regs.group(1)) if regs else None, local=int(local.group(1)) if local else None,
                  shared=int(shared.group(1)) if shared else None, total=len(rows),
                  channel=dict(instructions=sum(ch_ops.values()), categories=dict(ch_cats.most_common()),
                               ops=dict(ch_ops.most_common())))
    per_edge_ops = collections.Counter({k: v * trips for k, v in ch_ops.items()})
    if edge:
        e_ops, e_cats = count(rows, edge["start"], edge["end"], exclude=[(channel["start"], channel["end"])])
        result["edge_rest"] = dict(instructions=sum(e_ops.values()), categories=dict(e_cats.most_common()),
                                   ops=dict(e_ops.most_common()))
        per_edge_ops.update(e_ops)
    cats = collections.Counter()
    for op, n in per_edge_ops.items():
        cats[category(op)] += n
    total = sum(per_edge_ops.values())
    fp = sum(per_edge_ops.get(k, 0) for k in ("FFMA", "FMUL", "FADD"))
    result["per_edge"] = dict(trips=trips, instructions=total, ffma_fmul_fadd=fp, non_fma=total - fp,
                              fp_share=round(fp / total, 4) if total else None,
                              categories=dict(cats.most_common()), ops=dict(per_edge_ops.most_common()))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=int, default=90)
    p.add_argument("--json")
    p.add_argument("--sass-dir")
    p.add_argument("--option", action="append", default=[])
    p.add_argument("files", nargs="+")
    args = p.parse_args()
    results = []
    for f in args.files:
        trips = 1 if "vec4" in f else 2 if "vec2" in f else 4  # channel passes per edge (CHANNEL_DIM / 32 / N)
        r = census(f, args.arch, args.option, args.sass_dir, trips)
        results.append(r)
        pe = r["per_edge"]
        print(f"{r['file']} sm_{args.arch}: regs {r['registers']} local {r['local']} | channel pass "
              f"{r['channel']['instructions']} ({r['channel']['categories']}) | edge rest "
              f"{r.get('edge_rest', {}).get('instructions')} ({r.get('edge_rest', {}).get('categories')}) | per edge "
              f"{pe['instructions']} = FP {pe['ffma_fmul_fadd']} + other {pe['non_fma']} (FP {pe['fp_share']})",
              flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
