"""Scheduling census of cuobjdump SASS (torch 2.15 trial, 12 November 2026): per loop (backward branch), the
instruction mix and the control bits ptxas wrote into each instruction's upper word (Volta+ layout): stall cycles
(bits 41-44), yield (45), write / read barrier set (46-48 / 49-51), wait mask (52-57), operand reuse (58-61).
The static stall sum is the issue cycles one warp needs for a pass with no other warp to hide them.

    python sass_sched.py A.sass [B.sass ...]
"""
import collections
import re
import sys

LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?);\s+/\* 0x([0-9a-f]{16}) \*/\s*\n\s*/\* 0x([0-9a-f]{16}) \*/")


def parse(text):
    rows = []
    for m in LINE.finditer(text):
        addr, ins, hi = int(m.group(1), 16), m.group(2).strip(), int(m.group(4), 16)
        op = re.sub(r"^@!?U?P\w+\s+", "", ins).split()[0] if ins else "?"
        rows.append(dict(addr=addr, ins=ins, op=op, stall=(hi >> 41) & 15, yld=(hi >> 45) & 1, wbar=(hi >> 46) & 7,
                         rbar=(hi >> 49) & 7, wait=(hi >> 52) & 63, reuse=(hi >> 58) & 15))
    return rows


def loops(rows):
    out = []
    for r in rows:
        t = re.search(r"BRA[.\w]*\s+(?:[^,;]+,\s*)?(?:`\(\.L_x_\d+\)\s*)?0x([0-9a-f]+)", r["ins"]) or \
            re.search(r"BRA[.\w]*\s+`?\(?\.?L?_?x?_?\d*\)?\s*0x([0-9a-f]+)", r["ins"])
        if r["op"].startswith("BRA") and t and int(t.group(1), 16) < r["addr"]:
            out.append((int(t.group(1), 16), r["addr"]))
    return out


def summary(rows):
    ops = collections.Counter(r["op"].split(".")[0] for r in rows)
    mem = sum(ops[k] for k in ("LDG", "LDS", "STS", "STG", "LDL", "STL", "RED", "ATOMS", "LDSM", "SHFL"))
    return dict(n=len(rows), stall=sum(r["stall"] for r in rows), yields=sum(r["yld"] for r in rows),
                waits=sum(1 for r in rows if r["wait"]), reuse=sum(bin(r["reuse"]).count("1") for r in rows),
                ffma=ops["FFMA"], fmul=ops["FMUL"], fadd=ops["FADD"], mem=mem, lds=ops["LDS"], ldg=ops["LDG"],
                shfl=ops["SHFL"], mov=ops["MOV"] + ops["IMAD"] if False else ops["MOV"], imad=ops["IMAD"],
                iadd=ops["IADD3"], other=len(rows) - ops["FFMA"] - ops["FMUL"] - ops["FADD"] - mem - ops["MOV"]
                - ops["IMAD"] - ops["IADD3"])


for path in sys.argv[1:]:
    rows = parse(open(path).read())
    print(f"== {path.split('/')[-2]}/{path.split('/')[-1][-14:]}  total {len(rows)} instr, stall sum {sum(r['stall'] for r in rows)}")
    ls = sorted(loops(rows), key=lambda lh: -(lh[1] - lh[0]))
    for lo, hi in ls[:6]:
        body = [r for r in rows if lo <= r["addr"] <= hi]
        s = summary(body)
        print(f"   loop 0x{lo:05x}-0x{hi:05x}", " ".join(f"{k}={v}" for k, v in s.items()))


def load_use(body):
    """Per load in a loop body: the stall cycles ptxas scheduled between the load and the first instruction reading
    its destination register (wrapping around the loop once). Short distances leave the load's latency exposed."""
    out = collections.defaultdict(list)
    n = len(body)
    for i, r in enumerate(body):
        if r["op"].split(".")[0] not in ("LDS", "LDG", "LDSM", "SHFL"):
            continue
        dst = re.match(r"\S+\s+(R\d+)", re.sub(r"^@!?U?P\w+\s+", "", r["ins"]))
        if not dst:
            continue
        reg = int(dst.group(1)[1:])
        width = {"64": 2, "128": 4}.get((re.search(r"\.(64|128)", r["op"]) or [None, "32"])[1], 1)
        regs = {f"R{reg + k}" for k in range(width)}
        cyc = 0
        for j in range(1, n):
            q = body[(i + j) % n]
            cyc += body[(i + j - 1) % n]["stall"]
            srcs = set(re.findall(r"R\d+", q["ins"].split(",", 1)[1] if "," in q["ins"] else ""))
            if regs & srcs:
                out[r["op"].split(".")[0]].append(cyc)
                break
    return {k: (len(v), round(sum(v) / len(v), 1), sorted(v)[:6]) for k, v in out.items()}


if __name__ == "__main__" and "--load-use" in sys.argv:
    pass
