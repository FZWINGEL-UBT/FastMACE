"""Source variants of today's H200 w' kernel (sources/wprime-h200.cu, render_sources.py) for the sm_90 census.

  skip      exact: an edge at or beyond the cutoff (inside = false: the engine's padding edges, since the skin list
            is compacted to the cutoff on the device) writes dsh = 0 and dr = 0 and skips the channel loop; the
            table loads in the loop are then unconditional (no zeroed gv registers, no predicate).
  hoist     exact: the dx atomics' address (dx + input_base + channel) as one pointer per lane, stepped per pass.
  vecN      rounding (N = 2, 4): each lane owns N consecutive channels; the staged dout row, x and dx are read and
            added as floatN (LDS.64/128, LDG.64/128, vector RED on sm_90); per channel the arithmetic is unchanged,
            but each lane's edge-gradient and radial-gradient sums take its channels in a new order.

Combinations join with '+', e.g. ``skip+vec4``. ``python variants.py`` writes sources/wprime-h200-<name>.cu.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "sources" / "wprime-h200.cu"
LOOP = ("    for (int channel = lane; channel < CHANNEL_DIM;\n"
        "         channel += LOGICAL_GROUP_SIZE) {\n")
COMP = "xyzw"


def _sub(source, old, new, count=1):
    found = source.count(old)
    if found != count:
        raise ValueError(f"pattern found {found} times (want {count}): {old[:80]!r}")
    return source.replace(old, new)


def _block(text, start):
    """The balanced {...} block whose '{' is at or after ``start``: (open index, end index past '}')."""
    open_ = text.index("{", start)
    depth = 0
    for i in range(open_, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return open_, i + 1
    raise ValueError("unbalanced block")


def loop_parts(source):
    """(head, declarations, path blocks, tail, rest) of the channel loop."""
    begin = source.index(LOOP)
    open_, end = _block(source, begin + len(LOOP) - 2)
    body = source[open_ + 1:end - 1]
    first = re.search(r"\n\s*\{\s*\n\s*const int weight_index =", body).start()
    decl, rest = body[:first], body[first:]
    blocks, pos = [], 0
    while True:
        m = re.search(r"\{\s*\n\s*const int weight_index =", rest[pos:])
        if not m:
            break
        a, b = _block(rest, pos + m.start())
        blocks.append(rest[a:b])
        pos = b
    tail = rest[pos:]
    return source[:begin], decl, blocks, tail, source[end:]


def skip(source):
    marker = "    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};\n"
    source = _sub(source, marker,
                  "    if (!inside) {  // skip: at or beyond the cutoff (padding edges) w = 0: nothing but zeros\n"
                  "        for (int c = lane; c < EDGE_DIM; c += LOGICAL_GROUP_SIZE) dsh[edge_base + c] = scalar_t(0);\n"
                  "        if (lane == 0) dr[e] = scalar_t(0);\n"
                  "        continue;\n"
                  "    }\n" + marker)
    n = source.count("                if (inside) {\n                    const float4* gq =")
    source = _sub(source, "                if (inside) {\n                    const float4* gq =",
                  "                if (true) {\n                    const float4* gq =", n)
    return source


def hoist(source):
    head, decl, blocks, tail, rest = loop_parts(source)
    head = head + "    scalar_t* dx_lane = dx + input_base + lane;  // hoist\n"
    tail = re.sub(r"dx \+ input_base \+ (\d+) \+ channel", r"dx_lane + \1", tail)
    tail = tail.rstrip() + "\n            dx_lane += LOGICAL_GROUP_SIZE;\n"
    return head + LOOP + decl + "".join(blocks) + tail + "\n    }" + rest


def vec(source, n):
    head, decl, blocks, tail, rest = loop_parts(source)
    vt = f"float{n}"
    names = re.findall(r"acc_t (input_gradient_\w+) = acc_t\(0\);", decl)
    xs = sorted({int(o) for b in blocks for o in re.findall(r"x\[input_base\s*\+\s*(\d+)\s*\+\s*channel\]", b)})
    out = [f"    for (int cbase = lane * {n}; cbase < CHANNEL_DIM; cbase += LOGICAL_GROUP_SIZE * {n}) {{  // vec{n}\n"]
    for name in names:
        out.append("".join(f"            acc_t {name}_{k} = acc_t(0);\n" for k in range(n)))
    for o in xs:
        out.append(f"            const {vt} xv_{o} = *reinterpret_cast<const {vt}*>(x + input_base + {o} + cbase);\n")
    for b in blocks:
        rows = []
        for m in re.finditer(r"stage_row\[(\d+)\s*\+\s*(\d+)\s*\+\s*channel\]", b):
            key = (int(m.group(1)), int(m.group(2)))
            if key not in rows:
                rows.append(key)
        out.append("            {\n")
        for i, (a, c) in enumerate(rows):
            out.append(f"            const {vt} dv_{i} = *reinterpret_cast<const {vt}*>(stage_row + {a} + {c} + cbase);\n")
        for k in range(n):
            text = b
            text = re.sub(r"stage_row\[(\d+)\s*\+\s*(\d+)\s*\+\s*channel\]",
                          lambda m: f"dv_{rows.index((int(m.group(1)), int(m.group(2))))}.{COMP[k]}", text)
            text = re.sub(r"x\[input_base\s*\+\s*(\d+)\s*\+\s*channel\]", lambda m: f"xv_{m.group(1)}.{COMP[k]}", text)
            for name in names:
                text = re.sub(rf"\b{name}\b", f"{name}_{k}", text)
            text = re.sub(r"\bchannel\b", f"(cbase + {k})", text)
            out.append(text + "\n")
        out.append("            }\n")
    adds = re.findall(r"atomicAdd\(\s*dx \+ input_base \+ (\d+) \+ channel,\s*scalar_t\((input_gradient_\w+)\)\);", tail)
    if len(adds) != len(names):
        raise ValueError(f"atomics {len(adds)} against accumulators {len(names)}")
    for o, name in adds:
        values = ", ".join(f"{name}_{k}" for k in range(n))
        scalars = "".join(f"            atomicAdd(dx + input_base + {o} + cbase + {k}, {name}_{k});\n" for k in range(n))
        # vector atomicAdd (float2 / float4) exists from sm_90; older GPUs add the N values one by one
        out.append("#if __CUDA_ARCH__ >= 900\n"
                   f"            atomicAdd(reinterpret_cast<{vt}*>(dx + input_base + {o} + cbase), make_{vt}({values}));\n"
                   "#else\n" + scalars + "#endif\n")
    return head + "".join(out) + "    }" + rest


TRANSFORMS = {"skip": skip, "hoist": hoist, "vec2": lambda s: vec(s, 2), "vec4": lambda s: vec(s, 4)}
VARIANTS = ["skip", "hoist", "skip+hoist", "vec2", "vec4", "skip+vec2", "skip+vec4"]


def build(name, source=None):
    source = BASE.read_text() if source is None else source
    for part in name.split("+"):
        source = TRANSFORMS[part](source)
    return source


def main(names=None):
    for name in names or VARIANTS:
        (HERE / "sources" / f"wprime-h200-{name}.cu").write_text(build(name))
        print("wrote", name)


if __name__ == "__main__":
    main(sys.argv[1:])
