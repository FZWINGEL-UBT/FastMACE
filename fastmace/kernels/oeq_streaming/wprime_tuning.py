"""Post-render source transforms of the w' kernel (H200 instruction tuning, 12 November 2026;
results/wprime-h200-20261111): ``wprime_backward.HOIST_DX`` and ``wprime_backward.CHANNEL_VECTOR``.

They rewrite the rendered float32 kernel's channel loop (one block per path, ``x[input_base + k + channel]``
reads, one ``atomicAdd`` per input-gradient accumulator, STAGE_DOUT's ``stage_row`` reads) and raise ``ValueError``
where its patterns do not match. The kernel lab's ``h2-*`` variants use the same functions.
"""
import re

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
    """SKIP_OUTSIDE's lab form, as a transform of a rendered source without it (the template's own code; the kernel
    lab's h2-skip variants): an edge at or beyond the cutoff writes its zeros and leaves before the channel loop."""
    marker = "    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};\n"
    source = _sub(source, marker,
                  "    if (!inside) {  // skip: at or beyond the cutoff (padding edges) w = 0: nothing but zeros\n"
                  "        for (int c = lane; c < EDGE_DIM; c += LOGICAL_GROUP_SIZE) dsh[edge_base + c] = scalar_t(0);\n"
                  "        if (lane == 0) dr[e] = scalar_t(0);\n"
                  "        continue;\n"
                  "    }\n" + marker)
    # the table loads' guard (the template's only 'if (inside)'): unconditional
    source, n = re.subn(r"(\n\s*)if \(inside\) \{(\n\s*(?://[^\n]*\n\s*)?const float\d?\* gq)", r"\1if (true) {\2",
                        source)
    if n == 0:
        raise ValueError("no table loads under 'if (inside)'")
    return source


def hoist(source):
    """HOIST_DX: the dx atomics' address as one pointer per lane, stepped per pass (exact)."""
    head, decl, blocks, tail, rest = loop_parts(source)
    head = head + "    scalar_t* dx_lane = dx + input_base + lane;  // hoist\n"
    tail = re.sub(r"dx \+ input_base \+ (\d+) \+ channel", r"dx_lane + \1", tail)
    tail = tail.rstrip() + "\n            dx_lane += LOGICAL_GROUP_SIZE;\n"
    return head + LOOP + decl + "".join(blocks) + tail + "\n    }" + rest


def vec(source, n):
    """CHANNEL_VECTOR = n (2 or 4): n consecutive channels per lane, the staged dout row, x and dx as float<n>
    (vector atomics from sm_90; one by one below). Per channel the same arithmetic; each lane's dsh and dr sums take
    its channels in another order (rounding). STAGE_DOUT's staged row is read as vectors; other dout reads stay
    scalar per channel."""
    if n not in (2, 4):
        raise ValueError(f"channel vector {n!r}: 2 or 4")
    head, decl, blocks, tail, rest = loop_parts(source)
    vt = f"float{n}"
    names = re.findall(r"acc_t (\w+) = acc_t\(0\);", decl)
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
    adds = re.findall(r"atomicAdd\(\s*dx \+ input_base \+ (\d+) \+ channel,\s*scalar_t\((\w+)\)\);", tail)
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


def apply(source, hoist_dx=False, channel_vector=0):
    """``source`` with CHANNEL_VECTOR (which subsumes HOIST_DX: one pass per N channels) or HOIST_DX applied."""
    if channel_vector:
        return vec(source, int(channel_vector))
    return hoist(source) if hoist_dx else source


def variant(source, name):
    """The kernel lab's h2-<name> variant: '+'-joined parts of skip, hoist, vec2, vec4 (e.g. "skip+vec2")."""
    parts = {"skip": skip, "hoist": hoist, "vec2": lambda s: vec(s, 2), "vec4": lambda s: vec(s, 4)}
    for part in name.split("+"):
        source = parts[part](source)
    return source


__all__ = ["apply", "hoist", "skip", "variant", "vec"]
