"""fp32 idea 4 check: how much of a compiled graph's kernel time lies off its critical path.

Reads Inductor's generated wrapper code (``output_code.py`` files under a ``TORCH_COMPILE_DEBUG`` directory) and a
``tools/step_breakdown.py`` run's ``ops.csv`` (eager kernel time per operator and input shapes). Every launch in the
wrapper's ``call`` (a Triton kernel's ``.run``, an ``extern_kernels`` call, a ``torch.ops`` custom op) becomes a
node; a node depends on the last writer of every buffer it reads, and on the last readers of every buffer it
writes. Triton arguments are classified by the kernel's parameter names (``in_ptr``, ``out_ptr``, ``in_out_ptr``);
an extern call writes its ``out=`` buffer; a custom op writes its result and, conservatively, every buffer it is
passed. Aliases (``reinterpret_tensor``, ``as_strided``, indexing a result tuple) map to their base buffer.

Node times come from the same run's ``kernel_sequence.csv`` (one force call's kernels in launch order) joined with
``ops.csv`` (time per kernel, operator and input shapes): the wrapper's launches are walked alongside the sequence,
each launch takes the sequence rows of its operator (a Triton kernel's name, ``aten::mm`` for an extern product, a
custom op's name), and rows in between go to the custom op before them (its inner kernels). Prints, per graph, the
summed kernel time, the critical path (the longest dependency chain) and what is off it by category, which bounds
what running independent branches on a second stream could hide (ignoring that concurrent kernels share the GPU).
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

CALL = re.compile(r"^\s+(?:(?P<lhs>[\w, ]+?)\s*=\s*)?(?P<callee>[\w.]+)\((?P<args>.*)\)\s*$")
TRITON_DEF = re.compile(r"def (triton_\w+)\((?P<params>[^)]*)\)")
NAME = re.compile(r"\b(buf\d+|arg\d+_\d+|primals_\d+|tangents_\d+|getitem_\d+|\w+_\d+)\b")
ALIAS = re.compile(r"^\s+(?P<lhs>buf\d+|\w+)\s*=\s*(?:reinterpret_tensor|torch\.as_strided)?\(?(?P<base>buf\d+)\b")
TUPLE_ITEM = re.compile(r"^\s+(?P<lhs>buf\d+)\s*=\s*(?P<base>buf\d+)\[\d+\]\s*$")


def triton_params(text):
    """{kernel name: [parameter names]} from the triton kernel definitions in the code."""
    return {m.group(1): [p.split(":")[0].strip() for p in m.group("params").split(",")]
            for m in TRITON_DEF.finditer(text)}


def sequence_times(run_dir):
    """[(op, kernel, us)] of one force call in launch order (kernel_sequence.csv joined with ops.csv)."""
    per = {}
    for row in csv.DictReader(open(run_dir / "ops.csv")):
        launches = float(row.get("launches_per_call") or 1) or 1.0
        per[(row["op"], row["kernel"], row["input_dims"])] = float(row["us_per_call"]) / launches
    return [(row["op"], row["kernel"], per.get((row["op"], row["kernel"], row["input_dims"]), 0.0))
            for row in csv.DictReader(open(run_dir / "kernel_sequence.csv"))]


def _expected(label):
    if label.startswith("extern_kernels."):
        return "aten::" + label.split(".", 1)[1]
    return label


def assign_times(nodes, sequence, start):
    """Per node the time of its sequence rows, starting the walk at ``start``; returns (times, next index)."""
    times, p = [], start
    for label, _, _ in nodes:
        want = _expected(label)
        q = p
        while q < len(sequence) and not sequence[q][0].startswith(want):
            q += 1
        if q == len(sequence):  # not found: no time, the walk stays
            times.append(0.0)
            continue
        if times and q > p:  # rows skipped belong to the previous launch (a custom op's inner kernels)
            times[-1] += sum(t for _, _, t in sequence[p:q])
        times.append(sequence[q][2])
        p = q + 1
    return times, p


def parse_call(text, params, prefix="", rename=None):
    """Nodes [(label, reads, writes)] of the wrapper's ``def call``, buffer names prefixed with ``prefix`` (``rename``
    maps names first, the backward's saved inputs to the forward's buffers)."""
    body = text.split("def call(", 1)[-1]
    alias, nodes = {}, []

    def base(name):
        while name in alias:
            name = alias[name]
        if rename and name in rename:
            return rename[name]
        return name if name.startswith("primals_") else prefix + name

    for line in body.splitlines():
        if line.startswith("def ") or line.startswith("class "):
            break
        # Inductor's memory reuse (``bufX = bufY  # reuse``) is kept as an alias: a reused buffer can be updated in
        # place (``in_out_ptr``), so the old contents may matter. Conservative: it can only lengthen the path.
        item = TUPLE_ITEM.match(line)
        if item:
            alias[item.group("lhs")] = item.group("base")
            continue
        m = CALL.match(line)
        if not m:
            continue
        callee, args, lhs = m.group("callee"), m.group("args"), m.group("lhs")
        names = [base(n) for n in NAME.findall(args)]
        if callee.endswith(".run") and callee[:-4] in params:
            kernel = callee[:-4]
            positional = [a.strip() for a in re.split(r",(?![^(]*\))", args) if "=" not in a]
            reads, writes = set(), set()
            for param, arg in zip(params[kernel], positional):
                found = NAME.findall(arg)
                if not found:
                    continue
                buffer = base(found[0])
                if param.startswith("in_out_ptr"):
                    reads.add(buffer)
                    writes.add(buffer)
                elif param.startswith("out_ptr"):
                    writes.add(buffer)
                elif param.startswith("in_ptr"):
                    reads.add(buffer)
            nodes.append((kernel, reads, writes))
        elif callee.startswith("extern_kernels."):
            out = re.search(r"out=\s*(?:reinterpret_tensor\()?(buf\d+)", args)
            writes = {base(out.group(1))} if out else set()
            nodes.append((callee, set(names) - writes, writes))
        elif callee.startswith("torch.ops."):
            op = callee[len("torch.ops."):].rsplit(".", 1)[0].replace(".", "::", 1)
            writes = {base(n.strip()) for n in (lhs or "").split(",") if n.strip()} | set(names)
            nodes.append((op, set(names), writes))
        elif lhs and callee in ("reinterpret_tensor", "torch.as_strided"):
            first = NAME.findall(args)
            if first:
                alias[lhs.strip()] = first[0]
    return nodes


def _items(text):
    """Top-level comma-separated items of a parenthesised list."""
    depth, item, items = 0, "", []
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            items.append(item.strip())
            item = ""
        else:
            item += char
    if item.strip():
        items.append(item.strip())
    return items


def joint_rename(forward_text, backward_text):
    """{backward saved-input name: forward buffer} from the forward's return list and the backward's arguments."""
    body = forward_text.split("def call(", 1)[-1]
    returned = _items(body[body.rindex("return (") + len("return ("):].splitlines()[0].rstrip().rstrip(")"))
    forward_alias = {}
    for line in body.splitlines():
        m = re.match(r"^\s+(buf\d+)\s*=\s*(?:reinterpret_tensor\()?(buf\d+)(?:\[\d+\])?", line)
        if m:
            forward_alias[m.group(1)] = m.group(2)

    def resolve(name):
        while name in forward_alias:
            name = forward_alias[name]
        return name

    saved = []
    for item in returned:
        found = NAME.findall(item)
        if not found or re.fullmatch(r"\d+|s\d+|primals_\d+", item):
            continue
        saved.append("f:" + resolve(found[0]) if not found[0].startswith("primals_") else found[0])
    head = backward_text.split("def call(", 1)[-1].splitlines()[1]
    arguments = [a.strip() for a in head.split("=")[0].split(",") if a.strip()]
    inputs = [a for a in arguments if not a.startswith(("primals_", "tangents_"))]
    if len(inputs) > len(saved):
        raise SystemExit(f"cannot link: {len(inputs)} saved inputs, {len(saved)} forward items")
    return dict(zip(inputs, saved[len(saved) - len(inputs):])), len(inputs)


def critical_path(nodes, times):
    """(total, critical path, [(label, time, on_path)])."""
    last_writer, readers = {}, defaultdict(list)
    finish, preds = [], []
    for i, (label, reads, writes) in enumerate(nodes):
        deps = {last_writer[b] for b in reads if b in last_writer}
        for b in writes:
            deps |= set(readers[b])
            if b in last_writer:
                deps.add(last_writer[b])
        start = max((finish[d] for d in deps), default=0.0)
        finish.append(start + times[i])
        preds.append(max(deps, key=lambda d: finish[d]) if deps else None)
        for b in reads:
            readers[b].append(i)
        for b in writes:
            last_writer[b] = i
            readers[b] = []
    end = max(range(len(nodes)), key=lambda i: finish[i]) if nodes else None
    on_path = set()
    while end is not None:
        on_path.add(end)
        end = preds[end]
    total = sum(times)
    path = sum(times[i] for i in on_path)
    return total, path, [(nodes[i][0], times[i], i in on_path) for i in range(len(nodes))]


def category(label):
    if label.startswith("extern_kernels") or "joined_products" in label or "segment_gemm" in label:
        return "GEMM"
    if label.startswith("triton_"):
        return "Triton pointwise/reduction"
    return f"custom op {label}"


def report(name, nodes, times):
    total, path, rows = critical_path(nodes, times)
    off = defaultdict(float)
    for label, t, on in rows:
        if not on:
            off[category(label)] += t
    print(f"{name}: {len(nodes)} launches ({sum(1 for t in times if t == 0)} without a time), "
          f"kernel sum {total / 1e3:.2f} ms, critical path {path / 1e3:.2f} ms, off the path "
          f"{(total - path) / 1e3:.2f} ms ({(total - path) / max(total, 1e-9):.0%})")
    for kind, t in sorted(off.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    off path: {kind}: {t / 1e3:.3f} ms")
    for label, t, _ in sorted((r for r in rows if not r[2]), key=lambda r: -r[1])[:8]:
        print(f"      {t:8.1f} us  {label[:90]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("debug_dir", type=Path)
    p.add_argument("run_dir", type=Path, help="a tools/step_breakdown.py output directory")
    p.add_argument("--forward", default="model__0_forward", help="the forward graph's directory name (prefix)")
    p.add_argument("--backward", default="model__0_backward", help="the backward graph's directory name (prefix)")
    args = p.parse_args()
    sequence = sequence_times(args.run_dir)
    print(f"profile: {len(sequence)} kernels, {sum(t for _, _, t in sequence) / 1e3:.2f} ms per force call")
    codes = {c.parent.name: c for c in args.debug_dir.rglob("output_code.py")}
    forward = next(c for n, c in sorted(codes.items()) if n.startswith(args.forward)).read_text()
    backward = next(c for n, c in sorted(codes.items()) if n.startswith(args.backward)).read_text()
    f_nodes = parse_call(forward, triton_params(forward), prefix="f:")
    rename, linked = joint_rename(forward, backward)
    b_nodes = parse_call(backward, triton_params(backward), prefix="b:", rename=rename)
    f_times, position = assign_times(f_nodes, sequence, 0)
    b_times, position = assign_times(b_nodes, sequence, position)
    print(f"linked {linked} saved values; sequence rows used {position} of {len(sequence)}")
    report("forward alone", f_nodes, f_times)
    report("backward alone", b_nodes, b_times)
    report("forward + backward (backward starts as soon as its saved values exist)", f_nodes + b_nodes,
           f_times + b_times)


if __name__ == "__main__":
    main()
