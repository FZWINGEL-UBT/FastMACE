"""2-T08c: which kernels of stage B differ between FastMACE and the research engine (MH-1 plain cuEq float64).

``test_engine_gpu.py::test_mh1_plain_cueq_float64_against_the_research_engine`` compares the captured
graphs' kernel sequences (``kernel_sequence``). This runs both of its children once, in the
"checked" mode only, and prints for every captured graph the node counts, whether the sequences
match, and a diff of the two sequences (``difflib``: ``-`` FastMACE only, ``+`` research engine only).

Variants (``--variant``), each a FastMACE child with one change, to bisect a difference:

- ``main``: the test's own child;
- ``no-fresh-basis``: ``fastmace.model.loading._fresh_cueq_basis_cache`` replaced by a no-op (2-T14).

Run from a checkout with the environment of the test (``FASTMACE_MODELS``, ``FASTMACE_BASELINE``):

    python tools/validation/stage_b_kernel_diff.py --steps 100 --variant main --variant no-fresh-basis
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests" / "gpu"), str(ROOT / "tests")]
import test_engine_gpu as t  # noqa: E402

VARIANTS = {
    "main": "",
    "no-fresh-basis": ("import contextlib\nfrom fastmace.model import loading as _loading\n"
                       "_loading._fresh_cueq_basis_cache = contextlib.nullcontext\n"),
}


def fastmace(spec, variant):
    child = t.FASTMACE_CHILD.format()
    child = child.replace("import fastmace\n", "import fastmace\n" + VARIANTS[variant], 1)
    return t.run_fresh(t.COMMON.format(spec=spec) + child)["modes"]["checked"]


def compare(ours, theirs):
    out = {}
    for key in sorted(set(ours["kinds"]) | set(theirs["kinds"])):
        a, b = ours["kinds"].get(key) or [], theirs["kinds"].get(key) or []
        sa, sb = t.kernel_sequence(a), t.kernel_sequence(b)
        diff = [line for line in difflib.ndiff(sa, sb) if line[:1] in "+-"]
        out[key] = {"nodes": [len(a), len(b)], "sequence": [len(sa), len(sb)], "equal": sa == sb, "diff": diff}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    spec = t.spec("cueq", modes=[("checked", args.steps)])
    theirs = t.baseline_run(spec)["modes"]["checked"]
    result = {"baseline": {"kinds": theirs["kinds"], "edge_sizes_used": theirs["edge_sizes_used"]}}
    for variant in args.variant or ["main"]:
        ours = fastmace(spec, variant)
        result[variant] = {"kinds": ours["kinds"], "edge_sizes_used": ours["info"]["edge_sizes_used"],
                           "compare": compare(ours, theirs)}
        print(f"== {variant}")
        print(json.dumps(result[variant]["compare"], indent=1))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
