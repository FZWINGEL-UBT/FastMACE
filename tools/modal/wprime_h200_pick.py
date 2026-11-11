"""Pick the H200 lab's best w' variant for the MD-step A/B (tools/modal/wprime_h200_lab_ab.sh): prints NAME, KIND, CONSTANTS.

Valid: an exact variant with dsh and dr bitwise; a rounding variant with dsh and dr within 1e-5 of the base (relative
to max |a|). Best: the lowest lab ratio, which must be under 0.99 (else NAME=none).
"""
import json
import re
import sys

MODULE = "fastmace.kernels.oeq_streaming.wprime_backward"
CONSTANTS = {"skip": ["SKIP_OUTSIDE=True"], "hoist": ["HOIST_DX=True"], "vec2": ["CHANNEL_VECTOR=2"],
             "vec4": ["CHANNEL_VECTOR=4"]}


def valid(entry):
    compare = entry.get("compare") or {}
    if entry.get("median_us") is None or entry.get("error"):
        return False
    for key in ("dsh", "dr"):
        text = compare.get(key, "")
        if text == "bitwise":
            continue
        if entry["kind"] == "exact":
            return False
        m = re.search(r"= ([0-9.e+-]+)", text)
        if not m or float(m.group(1)) > 1e-5:
            return False
    return True


def main(path):
    variants = json.load(open(path))["kernels"]["wprime"]["variants"]
    ranked = sorted((v["ratio"], name) for name, v in variants.items()
                    if name.startswith("h2-") and v.get("ratio") is not None and valid(v))
    if not ranked or ranked[0][0] >= 0.99:
        print("NAME=none KIND=none CONSTANTS=")
        return
    ratio, name = ranked[0]
    parts = name[3:].split("+")
    constants = " ".join(f"{MODULE}:{c}" for part in parts for c in CONSTANTS[part])
    print(f"NAME={name[3:].replace('+', '-')} KIND={variants[name]['kind']} RATIO={ratio:.4f} CONSTANTS='{constants}'")


if __name__ == "__main__":
    main(sys.argv[1])
