"""Turn the 23 October A100 shapes.csv (FastMACE gemm_precision_bench) into the CUTLASS bench input.

Writes shapes_in.tsv (one GEMM per row: index, structure, m, k, n, layout/ld/alignment of A and B, alignment
of C, combo name, invocations per call) and k_<combo>.cu + combos.inc (the CUTLASS instantiations needed).
Usage: python prep.py SHAPES_CSV OUTDIR
"""
import ast
import csv
import sys
from pathlib import Path


def layout(rows, cols, stride):
    """('R'|'C', leading dimension, contiguous extent) of a rows x cols operand with torch strides."""
    if stride[1] == 1 and not (stride[0] == 1 and cols != 1):
        return "R", stride[0], cols
    return "C", stride[1], rows


def alignment(ld, extent):
    return next(a for a in (4, 2, 1) if ld % a == 0 and extent % a == 0)


def main(src, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(src, newline="")))
    combos = {}
    lines = []
    for i, r in enumerate(rows):
        m, k, n = int(r["m"]), int(r["k"]), int(r["n"])
        la, lda, ea = layout(m, k, ast.literal_eval(r["a_stride"]))
        lb, ldb, eb = layout(k, n, ast.literal_eval(r["b_stride"]))
        aa, ab, ac = alignment(lda, ea), alignment(ldb, eb), alignment(n, n)
        name = f"{la}{lb}_{aa}_{ab}_{ac}"
        combos[name] = (la, lb, aa, ab, ac)
        lines.append("\t".join(map(str, (i, r["structure"], m, k, n, la, lda, aa, lb, ldb, ab, ac, name,
                                         r["invocations_per_call"]))))
    combos.setdefault("RR_4_4_4", ("R", "R", 4, 4, 4))  # 4096^3 peak
    (out / "shapes_in.tsv").write_text("\n".join(lines) + "\n", newline="\n")
    lay = {"R": "cutlass::layout::RowMajor", "C": "cutlass::layout::ColumnMajor"}
    inc = []
    for name, (la, lb, aa, ab, ac) in sorted(combos.items()):
        for t in range(3):
            (out / f"k_{name}_t{t}.cu").write_text(
                '#include "kernels.cuh"\n'
                f"Runner* make_{name}_t{t}(bool fast) {{\n"
                f"  return make_runner<{lay[la]}, {lay[lb]}, {aa}, {ab}, {ac}, {t}>(fast);\n}}\n", newline="\n")
        inc.append(f"X({name})")
    (out / "combos.inc").write_text("\n".join(inc) + "\n", newline="\n")
    print(len(lines), "shapes,", len(combos), "combos:", " ".join(sorted(combos)))


if __name__ == "__main__":
    main(*sys.argv[1:3])
