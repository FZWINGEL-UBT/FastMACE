"""Idea 13 step 1 (plans/ideas-20261111-tiled-round2.md): offline SASS census of the w' backward with the edge force
contracted in its epilogue, against today's kernel (no GPU).

    python tools/bench/sh_recompute_census.py SOURCE.cu OUT_DIR [--arch 80 89 120]

SOURCE is a rendered production w' backward with the BUTTERFLY epilogue (lane c << SHIFT ends with dsh[e, c]). The
variants replace its dsh store by the vector gradient of the spherical harmonics, d L / d v[e] (12 B per edge instead
of dsh's 64), with the formulas of ``fastmace/kernels/triton/edge_sh.py``'s backward:

- ``gather``: 16 shuffles bring the 16 components to every lane; one lane contracts them (about 70 operations, the
  other lanes idle) and writes dv[e, 0:3].
- ``split``: each component's lane computes its own share of dv from the monomials of the unit vector (each Y_c is a
  homogeneous polynomial of degree l, so dY_c/du is a combination of 10 monomials of degree <= 2 with the lane's
  coefficients, read from a shared-memory table); a 3-value xor reduction over the 16 lanes; lane 0 writes dv.

Prints, per architecture and file: registers, spills, SASS instructions in all and in the longest basic block, and the
per-edge epilogue's instruction count (instructions between the last dx atomic and the loop's back edge).
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

S3, S5, S56, S38 = math.sqrt(3.0), math.sqrt(5.0), math.sqrt(5.0 / 6.0), math.sqrt(3.0 / 8.0)
C1, C2, C3 = math.sqrt(3.0), math.sqrt(5.0), math.sqrt(7.0)

STORE = "    if ((lane & ((1 << SHIFT) - 1)) == 0) dsh[edge_base + (lane >> SHIFT)] = scalar_t(edge_gradient[0]);\n"
SIGNATURE_END = "    scalar_t* dr) {"


def f(x):
    text = f"{x:.9g}"
    return (text if any(ch in text for ch in ".e") else text + ".0") + "f"


def contraction(g):
    """The edge_sh backward's gx, gy, gz for the unit vector (x, y, z), components g(c) already loaded."""
    return f"""
            const float g1x = {g(1)} * {f(C1)}, g1y = {g(2)} * {f(C1)}, g1z = {g(3)} * {f(C1)};
            const float g20 = {g(4)} * {f(C2)}, g21 = {g(5)} * {f(C2)}, g22 = {g(6)} * {f(C2)};
            const float g23 = {g(7)} * {f(C2)}, g24 = {g(8)} * {f(C2)};
            const float g30 = {g(9)} * {f(C3)}, g31 = {g(10)} * {f(C3)}, g32 = {g(11)} * {f(C3)};
            const float g33 = {g(12)} * {f(C3)}, g34 = {g(13)} * {f(C3)}, g35 = {g(14)} * {f(C3)};
            const float g36 = {g(15)} * {f(C3)};
            const float s20 = {f(S3)} * x * z, s24 = {f(S3 / 2)} * (z * z - x * x);
            const float y2 = y * y, x2z2 = x * x + z * z;
            const float a = 4.0f * y2 - x2z2, b = 2.0f * y2 - 3.0f * x2z2;
            const float gs20 = g20 + {f(S56)} * (g30 * z - g36 * x) + {f(S5)} * g31 * y;
            const float gs24 = g24 + {f(S56)} * (g30 * x + g36 * z) + {f(S5)} * g35 * y;
            const float ga = {f(S38)} * (g32 * x + g34 * z), gb = 0.5f * y * g33;
            const float gy2 = g22 + 4.0f * ga + 2.0f * gb, gx2z2 = -0.5f * g22 - ga - 3.0f * gb;
            gx = g1x + {f(S3)} * z * gs20 + {f(S3)} * y * g21 - {f(S3)} * x * gs24 + 2.0f * x * gx2z2
                 + {f(S56)} * (g30 * s24 - g36 * s20) + {f(S38)} * a * g32;
            gy = g1y + {f(S3)} * x * g21 + {f(S3)} * z * g23 + 2.0f * y * gy2 + {f(S5)} * (g31 * s20 + g35 * s24)
                 + 0.5f * b * g33;
            gz = g1z + {f(S3)} * x * gs20 + {f(S3)} * y * g23 + {f(S3)} * z * gs24 + 2.0f * z * gx2z2
                 + {f(S56)} * (g30 * s20 + g36 * s24) + {f(S38)} * a * g34;"""


UNIT = """
        const float vx = __ldg(vec + 3 * e), vy = __ldg(vec + 3 * e + 1), vz = __ldg(vec + 3 * e + 2);
        const float n = sqrtf(vx * vx + vy * vy + vz * vz), d = fmaxf(n, 1e-12f), inv = 1.0f / d;
        const float x = vx * inv, y = vy * inv, z = vz * inv;"""

PROJECT_STORE = """
        if (lane == 0) {
            const float dot = n > 1e-12f ? x * gx + y * gy + z * gz : 0.0f;
            dsh[3 * e + 0] = scalar_t((gx - x * dot) * inv);
            dsh[3 * e + 1] = scalar_t((gy - y * dot) * inv);
            dsh[3 * e + 2] = scalar_t((gz - z * dot) * inv);
        }
"""


def gather():
    return ("    {  // idea 13 (gather): every lane gets the 16 components, lane 0 contracts them\n"
            "        float gc[EDGE_DIM];\n        #pragma unroll\n"
            "        for (int c = 0; c < EDGE_DIM; ++c)\n"
            "            gc[c] = __shfl_sync(FULL_MASK, edge_gradient[0], c << SHIFT, 32);\n"
            "        float gx = 0.0f, gy = 0.0f, gz = 0.0f;"
            + UNIT + "\n        if (lane == 0) {" + contraction(lambda c: f"gc[{c}]") + "\n        }"
            + PROJECT_STORE + "    }\n")


MONOMIALS = ("1", "x", "y", "z", "x * x", "x * y", "x * z", "y * y", "y * z", "z * z")


def gradient_table():
    """[16][3][10]: dY_c/du_k as coefficients of MONOMIALS, from the edge_sh backward with one component set."""
    import sympy

    x, y, z = sympy.symbols("x y z")
    names = {m: sympy.sympify(m.replace(" ", "")) for m in MONOMIALS}
    table = []
    for c in range(16):
        src = contraction(lambda k: "1.0f" if k == c else "0.0f")
        local = {"x": x, "y": y, "z": z}
        for line in src.replace("\n                 ", " ").splitlines():
            line = line.strip().removeprefix("const float ").rstrip(";")
            if not line:
                continue
            for assignment in re.split(r",\s(?=[a-z0-9]+ =)", line):
                name, value = assignment.split(" = ", 1)
                value = re.sub(r"(\d+\.?\d*(?:e-?\d+)?)f\b", r"\1", value)
                local[name.strip()] = sympy.sympify(value, locals=local)
        exprs = [sympy.expand(local[k]) for k in ("gx", "gy", "gz")]
        row = []
        for e in exprs:
            poly = sympy.Poly(e, x, y, z)
            coeffs = [float(poly.coeff_monomial(names[m])) for m in MONOMIALS]
            assert sympy.simplify(sum(cf * names[m] for cf, m in zip(coeffs, MONOMIALS)) - e) == 0
            row.append(coeffs)
        table.append(row)
    return table


def split():
    table = gradient_table()
    flat = ", ".join(f(v) for row in table for k in row for v in k)
    head = (f"__device__ __constant__ float SH_GRAD[16 * 30] = {{{flat}}};\n")
    body = ("    {  // idea 13 (split): component c's lane adds g_c dY_c/du; a 3-value xor reduction, 16 lanes\n"
            + UNIT + "\n"
            "        const float m[10] = {1.0f, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z};\n"
            "        const float* t = sh_grad + (lane >> SHIFT) * 30;\n"
            "        const float g = (lane & ((1 << SHIFT) - 1)) == 0 ? edge_gradient[0] : 0.0f;\n"
            "        float gx = 0.0f, gy = 0.0f, gz = 0.0f;\n        #pragma unroll\n"
            "        for (int k = 0; k < 10; ++k) {\n"
            "            gx += t[k] * m[k]; gy += t[10 + k] * m[k]; gz += t[20 + k] * m[k];\n        }\n"
            "        gx *= g; gy *= g; gz *= g;\n        #pragma unroll\n"
            "        for (int offset = 16; offset >= (1 << SHIFT); offset >>= 1) {\n"
            "            gx += __shfl_xor_sync(FULL_MASK, gx, offset, 32);\n"
            "            gy += __shfl_xor_sync(FULL_MASK, gy, offset, 32);\n"
            "            gz += __shfl_xor_sync(FULL_MASK, gz, offset, 32);\n        }"
            + PROJECT_STORE + "    }\n")
    return head, body


def variant(source: str, name: str) -> str:
    assert source.count(STORE) == 1 and source.count(SIGNATURE_END) == 1, "not a BUTTERFLY w' source"
    s = source.replace(SIGNATURE_END, "    scalar_t* dr,\n    const float* __restrict__ vec) {")
    if name == "gather":
        return s.replace(STORE, gather())
    head, body = split()
    s = s.replace(STORE, body)
    # the table in shared memory, copied once per block (constant memory would serialise the lanes' 16 addresses)
    first = s.index("{", s.index("const float* __restrict__ vec) {")) + 1
    copy = ("\n    __shared__ float sh_grad[16 * 30];\n"
            "    for (int i = threadIdx.x; i < 16 * 30; i += blockDim.x) sh_grad[i] = SH_GRAD[i];\n"
            "    __syncthreads();\n")
    s = s[:first] + copy + s[first:]
    marker = 'extern "C" __global__'
    return s.replace(marker, head + marker, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("out")
    p.add_argument("--arch", nargs="+", default=["80", "89", "120"])
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(a.source).read_text()
    files = {"base": base, "gather": variant(base, "gather"), "split": variant(base, "split")}
    for name, text in files.items():
        (out / f"{name}.cu").write_text(text)
    import cu_census

    for arch, name in itertools.product(a.arch, files):
        print(f"== sm_{arch} {name}", flush=True)
        print(cu_census.census(str(out / f"{name}.cu"), arch), flush=True)


if __name__ == "__main__":
    main()
