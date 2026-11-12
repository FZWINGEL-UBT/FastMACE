"""CG term counts of interaction 2's paths in the complex (m) basis against the real basis (e3nn, CPU)."""
import json
from pathlib import Path
import torch
torch.serialization.add_safe_globals([slice])  # e3nn constants.pt under torch >= 2.6
from e3nn import o3


def change_basis_real_to_complex(l, dtype=torch.complex128):
    """Q [complex m, real m] (index m + l): Y_m = (-1)^m (R_m + i R_-m) / sqrt 2 for m > 0, (R_|m| - i R_-|m|) /
    sqrt 2 for m < 0, Y_0 = R_0 (the term counts depend only on this block structure)."""
    q = torch.zeros(2 * l + 1, 2 * l + 1, dtype=dtype)
    r = 2 ** -0.5
    for m in range(-l, l + 1):
        if m == 0:
            q[l, l] = 1
        elif m > 0:
            q[l + m, l + m] = (-1) ** m * r
            q[l + m, l - m] = (-1) ** m * r * 1j
        else:
            q[l + m, l - m] = r
            q[l + m, l + m] = -r * 1j
    return q

rows = []
for (l1, l2, l3) in [(0, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 2, 1), (0, 2, 2), (1, 1, 2), (1, 3, 2), (0, 3, 3), (1, 2, 3)]:
    c = o3.wigner_3j(l1, l2, l3, dtype=torch.float64).to(torch.complex128)
    q1, q2, q3 = (change_basis_real_to_complex(l, dtype=torch.complex128) for l in (l1, l2, l3))
    cc = torch.einsum("ai,bj,ck,ijk->abc", q1, q2, q3.conj(), c)  # wigner_3j is [i, j, k]
    real = int((c.abs() > 1e-12).sum())
    comp = int((cc.abs() > 1e-12).sum())
    # a complex term c x_i y_j g_o costs about 4 real FFMA per bilinear product (2 if c real), and the m and -m
    # components of a real field are conjugates: half the terms need computing
    rows.append(dict(path=(l1, l2, l3), real_terms=real, complex_terms=comp, complex_real_ffma_est=2 * comp))
print(json.dumps(rows))
Path(__file__).with_suffix(".json").write_text(json.dumps(rows, indent=1) + "\n")
print("real", sum(r["real_terms"] for r in rows), "complex", sum(r["complex_terms"] for r in rows),
      "complex in real FFMA (conjugate-halved, 4 per product)", sum(r["complex_real_ffma_est"] for r in rows))
