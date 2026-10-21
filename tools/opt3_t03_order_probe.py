"""OPT3-T03 feasibility probe (measurement only): can a fused integrator kernel reproduce PyTorch bitwise?

1. ``x.sum(dim=1)`` of an (M, 3) float64 CUDA tensor against the orders ((a+b)+c), (a+(b+c)) and ((a+c)+b),
   for gathered differences as the compaction has them and for squared displacements.
2. A Triton kernel with ``enable_fp_fusion=False`` computing the Langevin half kick
   ``v + (((c1 * f) / m - c2 * v) + r)`` against the same PyTorch expression, and with fusion on.
Prints one JSON line.
"""

import json  # noqa: I001  (fastmace before torch)

import fastmace  # noqa: F401  (before torch)
import torch
import triton
import triton.language as tl


@triton.jit
def kick(v_ptr, f_ptr, m_ptr, r_ptr, out_ptr, c_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    v = tl.load(v_ptr + i, mask=mask)
    f = tl.load(f_ptr + i, mask=mask)
    m = tl.load(m_ptr + i // 3, mask=mask)
    r = tl.load(r_ptr + i, mask=mask)
    c1 = tl.load(c_ptr)
    c2 = tl.load(c_ptr + 1)
    a = c1 * f
    b = a / m
    c = c2 * v
    d = b - c
    e = d + r
    tl.store(out_ptr + i, v + e, mask=mask)


@triton.jit
def sum3(x_ptr, out_ptr, n, ORDER: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    a = tl.load(x_ptr + 3 * i, mask=mask)
    b = tl.load(x_ptr + 3 * i + 1, mask=mask)
    c = tl.load(x_ptr + 3 * i + 2, mask=mask)
    if ORDER == 0:
        s = (a + b) + c
    elif ORDER == 1:
        s = a + (b + c)
    else:
        s = (a + c) + b
    tl.store(out_ptr + i, s, mask=mask)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    out = {}
    m_rows = 1_000_003
    positions = torch.randn(4000, 3, dtype=torch.float64, device=dev) * 7
    sender = torch.randint(0, 4000, (m_rows,), device=dev)
    receiver = torch.randint(0, 4000, (m_rows,), device=dev)
    shifts = torch.randn(m_rows, 3, dtype=torch.float64, device=dev) * 3
    vectors = positions[receiver] - positions[sender] + shifts
    squares = vectors * vectors
    reference = squares.sum(dim=1)
    names = ["(a+b)+c", "a+(b+c)", "(a+c)+b"]
    for order, name in enumerate(names):
        result = torch.empty(m_rows, dtype=torch.float64, device=dev)
        sum3[(triton.cdiv(m_rows, 1024),)](squares, result, m_rows, ORDER=order, BLOCK=1024)
        out[f"edges {name}"] = int((result != reference).sum())
    displacement = (positions - positions.roll(1, 0) * 0.999) ** 2
    reference = displacement.sum(dim=1)
    for order, name in enumerate(names):
        result = torch.empty(4000, dtype=torch.float64, device=dev)
        sum3[(triton.cdiv(4000, 1024),)](displacement, result, 4000, ORDER=order, BLOCK=1024)
        out[f"atoms {name}"] = int((result != reference).sum())
    n = 4000 * 3
    v = torch.randn(4000, 3, dtype=torch.float64, device=dev)
    f = torch.randn(4000, 3, dtype=torch.float64, device=dev) * 5
    masses = torch.rand(4000, 1, dtype=torch.float64, device=dev) * 100 + 1
    r = torch.randn(4000, 3, dtype=torch.float64, device=dev) * 1e-3
    c1, c2 = 0.123456789123, 0.0098765432109
    reference = v + (c1 * f / masses - c2 * v + r)
    for fusion in (False, True):
        result = torch.empty_like(v)
        coefficients = torch.tensor([c1, c2], dtype=torch.float64, device=dev)  # float64: a Python float arg is fp32
        kick[(triton.cdiv(n, 1024),)](v, f, masses, r, result, coefficients, n, BLOCK=1024, enable_fp_fusion=fusion)
        out[f"kick fp_fusion={fusion}"] = int((result != reference).sum())
    out.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
