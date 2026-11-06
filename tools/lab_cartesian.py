"""Two new w' backward candidates for OMAT-0 medium on the H200 (owner, 6 November 2026; Codex report
plans/codex-ideas-20261106.md, "Second pass"): the kernel-lab bound of each against today's default w' kernel.

N1, Cartesian harmonic adjoints. Per path (x degree a in {0, 1}, edge degree l, output degree L) the scalar
contraction of the message with the receiver's adjoint g is, up to a constant, a Cartesian expression in the edge
direction n and a harmonic polynomial H_L(n) = sum_k q_k P_k(n) whose 2L + 1 coefficients q are a fixed linear map of
g (``fit_conversion``: least squares in float64 against e3nn's harmonics and the kernel schedule's own coupling
coefficients, which absorbs every normalisation):

- a = 0:             f = x0 H_L(n)
- a = 1, L = l + 1:  f = x . grad H_L(n)
- a = 1, L = l - 1:  f = (x . n) H_L(n) - |n|^2 (x . grad H_L(n)) / (2L + 1)

The kernel then gives dx = w df/dx and the edge-vector gradient w' f n + (w / r) (grad f - l f n) directly (three
components per edge instead of the 16 sh components and dL/dr). A node kernel converts the adjoint rows to q once
per force call (``converter_source``; timed separately and with the w' kernel). Rounding class.

N2, receiver-tiled FP64 matrix adjoints. Bound only: the Codex compile-only skeleton (one vector path, 16 edges per
block, U = S @ D and M = (w x) @ D^T on FP64 mma.sync) launched for every 16 edges, with U stored (``N2-skel``) or
consumed on chip (``N2-noU``, no S preparation, no receiver gather, no radial tables, no geometry, no atomics). Six of
OMAT-0's ten paths have a vector x: 6 x N2-noU is an optimistic floor of N2's non-radial part.

Stop rule (the brief): a candidate whose ratio against the default kernel is above 0.95 is not taken further.

Usage (hybrid environment, FastMACE on PYTHONPATH; an idle GPU):
  python tools/lab_cartesian.py --checkpoint mace-omat-0-medium.model --output DIR
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402

# ------------------------------------------------------------------------------------------------- algebra (CPU)


def sh_frame():
    """S [3, 3] with sh_1(v) = sqrt(3) v @ S: the sh frame n' = v @ S in which the kernel reads the l = 1 harmonics
    (and the l = 1 features), and v = n' @ S^T."""
    import torch
    torch.serialization.add_safe_globals([slice])  # e3nn's constants.pt under torch.load(weights_only=True)
    from e3nn import o3
    eye = torch.eye(3, dtype=torch.float64)
    return o3.spherical_harmonics(1, eye, normalize=False, normalization="component") / math.sqrt(3.0)


def harmonics(l, n, frame, normalize=False):
    """e3nn's component-normalised harmonics of degree l at the sh-frame vectors n [..., 3]."""
    from e3nn import o3
    return o3.spherical_harmonics(l, n @ frame.T, normalize=normalize, normalization="component")


def _sym2(q):
    import torch
    q0, q1, q2, q3, q4 = q.unbind(-1)
    return torch.stack([torch.stack([q0, q1, q2], -1), torch.stack([q1, q3, q4], -1),
                        torch.stack([q2, q4, -q0 - q3], -1)], -2)


def _sym3n(q, n):
    """Q(n) = T . n for the symmetric traceless rank-3 tensor T with independent entries q0..q6 (the kernel's form)."""
    import torch
    q0, q1, q2, q3, q4, q5, q6 = q.unbind(-1)
    nx, ny, nz = n.unbind(-1)
    qxx = q0 * nx + q1 * ny + q2 * nz
    qxy = q1 * nx + q3 * ny + q4 * nz
    qxz = q2 * nx + q4 * ny + (-q0 - q3) * nz
    qyy = q3 * nx + q5 * ny + q6 * nz
    qyz = q4 * nx + q6 * ny + (-q1 - q5) * nz
    qzz = -qxx - qyy
    return torch.stack([torch.stack([qxx, qxy, qxz], -1), torch.stack([qxy, qyy, qyz], -1),
                        torch.stack([qxz, qyz, qzz], -1)], -2)


def cartesian_h(L, q, n):
    """(H_L(q; n), grad_n H_L) in the kernel's Cartesian form; q [..., 2L+1], n [..., 3]."""
    import torch
    if L == 0:
        return q[..., 0], torch.zeros_like(n)
    if L == 1:
        return (q * n).sum(-1), q.expand_as(n)
    if L == 2:
        v = torch.einsum("...ij,...j->...i", _sym2(q), n)
        return (n * v).sum(-1), 2.0 * v
    if L == 3:
        v = torch.einsum("...ij,...j->...i", _sym3n(q, n), n)
        return (n * v).sum(-1), 3.0 * v
    raise ValueError(f"L = {L} not supported")


def cartesian_f(a, l, L, x, n, q):
    """The path's scalar contraction in the Cartesian form (homogeneous of degree l in n)."""
    h, gh = cartesian_h(L, q, n)
    if a == 0:
        return x[..., 0] * h
    if L == l + 1:
        return (x * gh).sum(-1)
    if L == l - 1:
        return (x * n).sum(-1) * h - (n * n).sum(-1) * (x * gh).sum(-1) / (2 * L + 1)
    raise ValueError(f"path ({a}, {l}, {L}) not of a form N1 covers")


def fit_conversion(a, l, L, coef, frame, samples=96, seed=20261106):
    """M [2L+1, 2L+1] with q = M g, so that sum_ijo coef[i,j,o] x_i Y_lj(n) g_o == cartesian_f(x, n, q) for every
    x, n (n not normalised: the identity of homogeneous polynomials), and the fit's relative residual."""
    import torch
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(samples, 2 * a + 1, generator=gen, dtype=torch.float64)
    n = torch.randn(samples, 3, generator=gen, dtype=torch.float64)
    y = harmonics(l, n, frame)
    target = torch.einsum("ijo,si,sj->so", torch.as_tensor(coef, dtype=torch.float64), x, y)  # [S, 2L+1]
    phi = torch.stack([cartesian_f(a, l, L, x, n, torch.eye(2 * L + 1, dtype=torch.float64)[k].expand(samples, -1))
                       for k in range(2 * L + 1)], -1)  # [S, 2L+1]
    m = torch.linalg.lstsq(phi, target).solution  # [k, o]
    residual = float((phi @ m - target).abs().max() / target.abs().max().clamp_min(1e-300))
    return m, residual


# ------------------------------------------------------------------------------------------------ the schedule


def path_table(kernel):
    """The w' kernel's paths: degrees, offsets, coupling tensor and each output component's dout address
    (group, receiver row width, column offset without the channel)."""
    import numpy as np

    from fastmace.kernels.oeq_streaming.fused_forward import block_map
    channels = int(kernel.channels)
    bmap = block_map(kernel.blocks, channels) if kernel.blocks else None
    out = []
    for path in kernel.schedule.paths:
        if path.edge_mul != 1:
            raise ValueError("N1 lab: edge_mul != 1")
        outputs = list(path.outputs)
        dim_out = outputs[0].accumulator.irrep_dim
        a, l, L = (path.input_irrep_dim - 1) // 2, (path.edge_irrep_dim - 1) // 2, (dim_out - 1) // 2
        coef = np.zeros((path.input_irrep_dim, path.edge_irrep_dim, dim_out))
        for output in outputs:
            for term in output.terms:
                coef[term.input_component, term.edge_component, output.output_component] += term.coefficient
        start = outputs[0].accumulator.start
        if bmap is not None:
            g, width, rank, stride = bmap[int(start)]
            address = [(g, width, rank + m * stride) for m in range(dim_out)]
        else:
            address = [(0, int(kernel.output_dim), start + m * channels) for m in range(dim_out)]
        out.append(dict(a=a, l=l, L=L, weight_start=path.weight_start, input_start=path.input_start,
                        edge_start=path.edge_start, coef=coef, dout=address))
    return out


# --------------------------------------------------------------------------------------------- CUDA generation


def _render(kernel, stage):
    from fastmace.kernels.oeq_streaming import wprime_backward as wp
    return wp.render(kernel.schedule, kernel.scalar, kernel.terms, vector=kernel.vector, table_w=kernel.table_w,
                     blocks=kernel.blocks, factored=kernel.factored, horner=kernel.horner,
                     scalar_paths=kernel.scalar_paths, table_pairs=kernel.table_pairs,
                     hoist_scale=kernel.hoist_scale, stage_dout=stage)


def _f(v):
    text = f"{float(v):.9g}"
    if not any(c in text for c in ".en"):  # "1" -> "1.0f" (a bare "1f" is a user-defined literal in C++)
        text += ".0"
    return text + "f"


def n1_source(kernel, paths, stage):
    """The N1 w' kernel (the plan kernel's signature; dsh[e, 0:3] holds the sh-frame edge-vector gradient, dr 0)."""
    from fastmace.kernels.oeq_streaming import wprime_backward as wp
    if not (kernel.table_w and kernel.horner and not kernel.table_pairs and not kernel.hoist_scale):
        raise ValueError("N1 lab: needs TABLE_W + HORNER, q-major table (the H200 default)")
    rendered = _render(kernel, stage)
    head = rendered[:rendered.index("    acc_t edge_gradient[EDGE_DIM]")]
    horner = rendered[rendered.index("                scalar_t gv[PADDED_TERMS];"):
                      rendered.index("                acc_t weight_gradient")]
    tail = rendered[rendered.index("    }  // STAGE_DOUT: the run's edges"):] if stage else "}\n"
    channels = int(kernel.channels)
    if stage:
        _, stage_offsets, _ = wp._stage_layout(kernel.schedule, kernel.blocks)

    def dout(g, width, column):
        if stage:
            return f"stage_row[{stage_offsets[g] + column} + channel]"
        pointer = f"dout{g}" if kernel.blocks else "dout"
        return f"{pointer}[int64_t(receiver) * {width} + {column} + channel]"

    inputs = sorted({(p["input_start"], 2 * p["a"] + 1) for p in paths})
    lines = [head,
             "    // N1 (lab_cartesian): Cartesian harmonic adjoints; dout already converted to q per path.",
             "    const float nx = sh[edge_base + 1] * 0.57735026918962576451f;",
             "    const float ny = sh[edge_base + 2] * 0.57735026918962576451f;",
             "    const float nz = sh[edge_base + 3] * 0.57735026918962576451f;",
             "    const float ri = 1.f / r;",
             "    float fx = 0.f, fy = 0.f, fz = 0.f;",
             "    for (int channel = lane; channel < CHANNEL_DIM; channel += LOGICAL_GROUP_SIZE) {"]
    names = {}
    for start, dim in inputs:
        if dim == 1:
            names[start] = ("s", f"x{start}")
            lines.append(f"        const float x{start} = x[input_base + {start} + channel];")
            lines.append(f"        float dx{start} = 0.f;")
        elif dim == 3:
            names[start] = ("v", f"x{start}")
            for m, c in enumerate("xyz"):
                lines.append(f"        const float x{start}{c} = x[input_base + {start + m * channels} + channel];")
                lines.append(f"        float dx{start}{c} = 0.f;")
        else:
            raise ValueError("N1 lab: x of degree > 1")
    for p in paths:
        a, l, L = p["a"], p["l"], p["L"]
        s = ["        {", f"        const int weight_index = {p['weight_start']} + channel;", horner]
        for m, (g, width, column) in enumerate(p["dout"]):
            s.append(f"        const float q{m} = {dout(g, width, column)};")
        if L == 0:
            s.append("        const float h = q0, hx = 0.f, hy = 0.f, hz = 0.f;")
        elif L == 1:
            s.append("        const float h = q0 * nx + q1 * ny + q2 * nz, hx = q0, hy = q1, hz = q2;")
        elif L == 2:
            s += ["        const float qzz = -q0 - q3;",
                  "        const float vx = q0 * nx + q1 * ny + q2 * nz, vy = q1 * nx + q3 * ny + q4 * nz,"
                  " vz = q2 * nx + q4 * ny + qzz * nz;",
                  "        const float h = nx * vx + ny * vy + nz * vz, hx = 2.f * vx, hy = 2.f * vy, hz = 2.f * vz;"]
        elif L == 3:
            s += ["        const float q7 = -q0 - q3, q8 = -q1 - q5;",
                  "        const float Qxx = q0 * nx + q1 * ny + q2 * nz, Qxy = q1 * nx + q3 * ny + q4 * nz,"
                  " Qxz = q2 * nx + q4 * ny + q7 * nz;",
                  "        const float Qyy = q3 * nx + q5 * ny + q6 * nz, Qyz = q4 * nx + q6 * ny + q8 * nz,"
                  " Qzz = -Qxx - Qyy;",
                  "        const float vx = Qxx * nx + Qxy * ny + Qxz * nz, vy = Qxy * nx + Qyy * ny + Qyz * nz,"
                  " vz = Qxz * nx + Qyz * ny + Qzz * nz;",
                  "        const float h = nx * vx + ny * vy + nz * vz, hx = 3.f * vx, hy = 3.f * vy, hz = 3.f * vz;"]
        else:
            raise ValueError("N1 lab: L > 3")
        kind, xn = names[p["input_start"]]
        if a == 1:
            s.append(f"        const float xx = {xn}x, xy = {xn}y, xz = {xn}z;")
            # Hessian of H times x (L >= 2; zero below)
            if L == 2:
                s.append("        const float hhx = 2.f * (q0 * xx + q1 * xy + q2 * xz), hhy = 2.f * (q1 * xx + q3 * xy"
                         " + q4 * xz), hhz = 2.f * (q2 * xx + q4 * xy + qzz * xz);")
            elif L == 3:
                s.append("        const float hhx = 6.f * (Qxx * xx + Qxy * xy + Qxz * xz), hhy = 6.f * (Qxy * xx"
                         " + Qyy * xy + Qyz * xz), hhz = 6.f * (Qxz * xx + Qyz * xy + Qzz * xz);")
            else:
                s.append("        const float hhx = 0.f, hhy = 0.f, hhz = 0.f;")
        if a == 0:
            s += [f"        const float f = {xn} * h, gx = {xn} * hx, gy = {xn} * hy, gz = {xn} * hz;",
                  f"        dx{p['input_start']} = fmaf(radial_weight, h, dx{p['input_start']});"]
        elif L == l + 1:
            s += ["        const float f = xx * hx + xy * hy + xz * hz, gx = hhx, gy = hhy, gz = hhz;"]
            for c in "xyz":
                s.append(f"        dx{p['input_start']}{c} = fmaf(radial_weight, h{c}, dx{p['input_start']}{c});")
        else:
            s += ["        const float A = xx * nx + xy * ny + xz * nz, B = xx * hx + xy * hy + xz * hz;",
                  f"        const float ik = {_f(1.0 / (2 * L + 1))};",
                  "        const float f = A * h - ik * B;",
                  "        const float gx = xx * h + A * hx - ik * (2.f * nx * B + hhx),"
                  " gy = xy * h + A * hy - ik * (2.f * ny * B + hhy), gz = xz * h + A * hz - ik * (2.f * nz * B + hhz);"]
            for c in "xyz":
                s.append(f"        dx{p['input_start']}{c} = fmaf(radial_weight, n{c} * h - ik * h{c},"
                         f" dx{p['input_start']}{c});")
        s += ["        const float wr = radial_weight * ri;",
              f"        const float radial = (sloped ? radial_dt * sc : 0.f) * f - {_f(l)} * wr * f;",
              "        fx = fmaf(wr, gx, fmaf(radial, nx, fx));",
              "        fy = fmaf(wr, gy, fmaf(radial, ny, fy));",
              "        fz = fmaf(wr, gz, fmaf(radial, nz, fz));",
              "        }"]
        lines += s
    for start, dim in inputs:
        if dim == 1:
            lines.append(f"        atomicAdd(dx + input_base + {start} + channel, dx{start});")
        else:
            for m, c in enumerate("xyz"):
                lines.append(f"        atomicAdd(dx + input_base + {start + m * channels} + channel, dx{start}{c});")
    lines += ["    }",
              "    for (int o = 16; o; o >>= 1) {",
              "        fx += __shfl_down_sync(FULL_MASK, fx, o);",
              "        fy += __shfl_down_sync(FULL_MASK, fy, o);",
              "        fz += __shfl_down_sync(FULL_MASK, fz, o);",
              "    }",
              "    if (lane == 0) { dsh[edge_base] = fx; dsh[edge_base + 1] = fy; dsh[edge_base + 2] = fz; dr[e] = 0.f; }",
              tail]
    return "\n".join(lines)


def converter_source(kernel, paths, matrices):
    """The node kernel q = M g per (node, path, channel), group buffers in and out (same layout)."""
    channels = int(kernel.channels)
    groups = len(kernel.blocks) if kernel.blocks else 1
    args = ", ".join([f"const float* __restrict__ in{g}" for g in range(groups)]
                     + [f"float* __restrict__ out{g}" for g in range(groups)] + ["long long nodes"])
    body = [f'extern "C" __global__ void cart_convert({args}) {{',
            "    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;",
            f"    if (t >= nodes * {len(paths) * channels}) return;",
            f"    const long long node = t / {len(paths) * channels};",
            f"    const int rem = int(t - node * {len(paths) * channels});",
            f"    const int path = rem / {channels}, channel = rem - path * {channels};",
            "    switch (path) {"]
    for index, (p, m) in enumerate(zip(paths, matrices)):
        dim = 2 * p["L"] + 1
        body.append(f"    case {index}: {{")
        for o, (g, width, column) in enumerate(p["dout"]):
            body.append(f"        const float g{o} = in{g}[node * {width} + {column} + channel];")
        for k, (g, width, column) in enumerate(p["dout"]):
            terms = " + ".join(f"{_f(m[k, o])} * g{o}" for o in range(dim) if float(m[k, o]) != 0.0) or "0.f"
            body.append(f"        out{g}[node * {width} + {column} + channel] = {terms};")
        body.append("        break; }")
    body += ["    }", "}"]
    return "\n".join(body) + "\n"


N2_SKELETON = r"""
__device__ __forceinline__ void mm(double &d0,double &d1,double a,double b) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};" : "+d"(d0),"+d"(d1) : "d"(a),"d"(b));
}
extern "C" __global__ __launch_bounds__(128)
void adjoint_mma(const float* S, const float* D, const float* X, const float* W, float* Uout, double* Mout) {
    __shared__ double ss[48*8];
    __shared__ double dd[8*32];
    __shared__ double wx[48*32];
    const int t=threadIdx.x,l=t&31,w=t/32;
    for(int i=t;i<48*8;i+=128) ss[i]=double(S[blockIdx.x*48*8+i]);
    double m0[2]={},m1[2]={};
    double keep=0;
    for(int cc=0;cc<128;cc+=32) {
        for(int i=t;i<8*32;i+=128) dd[i]=double(D[(i/32)*128+cc+i%32]);
        for(int i=t;i<48*32;i+=128) {
            int row=i/32,ch=cc+i%32;
            wx[i]=double(X[(blockIdx.x*48+row)*128+ch])*double(W[(blockIdx.x*16+row/3)*128+ch]);
        }
        __syncthreads();
        #pragma unroll
        for(int rt=0;rt<6;++rt) {
            double u0=0,u1=0;
            #pragma unroll
            for(int k=0;k<8;k+=4) {
                double a=ss[(rt*8+l/4)*8+k+l%4];
                double b=dd[(k+l%4)*32+w*8+l/4];
                mm(u0,u1,a,b);
            }
            int row=rt*8+l/4,ch=cc+w*8+2*(l%4);
            STORE_U
        }
        #pragma unroll
        for(int it=0;it<2;++it) {
            int rt=w+4*it;
            if(rt<6) {
                #pragma unroll
                for(int k=0;k<32;k+=4) {
                    double a=wx[(rt*8+l/4)*32+k+l%4];
                    double b=dd[(l/4)*32+k+l%4];
                    mm(m0[it],m1[it],a,b);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int it=0;it<2;++it) {
        int rt=w+4*it;
        if(rt<6) {
            int row=rt*8+l/4,col=2*(l%4);
            Mout[(blockIdx.x*48+row)*8+col]=m0[it];
            Mout[(blockIdx.x*48+row)*8+col+1]=m1[it];
        }
    }
    if (keep == 1.2345e-300) Mout[0] = keep;
}
"""
N2_STORE = ("Uout[(blockIdx.x*48+row)*128+ch]=float(u0);\n"
            "            Uout[(blockIdx.x*48+row)*128+ch+1]=float(u1);")
N2_KEEP = "keep += u0 * double(ch) + u1;"


# ------------------------------------------------------------------------------------------------- GPU harness


def _load(source, name, extra=()):
    import torch
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin
    cubin = compile_cubin(source, torch.cuda.get_device_capability(), tuple(extra))
    module = _check(driver.cuModuleLoadData(cubin))
    function = _check(driver.cuModuleGetFunction(module, name.encode()))
    regs = _check(driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, function))
    local = _check(driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function))
    return SimpleNamespace(module=module, function=function, registers=int(regs), local_bytes=int(local))


def _launch(function, grid, threads, values, types, shared=0):
    import torch
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.runtime import _check
    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    _check(driver.cuLaunchKernel(function, grid, 1, 1, threads, 1, 1, shared, stream, (values, types), 0))


def wprime_launch(kernel, loaded, stage):
    """A launch of a kernel with the plan kernel's signature, as WPrimeBackward.__call__."""
    def launch(args):
        (x, sh, weights, senders, receivers, dout, dx, dsh, r, pair, gprime, starts, scales, bins, r_lo, bin_scale,
         r_max, dr) = args
        nodes, edges = x.shape[0], sh.shape[0]
        blocks = ((edges + kernel.stage_chunk - 1) // kernel.stage_chunk if stage
                  else (edges * kernel.cohort + kernel.num_threads - 1) // kernel.num_threads)
        p = ctypes.c_void_p
        douts = list(dout) if isinstance(dout, (list, tuple)) else [dout]
        values = (nodes, edges, x.data_ptr(), sh.data_ptr(), weights.data_ptr(), senders.data_ptr(),
                  receivers.data_ptr(), *[d.data_ptr() for d in douts], dx.data_ptr(), dsh.data_ptr(), r.data_ptr(),
                  pair.data_ptr(), gprime.data_ptr(), starts.data_ptr(), scales.data_ptr(), bins.data_ptr(),
                  r_lo, bin_scale, r_max, bins.shape[0], starts.shape[0], dr.data_ptr())
        types = ((ctypes.c_longlong, ctypes.c_longlong) + (p,) * (13 + len(douts))
                 + (ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int, p))
        _launch(loaded.function, blocks, kernel.num_threads, values, types, kernel.shared_memory)
    return launch


def converter_launch(loaded, nodes, npaths, channels):
    def launch(ins, outs):
        p = ctypes.c_void_p
        values = (*[t.data_ptr() for t in ins], *[t.data_ptr() for t in outs], nodes)
        types = (p,) * (len(ins) + len(outs)) + (ctypes.c_longlong,)
        total = nodes * npaths * channels
        _launch(loaded.function, (total + 255) // 256, 256, values, types)
    return launch


def edge_vector_reference(args, frame, dsh, dr):
    """dL/d(edge vector) in the sh frame from the default kernel's dL/dsh and dL/dr: dr n' + J^T dsh (float64)."""
    import torch
    sh, r = args[1], args[8]
    n = (sh[:, 1:4].double() / math.sqrt(3.0))
    n = n / n.norm(dim=1, keepdim=True)
    vec = (n * r.double()[:, None]).requires_grad_(True)
    f = frame.to(sh.device)
    ys = torch.cat([harmonics(l, vec, f, normalize=True) for l in range(4)], dim=1)
    sh_error = float((ys.detach() - sh.double()).abs().max())
    (grad,) = torch.autograd.grad(ys, vec, dsh.double())
    return grad + dr.double()[:, None] * n, sh_error


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@10x10x10")
    p.add_argument("--rewrites", nargs="*", default=[])
    p.add_argument("--min-edges", type=int, default=100000)
    p.add_argument("--equilibrate-steps", type=int, default=300)
    p.add_argument("--rounds", type=int, default=21)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--flush-l2-mb", type=int, default=256)
    p.add_argument("--skip-n2", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    import torch

    import omat0_kernel_lab as lab
    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cap_args = SimpleNamespace(checkpoint=args.checkpoint, model=args.model, structure=args.structure,
                               rewrites=args.rewrites, min_edges=args.min_edges,
                               equilibrate_steps=args.equilibrate_steps)
    captured, meta = lab.capture(cap_args)
    kernel, cargs = captured["wprime"]
    captured.pop("forward", None)
    gc.collect()
    torch.cuda.empty_cache()
    meta.update(tool="lab_cartesian", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
                gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                fastmace_commit=ab_step_time.fastmace_commit(), rounds=args.rounds,
                capture_s=time.perf_counter() - started,
                plan_kernel=dict(stage_dout=bool(kernel.stage_dout), factored=bool(kernel.factored),
                                 horner=bool(kernel.horner), scalar_paths=bool(kernel.scalar_paths),
                                 butterfly=bool(kernel.butterfly), redux=bool(kernel.redux),
                                 blocks=bool(kernel.blocks), threads=kernel.num_threads, cohort=kernel.cohort))
    (args.output / "wprime.cu").write_text(kernel.source)

    # ---- N1 algebra
    frame = sh_frame()
    paths = path_table(kernel)
    matrices, fits = [], []
    for q in paths:
        m, residual = fit_conversion(q["a"], q["l"], q["L"], q["coef"], frame)
        matrices.append(m)
        fits.append(dict(path=[q["a"], q["l"], q["L"]], weight_start=q["weight_start"], residual=residual,
                         matrix=m.tolist()))
        print(f"N1 fit path a={q['a']} l={q['l']} L={q['L']}: residual {residual:.2e}", flush=True)
    meta["n1_fits"] = fits
    sources = {"N1-stage": n1_source(kernel, paths, True), "N1-nostage": n1_source(kernel, paths, False),
               "base-nostage": _render(kernel, False), "convert": converter_source(kernel, paths, matrices)}
    for name, source in sources.items():
        (args.output / f"{name}.cu").write_text(source)

    # ---- buffers
    def outputs():
        a = list(cargs)
        a[6] = torch.zeros_like(cargs[6])
        a[7] = torch.zeros_like(cargs[7])
        a[17] = torch.zeros_like(cargs[17])
        return a

    douts = list(cargs[5]) if isinstance(cargs[5], (list, tuple)) else [cargs[5]]
    converted = [torch.empty_like(d) for d in douts]
    base_args, n1_args, shared_args = outputs(), outputs(), outputs()
    n1_args[5] = converted if isinstance(cargs[5], (list, tuple)) else converted[0]
    nodes = cargs[0].shape[0]
    edges = cargs[1].shape[0]
    loaded = {name: _load(src, "oeq_projected_backward_wprime") for name, src in sources.items() if name != "convert"}
    loaded["convert"] = _load(sources["convert"], "cart_convert")
    convert = converter_launch(loaded["convert"], nodes, len(paths), int(kernel.channels))

    def base_launch(a):
        kernel(*a)

    n1_stage = wprime_launch(kernel, loaded["N1-stage"], True)
    n1_nostage = wprime_launch(kernel, loaded["N1-nostage"], False)
    base_nostage = wprime_launch(kernel, loaded["base-nostage"], False)
    variants = {
        "base": (lambda: base_launch(base_args), base_args),
        "base-nostage": (lambda: base_nostage(shared_args), shared_args),
        "N1-convert": (lambda: convert(douts, converted), None),
        "N1-stage": (lambda: n1_stage(n1_args), n1_args),
        "N1-nostage": (lambda: n1_nostage(n1_args), n1_args),
        "N1-stage+convert": (lambda: (convert(douts, converted), n1_stage(n1_args)), n1_args),
        "N1-nostage+convert": (lambda: (convert(douts, converted), n1_nostage(n1_args)), n1_args),
    }
    info = {"base": dict(registers=None), "base-nostage": vars(loaded["base-nostage"]),
            "N1-convert": vars(loaded["convert"]), "N1-stage": vars(loaded["N1-stage"]),
            "N1-nostage": vars(loaded["N1-nostage"]), "N1-stage+convert": vars(loaded["N1-stage"]),
            "N1-nostage+convert": vars(loaded["N1-nostage"])}
    info = {k: {kk: vv for kk, vv in v.items() if kk in ("registers", "local_bytes")} for k, v in info.items()}

    # ---- correctness of N1 against the default kernel
    base_launch(base_args)
    convert(douts, converted)
    n1_stage(n1_args)
    torch.cuda.synchronize()
    reference, sh_error = edge_vector_reference(cargs, frame, base_args[7], base_args[17])
    real = (cargs[3] >= 0) & (cargs[3] < nodes) & (cargs[4] >= 0) & (cargs[4] < nodes)
    got = n1_args[7][:, :3].double()
    scale = float(reference[real].abs().max())
    diff = (got[real] - reference[real]).abs()
    rel_rows = (diff.norm(dim=1) / reference[real].norm(dim=1).clamp_min(1e-30 + 1e-6 * scale))
    dx_scale = float(base_args[6].abs().max())
    check = dict(sh_recompute_max_abs=sh_error, edge_vector_max_abs_over_max=float(diff.max()) / scale,
                 edge_vector_row_rel_median=float(rel_rows.median()), edge_vector_row_rel_p99=float(
                     torch.quantile(rel_rows[:200000].float(), 0.99)),
                 dx_max_abs_over_max=float((n1_args[6] - base_args[6]).abs().max()) / dx_scale,
                 real_edges=int(real.sum()))
    # The default's own float32 rounding: the same reference from a second launch order (base-nostage, exact in
    # dsh and dr) is no help, so report the float32 vs float64 recompute of the reference itself.
    meta["n1_check"] = check
    print("N1 check:", json.dumps(check), flush=True)
    del reference, diff, rel_rows, got
    torch.cuda.empty_cache()

    # ---- N2 skeleton bound
    if not args.skip_n2:
        nb = (edges + 15) // 16
        dev = "cuda"
        gen = torch.Generator(device=dev).manual_seed(7)
        n2 = dict(S=torch.randn(nb * 48 * 8, device=dev, generator=gen),
                  D=torch.randn(8 * 128, device=dev, generator=gen),
                  X=torch.randn(nb * 48 * 128, device=dev, generator=gen),
                  W=torch.randn(nb * 16 * 128, device=dev, generator=gen),
                  U=torch.empty(nb * 48 * 128, device=dev),
                  M=torch.empty(nb * 48 * 8, device=dev, dtype=torch.float64))
        skel = _load(N2_SKELETON.replace("STORE_U", N2_STORE), "adjoint_mma")
        keep = _load(N2_SKELETON.replace("STORE_U", N2_KEEP), "adjoint_mma")
        values = tuple(n2[k].data_ptr() for k in ("S", "D", "X", "W", "U", "M"))
        types = (ctypes.c_void_p,) * 6
        variants["N2-skel"] = (lambda: _launch(skel.function, nb, 128, values, types), None)
        variants["N2-noU"] = (lambda: _launch(keep.function, nb, 128, values, types), None)
        info["N2-skel"] = dict(registers=skel.registers, local_bytes=skel.local_bytes)
        info["N2-noU"] = dict(registers=keep.registers, local_bytes=keep.local_bytes)
        meta["n2_blocks"] = nb

    # ---- timing, round robin
    flush = torch.empty(int(args.flush_l2_mb) << 18, dtype=torch.float32, device="cuda") if args.flush_l2_mb else None
    names = list(variants)
    for name in names:
        for _ in range(args.warmup):
            launch, a = variants[name]
            if a is not None:
                a[6].zero_()
            launch()
    torch.cuda.synchronize()
    times = {v: [] for v in names}
    for _ in range(args.rounds):
        for name in names:
            launch, a = variants[name]
            if a is not None:
                a[6].zero_()
            if flush is not None:
                flush.fill_(0.0)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            launch()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) * 1000.0)
    base_median = sorted(times["base"])[len(times["base"]) // 2]
    rows = {}
    for name in names:
        ordered = sorted(times[name])
        median = ordered[len(ordered) // 2]
        rows[name] = dict(median_us=median, min_us=ordered[0], max_us=ordered[-1], ratio=median / base_median,
                          **info.get(name, {}))
        print(f"wprime {name:20s} {median:9.1f} us  ratio {median / base_median:.3f}  {info.get(name, {})}",
              flush=True)
    if "N2-noU" in rows:
        floor = 6 * rows["N2-noU"]["median_us"]
        rows["N2-noU-x6"] = dict(median_us=floor, ratio=floor / base_median,
                                 note="6 vector paths x the optimistic skeleton; scalar paths, S preparation, "
                                      "receiver gather, radial tables, geometry and atomics not counted")
        print(f"wprime {'N2-noU-x6':20s} {floor:9.1f} us  ratio {floor / base_median:.3f}", flush=True)
    meta["variants"] = rows
    meta["shapes"] = dict(edges=edges, nodes=nodes, paths=[[q["a"], q["l"], q["L"]] for q in paths])
    (args.output / "lab.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
