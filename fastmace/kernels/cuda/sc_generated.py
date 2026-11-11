"""A cuEquivariance symmetric contraction with the density division before it, as generated straight-line kernels
(results/sc0-bound-20261109: bound and lab; Codex report v3 proposal 5).

On the H200, cuEq's ``uniform_1d`` kernels for OMAT-0 medium's interaction 0 are FP32-issue bound and execute
1,381 / 2,935 FP32 instructions per (atom, channel) forward / backward. The polynomial itself has 480 distinct
(weight, monomial, output) terms over 410 monomials; cuEq rebuilds each monomial's product per term. The kernels here
are rendered from the module's own descriptor (``merged_terms``: the descriptor's paths with the same weight segment,
sorted monomial and output summed) and compiled with NVRTC, one block of ``mul`` threads per atom, ``ch =
threadIdx.x``, on cuEq's ir_mul layouts (x [n, segments * mul], weights [elements, weight segments * mul]):

- forward ``x = m * s`` with ``s = 1 / (D + 1)`` (the division fused); each degree-2 prefix's product, then its
  degree-3 monomials, each term added to its (weight, output) accumulator with the coefficient as an immediate;
  ``y_C = sum_a w_a t_aC``;
- backward: ``q_aC = w_a dy_C``; per monomial the adjoint ``h = sum coef q``; for a prefix ``p = x_i x_j`` each
  monomial (i, j, k) adds ``h p`` to ``dx_k`` and ``h x_k`` to the pair adjoint ``H``, then ``dx_i += H x_j``,
  ``dx_j += H x_i`` (repeated indices come out right); ``dm = dx s``, ``dD = -s sum(dx x)`` (a block reduction).

Rounding: another summation order than cuEq's (rms differences about 3e-7 relative), so a switch that uses it is
checked with the force/stress gate (decision 66).
"""

from __future__ import annotations

import ctypes
from collections import defaultdict

import torch

_REGISTRY = {}  # kernel id -> GeneratedContraction (custom ops take the id; see oeq_streaming.runtime)


def merged_terms(polynomial):
    """``{(weight segment, sorted input segments, output segment): coefficient}`` of a cuEq ``SegmentedPolynomial``
    with inputs (weights, x) and one output, the paths with equal keys summed (x's operands are identical, so a
    monomial's permutations are one term)."""
    terms = defaultdict(float)
    for operation, stp in polynomial.operations:
        buffers = operation.buffers
        out_buffer = buffers[-1]
        for path in stp.paths:
            a, xs, c = None, [], None
            for buffer, index in zip(buffers, path.indices):
                if buffer == out_buffer and buffer >= polynomial.num_inputs:
                    c = index
                elif buffer == 0:
                    a = index
                else:
                    xs.append(index)
            coefficient = path.coefficients
            coefficient = float(coefficient.reshape(-1)[0]) if hasattr(coefficient, "reshape") else float(coefficient)
            terms[(a, tuple(sorted(xs)), c)] += coefficient
    return {key: value for key, value in terms.items() if value != 0.0}


def _literal(value):
    import numpy as np

    return float(np.float32(value)).hex() + "f"


def _prefix_order(monomials):
    return sorted({m[:2] for m in monomials if len(m) >= 2})


def render_forward(terms, segments_in, segments_w, segments_out, mul):
    """CUDA source of ``sc_generated_forward(m, d, w, sp, y)``."""
    monomials = sorted({m for _, m, _ in terms}, key=lambda m: (len(m), m))
    by_monomial = defaultdict(list)
    for (wa, mono, c), coefficient in terms.items():
        by_monomial[mono].append((wa, c, coefficient))
    groups = sorted({(wa, c) for wa, _, c in terms})
    acc = {g: f"t{g[0]}_{g[1]}" for g in groups}
    lines = [f'extern "C" __global__ void __launch_bounds__({mul}) sc_generated_forward(',
             "    const float* __restrict__ m, const float* __restrict__ d, const float* __restrict__ w,",
             "    const int* __restrict__ sp, float* __restrict__ y) {",
             "  const int a = blockIdx.x, ch = threadIdx.x;",
             f"  const float* xr = m + (long long)a * {segments_in * mul} + ch;",
             f"  const float* wr = w + (long long)sp[a] * {segments_w * mul} + ch;",
             "  const float s = 1.0f / (d[a] + 1.0f);"]
    lines += [f"  const float x{i} = __ldg(xr + {i * mul}) * s;" for i in range(segments_in)]
    lines.append("  float " + ", ".join(acc[g] for g in groups) + ";")
    started = set()

    def consume(mono, expr):
        for wa, c, coefficient in sorted(by_monomial.get(mono, ())):
            t = acc[(wa, c)]
            if (wa, c) in started:
                lines.append(f"  {t} = fmaf({_literal(coefficient)}, {expr}, {t});")
            else:
                lines.append(f"  {t} = {_literal(coefficient)} * {expr};")
                started.add((wa, c))

    for mono in (mm for mm in monomials if len(mm) == 1):
        consume(mono, f"x{mono[0]}")
    for prefix in _prefix_order(monomials):
        lines.append(f"  {{ const float p = x{prefix[0]} * x{prefix[1]};")
        consume(prefix, "p")
        for mono in (mm for mm in monomials if len(mm) == 3 and mm[:2] == prefix):
            lines.append(f"  {{ const float q = p * x{mono[2]};")
            consume(mono, "q")
            lines.append("  }")
        lines.append("  }")
    lines += [f"  float y{c} = 0.0f;" for c in range(segments_out)]
    lines += [f"  y{c} = fmaf(__ldg(wr + {wa * mul}), {acc[(wa, c)]}, y{c});" for wa, c in groups]
    lines += [f"  y[(long long)a * {segments_out * mul} + {c * mul} + ch] = y{c};" for c in range(segments_out)]
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_backward(terms, segments_in, segments_w, segments_out, mul):
    """CUDA source of ``sc_generated_backward(m, d, w, sp, g, dm, dd)``."""
    monomials = sorted({m for _, m, _ in terms}, key=lambda m: (len(m), m))
    by_monomial = defaultdict(list)
    for (wa, mono, c), coefficient in terms.items():
        by_monomial[mono].append((wa, c, coefficient))
    groups = sorted({(wa, c) for wa, _, c in terms})
    q = {g: f"q{g[0]}_{g[1]}" for g in groups}
    warps = mul // 32
    lines = [f'extern "C" __global__ void __launch_bounds__({mul}) sc_generated_backward(',
             "    const float* __restrict__ m, const float* __restrict__ d, const float* __restrict__ w,",
             "    const int* __restrict__ sp, const float* __restrict__ g, float* __restrict__ dm,",
             "    float* __restrict__ dd) {",
             "  const int a = blockIdx.x, ch = threadIdx.x;",
             f"  const float* xr = m + (long long)a * {segments_in * mul} + ch;",
             f"  const float* wr = w + (long long)sp[a] * {segments_w * mul} + ch;",
             f"  const float* gr = g + (long long)a * {segments_out * mul} + ch;",
             "  const float s = 1.0f / (d[a] + 1.0f);"]
    lines += [f"  const float x{i} = __ldg(xr + {i * mul}) * s;" for i in range(segments_in)]
    lines += [f"  const float g{c} = __ldg(gr + {c * mul});" for c in range(segments_out)]
    lines += [f"  const float {q[(wa, c)]} = __ldg(wr + {wa * mul}) * g{c};" for wa, c in groups]
    lines += [f"  float dx{i} = 0.0f;" for i in range(segments_in)]

    def adjoint(mono, target, started):
        for wa, c, coefficient in sorted(by_monomial.get(mono, ())):
            if started:
                lines.append(f"    {target} = fmaf({_literal(coefficient)}, {q[(wa, c)]}, {target});")
            else:
                lines.append(f"    {target} = {_literal(coefficient)} * {q[(wa, c)]};")
                started = True
        return started

    for mono in (mm for mm in monomials if len(mm) == 1):
        adjoint(mono, f"dx{mono[0]}", True)
    for i, j in _prefix_order(monomials):
        lines.append(f"  {{ const float p = x{i} * x{j}; float H = 0.0f;")
        pair_set = adjoint((i, j), "H", False)
        for mono in (mm for mm in monomials if len(mm) == 3 and mm[:2] == (i, j)):
            k = mono[2]
            lines.append("  { float h;")
            adjoint(mono, "h", False)
            lines.append(f"    dx{k} = fmaf(h, p, dx{k});")
            lines.append(f"    H = {f'fmaf(h, x{k}, H)' if pair_set else f'h * x{k}'}; }}")
            pair_set = True
        lines.append(f"    dx{i} = fmaf(H, x{j}, dx{i}); dx{j} = fmaf(H, x{i}, dx{j}); }}")
    lines.append("  float dot = 0.0f;")
    for i in range(segments_in):
        lines.append(f"  dot = fmaf(dx{i}, x{i}, dot);")
        lines.append(f"  dm[(long long)a * {segments_in * mul} + {i * mul} + ch] = dx{i} * s;")
    lines += ["  #pragma unroll",
              "  for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);",
              f"  __shared__ float red[{warps}];",
              "  if ((ch & 31) == 0) red[ch >> 5] = dot;",
              "  __syncthreads();",
              "  if (ch == 0) {",
              "    float total = 0.0f;",
              f"    for (int k = 0; k < {warps}; ++k) total += red[k];",
              "    dd[a] = -s * total;",
              "  }",
              "}"]
    return "\n".join(lines) + "\n"


def shape_of(module):
    """``(segments_in, segments_w, segments_out, mul)`` of a cuEq ``SymmetricContraction`` whose kernels can be
    generated, else None: float32, degree 1-3 in x, one multiplicity that is a multiple of 32 up to 1024, uniform
    segments of that size."""
    etp = getattr(module, "etp", None)
    polynomial = getattr(etp, "polynomial", etp)
    weight = getattr(module, "weight", None)
    if polynomial is None or not isinstance(weight, torch.Tensor) or weight.dtype != torch.float32:
        return None
    try:
        if polynomial.num_inputs != 2 or polynomial.num_outputs != 1:
            return None
        operands = list(polynomial.inputs) + list(polynomial.outputs)
        mul = int(weight.shape[-1])
        for operand in operands:
            if any(tuple(seg) != (mul,) for seg in operand.segments):
                return None
        for operation, stp in polynomial.operations:
            if len(operation.buffers) - 2 > 3:
                return None
    except (AttributeError, TypeError):
        return None
    if mul % 32 or not 32 <= mul <= 1024:
        return None
    return (len(operands[1].segments), len(operands[0].segments), len(operands[2].segments), mul)


class GeneratedContraction:
    """The compiled forward and backward of one contraction on the current CUDA device."""

    def __init__(self, module, factored=False):
        """``factored`` (``rewrites.sc_generated.SC_FACTORED``): the kernels of ``kernels.cuda.sc_factored``, the
        contraction through memoised CG pair products in a cheaper basis, with ``fold`` mapping the module's weights
        to that basis; falls back to the monomial kernels (``self.factored`` False) when the basis is not found."""
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check, _device_key, compile_cubin, content_id

        shape = shape_of(module)
        if shape is None:
            raise ValueError("sc_generated: the contraction's kernels cannot be generated")
        self.segments_in, self.segments_w, self.segments_out, self.mul = shape
        etp = module.etp
        terms = merged_terms(getattr(etp, "polynomial", etp))
        self.factored, self.basis = False, None
        if factored:
            from fastmace.kernels.cuda import sc_factored

            found = sc_factored.factor(terms, self.segments_in, self.segments_w, self.segments_out)
            if found is not None:
                program, features, self.basis = found
                self.factored = True
                # the backward capped at 128 registers per thread for mul = 128 (fac-pair4: 0.65 of the monomial pair
                # on the A100, against 0.72 uncapped)
                self.source = (sc_factored.render_forward(program, features, *shape, name="sc_generated_forward")
                               + "\n" + sc_factored.render_backward(program, features, *shape,
                                                                    name="sc_generated_backward", min_blocks=4))
        if not self.factored:
            self.source = (render_forward(terms, *shape) + "\n" + render_backward(terms, *shape))
        cubin = compile_cubin(self.source, torch.cuda.get_device_capability())
        torch.cuda.current_stream()  # make the primary context current
        self._module = _check(driver.cuModuleLoadData(cubin))
        self._forward = _check(driver.cuModuleGetFunction(self._module, b"sc_generated_forward"))
        self._backward = _check(driver.cuModuleGetFunction(self._module, b"sc_generated_backward"))
        self.id = content_id("GeneratedContraction", self.source, _device_key())
        _REGISTRY[self.id] = self

    def fold(self, weight):
        """The weights the kernels read: ``weight [elements, segments_w * mul]`` itself, or (factored) its image in the
        factored basis (float64 product, weight's dtype)."""
        if not self.factored:
            return weight
        from fastmace.kernels.cuda.sc_factored import fold_weights

        return fold_weights(weight, self.basis, self.segments_w, self.mul)

    def _launch(self, function, atoms, tensors):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check
        if atoms == 0:
            return
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
        values = tuple(t.data_ptr() for t in tensors)
        _check(driver.cuLaunchKernel(function, atoms, 1, 1, self.mul, 1, 1, 0, stream,
                                     (values, (ctypes.c_void_p,) * len(values)), 0))

    def forward(self, m, density, weight, species, y):
        self._launch(self._forward, m.shape[0], (m, density, weight, species, y))

    def backward(self, m, density, weight, species, dy, dm, dd):
        self._launch(self._backward, m.shape[0], (m, density, weight, species, dy, dm, dd))


@torch.library.custom_op("fastmace::sc_generated_forward", mutates_args=())
def sc_generated_forward(m: torch.Tensor, density: torch.Tensor, weight: torch.Tensor, species: torch.Tensor,
                         kernel_id: int) -> torch.Tensor:
    """``contraction(m / (density + 1))``: m [n, segments_in * mul] (ir_mul), density [n] or [n, 1], weight
    [elements, segments_w * mul], species [n] int32."""
    kernels = _REGISTRY[kernel_id]
    n = m.shape[0]
    y = m.new_empty((n, kernels.segments_out * kernels.mul))
    kernels.forward(m.contiguous(), density.reshape(n).contiguous(), weight.contiguous(), species.contiguous(), y)
    return y


@sc_generated_forward.register_fake
def _(m, density, weight, species, kernel_id):
    kernels = _REGISTRY[kernel_id]
    return m.new_empty((m.shape[0], kernels.segments_out * kernels.mul))


@torch.library.custom_op("fastmace::sc_generated_backward", mutates_args=())
def sc_generated_backward(m: torch.Tensor, density: torch.Tensor, weight: torch.Tensor, species: torch.Tensor,
                          dy: torch.Tensor, kernel_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(dm, ddensity)`` of ``sc_generated_forward`` for the output gradient ``dy``; ddensity has density's
    shape."""
    kernels = _REGISTRY[kernel_id]
    n = m.shape[0]
    dm = torch.empty_like(m, memory_format=torch.contiguous_format)
    dd = m.new_empty((n,))
    kernels.backward(m.contiguous(), density.reshape(n).contiguous(), weight.contiguous(), species.contiguous(),
                     dy.contiguous(), dm, dd)
    return dm, dd.reshape(density.shape)


@sc_generated_backward.register_fake
def _(m, density, weight, species, dy, kernel_id):
    return torch.empty_like(m, memory_format=torch.contiguous_format), density.new_empty(density.shape)


def _setup_context(ctx, inputs, output):
    m, density, weight, species, kernel_id = inputs
    ctx.save_for_backward(m, density, weight, species)
    ctx.kernel_id = kernel_id


def _backward(ctx, dy):
    m, density, weight, species = ctx.saved_tensors
    dm, dd = sc_generated_backward(m, density, weight, species, dy, ctx.kernel_id)
    return dm, dd, None, None, None


sc_generated_forward.register_autograd(_backward, setup_context=_setup_context)


__all__ = ["GeneratedContraction", "merged_terms", "render_backward", "render_forward", "sc_generated_backward",
           "sc_generated_forward", "shape_of"]
