"""The radial path's r-gradient inside the streaming backward (``wprime_backward``, opt-9 OPT9-T03).

With ``radial_tables`` the folded radial layer's input is a piecewise Chebyshev series in the edge length,
``a(r) = sum_t T_t(x) C_t`` per (species pair, piece), and the per-edge weights are ``w = a [W, b]^T``. Today's
backward (``chunked_conv_backward``) writes the weight gradient ``dw [E, Q]`` in L2-sized chunks, forms ``da =
dw [W, b]`` on a side stream and hands ``da`` to the table's backward, ``dr = <da, a'(r)>``. Within a piece

    dL/dr = sum_c dw_c w'_c(r),   w'(r) = sum_t T_t(x) G'_t,   G' = C' [W, b]^T  ([terms, Q] per (pair, piece)),

so the edge-owned streaming backward can dot each lane's weight gradients with ``w'`` in registers and never
write ``dw``: no chunk stores, no input gradient, no 66-column table backward (``plans/opt-9.md``). ``G'`` is
the derivative half of ``w_table.fold_tables`` without its padding (float32 tables: ``D`` terms).

The kernel is generated from the streaming schedule (``FactorizedComputationSchedule``) with the vendored Jinja
environment. Its body is the vendored ``oeq_projected_backward`` (OpenEquivariance pull request 223, BSD
3-Clause, see NOTICE.md; ``factorized_projected.cuh`` itself stays byte-identical), so ``dx`` and ``dsh`` are the
same sums in the same order; the ``dweights`` store is replaced by the dot. Each edge locates its piece with the
tables' rule (one bin gather; ``x = (max(r, r_lo) - start) * scale - 1``); at or beyond the cutoff (also the
neighbour list's padding) and below ``r_lo`` the slope is zero, as the tables' own, and ``G'`` is not read.

The edge density's gradient still reaches ``r`` through the table's backward on the density column alone
(``density_backward``). Exactness: rounding (the series and the folded product are reassociated); the sender
gradients are added atomically, as in the vendored backward.
"""

from __future__ import annotations

import ctypes
import itertools

import torch

_REGISTRY = {}
_IDS = itertools.count()

TEMPLATE = r"""// oeq_projected_backward_wprime (FastMACE opt-9 OPT9-T03), adapted from the vendored
// oeq_projected_backward (OpenEquivariance PR 223, BSD 3-Clause, see NOTICE.md).
using int64_t = signed long long;
using int32_t = int;
using scalar_t = {{ scalar }};
using acc_t = {{ scalar }};

constexpr int INPUT_DIM = {{ schedule.input_dim }};
constexpr int EDGE_DIM = {{ schedule.edge_dim }};
constexpr int OUTPUT_DIM = {{ schedule.output_dim }};
constexpr int WEIGHT_DIM = {{ schedule.weight_numel }};
constexpr int CHANNEL_DIM = {{ schedule.channels }};
constexpr int LOGICAL_GROUP_SIZE =
    {{ schedule.launch_config.logical_cohort_width }};
constexpr int TERMS = {{ terms }};
constexpr unsigned int FULL_MASK = 0xffffffffu;

{% macro node_offset(start, component, irrep_dim) -%}
    {%- if layout == "ir_mul" -%}
        {{ start + component * schedule.channels }} + channel
    {%- else -%}
        {{ start + component }} + channel * {{ irrep_dim }}
    {%- endif -%}
{%- endmacro %}

// One logical thread group owns an edge; each lane owns channels separated by LOGICAL_GROUP_SIZE.
// dx and dsh as in oeq_projected_backward; in place of dweights, dr[e] = sum over the edge's weights of
// dweights * w'(r_e), w' from the derivative table gprime [pairs * pieces, TERMS, WEIGHT_DIM].
extern "C" __global__ void oeq_projected_backward_wprime(
    int64_t node_count,
    int64_t edge_count,
    const scalar_t* x,
    const scalar_t* sh,
    const scalar_t* weights,
    const int32_t* senders,
    const int32_t* receivers,
    const scalar_t* dout,
    scalar_t* dx,
    scalar_t* dsh,
    const scalar_t* lengths,
    const int32_t* pairs,
    const scalar_t* __restrict__ gprime,
    const scalar_t* starts,
    const scalar_t* scales,
    const int32_t* bins,
    scalar_t r_lo,
    scalar_t bin_scale,
    scalar_t r_max,
    int32_t n_bins,
    int32_t pieces,
    scalar_t* dr) {

    const int lane = threadIdx.x % LOGICAL_GROUP_SIZE;
    const int logical_group = threadIdx.x / LOGICAL_GROUP_SIZE;
    const int64_t e =
        int64_t(blockIdx.x) * (blockDim.x / LOGICAL_GROUP_SIZE) + logical_group;
    if (e >= edge_count) return;

    const int32_t sender = senders[e];
    const int32_t receiver = receivers[e];
    if (sender < 0 || sender >= node_count
        || receiver < 0 || receiver >= node_count) {
        if (lane == 0) dr[e] = scalar_t(0);
        return;
    }

    const int64_t input_base = int64_t(sender) * INPUT_DIM;
    const int64_t output_base = int64_t(receiver) * OUTPUT_DIM;
    const int64_t edge_base = e * EDGE_DIM;
    const int64_t weight_base = e * WEIGHT_DIM;

    // The radial tables' piece of r and its Chebyshev argument; no slope at or beyond the cutoff, below r_lo.
    const scalar_t r = lengths[e];
    const bool sloped = r < r_max && r >= r_lo;
    const scalar_t clamped = r > r_lo ? r : r_lo;
    int32_t bin = int32_t((clamped - r_lo) * bin_scale);
    bin = bin < n_bins - 1 ? bin : n_bins - 1;
    const int32_t piece = bins[bin];
    const scalar_t t = (clamped - starts[piece]) * scales[piece] - scalar_t(1);
    const scalar_t* g = gprime + (int64_t(pairs[e]) * pieces + piece) * TERMS * WEIGHT_DIM;
    scalar_t cheb[TERMS];
    cheb[0] = scalar_t(1);
    if (TERMS > 1) cheb[1] = t;
    #pragma unroll
    for (int k = 2; k < TERMS; ++k) cheb[k] = scalar_t(2) * t * cheb[k - 1] - cheb[k - 2];

    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};
    acc_t radial_gradient = acc_t(0);
    for (int channel = lane; channel < CHANNEL_DIM;
         channel += LOGICAL_GROUP_SIZE) {
        {%- for slot in schedule.input_gradient_slots %}
            acc_t {{ slot.name }} = acc_t(0);
        {%- endfor %}

        {% for path in schedule.paths %}
            {% for edge_channel in range(path.edge_mul) %} {
                const int weight_index =
                    {{ path.weight_start + edge_channel * schedule.channels }} + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);

                {% for output in path.outputs %} {
                    {%- set slot = output.accumulator %}
                    const int64_t output_index =
                        output_base + {{ node_offset(slot.start, slot.component, slot.irrep_dim) }};
                    const scalar_t output_gradient = dout[output_index];

                    {% for term in output.terms %} {
                        const int64_t input_index = input_base
                            + {{ node_offset(path.input_start, term.input_component, path.input_irrep_dim) }};
                        const int edge_component_index =
                            {{ path.edge_start + term.edge_component }}
                            + {{ edge_channel }} * {{ path.edge_irrep_dim }};

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * {{ cpp_scalar_literal(term.coefficient, scalar) }};
                        {{ term.input_accumulator.name }} += output_gradient
                            * radial_weight * edge_value * {{ cpp_scalar_literal(term.coefficient, scalar) }};
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * {{ cpp_scalar_literal(term.coefficient, scalar) }};
                    } {%- endfor %}
                } {%- endfor %}
                if (sloped) {
                    acc_t slope = acc_t(0);
                    #pragma unroll
                    for (int k = 0; k < TERMS; ++k)
                        slope += cheb[k] * __ldg(g + k * WEIGHT_DIM + weight_index);
                    radial_gradient += weight_gradient * slope;
                }
            } {%- endfor %}
        {%- endfor %}

        {%- for slot in schedule.input_gradient_slots %}
            {{ atomic_add }}(
                dx + input_base + {{ node_offset(slot.start, slot.component, slot.irrep_dim) }},
                scalar_t({{ slot.name }}));
        {%- endfor %}
    }

    // Reduce within the logical 32-thread edge owner. Lane 0 writes dsh[e, :] and dr[e].
    for (int offset = 16; offset > 0; offset >>= 1) {
        for (int component = 0; component < EDGE_DIM; ++component)
            edge_gradient[component] +=
                {{ shfl_down_width("edge_gradient[component]", "offset") }};
        radial_gradient += {{ shfl_down_width("radial_gradient", "offset") }};
    }

    if (lane == 0) {
        for (int component = 0; component < EDGE_DIM; ++component)
            dsh[edge_base + component] = scalar_t(edge_gradient[component]);
        dr[e] = scalar_t(radial_gradient);
    }
}
"""


def render(schedule, scalar, terms):
    """CUDA source of ``oeq_projected_backward_wprime`` for a schedule, ``scalar`` ("float" or "double") and the
    derivative table's ``terms``."""
    from fastmace.kernels.oeq_streaming.jinja_env import get_jinja_environment

    if schedule.layout not in ("ir_mul", "mul_ir"):
        raise ValueError(f"unknown layout {schedule.layout!r}")
    if int(terms) < 1:
        raise ValueError("the derivative table needs at least one term")
    env = get_jinja_environment(backend="cuda", warp_size=schedule.launch_config.logical_cohort_width)
    return env.from_string(TEMPLATE).render(scalar=scalar, schedule=schedule, layout=schedule.layout,
                                            terms=int(terms))


def derivative_table(dcoef, weight):
    """``G' [pairs * pieces, D', Q]`` from a table's ``dcoef [P, K, D', J]`` and the folded weight ``[Q, J - 1]``
    (the table's last column is the edge density): ``w_table.fold_tables``' derivative half without its padding
    term and zero group. Computed in float64, stored in ``dcoef``'s dtype."""
    pairs, pieces, terms, _ = dcoef.shape
    width = weight.shape[1]
    d = torch.einsum("pktj,qj->pktq", dcoef[..., :width].to(torch.float64),
                     weight.detach().to(device=dcoef.device, dtype=torch.float64))
    return d.reshape(pairs * pieces, terms, -1).to(dcoef.dtype).contiguous()


def density_column(dcoef):
    """The derivative series of the edge density alone, ``[P, K, D', 1]`` (``density_backward``)."""
    return dcoef[..., -1:].contiguous()


def density_backward(r, pair, density_dcoef, starts, scales, bins, grad_density, r_lo, bin_scale, r_max):
    """``dL/dr [E]`` of the edge density alone: the table's backward on its density column, with an ``[E, 0]``
    folded-input gradient (a view of ``grad_density``)."""
    from fastmace.kernels.triton.radial_table import radial_table_backward

    grad_density = grad_density.reshape(-1, 1).contiguous()
    return radial_table_backward(r, pair, density_dcoef, starts, scales, bins, grad_density[:, :0], grad_density,
                                 float(r_lo), float(bin_scale), float(r_max)).reshape(-1)


def wprime_arguments(r, pair, starts, scales, bins, r_lo, bin_scale, r_max):
    """Per edge the Chebyshev argument ``x`` and the ``G'`` group ``pair * pieces + piece`` (int32), ``-1`` where the
    slope is zero (at or beyond the cutoff, below ``r_lo``): the tables' rule, as the kernel computes it."""
    r = r.reshape(-1).to(starts.dtype)
    clamped = torch.clamp(r, min=r_lo)
    index = torch.clamp(((clamped - r_lo) * bin_scale).to(torch.int64), 0, bins.shape[0] - 1)
    piece = bins[index].to(torch.int64)
    x = (clamped - starts[piece]) * scales[piece] - 1.0
    sloped = (r < r_max) & (r >= r_lo)
    group = torch.where(sloped, pair.reshape(-1).to(torch.int64) * starts.shape[0] + piece,
                        torch.full_like(piece, -1))
    return x.contiguous(), group.to(torch.int32).contiguous()


def wprime_reference(dweights, r, pair, gprime, starts, scales, bins, r_lo, bin_scale, r_max):
    """PyTorch ``dL/dr [E]`` of the per-edge weights' gradient ``dweights [E, Q]`` (CPU tests and checks)."""
    x, group = wprime_arguments(r, pair, starts, scales, bins, r_lo, bin_scale, r_max)
    terms = gprime.shape[1]
    cheb = [torch.ones_like(x), x]
    for _ in range(2, terms):
        cheb.append(2.0 * x * cheb[-1] - cheb[-2])
    basis = torch.stack(cheb[:terms], dim=1)  # [E, terms]
    slopes = torch.einsum("et,etq->eq", basis, gprime[group.clamp(min=0).to(torch.int64)])
    dr = (dweights.to(slopes.dtype) * slopes).sum(dim=1)
    return torch.where(group >= 0, dr, torch.zeros_like(dr))


class WPrimeBackward:
    """Compiled ``oeq_projected_backward_wprime`` for the streaming kernels of one interaction
    (``StreamingKernels``) and a derivative table of ``terms`` terms."""

    def __init__(self, kernels, terms, device=None):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin
        scalar = "float" if "using scalar_t = float;" in kernels.source else "double"
        device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        self.terms = int(terms)
        self.scalar = scalar
        self.num_threads, self.cohort, self.shared_memory = kernels.num_threads, kernels.cohort, kernels.shared_memory
        self.output_dim = kernels.output_dim
        self.source = render(kernels.schedule, scalar, self.terms)
        cubin = compile_cubin(self.source, torch.cuda.get_device_capability(device))
        torch.cuda.current_stream(device)
        self._module = _check(driver.cuModuleLoadData(cubin))
        self._function = _check(driver.cuModuleGetFunction(self._module, b"oeq_projected_backward_wprime"))
        self.id = next(_IDS)
        _REGISTRY[self.id] = self

    def __call__(self, x, sh, weights, senders, receivers, dout, dx, dsh, r, pair, gprime, starts, scales, bins,
                 r_lo, bin_scale, r_max, dr):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check
        nodes, edges = x.shape[0], sh.shape[0]
        blocks = (edges * self.cohort + self.num_threads - 1) // self.num_threads
        if not blocks:
            return
        p = ctypes.c_void_p
        real = ctypes.c_float if self.scalar == "float" else ctypes.c_double
        values = (nodes, edges, x.data_ptr(), sh.data_ptr(), weights.data_ptr(), senders.data_ptr(),
                  receivers.data_ptr(), dout.data_ptr(), dx.data_ptr(), dsh.data_ptr(), r.data_ptr(),
                  pair.data_ptr(), gprime.data_ptr(), starts.data_ptr(), scales.data_ptr(), bins.data_ptr(),
                  r_lo, bin_scale, r_max, bins.shape[0], starts.shape[0], dr.data_ptr())
        types = (ctypes.c_longlong, ctypes.c_longlong, p, p, p, p, p, p, p, p, p, p, p, p, p, p, real, real, real,
                 ctypes.c_int, ctypes.c_int, p)
        stream = driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        _check(driver.cuLaunchKernel(self._function, blocks, 1, 1, self.num_threads, 1, 1, self.shared_memory,
                                     stream, (values, types), 0))


@torch.library.custom_op("fastmace::streaming_conv_wprime_backward", mutates_args=())
def streaming_conv_wprime_backward(x: torch.Tensor, sh: torch.Tensor, weights: torch.Tensor, senders: torch.Tensor,
                                   receivers: torch.Tensor, dout: torch.Tensor, r: torch.Tensor, pair: torch.Tensor,
                                   gprime: torch.Tensor, starts: torch.Tensor, scales: torch.Tensor,
                                   bins: torch.Tensor, r_lo: float, bin_scale: float, r_max: float,
                                   wprime_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dx, dsh, dL/dr [E] of the radial path) through the kernel ``wprime_id``; one launch over all edges."""
    kernel = _REGISTRY[wprime_id]
    x, sh, weights, dout = x.contiguous(), sh.contiguous(), weights.contiguous(), dout.contiguous()
    dx = torch.zeros_like(x)  # sender gradients are added atomically
    dsh = torch.empty_like(sh)
    dr = sh.new_empty((sh.shape[0],))
    kernel(x, sh, weights, senders.contiguous(), receivers.contiguous(), dout, dx, dsh,
           r.reshape(-1).contiguous(), pair.contiguous(), gprime, starts, scales, bins, r_lo, bin_scale, r_max, dr)
    return dx, dsh, dr


@streaming_conv_wprime_backward.register_fake
def _(x, sh, weights, senders, receivers, dout, r, pair, gprime, starts, scales, bins, r_lo, bin_scale, r_max,
      wprime_id):
    return torch.empty_like(x), torch.empty_like(sh), sh.new_empty((sh.shape[0],))


@torch.library.custom_op("fastmace::streaming_conv_wprime_backward_recompute", mutates_args=())
def streaming_conv_wprime_backward_recompute(x: torch.Tensor, sh: torch.Tensor, senders: torch.Tensor,
                                             receivers: torch.Tensor, dout: torch.Tensor, r: torch.Tensor,
                                             pair: torch.Tensor, gprime: torch.Tensor, coef: torch.Tensor,
                                             weight: torch.Tensor, starts: torch.Tensor, scales: torch.Tensor,
                                             bins: torch.Tensor, r_lo: float, bin_scale: float, r_max: float,
                                             wprime_id: int, rows: int) -> tuple[torch.Tensor, torch.Tensor,
                                                                                 torch.Tensor]:
    """``streaming_conv_wprime_backward`` without the stored per-edge weights (``recompute_conv_weights``,
    plans/capacity.md CAP-T07): the edges run in chunks of ``rows``, and each chunk's weights are built again from
    its edge lengths (the radial table, then ``a @ weight.T``, as the forward's chain builds them) into a
    chunk-sized buffer just before the kernel reads them. Per edge the kernel computes what the one-launch backward
    does; only the order of the atomic sender additions differs, as between any two runs."""
    from fastmace.kernels.triton.radial_table import radial_table_forward

    kernel = _REGISTRY[wprime_id]
    x, sh, dout = x.contiguous(), sh.contiguous(), dout.contiguous()
    senders, receivers = senders.contiguous(), receivers.contiguous()
    r, pair = r.reshape(-1).contiguous(), pair.contiguous()
    edges = sh.shape[0]
    dx = torch.zeros_like(x)  # sender gradients are added atomically, over all chunks
    dsh = torch.empty_like(sh)
    dr = sh.new_empty((edges,))
    rows = max(1, min(int(rows), edges))
    buffer = sh.new_empty((rows, weight.shape[0]))
    for start in range(0, edges, rows):
        stop = min(edges, start + rows)
        a, _ = radial_table_forward(r[start:stop], pair[start:stop], coef, starts, scales, bins, r_lo, bin_scale,
                                    r_max)
        weights = torch.matmul(a, weight.t(), out=buffer[:stop - start])
        kernel(x, sh[start:stop], weights, senders[start:stop], receivers[start:stop], dout, dx, dsh[start:stop],
               r[start:stop], pair[start:stop], gprime, starts, scales, bins, r_lo, bin_scale, r_max,
               dr[start:stop])
    return dx, dsh, dr


@streaming_conv_wprime_backward_recompute.register_fake
def _(x, sh, senders, receivers, dout, r, pair, gprime, coef, weight, starts, scales, bins, r_lo, bin_scale, r_max,
      wprime_id, rows):
    return torch.empty_like(x), torch.empty_like(sh), sh.new_empty((sh.shape[0],))


class _WPrimeConvolution(torch.autograd.Function):
    """Edge lengths -> (message, edge density) through the radial table, the folded layer and the streaming
    convolution; the backward is ``streaming_conv_wprime_backward`` plus the density column's."""

    @staticmethod
    def forward(ctx, r, pair, coef, density_dcoef, starts, scales, bins, r_lo, bin_scale, r_max, weight, gprime, x,
                sh, senders, receivers, row_ptr, kernel_id, fused_id, wprime_id, recompute_rows=0):
        from fastmace.kernels.oeq_streaming.runtime import streaming_conv_forward
        from fastmace.kernels.triton.radial_table import radial_table_forward

        a, density = radial_table_forward(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max)
        if fused_id >= 0:  # fused_folded_forward (opt-8 OPT8-T02): w stored by the message kernel
            from fastmace.kernels.oeq_streaming.fused_forward import fused_folded_forward
            message, weights = fused_folded_forward(a, weight, x, sh, senders, receivers, row_ptr, fused_id)
        else:
            weights = a @ weight.t()
            message = streaming_conv_forward(x, sh, weights, senders, row_ptr, kernel_id)
        if recompute_rows:  # recompute_conv_weights (CAP-T07): the backward builds the weights again per chunk
            weights = coef.new_empty((0,))
        ctx.save_for_backward(x, sh, weights, senders, receivers, r, pair, gprime, density_dcoef, starts, scales,
                              bins, coef, weight)
        ctx.constants, ctx.wprime_id, ctx.recompute_rows = (r_lo, bin_scale, r_max), wprime_id, int(recompute_rows)
        return message, density

    @staticmethod
    def backward(ctx, dout, ddensity):
        (x, sh, weights, senders, receivers, r, pair, gprime, density_dcoef, starts, scales, bins, coef,
         weight) = ctx.saved_tensors
        if dout is None:
            dout = x.new_zeros((x.shape[0], _REGISTRY[ctx.wprime_id].output_dim))
        if ctx.recompute_rows:
            dx, dsh, dr = streaming_conv_wprime_backward_recompute(
                x, sh, senders, receivers, dout, r, pair, gprime, coef, weight, starts, scales, bins, *ctx.constants,
                ctx.wprime_id, ctx.recompute_rows)
        else:
            dx, dsh, dr = streaming_conv_wprime_backward(x, sh, weights, senders, receivers, dout, r, pair, gprime,
                                                         starts, scales, bins, *ctx.constants, ctx.wprime_id)
        if ddensity is not None:
            dr = dr + density_backward(r, pair, density_dcoef, starts, scales, bins, ddensity, *ctx.constants)
        return (dr.reshape(r.shape), None, None, None, None, None, None, None, None, None, None, None, dx, dsh,
                None, None, None, None, None, None, None)


class WPrime(torch.nn.Module):
    """One interaction's ``wprime_backward``: the derivative table ``G'``, the density's derivative series and
    the compiled kernel, built from its radial table, folded weight and streaming kernels."""

    def __init__(self, table, weight, kernels):
        super().__init__()
        self.register_buffer("gprime", derivative_table(table.dcoef, weight), persistent=False)
        self.register_buffer("density_dcoef", density_column(table.dcoef), persistent=False)
        self.kernel = WPrimeBackward(kernels, self.gprime.shape[1], self.gprime.device)

    def convolution(self, lengths, pair, table, weight, x, sh, senders, receivers, row_ptr, kernels, fused=None,
                    recompute_rows=0):
        """``(message, edge density)`` of edge lengths ``lengths`` ``[E, 1]`` (differentiable in lengths, x, sh).
        ``recompute_rows`` (``recompute_conv_weights``): keep no per-edge weights for the backward, which builds
        them again in chunks of that many edges."""
        if weight.requires_grad:
            raise ValueError("wprime_backward: the folded weight must not require a gradient")
        return _WPrimeConvolution.apply(lengths, pair, table.coef, self.density_dcoef, table.starts, table.scales,
                                        table.bins, float(table.r_lo), float(table.bin_scale), float(table.r_max),
                                        weight, self.gprime, x, sh, senders, receivers, row_ptr, kernels.id,
                                        -1 if fused is None else fused.id, self.kernel.id, int(recompute_rows))

    def extra_repr(self):
        return f"groups={self.gprime.shape[0]}, terms={self.gprime.shape[1]}, outputs={self.gprime.shape[2]}"


__all__ = ["WPrime", "WPrimeBackward", "density_backward", "density_column", "derivative_table", "render",
           "streaming_conv_wprime_backward", "streaming_conv_wprime_backward_recompute", "wprime_arguments",
           "wprime_reference"]
