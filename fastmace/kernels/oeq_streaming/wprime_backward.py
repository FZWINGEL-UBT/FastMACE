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

import torch

from fastmace.kernels.gpu_models import GPUModels

_REGISTRY = {}
# The w-rebuilding backward's launch for recompute_conv_weights: one path part, four edges per batch (the fastest of
# the shapes timed in docs/validation/capacity-T08-inventory.md section 2).
REBUILD_PARTS = 1
REBUILD_BATCH = 4
# The stored per-edge weights are read once: load them with the streaming cache hint (``__ldcs``, evict-first), so
# they do not displace the derivative table and the receivers' output gradients in L1/L2 (OMAT-0 kernel lab,
# 3 November 2026: the wprime kernel 3.4 % faster on the RTX 5060 Ti). Bitwise. A module constant for the A/B.
STREAM_W = False
# The derivative table's layout for the slope loop. 0: term-major G' [groups, TERMS, Q], one scalar load per term and
# weight. 2 or 4 (float32 tables): q-major [groups, Q, TERMS rounded up to a multiple of SLOPE_VECTOR], each weight's
# terms loaded as float2/float4 (OMAT-0 kernel lab, 3 November 2026: the wprime kernel 8.7 % faster with float2 on
# the RTX 5060 Ti, 2.6 % with float4). The same sums in the same order; the padding terms are zero. The rebuild
# backward (recompute_conv_weights) reads the term-major table. A module constant for the A/B.
SLOPE_VECTOR = 0
# w and its slope from one table (OMAT-0 kernel lab, 4 November 2026). False: the fused forward stores w [E, Q] and the
# backward reads it with the derivative table G'. True (float32 tables): the backward reads the value table
# G = C [W, b]^T instead, q-major with float2 loads ([groups, Q, D + 1 rounded up to even]), and evaluates both
# w = sum_k T_k(x) G_k and w' = sum_k T_k'(x) scale G_k from the one slab, so the forward stores no w (its
# message-only kernel) and the backward reads neither w nor G'. Rounding: the backward's w is the table's series, not
# the forward's a . W product. None: on for the generic family's wprime_backward (OMAT-0, where it was measured) on
# TABLE_W_GPUS (decision 66; owner, 4 November 2026: "Enable it", every A100 model included), off for MH-1;
# True / False: an A/B side's choice on every GPU and model.
TABLE_W = None
# Benchmark_loup (OMAT-0 medium, SrTiO3 5,000 atoms), MD step b/a in both orders: RTX 5060 Ti 0.888 / 0.892, RTX 4000
# Ada 0.928 / 0.931, A100-SXM4-80GB 0.943 / 0.943; force gate passes (results/omat0-kernel-lab-20261103 section 16).
TABLE_W_GPUS = GPUModels(("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation", "NVIDIA A100-SXM4-80GB"))
TABLE_W_VECTOR = 2
# The message in linear_1's block layout (OMAT-0 work, 4 November 2026): with TABLE_W and the fused forward, the
# message-only kernel writes each same-irrep group of the grouped BlockLinear's input as its own [N * dim, P * C]
# tensor, the backward reads the groups' gradients through one pointer each, and linear_1 multiplies the groups
# without the relayout copies (RTX 5060 Ti: about 1.15 ms per step). Bitwise the same message. None: on for
# BLOCK_MESSAGE_GPUS (decision 66; owner, 4 November 2026: "Yes, Ada and A100"); True / False: an A/B side's
# choice on every GPU.
BLOCK_MESSAGE = None
# Benchmark_loup MD step b/a (a = the default with TABLE_W) in both orders: RTX 4000 Ada 0.974 / 0.971,
# A100-SXM4-80GB 0.983 / 0.977 (every A100 model, GPUModels); the RTX 5060 Ti measured 1.060 and keeps it off.
BLOCK_MESSAGE_GPUS = GPUModels(("NVIDIA RTX 4000 Ada Generation", "NVIDIA A100-SXM4-80GB"))
# The tensor product's backward factored per path (OMAT-0 TP backward screen, 4 November 2026). False: per
# Clebsch-Gordan term the vendored three updates (weight, input and edge gradient; about 7 floating-point operations).
# True: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i (3 operations per term), then the weight gradient
# sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j. Rounding (the same sums regrouped); every wprime kernel built while
# it is set. A module constant for the A/B (kernel lab on the RTX 4000 Ada: 0.903 of the TABLE_W wprime kernel).
FACTORED = False

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
constexpr int PADDED_TERMS = {{ padded_terms }};  // a q-major G' row: TERMS, then zero terms (SLOPE_VECTOR)
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
// dweights * w'(r_e), w' from the derivative table gprime [pairs * pieces, TERMS, WEIGHT_DIM] (q-major with
// SLOPE_VECTOR: [pairs * pieces, WEIGHT_DIM, PADDED_TERMS]).
extern "C" __global__ void oeq_projected_backward_wprime(
    int64_t node_count,
    int64_t edge_count,
    const scalar_t* x,
    const scalar_t* sh,
    const scalar_t* weights,
    const int32_t* senders,
    const int32_t* receivers,
{%- if blocks %}
{%- for g in range(n_blocks) %}
    const scalar_t* __restrict__ dout{{ g }},
{%- endfor %}
{%- else %}
    const scalar_t* dout,
{%- endif %}
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
    const scalar_t* g = gprime + (int64_t(pairs[e]) * pieces + piece) * PADDED_TERMS * WEIGHT_DIM;
    scalar_t cheb[PADDED_TERMS];
    cheb[0] = scalar_t(1);
    if (PADDED_TERMS > 1) cheb[1] = t;
    #pragma unroll
    for (int k = 2; k < PADDED_TERMS; ++k) cheb[k] = scalar_t(2) * t * cheb[k - 1] - cheb[k - 2];
    {%- if table_w %}
    // TABLE_W: w from the value series (zero at or beyond the cutoff) and its r-derivative T_k'(x) * scale,
    // T_k' = k U_(k-1).
    const bool inside = r < r_max;
    scalar_t dcheb[TERMS];
    {
        const scalar_t sc = scales[piece];
        scalar_t um1 = scalar_t(0), u = scalar_t(1);  // U_(k-2), U_(k-1)
        dcheb[0] = scalar_t(0);
        #pragma unroll
        for (int k = 1; k < TERMS; ++k) {
            dcheb[k] = scalar_t(k) * u * sc;
            const scalar_t un = scalar_t(2) * t * u - um1;
            um1 = u;
            u = un;
        }
    }
    {%- endif %}

    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};
    acc_t radial_gradient = acc_t(0);
    for (int channel = lane; channel < CHANNEL_DIM;
         channel += LOGICAL_GROUP_SIZE) {
        {%- for slot in schedule.input_gradient_slots %}
            acc_t {{ slot.name }} = acc_t(0);
        {%- endfor %}

        {% for path in schedule.paths %}
            {%- set path_index = loop.index0 %}
            {% for edge_channel in range(path.edge_mul) %} {
                const int weight_index =
                    {{ path.weight_start + edge_channel * schedule.channels }} + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                {%- if table_w %}
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const {{ vector_type }}* gq =
                        reinterpret_cast<const {{ vector_type }}*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / {{ slope_vector }}; ++k) {
                        const {{ vector_type }} v = __ldg(gq + k);
                        {%- for component in "xyzw"[:slope_vector] %}
                        gv[{{ slope_vector }} * k + {{ loop.index0 }}] = v.{{ component }};
                        {%- endfor %}
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = scalar_t(0);
                #pragma unroll
                for (int k = 0; k < TERMS; ++k) radial_weight += cheb[k] * gv[k];
                {%- else %}
                const scalar_t radial_weight = {{ "__ldcs(weights + radial_weight_index)" if stream_w
                                                  else "weights[radial_weight_index]" }};
                {%- endif %}
                acc_t weight_gradient = acc_t(0);
                {%- if factored %}
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                {%- for i in factored[path_index].inputs %}
                acc_t u_{{ i }} = acc_t(0);
                {%- endfor %}
                {%- for j in factored[path_index].edges %}
                acc_t v_{{ j }} = acc_t(0);
                {%- endfor %}
                {%- endif %}

                {% for output in path.outputs %} {
                    {%- set slot = output.accumulator %}
                    {%- if blocks %}
                    {%- set bk = blocks[slot.start] %}
                    const scalar_t output_gradient = dout{{ bk[0] }}[int64_t(receiver) * {{ bk[1] }}
                        + {{ bk[2] + slot.component * bk[3] }} + channel];
                    {%- else %}
                    const int64_t output_index =
                        output_base + {{ node_offset(slot.start, slot.component, slot.irrep_dim) }};
                    const scalar_t output_gradient = dout[output_index];
                    {%- endif %}

                    {%- if factored %}
                    {%- for term in output.terms %} {
                        const scalar_t input_value =
                            x[input_base
                              + {{ node_offset(path.input_start, term.input_component, path.input_irrep_dim) }}];
                        const scalar_t edge_value =
                            sh[edge_base
                               + {{ path.edge_start + term.edge_component + edge_channel * path.edge_irrep_dim }}];
                        {%- if term.coefficient == 1.0 %}
                        const scalar_t scaled = output_gradient;
                        {%- else %}
                        const scalar_t scaled = output_gradient * {{ cpp_scalar_literal(term.coefficient, scalar) }};
                        {%- endif %}
                        u_{{ term.input_component }} += scaled * edge_value;
                        v_{{ term.edge_component }} += scaled * input_value;
                    } {%- endfor %}
                    {%- else %}
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
                    {%- endif %}
                } {%- endfor %}
                {%- if factored %}
                {%- for i, slot in factored[path_index].inputs.items() %}
                weight_gradient +=
                    x[input_base + {{ node_offset(path.input_start, i, path.input_irrep_dim) }}] * u_{{ i }};
                {{ slot }} += radial_weight * u_{{ i }};
                {%- endfor %}
                {%- for j in factored[path_index].edges %}
                edge_gradient[{{ path.edge_start + j + edge_channel * path.edge_irrep_dim }}] +=
                    radial_weight * v_{{ j }};
                {%- endfor %}
                {%- endif %}
                if (sloped) {
                    acc_t slope = acc_t(0);
                    {%- if table_w %}
                    #pragma unroll
                    for (int k = 1; k < TERMS; ++k) slope += dcheb[k] * gv[k];
                    {%- elif slope_vector %}
                    const {{ vector_type }}* gq =
                        reinterpret_cast<const {{ vector_type }}*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / {{ slope_vector }}; ++k) {
                        const {{ vector_type }} v = __ldg(gq + k);
                        {%- for component in "xyzw"[:slope_vector] %}
                        slope += cheb[{{ slope_vector }} * k + {{ loop.index0 }}] * v.{{ component }};
                        {%- endfor %}
                    }
                    {%- else %}
                    #pragma unroll
                    for (int k = 0; k < TERMS; ++k)
                        slope += cheb[k] * __ldg(g + k * WEIGHT_DIM + weight_index);
                    {%- endif %}
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


def padded_terms(terms, vector):
    """A q-major derivative table's row length: ``terms`` rounded up to a multiple of ``vector`` (0: ``terms``)."""
    return -(-int(terms) // vector) * vector if vector else int(terms)


def factored_paths(schedule):
    """Per path of ``schedule`` (``FACTORED``): ``inputs`` {input component: its input-gradient accumulator} and
    ``edges`` (the edge components), in first-use order."""
    result = []
    for path in schedule.paths:
        inputs, edges = {}, {}
        for output in path.outputs:
            for term in output.terms:
                inputs.setdefault(term.input_component, term.input_accumulator.name)
                edges.setdefault(term.edge_component, None)
        result.append(dict(inputs=inputs, edges=tuple(edges)))
    return result


def render(schedule, scalar, terms, stream_w=None, vector=0, table_w=False, blocks=None, factored=False):
    """CUDA source of ``oeq_projected_backward_wprime`` for a schedule, ``scalar`` ("float" or "double") and the
    derivative table's ``terms``; ``stream_w`` (default ``STREAM_W``): the stored weights loaded with ``__ldcs``;
    ``vector`` 2 or 4: the q-major table's terms loaded as float2/float4 (``SLOPE_VECTOR``; float only).
    ``table_w`` (``TABLE_W``): the table is the q-major value table (``value_table``, ``terms`` = D + 1), from which
    the kernel evaluates w and its slope; the stored weights are not read (needs ``vector``). ``blocks``
    (``fused_forward.block_map``'s input; ir_mul): the message gradient read per group, one pointer each.
    ``factored`` (``FACTORED``): each path's three trilinear sums regrouped (rounding)."""
    from fastmace.kernels.oeq_streaming.fused_forward import block_map

    if blocks is not None and schedule.layout != "ir_mul":
        raise ValueError("a block-layout message needs the ir_mul layout")
    from fastmace.kernels.oeq_streaming.jinja_env import get_jinja_environment

    if schedule.layout not in ("ir_mul", "mul_ir"):
        raise ValueError(f"unknown layout {schedule.layout!r}")
    if int(terms) < 1:
        raise ValueError("the derivative table needs at least one term")
    if vector not in (0, 2, 4) or (vector and scalar != "float"):
        raise ValueError(f"slope vector {vector!r}: 0, or 2 or 4 for float kernels")
    if table_w and not vector:
        raise ValueError("table_w reads the q-major value table: it needs a slope vector of 2 or 4")
    env = get_jinja_environment(backend="cuda", warp_size=schedule.launch_config.logical_cohort_width)
    return env.from_string(TEMPLATE).render(scalar=scalar, schedule=schedule, layout=schedule.layout,
                                            terms=int(terms), stream_w=STREAM_W if stream_w is None else stream_w,
                                            slope_vector=vector, vector_type=f"float{vector}",
                                            padded_terms=padded_terms(terms, vector), table_w=bool(table_w),
                                            blocks=None if blocks is None else block_map(blocks, schedule.channels),
                                            n_blocks=0 if blocks is None else len(blocks),
                                            factored=factored_paths(schedule) if factored else None)


def derivative_table(dcoef, weight, vector=0):
    """``G' [pairs * pieces, D', Q]`` from a table's ``dcoef [P, K, D', J]`` and the folded weight ``[Q, J - 1]``
    (the table's last column is the edge density): ``w_table.fold_tables``' derivative half without its padding
    term and zero group. Computed in float64, stored in ``dcoef``'s dtype. ``vector`` 2 or 4 (``SLOPE_VECTOR``):
    q-major, ``[pairs * pieces, Q, D']`` with zero terms up to a multiple of ``vector``, the same values."""
    pairs, pieces, terms, _ = dcoef.shape
    width = weight.shape[1]
    d = torch.einsum("pktj,qj->pktq", dcoef[..., :width].to(torch.float64),
                     weight.detach().to(device=dcoef.device, dtype=torch.float64))
    d = d.reshape(pairs * pieces, terms, -1)
    if vector:
        d = torch.nn.functional.pad(d.transpose(1, 2), (0, padded_terms(terms, vector) - terms))
    return d.to(dcoef.dtype).contiguous()


def value_table(coef, weight, vector=TABLE_W_VECTOR):
    """``G [pairs * pieces, Q, D + 1 rounded up to a multiple of vector]`` (``TABLE_W``) from a table's ``coef [P, K,
    D + 1, J]`` and the folded weight ``[Q, J - 1]``: ``w_table.fold_tables``' value half without its zero group,
    q-major with zero padding terms. Computed in float64, stored in ``coef``'s dtype."""
    pairs, pieces, terms, _ = coef.shape
    width = weight.shape[1]
    g = torch.einsum("pktj,qj->pkqt", coef[..., :width].to(torch.float64),
                     weight.detach().to(device=coef.device, dtype=torch.float64))
    g = torch.nn.functional.pad(g.reshape(pairs * pieces, -1, terms), (0, padded_terms(terms, vector) - terms))
    return g.to(coef.dtype).contiguous()


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
    (``StreamingKernels``) and a derivative table of ``terms`` terms (``vector``: its q-major layout's load width,
    ``SLOPE_VECTOR``)."""

    def __init__(self, kernels, terms, device=None, vector=0, table_w=False, blocks=None, factored=None):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check, _device_key, compile_cubin, content_id
        scalar = "float" if "using scalar_t = float;" in kernels.source else "double"
        device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        self.terms, self.vector = int(terms), int(vector)
        self.scalar = scalar
        self.num_threads, self.cohort, self.shared_memory = kernels.num_threads, kernels.cohort, kernels.shared_memory
        self.output_dim, self.channels = kernels.output_dim, kernels.schedule.channels
        self.table_w = bool(table_w)
        self.blocks = blocks
        self.schedule = kernels.schedule
        self.factored = FACTORED if factored is None else bool(factored)
        self.source = render(kernels.schedule, scalar, self.terms, vector=self.vector, table_w=self.table_w,
                             blocks=blocks, factored=self.factored)
        cubin = compile_cubin(self.source, torch.cuda.get_device_capability(device))
        torch.cuda.current_stream(device)
        self._module = _check(driver.cuModuleLoadData(cubin))
        self._function = _check(driver.cuModuleGetFunction(self._module, b"oeq_projected_backward_wprime"))
        self.id = content_id("WPrimeBackward", self.source, _device_key(device))
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
                  receivers.data_ptr(), *([d.data_ptr() for d in dout] if self.blocks else [dout.data_ptr()]),
                  dx.data_ptr(), dsh.data_ptr(), r.data_ptr(),
                  pair.data_ptr(), gprime.data_ptr(), starts.data_ptr(), scales.data_ptr(), bins.data_ptr(),
                  r_lo, bin_scale, r_max, bins.shape[0], starts.shape[0], dr.data_ptr())
        douts = len(self.blocks) if self.blocks else 1
        types = ((ctypes.c_longlong, ctypes.c_longlong) + (p,) * (13 + douts) + (real, real, real)
                 + (ctypes.c_int, ctypes.c_int, p))
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


@torch.library.custom_op("fastmace::streaming_conv_wprime_backward_blocks", mutates_args=())
def streaming_conv_wprime_backward_blocks(x: torch.Tensor, sh: torch.Tensor, senders: torch.Tensor,
                                          receivers: torch.Tensor, douts: list[torch.Tensor], r: torch.Tensor,
                                          pair: torch.Tensor, gprime: torch.Tensor, starts: torch.Tensor,
                                          scales: torch.Tensor, bins: torch.Tensor, r_lo: float, bin_scale: float,
                                          r_max: float, wprime_id: int) -> tuple[torch.Tensor, torch.Tensor,
                                                                                 torch.Tensor]:
    """``streaming_conv_wprime_backward`` of a ``TABLE_W`` kernel built with ``blocks``: the message gradient as
    one tensor per group (``BLOCK_MESSAGE``)."""
    kernel = _REGISTRY[wprime_id]
    x, sh = x.contiguous(), sh.contiguous()
    douts = [d.contiguous() for d in douts]
    dx = torch.zeros_like(x)  # sender gradients are added atomically
    dsh = torch.empty_like(sh)
    dr = sh.new_empty((sh.shape[0],))
    kernel(x, sh, sh.new_empty((0,)), senders.contiguous(), receivers.contiguous(), douts, dx, dsh,
           r.reshape(-1).contiguous(), pair.contiguous(), gprime, starts, scales, bins, r_lo, bin_scale, r_max, dr)
    return dx, dsh, dr


@streaming_conv_wprime_backward_blocks.register_fake
def _(x, sh, senders, receivers, douts, r, pair, gprime, starts, scales, bins, r_lo, bin_scale, r_max, wprime_id):
    return torch.empty_like(x), torch.empty_like(sh), sh.new_empty((sh.shape[0],))


class _WPrimeBlockConvolution(torch.autograd.Function):
    """``_WPrimeConvolution``'s TABLE_W path with the fused message-only forward, the message in the block layout:
    returns the groups, then the edge density."""

    @staticmethod
    def forward(ctx, r, pair, coef, density_dcoef, starts, scales, bins, r_lo, bin_scale, r_max, weight, gprime, x,
                sh, senders, receivers, row_ptr, message_id, wprime_id):
        from fastmace.kernels.oeq_streaming.fused_forward import fused_folded_message_blocks
        from fastmace.kernels.triton.radial_table import radial_table_forward

        a, density = radial_table_forward(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max)
        groups = fused_folded_message_blocks(a, weight, x, sh, senders, receivers, row_ptr, message_id)
        ctx.save_for_backward(x, sh, senders, receivers, r, pair, gprime, density_dcoef, starts, scales, bins)
        ctx.constants, ctx.wprime_id = (r_lo, bin_scale, r_max), wprime_id
        return (*groups, density)

    @staticmethod
    def backward(ctx, *grads):
        x, sh, senders, receivers, r, pair, gprime, density_dcoef, starts, scales, bins = ctx.saved_tensors
        kernel = _REGISTRY[ctx.wprime_id]
        from fastmace.kernels.oeq_streaming.fused_forward import block_shapes
        shapes = block_shapes(kernel.blocks, kernel.channels, x.shape[0])
        douts = [x.new_zeros(shape) if g is None else g for g, shape in zip(grads[:-1], shapes)]
        dx, dsh, dr = streaming_conv_wprime_backward_blocks(x, sh, senders, receivers, douts, r, pair, gprime,
                                                            starts, scales, bins, *ctx.constants, ctx.wprime_id)
        if grads[-1] is not None:
            dr = dr + density_backward(r, pair, density_dcoef, starts, scales, bins, grads[-1], *ctx.constants)
        return (dr.reshape(r.shape), None, None, None, None, None, None, None, None, None, None, None, dx, dsh,
                None, None, None, None, None)


def block_spec(linear, schedule):
    """The block layout of linear_1's grouped input (``BLOCK_MESSAGE``), ``((dim, segment starts), ...)`` in the
    order of its products, or None where the message cannot feed ``linear`` directly: a grouped ``BlockLinear``
    whose input transpose is the identity and whose input segments are exactly the convolution's output segments
    (ir_mul, ``schedule.channels`` channels each)."""
    from fastmace.rewrites.linear.modules import BlockLinear

    if not isinstance(linear, BlockLinear) or not linear.grouped or schedule.layout != "ir_mul":
        return None
    transpose = linear.transpose_in
    if not isinstance(getattr(transpose, "f", transpose), torch.nn.Identity):
        return None
    segments = {(int(s.start), int(s.irrep_dim)) for s in schedule.output_slots}
    blocks, seen = [], set()
    for inputs, _ in linear.block_ends:
        dims = {linear.segments_in[i][2] for i in inputs}
        if len(dims) != 1 or any(linear.segments_in[i][1] != schedule.channels for i in inputs):
            return None
        starts = tuple(int(linear.segments_in[i][0]) for i in inputs)
        if seen & set(starts):
            return None
        seen |= set(starts)
        blocks.append((dims.pop(), starts))
    if {(s, d) for d, st in blocks for s in st} != segments:
        return None
    return tuple(blocks)


def dense_message(groups, blocks, channels, output_dim):
    """The ``[N, output_dim]`` ir_mul message of a block-layout one (the paths that need the whole message)."""
    n = groups[0].shape[0] // blocks[0][0]
    pieces = {}
    for group, (dim, starts) in zip(groups, blocks):
        view = group.reshape(n, dim, len(starts), channels)
        for rank, start in enumerate(starts):
            pieces[start] = view[:, :, rank].reshape(n, dim * channels)
    out = torch.cat([pieces[s] for s in sorted(pieces)], dim=1)
    if out.shape[1] != output_dim:
        raise ValueError(f"dense_message: {out.shape[1]} columns, expected {output_dim}")
    return out


class _WPrimeConvolution(torch.autograd.Function):
    """Edge lengths -> (message, edge density) through the radial table, the folded layer and the streaming
    convolution; the backward is ``streaming_conv_wprime_backward`` plus the density column's."""

    @staticmethod
    def forward(ctx, r, pair, coef, density_dcoef, starts, scales, bins, r_lo, bin_scale, r_max, weight, gprime, x,
                sh, senders, receivers, row_ptr, kernel_id, fused_id, wprime_id, recompute_rows=0, message_id=-1,
                rebuild_id=-1, table_w=False):
        from fastmace.kernels.oeq_streaming.runtime import streaming_conv_forward
        from fastmace.kernels.triton.radial_table import radial_table_forward

        a, density = radial_table_forward(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max)
        if rebuild_id >= 0 or (table_w and message_id >= 0):  # w built on chip, never stored
            from fastmace.kernels.oeq_streaming.fused_forward import fused_folded_message
            message = fused_folded_message(a, weight, x, sh, senders, receivers, row_ptr, message_id)
            weights = None
        elif fused_id >= 0:  # fused_folded_forward (opt-8 OPT8-T02): w stored by the message kernel
            from fastmace.kernels.oeq_streaming.fused_forward import fused_folded_forward
            message, weights = fused_folded_forward(a, weight, x, sh, senders, receivers, row_ptr, fused_id)
        else:
            weights = a @ weight.t()
            message = streaming_conv_forward(x, sh, weights, senders, row_ptr, kernel_id)
        if recompute_rows or weights is None or table_w:  # recompute_conv_weights (CAP-T07), TABLE_W: none
            weights = coef.new_empty((0,))
        ctx.save_for_backward(x, sh, weights, senders, receivers, r, pair, gprime, density_dcoef, starts, scales,
                              bins, coef, weight)
        ctx.constants, ctx.wprime_id, ctx.recompute_rows = (r_lo, bin_scale, r_max), wprime_id, int(recompute_rows)
        ctx.rebuild_id = int(rebuild_id)
        return message, density

    @staticmethod
    def backward(ctx, dout, ddensity):
        (x, sh, weights, senders, receivers, r, pair, gprime, density_dcoef, starts, scales, bins, coef,
         weight) = ctx.saved_tensors
        if dout is None:
            dout = x.new_zeros((x.shape[0], _REGISTRY[ctx.wprime_id].output_dim))
        if ctx.rebuild_id >= 0:
            from fastmace.kernels.oeq_streaming.wrebuild_backward import wrebuild_backward
            dx, dsh, dr = wrebuild_backward(x, sh, senders, receivers, dout, r, pair, gprime, coef, weight, starts,
                                            scales, bins, *ctx.constants, ctx.rebuild_id)
        elif ctx.recompute_rows:
            dx, dsh, dr = streaming_conv_wprime_backward_recompute(
                x, sh, senders, receivers, dout, r, pair, gprime, coef, weight, starts, scales, bins, *ctx.constants,
                ctx.wprime_id, ctx.recompute_rows)
        else:
            dx, dsh, dr = streaming_conv_wprime_backward(x, sh, weights, senders, receivers, dout, r, pair, gprime,
                                                         starts, scales, bins, *ctx.constants, ctx.wprime_id)
        if ddensity is not None:
            dr = dr + density_backward(r, pair, density_dcoef, starts, scales, bins, ddensity, *ctx.constants)
        return (dr.reshape(r.shape), None, None, None, None, None, None, None, None, None, None, None, dx, dsh,
                None, None, None, None, None, None, None, None, None, None)


def table_w_default(device, generic):
    """``TABLE_W`` for a WPrime on ``device``: the constant when set, else on for the generic family on
    ``TABLE_W_GPUS``."""
    if TABLE_W is not None:
        return bool(TABLE_W)
    device = torch.device(device)
    return bool(generic) and device.type == "cuda" and torch.cuda.get_device_name(device) in TABLE_W_GPUS


def block_message_default(device):
    """``BLOCK_MESSAGE`` on ``device``: the constant when set, else on for ``BLOCK_MESSAGE_GPUS``."""
    if BLOCK_MESSAGE is not None:
        return bool(BLOCK_MESSAGE)
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_name(device) in BLOCK_MESSAGE_GPUS


def prepare_block_messages(model):
    """``WPrime.prepare_blocks`` in every interaction of ``model`` with ``wprime_backward`` (after the rewrites;
    a no-op where ``BLOCK_MESSAGE`` is off). Returns how many interactions write the block layout."""
    count = 0
    if BLOCK_MESSAGE is False:
        return count
    for module in model.modules():
        wprime = module._modules.get("wprime_backward") if hasattr(module, "_modules") else None
        if isinstance(wprime, WPrime):
            wprime.prepare_blocks(module)
            count += wprime.block is not None
    return count


class WPrime(torch.nn.Module):
    """One interaction's ``wprime_backward``: the derivative table ``G'``, the density's derivative series and
    the compiled kernel, built from its radial table, folded weight and streaming kernels."""

    def __init__(self, table, weight, kernels, generic=False):
        super().__init__()
        # TABLE_W: the value table G in place of G' (w and its slope from one slab; the forward stores no w).
        self.table_w = table_w_default(table.coef.device, generic) and table.coef.dtype == torch.float32
        if self.table_w:
            self.terms, self.vector = int(table.coef.shape[2]), TABLE_W_VECTOR
            gprime = value_table(table.coef, weight, self.vector)
        else:
            self.terms = int(table.dcoef.shape[2])
            self.vector = SLOPE_VECTOR if table.dcoef.dtype == torch.float32 else 0
            gprime = derivative_table(table.dcoef, weight, self.vector)
        self.register_buffer("gprime", gprime, persistent=False)
        self.register_buffer("density_dcoef", density_column(table.dcoef), persistent=False)
        self.kernel = WPrimeBackward(kernels, self.terms, self.gprime.device, vector=self.vector,
                                     table_w=self.table_w)
        self.message = None  # TABLE_W with the fused forward: (its id, the message-only FusedFoldedForward)
        self.block = None  # BLOCK_MESSAGE: (fused id, blocks, block FusedFoldedForward, block WPrimeBackward)
        self.kernels = kernels
        # recompute_conv_weights on the fused kernels (attach_rebuild): the message-only forward and the backward that
        # builds w on chip, as (FusedFoldedForward(store_w=False), WRebuildBackward); None: the chunked recompute.
        self.rebuild = None

    def prepare_blocks(self, interaction):
        """Compile the block-layout kernels (``BLOCK_MESSAGE``) for ``interaction``'s fused forward and linear_1,
        after every rewrite is applied (``prepare_block_messages``); nothing where the layout does not fit."""
        from fastmace.kernels.oeq_streaming.fused_forward import FusedFoldedForward

        self.block = None
        fused, linear = getattr(interaction, "fused_folded_forward", None), getattr(interaction, "linear", None)
        if not (block_message_default(self.gprime.device) and self.table_w and fused is not None
                and linear is not None):
            return
        blocks = block_spec(linear, self.kernels.schedule)
        if blocks is None:
            return
        device = self.gprime.device
        self.block = (fused.id, blocks,
                      FusedFoldedForward(self.kernels, fused.k, device, store_w=False, used=fused.used, blocks=blocks),
                      WPrimeBackward(self.kernels, self.terms, device, vector=self.vector, table_w=True, blocks=blocks))

    def attach_rebuild(self, kernels, fused):
        """Compile the message-only fused forward and the w-rebuilding backward for ``recompute_conv_weights``
        (float32, the interaction's ``fused_folded_forward`` of width ``fused.k``)."""
        from fastmace.kernels.oeq_streaming.fused_forward import FusedFoldedForward
        from fastmace.kernels.oeq_streaming.wrebuild_backward import WRebuildBackward

        if self.vector or self.table_w:
            raise ValueError("wprime_backward: the rebuild backward (recompute_conv_weights) reads the term-major "
                             "derivative table; it needs SLOPE_VECTOR = 0 and TABLE_W off")
        device = self.gprime.device
        self.rebuild = (FusedFoldedForward(kernels, fused.k, device, store_w=False),
                        WRebuildBackward(kernels, fused.k, self.terms, device, parts=REBUILD_PARTS,
                                         batch=REBUILD_BATCH))

    def convolution(self, lengths, pair, table, weight, x, sh, senders, receivers, row_ptr, kernels, fused=None,
                    recompute_rows=0, linear=None):
        """``(message, edge density)`` of edge lengths ``lengths`` ``[E, 1]`` (differentiable in lengths, x, sh).
        ``recompute_rows`` (``recompute_conv_weights``): keep no per-edge weights for the backward. With the rebuild
        kernels attached (``attach_rebuild``) and ``fused``, the forward stores no ``w`` and the backward builds it on
        chip; otherwise the backward builds it again in chunks of that many edges."""
        if weight.requires_grad:
            raise ValueError("wprime_backward: the folded weight must not require a gradient")
        if (self.block is not None and fused is not None and linear is not None and self.block[0] == fused.id
                and BLOCK_MESSAGE is not False):  # a tuple of linear_1's input groups (prepare_blocks)
            *groups, density = _WPrimeBlockConvolution.apply(
                lengths, pair, table.coef, self.density_dcoef, table.starts, table.scales, table.bins,
                float(table.r_lo), float(table.bin_scale), float(table.r_max), weight, self.gprime, x, sh,
                senders, receivers, row_ptr, self.block[2].id, self.block[3].id)
            return tuple(groups), density
        if self.table_w:  # no w is kept anyway; the fused forward's message-only kernel stores none
            if fused is not None and (self.message is None or self.message[0] != fused.id):
                from fastmace.kernels.oeq_streaming.fused_forward import FusedFoldedForward
                self.message = (fused.id, FusedFoldedForward(kernels, fused.k, self.gprime.device, store_w=False,
                                                             used=fused.used))
            message_id = self.message[1].id if fused is not None else -1
            return _WPrimeConvolution.apply(lengths, pair, table.coef, self.density_dcoef, table.starts, table.scales,
                                            table.bins, float(table.r_lo), float(table.bin_scale), float(table.r_max),
                                            weight, self.gprime, x, sh, senders, receivers, row_ptr, kernels.id,
                                            -1 if fused is None else fused.id, self.kernel.id, 0, message_id, -1, True)
        rebuild = (self.rebuild[0].id, self.rebuild[1].id) if recompute_rows and fused is not None and self.rebuild \
            else None
        return _WPrimeConvolution.apply(lengths, pair, table.coef, self.density_dcoef, table.starts, table.scales,
                                        table.bins, float(table.r_lo), float(table.bin_scale), float(table.r_max),
                                        weight, self.gprime, x, sh, senders, receivers, row_ptr, kernels.id,
                                        -1 if fused is None else fused.id, self.kernel.id, int(recompute_rows),
                                        *(rebuild if rebuild else (-1, -1)))

    def extra_repr(self):
        outputs = self.gprime.shape[1 if self.vector else 2]
        vector = f", q-major float{self.vector}" if self.vector else ""
        table = ", value table (w from the table)" if self.table_w else ""
        return f"groups={self.gprime.shape[0]}, terms={self.terms}, outputs={outputs}{vector}{table}"


__all__ = ["BLOCK_MESSAGE", "BLOCK_MESSAGE_GPUS", "FACTORED", "TABLE_W", "TABLE_W_GPUS", "WPrime", "WPrimeBackward",
           "block_message_default", "block_spec", "dense_message",
           "density_backward", "density_column", "derivative_table", "factored_paths", "padded_terms",
           "prepare_block_messages",
           "render", "streaming_conv_wprime_backward", "streaming_conv_wprime_backward_blocks",
           "streaming_conv_wprime_backward_recompute", "table_w_default", "value_table", "wprime_arguments",
           "wprime_reference"]
