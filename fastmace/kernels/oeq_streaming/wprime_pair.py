"""The ``TABLE_W`` wprime backward with two adjacent channels per lane: a kernel-lab record
(``tools/omat0_kernel_lab.py`` variant ``pair``), not used by any plan. Measured against ``FACTORED`` (OMAT-0 medium,
4 November 2026): RTX 4000 Ada 0.898 vs 0.900, A100-SXM4-80GB 0.708 vs 0.722 of the default kernel (128 registers);
not built as a switch.

The vendored schedule gives each lane the channels ``lane, lane + 32, ...``, so every node-feature, output-gradient
and table load is a scalar (or float2 for the table row). Here a lane owns the pairs ``2 lane + 64 k, + 1``: the
input features and output gradients (ir_mul, channel-minor) are float2 loads, and the two weights' q-major value-table
rows (``2 PADDED_TERMS`` floats, adjacent) float4 loads, which halves the load instructions of the kernel lab's
load-bound kernel. The tensor product is the factored one (``FACTORED``): per path ``u_i = sum c g_o y_j`` and ``v_j
= sum c g_o x_i``, then the weight gradient ``sum_i x_i u_i``, ``dx_i += w u_i`` and ``dsh_j += w v_j``, per channel
of the pair. Rounding (the factored sums; dsh and dr add the pair's two channels per lane before the warp
reduction). The kernel's name and arguments are ``oeq_projected_backward_wprime``'s.

Generated in Python from the streaming schedule (the head, up to the Chebyshev bases, is the Jinja template's).
"""

from __future__ import annotations

from fastmace.kernels.oeq_streaming.jinja_env import cpp_scalar_literal

_SPLIT = "    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};"


def supported(schedule, scalar, vector, table_w):
    """Whether the pair kernel applies: float, ir_mul, ``TABLE_W`` with float2 rows, channels a multiple of 64."""
    return (scalar == "float" and schedule.layout == "ir_mul" and table_w and vector == 2
            and schedule.channels % 64 == 0 and schedule.launch_config.logical_cohort_width == 32)


def render(schedule, terms, blocks=None):
    """CUDA source of the channel-pair ``oeq_projected_backward_wprime`` (float, ir_mul, ``TABLE_W``, float2)."""
    from fastmace.kernels.oeq_streaming import wprime_backward as wp
    from fastmace.kernels.oeq_streaming.fused_forward import block_map

    if not supported(schedule, "float", 2, True):
        raise ValueError("the channel-pair wprime kernel needs float, ir_mul, TABLE_W (float2) and 64k channels")
    head = wp.render(schedule, "float", terms, vector=2, table_w=True, blocks=blocks)
    head = head[:head.index(_SPLIT)]
    padded = wp.padded_terms(terms, 2)
    channels = schedule.channels
    bmap = None if blocks is None else block_map(blocks, channels)
    lit = lambda c: cpp_scalar_literal(c, "float")  # noqa: E731

    def x_offset(start, component):
        return f"input_base + {start + component * channels} + channel"

    def out_pointer(slot):
        if bmap is None:
            return f"dout + output_base + {slot.start + slot.component * channels} + channel"
        g, width, rank, stride = bmap[slot.start]
        return f"dout{g} + int64_t(receiver) * {width} + {rank + slot.component * stride} + channel"

    lines = [_SPLIT, "    acc_t radial_gradient = acc_t(0);",
             f"    for (int channel = 2 * lane; channel < CHANNEL_DIM; channel += {2 * 32}) {{"]
    for slot in schedule.input_gradient_slots:
        lines.append(f"        acc_t {slot.name}_0 = acc_t(0), {slot.name}_1 = acc_t(0);")
    for path_index, path in enumerate(schedule.paths):
        fp = wp.factored_paths(schedule)[path_index]
        for ec in range(path.edge_mul):
            lines.append("        {")
            lines.append(f"            const int weight_index = {path.weight_start + ec * channels} + channel;")
            lines.append(f"            scalar_t gv0[{padded}], gv1[{padded}];")
            lines.append("            if (inside) {")
            lines.append("                const float4* gq = reinterpret_cast<const float4*>(g + weight_index * "
                         "PADDED_TERMS);")
            for q in range(padded // 2):
                comps = "xyzw"
                lines.append(f"                {{ const float4 v = __ldg(gq + {q});")
                for c in range(4):
                    flat = 4 * q + c
                    h, k = divmod(flat, padded)
                    lines.append(f"                  gv{h}[{k}] = v.{comps[c]};")
                lines.append("                }")
            lines.append("            } else {")
            lines.append("                #pragma unroll")
            lines.append("                for (int k = 0; k < PADDED_TERMS; ++k) { gv0[k] = scalar_t(0); "
                         "gv1[k] = scalar_t(0); }")
            lines.append("            }")
            lines.append("            scalar_t w0 = scalar_t(0), w1 = scalar_t(0);")
            lines.append("            #pragma unroll")
            lines.append("            for (int k = 0; k < TERMS; ++k) { w0 += cheb[k] * gv0[k]; "
                         "w1 += cheb[k] * gv1[k]; }")
            for i in fp["inputs"]:
                lines.append(f"            acc_t u0_{i} = acc_t(0), u1_{i} = acc_t(0);")
            for j in fp["edges"]:
                lines.append(f"            acc_t v0_{j} = acc_t(0), v1_{j} = acc_t(0);")
            for output in path.outputs:
                slot = output.accumulator
                lines.append("            {")
                lines.append("                const float2 og = *reinterpret_cast<const float2*>("
                             f"{out_pointer(slot)});")
                for term in output.terms:
                    i, j = term.input_component, term.edge_component
                    lines.append("                {")
                    lines.append("                    const float2 xv = *reinterpret_cast<const float2*>("
                                 f"x + {x_offset(path.input_start, i)});")
                    lines.append(f"                    const scalar_t y = sh[edge_base + "
                                 f"{path.edge_start + j + ec * path.edge_irrep_dim}];")
                    if float(term.coefficient) == 1.0:
                        lines.append("                    const scalar_t s0 = og.x, s1 = og.y;")
                    else:
                        c = lit(term.coefficient)
                        lines.append(f"                    const scalar_t s0 = og.x * {c}, s1 = og.y * {c};")
                    lines.append(f"                    u0_{i} += s0 * y; u1_{i} += s1 * y;")
                    lines.append(f"                    v0_{j} += s0 * xv.x; v1_{j} += s1 * xv.y;")
                    lines.append("                }")
                lines.append("            }")
            lines.append("            acc_t wg0 = acc_t(0), wg1 = acc_t(0);")
            for i, name in fp["inputs"].items():
                lines.append("            {")
                lines.append("                const float2 xv = *reinterpret_cast<const float2*>("
                             f"x + {x_offset(path.input_start, i)});")
                lines.append(f"                wg0 += xv.x * u0_{i}; wg1 += xv.y * u1_{i};")
                lines.append(f"                {name}_0 += w0 * u0_{i}; {name}_1 += w1 * u1_{i};")
                lines.append("            }")
            for j in fp["edges"]:
                e_index = path.edge_start + j + ec * path.edge_irrep_dim
                lines.append(f"            edge_gradient[{e_index}] += w0 * v0_{j};")
                lines.append(f"            edge_gradient[{e_index}] += w1 * v1_{j};")
            lines.append("            if (sloped) {")
            lines.append("                acc_t s0 = acc_t(0), s1 = acc_t(0);")
            lines.append("                #pragma unroll")
            lines.append("                for (int k = 1; k < TERMS; ++k) { s0 += dcheb[k] * gv0[k]; "
                         "s1 += dcheb[k] * gv1[k]; }")
            lines.append("                radial_gradient += wg0 * s0;")
            lines.append("                radial_gradient += wg1 * s1;")
            lines.append("            }")
            lines.append("        }")
    for slot in schedule.input_gradient_slots:
        offset = slot.start + slot.component * channels
        lines.append(f"        atomicAdd(dx + input_base + {offset} + channel, scalar_t({slot.name}_0));")
        lines.append(f"        atomicAdd(dx + input_base + {offset} + channel + 1, scalar_t({slot.name}_1));")
    lines += ["    }",
              "    for (int offset = 16; offset > 0; offset >>= 1) {",
              "        for (int component = 0; component < EDGE_DIM; ++component)",
              "            edge_gradient[component] +=",
              "                __shfl_down_sync(FULL_MASK, edge_gradient[component], offset, 32);",
              "        radial_gradient += __shfl_down_sync(FULL_MASK, radial_gradient, offset, 32);",
              "    }",
              "    if (lane == 0) {",
              "        for (int component = 0; component < EDGE_DIM; ++component)",
              "            dsh[edge_base + component] = scalar_t(edge_gradient[component]);",
              "        dr[e] = scalar_t(radial_gradient);",
              "    }",
              "}", ""]
    return head + "\n".join(lines)


__all__ = ["render", "supported"]
