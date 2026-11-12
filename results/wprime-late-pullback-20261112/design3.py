"""Design 3 (compile-only): the irrep-first loop nest of today's H200 kernel. The channel loop is split by edge degree
l = 0..3; each group's loop holds only its paths, its own input-gradient accumulators and dx atomics (hoisted
pointer per group), and right after it the group's 2l + 1 edge-gradient slots are reduced over the warp (xor
shuffles) and stored, so only one harmonic block is live at a time. The butterfly goes; dr is reduced at the end.

    python design3.py   -> sources/d3-irrepfirst.cu (from results/wprime-h200-20261111/sources/wprime-h200-skip.cu)
"""
import re

from common import H200, HERE, setup

BASE = (H200 / "sources" / "wprime-h200-skip.cu").read_text()


def main():
    from fastmace.kernels.oeq_streaming import wprime_pullback as wpb
    from fastmace.kernels.oeq_streaming.wprime_tuning import LOOP, _block
    s = setup()
    ch = int(s.schedule.channels)
    path_list = wpb.paths(s.schedule, s.blocks, True)
    l_of = {p["weight_start"]: p["l"] for p in path_list}
    head, decl, blocks, tail, rest = wpb._loop_parts(BASE)
    slots = {sl.name: (int(sl.start), int(sl.component)) for sl in s.schedule.input_gradient_slots}
    edge_start = {p["l"]: sum(2 * k + 1 for k in range(p["l"])) for p in path_list}
    groups = {}
    for blk in blocks:
        ws = int(re.search(r"const int weight_index =\s*(\d+) \+ channel;", blk).group(1))
        groups.setdefault(l_of[ws], []).append(blk)
    code = []
    for l in sorted(groups):
        text = "".join(groups[l])
        used = [n for n in slots if re.search(rf"\b{n}\b", text)]
        code += [f"    {{  // design 3: edge degree {l}",
                 "    scalar_t* dx_lane = dx + input_base + lane;",
                 LOOP.rstrip("\n")]
        code += [f"            acc_t {n} = acc_t(0);" for n in used]
        code.append(text)
        code += [f"            atomicAdd(dx_lane + {slots[n][0] + slots[n][1] * ch}, scalar_t({n}));" for n in used]
        code += ["            dx_lane += LOGICAL_GROUP_SIZE;", "    }"]
        for j in range(edge_start[l], edge_start[l] + 2 * l + 1):
            code += ["    #pragma unroll",
                     "    for (int offset = 16; offset > 0; offset >>= 1)",
                     f"        edge_gradient[{j}] += __shfl_xor_sync(FULL_MASK, edge_gradient[{j}], offset, 32);",
                     f"    if (lane == 0) dsh[edge_base + {j}] = scalar_t(edge_gradient[{j}]);"]
        code.append("    }")
    reduce_ = ["    #pragma unroll",
               "    for (int offset = 16; offset > 0; offset >>= 1)",
               "        radial_gradient += __shfl_xor_sync(FULL_MASK, radial_gradient, offset, 32);",
               "    if (lane == 0) dr[e] = scalar_t(radial_gradient);"]
    after = rest[rest.index("    }  // STAGE_DOUT: the run's edges"):]
    src = head + "\n".join(code) + "\n" + "\n".join(reduce_) + "\n" + after
    assert "atomicAdd(\n" not in src and "BUTTERFLY" not in src
    (HERE / "sources" / "d3-irrepfirst.cu").write_text(src)
    print("wrote d3-irrepfirst", {l: len(b) for l, b in groups.items()})


if __name__ == "__main__":
    main()
