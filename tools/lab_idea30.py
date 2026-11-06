"""Idea 30's falsifier (H200 ideas plan, revision 14; user approval 6 November 2026): a warp-specialized wgmma
forward for part 1 alone, as a kernel-lab variant of the fused forward. Lab prototype only; nothing for production.

The plan's text: "one producer warpgroup stages a, x and sh (TMA or cp.async) and issues wgmma.mma_async for
w = a . W, with W pre-split (big and small parts) in shared memory; two consumer warpgroups run the TP from the
finished w, synchronized by mbarriers ... Falsifier: a lab prototype of part 1 alone with the TP's FP32 work on
consumer warps and the product on a producer, against F-tc-no-product's bound. Stop if it is not <= 0.85 of the
part's time."

Design of ``ws30`` (a new kernel body for part 1, generated from the plan's render; part-0 blocks return at once,
as in the lab's F-part1-only):
- 12 warps per block as today: warpgroup 0 is the producer, warpgroups 1 and 2 hold eight consumer warps.
- Each consumer warp walks its own receiver chunks in 8-edge batches exactly as today's warps do (the same chunk
  counter, the same CSR order per receiver), so no message needs atomics. One round is the eight consumer warps'
  batches together: 64 edges, the M of one wgmma.
- A ring of two stages in shared memory. Per stage: the w tile [paths][64 edges][RS] (the producer's D, read by the
  consumers with lane = channel), the a tile [64][KP + 4] laid over the start of the w tile (the producer reads all
  of a into registers before it writes D), and the sh tile [64][ED].
- Consumers stage their batch's a rows and sh rows with cp.async (16 B, zero-filled past the batch) and arrive on the
  stage's ``full`` mbarrier through cp.async.mbarrier.arrive (the arrival lands when the copies have) and a plain
  arrive. Then they run the TP of the previous round (x gathers first, then wait on that stage's ``wready``), and
  arrive on its ``empty``. The staging side claims chunks one round ahead of the TP side; each round carries its
  batch's chunk bounds, so the receiver flushes are today's (rows of empty chunks are written as zeros at claim).
- The producer waits on ``full``, loads the 64 x KP a tile as wgmma A fragments (registers; split into TF32 high
  and low parts by the mask split, as today), and issues per half of the part's paths (N = 96) and per k step the
  three passes lo.hi, hi.lo, hi.hi as wgmma.mma_async m64n96k8 against W's high and low planes in shared memory
  (filled once per block in core-matrix order, no swizzle; round 1's WGMMA layout and descriptors). It writes D to
  the w tile and arrives on ``wready``. A round with no active consumer ends the producer and, through a flag, the
  consumers.
- The products are the same 3xTF32 terms as today's mma.sync product but accumulated in the wgmma's order:
  rounding class, not bitwise.

Falsifiers of the prototype itself (wrong outputs): ``F-ws30-no-tp`` (consumers skip the TP: staging and product
alone) and ``F-ws30-no-product`` (the producer issues no wgmma and stores no D: staging and TP alone).
``F-tc-no-product-p1`` is the lab's F-tc-no-product restricted to part 1 (the bound the plan names).

Debug render (``render_ws(..., consumers=4, product="simt", w_source="global")``): the same protocol with four
consumer warps (32-edge rounds) and an FP32 SIMT product reading W from global memory, which fits the shared memory
of an sm_120 GPU; used by ``check_idea30_local.py`` against a naive reference kernel generated from the same TP text.

Hook (``tools/omat0_kernel_lab.py``): ``_idea30_variants(kernel)``, which also adds the references (ws30 and its
falsifiers are compared with F-part1-only).

Since the full version (6 November 2026) the generator itself is production code,
``fastmace/kernels/oeq_streaming/ws_forward.py`` (``fused_forward.WS_FORWARD``); this module wraps it for the lab and
keeps the naive reference kernel of the debug check.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastmace.kernels.oeq_streaming import ws_forward as WS  # noqa: E402  (the generator lives in production)

SM90A_MARK = WS.MARK
VEC_A_MARK = WS.VEC_A_MARK
parts_of = WS.parts_of
default_halves = WS.default_halves
_case = WS.case_text
_kp = WS.kp_of


def extract(source, part=1):
    return WS.extract(source, part)


def part_shared_bytes(info, consumers=8, w_source="smem", kp=64, rs=36, ed=16, stages=2):
    return WS.part_shared_bytes(info["npl"], kp, rs, ed, stages, consumers, w_source)


def render_parts(source, parts=None, others="return", consumers=8, w_source="smem", order="z"):
    """``ws_forward.render_parts`` (``parts``: {part: dict(product=, tp=, halves=, stages=)}; default part 1)."""
    return WS.render_parts(source, {1: {}} if parts is None else parts, others, consumers, w_source, order)


def render_ws(source, consumers=8, product="wgmma", w_source="smem", tp=True, halves=None, part=1):
    """One part warp-specialized, the other parts' blocks returning at once (round 1's ws30 and its falsifiers)."""
    return render_parts(source, {part: dict(product=product, tp=tp, halves=halves)}, "return", consumers, w_source)


def rule_shared_bytes(source, parts, plan_bytes, others="return", consumers=8, w_source="smem"):
    """The launch rule's shared memory: the largest warp-specialized part's, or the plan's when today's parts stay."""
    if not isinstance(parts, dict):
        parts = {p: {} for p in parts}
    sizes = [WS.shared_bytes(source, parts, consumers, w_source)] if parts else []
    if others == "keep":
        sizes.append(plan_bytes)
    return max(sizes)


def reference_kernel(source, part=1):
    """A naive reference of part ``part``'s messages (debug check): one block per receiver row and channel block, one
    thread per channel; w by FP32 FMAs in k order (as the simt debug product); the same TP text, one edge per batch."""
    info = extract(source, part)
    flush = "\n".join(f"    {line}" for line in info["flush_lines"])
    return f"""{info['head']}
extern "C" __global__ void ref_part{part}(int64_t N, int64_t E, const float* __restrict__ a,
    const float* __restrict__ W, const float* __restrict__ x, const float* __restrict__ sh,
    const int* __restrict__ senders,
    const int* __restrict__ receivers, const int* __restrict__ row_ptr,
    float* __restrict__ out0, float* __restrict__ out1, float* __restrict__ out2, float* __restrict__ out3) {{
    const int lane = threadIdx.x, cb = blockIdx.y, ch = cb * CB + lane;
    int64_t cur = blockIdx.x;
    {info['wst']}
    {info['cpf']}
    float acc[{info['nacc']}];
    for (int i = 0; i < {info['nacc']}; ++i) acc[i] = 0.f;
    for (int64_t e0 = row_ptr[cur]; e0 < row_ptr[cur + 1]; ++e0) {{
        const int nb = 1;
        const bool tail = false;
        const int lane_rv = (int)cur, lane_snd = senders[e0];
        float w[{info['npl']}][B];
        for (int p = 0; p < {info['npl']}; ++p) {{
            const int l = lane;
            float sum = 0.f;
            for (int k = 0; k < KP; ++k) sum = fmaf(a[e0 * AS + k], {info['wexpr']}, sum);
            w[p][0] = sum;
        }}
        const float* sh_s = sh + e0 * ED;
{info['xv']}{info['tp']}
    }}
{flush}
}}
"""


def idea30_variants(kernel, rules, part_only, no_product_pattern):
    """name -> (kind, transform, extra) for the lab's ``forward_variants``; ``rules``: the lab's ``_RULES`` (each
    ws source gets its launch rule: 12 warps, its shared memory); ``part_only``: the lab's ``_part_only``;
    ``no_product_pattern``: (old, new) of the lab's F-tc-no-product."""
    import dataclasses

    def ws(parts, others="return", order="z"):
        def transform(s):
            out = render_parts(s, parts, others, order=order)
            rule = kernel.rule if kernel is not None else None
            if rule is not None and parts:
                rules[out] = dataclasses.replace(rule, warps=12, shared_bytes=rule_shared_bytes(
                    s, parts, rule.shared_bytes, others))
            return out
        return transform

    def no_product_part(part):
        def transform(s):
            old, new = no_product_pattern
            if s.count(old) != 2:
                raise ValueError("F-tc-no-product-p*: the k-loop pattern is not in both parts")
            return part_only(part)(s.replace(old, new))
        return transform

    both = {0: {}, 1: {}}
    both_s3 = {0: dict(stages=3), 1: {}}
    return {
        "ws30": ("rounding", ws({1: {}}), ()),
        "ws30-n192": ("rounding", ws({1: dict(halves=1)}), ()),
        "F-ws30-no-tp": ("falsifier", ws({1: dict(tp=False)}), ()),
        "F-ws30-no-product": ("falsifier", ws({1: dict(product="none")}), ()),
        "F-tc-no-product-p1": ("falsifier", no_product_part(1), ()),
        # Full version, step 2: part 0 (4 paths, N = 128) the same way, alone and with part 1.
        "ws30-p0": ("rounding", ws({0: {}}), ()),
        "ws30-p0-n64": ("rounding", ws({0: dict(halves=2)}), ()),
        "F-ws30-p0-no-tp": ("falsifier", ws({0: dict(tp=False)}), ()),
        "F-tc-no-product-p0": ("falsifier", no_product_part(0), ()),
        "ws-both": ("rounding", ws(both), ()),
        "ws-p1-keep0": ("rounding", ws({1: {}}, "keep"), ()),
        # Extras (r4): part 0's third stage (the TP two rounds behind its staging); the block order of the parts.
        "ws30-p0-s3": ("rounding", ws({0: dict(stages=3)}), ()),
        "ws-both-s3": ("rounding", ws(both_s3), ()),
        "ws-both-swap": ("rounding", ws(both, order="swap"), ()),
        "ws-both-inter": ("rounding", ws(both, order="interleave"), ()),
        "ws-both-s3-swap": ("rounding", ws(both_s3, order="swap"), ()),
        "ws-both-s3-inter": ("rounding", ws(both_s3, order="interleave"), ()),
        "base-swap": ("exact", ws({}, "keep", order="swap"), ()),
        "base-inter": ("exact", ws({}, "keep", order="interleave"), ()),
    }


def idea30_references():
    return {"ws30": "F-part1-only", "ws30-n192": "F-part1-only", "ws30-p0": "F-part0-only",
            "ws30-p0-n64": "F-part0-only", "ws30-p0-s3": "F-part0-only"}
