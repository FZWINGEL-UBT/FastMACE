"""sx-layout lab (owner, 7 November 2026): Symmetrix-XL's interaction-2 forward layout in the OMAT-0 kernel lab.

Symmetrix-XL 0.1.2's ``symmetrix_factorized_forward_v2`` (``symmetrix/jit_codegen.py``, ``_render_forward_owner``
and ``_render_tiled_forward_kernel``) runs one thread per (receiver, channel) with a grid stride in 256-thread
blocks. Each thread walks its receiver's CSR edges serially, skips edges at or beyond the cutoff, evaluates w per
(path, channel) from a cubic spline (4 coefficients, laid out [pair][interval][coefficient][path x channel], so
neighbouring threads' loads coalesce), forms the unrolled Clebsch-Gordan terms and accumulates the message in
registers; no shared memory, atomics or tensor cores. This module builds that layout around FastMACE's own angular
code: the CG terms, the x offsets and the output stores are taken from the production forward's rendered source
(``fused_folded_forward``, one ``case`` per path part), so the message is the production message.

Forward variants (``forward_variants``; every one computes the real message and is compared with the base):

- ``sx-spline<N>``: w from an N-knot cubic Hermite spline of the radial network, fitted in float64 on the GPU at lab
  time (values and exact r-derivatives at uniform knots on [R_LO, r_max)), float32 power-basis coefficients in
  t = (r - x_i) / h, one row per unordered species pair (the chain is symmetric in the pair).
- ``sx-table6``: w from our 6-term table (``wprime_backward.TABLE_W``'s value table, the w' call's own arguments:
  pieces, bins, power basis with HORNER), without FOLD_ALPHA's per-path factor.
- ``sx-wbuf``: w read from a [E, Q] buffer (a @ W in FP32): the layout's correctness check.
- ``sx-F-w1``: w = 1, no table (falsifier: the layout's floor without any w work).

Backward variants (``wprime_variants``): the production w' kernel with its 6-term table replaced by the N-knot
spline (4 terms, q-major float2 loads, the same dual Horner; FOLD_ALPHA's factor folded in as production folds it),
so ``dr`` is the spline's slope.
"""

from __future__ import annotations

import ctypes
import itertools
import re

CONTEXT = {}  # main() fills: checkpoint, model, numbers (the structure's atomic numbers), forward args, wprime args
_CACHE = {}

CHANNELS = 128
THREADS = 256


# --------------------------------------------------------------------------------------------- parsing production
def parse_forward(source):
    """Per path part of the production forward: path indices, x offsets, angular code and output stores."""
    channels = int(re.search(r"constexpr int CB = (\d+);", source).group(1)) * 4  # 4 channel blocks of CB
    parts = []
    for body in re.split(r"\n    case \d+: \{", source)[1:]:
        body = body.split("} break;")[0]
        wst = [int(v) for v in re.search(r"const int wst\[\d+\] = \{([^}]*)\};", body).group(1).split(",")]
        xs = re.findall(r"xv\[b\]\[(\d+)\] = \(snd >= 0\) \? x\[\(int64_t\)snd \* IN_DIM \+ (\d+) \+ ch\] : 0\.f;",
                        body)
        count = int(re.search(r"float acc\[(\d+)\];", body).group(1))
        blocks = re.findall(r"\{\s*((?:const float e\d+ = sh_s\[b \* ED \+ \d+\];\s*)+)"
                            r"((?:acc\[\d+\] = fmaf\(w\[\d+\]\[b\], [^;]*?, acc\[\d+\]\);\s*)+)\}", body)
        final = body[body.rindex("while (cur < r1) {"):]
        stores = re.findall(r"(out\d+)\[cur \* (\d+) \+ (\d+) \+ ch\] = acc\[(\d+)\];", final)
        terms = sum(len(re.findall(r"acc\[\d+\] = fmaf", b[1])) for b in blocks)
        if not xs or not blocks or len(stores) != count or terms != count:
            raise ValueError(f"sx-layout: could not parse a path part (x {len(xs)}, blocks {len(blocks)}, "
                             f"stores {len(stores)}, terms {terms}, acc {count})")
        parts.append(dict(paths=[v // CHANNELS for v in wst], xs=[(int(k), int(o)) for k, o in xs], count=count,
                          blocks=blocks, stores=[(o, int(s), int(off), int(i)) for o, s, off, i in stores]))
    if not parts or channels != CHANNELS:
        raise ValueError(f"sx-layout: {len(parts)} parts, {channels} channels (expects {CHANNELS})")
    return parts


def _angular(parts):
    """The per-edge code (after wv[] and shv are set) and the stores, with per-part names."""
    lines, stores, declarations = [], [], []
    for index, part in enumerate(parts):
        declarations.append(f"        float acc{index}[{part['count']}];\n"
                            f"        #pragma unroll\n"
                            f"        for (int i = 0; i < {part['count']}; ++i) acc{index}[i] = 0.f;")
        for k, offset in part["xs"]:
            lines.append(f"            const float x{index}_{k} = x[(int64_t)snd * IN_DIM + {offset} + ch];")
        for loads, products in part["blocks"]:
            text = "            {\n            " + loads.strip() + "\n            " + products.strip() + "\n            }"
            text = re.sub(r"sh_s\[b \* ED \+ (\d+)\]", r"__ldg(sh + e * ED + \1)", text)
            text = re.sub(r"xv\[b\]\[(\d+)\]", lambda m: f"x{index}_{m.group(1)}", text)
            text = re.sub(r"w\[(\d+)\]\[b\]", lambda m: f"wv[{part['paths'][int(m.group(1))]}]", text)
            text = re.sub(r"acc\[(\d+)\]", lambda m: f"acc{index}[{m.group(1)}]", text)
            lines.append(text)
        for out, stride, offset, i in part["stores"]:
            stores.append(f"        {out}[(int64_t)rcv * {stride} + {offset} + ch] = acc{index}[{i}];")
    return "\n".join(declarations), "\n".join(lines), "\n".join(stores)


# --------------------------------------------------------------------------------------------- the forward source
_W_CODE = {
    "spline": """            const float u = fmaxf((re - SX0) * SXINV, 0.f);
            int iv = (int)u;
            iv = iv < SXNI - 1 ? iv : SXNI - 1;
            const float t = u - (float)iv;
            const float* cp = tab + (int64_t)(pmap[pair[e]] * SXNI + iv) * 4 * Q + ch;
            float wv[P];
            #pragma unroll
            for (int p = 0; p < P; ++p) {
                const float c0 = cp[p * C], c1 = cp[Q + p * C], c2 = cp[2 * Q + p * C], c3 = cp[3 * Q + p * C];
                wv[p] = fmaf(t, fmaf(t, fmaf(t, c3, c2), c1), c0);
            }""",
    "table6": """            const float cl = fmaxf(re, T_RLO);
            int bin = (int)((cl - T_RLO) * T_BINSCALE);
            bin = bin < T_NBINS - 1 ? bin : T_NBINS - 1;
            const int piece = bins[bin];
            const float t = (cl - starts[piece]) * scales[piece] - 1.f;
            const float2* g = reinterpret_cast<const float2*>(tab + ((int64_t)pair[e] * T_PIECES + piece) * Q * 6)
                + ch * 3;
            float wv[P];
            #pragma unroll
            for (int p = 0; p < P; ++p) {
                const float2 v0 = __ldg(g + p * C * 3), v1 = __ldg(g + p * C * 3 + 1), v2 = __ldg(g + p * C * 3 + 2);
                float w = v2.y;
                w = fmaf(w, t, v2.x);
                w = fmaf(w, t, v1.y);
                w = fmaf(w, t, v1.x);
                w = fmaf(w, t, v0.y);
                wv[p] = fmaf(w, t, v0.x);
            }""",
    "wbuf": """            float wv[P];
            #pragma unroll
            for (int p = 0; p < P; ++p) wv[p] = tab[e * Q + p * C + ch];""",
    "w1": """            float wv[P];
            #pragma unroll
            for (int p = 0; p < P; ++p) wv[p] = 1.f;""",
}


def forward_source(production, mode, constants):
    """The sx-layout forward for ``mode`` (a key of ``_W_CODE``) with ``constants`` (name -> value)."""
    parts = parse_forward(production)
    declarations, body, stores = _angular(parts)
    paths = 1 + max(p for part in parts for p in part["paths"])
    in_dim = int(re.search(r"constexpr int IN_DIM = (\d+);", production).group(1))
    ed = int(re.search(r"constexpr int ED = (\d+);", production).group(1))
    outs = sorted({s[0] for part in parts for s in part["stores"]}, key=lambda o: int(o[3:]))
    consts = "\n".join(f"constexpr {'int' if isinstance(v, int) else 'float'} {k} = "
                       f"{v if isinstance(v, int) else repr(float(v)) + 'f'};" for k, v in constants.items())
    return f"""// sx_layout_forward (sx-layout lab, 7 November 2026): Symmetrix-XL's forward layout, w by {mode}.
using int64_t = long long;
constexpr int C = {CHANNELS};
constexpr int P = {paths};
constexpr int64_t Q = {paths * CHANNELS};
constexpr int IN_DIM = {in_dim};
constexpr int ED = {ed};
{consts}

extern "C" __global__ void __launch_bounds__({THREADS}) sx_layout_forward(
    int64_t N, int64_t E, const float* __restrict__ x, const float* __restrict__ sh,
    const int* __restrict__ senders, const int* __restrict__ row_ptr, const float* __restrict__ r,
    const int* __restrict__ pair, const int* __restrict__ pmap, const float* __restrict__ tab,
    const float* __restrict__ starts, const float* __restrict__ scales, const int* __restrict__ bins,
    {", ".join(f"float* __restrict__ {o}" for o in outs)}) {{
    const int64_t owners = N * C;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t owner = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; owner < owners; owner += stride) {{
        const int rcv = (int)(owner / C), ch = (int)(owner % C);
{declarations}
        int64_t eb = row_ptr[rcv], ee = row_ptr[rcv + 1];
        eb = eb < 0 ? 0 : eb;
        ee = ee > E ? E : ee;
        for (int64_t e = eb; e < ee; ++e) {{
            const float re = r[e];
            if (!(re < R_MAX)) continue;  // Symmetrix-XL's edge_is_active
            const int snd = senders[e];
            if (snd < 0 || snd >= N) continue;
{_W_CODE[mode]}
{body}
        }}
{stores}
    }}
}}
"""


def forward_launcher(kernel, source, extra):
    import torch
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin
    cubin = compile_cubin(source, torch.cuda.get_device_capability(), extra)
    module = _check(driver.cuModuleLoadData(cubin))
    function = _check(driver.cuModuleGetFunction(module, b"sx_layout_forward"))
    registers = _check(driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, function))
    spills = _check(driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                              function))

    def launch(args):
        a, weight, x, sh, senders, receivers, row_ptr, out, w, sx = args
        nodes = x.shape[0]
        outs = out if isinstance(out, list) else [out]
        p = ctypes.c_void_p
        values = (nodes, sh.shape[0], x.data_ptr(), sh.data_ptr(), senders.data_ptr(), row_ptr.data_ptr(),
                  sx["r"].data_ptr(), sx["pair"].data_ptr(), sx["pmap"].data_ptr(), sx["tab"].data_ptr(),
                  sx["starts"].data_ptr(), sx["scales"].data_ptr(), sx["bins"].data_ptr(),
                  *[o.data_ptr() for o in outs])
        types = (ctypes.c_longlong, ctypes.c_longlong) + (p,) * (len(values) - 2)
        blocks = (nodes * CHANNELS + THREADS - 1) // THREADS
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
        _check(driver.cuLaunchKernel(function, blocks, 1, 1, THREADS, 1, 1, 0, stream, (values, types), 0))
    return launch, dict(registers=int(registers), local_bytes=int(spills))


# --------------------------------------------------------------------------------------------- tables
def _edges():
    """(r, pair, wprime args) of the captured w' call, checked to share the forward's edges."""
    import torch
    wp, fw = CONTEXT["wprime_args"], CONTEXT["forward_args"]
    if not (torch.equal(wp[3], fw[4]) and torch.equal(wp[1], fw[3])):
        raise ValueError("sx-layout: the captured forward and w' calls do not share their edges")
    return wp[8].reshape(-1).float().contiguous(), wp[9].int().contiguous(), wp


def _reference_w(rows):
    """a @ W in float64 for the edge indices ``rows`` (the forward's own w, up to its 3xTF32 rounding)."""
    fw = CONTEXT["forward_args"]
    a, weight = fw[0], fw[1]
    k = 64
    return a[rows, :k].double() @ weight[:, :k].double().T


def _chain_and_columns():
    """The float64 radial chain of interaction 2 on the GPU and its species columns, matched to the captured a."""
    if "chain" in _CACHE:
        return _CACHE["chain"]
    import hashlib
    from pathlib import Path

    import torch

    from fastmace.model import loading
    from fastmace.rewrites import radial as rt
    checkpoint = Path(CONTEXT["checkpoint"])
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = loading.load_pristine(CONTEXT["model"], checkpoint=str(checkpoint), sha256=sha)
    model = model.float()
    numbers = [int(z) for z in model.atomic_numbers]
    species = [numbers.index(int(z)) for z in sorted(set(CONTEXT["numbers"]))]
    r, pair, _ = _edges()
    fw = CONTEXT["forward_args"]
    s_count = len(species)
    best = None
    for columns in itertools.permutations(species):
        chain = rt._GenericChain(model, model.interactions[1], list(columns), torch.float64, torch.device("cuda"))
        worst = 0.0
        for value in torch.unique(pair).tolist():
            rows = ((pair == value) & (r < float(model.r_max) - 0.05)).nonzero().flatten()[:256]
            if not rows.numel():
                continue
            with torch.no_grad():
                g = chain(r[rows].double(), value // s_count, value % s_count)
            ref = fw[0][rows, :64].double()
            worst = max(worst, float((g[:, :64] - ref).abs().max() / ref.abs().max()))
        if best is None or worst < best[0]:
            best = (worst, list(columns), chain)
    print(f"sx-layout: species columns {best[1]} (a from the chain vs the captured a: {best[0]:.2e})", flush=True)
    if best[0] > 1e-4:
        raise ValueError(f"sx-layout: no species order reproduces the captured a ({best[0]:.2e})")
    _CACHE["chain"] = (best[2], best[1], float(model.r_max))
    return _CACHE["chain"]


def spline64(knots):
    """float64 [S*S, knots - 1, 4, Q] power-basis coefficients in t of the cubic Hermite spline of w per ordered
    pair (mirror pairs equal), the knot spacing and x0."""
    key = ("spline", knots)
    if key in _CACHE:
        return _CACHE[key]
    import torch

    from fastmace.rewrites import radial as rt
    chain, columns, r_max = _chain_and_columns()
    s_count = len(columns)
    r_lo = rt.R_LO
    h = (r_max - r_lo) / (knots - 1)
    x = r_lo + h * torch.arange(knots, dtype=torch.float64, device="cuda")
    rows = {}
    for s in range(s_count):
        for t in range(s, s_count):
            with torch.no_grad():
                g, dg = chain.with_derivative(x, s, t)
            y = g[:, :-1] @ chain.folded.T
            m = dg[:, :-1] @ chain.folded.T * h
            y0, y1, m0, m1 = y[:-1], y[1:], m[:-1], m[1:]
            rows[(s, t)] = torch.stack([y0, m0, 3 * (y1 - y0) - 2 * m0 - m1, 2 * (y0 - y1) + m0 + m1], dim=1)
    coef = torch.stack([rows[(min(s, t), max(s, t))] for s in range(s_count) for t in range(s_count)])
    _CACHE[key] = (coef, h, r_lo, s_count)
    return _CACHE[key]


def alphas():
    """FOLD_ALPHA's per-path factor of the w' value table (1 without it): table w over a @ W at sample edges."""
    if "alpha" in _CACHE:
        return _CACHE["alpha"]
    import torch
    r, pair, wp = _edges()
    rows = torch.arange(0, r.numel(), max(1, r.numel() // 4096), device=r.device)
    rows = rows[r[rows] < float(wp[16])]
    w, ref = table_w(rows), _reference_w(rows)
    paths = ref.shape[1] // CHANNELS
    out = []
    for p in range(paths):
        a, b = w[:, p * CHANNELS:(p + 1) * CHANNELS].reshape(-1), ref[:, p * CHANNELS:(p + 1) * CHANNELS].reshape(-1)
        keep = b.abs() > 0.1 * b.abs().max()
        out.append(float((a[keep] / b[keep]).median()))
    alpha = torch.tensor(out, dtype=torch.float64, device=r.device)
    err = (w / alpha.repeat_interleave(CHANNELS) - ref).abs().max() / ref.abs().max()
    print(f"sx-layout: w' table factors per path {[round(v, 6) for v in out]}; table w vs a.W after them "
          f"{float(err):.2e}", flush=True)
    if err > 1e-3:
        raise ValueError(f"sx-layout: the w' value table does not reproduce a.W ({float(err):.2e}); is HORNER on?")
    _CACHE["alpha"] = alpha
    return alpha


def table_w(rows):
    """float64 w [len(rows), Q] from the captured w' value table (power basis in t, as HORNER), FOLD_ALPHA included."""
    import torch
    r, pair, wp = _edges()
    g, starts, scales, bins, r_lo, bin_scale = wp[10], wp[11], wp[12], wp[13], float(wp[14]), float(wp[15])
    pieces = starts.numel()
    rr = torch.clamp(r[rows], min=r_lo)
    b = torch.clamp(((rr - r_lo) * bin_scale).int(), max=bins.numel() - 1).long()
    piece = bins[b].long()
    t = ((rr - starts[piece]) * scales[piece] - 1.0).double()
    slab = g[pair[rows].long() * pieces + piece].double()  # [n, Q, 6]
    w = slab[..., -1]
    for k in range(slab.shape[-1] - 2, -1, -1):
        w = w * t[:, None] + slab[..., k]
    return w


# --------------------------------------------------------------------------------------------- forward variants
def _forward_args(mode, knots=None):
    def transform(args):
        import torch
        r, pair, wp = _edges()
        a, weight, x, sh, senders, receivers, row_ptr, out, w = list(args)[:9]
        dummy_f = torch.zeros(1, dtype=torch.float32, device=x.device)
        dummy_i = torch.zeros(1, dtype=torch.int32, device=x.device)
        sx = dict(r=r, pair=pair, pmap=dummy_i, tab=dummy_f, starts=dummy_f, scales=dummy_f, bins=dummy_i)
        if mode == "spline":
            coef, h, r_lo, s_count = spline64(knots)
            unordered = [(s, t) for s in range(s_count) for t in range(s, s_count)]
            pmap = [unordered.index((min(s, t), max(s, t))) for s in range(s_count) for t in range(s_count)]
            sx["tab"] = torch.stack([coef[s * s_count + t] for s, t in unordered]).float().contiguous()
            sx["pmap"] = torch.tensor(pmap, dtype=torch.int32, device=x.device)
            rows = torch.arange(0, r.numel(), max(1, r.numel() // 8192), device=r.device)
            keep = rows[r[rows] < float(wp[16])]
            u = torch.clamp((r[keep].double() - r_lo) / h, min=0)
            iv = torch.clamp(u.long(), max=coef.shape[1] - 1)
            t = (u - iv)[:, None]
            c = coef[pair[keep].long(), iv]
            ws = c[:, 0] + t * (c[:, 1] + t * (c[:, 2] + t * c[:, 3]))
            ref = _reference_w(keep)
            touched = torch.unique(sx["pmap"][pair.long()].long() * coef.shape[1]
                                   + torch.clamp(((r.double() - r_lo) / h).long(), 0, coef.shape[1] - 1)).numel()
            print(f"sx-spline{knots}: {sx['tab'].numel() * 4 / 1e6:.1f} MB table ({len(unordered)} pairs, "
                  f"{coef.shape[1]} intervals, h = {h:.5f} A), {touched} rows touched "
                  f"({touched * 4 * coef.shape[-1] * 4 / 1e6:.1f} MB); spline w vs a.W at {keep.numel()} edges "
                  f"max|d|/max|a| = {float((ws - ref).abs().max() / ref.abs().max()):.2e}", flush=True)
        elif mode == "table6":
            g = wp[10]
            if g.dim() != 3 or g.shape[-1] != 6:
                raise ValueError(f"sx-table6 expects a q-major [groups, Q, 6] value table, got {tuple(g.shape)}")
            alpha = alphas()
            sx["tab"] = (g.double() / alpha.repeat_interleave(CHANNELS)[None, :, None]).float().contiguous()
            sx.update(starts=wp[11].float().contiguous(), scales=wp[12].float().contiguous(),
                      bins=wp[13].int().contiguous())
        elif mode == "wbuf":
            k = 64
            sx["tab"] = (a[:, :k] @ weight[:, :k].T).contiguous()
        return [a, weight, x, sh, senders, receivers, row_ptr, out, w, sx]
    return transform


def _forward_transform(mode, knots=None):
    def transform(production):
        import torch
        wp = CONTEXT["wprime_args"]
        constants = dict(R_MAX=float(wp[16]))
        if mode == "spline":
            coef, h, r_lo, _ = spline64(knots)
            constants.update(SX0=float(r_lo), SXINV=float(1.0 / h), SXNI=int(coef.shape[1]))
        elif mode == "table6":
            constants.update(T_RLO=float(wp[14]), T_BINSCALE=float(wp[15]), T_NBINS=int(wp[13].numel()),
                             T_PIECES=int(wp[11].numel()))
        del torch
        return forward_source(production, mode, constants)
    return transform


def forward_variants(kernel=None, knots=(256, 2048, 4096)):
    table = {"sx-wbuf": ("rounding", _forward_transform("wbuf"), (), _forward_args("wbuf")),
             "sx-table6": ("rounding", _forward_transform("table6"), (), _forward_args("table6")),
             "sx-F-w1": ("falsifier", _forward_transform("w1"), (), _forward_args("w1"))}
    for n in knots:
        table[f"sx-spline{n}"] = ("rounding", _forward_transform("spline", n), (), _forward_args("spline", n))
    return table


# --------------------------------------------------------------------------------------------- backward variants
_WP_PIECE = """    int32_t bin = int32_t((clamped - r_lo) * bin_scale);
    bin = bin < n_bins - 1 ? bin : n_bins - 1;
    const int32_t piece = bins[bin];
    const scalar_t t = (clamped - starts[piece]) * scales[piece] - scalar_t(1);"""


def _wp_transform(knots):
    def transform(s):
        from omat0_kernel_lab import _sub
        coef, h, r_lo, _ = spline64(knots)
        s = _sub(s, "constexpr int TERMS = 6;", "constexpr int TERMS = 4;", 1)
        s = _sub(s, re.search(r"constexpr int PADDED_TERMS = \d+;", s).group(0), "constexpr int PADDED_TERMS = 4;", 1)
        s = _sub(s, _WP_PIECE, f"""    // sx-layout lab: the cubic spline's interval and t in [0, 1) instead of the table's piece.
    const scalar_t sx_u = (clamped - scalar_t({float(r_lo)!r})) * scalar_t({1.0 / h!r});
    int32_t piece = int32_t(sx_u);
    piece = piece < {coef.shape[1]} - 1 ? piece : {coef.shape[1]} - 1;
    const scalar_t t = sx_u - scalar_t(piece);""", 1)
        return _sub(s, "const scalar_t sc = scales[piece];", f"const scalar_t sc = scalar_t({1.0 / h!r});", 1)
    return transform


def _wp_args(knots):
    def transform(args):
        import torch
        args = list(args)
        coef, h, r_lo, s_count = spline64(knots)
        alpha = alphas()
        # q-major [pairs * intervals, Q, 4], FOLD_ALPHA's per-path factor folded in as production folds it.
        tab = (coef * alpha.repeat_interleave(CHANNELS)[None, None, None, :]).permute(0, 1, 3, 2)
        args[10] = tab.reshape(-1, tab.shape[2], 4).float().contiguous()
        args[11] = torch.zeros(coef.shape[1], dtype=args[11].dtype, device=args[11].device)  # pieces = intervals
        return args
    return transform


def wprime_variants(kernel=None, knots=(256, 2048, 4096)):
    return {f"B-spline{n}": ("rounding", _wp_transform(n), (), _wp_args(n)) for n in knots}
