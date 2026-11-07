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
def _braced(text, open_index):
    """The text inside the braces opening at ``open_index`` and the index after the closing brace."""
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:i], i + 1
    raise ValueError("sx-layout: unbalanced braces")


def parse_forward(source):
    """Per path part of the production forward: path indices, per-path W factors, x offsets, the per-edge angular
    block and the output stores. The angular block is the braced block right after the receiver-change loop
    ``while (cur < rv[b]) {...}``, in terms of ``sh_s[b * ED + j]``, ``xv[b][k]``, ``w[g][b]`` and ``acc[i]``."""
    parts = []
    for body in re.split(r"\n    case \d+: \{", source)[1:]:
        body = body.split("} break;")[0]
        wst = [int(v) for v in re.search(r"const int wst\[\d+\] = \{([^}]*)\};", body).group(1).split(",")]
        factors = re.search(r"const float cpf\[\d+\] = \{([^}]*)\};", body)
        cpf = ([float.fromhex(v.strip().rstrip("f")) for v in factors.group(1).split(",")] if factors
               else [1.0] * len(wst))
        xs = re.findall(r"xv\[b\]\[(\d+)\] = \(snd >= 0\) \? x\[\(int64_t\)snd \* IN_DIM \+ (\d+) \+ ch\] : 0\.f;",
                        body)
        count = int(re.search(r"float acc\[(\d+)\];", body).group(1))
        loop = body.index("while (cur < rv[b]) {")
        change, after = _braced(body, body.index("{", loop))
        block, _ = _braced(body, body.index("{", after))
        stores = re.findall(r"(out\d+)\[cur \* (\d+) \+ (\d+) \+ ch\] = acc\[(\d+)\];", change)
        written = {int(i) for i in re.findall(r"\bacc\[(\d+)\] = fmaf", block)}
        code = re.sub(r"//.*", "", block)  # comments may say "(b)"
        stray = re.search(r"\bb\b", re.sub(r"(sh_s\[b \* ED|xv\[b\]|w\[\d+\]\[b\])", "", code))
        if not xs or len(stores) != count or written != set(range(count)) or len(cpf) != len(wst) or stray:
            raise ValueError(f"sx-layout: could not parse a path part (x {len(xs)}, stores {len(stores)}, "
                             f"acc written {len(written)} of {count}, cpf {len(cpf)}, wst {len(wst)}, "
                             f"stray b {bool(stray)})")
        parts.append(dict(paths=[v // CHANNELS for v in wst], cpf=cpf, xs=[(int(k), int(o)) for k, o in xs],
                          count=count, block=block,
                          stores=[(o, int(st), int(off), int(i)) for o, st, off, i in stores]))
    if not parts:
        raise ValueError("sx-layout: no path parts in the production forward")
    return parts



def _angular(parts):
    """The per-edge code (after wv[] is set) and the stores, with per-part names."""
    lines, stores, declarations = [], [], []
    for index, part in enumerate(parts):
        declarations.append(f"        float acc{index}[{part['count']}];\n"
                            f"        #pragma unroll\n"
                            f"        for (int i = 0; i < {part['count']}; ++i) acc{index}[i] = 0.f;")
        for k, offset in part["xs"]:
            lines.append(f"            const float x{index}_{k} = x[(int64_t)snd * IN_DIM + {offset} + ch];")
        for g, (path, factor) in enumerate(zip(part["paths"], part["cpf"])):
            value = f"wv[{path}]" if factor == 1.0 else f"wv[{path}] * {factor.hex()}f"
            lines.append(f"            const float w{index}_{g} = {value};  // production's per-path W factor")
        text = "            {" + part["block"] + "}"
        text = re.sub(r"sh_s\[b \* ED \+ (\d+)\]", r"__ldg(sh + e * ED + \1)", text)
        text = re.sub(r"xv\[b\]\[(\d+)\]", lambda m: f"x{index}_{m.group(1)}", text)
        text = re.sub(r"w\[(\d+)\]\[b\]", lambda m: f"w{index}_{m.group(1)}", text)
        text = re.sub(r"\bacc\[(\d+)\]", lambda m: f"acc{index}[{m.group(1)}]", text)
        lines.append(text)
        for out, stride, offset, i in part["stores"]:
            stores.append(f"        {out}[(int64_t)rcv * {stride} + {offset} + ch] = acc{index}[{i}];")
    return "\n".join(declarations), "\n".join(lines), "\n".join(stores)



# --------------------------------------------------------------------------------------------- the forward source
_W_CODE = {
    "spline": """SX_LOOKUP
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
    w_code = (_W_CODE[mode].replace("SX_LOOKUP", lookup_code(constants.pop("SX_GRID"), "re", "float", "            "))
              if mode == "spline" else _W_CODE[mode])
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
{w_code}
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
    del loading, sha  # the plain checkpoint, as tools/sx_spline_error.py: the chain needs the e3nn radial modules
    model = torch.load(str(checkpoint), map_location="cpu", weights_only=False).float()
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


# Piecewise-uniform knots (follow-up, 7 November 2026: dense where w curves, at short r; tools/sx_spline_knots.py):
# segments "start:end:intervals", each uniform. "a695" passes the radial certificate over the whole domain (slope
# 0.63), "p399" only at r >= 1 A (0.63; coarse below 1 A).
GRIDS = {"a695": "0.3:0.5:150,0.5:0.75:110,0.75:1.0:85,1.0:1.5:110,1.5:2.0:50,2.0:2.5:30,2.5:6.0:160",
         "p399": "0.3:1.0:32,1.0:1.5:110,1.5:2.5:100,2.5:6.0:157"}


def grid_of(spec):
    """[(start, end, intervals)] for an int knot count (uniform on [R_LO, r_max)) or a GRIDS name."""
    from fastmace.rewrites import radial as rt
    _, _, r_max = _chain_and_columns()
    if isinstance(spec, int):
        return [(rt.R_LO, r_max, spec - 1)]
    grid = [(float(a), float(b), int(n)) for a, b, n in (part.split(":") for part in GRIDS[spec].split(","))]
    if abs(grid[0][0] - rt.R_LO) > 1e-12 or abs(grid[-1][1] - r_max) > 1e-12:
        raise ValueError(f"sx-layout: grid {spec} does not span [{rt.R_LO}, {r_max})")
    return grid


def locate(r64, grid):
    """float64 (interval, t, 1/h) per length, as the kernels' lookup."""
    import torch
    iv = torch.zeros_like(r64, dtype=torch.long)
    t, inv = torch.zeros_like(r64), torch.zeros_like(r64)
    offset = 0
    for k, (a, b, n) in enumerate(grid):
        mask = (r64 >= a) if k else torch.ones_like(r64, dtype=torch.bool)
        u = torch.clamp((r64 - a) * (n / (b - a)), min=0)
        i = torch.clamp(u.long(), max=n - 1)
        iv, t = torch.where(mask, offset + i, iv), torch.where(mask, u - i, t)
        inv = torch.where(mask, torch.full_like(r64, n / (b - a)), inv)
        offset += n
    return iv, t, inv


def lookup_code(grid, r, scalar, indent):
    """C code defining ``iv`` (global interval), ``t`` in [0, 1) and ``sx_inv`` (1/h) from length ``r``: one compare
    per segment boundary (an edge is one warp's for every channel, so the branch is uniform), then one multiply."""
    def f(v):
        return f"{scalar}({float(v)!r})" if scalar != "float" else f"{float(v)!r}f"
    lines = [f"{indent}int iv; {scalar} t, sx_inv;"]
    offset = 0
    for k, (a, b, n) in enumerate(grid):
        head = "if" if k == 0 else "else if"
        cond = f"{head} ({r} < {f(b)}) " if k < len(grid) - 1 else ("else " if k else "")
        body = (f"{{ sx_inv = {f(n / (b - a))}; {scalar} u = ({r} - {f(a)}) * sx_inv; u = u > {f(0)} ? u : {f(0)}; "
                f"int il = (int)u; il = il < {n - 1} ? il : {n - 1}; t = u - ({scalar})il; iv = {offset} + il; }}")
        lines.append(f"{indent}{cond}{body}")
        offset += n
    return "\n".join(lines)


def spline64(spec):
    """float64 [S*S, intervals, 4, Q] power-basis coefficients in each interval's t of the cubic Hermite spline of w
    per ordered pair (mirror pairs equal), on the knots of ``spec`` (see ``grid_of``); the grid; r_lo; the species
    count."""
    key = ("spline", spec)
    if key in _CACHE:
        return _CACHE[key]
    import torch

    from fastmace.rewrites import radial as rt
    chain, columns, r_max = _chain_and_columns()
    s_count = len(columns)
    r_lo = rt.R_LO
    grid = grid_of(spec)
    xs = [torch.linspace(a, b, n + 1, dtype=torch.float64, device="cuda") for a, b, n in grid]
    x = torch.cat([xs[0]] + [v[1:] for v in xs[1:]])
    h = (x[1:] - x[:-1])[:, None]
    rows = {}
    for s in range(s_count):
        for t in range(s, s_count):
            with torch.no_grad():
                g, dg = chain.with_derivative(x, s, t)
            y = g[:, :-1] @ chain.folded.T
            m = dg[:, :-1] @ chain.folded.T
            y0, y1, m0, m1 = y[:-1], y[1:], m[:-1] * h, m[1:] * h
            rows[(s, t)] = torch.stack([y0, m0, 3 * (y1 - y0) - 2 * m0 - m1, 2 * (y0 - y1) + m0 + m1], dim=1)
    coef = torch.stack([rows[(min(s, t), max(s, t))] for s in range(s_count) for t in range(s_count)])
    _CACHE[key] = (coef, grid, r_lo, s_count)
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
            coef, grid, r_lo, s_count = spline64(knots)
            unordered = [(s, t) for s in range(s_count) for t in range(s, s_count)]
            pmap = [unordered.index((min(s, t), max(s, t))) for s in range(s_count) for t in range(s_count)]
            sx["tab"] = torch.stack([coef[s * s_count + t] for s, t in unordered]).float().contiguous()
            sx["pmap"] = torch.tensor(pmap, dtype=torch.int32, device=x.device)
            rows = torch.arange(0, r.numel(), max(1, r.numel() // 8192), device=r.device)
            keep = rows[r[rows] < float(wp[16])]
            iv, t, _ = locate(r[keep].double(), grid)
            t = t[:, None]
            c = coef[pair[keep].long(), iv]
            ws = c[:, 0] + t * (c[:, 1] + t * (c[:, 2] + t * c[:, 3]))
            ref = _reference_w(keep)
            touched = torch.unique(sx["pmap"][pair.long()].long() * coef.shape[1]
                                   + locate(r.double(), grid)[0]).numel()
            spacing = ", ".join(f"{a}-{b}: {(b - a) / n:.5f}" for a, b, n in grid)
            print(f"sx-spline{knots}: {sx['tab'].numel() * 4 / 1e6:.1f} MB table ({len(unordered)} pairs, "
                  f"{coef.shape[1]} intervals, h = {spacing} A), {touched} rows touched "
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
            sx["tab"] = (a[:, :k].double() @ weight[:, :k].double().T).float().contiguous()  # no TF32
        return [a, weight, x, sh, senders, receivers, row_ptr, out, w, sx]
    return transform


def _forward_transform(mode, knots=None):
    def transform(production):
        import torch
        wp = CONTEXT["wprime_args"]
        constants = dict(R_MAX=float(wp[16]))
        if mode == "spline":
            coef, grid, _, _ = spline64(knots)
            constants.update(SXNI=int(coef.shape[1]), SX_GRID=grid)
        elif mode == "table6":
            constants.update(T_RLO=float(wp[14]), T_BINSCALE=float(wp[15]), T_NBINS=int(wp[13].numel()),
                             T_PIECES=int(wp[11].numel()))
        del torch
        return forward_source(production, mode, constants)
    return transform


def forward_variants(kernel=None, knots=(256, 2048, 4096, *GRIDS)):
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
        coef, grid, _, _ = spline64(knots)
        s = _sub(s, "constexpr int TERMS = 6;", "constexpr int TERMS = 4;", 1)
        s = _sub(s, re.search(r"constexpr int PADDED_TERMS = \d+;", s).group(0), "constexpr int PADDED_TERMS = 4;", 1)
        spline = ("    // sx-layout lab: the cubic spline's interval and t in [0, 1) instead of the table's piece.\n"
                  + lookup_code(grid, "clamped", "scalar_t", "    ") + "\n    const int32_t piece = iv;")
        # The table's piece lookup, from the bin to t (the exact text of 6 November as _WP_PIECE; any later
        # rendering of the same four statements by the pattern).
        pattern = (r"    int32_t bin = int32_t\(\(clamped - r_lo\) \* bin_scale\);.*?"
                   r"const scalar_t t = \(clamped - starts\[piece\]\) \* scales\[piece\] - scalar_t\(1\);")
        if len(re.findall(pattern, s, flags=re.S)) != 1:
            raise ValueError("B-spline: the w' piece lookup was not found once")
        s = re.sub(pattern, lambda m: spline, s, flags=re.S)
        if len(re.findall(r"const scalar_t sc = [^;]*;", s)) != 1:
            raise ValueError("B-spline: the w' slope scale was not found once")
        return re.sub(r"const scalar_t sc = [^;]*;", "const scalar_t sc = sx_inv;", s)
    return transform


def _wp_args(knots):
    def transform(args):
        import torch
        args = list(args)
        coef, _, _, _ = spline64(knots)
        alpha = alphas()
        # q-major [pairs * intervals, Q, 4], FOLD_ALPHA's per-path factor folded in as production folds it.
        tab = (coef * alpha.repeat_interleave(CHANNELS)[None, None, None, :]).permute(0, 1, 3, 2)
        args[10] = tab.reshape(-1, tab.shape[2], 4).float().contiguous()
        args[11] = torch.zeros(coef.shape[1], dtype=args[11].dtype, device=args[11].device)  # pieces = intervals
        return args
    return transform


def wprime_variants(kernel=None, knots=(256, 2048, 4096, *GRIDS)):
    return {f"B-spline{n}": ("rounding", _wp_transform(n), (), _wp_args(n)) for n in knots}


# --------------------------------------------------------------------------------------------- guard
# Largest message error against the production base (max|d|/max|a|) before any timing: the layout with w = a.W must
# match production to rounding; the table and the splines to their approximation. A larger value is a bug, and the
# job stops before the timed rounds (owner's budget: a broken kernel should cost only its compile time).
GUARD = {"sx-wbuf": 1e-4, "sx-table6": 1e-3, "sx-spline": 0.05, "B-spline": 0.05}


def _worst(compare):
    values = []
    for key, text in compare.items():
        if key == "against":
            continue
        if text == "bitwise":
            values.append(0.0)
            continue
        found = re.search(r"max\|d\|/max\|a\| = ([0-9.eE+-]+|nan|inf)", str(text))
        values.append(float(found.group(1)) if found else float("inf"))
    return max(values) if values else float("inf")


def check_before_timing(kernel_name, built):
    """Raise SystemExit when a sx-layout variant failed to build or its outputs are off by more than GUARD."""
    problems = []
    for variant, b in built.items():
        limit = next((v for k, v in GUARD.items() if variant.startswith(k)), None)
        if limit is None:
            continue
        if "launch" not in b:
            problems.append(f"{variant}: not built ({b.get('error', '')[-300:]})")
            continue
        worst = _worst(b.get("compare", {}))
        print(f"{kernel_name} guard {variant}: worst max|d|/max|a| {worst:.2e} (limit {limit:g})", flush=True)
        if not worst <= limit:
            problems.append(f"{variant}: {worst:.2e} > {limit:g} {b.get('compare')}")
    if problems and kernel_name == "wprime":  # the forward's results are already in the log: report, keep going
        print("sx-layout guard (wprime, not fatal):\n" + "\n".join(problems), flush=True)
    elif problems:
        raise SystemExit("sx-layout guard failed before timing:\n" + "\n".join(problems))
