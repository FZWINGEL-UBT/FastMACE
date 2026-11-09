"""A symmetric contraction as a factored invariant circuit: the generated kernels of ``sc_generated`` with the
polynomial evaluated through shared Clebsch-Gordan pair tensors instead of one product per monomial (9 November 2026,
external LLM proposal "factored invariant SC circuit").

``sc_generated`` forms every (weight, monomial, output) term of the descriptor: 480 terms over 410 monomials for
OMAT-0 medium's interaction 0 (16 input components per channel: 0e + 1o + 2e + 3o; 4 outputs: 0e + 1o; 29 weights).
But the 29 basis polynomials ``B_aC(x)`` (one per weight) only span the equivariant polynomials of degree 1-3: 13
invariants and 16 vector covariants. Any other basis ``F_b`` of the same span gives the same contraction, with the
change of basis folded into the per-species weights once (``w' = w K``, at plan time):

    y_C = sum_a w_a B_aC(x) = sum_b w'_b F_bC(x),   B_aC = sum_b K_ab F_bC.

``factor`` builds the ``F_b`` from cuEquivariance's own Clebsch-Gordan tensors (the descriptor's convention):

- pair products ``x_i x_j`` (shared), pair tensors ``P(l1, l2 -> L)`` as linear combinations of them;
- degree-1 features ``x_l``, degree-2 features ``P(l1, l2 -> out)``, degree-3 features ``CG(P(l1, l2 -> L), x_l3 ->
  out)`` as bilinear forms (grouped on the cheaper side, the inner linear combinations shared);

then chooses greedily, cheapest marginal cost first (in FP32 operations, with every shared node counted once), the
candidates that raise the rank of the span, and fits K by least squares in float64 on random inputs. It refuses
(returns None) when the candidates do not span the basis exactly. The kernels are straight-line code over a small
SSA program (``Program``): the forward evaluates it; the backward re-evaluates it and runs its reverse mode. The
division ``x = m / (D + 1)`` and its backward are fused as in ``sc_generated`` (same signatures, the folded weights
in place of the module's).

Rounding: another evaluation order and folded weights, so not bitwise; a switch that uses it needs the force/stress
gate (decision 66).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _literal(value):
    return float(np.float32(value)).hex() + "f"


def irreps_of(segments):
    """[(l, parity, first segment)] for ``segments`` components laid out 0e, 1o, 2e, 3o, ... (spherical-harmonic
    parity), or None if ``segments`` is not 1 + 3 + 5 + ...."""
    out, start, l = [], 0, 0
    while start < segments:
        out.append((l, (-1) ** l, start))
        start += 2 * l + 1
        l += 1
    return out if start == segments else None


class Program:
    """Straight-line SSA over the inputs ``x0..x{n-1}``. Nodes: ``("mul", a, b)``, ``("lin", ((c, u), ...))``
    (sum c u), ``("dot", ((a, b), ...))`` (sum a b). Node ids are ints; inputs are ``-1 - i``. Identical nodes are
    shared (``node`` memoises by content)."""

    def __init__(self, inputs):
        self.inputs = inputs
        self.nodes = []
        self.memo = {}

    def node(self, op):
        key = op
        if key not in self.memo:
            self.memo[key] = len(self.nodes)
            self.nodes.append(op)
        return self.memo[key]

    @staticmethod
    def cost(op):
        kind = op[0]
        return 1 if kind == "mul" else len(op[1])

    def mul(self, a, b):
        return self.node(("mul",) + tuple(sorted((a, b))))

    def lin(self, terms):
        terms = tuple(sorted((u, float(c)) for c, u in terms if c != 0.0))
        terms = tuple((c, u) for u, c in terms)
        if len(terms) == 1 and terms[0][0] == 1.0:
            return terms[0][1]
        return self.node(("lin", terms))

    def dot(self, pairs):
        pairs = tuple(sorted(tuple(sorted(p)) for p in pairs))
        if len(pairs) == 1:
            return self.mul(*pairs[0])
        return self.node(("dot", pairs))

    def evaluate(self, x, upto=None):
        """Values of every node for inputs ``x [n, samples]`` (float64)."""
        values = []

        def get(i):
            return x[-1 - i] if i < 0 else values[i]

        for op in self.nodes[:upto]:
            if op[0] == "mul":
                values.append(get(op[1]) * get(op[2]))
            elif op[0] == "lin":
                values.append(sum(c * get(u) for c, u in op[1]))
            else:
                values.append(sum(get(a) * get(b) for a, b in op[1]))
        return values

    def used(self, roots):
        """Node ids needed for ``roots`` (sorted)."""
        seen, stack = set(), [r for r in roots if r >= 0]
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            op = self.nodes[i]
            children = (op[1], op[2]) if op[0] == "mul" else ([u for _, u in op[1]] if op[0] == "lin"
                                                              else [v for p in op[1] for v in p])
            stack += [c for c in children if c >= 0 and c not in seen]
        return sorted(seen)


def _cg(l1, p1, l2, p2, l3, p3):
    """cuEquivariance's real CG tensor [d1, d2, d3] for O3 (l1 p1) x (l2 p2) -> (l3 p3), or None."""
    import cuequivariance as cue

    c = cue.O3.clebsch_gordan(cue.O3(l1, p1), cue.O3(l2, p2), cue.O3(l3, p3))
    if c.shape[0] == 0:
        return None
    c = np.asarray(c[0], dtype=np.float64)
    c[np.abs(c) < 1e-12] = 0.0
    return c


class _Builder:
    """Candidate features on one Program."""

    def __init__(self, prog, irreps):
        self.prog, self.irreps = prog, irreps
        self.pairs = {}

    def comps(self, l):
        start = next(s for ll, _, s in self.irreps if ll == l)
        return [-1 - (start + m) for m in range(2 * l + 1)]

    def parity(self, l):
        return next(p for ll, p, _ in self.irreps if ll == l)

    def pair(self, l1, l2, big_l):
        """Components of P(l1, l2 -> L) (node ids), or None when the coupling does not exist or vanishes."""
        key = (l1, l2, big_l)
        if key in self.pairs:
            return self.pairs[key]
        p3 = self.parity(l1) * self.parity(l2)
        cg = _cg(l1, self.parity(l1), l2, self.parity(l2), big_l, p3)
        result = None
        if cg is not None:
            a, b = self.comps(l1), self.comps(l2)
            out = []
            for m in range(2 * big_l + 1):
                terms = defaultdict(float)
                for i in range(len(a)):
                    for j in range(len(b)):
                        c = cg[i, j, m]
                        if c:
                            terms[tuple(sorted((a[i], b[j])))] += c
                terms = {k: v for k, v in terms.items() if abs(v) > 1e-12}
                if not terms:
                    out.append(None)
                    continue
                out.append(self.prog.lin([(c, self.prog.mul(*k)) for k, c in terms.items()]))
            result = None if all(o is None for o in out) else out
        self.pairs[key] = result
        return result

    def bilinear(self, cg, left, right):
        """Components (node ids) of sum_{i,j} cg[i, j, m] left_i right_j, grouped on the side with fewer distinct
        inner sums; None entries in left/right are zeros."""
        n_out = cg.shape[2]
        options = []
        for flip in (False, True):
            outer, inner = (right, left) if flip else (left, right)
            c = np.transpose(cg, (1, 0, 2)) if flip else cg
            options.append((flip, outer, inner, c))
        best = None
        for flip, outer, inner, c in options:
            prog_nodes = len(self.prog.nodes)
            comps = []
            for m in range(n_out):
                pairs = []
                for i, o in enumerate(outer):
                    if o is None:
                        continue
                    terms = [(c[i, j, m], u) for j, u in enumerate(inner) if u is not None and abs(c[i, j, m]) > 1e-12]
                    if terms:
                        pairs.append((o, self.prog.lin(terms)))
                comps.append(self.prog.dot(pairs) if pairs else None)
            cost = sum(Program.cost(op) for op in self.prog.nodes[prog_nodes:])
            # roll back: the builder is re-run for the cheaper side below
            for op in self.prog.nodes[prog_nodes:]:
                del self.prog.memo[op]
            del self.prog.nodes[prog_nodes:]
            if best is None or cost < best[0]:
                best = (cost, flip, outer, inner, c)
        _, flip, outer, inner, c = best
        comps = []
        for m in range(n_out):
            pairs = []
            for i, o in enumerate(outer):
                if o is None:
                    continue
                terms = [(c[i, j, m], u) for j, u in enumerate(inner) if u is not None and abs(c[i, j, m]) > 1e-12]
                if terms:
                    pairs.append((o, self.prog.lin(terms)))
            comps.append(self.prog.dot(pairs) if pairs else None)
        return comps


def _candidates(irreps, out_l, out_p, degrees):
    """[(description, builder function)] of features transforming as (out_l, out_p)."""
    ls = [l for l, _, _ in irreps]
    par = {l: p for l, p, _ in irreps}
    found = []
    if 1 in degrees and out_l in par and par[out_l] == out_p:
        found.append((("x", out_l), lambda b: [u for u in b.comps(out_l)]))
    for l1 in ls:
        for l2 in ls:
            if l2 < l1:
                continue
            for big_l in range(abs(l1 - l2), l1 + l2 + 1):
                p = par[l1] * par[l2]
                if 2 in degrees and big_l == out_l and p == out_p:
                    found.append((("P", l1, l2, big_l), lambda b, k=(l1, l2, big_l): b.pair(*k)))
                if 3 not in degrees:
                    continue
                for l3 in ls:
                    if not abs(big_l - l3) <= out_l <= big_l + l3 or p * par[l3] != out_p:
                        continue

                    def build(b, k=(l1, l2, big_l, l3)):
                        pair = b.pair(k[0], k[1], k[2])
                        if pair is None:
                            return None
                        cg = _cg(k[2], par[k[0]] * par[k[1]], k[3], par[k[3]], out_l, out_p)
                        if cg is None:
                            return None
                        return b.bilinear(cg, pair, b.comps(k[3]))

                    found.append((("PX", l1, l2, big_l, l3), build))
    return found


def _basis_values(terms, x):
    """{(a, C): values [samples]} of the descriptor's basis polynomials at ``x [n, samples]``."""
    out = defaultdict(float)
    for (a, mono, c), coefficient in terms.items():
        value = coefficient
        for i in mono:
            value = value * x[i]
        out[(a, c)] = out[(a, c)] + value
    return out


def factor(terms, segments_in, segments_w, segments_out, samples=96, seed=0):
    """``(program, features, K)`` or None: ``features [segments_w]`` lists per new weight b its output segments'
    node ids ``{C: id}``; ``K [segments_w, segments_w]`` float64 with ``w'_b = sum_a w_a K[a, b]``."""
    irreps_in, irreps_out = irreps_of(segments_in), irreps_of(segments_out)
    if irreps_in is None or irreps_out is None:
        return None
    degrees = sorted({len(m) for _, m, _ in terms})
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((segments_in, samples))
    basis = _basis_values(terms, x)
    prog = Program(segments_in)
    builder = _Builder(prog, irreps_in)
    features = [None] * segments_w
    K = np.zeros((segments_w, segments_w))
    next_b = 0
    for out_l, out_p, start in irreps_out:
        cs = list(range(start, start + 2 * out_l + 1))
        weights = sorted({a for (a, c) in basis if c in cs})
        if not weights:
            continue
        # target matrix [samples * components, weights]
        target = np.stack([np.concatenate([np.broadcast_to(basis.get((a, c), 0.0), (samples,)) for c in cs])
                           for a in weights], axis=1)
        rank = np.linalg.matrix_rank(target)
        if rank != len(weights):
            return None
        q_target, _ = np.linalg.qr(target)
        chosen, chosen_values = [], []
        candidates = _candidates(irreps_in, out_l, out_p, degrees)
        while len(chosen) < rank:
            best = None
            for desc, build in candidates:
                if desc in [d for d, _ in chosen]:
                    continue
                mark = len(prog.nodes)
                comps = build(builder)
                cost = sum(Program.cost(op) for op in prog.nodes[mark:])
                values = None
                if comps is not None and any(u is not None for u in comps):
                    vals = prog.evaluate(x)
                    values = np.concatenate([(x[-1 - u] if u < 0 else vals[u]) if u is not None
                                             else np.zeros(samples) for u in comps])
                # roll back the trial nodes (pair caches that point at them too)
                for op in prog.nodes[mark:]:
                    del prog.memo[op]
                del prog.nodes[mark:]
                builder.pairs = {k: v for k, v in builder.pairs.items()
                                 if v is None or all(u is None or u < mark for u in v)}
                if values is None or np.linalg.norm(values) == 0.0:
                    continue
                # inside the target span, and raising the rank
                inside = np.linalg.norm(values - q_target @ (q_target.T @ values)) <= 1e-9 * np.linalg.norm(values)
                if not inside:
                    continue
                stack = np.stack(chosen_values + [values], axis=1)
                if np.linalg.matrix_rank(stack, tol=1e-9 * np.abs(stack).max()) <= len(chosen_values):
                    continue
                if best is None or cost < best[0]:
                    best = (cost, desc, build)
            if best is None:
                return None
            _, desc, build = best
            comps = build(builder)
            vals = prog.evaluate(x)
            values = np.concatenate([(x[-1 - u] if u < 0 else vals[u]) if u is not None else np.zeros(samples)
                                     for u in comps])
            chosen.append((desc, comps))
            chosen_values.append(values)
            factor.log.append((out_l, desc, best[0]))
        F = np.stack(chosen_values, axis=1)  # [samples * comps, rank]
        coef, *_ = np.linalg.lstsq(F, target, rcond=None)  # target ~ F @ coef: B_a = sum_b coef[b, a] F_b
        if np.linalg.norm(F @ coef - target) > 1e-9 * np.linalg.norm(target):
            return None
        for k, (desc, comps) in enumerate(chosen):
            b = next_b + k
            features[b] = {c: comps[m] for m, c in enumerate(cs)}
            for col, a in enumerate(weights):
                K[a, b] = coef[k, col]
        next_b += rank
    if next_b != segments_w:
        return None
    return prog, features, K


factor.log = []


def check(terms, prog, features, K, segments_in, samples=16, seed=1):
    """Max relative error of the factored polynomial against the descriptor's, float64, random x and w."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((segments_in, samples))
    w = rng.standard_normal((K.shape[0], samples))
    basis = _basis_values(terms, x)
    vals = prog.evaluate(x)
    get = lambda u: x[-1 - u] if u < 0 else vals[u]  # noqa: E731
    outputs = sorted({c for _, _, c in terms})
    ref = {c: sum(w[a] * basis[(a, cc)] for (a, cc) in basis if cc == c) for c in outputs}
    wp = K.T @ w
    got = {c: sum(wp[b] * get(f[c]) for b, f in enumerate(features) if f is not None and f.get(c) is not None)
           for c in outputs}
    scale = max(np.abs(ref[c]).max() for c in outputs)
    return max(np.abs(got[c] - ref[c]).max() for c in outputs) / scale


def counts(prog, features):
    roots = [u for f in features for u in f.values() if u is not None]
    used = prog.used(roots)
    by_kind = defaultdict(int)
    for i in used:
        by_kind[prog.nodes[i][0]] += Program.cost(prog.nodes[i])
    outputs = sum(1 for f in features for u in f.values() if u is not None)
    return dict(nodes=len(used), ops=sum(by_kind.values()), by_kind=dict(by_kind), output_fma=outputs)


def _expr(u):
    return f"x{-1 - u}" if u < 0 else f"v{u}"


def _emit_node(lines, i, op):
    if op[0] == "mul":
        lines.append(f"  const float v{i} = {_expr(op[1])} * {_expr(op[2])};")
        return
    if op[0] == "lin":
        (c0, u0), rest = op[1][0], op[1][1:]
        expr = f"{_literal(c0)} * {_expr(u0)}"
        for c, u in rest:
            expr = f"fmaf({_literal(c)}, {_expr(u)}, {expr})"
    else:
        (a0, b0), rest = op[1][0], op[1][1:]
        expr = f"{_expr(a0)} * {_expr(b0)}"
        for a, b in rest:
            expr = f"fmaf({_expr(a)}, {_expr(b)}, {expr})"
    lines.append(f"  const float v{i} = {expr};")


def _header(name, segments_in, segments_w, mul, extra_args, extra_rows):
    return [f'extern "C" __global__ void __launch_bounds__({mul}) {name}(',
            "    const float* __restrict__ m, const float* __restrict__ d, const float* __restrict__ w,",
            f"    const int* __restrict__ sp, {extra_args}) {{",
            "  const int a = blockIdx.x, ch = threadIdx.x;",
            f"  const float* xr = m + (long long)a * {segments_in * mul} + ch;",
            f"  const float* wr = w + (long long)sp[a] * {segments_w * mul} + ch;"] + extra_rows + [
            "  const float s = 1.0f / (d[a] + 1.0f);"] + [
            f"  const float x{i} = __ldg(xr + {i * mul}) * s;" for i in range(segments_in)]


def render_forward(prog, features, segments_in, segments_w, segments_out, mul, name="sc_factored_forward"):
    """CUDA source of ``name(m, d, w', sp, y)`` (w' the folded weights)."""
    roots = [u for f in features for u in f.values() if u is not None]
    lines = _header(name, segments_in, segments_w, mul, "float* __restrict__ y", [])
    for i in prog.used(roots):
        _emit_node(lines, i, prog.nodes[i])
    lines.append("  float " + ", ".join(f"y{c} = 0.0f" for c in range(segments_out)) + ";")
    for b, f in enumerate(features):
        items = [(c, u) for c, u in sorted(f.items()) if u is not None]
        if items:
            lines.append(f"  {{ const float wb = __ldg(wr + {b * mul});")
            lines += [f"    y{c} = fmaf(wb, {_expr(u)}, y{c});" for c, u in items]
            lines.append("  }")
    lines += [f"  y[(long long)a * {segments_out * mul} + {c * mul} + ch] = y{c};" for c in range(segments_out)]
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_backward(prog, features, segments_in, segments_w, segments_out, mul, name="sc_factored_backward"):
    """CUDA source of ``name(m, d, w', sp, g, dm, dd)``: the forward program, then its reverse mode."""
    roots = [u for f in features for u in f.values() if u is not None]
    used = prog.used(roots)
    rows = [f"  const float* gr = g + (long long)a * {segments_out * mul} + ch;"]
    lines = _header(name, segments_in, segments_w, mul,
                    "const float* __restrict__ g, float* __restrict__ dm, float* __restrict__ dd", rows)
    lines += [f"  const float g{c} = __ldg(gr + {c * mul});" for c in range(segments_out)]
    for i in used:
        _emit_node(lines, i, prog.nodes[i])
    # adjoint seeds: d F_bC = w'_b g_C
    adj = {}  # node id or input -> name of its adjoint accumulator (declared on first write)

    def add(u, expr_mul):
        """adj[u] += product given as (a, b) expressions or a single expression."""
        name = f"d{_expr(u)}"
        if isinstance(expr_mul, tuple):
            a, b = expr_mul
            if u in adj:
                lines.append(f"  {name} = fmaf({a}, {b}, {name});")
            else:
                lines.append(f"  float {name} = {a} * {b};")
        else:
            if u in adj:
                lines.append(f"  {name} += {expr_mul};")
            else:
                lines.append(f"  float {name} = {expr_mul};")
        adj[u] = name

    for b, f in enumerate(features):
        items = [(c, u) for c, u in sorted(f.items()) if u is not None]
        if not items:
            continue
        lines.append(f"  const float w{b} = __ldg(wr + {b * mul});")
        for c, u in items:
            add(u, (f"w{b}", f"g{c}"))
    for i in reversed(used):
        if i not in adj:
            continue
        op, da = prog.nodes[i], adj[i]
        if op[0] == "mul":
            add(op[1], (da, _expr(op[2])))
            add(op[2], (da, _expr(op[1])))
        elif op[0] == "lin":
            for c, u in op[1]:
                add(u, (_literal(c), da))
        else:
            for a_, b_ in op[1]:
                add(a_, (da, _expr(b_)))
                add(b_, (da, _expr(a_)))
    lines.append("  float dot = 0.0f;")
    for i in range(segments_in):
        u = -1 - i
        dx = adj.get(u, "0.0f")
        lines.append(f"  dot = fmaf({dx}, x{i}, dot);")
        lines.append(f"  dm[(long long)a * {segments_in * mul} + {i * mul} + ch] = {dx} * s;")
    warps = mul // 32
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


def fold_weights(weight, K, segments_w, mul):
    """``w' [elements, segments_w * mul]`` from the module's weight ``[elements, segments_w * mul]`` (float64
    product, returned in weight's dtype)."""
    import torch

    k = torch.as_tensor(K, dtype=torch.float64, device=weight.device)
    w = weight.reshape(weight.shape[0], segments_w, mul).to(torch.float64)
    return torch.einsum("zau,ab->zbu", w, k).reshape(weight.shape).to(weight.dtype).contiguous()


__all__ = ["Program", "check", "counts", "factor", "fold_weights", "irreps_of", "render_backward", "render_forward"]
