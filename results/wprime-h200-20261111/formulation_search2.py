"""Wider formulation search for the w' channel loop (part 1 of the H200 w' lane, 12 November 2026).

Input: schedule_terms.json (dump_schedule.py: interaction 2 of OMAT-0 medium as the kernel sees it; per path its
terms (o, i, j, c) and FOLD_ALPHA's alpha, so a term costs a multiply only where |c| / alpha != 1).

Per channel the kernel must produce
  dx_i   = sum_p w_p U_pi,      U_pi = sum_{o,j} C g_o y_j          (4 values, atomics)
  dy_j  += sum_p w_p V_pj,      V_pj = sum_{o,i} C g_o x_i          (16 lane accumulators)
  R     += sum_p dw_p T_p,      T_p  = sum_{o,i,j} C g_o x_i y_j    (1 lane accumulator)
with g (dout, 40 per channel, LDS), x (4, LDG), w_p and dw_p (dual Horner per path, fixed) channel-dependent and
y (16) edge-only.

Cost: issue slots per channel pass on sm_90 (the H200 is issue-bound and an FFMA issues once per warp and clock,
like any other instruction): FFMA / FMUL / FADD 1 each, a sign free; an edge-only value used in the channel pass costs
a quarter LDS (LDS.128 from a per-edge staged vector) unless it is one of the 16 harmonics the kernel already stages
(y is read as 4 LDS.128 per pass either way); edge-only arithmetic is per edge and lane, so it counts 1/4 per pass
(4 passes per edge); live = channel-dependent values held at once (a register proxy).

Strategies (all exact in real arithmetic; every one is a rounding change against today's kernel but the current):
  per path (the 11 November space, now with the real coefficients and loads):
    U: terms | M (M_io = sum_j C y_j per edge, staged) | N (N_ij = sum_o C g_o)
    V: terms | N | K (K_oj = sum_i C x_i)
    T: xU | yV | terms
    SCALAR_PATHS for l1 = 0 (today's): dy_j += (w x0) c g_o per term, T = x0 u.
  cross-path (new):
    H  pre-scale g by w per output (gh_o = w_p g_o: one FMUL per output, the alpha ratio folded where it is one
       constant per output), then dx_i = sum_o A_oi gh_o over all paths (A = M, staged), dy_j = sum C x_i gh_o
       (terms, or K forms shared across paths and outputs up to sign), T_p = sum_o g_o B_o (B_o = sum_i x_i A_oi).
    G  group paths with the same (input, edge) irreps ((1o,1o) -> 0e, 2e; (1o,2e) -> 1o, 3o): their V sums share
       the x_i products once g is pre-scaled (H), and their K forms coincide.
    CSE: distinct linear forms in x (K_oj up to sign and scale, across all paths) built once per pass; each use one
       FFMA. Greedy pairing of shared sub-sums inside the K forms (multi-output CSE).
  basis change: the complex (m) basis's term count per path (each complex product costs 4 real FFMA, conjugate
    pairs halve it): reported, not a candidate unless it beats the real count.
Writes formulation_search2.json; prints the best per-pass FP count and the bound in microseconds and step share.
"""
import collections
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EPS = 1e-9
LDS_PER_VALUE = 0.25          # one LDS.128 per four staged edge-only values
EDGE_ARITH_PER_PASS = 0.25    # per-edge arithmetic, once per edge and lane, over four passes
# today's census (census_variants.json, sm_90): FP per pass and instructions per edge
CENSUS = dict(fp_base=361, fp_skip=349, per_edge_base=2198, passes=4)
H200 = dict(wprime_us=1092.0, step_us=3369.0)  # 9 Nov profile; SrTiO3 10x10x10 MD step (BUTTERFLY A/B, ms 3.369)


def load():
    d = json.loads((HERE / "schedule_terms.json").read_text())
    paths = []
    for p in d["paths"]:
        terms = [(t["o"], t["i"], t["j"], t["c"] / p["alpha"]) for out in p["outputs"] for t in out]
        paths.append(dict(I=p["input_dim"], J=p["edge_dim"], O=len(p["outputs"]), terms=terms, alpha=p["alpha"]))
    return paths


def unit(c):
    return abs(abs(c) - 1.0) < 1e-6


def path_counts(p):
    t = p["terms"]
    oc = {(o, round(abs(c), 9)) for o, _, _, c in t}
    return dict(nnz=len(t), io=len({(o, i) for o, i, _, _ in t}), ij=len({(i, j) for _, i, j, _ in t}),
                oj=len({(o, j) for o, _, j, _ in t}), s_fmul=sum(1 for _, m in oc if not unit(m)),
                ii=len({i for _, i, _, _ in t}), jj=len({j for _, _, j, _ in t}), oo=p["O"])


def fixed_per_path(p):
    """Per path and pass, outside the tensor product: the dual Horner (5 FFMA for 4 terms), the slope (FMUL + FFMA)."""
    return 5 + 2


def per_path_strategies(p):
    """[(name, fp, edge-only loads, live)] for one path (the 11 November tree space with the real coefficients)."""
    k = path_counts(p)
    rows = []
    scalar = p["I"] == 1
    if scalar:  # SCALAR_PATHS (today): wx = w x0 (FMUL); per term u += s y_j and dy_j += wx s; T = x0 u; dx += w u
        fp = 1 + k["s_fmul"] + 2 * k["nnz"] + 1 + 1
        rows.append(("scalar_paths", fp, 0.0, 2))
        # M for U: u = sum_o M_o g_o (M_o = sum_j C y_j, staged), dy terms as before
        fp = 1 + k["s_fmul"] + k["io"] + k["nnz"] + 1 + 1
        rows.append(("scalar_paths/M", fp, k["io"] * LDS_PER_VALUE, 2))
        return rows
    for u, v, t in itertools.product(("terms", "M", "N"), ("terms", "N", "K"), ("xU", "yV", "terms")):
        fp, loads, live = 0.0, 0.0, k["ii"] + k["jj"]
        need_s = u == "terms" or v == "terms" or (u == "N" or v == "N")
        if need_s:
            fp += k["s_fmul"]
        if u == "N" or v == "N":
            fp += k["nnz"]           # N_ij += C g_o
            live += k["ij"]
        fp += {"terms": k["nnz"], "M": k["io"], "N": k["ij"]}[u]
        if u == "M":
            loads += k["io"] * LDS_PER_VALUE
        if v == "K":
            # K_oj = sum_i C x_i: here every (o, j) has one i, so K_oj = c x_i, a product of the global pool of
            # x_i |c| (counted once over all paths, ``xc_pool``); V_j += g_o K_oj: one FFMA per (o, j)
            fp += k["oj"]
            live += k["oj"]
        else:
            fp += {"terms": k["nnz"], "N": k["ij"]}[v]
        fp += {"xU": k["ii"], "yV": k["jj"], "terms": 2 * k["nnz"]}[t]
        if t == "yV" and v == "terms":
            pass
        fp += k["ii"] + k["jj"]      # dx_i += w U_i, dy_j += w V_j
        rows.append((f"{u}/{v}/{t}", fp, loads, live))
    return rows


def xc_pool(paths):
    """Distinct x_i |c| products (|c| != 1) over the l1 = 1 paths: the K strategy's shared FMULs per pass."""
    return len({(i, round(abs(c), 6)) for p in paths if p["I"] > 1 for _, i, _, c in p["terms"] if not unit(c)})


def factored_today(p):
    k = path_counts(p)
    if p["I"] == 1:
        return 1 + k["s_fmul"] + 2 * k["nnz"] + 1 + 1
    return k["s_fmul"] + 2 * k["nnz"] + k["ii"] + k["ii"] + k["jj"]


def k_forms(paths, scale_free=True):
    """Distinct linear forms in x (K_oj = sum_i C x_i over i, per path, output, edge component), up to sign (and
    scale when ``scale_free``), across all paths: (distinct forms, their build cost, uses)."""
    forms = collections.OrderedDict()
    uses = 0
    for n, p in enumerate(paths):
        if p["I"] == 1:
            continue
        by = collections.defaultdict(dict)
        for o, i, j, c in p["terms"]:
            by[(o, j)][i] = c
        for (o, j), f in by.items():
            uses += 1
            items = sorted(f.items())
            lead = items[0][1]
            key = tuple((i, round(c / lead if scale_free else c / abs(lead), 6)) for i, c in items)
            forms.setdefault(key, 0)
            forms[key] += 1
    build = sum(len(key) - 1 if len(key) > 1 else 0 for key in forms)  # FFMA per extra term (first is x_i itself
    # when its coefficient is the lead: scale moved to the use)
    return len(forms), build, uses, forms


def cse_build(forms):
    """Greedy multi-output CSE over the K forms' sub-sums: repeatedly extract the most frequent pair of
    (x_a, x_b, ratio) and count the FFMA saved."""
    exprs = [list(key) for key in forms]
    saved = 0
    while True:
        pairs = collections.Counter()
        for e in exprs:
            for (a, ca), (b, cb) in itertools.combinations(e, 2):
                pairs[(a, b, round(cb / ca, 6))] += 1
        if not pairs:
            break
        (a, b, r), count = pairs.most_common(1)[0]
        if count < 2:
            break
        new = ("t", a, b, r)
        for e in exprs:
            ia = next((n for n, (x, _) in enumerate(e) if x == a), None)
            ib = next((n for n, (x, _) in enumerate(e) if x == b), None)
            if ia is not None and ib is not None and abs(e[ib][1] / e[ia][1] - r) < 1e-6:
                ca = e[ia][1]
                e[:] = [t for n, t in enumerate(e) if n not in (ia, ib)] + [(new, ca)]
        saved += count - 1
    return saved


def h_form(paths, k_shared=True):
    """The cross-path H form: gh_o = w_p g_o; dx_i = sum_o A_oi gh_o; dy_j = sum C x_i gh_o (terms or shared K
    forms); T_p = sum_o g_o B_o, B_o = sum_i x_i A_oi; R += dw_p T_p."""
    fp, loads = 0.0, 0.0
    outputs = sum(p["O"] for p in paths)
    # gh: one FMUL per output; a per-output constant ratio (all terms of an output with one |c|) folds in too
    fp += outputs
    s_extra = 0
    for p in paths:
        by_o = collections.defaultdict(set)
        for o, _, _, c in p["terms"]:
            by_o[o].add(round(abs(c), 9))
        s_extra += sum(len(m) - 1 for m in by_o.values())  # outputs with several |c|: one FMUL per extra one
    fp += s_extra
    io = sum(path_counts(p)["io"] for p in paths)
    fp += io                       # dx_i = sum A_oi gh_o
    fp += io                       # B_o = sum_i x_i A_oi
    fp += outputs                  # T_p = sum_o g_o B_o
    fp += len(paths)               # R += dw_p T_p
    loads += io * LDS_PER_VALUE    # A staged per edge
    edge_arith = sum(path_counts(p)["nnz"] for p in paths) * EDGE_ARITH_PER_PASS  # A = sum_j C y_j
    # dy: l1 = 0 paths: dy_j += x0 gh_o c (per term: FFMA with s = x0 gh_o? x0 is one value: wx-like: 1 FMUL per
    # output, then FFMA per term); l1 = 1 paths: terms (FFMA per term) or K forms
    dy_terms = 0
    for p in paths:
        k = path_counts(p)
        dy_terms += (k["oo"] + k["nnz"]) if p["I"] == 1 else k["nnz"]
    n_forms, build, uses, forms = k_forms(paths)
    dy_k = sum((path_counts(p)["oo"] + path_counts(p)["nnz"]) for p in paths if p["I"] == 1) + build - cse_build(forms) + uses
    fp_terms, fp_k = fp + dy_terms, fp + dy_k
    return dict(fp_terms=fp_terms, fp_k=fp_k, loads=loads, edge_arith=edge_arith, forms=n_forms, form_uses=uses,
                form_build=build, outputs=outputs, io=io)


def complex_terms(paths):
    """Term counts of each path's CG tensor in the complex (m) basis (e3nn's change of basis); None without e3nn."""
    try:
        import torch
        from e3nn import o3
    except Exception:
        return None
    from e3nn.o3._wigner import _so3_clebsch_gordan  # noqa: F401  (complex-basis CG, if exposed)
    return None


def main():
    paths = load()
    today = [factored_today(p) for p in paths]
    fixed = sum(fixed_per_path(p) for p in paths)
    best_per_path = []
    for p in paths:
        rows = per_path_strategies(p)
        best = min(rows, key=lambda r: (r[1] + r[2], r[3]))
        best_per_path.append(dict(I=p["I"], J=p["J"], O=p["O"], counts=path_counts(p), today=factored_today(p),
                                  best=best[0], best_slots=best[1] + best[2], best_live=best[3],
                                  options=sorted(((r[0], r[1] + r[2]) for r in rows), key=lambda r: r[1])[:4]))
    tp_today = sum(today)
    tp_best_mix = sum(b["best_slots"] for b in best_per_path)
    # the per-path best may pick K (V via the shared x|c| pool): add the pool once when any path does
    pool = xc_pool(paths)
    if any("/K/" in b["best"] for b in best_per_path):
        tp_best_mix += pool
    # without K anywhere: the best per path among non-K strategies
    no_k = 0.0
    for p in paths:
        rows = [r for r in per_path_strategies(p) if "/K/" not in r[0]]
        no_k += min(r[1] + r[2] for r in rows)
    h = h_form(paths)
    model_today = tp_today + fixed
    report = dict(
        model_fp_per_pass_today=model_today, census_fp_per_pass=CENSUS,
        tp_today=tp_today, fixed=fixed, tp_best_per_path_mix=tp_best_mix, xc_pool=pool, tp_best_without_k=no_k,
        h_form=h,
        per_path=best_per_path)
    best_total = min(tp_best_mix, no_k, h["fp_terms"] + h["loads"] + h["edge_arith"], h["fp_k"] + h["loads"] + h["edge_arith"])
    saved_per_pass = tp_today - best_total
    per_edge = saved_per_pass * CENSUS["passes"]
    us = H200["wprime_us"] * per_edge / CENSUS["per_edge_base"]
    report["best"] = dict(tp_slots=best_total, saved_per_pass=saved_per_pass, saved_per_edge=per_edge,
                          bound_us=round(us, 1), bound_step_pct=round(100 * us / H200["step_us"], 2))
    (HERE / "formulation_search2.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_path"}, indent=1))
    for b in best_per_path:
        print(b["I"], b["J"], b["O"], b["counts"], "today", b["today"], "best", b["best"], b["best_slots"], b["options"])


if __name__ == "__main__":
    main()
