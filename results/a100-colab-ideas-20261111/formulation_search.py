"""Idea 3 (11 Nov 2026), bounded: exhaustive search over contraction orders of interaction 2's trilinear sums.

Per path p of OMAT-0 medium's interaction 2 (input 128x0e+128x1o, edge 0e+1o+2e+3o, outputs up to l = 3; 10 paths)
the CG tensor C[o, i, j] (e3nn ``wigner_3j``; its sparsity and coefficient values decide the counts) defines

  forward   m_o  = w * sum_ij C_oij x_i y_j
  backward  T    = sum_oij C_oij g_o x_i y_j       (weight gradient)
            U_i  = sum_oj  C_oij g_o y_j           (dx_i += w U_i)
            V_j  = sum_oi  C_oij g_o x_i           (dy_j += w V_j, summed over channels)

with g (dout), x (sender) and w channel-dependent and y (harmonics) edge-only. Cost = channel-dependent FP32
instructions per edge and channel (FFMA, FMUL, FADD = 1 each; a sign is free; products of edge-only values are
amortised over 128 channels and counted / 128); ``live`` = channel-dependent values that must be held at once
(accumulators and intermediates; a register-pressure proxy). Every strategy is a contraction tree over the index
sets with shared intermediates:

  U: "terms" (U_i += s_oc y_j, s_oc = c g_o shared with V's terms), "M" (M_io = sum_j C y_j per edge, U_i = sum_o
     M_io g_o), "N" (U_i = sum_j N_ij y_j with N_ij = sum_o C g_o).
  V: "terms" (v_j += s_oc x_i), "N" (N_ij = sum_o C g_o, V_j = sum_i N_ij x_i), "K" (K_oj = sum_i C x_i, V_j =
     sum_o g_o K_oj).
  T: "xU" (sum_i x_i U_i), "yV" (sum_j y_j V_j), "terms" (direct).
  forward: "terms" (acc_o += w * (sum C x_i y_j), per-term products), "ghoist" (G_oi = sum_j C y_j per edge,
     acc_o += (w x_i) G_oi), "Mx" (wx_i = w x_i, then acc_o += sum_i wx_i G_oi; = ghoist), "yfirst" (Q_oi ... same).

The unfactored form (the pre-FACTORED kernel: each of T, U, V its own term sum) is the start; FACTORED is
U=terms, V=terms, T=xU. The first test of the idea: the search must rank FACTORED at or near the top from the
unfactored start. CPU only. Writes formulation_search.json beside this file.
"""
import itertools
import json
from pathlib import Path

import torch

torch.serialization.add_safe_globals([slice])  # e3nn constants.pt under torch >= 2.6
from e3nn import o3

CHANNELS = 128
IN = [(0, 1), (1, -1)]                 # (l, parity) of the sender irreps
EDGE = [(l, (-1) ** l) for l in range(4)]
OUT = [(l, (-1) ** l) for l in range(4)]


def paths():
    out = []
    for (l1, p1), (l2, p2), (l3, p3) in itertools.product(IN, EDGE, OUT):
        if p1 * p2 == p3 and abs(l1 - l2) <= l3 <= l1 + l2:
            out.append((l1, l2, l3))
    return out


def cg(l1, l2, l3):
    w = o3.wigner_3j(l1, l2, l3, dtype=torch.float64)  # [i, j, o]
    terms = []
    for i, j, o in zip(*torch.nonzero(w.abs() > 1e-12, as_tuple=True)):
        terms.append((int(o), int(i), int(j), float(w[i, j, o])))
    return terms


def mag_key(c):
    return round(abs(c), 12)


def costs(terms, n_i, n_j, n_o):
    nnz = len(terms)
    oc = {(o, mag_key(c)) for o, _, _, c in terms}
    s_fmul = sum(1 for _, m in oc if abs(m - 1.0) > 1e-12)  # s_oc = c g_o (|c| = 1 is free)
    ij = {(i, j) for _, i, j, _ in terms}
    io = {(i, o) for o, i, _, _ in terms}
    oj = {(o, j) for o, _, j, _ in terms}
    ii = {i for _, i, _, _ in terms}
    jj = {j for _, _, j, _ in terms}
    edge = 1.0 / CHANNELS
    rows = []
    for u, v, t in itertools.product(("terms", "M", "N"), ("terms", "N", "K"), ("xU", "yV", "terms")):
        cost, live, inter = 0.0, len(ii) + len(jj), set()
        if u == "terms" or v == "terms":
            inter.add("s")
        if u == "N" or v == "N":
            inter.add("N")
        if "s" in inter:
            cost += s_fmul
        if "N" in inter:
            cost += nnz  # N_ij += C g_o (FFMA with an immediate)
            live += len(ij)
        cost += {"terms": nnz, "M": len(io) + nnz * edge, "N": len(ij)}[u]
        if v == "K":
            cost += nnz + len(oj)  # K_oj = sum_i C x_i, then V_j = sum_o g_o K_oj
            live += len(oj)
        else:
            cost += {"terms": nnz, "N": len(ij)}[v]
        cost += {"xU": len(ii), "yV": len(jj), "terms": 2 * nnz}[t]
        rows.append(dict(U=u, V=v, T=t, cost=cost, live=live))
    # the unfactored start: T, U, V each its own term sum (T: c g x y = FMUL + FMUL + FFMA; U, V: FMUL + FFMA)
    unfactored = dict(U="sep", V="sep", T="sep", cost=3 * nnz + 2 * nnz + 2 * nnz, live=len(ii) + len(jj))
    # forward per channel: terms (x_i y_j c products, then FFMA with w per output) against ghoist
    fwd_terms = 2 * nnz + n_o  # x*y*c (FMUL with folded c y? count FMUL + FFMA per term) + w FFMA per output
    fwd_ghoist = len(ii) + len(io) + nnz * edge  # w x_i, then FFMA per (o, i); G_oi per edge
    return dict(nnz=nnz, nnz_ij=len(ij), nnz_io=len(io), nnz_oj=len(oj), s_fmul=s_fmul, rows=rows,
                unfactored=unfactored, forward=dict(terms=fwd_terms, ghoist=fwd_ghoist))


def main():
    result, per_path = {}, []
    totals = {}
    for l1, l2, l3 in paths():
        terms = cg(l1, l2, l3)
        c = costs(terms, 2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)
        c["path"] = (l1, l2, l3)
        per_path.append(c)
        for r in c["rows"]:
            key = (r["U"], r["V"], r["T"])
            totals.setdefault(key, [0.0, 0])
            totals[key][0] += r["cost"]
            totals[key][1] = max(totals[key][1], r["live"])
    # best per path (a per-path choice, as the generator could make)
    best_mix = sum(min(r["cost"] for r in c["rows"]) for c in per_path)
    unf = sum(c["unfactored"]["cost"] for c in per_path)
    fac = totals[("terms", "terms", "xU")][0]
    ranking = sorted(((v[0], v[1], k) for k, v in totals.items()))
    result = dict(paths=[c["path"] for c in per_path], cg_terms=sum(c["nnz"] for c in per_path),
                  unfactored=unf, factored=fac, best_uniform=dict(cost=ranking[0][0], max_live=ranking[0][1],
                                                                   strategy=ranking[0][2]),
                  best_per_path=best_mix, factored_rank=[r[2] for r in ranking].index(("terms", "terms", "xU")) + 1,
                  ranking=[dict(strategy="/".join(r[2]), cost=r[0], max_live=r[1]) for r in ranking[:12]],
                  per_path=[dict(path=c["path"], nnz=c["nnz"], nnz_ij=c["nnz_ij"], nnz_io=c["nnz_io"],
                                 nnz_oj=c["nnz_oj"], s_fmul=c["s_fmul"],
                                 factored=next(r["cost"] for r in c["rows"]
                                               if (r["U"], r["V"], r["T"]) == ("terms", "terms", "xU")),
                                 best=min(c["rows"], key=lambda r: (r["cost"], r["live"])),
                                 forward=c["forward"]) for c in per_path],
                  forward=dict(terms=sum(c["forward"]["terms"] for c in per_path),
                               ghoist=sum(c["forward"]["ghoist"] for c in per_path)))
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_path"}, indent=1))
    for p in result["per_path"]:
        print(p["path"], "nnz", p["nnz"], "ij", p["nnz_ij"], "io", p["nnz_io"], "oj", p["nnz_oj"], "sFMUL",
              p["s_fmul"], "factored", p["factored"], "best", p["best"]["U"], p["best"]["V"], p["best"]["T"],
              round(p["best"]["cost"], 2), "live", p["best"]["live"])


if __name__ == "__main__":
    main()
