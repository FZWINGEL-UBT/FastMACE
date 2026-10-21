"""OPT-T08 audit: GEMM time each candidate removes, from 2-T20a's gemms.csv (usage: python estimate.py results/2-T20a-step-breakdown-20261020)."""
import csv, json, sys, os
R = sys.argv[1]
B = {("local","NaCl"):3.39,("local","NbCl5"):26.98,("murphy","NaCl"):3.48,("murphy","NbCl5"):31.30}
def load(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        dims = json.loads(r["input_dims"]); key = (r["op"], tuple(tuple(d) for d in dims if d))
        out[key] = (float(r["us_per_call"]), float(r["invocations_per_call"]), r["input_strides"])
    return out
def per_inv(g, M, K, N, op="aten::mm"):
    for (o, dims), (us, inv, st) in g.items():
        if o == op and len(dims) >= 2 and dims[0+(1 if op=="aten::addmm" else 0)][0] == M and \
           dims[0+(1 if op=="aten::addmm" else 0)][1] == K and dims[1+(1 if op=="aten::addmm" else 0)][1] == N:
            return us / inv, inv
    return None, 0
for machine, base in (("local", R), ("murphy", R + "/murphy")):
    for c, n in (("NaCl", 216), ("NbCl5", 1944)):
        p = f"{base}/{c}/gemms.csv"
        if not os.path.exists(p): continue
        g = load(p)
        total_atom = sum(us for (o, d), (us, inv, st) in g.items())
        l0, _ = per_inv(g, n, 512, 512); l1, _ = per_inv(g, 3*n, 512, 512)
        emb, emb_inv = per_inv(g, n, 89, 512); proj, proj_inv = per_inv(g, n, 512, 128, "aten::addmm")
        up, up_inv = per_inv(g, n, 512, 128)
        attr_bwd, ab_inv = per_inv(g, n, 512, 89)
        m0f, _ = per_inv(g, n, 256, 2048); m0b, _ = per_inv(g, n, 2048, 256)
        ro1f, _ = per_inv(g, n, 512, 96); ro1b, _ = per_inv(g, n, 96, 512)
        thin_f, _ = per_inv(g, n, 512, 6); thin_b, _ = per_inv(g, n, 6, 512); thin_f = thin_f or 0; thin_b = thin_b or thin_f; ro1b = ro1b or ro1f; m0b = m0b or m0f; up = up or 0; attr_bwd = attr_bwd or 0
        c6 = (attr_bwd or 0) * ab_inv
        c1 = emb*emb_inv + proj*proj_inv + 2*l0 + 2*up + 0.5*(m0f + m0b)   # int0 skip + linear_up fwd+bwd, residual half
        c2 = 4*l0 + ro1f + ro1b - 2*(thin_f + thin_b)
        c3 = 2*l0 + 2*l1
        ms = B[(machine, c)]
        print(f"{machine:6} {c:5} B={ms:6.2f} ms  C1 {c1:6.1f}us {100*c1/1000/ms:4.1f}%  C2 {c2:6.1f}us {100*c2/1000/ms:4.1f}%  "
              f"C3 {c3:6.1f}us {100*c3/1000/ms:4.1f}%  C6 {c6:5.1f}us {100*c6/1000/ms:4.1f}%  | l0 {l0:.1f} l1 {l1:.1f} up {up} emb {emb} proj {proj}")
