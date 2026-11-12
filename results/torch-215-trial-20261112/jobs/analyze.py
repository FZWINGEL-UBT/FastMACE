"""PyTorch 2.15 trial: summarise one machine's out/ directory (job.sh) into summary-<machine>.json and print tables.

  python analyze.py OUT_DIR MACHINE DEST_DIR

- A/B: per structure the eight processes (a b a b b a b a), ms/step per process, b/a for the ABAB half (mean of b over
  mean of a) and the BABA half, and overall; warm build seconds (median per side); peak allocated / reserved MiB
  in the steady phase and over the warm-up (captures).
- Cold compile: build seconds of cold-a / cold-b (emptied caches) against the warm median.
- Kernels: prof-a against prof-b, us per MD step by category and the largest per-kernel changes (names matched
  exactly; unmatched kernels listed per side).
- Compat: status of every check per environment.
"""
import json
import statistics
import sys
from pathlib import Path

out, machine, dest = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
summary = {"machine": machine}


def load_point(name):
    files = sorted((out / "points" / name).glob("point-*.json"))
    if not files:
        return None
    return json.loads(files[0].read_text())


def mib(x):
    return None if x is None else round(x / 2**20, 1)


ab = {}
for st in ("SrTiO3", "NbCl5"):
    rows = []
    for i, side in enumerate("ababbaba", start=1):
        p = load_point(f"ab-{st}-{i}{side}")
        if p is None or p.get("ms_per_step") is None:
            rows.append({"i": i, "side": side, "status": None if p is None else p.get("status")})
            continue
        rows.append({"i": i, "side": side, "ms_per_step": p["ms_per_step"], "build_s": p["times"]["build_s"],
                     "natoms": p["natoms"], "torch": p.get("torch"),
                     "steady_max_allocated_MiB": mib(p["phases"]["steady"]["max_allocated"]),
                     "steady_max_reserved_MiB": mib(p["phases"]["steady"]["max_reserved"]),
                     "warmup_max_allocated_MiB": mib(p["phases"]["warmup"]["max_allocated"]),
                     "warmup_max_reserved_MiB": mib(p["phases"]["warmup"]["max_reserved"]),
                     "plan_text_hash": hash(p.get("plan", {}).get("text")) & 0xFFFFFFFF})

    def ratio(sub):
        a = [r["ms_per_step"] for r in sub if r["side"] == "a" and "ms_per_step" in r]
        b = [r["ms_per_step"] for r in sub if r["side"] == "b" and "ms_per_step" in r]
        return round(statistics.mean(b) / statistics.mean(a), 4) if a and b else None

    entry = {"rows": rows, "b_over_a_ABAB": ratio(rows[:4]), "b_over_a_BABA": ratio(rows[4:]),
             "b_over_a_all": ratio(rows)}
    for side in "ab":
        ok = [r for r in rows if r["side"] == side and "ms_per_step" in r]
        if ok:
            entry[f"{side}_ms_per_step_median"] = round(statistics.median(r["ms_per_step"] for r in ok), 3)
            entry[f"{side}_warm_build_s_median"] = round(statistics.median(r["build_s"] for r in ok), 1)
            for k in ("steady_max_allocated_MiB", "steady_max_reserved_MiB", "warmup_max_allocated_MiB",
                      "warmup_max_reserved_MiB"):
                entry[f"{side}_{k}"] = max(r[k] for r in ok)
    entry["same_plan_text"] = len({r.get("plan_text_hash") for r in rows if "plan_text_hash" in r}) == 1
    ab[st] = entry
summary["ab"] = ab

cold = {}
for side in "ab":
    p = load_point(f"cold-{side}-SrTiO3")
    if p:
        cold[side] = {"build_s": round(p["times"]["build_s"], 1), "load_s": round(p["times"]["load_s"], 1),
                      "warmup_s": round(p["times"]["warmup_s"], 1), "ms_per_step": p.get("ms_per_step")}
summary["cold_compile"] = cold

kern = {}
pa, pb = out / "profile" / "kernels-prof-a.json", out / "profile" / "kernels-prof-b.json"
if pa.exists() and pb.exists():
    A, B = json.loads(pa.read_text()), json.loads(pb.read_text())
    sys.path.insert(0, str(Path(__file__).parent))
    from kernel_category import category  # the corrected rule (bare Triton names were counted as cuBLAS before 12 Nov)
    for P in (A, B):
        by = {}
        for r in P.get("kernels", []):
            r["category"] = category(r["name"])
            by[r["category"]] = round(by.get(r["category"], 0) + r["us_per_step"], 2)
        P["info"]["us_per_step_by_category"] = by
    ka = {r["name"]: r for r in A.get("kernels", [])}
    kb = {r["name"]: r for r in B.get("kernels", [])}
    kern["a_by_category"] = A["info"].get("us_per_step_by_category")
    kern["b_by_category"] = B["info"].get("us_per_step_by_category")
    kern["a_total_us"] = A["info"].get("us_per_step_total")
    kern["b_total_us"] = B["info"].get("us_per_step_total")
    common = sorted(set(ka) & set(kb), key=lambda n: -abs(kb[n]["us_per_step"] - ka[n]["us_per_step"]))
    kern["largest_changes"] = [{"name": n[:160], "category": ka[n]["category"],
                                "a_us": round(ka[n]["us_per_step"], 2), "b_us": round(kb[n]["us_per_step"], 2),
                                "a_count": ka[n]["count_per_step"], "b_count": kb[n]["count_per_step"]}
                               for n in common[:25]]
    for side, mine, other in (("only_a", ka, kb), ("only_b", kb, ka)):
        only = sorted((n for n in mine if n not in other), key=lambda n: -mine[n]["us_per_step"])
        kern[side] = [{"name": n[:160], "category": mine[n]["category"], "us": round(mine[n]["us_per_step"], 2),
                       "count": mine[n]["count_per_step"]} for n in only[:40]]
        kern[side + "_us_total"] = round(sum(mine[n]["us_per_step"] for n in only), 2)
    cats = set(kern["a_by_category"] or {}) | set(kern["b_by_category"] or {})
    kern["category_delta_us"] = {c: round((kern["b_by_category"] or {}).get(c, 0) - (kern["a_by_category"] or {}).get(c, 0), 2)
                                 for c in sorted(cats)}
pc = out / "profile" / "kernels-prof-c.json"
if pc.exists():
    kern["pystacks"] = {k: v for k, v in json.loads(pc.read_text())["info"].items() if k != "frames"}
summary["kernels"] = kern

compat = {}
for side in "ab":
    f = out / f"compat-{side}.json"
    if f.exists():
        c = json.loads(f.read_text())["checks"]
        compat[side] = {k: {kk: vv for kk, vv in v.items() if kk in ("status", "seconds", "error", "value")}
                        for k, v in c.items()}
summary["compat"] = compat

dest.mkdir(parents=True, exist_ok=True)
(dest / f"summary-{machine}.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
for st, e in ab.items():
    print(st, {k: v for k, v in e.items() if k != "rows"})
print("cold", cold)
print("kernels", json.dumps({k: kern.get(k) for k in ("a_total_us", "b_total_us", "category_delta_us",
                                                      "only_a_us_total", "only_b_us_total")}))
print("compat", {s: {k: v["status"] for k, v in c.items()} for s, c in compat.items()})
