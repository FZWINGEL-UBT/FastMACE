"""2-T14e: attribute each test's seconds to reference work (events from the timing hook) and the rest."""
import collections, json, sys, xml.etree.ElementTree as ET

events_path, junit_path = sys.argv[1], sys.argv[2]
events = [json.loads(line) for line in open(events_path)]
tests = {}
for c in ET.parse(junit_path).getroot().iter("testcase"):
    if c.find("skipped") is None:
        tests[c.get("classname").replace(".", "/") + ".py::" + c.get("name")] = float(c.get("time"))

def test_of(e):
    t = e["test"]
    return t.rsplit(" (", 1)[0] if t else ""

selfcheck_children = {e["pid"] for e in events if e["kind"] == "process" and e.get("argv0", "").endswith("fastmace/planner/selfcheck.py")}
rows = collections.defaultdict(lambda: collections.Counter())
dups = collections.Counter()
for e in events:
    t = test_of(e)
    k = e["kind"]
    if k == "gate_references":
        rows[t]["gate_refs"] += e["seconds"]
    elif k == "process" and e.get("research_engine"):
        rows[t]["research_engine"] += e["seconds"]
    elif k in ("reference_evaluate", "reference_load") and e["pid"] not in selfcheck_children and e.get("nested", 0) == 0:
        rows[t]["reference_tier"] += e["seconds"]
    elif k == "mace_calculate" and e.get("nested", 0) == 0 and e["pid"] not in selfcheck_children \
            and not e.get("caller", "").startswith("fastmace.reference"):
        rows[t]["mace_calc"] += e["seconds"]
        if e.get("model") and e.get("positions"):
            dups[(e["model"], e["positions"], e.get("dtype"))] += 1
    elif k == "process" and e["pid"] in selfcheck_children:
        rows[t]["gate_child_process"] += e["seconds"]
total = sum(tests.values())
cols = ["gate_refs", "reference_tier", "mace_calc", "research_engine"]
sums = collections.Counter()
print(f"tests {len(tests)}, total {total/60:.1f} min")
table = []
for t, secs in tests.items():
    r = rows.get(t, {})
    ref = sum(r.get(c, 0) for c in cols)
    for c in cols:
        sums[c] += r.get(c, 0)
    table.append((ref, secs, t, {c: round(r.get(c, 0), 1) for c in cols if r.get(c)}))
for c in cols:
    print(f"{c:16s} {sums[c]:8.1f} s  {100*sums[c]/total:5.1f} %")
print(f"{'all reference':16s} {sum(sums.values()):8.1f} s  {100*sum(sums.values())/total:5.1f} %")
print("--- tests with reference work")
for ref, secs, t, detail in sorted(table, reverse=True):
    if ref > 0.5:
        print(f"{ref:7.1f} / {secs:7.1f} s  {t}  {detail}")
print("--- repeated MACE single points (model, positions, dtype) seen more than once:", sum(1 for v in dups.values() if v > 1))
json.dump({"total_s": total, "sums": sums, "tests": [{"test": t, "seconds": s, "reference_s": r, **d} for r, s, t, d in table]},
          open(events_path.replace(".events.jsonl", ".attribution.json"), "w"), indent=1)
