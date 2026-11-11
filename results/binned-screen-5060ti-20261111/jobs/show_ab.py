"""Print the A/B records of the BINNED_SCREEN run: medians, per-pair ratios, rebuild counters, validity."""
import json
import sys
from pathlib import Path


def brief(value, limit=700):
    text = json.dumps(value)
    return text if len(text) <= limit else text[:limit] + "..."


for f in sorted(Path(sys.argv[1]).glob("ab-*.json")):
    d = json.loads(f.read_text())
    print("==", f.name, "| order", d.get("order"), "| pattern", d.get("pattern"), "| protocol", brief(d.get("protocol")))
    print("   constants", brief(d.get("constant_sides")))
    structures = d.get("structures")
    items = structures.items() if isinstance(structures, dict) else enumerate(structures or [])
    for name, rec in items:
        if isinstance(rec, dict):
            for key, value in rec.items():
                print("   ", name, key, brief(value))
        else:
            print("   ", name, brief(rec))
