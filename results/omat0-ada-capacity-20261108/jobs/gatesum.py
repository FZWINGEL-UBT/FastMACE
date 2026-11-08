"""One line per structure from a force_stress_gate.py log: energy max abs (eV), forces and stress ratios, passed."""
import json
import re
import sys

for path in sys.argv[1:]:
    for line in open(path, errors="replace"):
        m = re.match(r"(\S+) summary (\{.*\})\s*$", line)
        if not m:
            continue
        r = json.loads(m.group(2))
        parts = [m.group(1), f"energy {r['energy']['production_max_abs']:.1e}"]
        for key in ("forces", "stress"):
            if key in r:
                parts.append(f"{key} {r[key].get('ratio', float('nan')):.2f}")
        parts.append("pass" if all(v.get("passed", True) for v in r.values() if isinstance(v, dict)) else "FAIL")
        print(" ".join(parts))
    for line in open(path, errors="replace"):
        if re.search(r"verdict|revised|rule 13", line, re.I):
            print("  " + line.strip()[:200])
