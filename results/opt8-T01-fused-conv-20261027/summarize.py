import json
import sys

for line in sys.stdin:
    try:
        d = json.loads(line)
    except Exception:
        if "exactness" not in line and "falling back" not in line and line.strip():
            print(line.rstrip())
        continue
    ms = d["ms"]
    fc = d.get("fused_config", {})
    print(d["structure"], "chain f/b %.3f/%.3f" % (ms["chain_fwd"], ms["chain_bwd"]),
          "fused f/b %.3f/%.3f" % (ms.get("fused_fwd", 0), ms.get("fused_bwd", 0)),
          "ratio", d.get("fused_ratio"), "regs", fc.get("fwd"), fc.get("bwd"),
          "err %.1e %.1e %.1e %.1e" % (d["fused_msg"], d["fused_dx"], d["fused_dsh"], d["fused_da"]), flush=True)
