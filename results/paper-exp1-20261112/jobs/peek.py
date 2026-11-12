import json,sys
r=json.load(open(sys.argv[1]))
for k,e in r["structures"].items():
    m=e["manifest"]; print(k,e["natoms"],"setup",round(e["setup_s"]),"evals",e["evaluations"],round(e["evaluate_s"],1))
    print(" attached",m["wprime_attached"][:3] if len(str(m["wprime_attached"]))<600 else str(m["wprime_attached"])[:600])
    sf=m["switches"]["fastmace.kernels.oeq_streaming.spline_forward"]["resolved"]; print(" spline_forward resolved",sf)
    print(" enabled",m["plan"]["enabled"])
    for t in ("derivative","knots","loop","line_choice"):
        if t in e: print(" ",t,{a:b for a,b in e[t].items() if not isinstance(b,(list,dict))}, e[t].get("summary",""))
