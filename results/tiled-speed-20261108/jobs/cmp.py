import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
def short(n): return n.replace("void ","")[:95]
ka={k["name"]:k for k in a["kernels"]}; kb={k["name"]:k for k in b["kernels"]}
names=sorted(set(ka)|set(kb), key=lambda n:-(kb.get(n,{}).get("ns_per_atom",0)+ka.get(n,{}).get("ns_per_atom",0)))
print("%-95s %8s %5s %8s %6s"%("kernel","A ns/at","calls","B ns/at","calls"))
ta=tb=0
for n in names[:int(sys.argv[3]) if len(sys.argv)>3 else 60]:
    x=ka.get(n,{}); y=kb.get(n,{})
    print("%-95s %8.1f %5s %8.1f %6s"%(short(n),x.get("ns_per_atom",0),x.get("calls",""),y.get("ns_per_atom",0),y.get("calls","")))
print("total kernel ns/atom", a["kernel_s"]/a["atoms"]*1e9, b["kernel_s"]/b["atoms"]*1e9)
for h in b["host"][:15]: print(" host", h)
