import json,glob,sys
for f in sorted(glob.glob(sys.argv[1]+"/*/*.json")):
    d=json.load(open(f)); n=d["atoms"]
    print("==",f.split("/")[-2], n, d["phase"], d.get("error","")[:300])
    for c in d.get("calls",[]):
        w=c.get("wavefront",{})
        print("  call %.3f s %s tiles=%s R=%s halo=%s %.2f us/atom peak=%.2fGiB host=%.1fGiB wave=%s"%(c["seconds"],c["mode"],c["tiles"],c["receivers"],c.get("max_halo"),c["seconds"]/n*1e6,c.get("total_peak",0)/2**30,c.get("host_peak",0)/2**30,w.get("on")))
    st=d.get("sweep_times")
    if st: print("  sweeps", {k:round(v,3) for k,v in st["seconds"].items()}, "wall", round(st["wall_s"],3), "timed", round(st["timed_s"],3))
    sp=d.get("sweep_profile")
    if sp:
        for s,v in sp.items():
            print("  profile sweep",s, str(v)[:1500])
    if "tiled" in d: print("  E/atom", d["tiled"]["energy_per_atom_eV"], "fmax", d["tiled"]["max_abs_force"])
