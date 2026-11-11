import sys, torch
sys.path.insert(0, "/mnt/d/Margraf/fm-a100-colab")
from pathlib import Path
from fastmace.kernels.oeq_streaming import half_far
src = Path("/mnt/d/Margraf/fm-a100-colab/results/a100-colab-ideas-20261111/screen1/ghoist/sources/wprime-real.cu").read_text()
st = half_far.statements(half_far.channel_body(src))
p = -1; info = {}
for s in st:
    if s[0] == "path": p += 1; info[p] = dict(wb=s[1], d=set(), x=set(), sh=set())
    elif s[0] == "let":
        for a in s[2]:
            if a[0] in ("d", "x", "sh"): info[p][a[0]].add(a[1])
for k, v in info.items(): print(k, v["wb"], "d", sorted(v["d"]), "x", sorted(v["x"]), "sh", sorted(v["sh"]))
m = torch.load("/home/fzwin/models/mace-omat-0-medium.model", map_location="cpu", weights_only=False)
tp = m.interactions[1].conv_tp
print(tp.irreps_in1, tp.irreps_in2, tp.irreps_out)
for i in tp.instructions: print(i)
print(tp.irreps_out.slices())
