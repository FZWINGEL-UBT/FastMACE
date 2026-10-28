import torch, numpy as np
m = torch.load("/home/"+__import__("os").environ["USER"]+"/models/mace-mh-1.model", map_location="cpu", weights_only=False)
rows=[]
for n,p in m.named_parameters():
    if p.ndim==1 and p.numel()>=4096:
        k=int(round(p.numel()**0.5))
        # e3nn linear weights are flat per instruction; try square blocks of 512*512 / 128*128
        for d in (512,256,128):
            if p.numel()%(d*d)==0:
                W=p.detach().double().reshape(-1,d,d); break
        else: continue
    elif p.ndim==2 and min(p.shape)>=16: W=p.detach().double()[None]
    elif p.ndim==3 and min(p.shape[1:])>=16: W=p.detach().double()
    else: continue
    for i,w in enumerate(W):
        s=torch.linalg.svdvals(w); s=s/s[0]
        r7=int((s>1e-7).sum()); r4=int((s>1e-4).sum()); r2=int((s>1e-2).sum())
        rows.append((n,i,tuple(w.shape),r2,r4,r7,float(s[-1])))
for r in rows: print(*r)
