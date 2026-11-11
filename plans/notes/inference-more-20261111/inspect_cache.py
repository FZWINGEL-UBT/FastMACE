"""Inspect radial cache metadata on CPU; do not alter cached data."""
from pathlib import Path
import json
import torch

torch.set_num_threads(1)
root=Path.home()/'.cache/fastmace/radial-tables'
files=sorted(root.glob('*.pt'),key=lambda p:p.stat().st_mtime,reverse=True)
print('cache files',len(files),flush=True)
for p in files[:18]:
    try:
        d=torch.load(p,map_location='cpu',weights_only=True)
        def brief(v):
            if isinstance(v,torch.Tensor): return {'shape':list(v.shape),'dtype':str(v.dtype)}
            if isinstance(v,dict): return {str(k):brief(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)):
                if len(v)<20: return [brief(x) for x in v]
                return {'type':type(v).__name__,'length':len(v)}
            if isinstance(v,(int,float,str,bool)) or v is None: return v
            return type(v).__name__
        print(json.dumps({'file':str(p),'bytes':p.stat().st_size,'content':brief(d)}),flush=True)
    except Exception as e:
        print(json.dumps({'file':str(p),'error':str(e)[:180]}),flush=True)
