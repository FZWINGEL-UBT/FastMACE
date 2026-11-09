"""CPU-only counterexample to a cheap exact residual and dense channel gauge.

No force calls, test suites, GPU access, or production edits.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import fastmace
import torch
from fastmace.rewrites.generic import modules as gm
from fastmace.rewrites import radial

torch.set_num_threads(1)
torch.serialization.add_safe_globals([slice])
checkpoint = Path("/home/fzwin/models/mace-omat-0-medium.model")
sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert sha == "d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a"
model = torch.load(checkpoint, map_location="cpu", weights_only=False)
w = gm.fc_last_weight(gm.fc_split(radial._radial_net(model.interactions[1]))[1]).detach().float()
quantized = w.half().float()
residual = w.double() - quantized.double()
s = torch.linalg.svdvals(residual)
signatures = w.reshape(10,128,64).permute(1,0,2).reshape(128,640).double()
distance = torch.cdist(signatures, signatures)
distance.fill_diagonal_(float("inf"))
result = {
    "checkpoint_sha256": sha,
    "torch": torch.__version__,
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "matrix": "interaction 2 radial final layer W",
    "shape": list(w.shape),
    "coarse": "float32(float16(W)); residual formed in float64",
    "residual_rank_relative_1e12": int((s > s[0]*1e-12).sum()),
    "relative_residual_frobenius_norm": float(residual.norm()/w.double().norm()),
    "residual_relative_smin": float(s[-1]/s[0]),
    "residual_tail_frobenius_fraction_by_rank": {
        str(k):float(s[k:].norm()/s.norm()) for k in (8,16,32,48,63)
    },
    "rank_for_99percent_residual_squarednorm": int((s.square().cumsum(0)<.99*s.square().sum()).sum())+1,
    "channel_gauge": {
        "signature_shape": list(signatures.shape),
        "meaning": "Each channel's coefficients over ten independent radial paths and 64 radial hidden basis functions",
        "distinct_signature_rows": int(torch.unique(signatures,dim=0).shape[0]),
        "minimum_distinct_row_distance_l2": float(distance.min()),
        "min_row_norm": float(signatures.norm(dim=1).min()),
        "max_row_norm": float(signatures.norm(dim=1).max()),
        "boundary": "Distinct coefficient signatures rule out a fixed dense channel gauge commuting with all coefficient-space diagonal filters. They do not prove every sampled radial function is independent. Diagonal scalings and permutations remain possible, but do not eliminate operations."
    }
}
out = Path(__file__).with_name("residual-rank.json")
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
