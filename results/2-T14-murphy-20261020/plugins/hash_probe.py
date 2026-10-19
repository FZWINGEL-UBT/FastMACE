"""2-T14: hash of the float32 (and float64) cuEq models of a tiny MH-1, in production order (pristine load first)."""
import hashlib, os, sys, tempfile, warnings
import fastmace  # noqa: F401
import torch
sys.path[:0] = ["tests"]
warnings.simplefilter("ignore")
path = sys.argv[1]
from fastmace.model import loading
if not os.path.exists(path):
    from _support.tiny_models import FAMILIES, tiny_mace_model
    torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
    print("saved; rerun"); sys.exit()
loading.load_pristine(path, trusted=True)
for dtype in ("float32", "float64", "float32"):
    model = loading.load(path, backend="cueq", dtype=dtype, device="cpu", trusted=True, allow_naive_cueq=True).calculator.models[0]
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    print(dtype, digest.hexdigest()[:16])
