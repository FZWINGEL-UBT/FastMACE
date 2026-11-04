import torch._inductor.config as ic

d = ic.to_dict() if hasattr(ic, "to_dict") else vars(ic)
for k, val in sorted(d.items()):
    if any(x in k.lower() for x in ("cublas", "gemm", "matmul", "autotune", "epilogue", "coordinate", "tile")):
        if not callable(val):
            print(f"{k} = {val!r}")
