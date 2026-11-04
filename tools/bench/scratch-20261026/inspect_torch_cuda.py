import torch

print("torch.backends.cuda.matmul.allow_tf32:", torch.backends.cuda.matmul.allow_tf32)
print("torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction:", torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
print("torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:", torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
for k in dir(torch.backends.cuda):
    if not k.startswith("_"):
        v = getattr(torch.backends.cuda, k)
        if not callable(v):
            print(f"torch.backends.cuda.{k} = {v!r}")
        elif hasattr(v, "__call__") and not isinstance(v, type):
            try:
                print(f"torch.backends.cuda.{k}() = {v()!r}")
            except Exception:
                pass
