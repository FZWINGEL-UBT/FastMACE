import torch
import torch._inductor.config as ic

def mm_fn(a, b):
    return torch.mm(a, b)

M, N, K = 13608, 512, 512
a = torch.randn(M, K, device="cuda", dtype=torch.float32)
b = torch.randn(K, N, device="cuda", dtype=torch.float32)

print("Baseline uncompiled mm...")
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
for _ in range(10): mm_fn(a, b)
start.record()
for _ in range(50): mm_fn(a, b)
end.record()
torch.cuda.synchronize()
print(f"Uncompiled: {start.elapsed_time(end) * 20:.2f} us")

print("Compiling with mode=default...")
compiled_default = torch.compile(mm_fn, mode="default")
for _ in range(10): compiled_default(a, b)
torch.cuda.synchronize()
start.record()
for _ in range(50): compiled_default(a, b)
end.record()
torch.cuda.synchronize()
print(f"Compiled default: {start.elapsed_time(end) * 20:.2f} us")

print("Compiling with max_autotune_gemm=True...")
ic.max_autotune_gemm = True
ic.autotune_cublasLt = True
compiled_autotune = torch.compile(mm_fn, mode="default")
for _ in range(10): compiled_autotune(a, b)
torch.cuda.synchronize()
start.record()
for _ in range(50): compiled_autotune(a, b)
end.record()
torch.cuda.synchronize()
print(f"Compiled max_autotune_gemm: {start.elapsed_time(end) * 20:.2f} us")
