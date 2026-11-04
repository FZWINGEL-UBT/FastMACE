import torch
import time

def bench_mm(M, N, K, reps=50):
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(reps):
        C = torch.mm(A, B)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) * 1000.0 / reps

shapes = [
    (13608, 512, 512),
    (1512, 512, 512),
    (1944, 1024, 1024),
    (5832, 512, 512),
    (9720, 512, 512),
]

print("=== Baseline: cuBLAS ===")
torch.backends.cuda.preferred_blas_library("cublas")
for M, N, K in shapes:
    t_cublas = bench_mm(M, N, K)
    print(f"[{M}, {N}, {K}]: {t_cublas:.2f} us")

print("\n=== Preferred BLAS: cuBLASLt with 32MB workspace ===")
torch.backends.cuda.preferred_blas_library("cublaslt")
torch.backends.cuda.cublaslt_workspace_size(33554432)
for M, N, K in shapes:
    t_lt = bench_mm(M, N, K)
    print(f"[{M}, {N}, {K}]: {t_lt:.2f} us")
