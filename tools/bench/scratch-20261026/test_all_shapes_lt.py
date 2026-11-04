import ctypes
import torch

lib = ctypes.CDLL("/home/fzwin/tmp/libcublaslt_dispatcher.so")
lib.init_cublaslt.restype = ctypes.c_int
lib.cublaslt_gemm_fp32.restype = ctypes.c_int
lib.cublaslt_gemm_fp32.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p
]
assert lib.init_cublaslt() == 0

def cublaslt_mm_out(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
    M, K = A.shape
    K2, N = B.shape
    stream = torch.cuda.current_stream().cuda_stream
    st = lib.cublaslt_gemm_fp32(
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_void_p(B.data_ptr()),
        ctypes.c_void_p(C.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
        ctypes.c_void_p(stream)
    )
    if st != 0: raise RuntimeError(f"failed: {st}")

shapes = [
    (13608, 512, 512),
    (1512, 512, 512),
    (1944, 1024, 1024),
    (5832, 512, 512),
    (9720, 512, 512),
]

print(f"{'Shape':<22} | {'torch.mm':<12} | {'cuBLASLt':<12} | {'Speedup':<10} | {'Max Diff':<10}")
print("-" * 75)

for M, N, K in shapes:
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C_torch = torch.empty((M, N), device="cuda", dtype=torch.float32)
    C_lt = torch.empty((M, N), device="cuda", dtype=torch.float32)
    
    # Warmup
    for _ in range(5):
        torch.mm(A, B, out=C_torch)
        cublaslt_mm_out(A, B, C_lt)
    torch.cuda.synchronize()
    
    reps = 40
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(reps):
        torch.mm(A, B, out=C_torch)
    end.record()
    torch.cuda.synchronize()
    t_torch = start.elapsed_time(end) * 1000.0 / reps
    
    start.record()
    for _ in range(reps):
        cublaslt_mm_out(A, B, C_lt)
    end.record()
    torch.cuda.synchronize()
    t_lt = start.elapsed_time(end) * 1000.0 / reps
    
    speedup = (t_torch / t_lt - 1.0) * 100.0
    maxdiff = (C_torch - C_lt).abs().max().item()
    print(f"[{M}, {N}, {K}]".ljust(22) + f" | {t_torch:8.2f} µs | {t_lt:8.2f} µs | {speedup:+6.1f} % | {maxdiff:8.2e}")
