import ctypes
import torch

# Load library
lib = ctypes.CDLL("/home/fzwin/tmp/libcublaslt_dispatcher.so")
lib.init_cublaslt.restype = ctypes.c_int
lib.cublaslt_gemm_fp32.restype = ctypes.c_int
lib.cublaslt_gemm_fp32.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p
]

assert lib.init_cublaslt() == 0, "Failed to initialize cuBLASLt"

def cublaslt_mm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    stream = torch.cuda.current_stream().cuda_stream
    st = lib.cublaslt_gemm_fp32(
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_void_p(B.data_ptr()),
        ctypes.c_void_p(C.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
        ctypes.c_void_p(stream)
    )
    if st != 0:
        raise RuntimeError(f"cublaslt_gemm_fp32 failed with status {st}")
    return C

shapes = [
    (13608, 512, 512),
    (1512, 512, 512),
    (1944, 1024, 1024),
    (5832, 512, 512),
    (9720, 512, 512),
]

print(f"{'Shape':<22} | {'PyTorch mm':<12} | {'cuBLASLt mm':<12} | {'Speedup':<10} | {'Max Diff':<10}")
print("-" * 75)

for M, N, K in shapes:
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    # Correctness
    C_torch = torch.mm(A, B)
    C_lt = cublaslt_mm(A, B)
    torch.cuda.synchronize()
    maxdiff = (C_torch - C_lt).abs().max().item()
    
    # Benchmark PyTorch mm
    reps = 30
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    for _ in range(5): torch.mm(A, B)
    torch.cuda.synchronize()
    start.record()
    for _ in range(reps): torch.mm(A, B)
    end.record()
    torch.cuda.synchronize()
    t_torch = start.elapsed_time(end) * 1000.0 / reps
    
    # Benchmark cuBLASLt mm
    for _ in range(5): cublaslt_mm(A, B)
    torch.cuda.synchronize()
    start.record()
    for _ in range(reps): cublaslt_mm(A, B)
    end.record()
    torch.cuda.synchronize()
    t_lt = start.elapsed_time(end) * 1000.0 / reps
    
    speedup = (t_torch / t_lt - 1.0) * 100.0
    print(f"[{M}, {N}, {K}]".ljust(22) + f" | {t_torch:8.2f} µs | {t_lt:8.2f} µs | {speedup:+6.1f} % | {maxdiff:8.2e}")
