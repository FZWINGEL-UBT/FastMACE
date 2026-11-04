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

# Top atom-level GEMMs per NbCl5 force call from gemms.csv:
# Shape, count
gemm_workload = [
    ((5832, 512, 512), 8),
    ((13608, 512, 512), 4),
    ((9720, 512, 512), 4),
    ((1944, 512, 512), 11),
]

# Preallocate tensors for each unique shape
buffers = {}
for (M, K, N), count in gemm_workload:
    if (M, K, N) not in buffers:
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)
        C1 = torch.empty((M, N), device="cuda", dtype=torch.float32)
        C2 = torch.empty((M, N), device="cuda", dtype=torch.float32)
        buffers[(M, K, N)] = (A, B, C1, C2)

# Warmup
for (shape, count) in gemm_workload:
    A, B, C1, C2 = buffers[shape]
    for _ in range(count):
        torch.mm(A, B, out=C1)
        cublaslt_mm_out(A, B, C2)
torch.cuda.synchronize()

reps = 30
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Timed cuBLAS baseline
start.record()
for _ in range(reps):
    for shape, count in gemm_workload:
        A, B, C1, _ = buffers[shape]
        for _ in range(count):
            torch.mm(A, B, out=C1)
end.record()
torch.cuda.synchronize()
t_cublas_total = start.elapsed_time(end) * 1000.0 / reps

# Timed cuBLASLt
start.record()
for _ in range(reps):
    for shape, count in gemm_workload:
        A, B, _, C2 = buffers[shape]
        for _ in range(count):
            cublaslt_mm_out(A, B, C2)
end.record()
torch.cuda.synchronize()
t_cublaslt_total = start.elapsed_time(end) * 1000.0 / reps

delta_us = t_cublas_total - t_cublaslt_total
speedup = (t_cublas_total / t_cublaslt_total - 1.0) * 100.0

print(f"=== NbCl5 Top Atom GEMMs (27 products / force call) ===")
print(f"cuBLAS total time:   {t_cublas_total:.2f} µs")
print(f"cuBLASLt total time: {t_cublaslt_total:.2f} µs")
print(f"Net time saved:      {delta_us:+.2f} µs per force call")
print(f"GEMM speedup:        {speedup:+.1f} %")
print(f"Full-step impact:    ~{delta_us / 27430.0 * 100.0:.2f} % of 27.4 ms step")
