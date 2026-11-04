import torch

print("preferred_blas_library:", torch.backends.cuda.preferred_blas_library())
try:
    from torch.backends.cuda import _BlasBackend
    print("BlasBackend members:", list(_BlasBackend))
except Exception as e:
    print("Error getting members:", e)

try:
    torch.backends.cuda.preferred_blas_library("cublaslt")
    print("Set to cublaslt successfully!")
    print("Now preferred_blas_library:", torch.backends.cuda.preferred_blas_library())
except Exception as e:
    print("Setting to cublaslt error:", e)
