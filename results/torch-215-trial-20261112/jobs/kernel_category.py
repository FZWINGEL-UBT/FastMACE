"""Kernel family by kernel name (torch 2.15 trial, 12 November 2026); shared by profile_point.py and analyze.py."""


def category(name):
    """Kernel family by name. Checked against the Python stacks of an annotated capture (pystacks-prof-c.json, Ada,
    12 November 2026): our hand-written Triton kernels carry bare function names ("gemm" is tf32x3_gemm/species_gemm,
    "kernel"/"Kernel" is tabled_outer_sum), so a bare lower-case identifier is ours ("triton"), never cuBLAS."""
    import re
    n = name.lower()
    if n.startswith("triton_") or "triton_" in n:
        return "inductor"
    if "oeq_" in n or "wprime" in n or "streaming" in n or "w_rebuild" in n or "kspace" in n \
            or n.startswith(("spline_forward", "sc_generated", "fused_forward", "wrebuild")):
        return "nvrtc"
    if "cutlass" in n:
        return "cutlass"
    if "cuequivariance" in n or "segmented_polynomial" in n or "tensor_product" in n:
        return "cueq"
    if "memcpy" in n or "memset" in n:
        return "memory"
    if re.fullmatch(r"_?[a-z][a-z0-9_]*", n) and not n.startswith(("ampere_", "sm80_", "sm90_", "sm89_", "gemmk1")):
        return "triton"
    if "gemm" in n or "cublas" in n or "xmma" in n or "gemv" in n or n.startswith(("sm80_", "sm90_")) \
            or "_tensorop_" in n:
        return "cublas"
    return "other"
