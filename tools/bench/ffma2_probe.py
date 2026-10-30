"""Does sm_120 have a packed FP32 FMA (FFMA2)? NVRTC (CUDA 13.0) compiles fma.rn.f32x2 inline PTX for sm_120, sm_120a
and sm_100a; nvdisasm (Triton's copy) reads the SASS. CPU only: no GPU is used (plans/capacity.md, item A)."""
import subprocess, sys, tempfile, os
from cuda.bindings import nvrtc
NVDISASM = os.path.expanduser("~/environments/mace_cueq_oeq/lib/python3.10/site-packages/triton/backends/nvidia/bin/nvdisasm")
SRC_PACKED = r'''
extern "C" __global__ void packed(unsigned long long* out, const unsigned long long* a, const unsigned long long* b, const unsigned long long* c) {
  unsigned long long d;
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a[threadIdx.x]), "l"(b[threadIdx.x]), "l"(c[threadIdx.x]));
  out[threadIdx.x] = d;
}
'''
SRC_SCALAR = r'''
extern "C" __global__ void scalar(float2* out, const float2* a, const float2* b, const float2* c) {
  float2 x = a[threadIdx.x], y = b[threadIdx.x], z = c[threadIdx.x];
  out[threadIdx.x] = make_float2(fmaf(x.x, y.x, z.x), fmaf(x.y, y.y, z.y));
}
'''
for arch in ("sm_120", "sm_120a", "sm_100a"):
    for name, src in (("packed", SRC_PACKED), ("scalar", SRC_SCALAR)):
        prog = nvrtc.nvrtcCreateProgram(src.encode(), b"t.cu", 0, [], [])[1]
        opts = [f"--gpu-architecture={arch}".encode()]
        res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        n = nvrtc.nvrtcGetProgramLogSize(prog)[1]; log = b" " * n; nvrtc.nvrtcGetProgramLog(prog, log)
        if int(res[0]) != 0:
            print(arch, name, "REJECTED:", [l for l in log.decode(errors="replace").splitlines() if l.strip()][:3]); continue
        size = nvrtc.nvrtcGetCUBINSize(prog)[1]; cubin = b" " * size; nvrtc.nvrtcGetCUBIN(prog, cubin)
        with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
            f.write(cubin); path = f.name
        sass = subprocess.run([NVDISASM, path], capture_output=True, text=True).stdout
        fma = [l.strip() for l in sass.splitlines() if "FMA" in l or "FFMA" in l]
        print(arch, name, "SASS:", fma[:4] or sass[-300:])
