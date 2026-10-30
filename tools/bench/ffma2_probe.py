"""Does sm_120 have a packed FP32 FMA (FFMA2)? (plans/capacity.md, the conductor's item A.)

NVRTC (CUDA 13.0) compiles an ``fma.rn.f32x2`` inline-PTX kernel and a two-``fmaf`` kernel for sm_120, sm_120a and
sm_100a; nvdisasm (Triton's copy) prints the SASS's FMA lines. CPU only: no GPU is used.
"""

import os
import subprocess
import tempfile

from cuda.bindings import nvrtc

NVDISASM = os.path.expanduser("~/environments/mace_cueq_oeq/lib/python3.10/site-packages/triton/backends/nvidia/bin/"
                              "nvdisasm")
PACKED = r'''
extern "C" __global__ void packed(unsigned long long* out, const unsigned long long* a,
                                  const unsigned long long* b, const unsigned long long* c) {
  unsigned long long d;
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;"
               : "=l"(d) : "l"(a[threadIdx.x]), "l"(b[threadIdx.x]), "l"(c[threadIdx.x]));
  out[threadIdx.x] = d;
}
'''
SCALAR = r'''
extern "C" __global__ void scalar(float2* out, const float2* a, const float2* b, const float2* c) {
  float2 x = a[threadIdx.x], y = b[threadIdx.x], z = c[threadIdx.x];
  out[threadIdx.x] = make_float2(fmaf(x.x, y.x, z.x), fmaf(x.y, y.y, z.y));
}
'''


def compile_cubin(source, arch):
    """(cubin bytes or None, the log's non-empty lines)."""
    program = nvrtc.nvrtcCreateProgram(source.encode(), b"t.cu", 0, [], [])[1]
    options = [f"--gpu-architecture={arch}".encode()]
    result = nvrtc.nvrtcCompileProgram(program, len(options), options)
    size = nvrtc.nvrtcGetProgramLogSize(program)[1]
    log = b" " * size
    nvrtc.nvrtcGetProgramLog(program, log)
    lines = [line for line in log.decode(errors="replace").splitlines() if line.strip()]
    if int(result[0]) != 0:
        return None, lines
    size = nvrtc.nvrtcGetCUBINSize(program)[1]
    cubin = b" " * size
    nvrtc.nvrtcGetCUBIN(program, cubin)
    return cubin, lines


def main():
    for arch in ("sm_120", "sm_120a", "sm_100a"):
        for name, source in (("packed", PACKED), ("scalar", SCALAR)):
            cubin, lines = compile_cubin(source, arch)
            if cubin is None:
                print(arch, name, "REJECTED:", lines[:3])
                continue
            with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as handle:
                handle.write(cubin)
            sass = subprocess.run([NVDISASM, handle.name], capture_output=True, text=True).stdout
            os.unlink(handle.name)
            print(arch, name, "SASS:", [line.strip() for line in sass.splitlines() if "FMA" in line][:4])


if __name__ == "__main__":
    main()
