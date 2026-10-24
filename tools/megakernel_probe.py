"""Single-SM co-execution probe: CUDA-core FP32 FFMA beside tensor-core MMA (plan opt-4, OPT4-T07).

The gate of docs/design/2026-10-23-heterogeneous-megakernel.md ("The gate before any work"). One kernel, one
block per SM; warps ``0 .. S-1`` run an FP32 FFMA workload (the SIMT side), warps ``S .. S+M-1`` run
independent ``mma.sync`` work (TF32 m16n8k8 or BF16 m16n8k16, FP32 accumulate; the MMA side). Warp ``w`` sits
on sub-partition ``w % 4``, so with ``S`` and ``M`` multiples of 4 every scheduler holds both kinds. Each
launch runs one of three modes with the same kernel (same registers): SIMT warps only, MMA warps only, both.

Timing is inside the kernel. All warps of a block start from one ``clock64`` value and loop in chunks until a
common deadline (``--budget`` cycles), so in the concurrent mode the two sides overlap for the whole window.
Each warp records its chunks and its end clock (and ``%globaltimer``); a side's rate is the sum of its warps'
FLOPs per cycle. Per-SM rates are the median over blocks, and over ``--repeats`` launches.

Variants, from a pure-issue test towards a real GEMM's registers and shared-memory traffic:

- ``reg``: operands in registers. SIMT: 16 independent FFMA chains per thread. MMA: 8 independent tiles.
- ``smem``: SIMT is an SGEMM inner loop (8x8 thread tile, 64 accumulators, four ``ld.shared.v4`` per 64 FFMA);
  MMA is a 32x32 warp tile (fragments by ``ldmatrix.x4``, 8 MMAs per k-step, 32 accumulators).
- ``gemm``: as ``smem``, with a 64x64 MMA warp tile (32 MMAs per k-step, 128 accumulators), the register use of a
  CUTLASS-class tensor-core GEMM.

Pass rule (design note): the SIMT side keeps >= 90 % of its standalone rate, and the MMA side adds enough
useful throughput to pay for an FP32-accurate decomposition. The useful FP32 rate of the MMA side is its rate
divided by the products the decomposition needs (``c``: 3 for 3xTF32, 6 for BF16x6; BF16x3 is shown as well),
with the split's own SIMT cost left out, so the gain is an upper bound. The row's verdict is ``pass`` when

    keep = simt_both / simt_alone >= 0.90  and
    gain = (simt_both + mma_both / c) / max(simt_best, mma_best / c) >= 1.10,

the second term comparing against the better of the two units used alone (all FP32 work on CUDA cores, or all
of it as the decomposition on tensor cores) with the 10 % gain the design note asks of each later stage. The
``*_best`` rates are the higher of the side's own warps alone and all ``S + M`` warps of the block on that side:
the alternative to giving warps to the tensor cores is giving them to the CUDA cores, and a SIMT side of one
warp per scheduler is latency-bound, so its idle issue slots would otherwise count as co-execution gain.

Speed only, untimed in the A/B sense; use an idle GPU. Needs torch with CUDA and ``cuda-bindings`` (NVRTC and
the driver API; no nvcc). Output: ``<output>/summary.json`` and a table on stdout.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

VARIANTS = ("reg", "smem", "gemm")
KINDS = ("tf32", "bf16")
CONFIGS = ("4x4", "8x4", "4x8", "8x8")  # S x M warps per block
# FP32-equivalent products per MMA product for an FP32-accurate decomposition.
DECOMPOSITIONS = {"tf32": {"3xTF32": 3}, "bf16": {"BF16x6": 6, "BF16x3": 3}}
PRIMARY = {"tf32": "3xTF32", "bf16": "BF16x6"}
KEEP_MIN = 0.90
GAIN_MIN = 1.10
REC = 6  # per-warp record: role, chunks, start, end (clock64), gstart, gend (globaltimer)

SOURCE = r"""
#define KS 8          // k-steps held in shared memory per side
#define REG_CHAINS 16
#define REG_TILES 8

__device__ __forceinline__ void mma_op(int kind, float* d, const unsigned* a, const unsigned* b) {
  if (kind == 0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
}

__device__ __forceinline__ void ldsm_x4(unsigned* r, unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ float4 lds4(unsigned addr) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(addr));
  return v;
}

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

// VAR: 0 reg, 1 smem (32x32 MMA warp tile), 2 gemm (64x64 MMA warp tile). KIND: 0 tf32, 1 bf16.
template <int VAR, int KIND>
__device__ void body(int s_warps, long long budget, unsigned fill, const float* __restrict__ in,
                     float* __restrict__ out, long long* __restrict__ rec, int* __restrict__ smid) {
  constexpr int MT = VAR == 2 ? 4 : 2;  // m16 tiles per warp tile
  constexpr int NT = VAR == 2 ? 8 : 4;  // n8 tiles per warp tile
  __shared__ __align__(16) float sA[KS * 32];        // SIMT A: 8 float4 per k-step
  __shared__ __align__(16) float sB[KS * 64];        // SIMT B: 16 float4 per k-step
  __shared__ __align__(16) unsigned mA[KS * 4 * 128];  // MMA A: KS x up to 4 fragments of 512 B
  __shared__ __align__(16) unsigned mB[KS * 4 * 128];  // MMA B: KS x up to 4 x4-loads of 512 B
  __shared__ long long t0s;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid; i < KS * 32; i += blockDim.x) sA[i] = in[i & 63];
  for (int i = tid; i < KS * 64; i += blockDim.x) sB[i] = in[(i + 7) & 63];
  for (int i = tid; i < KS * 4 * 128; i += blockDim.x) { mA[i] = fill; mB[i] = fill; }
  __syncthreads();
  if (tid == 0) {
    unsigned s; asm volatile("mov.u32 %0, %smid;" : "=r"(s)); smid[blockIdx.x] = (int)s;
    t0s = clock64();
  }
  __syncthreads();
  const long long t0 = t0s, deadline = t0 + budget;
  const unsigned long long g0 = gtimer();
  long long chunks = 0;
  float sum = 0.f;
  const bool simt = warp < s_warps;
  if (simt) {
    if (VAR == 0) {
      float acc[REG_CHAINS];
      const float b = in[64], c = in[65];
#pragma unroll
      for (int i = 0; i < REG_CHAINS; ++i) acc[i] = in[(lane + i) & 63];
      do {
#pragma unroll
        for (int j = 0; j < 32; ++j)
#pragma unroll
          for (int i = 0; i < REG_CHAINS; ++i) acc[i] = fmaf(acc[i], b, c);
        ++chunks;
      } while (clock64() < deadline);
#pragma unroll
      for (int i = 0; i < REG_CHAINS; ++i) sum += acc[i];
    } else {
      float acc[8][8];
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;
      const unsigned a_base = (unsigned)__cvta_generic_to_shared(sA) + 16u * (lane >> 3);
      const unsigned b_base = (unsigned)__cvta_generic_to_shared(sB) + 16u * (lane & 7);
      do {
#pragma unroll
        for (int k = 0; k < KS; ++k) {
          const float4 a0 = lds4(a_base + k * 128), a1 = lds4(a_base + k * 128 + 64);
          const float4 b0 = lds4(b_base + k * 256), b1 = lds4(b_base + k * 256 + 128);
          const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
          const float bb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
          for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(a[i], bb[j], acc[i][j]);
        }
        ++chunks;
      } while (clock64() < deadline);
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) sum += acc[i][j];
    }
  } else {
    if (VAR == 0) {
      float d[REG_TILES][4];
      unsigned a[4], b[2];
#pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = fill ^ (unsigned)(lane & 1);
      b[0] = fill; b[1] = fill ^ 2u;
#pragma unroll
      for (int t = 0; t < REG_TILES; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) d[t][i] = 0.f;
      do {
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int t = 0; t < REG_TILES; ++t) mma_op(KIND, d[t], a, b);
        ++chunks;
      } while (clock64() < deadline);
#pragma unroll
      for (int t = 0; t < REG_TILES; ++t) sum += d[t][0] + d[t][1] + d[t][2] + d[t][3];
    } else {
      float d[MT][NT][4];
#pragma unroll
      for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i) d[m][n][i] = 0.f;
      // Each lane addresses one 16-byte row; one x4 load reads 512 contiguous bytes (conflict-free).
      const unsigned a_base = (unsigned)__cvta_generic_to_shared(mA) + 16u * lane;
      const unsigned b_base = (unsigned)__cvta_generic_to_shared(mB) + 16u * lane;
      do {
#pragma unroll
        for (int k = 0; k < KS; ++k) {
          unsigned fa[MT][4], fb[NT / 2][4];
#pragma unroll
          for (int m = 0; m < MT; ++m) ldsm_x4(fa[m], a_base + (k * 4 + m) * 512);
#pragma unroll
          for (int n = 0; n < NT / 2; ++n) ldsm_x4(fb[n], b_base + (k * 4 + n) * 512);
#pragma unroll
          for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int n = 0; n < NT; ++n) mma_op(KIND, d[m][n], fa[m], &fb[n >> 1][(n & 1) * 2]);
        }
        ++chunks;
      } while (clock64() < deadline);
#pragma unroll
      for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < NT; ++n) sum += d[m][n][0] + d[m][n][1] + d[m][n][2] + d[m][n][3];
    }
  }
  const long long t1 = clock64();
  const unsigned long long g1 = gtimer();
  out[blockIdx.x * blockDim.x + tid] = sum;
  if (lane == 0) {
    long long* r = rec + ((long long)blockIdx.x * (blockDim.x >> 5) + warp) * 6;
    r[0] = simt ? 0 : 1; r[1] = chunks; r[2] = t0; r[3] = t1; r[4] = (long long)g0; r[5] = (long long)g1;
  }
}

// The gemm variant's 128 MMA accumulators need up to 255 registers, hence at most 256 threads.
#define PROBE(NAME, VAR, KIND) \
extern "C" __global__ void __launch_bounds__(VAR == 2 ? 256 : 512) NAME(int s, long long budget, unsigned fill, const float* in, \
    float* out, long long* rec, int* smid) { body<VAR, KIND>(s, budget, fill, in, out, rec, smid); }
PROBE(probe_reg_tf32, 0, 0)
PROBE(probe_reg_bf16, 0, 1)
PROBE(probe_smem_tf32, 1, 0)
PROBE(probe_smem_bf16, 1, 1)
PROBE(probe_gemm_tf32, 2, 0)
PROBE(probe_gemm_bf16, 2, 1)
"""


def ops_per_chunk(variant, kind, role):
    """FLOPs one warp does per chunk (SIMT: 32 lanes x FMAs x 2; MMA: products x 2*m*n*k)."""
    mma_flops = 2 * 16 * 8 * (8 if kind == "tf32" else 16)
    if role == 0:
        return 32 * 2 * (32 * 16 if variant == "reg" else 8 * 64)
    if variant == "reg":
        return 8 * 8 * mma_flops
    mt, nt = (4, 8) if variant == "gemm" else (2, 4)
    return 8 * mt * nt * mma_flops


def check(result, what):
    err = result[0]
    if int(err) != 0:
        raise RuntimeError(f"{what}: {err!r}")
    return result[1:] if len(result) > 2 else (result[1] if len(result) == 2 else None)


def compile_module(capability):
    from cuda.bindings import driver, nvrtc

    prog = check(nvrtc.nvrtcCreateProgram(SOURCE.encode(), b"megakernel_probe.cu", 0, [], []), "nvrtcCreateProgram")
    opts = [f"-arch=sm_{capability[0]}{capability[1]}".encode(), b"-std=c++17"]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    log_size = check(nvrtc.nvrtcGetProgramLogSize(prog), "log size")
    log = b" " * log_size
    nvrtc.nvrtcGetProgramLog(prog, log)
    if int(err) != 0:
        raise RuntimeError(f"NVRTC failed: {log.decode(errors='replace')}")
    size = check(nvrtc.nvrtcGetCUBINSize(prog), "cubin size")
    cubin = b" " * size
    check(nvrtc.nvrtcGetCUBIN(prog, cubin), "cubin")
    module = check(driver.cuModuleLoadData(cubin), "cuModuleLoadData")
    kernels = {}
    for variant in VARIANTS:
        for kind in KINDS:
            name = f"probe_{variant}_{kind}"
            fn = check(driver.cuModuleGetFunction(module, name.encode()), name)
            attr = driver.CUfunction_attribute
            regs = check(driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_NUM_REGS, fn), "regs")
            maxt = check(driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn), "maxt")
            smem = check(driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn), "smem")
            spill = check(driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn), "local")
            kernels[(variant, kind)] = {"fn": fn, "regs": int(regs), "max_threads": int(maxt),
                                        "smem_bytes": int(smem), "local_bytes": int(spill)}
    return module, kernels, log.decode(errors="replace").strip()


def launch(torch, kernel, grid, s_warps, m_warps, budget, fill, buffers):
    from cuda.bindings import driver

    warps = s_warps + m_warps
    inp, out, rec, smid = buffers
    rec.zero_()
    args = [ctypes.c_int(s_warps), ctypes.c_longlong(budget), ctypes.c_uint(fill),
            ctypes.c_void_p(inp.data_ptr()), ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(rec.data_ptr()),
            ctypes.c_void_p(smid.data_ptr())]
    argv = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    check(driver.cuLaunchKernel(kernel["fn"], grid, 1, 1, 32 * warps, 1, 1, 0, stream, ctypes.addressof(argv), 0),
          "cuLaunchKernel")
    torch.cuda.synchronize()
    r = rec[: grid * warps * REC].view(grid, warps, REC).cpu().tolist()
    return r, smid[:grid].cpu().tolist()


def side_rates(records, variant, kind, role):
    """Per block: the side's FLOPs per cycle (sum over its warps); and the chip's FLOPs per ns."""
    per_block, per_ns = [], 0.0
    for block in records:
        warps = [w for w in block if w[0] == role]
        if not warps:
            continue
        ops = ops_per_chunk(variant, kind, role)
        per_block.append(sum(w[1] * ops / (w[3] - w[2]) for w in warps))
        per_ns += sum(w[1] * ops / max(1, w[5] - w[4]) for w in warps)
    return (statistics.median(per_block) if per_block else 0.0), per_ns


def measure(torch, kernel, variant, kind, grid, s, m, budget, fill, buffers, repeats):
    # *_all: every warp of the block on one side, the alternative to splitting the warps between the sides.
    modes = {"simt_alone": (s, 0), "mma_alone": (0, m), "both": (s, m), "simt_all": (s + m, 0),
             "mma_all": (0, s + m)}
    raw = {}
    unique = True
    for mode, (sw, mw) in modes.items():
        samples = []
        for _ in range(repeats):
            recs, smids = launch(torch, kernel, grid, sw, mw, budget, fill, buffers)
            unique &= len(set(smids)) == len(smids)
            # role id: warps < sw are SIMT (0); the rest MMA (1)
            simt = side_rates(recs, variant, kind, 0) if sw else (0.0, 0.0)
            mma = side_rates(recs, variant, kind, 1) if mw else (0.0, 0.0)
            samples.append((simt, mma))
        raw[mode] = {
            "simt_flop_per_cycle_sm": statistics.median(x[0][0] for x in samples),
            "mma_flop_per_cycle_sm": statistics.median(x[1][0] for x in samples),
            "simt_tflops": statistics.median(x[0][1] for x in samples) / 1e3,
            "mma_tflops": statistics.median(x[1][1] for x in samples) / 1e3,
        }
    sa = raw["simt_alone"]["simt_flop_per_cycle_sm"]
    ma = raw["mma_alone"]["mma_flop_per_cycle_sm"]
    s_all = max(sa, raw["simt_all"]["simt_flop_per_cycle_sm"])
    m_all = max(ma, raw["mma_all"]["mma_flop_per_cycle_sm"])
    sb = raw["both"]["simt_flop_per_cycle_sm"]
    mb = raw["both"]["mma_flop_per_cycle_sm"]
    row = {"variant": variant, "kind": kind, "grid": grid, "simt_warps": s, "mma_warps": m,
           "regs": kernel["regs"], "one_block_per_sm": unique, "modes": raw,
           "keep": sb / sa, "mma_share": mb / ma, "decompositions": {}}
    for name, c in DECOMPOSITIONS[kind].items():
        useful = sb + mb / c
        row["decompositions"][name] = {"c": c, "useful_vs_simt": useful / sa,
                                       "gain_vs_best_single": useful / max(s_all, m_all / c),
                                       "mma_alone_fp32_equiv_vs_simt": (ma / c) / sa}
    p = row["decompositions"][PRIMARY[kind]]
    row["pass"] = bool(row["keep"] >= KEEP_MIN and p["gain_vs_best_single"] >= GAIN_MIN)
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="Fresh directory")
    p.add_argument("--label", default="", help="Machine label for the record")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), help="SxM warps per block, multiples of 4")
    p.add_argument("--grids", nargs="+", default=["1", "sm"], help="blocks: integers or 'sm' (one per SM)")
    p.add_argument("--budget", type=int, default=2_000_000, help="cycles per launch")
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from cuda.bindings import driver

    torch.cuda.init()
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    cap = torch.cuda.get_device_capability(dev)
    torch.zeros(1, device="cuda")  # make torch's primary context current for the driver calls
    check(driver.cuInit(0), "cuInit")
    _, kernels, log = compile_module(cap)
    clock_khz = check(driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev),
                      "clock")
    sms = props.multi_processor_count
    grids = [sms if g == "sm" else int(g) for g in args.grids]
    max_warps = max(sum(int(v) for v in c.split("x")) for c in args.configs)
    inp = torch.full((128,), 1e-3, device="cuda")
    inp[64] = 0.999
    buffers = (inp, torch.empty(max(grids) * 32 * max_warps, device="cuda"),
               torch.zeros(max(grids) * max_warps * REC, dtype=torch.int64, device="cuda"),
               torch.zeros(max(grids), dtype=torch.int32, device="cuda"))
    fills = {"tf32": int.from_bytes(torch.tensor([1e-3]).numpy().tobytes(), "little"), "bf16": 0x3A833A83}
    meta = {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "label": args.label,
            "gpu": props.name, "capability": f"{cap[0]}.{cap[1]}", "sms": sms, "clock_khz_nominal": int(clock_khz),
            "torch": torch.__version__, "cuda": torch.version.cuda, "budget_cycles": args.budget,
            "repeats": args.repeats, "nvrtc_log": log,
            "kernels": {f"{v}_{k}": {x: y for x, y in d.items() if x != "fn"} for (v, k), d in kernels.items()},
            "rule": {"keep_min": KEEP_MIN, "gain_min": GAIN_MIN, "primary": PRIMARY}}
    print(json.dumps({k: v for k, v in meta.items() if k != "nvrtc_log"}, indent=1), flush=True)
    rows = []
    # Warm-up: clocks up, module resident.
    for _ in range(3):
        launch(torch, kernels[("reg", "tf32")], sms, 4, 4, args.budget, fills["tf32"], buffers)
    print(f"{'variant':6} {'kind':4} {'grid':>4} {'SxM':>4} {'regs':>4} | {'simt':>6} {'mma':>6} {'s_all':>6} {'m_all':>6} | {'s_both':>6} "
          f"{'m_both':>6} | {'keep':>5} {'mma%':>5} {'gain':>5} {'decomp':>7} | verdict", flush=True)
    for variant in args.variants:
        for kind in args.kinds:
            kernel = kernels[(variant, kind)]
            for config in args.configs:
                s, m = (int(v) for v in config.split("x"))
                if 32 * (s + m) > kernel["max_threads"]:
                    rows.append({"variant": variant, "kind": kind, "simt_warps": s, "mma_warps": m,
                                 "skipped": f"{32 * (s + m)} threads > {kernel['max_threads']} at "
                                            f"{kernel['regs']} registers"})
                    print(f"{variant:6} {kind:4} {'':>4} {config:>4} {kernel['regs']:>4} | skipped "
                          f"({kernel['max_threads']} threads max)", flush=True)
                    continue
                for grid in grids:
                    row = measure(torch, kernel, variant, kind, grid, s, m, args.budget, fills[kind], buffers,
                                  args.repeats)
                    rows.append(row)
                    md = row["modes"]
                    d = row["decompositions"][PRIMARY[kind]]
                    print(f"{variant:6} {kind:4} {grid:>4} {config:>4} {row['regs']:>4} | "
                          f"{md['simt_alone']['simt_flop_per_cycle_sm']:6.1f} "
                          f"{md['mma_alone']['mma_flop_per_cycle_sm']:6.1f} "
                          f"{md['simt_all']['simt_flop_per_cycle_sm']:6.1f} {md['mma_all']['mma_flop_per_cycle_sm']:6.1f} | "
                          f"{md['both']['simt_flop_per_cycle_sm']:6.1f} {md['both']['mma_flop_per_cycle_sm']:6.1f} | "
                          f"{row['keep']:5.2f} {row['mma_share']:5.2f} {d['gain_vs_best_single']:5.2f} "
                          f"{PRIMARY[kind]:>7} | {'PASS' if row['pass'] else 'fail'}"
                          f"{'' if row['one_block_per_sm'] else ' (shared SM!)'}", flush=True)
    measured = [r for r in rows if "skipped" not in r]
    realistic = [r for r in measured if r["variant"] == "gemm"] or [r for r in measured if r["variant"] == "smem"]
    verdict = {"pass_any": any(r["pass"] for r in measured), "pass_realistic": any(r["pass"] for r in realistic),
               "best_realistic": max(realistic, key=lambda r: (r["keep"] >= KEEP_MIN,
                                     r["decompositions"][PRIMARY[r["kind"]]]["gain_vs_best_single"]),
                                     default=None)}
    summary = {"meta": meta, "rows": rows, "verdict": verdict}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"verdict: pass_realistic={verdict['pass_realistic']} pass_any={verdict['pass_any']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
