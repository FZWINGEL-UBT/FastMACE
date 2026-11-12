"""tf32x3 by an explicit register split (Ada follow-up, 12 November 2026): offline falsifier, CPU only.

Compiles FastMACE's two Triton GEMMs (tf32x3_gemm.gemm, species_gemm.gemm) for GPUTarget("cuda", ARCH, 32) with
production's tile (64 x 64 x 32, 4 warps, 3 stages), each as
  - "tf32x3": production, tl.dot(input_precision="tf32x3") (Triton's own split);
  - "split": a and b split in registers, hi rounded to nearest even TF32 by integer add and mask
    (hi = (u + 0x0FFF + ((u >> 13) & 1)) & 0xFFFFE000, lo = x - hi), then three tl.dot(..., "tf32"), the small
    products first (lo*hi, hi*lo, then hi*hi);
  - "tf32": one tl.dot(..., "tf32") (the floor: one product, no split).
Per variant: metadata shared bytes, local_alloc / local_load / convert_layout counts in the TTGIR, mma and cvt.rna
counts in the PTX, and ptxas -v registers / spill stores / spill loads.

  python split_lab.py OUT.json [ARCH ...]     (default arch 89; PATH must hold Triton's ptxas)
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

BM, BN, BK, WARPS, STAGES = 64, 64, 32, 4, 3


@triton.jit
def _split(x):
    u = x.to(tl.uint32, bitcast=True)
    hi = (((u + 0x0FFF + ((u >> 13) & 1)) >> 13) << 13).to(tl.float32, bitcast=True)  # & 0xFFFFE000, no int64 const
    return hi, x - hi


@triton.jit
def _product(acc, a, b, MODE: tl.constexpr):
    if MODE == "split":
        a_hi, a_lo = _split(a)
        b_hi, b_lo = _split(b)
        acc = tl.dot(a_lo, b_hi, acc, input_precision="tf32")
        acc = tl.dot(a_hi, b_lo, acc, input_precision="tf32")
        acc = tl.dot(a_hi, b_hi, acc, input_precision="tf32")
    elif MODE == "tf32x3":
        acc += tl.dot(a, b, input_precision="tf32x3")
    else:
        acc += tl.dot(a, b, input_precision="tf32")
    return acc


@triton.jit
def gemm(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, MODE: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, K, BLOCK_K):
        kk = start + rk
        a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak,
                    mask=(rm < M)[:, None] & (kk < K)[None, :], other=0.0)
        b = tl.load(b_ptr + kk[:, None] * sbk + rn[None, :] * sbn,
                    mask=(kk < K)[:, None] & (rn < N)[None, :], other=0.0)
        acc = _product(acc, a, b, MODE)
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm < M)[:, None] & (rn < N)[None, :])


@triton.jit
def species(x_ptr, w_ptr, y_ptr, order_ptr, counts_ptr, S, size, N, K, sxa, sxj, sxk, swk, swn, sya, syj, syn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, MODE: tl.constexpr):
    t, pn = tl.program_id(0), tl.program_id(1)
    sp, start, end = 0, 0, 0
    tiles_before, rows_before = 0, 0
    for s in range(S):
        rows = tl.load(counts_ptr + s) * size
        n_tiles = (rows + BLOCK_M - 1) // BLOCK_M
        mine = (t >= tiles_before) & (t < tiles_before + n_tiles)
        sp = tl.where(mine, s, sp)
        start = tl.where(mine, rows_before + (t - tiles_before) * BLOCK_M, start)
        end = tl.where(mine, rows_before + rows, end)
        tiles_before += n_tiles
        rows_before += rows
    if start >= end:
        return
    r = start + tl.arange(0, BLOCK_M)
    live = r < end
    i = tl.where(live, r // size, 0)
    j = r - i * size
    atom = tl.load(order_ptr + i, mask=live, other=0).to(tl.int64)
    xrow = atom * sxa + j.to(tl.int64) * sxj
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    wbase = w_ptr + sp.to(tl.int64) * K * N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        a = tl.load(x_ptr + xrow[:, None] + kk[None, :] * sxk, mask=live[:, None] & (kk < K)[None, :], other=0.0)
        b = tl.load(wbase + kk[:, None] * swk + rn[None, :] * swn, mask=(kk < K)[:, None] & (rn < N)[None, :],
                    other=0.0)
        acc = _product(acc, a, b, MODE)
    yrow = atom * sya + j.to(tl.int64) * syj
    tl.store(y_ptr + yrow[:, None] + rn[None, :] * syn, acc, mask=live[:, None] & (rn < N)[None, :])


def source(kernel, mode):
    sig, attrs = {}, {}
    for index, name in enumerate(kernel.arg_names):
        if name in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "MODE"):
            sig[name] = "constexpr"
        elif name.endswith("_ptr"):
            sig[name] = "*i32" if name in ("order_ptr", "counts_ptr") else "*fp32"
            attrs[(index,)] = [["tt.divisibility", 16]]
        else:
            sig[name] = "i32"
    const = {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK, "MODE": mode}
    return ASTSource(fn=kernel, signature=sig, constexprs=const, attrs=attrs)


def ptxas_v(ptx, arch):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "k.ptx"
        p.write_text(ptx)
        r = subprocess.run(["ptxas", "-v", f"--gpu-name=sm_{arch}", str(p), "-o", str(Path(d) / "k.cubin")],
                           capture_output=True, text=True)
    text = r.stderr + r.stdout
    regs = re.search(r"Used (\d+) registers", text)
    spill = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", text)
    stack = re.search(r"(\d+) bytes stack frame", text)
    return {"registers": int(regs.group(1)) if regs else None,
            "spill_stores": int(spill.group(1)) if spill else None,
            "spill_loads": int(spill.group(2)) if spill else None,
            "stack": int(stack.group(1)) if stack else None, "ptxas": text.strip()[-400:]}


def main():
    out = Path(sys.argv[1])
    archs = [int(a) for a in sys.argv[2:]] or [89]
    rows = []
    for arch in archs:
        for kname, kernel in (("tf32x3_gemm.gemm", gemm), ("species_gemm.gemm", species)):
            for mode in ("tf32x3", "split", "tf32"):
                row = {"triton": triton.__version__, "arch": arch, "kernel": kname, "mode": mode}
                try:
                    c = triton.compile(source(kernel, mode), target=GPUTarget("cuda", arch, 32),
                                       options={"num_warps": WARPS, "num_stages": STAGES})
                    ttgir, ptx = c.asm["ttgir"], c.asm["ptx"]
                    row.update(shared=c.metadata.shared,
                               local_alloc=ttgir.count("ttg.local_alloc"), local_load=ttgir.count("ttg.local_load"),
                               convert_layout=ttgir.count("ttg.convert_layout"),
                               async_copy=ttgir.count("async_copy_global_to_local"),
                               mma=len(re.findall(r"\bmma\.sync", ptx)), cvt_rna_tf32=ptx.count("cvt.rna.tf32"),
                               **ptxas_v(ptx, arch))
                except Exception as e:  # noqa: BLE001
                    row["error"] = f"{type(e).__name__}: {e}"[:600]
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "ptxas"}), flush=True)
    out.write_text(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    main()
