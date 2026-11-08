"""PACKED_COEF falsifier (idea 3, Codex H200 report): the tabled outer sum's walks compiled for sm_90 without a GPU.

Compiles ``tabled_outer_sum``'s forward walk and its fused backward (the H200's POWER_BACKWARD and the RECOMPUTE
side) with the unpacked and the packed coefficient rows through ``triton.compile`` for ``GPUTarget("cuda", 90, 32)``
and counts the global loads in the PTX (``ld.global`` scalar / ``.v2`` / ``.v4``) and the SASS (``LDG.E`` /
``.64`` / ``.128``), with ptxas's registers and spills. The premise under test: today each lane reads its two
channels of each term as one 8-byte load (TERMS loads), packed it reads two terms as one 16-byte load (TERMS / 2).

No GPU and no CUDA device are touched (run with ``CUDA_VISIBLE_DEVICES=``)::

    CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD python tools/packed_coef_lab.py

The shapes are OMAT-0 medium's interaction 1: K = 64 channels, J = 66 table columns, degree 5 (6 terms), D = 16.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

POINTER_TYPES = {"pair_ptr": "*i32", "bins_ptr": "*i32", "key_ptr": "*i64"}
INTS = ("num_edges", "a_stride", "E_DENSITY", "n_pieces", "n_bins")
ALIGNED_INTS = ("a_stride", "n_pieces", "n_bins")  # 64, PIECE_BUCKET multiples, 4096: Triton's 16-divisibility
FLOATS = ("r_lo", "bin_scale", "r_max")


def _source(kernel, constexprs):
    """``ASTSource`` of ``kernel`` with ``constexprs`` (name -> value), every pointer 16-byte aligned."""
    from triton.compiler import ASTSource

    signature, attrs = {}, {}
    for index, name in enumerate(kernel.arg_names):
        if name in constexprs:
            signature[name] = "constexpr"
        elif name.endswith("_ptr"):
            signature[name] = POINTER_TYPES.get(name, "*fp32")
            attrs[(index,)] = [["tt.divisibility", 16]]
        elif name in INTS:
            signature[name] = "i32"
            if name in ALIGNED_INTS:
                attrs[(index,)] = [["tt.divisibility", 16]]
        elif name in FLOATS:
            signature[name] = "fp32"
        else:
            raise ValueError(f"no type for argument {name}")
    return ASTSource(fn=kernel, signature=signature, constexprs=constexprs, attrs=attrs)


def compile_walk(which, packed, *, k=64, d=16, terms=6, columns=66, capability=90):
    """The compiled kernel (``triton.compile``) of walk ``which`` ("forward", "power" or "recompute")."""
    import triton
    from triton.backends.compiler import GPUTarget

    from fastmace.kernels.triton import tabled_outer_sum as tos

    block_k, pairs = triton.next_power_of_2(k), columns // 2  # ceil((columns - 1) / 2): packed_coef's pairs
    common = dict(K=k, D=d, INNER=1, J=columns, BLOCK_K=block_k, BLOCK_D=triton.next_power_of_2(d),
                  PACKED=bool(packed), PAIRS=pairs if packed else 1)
    if which == "forward":
        kernel, warps = tos._forward_kernel(), tos.FORWARD_WARPS
        constexprs = dict(common, TERMS=terms, CHUNK=tos.FORWARD_CHUNK)
    elif which in ("power", "recompute"):
        kernel, warps = tos._kernel(), tos.WARPS
        constexprs = dict(common, TERMS=terms - 1, CHUNK=tos.CHUNK, RECOMPUTE=True, VTERMS=terms,
                          POWER=which == "power")
    else:
        raise ValueError(which)
    target = GPUTarget("cuda", capability, 32)
    return triton.compile(_source(kernel, constexprs), target=target, options={"num_warps": warps})


def ptx_loads(ptx):
    """{"v4", "v2", "scalar"}: the PTX's ``ld.global`` instructions by vector width."""
    counts = {"v4": 0, "v2": 0, "scalar": 0}
    for line in ptx.splitlines():
        match = re.search(r"\bld\.global\S*", line)
        if not match:
            continue
        op = match.group(0)
        counts["v4" if ".v4." in op else "v2" if ".v2." in op else "scalar"] += 1
    return counts


def sass_loads(sass):
    """{"128", "64", "32"}: the SASS's ``LDG`` instructions by width."""
    counts = {"128": 0, "64": 0, "32": 0}
    for match in re.finditer(r"\bLDG(\.[A-Z0-9_]+)*", sass):
        op = match.group(0)
        counts["128" if ".128" in op else "64" if ".64" in op else "32"] += 1
    return counts


def ptxas_usage(ptx, capability=90):
    """(registers, spill store bytes, spill load bytes) from ``ptxas -v`` on ``ptx`` (None where unavailable)."""
    try:
        from triton.backends.nvidia.compiler import get_ptxas, sm_arch_from_capability

        ptxas = get_ptxas(capability).path
    except Exception:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "walk.ptx"
        source.write_text(ptx)
        result = subprocess.run([ptxas, "-v", f"--gpu-name={sm_arch_from_capability(capability)}", str(source), "-o",
                                 str(Path(tmp) / "walk.cubin")], capture_output=True, text=True, check=True)
    log = result.stderr + result.stdout
    registers = re.search(r"Used (\d+) registers", log)
    stores = re.search(r"(\d+) bytes spill stores", log)
    loads = re.search(r"(\d+) bytes spill loads", log)
    return (int(registers.group(1)) if registers else None, int(stores.group(1)) if stores else 0,
            int(loads.group(1)) if loads else 0)


def measure(which, packed, **shape):
    """One row: the walk's PTX and SASS load counts, registers and spills, and shared-memory traffic."""
    compiled = compile_walk(which, packed, **shape)
    ptx = compiled.asm["ptx"]
    row = {"walk": which, "packed": bool(packed), "ptx": ptx_loads(ptx),
           "shared": len(re.findall(r"\b(?:st|ld)\.shared", ptx)), "usage": ptxas_usage(ptx)}
    try:
        row["sass"] = sass_loads(compiled.asm["sass"])
    except Exception as error:  # cuobjdump missing
        row["sass"] = f"unavailable ({type(error).__name__})"
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--walks", nargs="+", default=["forward", "power", "recompute"])
    args = parser.parse_args(argv)
    for which in args.walks:
        for packed in (False, True):
            row = measure(which, packed)
            print(f"{which:9s} packed={row['packed']!s:5s} ptx={row['ptx']} sass={row['sass']} "
                  f"shared={row['shared']} regs/spill-st/spill-ld={row['usage']}")


if __name__ == "__main__":
    main()
