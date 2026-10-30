"""Same-process A/B of segment_gemm's irrep size D as a runtime argument (side a, ``main``'s kernel before the
change) against D as ``tl.constexpr`` (side b, the library's kernel); plans/capacity-ideas-20261030.md, fourth list.

``tools/ab_rewrites.py`` with every option at its default on both sides. Measurement only: this process replaces
``fastmace.kernels.triton.segment_gemm._kernel`` with a dispatcher that returns the runtime-D kernel while side a
builds or steps (its CUDA graphs capture that kernel, also when a new edge size re-captures in the warm-up) and the
library's kernel otherwise. Arguments are ``ab_rewrites.py``'s, without ``--a``/``--b`` options.

A record of docs/validation/speed-small.md's A/B, run at ``54942faf``, where the library's kernel took D as
``tl.constexpr``. The change was measured neutral and not adopted (reverted), so from then on both sides run the same
kernel unless the candidate is restored.
"""

from __future__ import annotations

import sys
from functools import cache
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ab_rewrites  # noqa: E402
import ab_step_time  # noqa: E402

SIDE = "b"  # the side being built or stepped; ab_rewrites sets it to "a" for side a's build (--a-constant)
_ENGINE_SIDES = {}


@cache
def runtime_d_kernel():
    """``segment_gemm``'s Triton GEMM as on ``main`` before the change: D a runtime argument."""
    import triton
    import triton.language as tl

    @triton.jit
    def gemm(a_ptr, b_ptr, c_ptr, rows, N, K, D, a_n, a_m, b_k, b_n, c_n, c_m,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PRECISION: tl.constexpr):
        pm, pn = tl.program_id(0), tl.program_id(1)
        r = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        j = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        atom, comp = r // D, r % D
        a_row = atom * a_n + comp * a_m
        kk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start in range(0, K, BLOCK_K):
            k = start + kk
            a = tl.load(a_ptr + a_row[:, None] + k[None, :], mask=(r < rows)[:, None] & (k < K)[None, :], other=0.0)
            b = tl.load(b_ptr + k[:, None] * b_k + j[None, :] * b_n, mask=(k < K)[:, None] & (j < N)[None, :],
                        other=0.0)
            acc += tl.dot(a, b, input_precision=PRECISION)
        tl.store(c_ptr + (atom * c_n + comp * c_m)[:, None] + j[None, :], acc,
                 mask=(r < rows)[:, None] & (j < N)[None, :])

    return gemm


def install():
    """The kernel dispatcher, and the side of each engine (recorded at its build, set before each of its steps)."""
    from fastmace.kernels.triton import segment_gemm
    from fastmace.planner import minimal

    library = segment_gemm._kernel
    this = sys.modules[__name__]
    segment_gemm._kernel = lambda: runtime_d_kernel() if this.SIDE == "a" else library()

    build = minimal.MinimalPlanner.build

    def recorded_build(self, *args, **kwargs):
        engine = build(self, *args, **kwargs)
        _ENGINE_SIDES[id(engine)] = this.SIDE
        return engine

    minimal.MinimalPlanner.build = recorded_build
    step = ab_step_time.Tf32Side.step

    def sided_step(self):
        this.SIDE = _ENGINE_SIDES[id(self.side.engine)]
        try:
            step(self)
        finally:
            this.SIDE = "b"

    ab_step_time.Tf32Side.step = sided_step


def main(argv=None):
    sys.modules.setdefault("ab_constexpr_d", sys.modules[__name__])
    argv = list(sys.argv[1:] if argv is None else argv)
    ab_step_time.prepare_fastmace_hybrid()  # before anything imports MACE
    install()
    return ab_rewrites.main([*argv, "--a-constant", "ab_constexpr_d:SIDE='a'"])


if __name__ == "__main__":
    main()
