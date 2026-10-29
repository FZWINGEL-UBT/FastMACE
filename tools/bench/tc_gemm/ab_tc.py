"""Same-process whole-step A/B of GEMM backends (tensor-core GEMM screen, 29 October 2026): ``tools/ab_rewrites.py``
with a ``cutlass_tc`` backend per side.

    python ab_tc.py [--a-backend VALUE] [--b-backend VALUE] -- AB_REWRITES_ARGS...

VALUE is a ``cutlass_tc`` backend (``rz3``, ``rn3``, ``rn4``, ``tf32``, ``simt-triton``, ``simt-all``,
``cublas-triton``, ``cublas-all``) or ``none`` (the library's own launches, the default). Side b normally also
needs ``--b tf32x3_gemms=on`` for a tensor-core variant (the record decides which products are rerouted).

Each side's ``FastMACESide`` is wrapped: when it is built (``ab_rewrites`` builds the sides in build order), its CUDA
graphs are dropped, and before each of its steps ``cutlass_tc.CURRENT`` is set to its backend, so whatever it
captures or launches eagerly uses its own GEMMs. Both sides' graphs are dropped after the build, so both re-capture
at their first step alike.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[1]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: ab_tc.py [--a-backend V] [--b-backend V] -- AB_REWRITES_ARGS...")
    split = argv.index("--")
    own, rest = argv[:split], argv[split + 1:]
    backends = {"a": None, "b": None}
    for i in range(0, len(own), 2):
        side = {"--a-backend": "a", "--b-backend": "b"}.get(own[i])
        if side is None or i + 1 >= len(own):
            raise SystemExit(f"unknown option {own[i]!r}")
        backends[side] = None if own[i + 1] == "none" else own[i + 1]
    order = "ab"
    if "--order" in rest:
        order = rest[rest.index("--order") + 1]
    sys.path.insert(0, str(TOOLS))
    sys.path.insert(0, str(HERE))
    import fastmace  # noqa: F401  (before torch, SPEC §14)
    import ab_rewrites
    import ab_step_time
    import cutlass_tc

    for value in backends.values():
        if value is not None:
            cutlass_tc.install(value)
    cutlass_tc.CURRENT = None
    plain = ab_step_time.FastMACESide
    built = [0]

    class BackendSide(plain):
        def __init__(self, engine, torch):
            super().__init__(engine, torch)
            self.backend = backends[order[built[0] % 2]]
            self.label = order[built[0] % 2]
            built[0] += 1
            engine.invalidate_graphs(f"GEMM screen: re-capture with backend {self.backend}")

        def step(self):
            cutlass_tc.CURRENT = self.backend
            super().step()

    ab_step_time.FastMACESide = BackendSide
    print(f"ab_tc: backends a={backends['a']} b={backends['b']} order={order}", flush=True)
    try:
        return ab_rewrites.main(rest)
    finally:
        print(f"ab_tc: rerouted launches {cutlass_tc.CALLS}", flush=True)


if __name__ == "__main__":
    main()
