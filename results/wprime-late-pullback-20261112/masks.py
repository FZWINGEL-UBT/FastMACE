"""Design 1 in production form and design 2's hybrids: ``wprime_pullback.apply`` on today's H200 render with
SKIP_OUTSIDE (results/wprime-h200-20261111/sources/wprime-h200-skip.cu, FOLD_ALPHA on), then HOIST_DX
(``wprime_tuning.hoist``), for every set of direct edge degrees (16 masks; mask bit l = edge degree l direct).

Writes sources/mask-<bits>.cu (with hoist; census) and sources/mask-<bits>-nohoist.cu (emulate2.py).

    python masks.py
"""
from common import H200, HERE, setup

BASE = (H200 / "sources" / "wprime-h200-skip.cu").read_text()


def main():
    from fastmace.kernels.oeq_streaming import wprime_pullback as wpb
    from fastmace.kernels.oeq_streaming import wprime_tuning as wt
    s = setup()
    assert wpb.supported(s.schedule, "float", s.blocks)
    out = HERE / "sources"
    out.mkdir(exist_ok=True)
    for mask in range(16):
        direct = [l for l in range(4) if mask >> l & 1]
        bits = "".join("D" if mask >> l & 1 else "s" for l in range(4))  # l = 0..3
        src = wpb.apply(BASE, s.schedule, s.blocks, fold_alpha=True, direct=direct)
        (out / f"mask-{bits}-nohoist.cu").write_text(src)
        (out / f"mask-{bits}.cu").write_text(wt.hoist(src))
        print("wrote", bits)


if __name__ == "__main__":
    main()
