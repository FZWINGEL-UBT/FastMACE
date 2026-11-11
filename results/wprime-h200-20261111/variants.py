"""Source variants of today's H200 w' kernel (sources/wprime-h200.cu, render_sources.py) for the sm_90 census.

  skip      exact: an edge at or beyond the cutoff (inside = false: the engine's padding edges, since the skin list
            is compacted to the cutoff on the device) writes dsh = 0 and dr = 0 and skips the channel loop; the
            table loads in the loop are then unconditional (no zeroed gv registers, no predicate).
  hoist     exact: the dx atomics' address (dx + input_base + channel) as one pointer per lane, stepped per pass.
  vecN      rounding (N = 2, 4): each lane owns N consecutive channels; the staged dout row, x and dx are read and
            added as floatN (LDS.64/128, LDG.64/128, vector RED on sm_90); per channel the arithmetic is unchanged,
            but each lane's edge-gradient and radial-gradient sums take its channels in a new order.

Combinations join with '+', e.g. ``skip+vec4``. ``python variants.py`` writes sources/wprime-h200-<name>.cu.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
BASE = HERE / "sources" / "wprime-h200.cu"
from fastmace.kernels.oeq_streaming import wprime_tuning as _t  # noqa: E402

LOOP, COMP, _sub, _block, loop_parts = _t.LOOP, _t.COMP, _t._sub, _t._block, _t.loop_parts
hoist, vec, skip = _t.hoist, _t.vec, _t.skip

TRANSFORMS = {"skip": skip, "hoist": hoist, "vec2": lambda s: vec(s, 2), "vec4": lambda s: vec(s, 4)}
VARIANTS = ["skip", "hoist", "skip+hoist", "vec2", "vec4", "skip+vec2", "skip+vec4"]


def build(name, source=None):
    source = BASE.read_text() if source is None else source
    for part in name.split("+"):
        source = TRANSFORMS[part](source)
    return source


def main(names=None):
    for name in names or VARIANTS:
        (HERE / "sources" / f"wprime-h200-{name}.cu").write_text(build(name))
        print("wrote", name)


if __name__ == "__main__":
    main(sys.argv[1:])
