"""Idea 11: where the float32 edge vectors of a perfect SrTiO3 crystal lose accuracy, by box size.

The tiled call forms ``pos32[receiver] - pos32[sender] + shift32`` (tiled._tile_vectors; float32 positions from
``field.positions``, float32 shifts from the float64 ``unit @ cell``). For sampled receivers (both box faces and the
middle) and every neighbour within 6 A (perfect lattice, analytic), the float32 vector against the float64 one, split
into the positions' rounding, the shift's rounding, and the float32 arithmetic.
"""
import itertools
import sys

import numpy as np

A = 3.905
BASIS = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])  # Sr Ti O O O
CUTOFF = 6.0
OFFSETS = np.array(list(itertools.product(range(-2, 3), repeat=3)))


def analyse(n, origin=0.0, samples=200, seed=0):
    rng = np.random.default_rng(seed)
    L = n * A
    rows = []
    # receivers: cells on the low face, the high face and the middle
    for _ in range(samples):
        cell = rng.integers(0, n, size=3)
        face = rng.integers(0, 3)
        cell[rng.integers(0, 3)] = [0, n - 1, n // 2][face]
        b = rng.integers(0, 5)
        for off in OFFSETS:
            for bs in range(5):
                raw = cell + off
                unit = np.floor_divide(raw, n)  # the sender's image: raw = wrapped + unit * n
                wrapped = raw - unit * n
                p_r = origin + (cell + BASIS[b]) * A
                p_s = origin + (wrapped + BASIS[bs]) * A
                shift = -unit * L  # the sender image p_s + unit L: v = p_r - p_s + shift
                v64 = p_r - p_s + shift
                d = np.linalg.norm(v64)
                if d == 0 or d >= CUTOFF:
                    continue
                r32, s32, sh32 = np.float32(p_r), np.float32(p_s), np.float32(shift)
                v32 = (r32 - s32) + sh32  # float32 arithmetic, as on the device
                pos_err = (r32.astype(np.float64) - s32.astype(np.float64)) - (p_r - p_s)
                shift_err = sh32.astype(np.float64) - shift
                total = v32.astype(np.float64) - v64
                rows.append((np.abs(total).max(), np.abs(pos_err).max(), np.abs(shift_err).max(),
                             bool(np.any(unit != 0))))
    rows = np.array(rows, dtype=float)
    cross = rows[:, 3] > 0
    return dict(n=n, box=L, origin=origin, edges=len(rows), cross=int(cross.sum()),
                total=rows[:, 0].max(), pos=rows[:, 1].max(), shift=rows[:, 2].max(),
                total_inner=rows[~cross, 0].max(), total_cross=rows[cross, 0].max() if cross.any() else 0.0,
                rms_total=float(np.sqrt((rows[:, 0] ** 2).mean())))


if __name__ == "__main__":
    print(f"{'n':>4} {'box A':>8} {'origin':>7} {'edges':>7} {'cross':>6} {'max|dv|':>9} {'pos':>9} {'shift':>9} "
          f"{'inner':>9} {'across':>9} {'rms':>9}")
    for n, origin in ((58, 0.0), (310, 0.0), (400, 0.0), (6, 0.0), (6, 1500.0), (58, -113.0)):
        r = analyse(n, origin)
        print(f"{r['n']:>4} {r['box']:>8.1f} {r['origin']:>7.0f} {r['edges']:>7} {r['cross']:>6} {r['total']:>9.2e} "
              f"{r['pos']:>9.2e} {r['shift']:>9.2e} {r['total_inner']:>9.2e} {r['total_cross']:>9.2e} "
              f"{r['rms_total']:>9.2e}")
    sys.exit(0)
