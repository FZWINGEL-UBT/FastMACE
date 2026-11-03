"""The wavefront's band against the tile shape, on a block model of a periodic cubic cell (CPU only, no torch).

Tiles are blocks of a layer-major grid: layers along the wave's axis (any thicknesses, periodic), a K x K lateral
grid in each. A tile reads the atoms within the cutoff of its receivers: with layers and lateral blocks at least
one cutoff thick, those are the blocks one step away in each direction (periodic). ``wavefront_schedule`` (copied
from fastmace/model/tiled.py at 91f646b7) then gives the row pool's peak. The halo of a tile is the shell of one
cutoff around its block (uniform density), clipped to the neighbouring layers' thicknesses.

Variants:
- cubic: equal layers (today's blocks);
- skin: the first and last layer each split into a cutoff-thick skin at the periodic boundary and the rest, so only
  the two skins stay in the band for the whole call (the wrap's readers);
- flat: thinner layers (half the thickness) with wider lateral blocks at the same receivers per tile.

Usage: python band_model.py
"""

import math

DENSITY = 5 / 3.905**3  # SrTiO3 atoms per cubic Angstrom (a = 3.905)
CUTOFF = 6.0  # MACE-OMAT-0 medium


def wavefront_schedule(needs, ready, counts, group=1):
    tiles = len(counts)
    order = sorted(range(tiles), key=lambda u: (int(ready[u]), u))
    done = [False] * tiles
    steps, live, peak, next3 = [], 0, 0, 0
    for start in range(0, tiles, max(1, int(group))):
        stop = min(tiles, start + max(1, int(group)))
        for t in range(start, stop):
            for u in needs[t]:
                if not done[u]:
                    done[u] = True
                    steps.append((1, u))
                    live += counts[u]
                    peak = max(peak, live)
        steps.extend((2, t) for t in range(start, stop))
        while next3 < tiles and ready[order[next3]] < stop:
            u = order[next3]
            next3 += 1
            steps.append((3, u))
            live -= counts[u]
    return steps, peak


def model(side, layers, lateral, group=16):
    """``layers``: the layers' thicknesses along the wave (sum = ``side``); ``lateral``: blocks per lateral axis."""
    nl, k = len(layers), lateral
    s = side / k
    index = {}
    for i in range(nl):
        for a in range(k):
            for b in range(k):
                index[(i, a, b)] = len(index)
    counts, halo, needs = [], [], []
    for (i, a, b), t in index.items():
        thick = layers[i]
        counts.append(DENSITY * thick * s * s)
        below, above = min(CUTOFF, layers[i - 1]), min(CUTOFF, layers[(i + 1) % nl])
        halo.append(DENSITY * ((thick + below + above) * (s + 2 * CUTOFF) ** 2 - thick * s * s))
        near = {index[((i + di) % nl, (a + da) % k, (b + db) % k)]
                for di in (-1, 0, 1) for da in (-1, 0, 1) for db in (-1, 0, 1)}
        needs.append(sorted(near))
    ready = [max(needs[t]) for t in range(len(counts))]  # symmetric reads: the last tile that reads t
    _, peak = wavefront_schedule(needs, ready, counts, group)
    n = sum(counts)
    return dict(tiles=len(counts), receivers=n / len(counts), pool=peak / n,
                halo=sum(halo) / n, pool_gib=peak * 4096 / 2**30)


def variants(atoms, receivers):
    side = (atoms / DENSITY) ** (1 / 3)
    cube = (receivers / DENSITY) ** (1 / 3)
    k = max(3, round(side / cube))
    layers = [side / k] * k
    skin = [CUTOFF, side / k - CUTOFF] + layers[1:-1] + [side / k - CUTOFF, CUTOFF]
    kf = max(3, round(side / (cube / 2)))  # half-thick layers; lateral blocks wider by sqrt(2)
    flat = [side / kf] * kf
    lateral_flat = max(3, round(side / (cube * math.sqrt(2))))
    skin_flat = [CUTOFF, side / kf - CUTOFF] + flat[1:-1] + [side / kf - CUTOFF, CUTOFF]
    return side, {"cubic": (layers, k), "cubic + skin": (skin, k), "flat": (flat, lateral_flat),
                  "flat + skin": (skin_flat, lateral_flat)}


if __name__ == "__main__":
    for atoms, receivers, group in ((98.4e6, 8192, 1), (121.9e6, 8192, 1), (149.0e6, 4096, 1), (149.0e6, 8192, 1),
                                    (149.0e6, 8192, 16), (300e6, 8192, 16)):
        side, cases = variants(atoms, receivers)
        print(f"{atoms / 1e6:.1f}M atoms, R {receivers}, group {group}, cell {side:.0f} A")
        for name, (layers, lateral) in cases.items():
            r = model(side, layers, lateral, group)
            print(f"  {name:13s} tiles {r['tiles']:6d}  receivers/tile {r['receivers']:7.0f}  halo/receivers "
                  f"{r['halo']:.2f}  pool {100 * r['pool']:5.1f} % ({r['pool_gib']:.0f} GiB at 4 KB per atom)")
