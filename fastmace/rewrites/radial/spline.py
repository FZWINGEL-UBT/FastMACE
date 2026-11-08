"""Cubic Hermite splines of a generic interaction's radial weights ``w(r) = [h c, c] [W, 0]^T`` on piecewise-uniform
knots (``SPLINE_W``, sx-layout lab, 7 November 2026; results/sx-layout-lab-20261107).

Each segment of ``[R_LO, r_max)`` between the ``BAND_EDGES`` is uniform; its spacing is the coarsest (at most the
base spacing ``(r_max - R_LO) / BASE_INTERVALS``) whose certificate score is at most ``TARGET``. The score is
``radial.certify``'s: per pair and output column, the error over the column's largest exact value, divided by the
float32 network's own error, for values and r-slopes, over the whole domain and over ``r >= R_PHYSICAL``. A Hermite
spline's slope error scales with the spacing cubed, so a band's spacing is refined by that rule until it passes; the
whole spline then passes the certificate (worst <= 1) or is not used.

The coefficients are the power basis of ``t = (r - x_i) / h_i`` in ``[0, 1)``: ``w = c0 + t (c1 + t (c2 + t c3))``
from float64 values and exact slopes at the knots (``_GenericChain.with_derivative``), stored as float32 and scored
in float32 as the kernels evaluate them (one compare per band edge, then one multiply).
"""
from __future__ import annotations

import hashlib
import inspect
import sys

BAND_EDGES = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5)  # Angstrom; the bands below r_max are used
BASE_INTERVALS = 255  # the coarsest spacing: (r_max - R_LO) / BASE_INTERVALS (256 uniform knots)
TARGET = 0.5  # each band's certificate score after refinement (the certificate passes at 1)
ROUNDS = 8  # refinement rounds before the fit gives up
SPLINE_FORMAT = 1
CACHE_KIND = "radial-tables"  # beside the tables, as spline-<key>.pt


def grid_knots(grid):
    """float64 knot positions of ``grid`` ``[(start, end, intervals)]`` (shared end knots once)."""
    import torch

    xs = [torch.linspace(a, b, n + 1, dtype=torch.float64) for a, b, n in grid]
    return torch.cat([xs[0]] + [x[1:] for x in xs[1:]])


def coefficients(chain64, weight64, pairs, grid):
    """``[P, intervals, 4, Q]`` float64 power-basis coefficients in each interval's t, per pair of ``pairs``."""
    import torch

    x = grid_knots(grid)
    h = (x[1:] - x[:-1])[:, None]
    out = []
    for s, t in pairs:
        with torch.no_grad():
            g, dg = chain64.with_derivative(x, s, t)
        y = g[:, :-1].double() @ weight64.T
        m = dg[:, :-1].double() @ weight64.T
        y0, y1, m0, m1 = y[:-1], y[1:], m[:-1] * h, m[1:] * h
        out.append(torch.stack([y0, m0, 3 * (y1 - y0) - 2 * m0 - m1, 2 * (y0 - y1) + m0 + m1], dim=1))
    return torch.stack(out)


def locate(r, grid):
    """(interval, t, 1/h) per length in ``r``'s dtype, as the kernels: the band by compares, then one multiply."""
    import torch

    iv = torch.zeros(r.shape, dtype=torch.long)
    t, inv = torch.zeros_like(r), torch.zeros_like(r)
    offset = 0
    for k, (a, b, n) in enumerate(grid):
        scale = torch.tensor(n / (b - a), dtype=r.dtype)
        mask = (r >= torch.tensor(a, dtype=r.dtype)) if k else torch.ones(r.shape, dtype=torch.bool)
        u = torch.clamp((r - torch.tensor(a, dtype=r.dtype)) * scale, min=0)
        i = torch.clamp(torch.floor(u).long(), 0, n - 1)
        iv, t = torch.where(mask, offset + i, iv), torch.where(mask, u - i.to(r.dtype), t)
        inv = torch.where(mask, scale, inv)
        offset += n
    return iv, t, inv


def evaluate(coef32, grid, r32, pair):
    """float32 ``(w, dw/dr)`` ``[n, Q]`` of the float32 spline at float32 lengths for pair indices ``pair``."""
    iv, t, inv = locate(r32, grid)
    t = t[:, None]
    c = coef32[pair, iv]
    value = c[:, 0] + t * (c[:, 1] + t * (c[:, 2] + t * c[:, 3]))
    slope = (c[:, 1] + t * (2 * c[:, 2] + 3 * t * c[:, 3])) * inv[:, None]
    return value, slope


class _Scorer:
    """The certificate's points, exact values and the float32 network's errors, per pair; scores per band."""

    def __init__(self, chain64, chain32, weight64, pairs, r_lo, r_max, points, physical_from, seed):
        import torch

        gen = torch.Generator().manual_seed(seed)
        r64 = torch.cat([torch.linspace(r_lo, r_max, points + 1, dtype=torch.float64)[:-1],
                         r_lo + (r_max - r_lo) * torch.rand(points, generator=gen, dtype=torch.float64)])
        self.r32 = r64.to(torch.float32)
        self.r64 = self.r32.to(torch.float64)
        self.regions = {"all": torch.ones_like(self.r64, dtype=torch.bool), "physical": self.r64 >= physical_from}
        weight32 = weight64.to(device=chain32.device, dtype=torch.float32)
        self.truth, self.scale, self.net = [], {}, {}
        for k, (s, t) in enumerate(pairs):
            with torch.no_grad():
                exact = [g[:, :-1].double() @ weight64.T for g in chain64.with_derivative(self.r64, s, t)]
                net = [(g[:, :-1].float() @ weight32.T).double()
                       for g in chain32.with_derivative(self.r32.to(chain32.device), s, t)]
            self.truth.append(exact)
            for j in range(2):
                for name, region in self.regions.items():
                    scale = exact[j][region].abs().amax(dim=0).clamp_min(1e-300)
                    self.scale[k, j, name] = scale
                    self.net[k, j, name] = ((net[j].cpu()[region] - exact[j][region]).abs() / scale).max().item()
        self.pairs = pairs

    def relative(self, coef32, grid):
        """{(pair, kind, region): per-point error over the network's region error}."""
        import torch

        out = {}
        for k in range(len(self.pairs)):
            got = evaluate(coef32, grid, self.r32, torch.full(self.r32.shape, k, dtype=torch.long))
            for j in range(2):
                diff = (got[j].double() - self.truth[k][j]).abs()
                for name in self.regions:
                    out[k, j, name] = (diff / self.scale[k, j, name]).amax(dim=1) / max(self.net[k, j, name], 1e-300)
        return out

    def certificate(self, relative):
        parts = {}
        for (k, j, name), v in relative.items():
            key = f"{('value', 'slope')[j]}/{name}"
            parts[key] = max(parts.get(key, 0.0), v[self.regions[name]].max().item())
        return dict(worst=max(parts.values()), parts=parts)

    def bands(self, relative, grid):
        """The worst score inside each band of ``grid``."""
        scores = []
        for a, b, _ in grid:
            inside = (self.r64 >= a) & (self.r64 < b)
            worst = 0.0
            for (k, j, name), v in relative.items():
                mask = inside & self.regions[name]
                if mask.any():
                    worst = max(worst, v[mask].max().item())
            scores.append(worst)
        return scores


def fit(chain64, chain32, weight, pairs, r_lo, r_max, points, physical_from, seed=0):
    """``(spline, certificate)``: ``spline = dict(grid=[(start, end, intervals)], coef=[P, I, 4, Q] float32)``
    on the bands of ``BAND_EDGES`` below ``r_max``, each refined until its score is at most ``TARGET``;
    the certificate as ``radial.certify``'s (``worst``, ``parts``) plus the band scores and rounds."""
    import math

    import torch

    weight64 = weight.detach().to(device="cpu", dtype=torch.float64)
    edges = [r_lo] + [e for e in BAND_EDGES if r_lo < e < r_max] + [r_max]
    base = (r_max - r_lo) / BASE_INTERVALS
    grid = [(a, b, max(1, math.ceil((b - a) / base - 1e-9))) for a, b in zip(edges, edges[1:])]
    scorer = _Scorer(chain64, chain32, weight64, pairs, r_lo, r_max, points, physical_from, seed)
    for rounds in range(1, ROUNDS + 1):
        coef = coefficients(chain64, weight64, pairs, grid).to(torch.float32)
        relative = scorer.relative(coef, grid)
        scores = scorer.bands(relative, grid)
        if all(s <= TARGET for s in scores):
            break
        grid = [(a, b, n if s <= TARGET else math.ceil(n * 1.05 * (s / TARGET) ** (1 / 3)))
                for (a, b, n), s in zip(grid, scores)]
    certificate = scorer.certificate(relative)
    certificate.update(bands=[dict(start=a, end=b, intervals=n, score=s) for (a, b, n), s in zip(grid, scores)],
                       rounds=rounds, intervals=sum(n for _, _, n in grid))
    return dict(grid=[(float(a), float(b), int(n)) for a, b, n in grid], coef=coef), certificate


def cache_key(table_key):
    """The spline's cache key from its interaction's table key (which covers the model, species, dtype, r_max and
    the radial module) and this module's source and constants."""
    constants = (BAND_EDGES, BASE_INTERVALS, TARGET, ROUNDS, SPLINE_FORMAT)
    parts = [table_key, repr(constants), hashlib.sha256(inspect.getsource(sys.modules[__name__]).encode()).hexdigest()]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _path(key):
    from fastmace import _cache

    return _cache.cache_dir(CACHE_KIND) / f"spline-{key}.pt"


def load_cached(key):
    """The cached ``(spline, certificate)`` of ``key``, or None."""
    import torch

    try:
        path = _path(key)
        if not path.is_file():
            return None
        entry = torch.load(path, map_location="cpu", weights_only=True)
        if entry.get("format") != SPLINE_FORMAT or entry.get("key") != key:
            return None
        grid = [(float(a), float(b), int(n)) for a, b, n in entry["grid"]]
        return dict(grid=grid, coef=entry["coef"]), entry["certificate"]
    except Exception:  # noqa: BLE001 - an unreadable cache entry is a miss
        return None


def store_cached(key, spline, certificate):
    """Write the entry atomically; a cache that cannot be written is skipped."""
    import os

    import torch

    try:
        path = _path(key)
        entry = {"format": SPLINE_FORMAT, "key": key, "certificate": certificate,
                 "grid": [list(g) for g in spline["grid"]], "coef": spline["coef"].detach().cpu()}
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save(entry, temporary)
        os.replace(temporary, path)
    except Exception:  # noqa: BLE001 - the spline is fitted either way
        return


__all__ = ["BAND_EDGES", "BASE_INTERVALS", "ROUNDS", "TARGET", "coefficients", "evaluate", "fit", "grid_knots",
           "locate"]
