"""``radial_tables`` (opt-6 OPT6-T05): certified per-species-pair Chebyshev tables for MH-1's radial chain.

In a factored, cutoff-folded MH-1 interaction, for a fixed (sender, receiver) species pair, everything from the
edge length ``r`` to the folded last radial layer's input ``[h c, c]`` and to the edge density
``tanh(density(r))^2 c`` is a function of ``r`` alone: the Bessel basis and Agnesi transform of the radial
embedding, the projected species embeddings, the first layers, the hidden LayerNorm/SiLU layers and the
cutoff. The record replaces that chain by piecewise Chebyshev tables (``fastmace.kernels.triton.radial_table``):

- **Fit (apply):** per interaction and ordered pair of the structure's species superset, in float64 on the CPU
  from copies of the model's own modules (the weights are the model's float32 weights). Degree ``DEGREE``
  pieces, a dyadic refinement of ``INITIAL_PIECES`` uniform pieces over ``[R_LO, r_max)``, shared by all pairs,
  split until every output's interpolation error at ``2 * DEGREE + 3`` test points is below ``TOLERANCE`` of its
  largest value on the piece (with a floor of ``FLOOR`` of its largest value on the domain).
- **Certificate (apply, held by the self-check):** on a dense grid the float32 table and the model's own float32
  radial chain are both compared with the float64 chain, at the per-edge tensor-product weights
  ``w = [h c, c] @ [W, b]^T``, their r-derivatives, the edge density and its r-derivative, each normalised per
  output, over the whole domain and over ``[R_PHYSICAL, r_max)``. The certificate's ratio is the table's error
  over the network's; the self-check fails, and rule 13 turns the record off, if any ratio exceeds 1.
- **Cache:** the fitted tables and their certificate are stored under ``FASTMACE_CACHE/radial-tables`` by
  ``cache_key`` (the model's state, the species set, dtype, ``r_max``, the constants and this module's source) and
  reused by later builds; the self-check still runs on cached tables, so a bad entry turns the record off.
- **Model:** the record sets each interaction's ``radial_table`` and replaces ``model.radial_embedding`` by
  ``LengthsEmbedding``, which hands the interactions the edge lengths ``[E, 1]`` and no cutoff. The folded forward
  takes the table path when its ``edge_feats`` have width 1 and the network path otherwise (the other records'
  self-checks feed random radial features).

Exactness class ``certified`` (SPEC §6.1, owner decision 61): float32 plans only; the planner refuses it in
float64 plans, whose rounding the certificate does not bound. Deterministic: no atomics.
"""

from __future__ import annotations

import copy

from fastmace.rewrites import memory as memory_terms
from fastmace.rewrites.registry import CaseErrors, CheckResult, Rewrite, RewriteError, Support

MACE_VERSIONS = ">=0.3.16,<0.3.17"
R_LO = 0.3  # Å: the domain's lower end, below any distance MD reaches; shorter edges are evaluated here
R_PHYSICAL = 1.0  # Å: the certificate's second region, where every MD edge lies
DEGREE = 10  # tiny MH-1-type model: 36 pieces, certificate 0.13 (degree 12: 24 pieces, the same ratios)
INITIAL_PIECES = 8
MAX_DEPTH = 9  # bins: INITIAL_PIECES * 2**MAX_DEPTH, so that every piece boundary is a bin boundary
TOLERANCE = 1e-7
FLOOR = 1e-6
# Float64 tables (float64 plans; owner request, 27 October 2026): the same fit held to float64 rounding.
DEGREE_64 = 16
TOLERANCE_64 = 3e-13  # MH-1 NaCl, NbCl5: certificate <= 0.52 (1e-12 failed at 1.25-1.96)
FLOOR_64 = 1e-1  # with TOLERANCE_64: within 3e-14 of each output's largest value on the domain
# Their certificate: the largest error, per output scale, over this bound (about 5,000 float64 ulps). The float32
# certificate's yardstick, the network's own rounding error, is zero when the float64 network is the reference.
CERTIFY_BOUND_64 = 1e-12
# Their domain's lower end: exact in float32, as Triton passes the kernels' scalar bounds in float32 (0.3 rounds
# up, and an edge at exactly 0.3 Å then reads as below the domain, with a zero slope).
R_LO_64 = 0.25
# The installed tables' piece count is rounded up to a multiple of this (setup-across-compositions F1). The table
# shapes are static in the compiled graph, so compositions whose fits differ by a few pieces (NbCl5 53, TaCl5 52)
# generated different code; padded, they share it. The extra pieces repeat the last one and no bin maps to them, so
# every result is bitwise the same. 1: no padding (A/Bs).
PIECE_BUCKET = 16
CERTIFY_POINTS = 6000  # uniform and as many random lengths; the w-level tensors are [12000, 1280] float64
FIXED_SPECIES_SUPERSET = "fixed_species_superset"


def _factored(model):
    from fastmace.model.capabilities import FACTORED_INTERACTIONS

    return [(index, interaction) for index, interaction in enumerate(getattr(model, "interactions", ()))
            if type(interaction).__name__ in FACTORED_INTERACTIONS]


def _support(model):
    import torch

    from fastmace.rewrites import mh1

    verdict = mh1.RECORDS["fold_cutoff"].applies_to(model)
    if not verdict.supported:
        return verdict
    if len(_factored(model)) != len(model.interactions):
        return Support(False, "every interaction must be a factored MH-1 interaction (the embedding is shared)")
    parameter = next(model.parameters(), None)
    if parameter is not None and parameter.dtype not in (torch.float32, torch.float64):
        # float32 tables (decision 61) and float64 ones held to float64 rounding (f64-T01).
        return Support(False, f"float32 or float64 models only, not {parameter.dtype}")
    return verdict


# --------------------------------------------------------------------------- modules


def _lengths_embedding(original):
    import torch

    class LengthsEmbedding(torch.nn.Module):
        """``radial_tables``: the radial embedding's place in MACE's forward, returning the edge lengths and no
        cutoff (the tables apply it). ``apply_cutoff`` stays False, as the folded model's embedding has it."""

        def __init__(self, original):
            super().__init__()
            self.original = original
            self.apply_cutoff = False
            self.out_dim = getattr(original, "out_dim", None)

        def forward(self, edge_lengths, node_attrs, edge_index, atomic_numbers):
            return edge_lengths, None

    return LengthsEmbedding(original)


def _padded(tables):
    """``tables`` with the pieces rounded up to a multiple of ``PIECE_BUCKET`` by repeating the last piece (``coef``
    and ``dcoef`` along their piece axis, ``starts`` and ``scales``); ``bins`` still maps only to the real pieces."""
    import torch

    pieces = tables["coef"].shape[1]
    extra = -pieces % max(int(PIECE_BUCKET), 1)
    if not extra:
        return tables

    def repeat_last(tensor, axis):
        last = tensor.narrow(axis, tensor.shape[axis] - 1, 1)
        return torch.cat([tensor, last.expand(*[extra if i == axis else n for i, n in enumerate(tensor.shape)])],
                         dim=axis).contiguous()

    return dict(tables, coef=repeat_last(tables["coef"], 1), dcoef=repeat_last(tables["dcoef"], 1),
                starts=repeat_last(tables["starts"], 0), scales=repeat_last(tables["scales"], 0))


def _table_module(tables, columns, device):
    import torch

    pieces = tables["coef"].shape[1]
    tables = _padded(tables)

    class RadialTableModule(torch.nn.Module):
        """One interaction's tables: ``forward(lengths, node_attrs, edge_index) -> (folded input, edge density)``."""

        def __init__(self):
            super().__init__()
            for name in ("coef", "dcoef", "starts", "scales", "bins"):
                self.register_buffer(name, tables[name].to(device), persistent=False)
            self.register_buffer("columns", torch.as_tensor(columns, dtype=torch.long, device=device),
                                 persistent=False)
            self.species = len(columns)
            self.pieces = pieces  # fitted; coef.shape[1] is padded to PIECE_BUCKET
            self.r_lo, self.bin_scale, self.r_max = tables["r_lo"], tables["bin_scale"], tables["r_max"]
            self.fit_error = tables["fit_error"]
            self.certificate = {}

        def pair(self, node_attrs, edge_index):
            """Per edge the table's (sender species, receiver species) index (int32)."""
            species = node_attrs[:, self.columns].argmax(dim=1).to(torch.int32)
            return species[edge_index[0]] * self.species + species[edge_index[1]]

        def forward(self, lengths, node_attrs, edge_index):
            from fastmace.kernels.triton.radial_table import radial_table

            return radial_table(lengths, self.pair(node_attrs, edge_index), self.coef, self.dcoef, self.starts,
                                self.scales, self.bins, self.r_lo, self.bin_scale, self.r_max)

        def extra_repr(self):
            return (f"pairs={self.coef.shape[0]}, pieces={self.pieces} (padded to {self.coef.shape[1]}), "
                    f"degree={self.coef.shape[2] - 1}, r=[{self.r_lo}, {self.r_max})")

    return RadialTableModule()


# --------------------------------------------------------------------------- the chain


class _Chain:
    """One interaction's radial chain ``r -> [h c, c, density]`` per species pair, in one dtype and device.

    ``radial`` is the radial embedding (default ``model.radial_embedding``); the networks are copies with
    PyTorch's own hidden layers, so forward-mode derivatives exist."""

    def __init__(self, model, interaction, columns, dtype, device, radial=None):
        import torch

        self.dtype, self.device = dtype, device
        self.elements = len(model.atomic_numbers)
        self.atomic_numbers = model.atomic_numbers.to(device)
        self.columns = list(columns)
        self.radial = copy.deepcopy(radial if radial is not None else model.radial_embedding).to(
            device=device, dtype=dtype)
        nets = []
        for net in (interaction.conv_tp_weights, interaction.density_fn):
            net = copy.deepcopy(net).to(device=device, dtype=dtype)
            net.fused_hidden = None
            nets.append(net)
        self.conv, self.density = nets
        parameter = next(interaction.parameters())
        onehot = torch.zeros((len(self.columns), self.elements), dtype=parameter.dtype, device=parameter.device)
        onehot[torch.arange(len(self.columns)), torch.as_tensor(self.columns)] = 1
        with torch.no_grad():
            self.source = interaction.source_embedding(onehot).to(device=device, dtype=dtype)
            self.target = interaction.target_embedding(onehot).to(device=device, dtype=dtype)
        self.folded = interaction.conv_tp_weights.folded_weight().detach().to(device=device, dtype=dtype)

    def __call__(self, r, sender, receiver):
        """``[n, Q_in + 1]``: the folded input ``[h c, c]`` and the edge density for edges ``sender -> receiver``
        (superset indices) of lengths ``r`` (``[n]``)."""
        import torch

        n = r.shape[0]
        node_attrs = torch.zeros((2, self.elements), dtype=self.dtype, device=self.device)
        node_attrs[0, self.columns[sender]] = 1
        node_attrs[1, self.columns[receiver]] = 1
        edge_index = torch.stack([torch.zeros(n, dtype=torch.long, device=self.device),
                                  torch.ones(n, dtype=torch.long, device=self.device)])
        feats, cutoff = self.radial(r[:, None], node_attrs, edge_index, self.atomic_numbers)
        inputs = torch.cat([feats, self.source[sender].expand(n, -1), self.target[receiver].expand(n, -1)], dim=1)
        folded = self.conv.augmented(inputs, cutoff)
        density = torch.tanh(self.density(inputs) ** 2) * cutoff
        return torch.cat([folded, density], dim=1)

    def with_derivative(self, r, sender, receiver):
        import torch

        return torch.func.jvp(lambda x: self(x, sender, receiver), (r,), (torch.ones_like(r),))


# --------------------------------------------------------------------------- fit


def _nodes(count, dtype):
    import torch

    k = torch.arange(count, dtype=dtype)
    return torch.cos(torch.pi * (k + 0.5) / count)


def _coefficients(values):
    """Chebyshev coefficients ``[..., T, J]`` of values at the ``T`` Chebyshev nodes (axis -2)."""
    import torch

    count = values.shape[-2]
    k = torch.arange(count, dtype=values.dtype)
    basis = torch.cos(torch.pi * torch.outer(k, k + 0.5) / count) * (2.0 / count)  # [term, node]
    basis[0] /= 2.0
    return torch.einsum("tn,...nj->...tj", basis, values)


def _series(coef, x):
    """Clenshaw sum of ``coef [..., T, J]`` at ``x [..., M]``: ``[..., M, J]``."""
    import torch

    b1 = torch.zeros((*x.shape, coef.shape[-1]), dtype=coef.dtype)
    b2 = torch.zeros_like(b1)
    for k in range(coef.shape[-2] - 1, 0, -1):
        b1, b2 = coef[..., k:k + 1, :] + 2.0 * x[..., None] * b1 - b2, b1
    return coef[..., 0:1, :] + x[..., None] * b1 - b2


def _truth(chain, pairs, r, derivative=False):
    """``[P, n, J]`` values of ``chain`` (and their r-derivatives, with ``derivative``) for every pair."""
    import torch

    with torch.no_grad():
        if not derivative:
            return torch.stack([chain(r, s, t) for s, t in pairs])
        both = [chain.with_derivative(r, s, t) for s, t in pairs]
        return torch.stack([v for v, _ in both]), torch.stack([d for _, d in both])


def fit(chain, pairs, r_max, table_dtype=None):
    """The tables of ``chain`` for ``pairs`` (float64 fit; float32 tables, or float64 ones with
    ``table_dtype=torch.float64``): a dict of tensors and constants.

    A piece is split until both its values and its r-derivatives (the derivative series) are within
    ``TOLERANCE`` (float64 tables: ``TOLERANCE_64``) of the local largest value or derivative at the test points."""
    import numpy as np
    import torch

    dtype = torch.float64
    table_dtype = torch.float32 if table_dtype is None else table_dtype
    wide = table_dtype == torch.float64
    degree, tolerance, floor_factor = (DEGREE_64, TOLERANCE_64, FLOOR_64) if wide else (DEGREE, TOLERANCE, FLOOR)
    terms = degree + 1
    r_lo = R_LO_64 if wide else R_LO
    domain = torch.linspace(r_lo, r_max, 4097, dtype=dtype)[:-1]
    values, slopes = _truth(chain, pairs, domain, derivative=True)
    floor = floor_factor * values.abs().amax(dim=1, keepdim=True)  # [P, 1, J]
    dfloor = floor_factor * slopes.abs().amax(dim=1, keepdim=True)
    nodes, tests = _nodes(terms, dtype), torch.linspace(-1.0, 1.0, 2 * terms + 1, dtype=dtype)
    points = torch.cat([nodes, tests])
    unit = (r_max - r_lo) / INITIAL_PIECES
    pending = [(r_lo + i * unit, r_lo + (i + 1) * unit, 0) for i in range(INITIAL_PIECES)]
    accepted, worst = [], 0.0
    while pending:
        starts = torch.tensor([a for a, _, _ in pending], dtype=dtype)
        widths = torch.tensor([b - a for a, b, _ in pending], dtype=dtype)
        r = (starts[:, None] + (points[None, :] + 1.0) * widths[:, None] / 2.0).reshape(-1)
        values, slopes = _truth(chain, pairs, r, derivative=True)
        shape = (len(pairs), len(pending), len(points), -1)
        values, slopes = values.reshape(shape), slopes.reshape(shape)  # [P, pieces, points, J]
        coef = _coefficients(values[:, :, :terms])
        if wide:  # float64: the derivative series from the exact derivatives, as differentiating the value series
            dcoef = _coefficients(slopes[:, :, :terms])  # amplifies its rounding by ~degree^2 / width (f64-T01)
        else:
            dcoef = torch.from_numpy(np.polynomial.chebyshev.chebder(coef.numpy(), axis=2)) * (
                2.0 / widths)[None, :, None, None]
        x = tests.expand(len(pairs), len(pending), -1)
        error = (_series(coef, x) - values[:, :, terms:]).abs() / torch.maximum(
            values.abs().amax(dim=2, keepdim=True), floor[:, None])
        derror = (_series(dcoef, x) - slopes[:, :, terms:]).abs() / torch.maximum(
            slopes.abs().amax(dim=2, keepdim=True), dfloor[:, None])
        relative = torch.maximum(error.amax(dim=(0, 2, 3)), derror.amax(dim=(0, 2, 3)))  # per piece
        following = []
        for (a, b, level), err, piece_coef, piece_dcoef in zip(pending, relative.tolist(), coef.unbind(1),
                                                               dcoef.unbind(1)):
            if err > tolerance and level < MAX_DEPTH:
                middle = 0.5 * (a + b)
                following += [(a, middle, level + 1), (middle, b, level + 1)]
            else:
                accepted.append((a, b, piece_coef, piece_dcoef))
                worst = max(worst, err)
        pending = following
    accepted.sort(key=lambda item: item[0])
    coef = torch.stack([c for _, _, c, _ in accepted], dim=1)  # [P, K, T, J]
    scales = torch.tensor([2.0 / (b - a) for a, b, _, _ in accepted], dtype=dtype)
    if wide:
        dcoef = torch.stack([d for _, _, _, d in accepted], dim=1)
    else:
        dcoef = torch.from_numpy(np.polynomial.chebyshev.chebder(coef.numpy(), axis=2)) * scales[None, :, None, None]
    starts = torch.tensor([a for a, _, _, _ in accepted], dtype=dtype)
    n_bins = INITIAL_PIECES * 2 ** MAX_DEPTH
    centres = r_lo + (torch.arange(n_bins, dtype=dtype) + 0.5) * (r_max - r_lo) / n_bins
    bins = (torch.searchsorted(starts, centres, right=True) - 1).to(torch.int32)
    return dict(coef=coef.to(table_dtype).contiguous(), dcoef=dcoef.to(table_dtype).contiguous(),
                starts=starts.to(table_dtype), scales=scales.to(table_dtype), bins=bins,
                r_lo=float(r_lo), bin_scale=float(n_bins / (r_max - r_lo)), r_max=float(r_max), fit_error=worst)


# --------------------------------------------------------------------------- certificate


def _table_values(table, r, pair):
    """``(values, r-derivatives)`` ``[n, J]`` of the float32 table, by the arithmetic that will run: the Triton
    kernels on CUDA (the derivatives column by column through the backward with one-hot gradients), else
    ``radial_table.evaluate``."""
    import torch

    from fastmace.kernels.triton.radial_table import evaluate, radial_table_backward, radial_table_forward

    args = (table.starts, table.scales, table.bins)
    constants = (table.r_lo, table.bin_scale, table.r_max)
    if not r.is_cuda:
        return (evaluate(r, pair, table.coef, *args, *constants),
                evaluate(r, pair, table.dcoef, *args, *constants, derivative=True))
    g, density = radial_table_forward(r, pair, table.coef, *args, *constants)
    width = table.coef.shape[-1]
    slopes = []
    for column in range(width):
        grad_g = torch.zeros_like(g)
        grad_density = torch.zeros_like(density)
        if column < width - 1:
            grad_g[:, column] = 1.0
        else:
            grad_density[:, 0] = 1.0
        slopes.append(radial_table_backward(r, pair, table.dcoef, *args, grad_g, grad_density, *constants).reshape(-1))
    return torch.cat([g, density], dim=1), torch.stack(slopes, dim=1)


def certify(table, chain64, chain32, pairs, r_max, seed=0):
    """Per pair, the float32 table's error over the float32 chain's, both against the float64 chain (module
    docstring). Returns ``{"worst": ratio, "pairs": {"s-t": ratio}, ...}`` (superset indices)."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    r_lo = table.r_lo
    r64 = torch.cat([torch.linspace(r_lo, r_max, CERTIFY_POINTS + 1, dtype=torch.float64)[:-1],
                     r_lo + (r_max - r_lo) * torch.rand(CERTIFY_POINTS, generator=generator, dtype=torch.float64)])
    r64 = r64.to(table.coef.dtype).to(torch.float64)  # the lengths the GPU sees (float32 for float32 tables)
    physical = r64 >= R_PHYSICAL
    folded64 = chain64.folded
    r32 = r64.to(device=table.coef.device, dtype=table.coef.dtype)
    wide = table.coef.dtype == torch.float64
    out, worst, largest = {}, 0.0, 0.0

    def to_w(g):
        g = g.detach().to(torch.float64).cpu()
        return torch.cat([g[:, :-1] @ folded64.T, g[:, -1:]], dim=1)  # [n, Q + 1]: w and the density

    for index, (s, t) in enumerate(pairs):
        truth, dtruth = chain64.with_derivative(r64, s, t)
        net, dnet = chain32.with_derivative(r32, s, t)
        pair = torch.full(r32.shape, index, dtype=torch.int32, device=r32.device)
        tab, dtab = _table_values(table, r32, pair)
        ratios = []
        for exact, network, tabled in ((truth, net, tab), (dtruth, dnet, dtab)):
            exact, network, tabled = to_w(exact), to_w(network), to_w(tabled)
            for region in (slice(None), physical):
                e = exact[region]
                scale = e.abs().amax(dim=0).clamp_min(1e-300)
                net_error = (network[region] - e).abs() / scale
                tab_error = (tabled[region] - e).abs() / scale
                for columns in (slice(0, -1), slice(-1, None)):  # w, then the density
                    err_net, err_tab = net_error[:, columns].max().item(), tab_error[:, columns].max().item()
                    largest = max(largest, err_tab)
                    if wide:  # the float64 network is the reference itself: a fixed bound instead (f64-T01)
                        ratios.append(err_tab / CERTIFY_BOUND_64)
                    else:
                        ratios.append(0.0 if err_tab == 0.0 else (err_tab / err_net if err_net > 0 else float("inf")))
        out[f"{s}-{t}"] = max(ratios)
        worst = max(worst, out[f"{s}-{t}"])
    return {"worst": worst, "pairs": out, "points": int(r64.numel()), "r": [r_lo, r_max],
            "physical_from": R_PHYSICAL, "largest_error": largest,
            "bound": "CERTIFY_BOUND_64 of each output's scale" if wide else "the float32 network's error"}


# --------------------------------------------------------------------------- apply and self-check


def _superset(model, context):
    import torch

    columns = (context.inputs or {}).get("species_columns")
    if columns is None:
        return list(range(len(model.atomic_numbers)))
    return [int(c) for c in (columns.tolist() if isinstance(columns, torch.Tensor) else columns)]


# --------------------------------------------------------------------------- the table cache

CACHE_KIND = "radial-tables"
CACHE_FORMAT = 1
TABLE_TENSORS = ("coef", "dcoef", "starts", "scales", "bins")
TABLE_SCALARS = ("r_lo", "bin_scale", "r_max", "fit_error")


def _state_digest(model):
    """SHA-256 over every parameter and buffer of ``model`` (names, dtypes, shapes, bytes): the tables read the
    model's own modules, so anything in its state may change them."""
    import hashlib

    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict(keep_vars=False).items()):
        tensor = tensor.detach()
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.view(torch.int16)
        digest.update(tensor.contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def cache_key(model, index, columns, dtype, r_max, state_digest=None):
    """The cache key of interaction ``index``'s tables: the model's state, the species set (atomic numbers, in
    superset order), the tables' dtype, ``r_max``, every fit and certificate constant, this module's source and the
    torch and FastMACE versions. Anything else that changes the tables changes one of these."""
    import hashlib
    import inspect
    import sys

    import torch

    import fastmace

    numbers = [int(model.atomic_numbers[c]) for c in columns]
    constants = (R_LO, R_PHYSICAL, DEGREE, INITIAL_PIECES, MAX_DEPTH, TOLERANCE, FLOOR, DEGREE_64, TOLERANCE_64,
                 FLOOR_64, CERTIFY_BOUND_64, CERTIFY_POINTS, CACHE_FORMAT)
    parts = [state_digest or _state_digest(model), str(index), repr(numbers), str(dtype), repr(float(r_max)),
             repr(constants), hashlib.sha256(inspect.getsource(sys.modules[__name__]).encode()).hexdigest(),
             torch.__version__, fastmace.__version__]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _cache_path(key):
    from fastmace import _cache

    return _cache.cache_dir(CACHE_KIND) / f"{key}.pt"


def _load_cached(key):
    """The cached ``(tables, certificate)`` of ``key``, or None (absent, unreadable or of another format)."""
    import torch

    try:
        path = _cache_path(key)
        if not path.is_file():
            return None
        entry = torch.load(path, map_location="cpu", weights_only=True)
        if entry.get("format") != CACHE_FORMAT or entry.get("key") != key:
            return None
        tables = {name: entry["tables"][name] for name in TABLE_TENSORS}
        tables.update({name: float(entry["tables"][name]) for name in TABLE_SCALARS})
        return tables, entry["certificate"]
    except Exception:  # a damaged or foreign file is a miss; the fit replaces it
        return None


def _store_cached(key, tables, certificate):
    """Write the entry atomically (temporary file, then rename), so processes sharing the cache never read half a
    file. A cache that cannot be written is skipped: the tables are already fitted."""
    import os

    import torch

    try:
        path = _cache_path(key)
        entry = {"format": CACHE_FORMAT, "key": key, "certificate": certificate,
                 "tables": {**{name: tables[name].detach().cpu() for name in TABLE_TENSORS},
                            **{name: float(tables[name]) for name in TABLE_SCALARS}}}
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save(entry, temporary)
        os.replace(temporary, path)
    except OSError:
        pass


def tabulate_radials(model, context):
    """Fit, certify and install the tables of every interaction of ``model``; returns the applied names."""
    import torch

    from fastmace.rewrites import mh1

    parameter = next(model.parameters())
    if parameter.dtype not in (torch.float32, torch.float64):
        raise RewriteError(f"radial_tables: float32 or float64 models only, not {parameter.dtype}")
    factored = _factored(model)
    if not factored or len(factored) != len(model.interactions):
        raise RewriteError("radial_tables: every interaction must be a factored MH-1 interaction")
    for index, interaction in factored:
        if getattr(interaction.forward, "__func__", None) is not mh1.modules.cutoff_folded_forward:
            raise RewriteError(f"radial_tables requires the folded forward (fold_cutoff) in interactions.{index}")
    columns = _superset(model, context)
    pairs = [(s, t) for s in range(len(columns)) for t in range(len(columns))]
    r_max = float(model.r_max)
    names = []
    digest = _state_digest(model)  # before any interaction gets its table: every key sees the same state
    for index, interaction in factored:
        key = cache_key(model, index, columns, parameter.dtype, r_max, state_digest=digest)
        cached = _load_cached(key)
        if cached is not None:
            tables, certificate = cached
            source = "cached"
        else:
            chain64 = _Chain(model, interaction, columns, torch.float64, torch.device("cpu"))
            chain32 = _Chain(model, interaction, columns, parameter.dtype, parameter.device)
            tables = fit(chain64, pairs, r_max, table_dtype=parameter.dtype)
            certificate = certify(_table_module(tables, columns, parameter.device), chain64, chain32, pairs, r_max,
                                  seed=index)
            _store_cached(key, tables, certificate)
            source = "fitted"
        table = _table_module(tables, columns, parameter.device)
        table.certificate = certificate
        interaction.radial_table = table
        names.append(f"interactions.{index} ({table.pieces} pieces, {len(pairs)} pairs, "
                     f"certificate {table.certificate['worst']:.2f}, {source})")
    model.radial_embedding = _lengths_embedding(model.radial_embedding)
    return names


def _selfcheck(original, rewritten, context):
    """The certificate (every ratio at most 1) and, per interaction, the table kernels against the model's float32
    chain at random lengths of ``[R_PHYSICAL, r_max)``: values at the w level, the edge density and the r-gradient
    for random output gradients, held to SPEC §6.4's float32 limit."""
    import torch

    from fastmace.kernels.triton.radial_table import radial_table_backward, radial_table_forward

    cases, failures, dtype = CaseErrors(), [], None
    generator = torch.Generator().manual_seed(context.seed + 11)
    radial = getattr(rewritten.radial_embedding, "original", None)
    for index, interaction in _factored(rewritten):
        table = getattr(interaction, "radial_table", None)
        if table is None or radial is None:
            failures.append(f"interactions.{index}: no table")
            continue
        if not table.certificate.get("worst", float("inf")) <= 1.0:
            failures.append(f"interactions.{index}: certificate {table.certificate.get('worst')} > 1 "
                            f"(pairs {table.certificate.get('pairs')})")
        device = table.coef.device
        dtype = table.coef.dtype
        chain = _Chain(rewritten, interaction, table.columns.tolist(), dtype, device, radial=radial)
        r = (R_PHYSICAL + (table.r_max - R_PHYSICAL) * torch.rand(257, generator=generator, dtype=dtype)).to(device)
        for number in range(table.species ** 2):
            s, t = divmod(number, table.species)
            net, dnet = chain.with_derivative(r, s, t)
            pair = torch.full((r.numel(),), number, dtype=torch.int32, device=device)
            g, density = radial_table_forward(r, pair, table.coef, table.starts, table.scales, table.bins,
                                              table.r_lo, table.bin_scale, table.r_max)
            name = f"interactions.{index}.{s}-{t}"
            cases.add(f"{name}.w", net[:, :-1] @ chain.folded.T, g @ chain.folded.T)
            cases.add(f"{name}.density", net[:, -1:], density)
            weights = torch.randn((r.numel(), net.shape[1]), generator=generator, dtype=dtype).to(device)
            slope = radial_table_backward(r, pair, table.dcoef, table.starts, table.scales, table.bins,
                                          weights[:, :-1].contiguous(), weights[:, -1:].contiguous(), table.r_lo,
                                          table.bin_scale, table.r_max)
            cases.add(f"{name}.dr", (weights * dnet).sum(dim=1), slope.reshape(-1))
    result = cases.result(dtype)
    if failures:
        return CheckResult(False, result.max_relative_error, result.threshold, result.modules,
                           "; ".join(failures), result.dtype)
    return result


def _apply(model, context):
    return tabulate_radials(model, context)


# --------------------------------------------------------------------------- w_tables (ideas-20261027 D2)

# The w-tables hold [pairs * pieces + 1, degree + 1, Q] values twice; above this many bytes the record does not
# apply (a size rule: superset pairs grow quadratically with the species).
W_TABLE_MAX_BYTES = 2 << 30


def attach_w_tables(model):
    """Per-piece w-tables (``fastmace.kernels.triton.w_table``) in every float64 interaction whose tables feed the
    chunked streaming convolution; returns their names. Reassociated interactions keep their own path."""
    import torch

    from fastmace.kernels.triton.w_table import WTable

    names = []
    for index, interaction in _factored(model):
        table = getattr(interaction, "radial_table", None)
        if table is None or getattr(interaction, "first_reassociation", None) is not None:
            continue
        if getattr(interaction, "streaming_kernels", None) is None or not getattr(interaction,
                                                                                 "chunked_backward_edges", 0):
            continue
        if table.coef.dtype != torch.float64:
            raise RewriteError(f"w_tables is float64 only, interactions.{index} has {table.coef.dtype} tables")
        weight = interaction.conv_tp_weights.folded_weight()
        pairs, pieces, terms, _ = table.coef.shape
        size = 2 * (pairs * pieces + 1) * terms * weight.shape[0] * 8
        if size > W_TABLE_MAX_BYTES:
            raise RewriteError(f"w_tables: interactions.{index} would hold {size / 2**30:.1f} GiB of tables "
                               f"(more than W_TABLE_MAX_BYTES)")
        interaction.w_table = WTable(table, weight)
        names.append(f"interactions.{index} ({pairs * pieces} groups, {size / 2**20:.0f} MiB)")
    if not names:
        raise RewriteError("w_tables: no interaction with float64 radial tables and the chunked streaming "
                           "convolution")
    return names


def _w_selfcheck(original, rewritten, context):
    """Per interaction and pair, the w-table's values and r-slopes against the chain it replaces (the table's
    ``a`` times the folded weight), at random lengths of ``[R_PHYSICAL, r_max)``, to SPEC §6.4's float64 limit."""
    import torch

    from fastmace.kernels.triton.radial_table import radial_table_backward, radial_table_forward
    from fastmace.kernels.triton.w_table import w_table_backward, w_table_forward

    cases, dtype = CaseErrors(), None
    generator = torch.Generator().manual_seed(context.seed + 13)
    for index, interaction in _factored(rewritten):
        wt = getattr(interaction, "w_table", None)
        if wt is None:
            continue
        table = interaction.radial_table
        device, dtype = table.coef.device, table.coef.dtype
        weight = interaction.conv_tp_weights.folded_weight().to(dtype)
        r = (R_PHYSICAL + (table.r_max - R_PHYSICAL) * torch.rand(257, generator=generator, dtype=dtype)).to(device)
        for number in range(table.species ** 2):
            pair = torch.full((r.numel(),), number, dtype=torch.int32, device=device)
            a, _ = radial_table_forward(r, pair, table.coef, table.starts, table.scales, table.bins, table.r_lo,
                                        table.bin_scale, table.r_max)
            w, xs, gs, perm, slope = w_table_forward(r, pair, wt.g, wt.starts, wt.scales, wt.bins, wt.r_lo,
                                                     wt.bin_scale, wt.r_max)
            name = f"interactions.{index}.pair{number}"
            cases.add(f"{name}.w", a @ weight.T, w)
            dw = torch.randn(w.shape, generator=generator, dtype=dtype).to(device)
            da = dw @ weight
            chain = radial_table_backward(r, pair, table.dcoef, table.starts, table.scales, table.bins,
                                          da.contiguous(), torch.zeros_like(da[:, :1]), table.r_lo, table.bin_scale,
                                          table.r_max).reshape(-1)
            cases.add(f"{name}.dr", chain, w_table_backward(dw, wt.dg, xs, gs, perm, slope))
    return cases.result(dtype)


def _w_support(model):
    return _support(model)


def _apply_w_tables(model, context):
    return attach_w_tables(model)


# --------------------------------------------------------------------------- wprime_backward (opt-9 OPT9-T03)


def attach_wprime(model):
    """``wprime_backward`` (``fastmace.kernels.oeq_streaming.wprime_backward``) in every float32 interaction whose
    tables feed the chunked streaming convolution; returns their names. Reassociated interactions keep their own
    path."""
    import torch

    from fastmace.kernels.oeq_streaming.wprime_backward import WPrime

    names = []
    for index, interaction in _factored(model):
        table = getattr(interaction, "radial_table", None)
        if table is None or getattr(interaction, "first_reassociation", None) is not None:
            continue
        kernels = getattr(interaction, "streaming_kernels", None)
        if kernels is None or not getattr(interaction, "chunked_backward_edges", 0):
            continue
        if table.coef.dtype != torch.float32:
            raise RewriteError(f"wprime_backward is float32 only, interactions.{index} has {table.coef.dtype} tables")
        if table.coef.device.type != "cuda":
            raise RewriteError(f"wprime_backward needs the model on a CUDA device (interactions.{index})")
        weight = interaction.conv_tp_weights.folded_weight().detach()
        pairs, pieces, terms, _ = table.dcoef.shape
        size = pairs * pieces * terms * weight.shape[0] * table.dcoef.element_size()
        if size > W_TABLE_MAX_BYTES:
            raise RewriteError(f"wprime_backward: interactions.{index} would hold {size / 2**30:.1f} GiB of tables "
                               f"(more than W_TABLE_MAX_BYTES)")
        interaction.wprime_backward = WPrime(table, weight, kernels)
        names.append(f"interactions.{index} ({pairs * pieces} groups, {terms} terms, {size / 2**20:.0f} MiB)")
    if not names:
        raise RewriteError("wprime_backward: no interaction with float32 radial tables and the chunked streaming "
                           "convolution")
    return names


def _wprime_case(interaction, generator, wprime, recompute_rows=0):
    """One random receiver-sorted graph through the interaction's table and folded streaming convolution, with
    the record (``wprime``) or the chain: ``[message, density, dL/dr, dx, dsh]``. A twelfth of the edges lie at or
    beyond the cutoff (the neighbour list's skin and padding)."""
    import torch

    from fastmace.kernels.oeq_streaming import folded_streaming_convolution
    from fastmace.kernels.triton.radial_table import radial_table

    table, kernels = interaction.radial_table, interaction.streaming_kernels
    radial = interaction.conv_tp_weights
    weight = radial.folded_weight().detach()
    device, dtype = table.coef.device, table.coef.dtype
    nodes, edges = 37, 720

    def rand(*shape):
        return torch.rand(shape, generator=generator, dtype=torch.float64).to(device=device, dtype=dtype)

    def randn(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float64).to(device=device, dtype=dtype)

    receivers = torch.sort(torch.randint(0, nodes, (edges,), generator=generator)).values.to(device)
    senders = torch.randint(0, nodes, (edges,), generator=generator).to(device)
    r = R_PHYSICAL + (table.r_max - R_PHYSICAL) * rand(edges, 1)
    r[::12] = table.r_max + 0.5 * rand(r[::12].shape[0], 1)
    pair = torch.randint(0, table.species ** 2, (edges,), generator=generator, dtype=torch.int32).to(device)
    x, sh = randn(nodes, kernels.input_dim), randn(edges, kernels.edge_dim)
    dout, dd = randn(nodes, kernels.output_dim), randn(edges, 1)
    row_ptr = torch.searchsorted(receivers, torch.arange(nodes + 1, device=device)).to(torch.int32)
    s32, r32 = senders.to(torch.int32), receivers.to(torch.int32)
    fused = getattr(interaction, "fused_folded_forward", None)
    with torch.enable_grad():
        leaves = [t.clone().requires_grad_() for t in (r, x, sh)]
        if wprime:
            message, density = interaction.wprime_backward.convolution(leaves[0], pair, table, weight, leaves[1],
                                                                       leaves[2], s32, r32, row_ptr, kernels, fused,
                                                                       recompute_rows)
        else:
            a, density = radial_table(leaves[0], pair, table.coef, table.dcoef, table.starts, table.scales,
                                      table.bins, table.r_lo, table.bin_scale, table.r_max)
            message = folded_streaming_convolution(a, weight, leaves[1], leaves[2], s32, r32, row_ptr, kernels,
                                                   interaction.chunked_backward_edges, radial.folded_tiles,
                                                   radial.folded_parts, fused)
        grads = torch.autograd.grad([message, density], leaves, [dout, dd])
    return [message.detach(), density.detach(), *grads]


def _wprime_selfcheck(original, rewritten, context):
    """Per interaction, the record's message, edge density and gradients to r, x and sh against the chain it
    replaces (the table, the folded streaming convolution and its chunked backward) on a random graph, to SPEC
    §6.4's float32 limit."""
    import torch

    cases, dtype = CaseErrors(), None
    for index, interaction in _factored(rewritten):
        if getattr(interaction, "wprime_backward", None) is None:
            continue
        dtype = interaction.radial_table.coef.dtype
        seed = context.seed + 17 + index
        chain = _wprime_case(interaction, torch.Generator().manual_seed(seed), wprime=False)
        got = _wprime_case(interaction, torch.Generator().manual_seed(seed), wprime=True)
        for name, a, b in zip(("message", "density", "dr", "dx", "dsh"), chain, got):
            cases.add(f"interactions.{index}.{name}", a, b)
    return cases.result(dtype)


def _apply_wprime(model, context):
    return attach_wprime(model)


# --------------------------------------------------------------------------- recompute_conv_weights (CAP-T07)

# Edges per chunk in the self-check: below its 720-edge graph, so the chunks and a partial last one run.
RECOMPUTE_CHECK_ROWS = 256
# recompute_conv_weights' kernel where fused_folded_forward is on: "rebuild" (the forward stores no w and the
# backward builds it on chip, plans/capacity.md CAP-T07b) or "chunked" (cuBLAS per edge chunk; same-process A/Bs).
RECOMPUTE_KERNEL = "rebuild"


def attach_recompute(model):
    """``recompute_conv_weights`` in every interaction with ``wprime_backward``: the per-edge weights (``[E, W]``) are
    not kept. With ``fused_folded_forward`` (and ``RECOMPUTE_KERNEL = "rebuild"``) the forward stores no ``w`` and the
    backward builds it on chip (``WPrime.attach_rebuild``); otherwise the backward builds it again with cuBLAS in
    chunks of the interaction's chunked-backward edges. Returns their names."""
    names = []
    for index, interaction in _factored(model):
        if getattr(interaction, "wprime_backward", None) is None:
            continue
        interaction.recompute_weight_rows = int(interaction.chunked_backward_edges)
        fused = getattr(interaction, "fused_folded_forward", None)
        if RECOMPUTE_KERNEL == "rebuild" and fused is not None:
            interaction.wprime_backward.attach_rebuild(interaction.streaming_kernels, fused)
            names.append(f"interactions.{index} (w built on chip in the backward)")
        else:
            names.append(f"interactions.{index} ({interaction.recompute_weight_rows} edges per chunk)")
    if not names:
        raise RewriteError("recompute_conv_weights: no interaction with wprime_backward")
    return names


def _recompute_selfcheck(original, rewritten, context):
    """Per interaction, the record's message, edge density and gradients against wprime_backward with the stored
    weights, on a random graph in chunks of ``RECOMPUTE_CHECK_ROWS`` edges."""
    import torch

    cases, dtype = CaseErrors(), None
    for index, interaction in _factored(rewritten):
        if not getattr(interaction, "recompute_weight_rows", 0):
            continue
        dtype = interaction.radial_table.coef.dtype
        seed = context.seed + 29 + index
        stored = _wprime_case(interaction, torch.Generator().manual_seed(seed), wprime=True)
        got = _wprime_case(interaction, torch.Generator().manual_seed(seed), wprime=True,
                           recompute_rows=RECOMPUTE_CHECK_ROWS)
        for name, a, b in zip(("message", "density", "dr", "dx", "dsh"), stored, got):
            cases.add(f"interactions.{index}.{name}", a, b)
    return cases.result(dtype)


def _apply_recompute(model, context):
    return attach_recompute(model)


# --------------------------------------------------------------------------- recompute_first_table (rebuild-a)


def attach_first_table(model):
    """``recompute_first_table`` in every interaction with a radial table and the reassociated first interaction's
    message (``modules.first_table_region``): the table's output is not kept for the backward. Returns their names."""
    names = []
    for index, interaction in _factored(model):
        if getattr(interaction, "radial_table", None) is None or getattr(interaction, "first_reassociation",
                                                                          None) is None:
            continue
        interaction.recompute_first_table = True
        names.append(f"interactions.{index}")
    if not names:
        raise RewriteError("recompute_first_table: no reassociated interaction with a radial table")
    return names


def _first_table_selfcheck(original, rewritten, context):
    """Per interaction with the record, its checkpointed region (``modules.recomputed_first_table``) against the table
    and the reassociation called directly: the message, the edge density and the gradients to the lengths and the
    spherical harmonics, on a random receiver-sorted graph of the superset's species with a random species table
    bound (the interaction's other modules are not run: species_tables reads its input as one-hot rows). A twelfth
    of the edges lie at or beyond the cutoff."""
    import torch

    from fastmace.rewrites.mh1 import modules as mm

    cases, dtype = CaseErrors(), None
    for index, interaction in _factored(rewritten):
        if not getattr(interaction, "recompute_first_table", False):
            continue
        generator = torch.Generator().manual_seed(context.seed + 53 + index)
        table, reassociation = interaction.radial_table, interaction.first_reassociation
        device, dtype = table.coef.device, table.coef.dtype
        nodes, edges = 37, 720

        def rand(*shape):
            return torch.rand(shape, generator=generator, dtype=torch.float64).to(device=device, dtype=dtype)

        def randn(*shape):
            return torch.randn(shape, generator=generator, dtype=torch.float64).to(device=device, dtype=dtype)

        columns = reassociation.columns.cpu()
        species = columns[torch.randint(0, len(columns), (nodes,), generator=generator)]
        attrs = torch.nn.functional.one_hot(species, interaction.node_attrs_irreps.dim).to(device=device, dtype=dtype)
        receivers = torch.sort(torch.randint(0, nodes, (edges,), generator=generator)).values
        senders = torch.randint(0, nodes, (edges,), generator=generator)
        edge_index = torch.stack([senders, receivers]).to(device)
        r = R_PHYSICAL + (table.r_max - R_PHYSICAL) * rand(edges, 1)
        r[::12] = table.r_max + 0.5 * rand(r[::12].shape[0], 1)
        sh = randn(edges, interaction.edge_attrs_irreps.dim)
        weight = interaction.conv_tp_weights.folded_weight().detach()
        own = reassociation.species_feats
        reassociation.bind(randn(len(columns), reassociation.channels))
        try:
            values = []
            grads = None
            for checkpointed in (False, True):
                with torch.enable_grad():
                    leaves = [t.clone().requires_grad_() for t in (r, sh)]
                    if checkpointed:
                        message, density = mm.recomputed_first_table(interaction, leaves[0], None, attrs, leaves[1],
                                                                     edge_index)
                    else:
                        folded, density = table(leaves[0], attrs, edge_index)
                        message = reassociation(None, attrs, leaves[1], folded, edge_index, weight)
                    if grads is None:
                        grads = [randn(*message.shape), randn(*density.shape)]
                    values.append([message.detach(), density.detach(),
                                   *torch.autograd.grad([message, density], leaves, grads)])
        finally:
            reassociation.bind(own)
        for name, a, b in zip(("message", "density", "dr", "dsh"), *values):
            cases.add(f"interactions.{index}.{name}", a, b)
    return cases.result(dtype)


def _apply_first_table(model, context):
    return attach_first_table(model)


RECORDS = {
    "radial_tables": Rewrite(
        name="radial_tables", applies_to=_support, requires=("fold_cutoff",), conflicts=(),
        mace_versions=MACE_VERSIONS, source_hashes={}, exactness="certified", default="auto", deterministic=True,
        workload_constraints=(FIXED_SPECIES_SUPERSET,), apply=_apply, selfcheck=_selfcheck,
        memory=memory_terms.radial_tables),
    # ideas-20261027 D2: rounding (the series and the folded product reassociated); float64; opt-in.
    "w_tables": Rewrite(
        name="w_tables", applies_to=_w_support, requires=("radial_tables", "chunked_conv_backward"), conflicts=(),
        mace_versions=MACE_VERSIONS, source_hashes={}, exactness="rounding", default="off", deterministic=False,
        workload_constraints=(FIXED_SPECIES_SUPERSET,), apply=_apply_w_tables, selfcheck=_w_selfcheck),
    # opt-9 OPT9-T03: rounding (the derivative series and the folded product reassociated); float32; default on
    # for MH-1 on the measured GPUs (decision 66, planner.fixed.DECISION_66_WPRIME_GPUS).
    "wprime_backward": Rewrite(
        name="wprime_backward", applies_to=_w_support, requires=("radial_tables", "chunked_conv_backward"),
        conflicts=(), mace_versions=MACE_VERSIONS, source_hashes={}, exactness="rounding", default="auto",
        deterministic=False, workload_constraints=(FIXED_SPECIES_SUPERSET,), apply=_apply_wprime,
        selfcheck=_wprime_selfcheck),
    # plans/capacity.md CAP-T07: rounding (the rebuilt weights equal the stored ones bitwise; dsh and dL/dr are summed
    # over channel slices in another order, the sender additions are atomic); off unless memory requires it
    # (planner.memory.MEMORY_RESCUE, fallback.memory_rescue).
    "recompute_conv_weights": Rewrite(
        name="recompute_conv_weights", applies_to=_w_support, requires=("wprime_backward",), conflicts=(),
        mace_versions=MACE_VERSIONS, source_hashes={}, exactness="rounding", default="off", deterministic=False,
        workload_constraints=(FIXED_SPECIES_SUPERSET,), apply=_apply_recompute, selfcheck=_recompute_selfcheck,
        memory=memory_terms.recompute_conv_weights),
    # Capacity ideas, rebuild-a (speed-def falsifier: about 0.45 % of the force call at NbCl5 9,216 atoms): exact
    # (the same table kernel evaluated again on the same lengths); off unless memory requires it, after
    # recompute_conv_weights (planner.memory.MEMORY_RESCUE, fallback.memory_rescue).
    "recompute_first_table": Rewrite(
        name="recompute_first_table", applies_to=_support, requires=("radial_tables", "reassociate_first"),
        conflicts=(), mace_versions=MACE_VERSIONS, source_hashes={}, exactness="exact", default="off",
        deterministic=True, workload_constraints=(FIXED_SPECIES_SUPERSET,), apply=_apply_first_table,
        selfcheck=_first_table_selfcheck, memory=memory_terms.recompute_first_table),
}

__all__ = ["DEGREE", "PIECE_BUCKET", "R_LO", "RECOMPUTE_CHECK_ROWS", "RECOMPUTE_KERNEL", "RECORDS", "W_TABLE_MAX_BYTES",
           "attach_first_table", "attach_recompute", "attach_w_tables", "attach_wprime", "certify", "fit",
           "tabulate_radials"]
