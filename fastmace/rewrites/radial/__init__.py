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
- **Model:** the record sets each interaction's ``radial_table`` and replaces ``model.radial_embedding`` by
  ``LengthsEmbedding``, which hands the interactions the edge lengths ``[E, 1]`` and no cutoff. The folded forward
  takes the table path when its ``edge_feats`` have width 1 and the network path otherwise (the other records'
  self-checks feed random radial features).

Exactness class ``certified`` (SPEC §6.1, owner decision 61): float32 plans only; the planner refuses it in
float64 plans, whose rounding the certificate does not bound. Deterministic: no atomics.
"""

from __future__ import annotations

import copy

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
CERTIFY_POINTS = 6000  # uniform and as many random lengths; the w-level tensors are [12000, 1280] float64
FIXED_SPECIES_SUPERSET = "fixed_species_superset"


def _factored(model):
    from fastmace.model.capabilities import FACTORED_INTERACTIONS

    return [(index, interaction) for index, interaction in enumerate(getattr(model, "interactions", ()))
            if type(interaction).__name__ in FACTORED_INTERACTIONS]


def _support(model):
    from fastmace.rewrites import mh1

    verdict = mh1.RECORDS["fold_cutoff"].applies_to(model)
    if not verdict.supported:
        return verdict
    if len(_factored(model)) != len(model.interactions):
        return Support(False, "every interaction must be a factored MH-1 interaction (the embedding is shared)")
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


def _table_module(tables, columns, device):
    import torch

    class RadialTableModule(torch.nn.Module):
        """One interaction's tables: ``forward(lengths, node_attrs, edge_index) -> (folded input, edge density)``."""

        def __init__(self):
            super().__init__()
            for name in ("coef", "dcoef", "starts", "scales", "bins"):
                self.register_buffer(name, tables[name].to(device), persistent=False)
            self.register_buffer("columns", torch.as_tensor(columns, dtype=torch.long, device=device),
                                 persistent=False)
            self.species = len(columns)
            self.r_lo, self.bin_scale, self.r_max = tables["r_lo"], tables["bin_scale"], tables["r_max"]
            self.fit_error = tables["fit_error"]
            self.certificate = {}

        def forward(self, lengths, node_attrs, edge_index):
            from fastmace.kernels.triton.radial_table import radial_table

            species = node_attrs[:, self.columns].argmax(dim=1).to(torch.int32)
            pair = species[edge_index[0]] * self.species + species[edge_index[1]]
            return radial_table(lengths, pair, self.coef, self.dcoef, self.starts, self.scales, self.bins,
                                self.r_lo, self.bin_scale, self.r_max)

        def extra_repr(self):
            return (f"pairs={self.coef.shape[0]}, pieces={self.coef.shape[1]}, degree={self.coef.shape[2] - 1}, "
                    f"r=[{self.r_lo}, {self.r_max})")

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
    for index, interaction in factored:
        chain64 = _Chain(model, interaction, columns, torch.float64, torch.device("cpu"))
        chain32 = _Chain(model, interaction, columns, parameter.dtype, parameter.device)
        table = _table_module(fit(chain64, pairs, r_max, table_dtype=parameter.dtype), columns, parameter.device)
        table.certificate = certify(table, chain64, chain32, pairs, r_max, seed=index)
        interaction.radial_table = table
        names.append(f"interactions.{index} ({table.coef.shape[1]} pieces, {len(pairs)} pairs, "
                     f"certificate {table.certificate['worst']:.2f})")
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


RECORDS = {
    "radial_tables": Rewrite(
        name="radial_tables", applies_to=_support, requires=("fold_cutoff",), conflicts=(),
        mace_versions=MACE_VERSIONS, source_hashes={}, exactness="certified", default="off", deterministic=True,
        workload_constraints=(FIXED_SPECIES_SUPERSET,), apply=_apply, selfcheck=_selfcheck),
}

__all__ = ["DEGREE", "R_LO", "RECORDS", "certify", "fit", "tabulate_radials"]
