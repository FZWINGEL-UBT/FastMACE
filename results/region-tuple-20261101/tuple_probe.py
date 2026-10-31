"""Falsifier (conductor, 1 November 2026): the combined node region's inputs passed through as a tuple instead of
``torch.cat`` in ``joined_inputs`` and ``split`` in ``region_forward``. The claim: about 23 kB per atom less
("gated_products_input_grads"). ``region_probe.py`` with the two functions of ``fastmace.rewrites.segments.modules``
replaced before the record is applied; the arguments are region_probe's.
    python tuple_probe.py --checkpoint PATH --structure NbCl5@5x5x12 --region combined --chunks 4 --output DIR
"""

import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chunked-region-20261031"))
import region_probe  # noqa: E402

from fastmace.rewrites.segments import modules  # noqa: E402


def joined_inputs(self, x, y, divisor=None):
    """The chain's inputs as they are: no copy into one buffer."""
    return (x, y, divisor)


def region_forward(self, node_feats, sc, node_attrs):
    """``modules.region_forward`` on the tuple: the same chunks and checkpoints, without the split."""
    import torch
    from torch.utils.checkpoint import checkpoint

    if isinstance(node_feats, torch.Tensor):  # the record's self-check passes the joined columns
        x, y, *rest = node_feats.split(self.region_widths, dim=1)
        divisor = rest[0] if rest else None
    else:
        x, y, divisor = node_feats
    body = self.region_body
    chunks = self.region_chunks
    if chunks <= 1:
        return checkpoint(body, x, y, sc, node_attrs, divisor, use_reentrant=False, preserve_rng_state=False)
    xs, ys = torch.tensor_split(x, chunks, dim=0), torch.tensor_split(y, chunks, dim=0)
    attrs = torch.tensor_split(node_attrs, chunks, dim=0)
    scs = torch.tensor_split(sc, chunks, dim=0) if sc is not None else (None,) * chunks
    divisors = torch.tensor_split(divisor, chunks, dim=0) if divisor is not None else (None,) * chunks
    outs = []
    for i in range(chunks):
        outs.append(checkpoint(body, xs[i], ys[i], scs[i], attrs[i], divisors[i], use_reentrant=False,
                               preserve_rng_state=False))
    return torch.cat(outs, dim=0)


modules.joined_inputs = joined_inputs
modules.region_forward = region_forward

if __name__ == "__main__":
    region_probe.main()
