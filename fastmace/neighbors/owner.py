"""Mirror pairs and owner edges of a receiver-sorted edge list (ideas plan idea 14(A), the owner-pass forward).

Every edge i <- j (receiver i, sender j, image shift S: the vector r_i - r_j + S h) has a mirror j <- i with shift
-S and a bitwise-equal length. Each pair has one owner edge: i <- j owns when i + j is even and i < j, or odd and
i > j; a self-image pair (i = j) by the sign of its shift (the first non-zero component positive); an edge without a
mirror owns itself. Each receiver's owned edges are ordered first; a search on the key 2 receiver + (not owned) then
gives both the CSR ``row_ptr`` and each receiver's split ``mid`` (``split_rows``). ``fused_forward.OWNER_PASS``'s
pass 1 runs the owned ranges [row_ptr[i], mid[i]) and pass 2 the partner ranges [mid[i], row_ptr[i + 1]).

Device operations of fixed shapes only (no host synchronisation): ``SkinNeighbourList(owner_pairs=True)`` pairs its
candidates at every build, the asynchronous one included.
"""

from __future__ import annotations

import torch

SHIFT_BITS = 21  # per shift component in the pair sort's second key: |S| < 2^20 cells


def _shift_key(shifts):
    """An int64 key of integer image shifts [E, 3] (components within +-2^20)."""
    offset = 1 << (SHIFT_BITS - 1)
    s = shifts + offset
    return (s[:, 0] << (2 * SHIFT_BITS)) | (s[:, 1] << SHIFT_BITS) | s[:, 2]


def _positive(shifts):
    """Whether each shift's first non-zero component is positive."""
    x, y, z = shifts.unbind(1)
    return (x > 0) | ((x == 0) & (y > 0)) | ((x == 0) & (y == 0) & (z > 0))


def mirror_pairs(sender, receiver, shifts, valid, natoms):
    """Each edge's mirror (int64 index, -1: none) among the ``valid`` edges: one stable sort by the canonical shift,
    one by the atom pair. ``shifts``: integer image shifts [E, 3] (int64)."""
    edges = sender.shape[0]
    index = torch.arange(edges, device=sender.device)
    lo, hi = torch.minimum(sender, receiver), torch.maximum(sender, receiver)
    flip = (receiver > sender) | ((receiver == sender) & ~_positive(shifts))  # the canonical form is the lo <- hi edge
    canonical = torch.where(flip[:, None], -shifts, shifts)
    pair = torch.where(valid, lo * natoms + hi, natoms * natoms + index)  # invalid edges pair with none
    order = torch.sort(_shift_key(canonical), stable=True).indices
    order = order[torch.sort(pair[order], stable=True).indices]
    pk, sk = pair[order], _shift_key(canonical)[order]
    same = (pk[1:] == pk[:-1]) & (sk[1:] == sk[:-1])  # adjacent edges of one pair and image
    after = torch.zeros(edges, dtype=torch.bool, device=sender.device)
    before = torch.zeros_like(after)
    after[:-1], before[1:] = same, same
    position = torch.arange(edges, device=sender.device)
    partner = torch.where(after, position + 1, torch.where(before, position - 1, -1))
    mirror = torch.full((edges,), -1, dtype=torch.int64, device=sender.device)
    mirror[order] = torch.where(partner >= 0, order[partner.clamp(min=0)], -1)
    return mirror


def owner_flags(sender, receiver, shifts, mirror):
    """Bool per edge: the parity rule, the shift's sign for a self-image pair; an edge without a mirror owns itself."""
    odd = ((receiver + sender) & 1).bool()
    parity = (~odd & (receiver < sender)) | (odd & (receiver > sender))
    return torch.where(receiver == sender, _positive(shifts), parity) | (mirror < 0)


def owned_first_order(receiver, owned, valid=None, natoms=None):
    """The stable permutation that orders each receiver's owned edges first (receiver-sorted input stays sorted);
    edges outside ``valid`` go last in their order (key 2 ``natoms``)."""
    key = 2 * receiver + (~owned).to(receiver.dtype)
    if valid is not None:
        key = torch.where(valid, key, 2 * natoms)
    return torch.sort(key, stable=True).indices


def split_rows(receiver, owned, count, nodes):
    """(row_ptr [nodes + 1], mid [nodes]) as int32 of the first ``count`` edges (a device scalar), receiver-sorted
    with each receiver's owned edges first: one search on the key 2 receiver + (not owned)."""
    position = torch.arange(receiver.shape[0], device=receiver.device)
    key = torch.where(position < count, 2 * receiver.to(torch.int64) + (~owned).to(torch.int64),
                      torch.full_like(position, 2 * nodes))
    bounds = torch.searchsorted(key, torch.arange(2 * nodes + 1, device=key.device)).to(torch.int32)
    return bounds[0::2].contiguous(), bounds[1::2].contiguous()


def pair_candidates(sender, receiver, shifts, valid, natoms):
    """(order, owned, mirror) of candidates at a rebuild: ``order`` the owned-first permutation, then each permuted
    candidate's owner flag and its mirror's permuted index (int32, -1: none). Invalid candidates (``valid`` false)
    keep their place at the end, owned and without a mirror."""
    mirror = mirror_pairs(sender, receiver, shifts, valid, natoms)
    owned = owner_flags(sender, receiver, shifts, mirror) | ~valid
    order = owned_first_order(receiver, owned, valid, natoms)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.shape[0], device=order.device)
    moved = mirror[order]
    return order, owned[order], torch.where(moved >= 0, inverse[moved.clamp(min=0)], -1).to(torch.int32)


__all__ = ["SHIFT_BITS", "mirror_pairs", "owned_first_order", "owner_flags", "pair_candidates", "split_rows"]
