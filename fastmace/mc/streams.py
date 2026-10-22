"""The MC random streams: counter-based Philox, so a number is a function of (seed, offset) (SPEC §8.5
"Random numbers per outcome"; plan task 3-T09, conductor answer 2).

Three streams (``config.MC_STREAMS``): proposal, acceptance and velocity, each seeded by
``config.mc_stream_seeds(seed)``. Number ``n`` of a stream with seed ``s`` is computed from

    Philox4x32-10(counter = (n mod 2**32, n >> 32, 0, 0), key = (s mod 2**32, s >> 32)) = (x0, x1, x2, x3)

(Salmon et al. 2011, Random123's ``philox4x32_10``; known-answer tests in ``tests/cpu/test_mc_streams.py``):
- a **uniform** in [0, 1) is ``((x0 >> 5) * 2**26 + (x1 >> 6)) / 2**53`` (53 random bits, exact in float64);
- a **normal** is ``sqrt(-2 log(1 - u)) cos(2 pi v)`` with ``u`` from (x0, x1) and ``v`` from (x2, x3) as above.

So the numbers are the same on the CPU and the GPU (the uniforms bit for bit), and a stream's whole
state is its seed and the offset of its next number. The offsets are the restart record
(``io.restart.MC_STREAM_KEYS``: seed, offset). Every attempt consumes a fixed count from each stream
(``attempt_counts``), whatever its outcome; attempt ``k`` of a move in a block reads the numbers
``base + k * count ... base + (k + 1) * count - 1`` of each stream, where ``base`` is the stream's offset
when the move's attempts began. The layout of those numbers (``decode_swap``, ``decode_transmute``,
``ACCEPTANCE_COUNT``, ``velocity_count``) is a restart compatibility promise
(``docs/design/contracts-3.md``, "MC streams").

Everything here is plain PyTorch on int64 and float64 tensors: device-side, fixed shapes, no host
synchronisation, so it runs inside a captured CUDA graph.
"""

from __future__ import annotations

import math

PHILOX_ROUNDS = 10
PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9  # the key schedule's Weyl constants
PHILOX_W1 = 0xBB67AE85
_M32 = 0xFFFFFFFF
_TWO_POW_M53 = 2.0 ** -53

# Numbers each stream gives one attempt (SPEC §8.5): fixed per move type and atom count.
ACCEPTANCE_COUNT = 1


def proposal_count(move: str, natoms: int) -> int:
    """Proposal numbers per attempt: a swap draws its pair-type uniform and two key vectors (2N + 1), a
    transmutation its new-species uniform and one key vector (N + 1)."""
    if move == "swap":
        return 2 * natoms + 1
    if move == "transmute":
        return natoms + 1
    raise ValueError(f"no proposal layout for {move!r} moves here (displacements: plan task 3-T10)")


def velocity_count(move: str, velocities: str) -> int:
    """Velocity numbers per attempt: ``redraw`` draws a normal per component of each changed atom (a swap
    changes two atoms, a transmutation one); the other treatments draw none per attempt. ``redraw_all``
    draws 3N per block instead (``block_velocity_count``)."""
    if velocities != "redraw":
        return 0
    return 6 if move == "swap" else 3


def block_velocity_count(velocities: str, natoms: int) -> int:
    """Velocity numbers a move consumes once per block after its attempts: 3N for ``redraw_all`` (used only
    after an acceptance), else none."""
    return 3 * natoms if velocities == "redraw_all" else 0


def attempt_counts(move: str, velocities: str, natoms: int) -> dict[str, int]:
    """The fixed counts per attempt, by stream (``config.MC_STREAMS`` order)."""
    return {"proposal": proposal_count(move, natoms), "acceptance": ACCEPTANCE_COUNT,
            "velocity": velocity_count(move, velocities)}


# --------------------------------------------------------------------------- Philox4x32-10


def _mulhilo(a, multiplier: int):
    """(hi, lo) 32-bit words of ``a * multiplier`` for int64 ``a`` in [0, 2**32): the multiplier is split
    into 16-bit halves so that no product leaves the int64 range."""
    low, high = multiplier & 0xFFFF, multiplier >> 16
    p_low = a * low  # < 2**48
    p_high = a * high  # < 2**48
    t = p_low + ((p_high & 0xFFFF) << 16)
    return (p_high >> 16) + (t >> 32), t & _M32


def philox4x32(c0, c1, c2, c3, key0: int, key1: int, rounds: int = PHILOX_ROUNDS):
    """Philox4x32 of the counter words ``c0..c3`` (int64 tensors holding values in [0, 2**32)) with the key
    ``(key0, key1)``: four int64 tensors of 32-bit words."""
    k0, k1 = int(key0) & _M32, int(key1) & _M32
    for index in range(rounds):
        if index:
            k0 = (k0 + PHILOX_W0) & _M32
            k1 = (k1 + PHILOX_W1) & _M32
        hi0, lo0 = _mulhilo(c0, PHILOX_M0)
        hi1, lo1 = _mulhilo(c2, PHILOX_M1)
        c0, c1, c2, c3 = (hi1 ^ c1) ^ k0, lo1, (hi0 ^ c3) ^ k1, lo0
    return c0, c1, c2, c3


def _words(seed: int, offsets):
    import torch

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 64:
        raise ValueError(f"a stream seed is an int in [0, 2**64), got {seed!r}")
    if offsets.dtype != torch.int64:
        raise TypeError(f"stream offsets must be int64, got {offsets.dtype}")
    zero = torch.zeros_like(offsets)
    return philox4x32(offsets & _M32, offsets >> 32, zero, zero, seed & _M32, seed >> 32)


def _unit(high, low):
    """A float64 in [0, 1) from two 32-bit words: their top 27 and 26 bits."""
    import torch

    return ((high >> 5) * (1 << 26) + (low >> 6)).to(torch.float64) * _TWO_POW_M53


def uniforms(seed: int, offsets):
    """Uniforms in [0, 1), float64, one per entry of the int64 tensor ``offsets`` (the numbers' positions)."""
    x0, x1, _, _ = _words(seed, offsets)
    return _unit(x0, x1)


def normals(seed: int, offsets):
    """Standard normals, float64, one per entry of ``offsets`` (Box–Muller on the two halves of one draw)."""
    import torch

    x0, x1, x2, x3 = _words(seed, offsets)
    radius = torch.sqrt(-2.0 * torch.log1p(-_unit(x0, x1)))
    return radius * torch.cos((2.0 * math.pi) * _unit(x2, x3))


def positions(base, count: int):
    """The int64 offsets ``base, base + 1, ..., base + count - 1`` for a device scalar or ``[1]`` ``base``."""
    import torch

    return base.reshape(1) + torch.arange(count, dtype=torch.int64, device=base.device)


# --------------------------------------------------------------------------- decoding


def select(keys, eligible):
    """Fixed-shape selection (SPEC §10.4): the atom with the smallest key among ``eligible`` ones, and whether
    there is one. Keys of other atoms are +inf; the decision reads the minimum value, never the index alone
    (an all-+inf vector has a valid-looking index 0). Returns int64 ``[1]`` index and bool ``[1]``."""
    import torch

    masked = torch.where(eligible, keys, torch.full_like(keys, math.inf))
    value, index = torch.min(masked, dim=0, keepdim=True)
    return index, torch.isfinite(value)


def decode_swap(u, species, pairs):
    """A swap proposal from its ``2N + 1`` proposal numbers ``u``: number 0 picks the pair type
    ``p = min(floor(u0 * P), P - 1)`` of the ``[P, 2]`` int64 table ``pairs`` (atomic numbers); numbers
    ``1..N`` are the keys of the first species' atoms, ``N + 1..2N`` those of the second's.

    Returns ``(atoms [2] int64, possible [1] bool)``: atom i of the first species and atom j of the second,
    impossible when either species is absent (then the indices mean nothing)."""
    import torch

    natoms = species.shape[0]
    count = pairs.shape[0]
    choice = torch.clamp((u[:1] * count).to(torch.int64), max=count - 1)
    pair = pairs[choice[0]]  # [2], a device gather
    first, first_ok = select(u[1:natoms + 1], species == pair[0])
    second, second_ok = select(u[natoms + 1:2 * natoms + 1], species == pair[1])
    return torch.cat((first, second)), first_ok & second_ok


def decode_transmute(u, species, set_, members, position):
    """A transmutation proposal from its ``N + 1`` proposal numbers ``u``: numbers ``1..N`` are the keys of the
    atoms whose species is in the transmutable set S (``members``: bool table by atomic number), number 0
    picks the new species uniformly from S without the atom's current one: ``r = min(floor(u0 (|S| - 1)),
    |S| - 2)``, then ``S[r + (r >= c)]`` with ``c`` the current species' index in S (``position``: int64
    table by atomic number; ``S`` is the int64 tensor ``set_``).

    Returns ``(atom [1] int64, new species [1] int64, possible [1] bool)``; impossible when no atom of S
    is present."""
    import torch

    natoms = species.shape[0]
    size = set_.shape[0]
    atom, possible = select(u[1:natoms + 1], members[species])
    current = position[species[atom]]  # [1]
    r = torch.clamp((u[:1] * (size - 1)).to(torch.int64), max=size - 2)
    return atom, set_[r + (r >= current).to(torch.int64)], possible


def stream_seeds(seed: int) -> dict[str, int]:
    """``config.mc_stream_seeds``: the three streams' seeds for a simulation seed."""
    from fastmace.config import mc_stream_seeds

    return mc_stream_seeds(seed)


__all__ = ["ACCEPTANCE_COUNT", "PHILOX_ROUNDS", "attempt_counts", "block_velocity_count", "decode_swap",
           "decode_transmute", "normals", "philox4x32", "positions", "proposal_count", "select",
           "stream_seeds", "uniforms", "velocity_count"]

