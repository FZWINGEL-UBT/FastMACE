"""The MC random streams (plan task 3-T09; SPEC §8.5 "Random numbers per outcome").

- Philox4x32-10 reproduces Random123's known-answer vectors;
- a stream's numbers depend only on (seed, offset): any slice of offsets gives the same numbers;
- the uniforms are in [0, 1) with 53 random bits, the normals have the right moments;
- the fixed counts per attempt and the proposal decoding (pair type, keys, +inf for other species,
  "impossible" from the minimum value, the new species of a transmutation).
"""

import numpy as np
import pytest
import torch

from fastmace.config import MC_STREAMS, mc_stream_seeds
from fastmace.mc import streams

M32 = 0xFFFFFFFF


def words(counter, key):
    tensors = [torch.tensor([value], dtype=torch.int64) for value in counter]
    return [int(word[0]) for word in streams.philox4x32(*tensors, *key)]


@pytest.mark.parametrize(("counter", "key", "expected"), [
    # Random123 kat_vectors, philox4x32_10.
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((M32, M32, M32, M32), (M32, M32), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
])
def test_philox_known_answers(counter, key, expected):
    assert tuple(words(counter, key)) == expected


def test_numbers_depend_only_on_seed_and_offset():
    seed = mc_stream_seeds(5)["proposal"]
    whole = streams.uniforms(seed, torch.arange(1000, dtype=torch.int64))
    part = streams.uniforms(seed, torch.arange(400, 700, dtype=torch.int64))
    assert torch.equal(whole[400:700], part)
    scattered = torch.tensor([999, 3, 2 ** 40 + 7, 3], dtype=torch.int64)
    again = streams.uniforms(seed, scattered)
    assert again[1] == whole[3] and again[3] == whole[3] and again[0] == whole[999]
    assert not torch.equal(streams.uniforms(seed + 1, torch.arange(1000, dtype=torch.int64)), whole)
    base = torch.tensor(400, dtype=torch.int64)
    assert torch.equal(streams.uniforms(seed, streams.positions(base, 300)), part)


def test_uniforms_and_normals_have_their_distributions():
    seed = mc_stream_seeds(1)["velocity"]
    offsets = torch.arange(200_000, dtype=torch.int64)
    u = streams.uniforms(seed, offsets)
    assert u.dtype == torch.float64 and float(u.min()) >= 0.0 and float(u.max()) < 1.0
    assert abs(float(u.mean()) - 0.5) < 0.005
    assert torch.all(u * 2.0 ** 53 == torch.round(u * 2.0 ** 53))  # 53-bit grid
    z = streams.normals(seed, offsets)
    assert abs(float(z.mean())) < 0.01 and abs(float(z.std()) - 1.0) < 0.01
    assert abs(float((z ** 4).mean()) - 3.0) < 0.05
    assert torch.isfinite(z).all()


def test_stream_seeds_are_distinct_and_fixed():
    seeds = streams.stream_seeds(3)
    assert tuple(seeds) == MC_STREAMS and len(set(seeds.values())) == 3
    assert seeds == mc_stream_seeds(3)


def test_fixed_counts_per_attempt():
    assert streams.attempt_counts("swap", "exchange", 8) == {"proposal": 17, "acceptance": 1, "velocity": 0}
    assert streams.attempt_counts("swap", "redraw", 8) == {"proposal": 17, "acceptance": 1, "velocity": 6}
    assert streams.attempt_counts("transmute", "redraw", 8) == {"proposal": 9, "acceptance": 1, "velocity": 3}
    assert streams.attempt_counts("transmute", "rescale", 8)["velocity"] == 0
    assert streams.attempt_counts("swap", "redraw_all", 8)["velocity"] == 0
    assert streams.block_velocity_count("redraw_all", 8) == 24
    assert streams.block_velocity_count("rescale", 8) == 0
    with pytest.raises(ValueError, match="3-T10"):
        streams.proposal_count("displace", 8)


def test_selection_reads_the_minimum_value():
    keys = torch.tensor([0.9, 0.2, 0.5, 0.1], dtype=torch.float64)
    index, ok = streams.select(keys, torch.tensor([True, True, True, False]))
    assert index.tolist() == [1] and ok.tolist() == [True]
    index, ok = streams.select(keys, torch.zeros(4, dtype=torch.bool))
    assert ok.tolist() == [False]  # the index (0) is valid-looking, the minimum is +inf


def test_decode_swap():
    species = torch.tensor([11, 17, 11, 17, 8], dtype=torch.int64)
    pairs = torch.tensor([[11, 17], [8, 17]], dtype=torch.int64)
    u = torch.tensor([0.7, 0.9, 0.1, 0.3, 0.2, 0.0, 0.5, 0.6, 0.05, 0.4, 0.8], dtype=torch.float64)
    atoms, ok = streams.decode_swap(u, species, pairs)  # 0.7 * 2 -> pair 1: O and Cl
    assert ok.tolist() == [True] and atoms.tolist() == [4, 3]
    u[0] = 0.2  # pair 0: Na (keys 0.9, -, 0.3) and Cl (keys of the second half: 0.6, 0.4)
    atoms, ok = streams.decode_swap(u, species, pairs)
    assert ok.tolist() == [True] and atoms.tolist() == [2, 3]
    species[4] = 11  # no O: pair 1 is impossible
    u[0] = 0.99
    _, ok = streams.decode_swap(u, species, pairs)
    assert ok.tolist() == [False]


def test_decode_transmute_never_proposes_the_current_species():
    species = torch.tensor([11, 17, 19, 17], dtype=torch.int64)
    set_ = torch.tensor([11, 19, 37], dtype=torch.int64)
    members = torch.zeros(119, dtype=torch.bool)
    members[set_] = True
    position = torch.zeros(119, dtype=torch.int64)
    position[set_] = torch.arange(3)
    seen = set()
    for r0 in np.linspace(0.0, 0.999, 41):
        for key0, key2 in ((0.1, 0.5), (0.5, 0.1)):
            u = torch.tensor([r0, key0, 0.0, key2, 0.0], dtype=torch.float64)
            atom, new, ok = streams.decode_transmute(u, species, set_, members, position)
            current = int(species[atom])
            assert ok.tolist() == [True] and current in (11, 19)
            assert int(new) in (11, 19, 37) and int(new) != current
            seen.add((current, int(new)))
    assert seen == {(11, 19), (11, 37), (19, 11), (19, 37)}
    _, _, ok = streams.decode_transmute(torch.zeros(5, dtype=torch.float64), torch.tensor([17] * 4), set_, members,
                                        position)
    assert ok.tolist() == [False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_same_numbers_on_the_gpu():  # pragma: no cover - GPU only
    seed = mc_stream_seeds(2)["acceptance"]
    offsets = torch.arange(10_000, dtype=torch.int64) + 2 ** 33
    assert torch.equal(streams.uniforms(seed, offsets.cuda()).cpu(), streams.uniforms(seed, offsets))
    np.testing.assert_allclose(streams.normals(seed, offsets.cuda()).cpu(), streams.normals(seed, offsets),
                               rtol=1e-14, atol=1e-14)
