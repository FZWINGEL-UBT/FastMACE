# Chunked neighbour-list compaction (exact)

The conductor's capacity step of 31 October 2026: `SkinNeighbourList.compact` without the whole-list float64
candidate arrays. The tiled path's NbCl5 build ran out of memory there at 2.3M atoms
([pair-search-grid.md](pair-search-grid.md)).

## The change

**The problem.** `compact` formed, for every candidate pair at once:

- the float64 shifts (`times_3x3` with its three intermediates);
- the gathered positions;
- the vectors;
- the int64 test, cumulative sums and destinations.

That is about 150 B per candidate, 11 GiB of temporaries at 2.3M NbCl5 atoms.

**The chunked compaction.** Above `COMPACT_CHUNK` (2²² = 4,194,304 candidates, about 150k MH-1 atoms) it now works
in two passes over chunks of candidates.

- **Pass 1:** each chunk's within-cutoff test, kept as one byte per candidate, and the count.
- **Pass 2:** each chunk's destinations, from the counts before it, then its copies:
  - inside pairs go to `before + local cumulative sum − 1`;
  - outside pairs go to `count + (start − before) + local cumulative sum of the outside − 1`.

  These are the same integers as the one-chunk cumulative sums, so the list is **bitwise** the same, padding and
  order included.
- **Small lists.** Up to `COMPACT_CHUNK` candidates the code is the one-chunk compaction of before.
- **Graph capture.** It stays capturable: the chunks are fixed by the capacity, with no host synchronisation.

**The fused compaction kernel** of the sixth idea list was not built (the conductor agreed). The chunking already
removes the whole-list temporaries. A fused Triton kernel would mainly save time, at a real risk to the bitwise
result: Triton contracts `a * b + c` into FMA, while PyTorch's eager kernels round each product.

## Checks

**Tests:**

- `test_neighbors.py`: the chunked compaction against the one-chunk compaction, bitwise, for chunks of 1, 7 and
  1,000 candidates, fixed and variable cell.
- `test_neighbors_gpu.py`: the same on the GPU build with sender groups, fixed and variable cell, chunk 777, and
  the chunked compaction captured in a CUDA graph and replayed.
- **CPU** (murphy, GPU hidden, `mace_cueq_latest`): `test_neighbors.py`, `test_d3_list.py` and `test_layering.py`,
  109 passed (91 CUDA-only cases skipped).
- **GPU** (murphy, baseline set): `test_neighbors_gpu.py`, `test_neighbors_lean_gpu.py`, `test_d3_list_gpu.py` and
  `test_async_rebuild_gpu.py`, **155 passed**.

**Same-process A/B against main** (`tools/bench/compact_ab.py`):

- main's `compact` (`bacf6e07`) against this one, on the same GPU list;
- NbCl5 rattled by up to 0.05 Å and moved by up to 0.1 Å;
- 10 rounds of ABBA, medians;
- record: `results/compact-chunked-20261031/compact-ab.json`.

| Cell | Atoms | Candidates (capacity) | Edges | Main | This branch | Ratio | Bitwise | Peak, main / this |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| NbCl5 4×4×8 | 9,216 | 316,152 (363,776) | 249,008 | 0.429 ms | 0.433 ms | 1.011 | yes | 37.5 / 33.3 MiB |
| NbCl5 6×6×20 | 51,840 | 1,779,114 (2,046,208) | 1,400,774 | 5.705 ms | 5.704 ms | 1.000 | yes | 211 / 187 MiB |

Both cells compact in one chunk, so the code path is the one of before. The 4 µs at 9,216 atoms is within
run-to-run noise.

## The tiled limits with this and the screen-launch fix

(rescan running; see the conductor's report)
