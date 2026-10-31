# The combined region's inputs as a tuple, and the contraction inside the region (31 October 2026)

**Verdict: STOP for both.** The tuple pass-through leaves the force call's peak unchanged, and the contraction output
is already a per-chunk working set inside the region. Measurement only; no library code changed.

## Tuple instead of cat (conductor's falsifier)

The claim: `segments/modules.py` copies the region's inputs into one buffer (`torch.cat` in `joined_inputs`, `split`
in `region_forward`), and passing them through as a tuple would free about 23 kB per atom.

`results/region-tuple-20261101/tuple_probe.py` replaces the two functions before the record is applied and otherwise
runs `region_probe.py`. Setup:
- the compiled force call's allocator peak above its start, median of 5 calls;
- combined region in 4 chunks, `recompute_conv_weights` and `recompute_node_segments` on;
- leeloo, RTX 4000 Ada, FastMACE 07175c8c;
- cat and tuple alternating, two rounds.

| NbCl5 atoms | cat: peak, call | tuple: peak, call |
|---:|---|---|
| 21,600 | 1.985 GiB, 281.8 / 284.2 ms | 1.985 GiB, 281.5 / 282.0 ms |
| 43,200 | 3.969 GiB, 559.2 / 558.9 ms | 3.969 GiB, 554.8 / 554.2 ms |

The peak is the same to the MiB. The call is 0.1–0.8 % shorter, within the noise of separate processes. As the
conductor read it:
- the region's checkpoint keeps its inputs either way;
- the message's gradient is needed for the convolution's backward anyway;
- so the cat costs at most a transient copy that is not live at the peak.

## The contraction output (M3 ledger, target 4)

The refreshed ledger listed cuEquivariance's `uniform_1d` output (the symmetric contraction, `[N, 2048]`) at
8.2 kB per atom at the peak. `peak_stacks.py` records the Python stacks of the blocks alive at the peak (NbCl5
21,600 atoms, same setup). The block comes from `symmetric_contraction` in `mace/modules/blocks.py:500`, called from
`segments/modules.py:63 region` inside the chunk loop. So the contraction is already inside the combined region.

| Region chunks | Contraction block at the peak | Live peak |
|---:|---:|---:|
| 1 | 32.8 kB per atom | 159.0 kB per atom |
| 4 (default) | 8.2 kB per atom | 98.7 kB per atom |
| 8 | 4.1 kB per atom | 112.0 kB per atom |

- **One chunk's working set.** The block is the contraction output of the chunk being recomputed in the backward,
  N/chunks rows. The ledger's per-atom figure divides it by all atoms.
- **More chunks cost more.** At 8 chunks the contraction halves again, but a 22.5 kB-per-atom block appears and the
  peak rises. This matches the chunk screen in [chunked-region.md](chunked-region.md).
- **What is left at 4 chunks** (98.7 kB per atom):
  - interaction 1's raw message, the region's input, 20.5 kB per atom (the division's backward needs it);
  - one chunk's recompute working set, about 25 kB per atom (the contraction 8.2, three divided-message blocks of
    5.1);
  - the radial table's output, 6.8 kB per atom;
  - the gate's input gradients.

None of these is a cheap target: the next memory gains need a different region layout or a smaller message, at more
recompute. The owner now weighs speed at large sizes, so the region-memory line stops here.

## Records

`results/region-tuple-20261101/`:
- `tuple_probe.py`, `peak_stacks.py`;
- `cat-*/` and `tuple-*/`: region_probe records;
- `run.log`;
- `peak_stacks*.txt`: the stacks at 4, 1 and 8 chunks.
