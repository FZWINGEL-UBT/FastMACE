# Tiled path: how much of a padded call is padding

**Question.** Is a tighter edge fit for the padded slots, together with equal-piece block splitting, worth building?
It was the conductor's last tiled-specific speed lever; the threshold was at least 5 % combined.

**Answer: no.** At production sizes the recoverable share is about 2-3 % of the edge work and at most about 1 % from
the remainder pieces, below the 5 % threshold. Nothing was built.

## The measurement

Tool: `results/tiled-pieces-20261102/padding.py`. It runs one padded call: eager slots of the same sizes as the
compiled modes, with tile lists, the host state aliased, the species index, and no budget. Before the call fills the
slots, `_slot_keys` is wrapped to record each tile's receivers (R), halo (H) and edges (E), and the slot each tile
lands in.

**Setup.** Murphy, NVIDIA RTX 4000 Ada, `mace_cueq_oeq`, `mh1-r2scan` float32, at 366aab4a (main df301c76 plus
the tool).

- **Real rows:** R + H, the rows interaction 0 runs on.
- **Padded rows:** the slot's receiver rows plus its halo rows.
- **Remainder piece:** a tile with fewer than half of the receiver cap, i.e. the last piece of a block that holds
  more atoms than the cap.

| NbCl5 atoms | Cap | Tiles | Row padding | Edge padding | Halo share of real rows | Remainder tiles | Their padded edges |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 884,736 | 4,096 | 312 | 11.4 % | 11.8 % | 38.9 % | 30.8 % | 6.0 % |
| 884,736 | 16,384 | 64 | 13.3 % | 7.4 % | 27.1 % | 0 | 0 |
| 9,261,000 | 4,096 | 2,395 | 11.2 % | 9.4 % | 37.4 % | 1.2 % | 0.2 % |
| 9,261,000 | 16,384 | 608 | **6.3 %** | **7.5 %** | 27.2 % | 5.3 % | **0.8 %** |

The full tiles are close in size. At 9.26M with the cap at 16,384:
- receivers range from 15,561 to 16,384, with a median of 16,114;
- edges range from 408,954 to 430,724, which is ±2.6 %.

## Where the padding comes from (9.26M, cap 16,384, the default budget's size)

- **Edges: 7.5 %.** All full tiles fall in one edge bucket. Its top, 453,376 edges, is 5.3 % above the largest
  tile; the remaining 2-3 % is the spread between tiles.
  - A tighter fit would size the slot at the largest tile plus a margin for the drift in MD: about 2.5 % in 20
    steps ([tiled-slot-growth.md](tiled-slot-growth.md)).
  - That leaves the spread plus the margin, about 4-5 %, and recovers **about 2-3 % of the edge work**.
- **Rows: 6.3 %.** Mostly the 10 % headroom on R and H, which MD needs: H drifted up to 3.3 % in 20 steps.
  - Cutting the headroom to about 5 % would recover about 3 % of the rows.
  - The node-wise work is the smaller part of a call: the convolution's edge work dominates. So this is about 1 %
    of the call, and it risks more slot growth in MD.
- **Remainder pieces: 0.8 % of the padded edges.** Equal-piece splitting removes **at most about 1 %**.
  - At 884,736 atoms with the cap at 4,096 they are 6 % of the edges.
  - The budget raises the cap to 16,384 at every size where compiled mode fits, which removes them there.

## Conclusion

The bound of 6-8 % in the lever list ([tiled-leftovers.md](tiled-leftovers.md)) assumed that most of the padding
can be removed. The measurement shows most of it is the headroom that keeps slots from growing in MD. The part that
can be recovered is about 3 % of a call, so the lever does not meet the 5 % threshold.

Runs: `results/tiled-pieces-20261102/runs/murphy-padding-*`.
