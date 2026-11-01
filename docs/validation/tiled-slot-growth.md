# Tiled path: padded slots that settle during MD

Compiled tiles run in padded slots of fixed receiver (R), halo (H) and edge (E) counts. A slot that grows compiles
again. In the MD of [tiled-leftovers.md](tiled-leftovers.md) the slots kept growing:
- 5 changes in 12 steps at 884,736 NbCl5 atoms;
- 9 changes in 6 steps at 9.26M, where the step time alternated between 135 and 170 s.

## What grows (20 compiled MD steps, before the fix)

**Setup.** `tiled_md.py` compiled, 2 fs at 600 K. Each step now records the slots, their changes, and each bucket's
largest tile sizes before headroom (`slot_needs`, a new field in the call's report). The runs are
`results/tiled-slot-growth-20261101/runs/*-sg0-*`.

| Slot | Over 20 steps | Changes after the budget's switch |
| --- | --- | ---: |
| 884,736: the one slot (R = 16,384) | R steady (±0.05 %), H +3.3 % (flat after step 8), E +2.5 % (+0.13 %/step) | 2 |
| 9.26M: the main slot | R at the cap, H +1.6 %, E +2.3 % | 1 |
| 9.26M: the small slot (bucket 1) | R 51 → 160, H 432 → 724, E 1,355 → 4,343 | 5 |

**Two causes:**
1. **The fit test included the headroom.** `_slot_keys` grew a slot as soon as its largest tile came within 10 %
   of the slot size, while the tile still fit. This caused every change of the large slots:
   - at 884,736 atoms, H 5,555 against a slot of 5,998, and R 13,853 against 15,239;
   - at 9.26M, H 6,584 against 7,151.
2. **The small slot holds the remainder pieces of blocks just above the cap.** Relative to their size these
   pieces grow fast, about 3× in 20 steps. A growth by a quarter of the slot's own size catches up only in many
   small steps.

## The fix

In `TiledForceField._slot_keys`:
1. **Fit test.** It takes the tiles' own sizes, with one padding row. The headroom (`TileConfig.headroom`, still
   10 %) applies only when a slot is sized or grown.
2. **First sizing.** Unchanged: 10 % headroom on R and H, and the bucket's edge count for E. The large slots'
   drift in 20 steps is under 3.5 %.
3. **Growth.** An exceeded size grows by at least a quarter of the largest slot's in that size. For the largest
   slot this is the previous ×1.25. A small slot takes one step to a size its tiles do not outgrow. The rule
   depends only on the slots' own sizes.

**CPU test.** `test_padded_slot_kept_while_its_tiles_fit_and_grown_by_a_quarter_of_the_largest` checks two things:
- a slot exactly as large as its largest tile is kept (this fails before the fix);
- a slot one halo row short grows in that size by at least a quarter of the largest slot's halo, and keeps the
  sizes that fit.

`test_tiled.py`, `test_layering.py` and `test_neighbors.py` on trillian at 260d67bb (rebased on main 058d5166):
162 passed, 91 skipped.

## After the fix (20 compiled MD steps; `runs/*-sg1-md-*`)

| NbCl5 atoms | Recompiles after the budget's switch | Steady step | Drift |
| ---: | --- | ---: | ---: |
| 884,736 (murphy) | **0** (before: 2) | 13.42 s (before: 14.21 s) | 0.050 meV/atom |
| 9,261,000 (leto) | **1** (before: 7), the small slot once at step 2 | 135.6 s, steps 2-19 within 134.4-136.0 s | 0.050 meV/atom |

At 9.26M the small slot grows once, to (4,155, 2,264, 65,536), and stays there. The large slot keeps
(16,385, 7,151, 458,752).

**What remains.** The large slot's E keeps rising slowly, about 0.13 % per step at 884,736 atoms, as the lattice
thermalises. After about 20 to 40 more steps its largest tile reaches the next edge bucket. That makes a new slot:
one more compilation, and then the bucket holds.

## The memory at 70.96M (leto)

`tiled_capacity.py --mode auto --calls 3`, default budget, at a882227d (`runs/leto-sg1-auto3-NbCl5-69x69x207`). The
run is valid.

| Call | Mode | Time | GPU peak | Host peak |
| ---: | --- | ---: | ---: | ---: |
| 1 | eager | 1,747 s | 14.49 GiB | 213 GiB |
| 2 | compiled (R = 16,384) | 1,057 s | **17.73 GiB** | 219 GiB |
| 3 | compiled | 1,035 s | **17.73 GiB** | 219 GiB |

- **Repeated calls:** the peak is unchanged from before the fix (17.73 GiB, [tiled-leftovers.md](tiled-leftovers.md)).
  The first sizing is the same, and repeated calls on the same positions do not grow a slot.
- **In MD the slots end a little larger.** In the 70.96M NVE run of [tiled-md-host.md](tiled-md-host.md) (this
  fix with the host integrator), the main slot ended at (16,385, 7,585, 460,544): H +2.2 % and E +1.6 % on the
  repeated calls' slot. That run completed four calls on the same 20 GB card. Its peak was not recorded; from the
  sizes, it is expected about 0.1-0.2 GiB above 17.73 GiB, within the 1.8 GiB left.
