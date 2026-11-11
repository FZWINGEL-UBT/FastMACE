# Two half-systems out of phase: the review retest (11 November 2026)

The review (`results/half-far-wprime-20261111/review-20261111-rejections.md`) found that the first falsifier started
both half-system graphs on one event every round, so they always ran in phase. This retest adds an `offset` side to
`tools/bench/half_system_streams.py`.
- Stream 2 is delayed once, at the start of a block of 40 replay pairs, by half of one half-system graph: a
  `torch.cuda._sleep` calibrated on the GPU (1,643 µs).
- After the delay both streams replay freely, so equal graphs stay about half a step apart. B's edge kernels then
  meet A's atom chain.
- `offset_net` subtracts the delay per pair.

Alex A100-SXM4-80GB (node a0534, driver 615.71.10), job 4983700, source 600942ca, OMAT-0 medium, default A100 plan.
Graphs: SrTiO3 10×10×10 (full) and two 10×10×5 halves. 21 alternating rounds of 40 replays. Record:
`half_system_streams.json`.

| side | µs per pair | over full graph |
| --- | --- | --- |
| full 10×10×10 graph | 5,987 | 1.000 |
| serial (A then B) | 6,593 | 1.101 |
| concurrent, in phase | 6,051 | 1.011 |
| offset, out of phase (net of the delay) | 6,081 | 1.016 |

Out of phase is 1.005× in phase and 1.016× the full graph. Even uncoupled halves never beat one full-size graph in
either phase, which is the bound for the coupled split. The idea stays rejected and is now closed. This matches the
review's expectation: the edge kernels fill 61,440 of 65,536 registers per SM.
