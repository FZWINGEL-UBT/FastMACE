# Prompt for the conductor: next steps, revision 2 (11 November 2026)

You are the FastMACE conductor. The first queue is done: two switches merged, C1 to C5 and the texture idea closed.
The owner asks you to act on the revised queue in `plans/next-steps-20261111-rev2.md`. Read it first; it supersedes
`plans/next-steps-20261111.md`. Sizes without a measurement are estimates to be tested.

## What I want from you

1. **Start one lane on the nested radial basis** (section 1 of the file; idea 1 of the outside note in
   `D:\Margraf\inference-more-20261111\ideas.md`). It is the only open kernel idea with a ceiling well over the
   bars: 5.4 % of the A100's force call and 4.1 % of the H200's. Run it in three gated steps:
   - **Certificate (CPU):** transform the real first table and the composed matrices, evaluate in float32, run the
     radial certificate against the original chain. Stop if any pair needs more than 32 far columns at the
     tolerance the certificate requires.
   - **Timing bound (5060 Ti, then Alex A100):** `tools/first_interaction_lab.py` with 32 columns for every edge.
     The gain is 0.84 × the difference from 64 columns. Stop under 130 µs on the A100.
   - **Real kernels:** a per-edge width branch, the rotated table, the composed matrices; then the packed A/B and
     the gate.
2. **Launch one Modal H200 pack with two existing switches:** `tf32x3.HYBRID` and `spline_forward.GEOM`, each as an
   A/B in both orders, plus the gate. Neither has been measured on the H200.
3. **Run the three packed jobs still open from the first queue**, unless they are already queued:
   `BINNED_SCREEN` on the RTX 4000 Ada (leto or murphy, after the idle check); receiver caps 32,768 and 65,536 for
   the A100's host-state and wavefront tiled calls (Alex); shuffled against ordered atoms at 24×24×24 (local
   5060 Ti).
4. **Do not build the other two outside ideas.** The compact angular products have the same operation count as
   `GEOM`; revisit them only if `GEOM` is slower on the H200 and a census shows at most 80 registers. The shared
   denominator is under both bars by today's one-load falsifier.
5. **Record.**
   - Copy the second outside note into `plans/notes/` and commit it with the revised queue and this prompt.
   - Ledger rows: the nested basis (open), the compact angular products (waits on `GEOM`), the shared denominator
     (stopped by the bound).
   - Check that the index rows of ideas 18–20 and the two first outside ideas show today's stops.

## Constraints

- The usual rules: free hardware before paid, packed jobs, the local GPU reservation, the workstation idle check,
  no local CPU suite.
- The nested basis is a rounding change under a certificate. It needs the 3× float32 gate, and it applies as
  designed only where `COMPOSE_SKIP` is on (H200, A100, Ada).
- Keep the first interaction's forward on sm_80 at or under its present 128 registers, and w′ within about 85
  registers and 27 KB of shared memory per block.
- The capacity-path items are not part of this request.

## What to report back

One short report: the certificate's far rank per pair and tolerance; the 32-column timing against the 130 µs bar;
the H200 ratios and gate results for `HYBRID` and `GEOM`; the three packed jobs' ratios; anything you stopped and
why.
