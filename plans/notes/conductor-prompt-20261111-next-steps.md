# Prompt for the conductor: next steps after 11 November 2026

You are the FastMACE conductor. The owner asks you to act on today's review and idea work. Everything is collected
in `plans/next-steps-20261111.md`: read it first. It ranks what is ready to merge, what one packed job decides, and
what needs a falsifier, each with a size and a stop rule. Sizes without a measurement are estimates; treat them as
bounds to test, not as results.

## What I want from you

1. **Merge two measured results.**
   - `BINNED_SCREEN` on the RTX 5060 Ti: add the GPU to `pair_search.BINNED_SCREEN_GPUS` (work removal, bit-identical
     list, decision 66). Evidence: `results/binned-screen-5060ti-20261111/README.md`.
   - `tf32x3.HYBRID` on the A100 family: branch `fastmace/review-retries-20261111` (`fd9a74ad`). Run the leto suite
     on the merged tip as usual.
2. **Launch three packed jobs** (section B of the file):
   - Modal H200: `HYBRID` A/B in both orders and the gate. The sm_90 cubin exists; nothing was measured there.
   - leto or murphy, after the idle check: `BINNED_SCREEN` A/B on the RTX 4000 Ada at SrTiO3 16×16×16 and 20×20×20.
   - Alex: receiver caps 32,768 and 65,536 for the A100's host-state and wavefront tiled calls.
3. **Give the first interaction a lane** (section C, items C1 and C2). Its two tabled kernels are 13 % of the
   A100's force call and 10 % of the H200's, and have had far fewer rounds than the interaction-2 kernels. Each
   item starts with a falsifier on free hardware and stops under the lab bar (68 µs on the H200, 130 µs on the
   A100).
4. **Run one offline compile screen** covering C3, C4 and C5 together (CPU only, NVRTC census for sm_80 and
   sm_90). These are 1 %-class each; do not give them GPU time unless the census clears its stop rule.
5. **Close the texture idea with one microbenchmark** (fetch rate of a float32 linear-filtered texture), or
   without it if you judge the argument in the file sufficient.
6. **Record and tidy.**
   - Commit this session's untracked files listed under "Housekeeping" in the file, and copy the outside note
     from `D:\Margraf\inference-ideas-20261111\` into `plans/notes/`.
   - Add ledger rows for ideas 18–20 and the three outside ideas with their bounds, and fix the four stale
     statuses named there.
   - `plans/notes/rejected-ledger-20261111/` holds a catalogue of all 288 rejected-type ledger rows and a list of
     27 inconsistencies the reading agents reported; none is verified. Fix the ones you can confirm.

## Constraints

- The usual rules apply: free hardware before paid, packed jobs, the local GPU reservation, the workstation idle
  check, no local CPU suite.
- Do not reopen what was closed today (half-precision far edges, half-systems on two streams, the formulation
  search, the device row pool) without new evidence.
- Any w′ variant for the A100 must stay within about 85 registers and 27 KB of shared memory per block.
- The capacity-path items (section D) are not part of this request; schedule them only if the owner asks.

## What to report back

One short report: what was merged, the three packed jobs' ratios and gate results, the falsifier results for C1
and C2 against their bars, the census counts for C3–C5, and anything you stopped and why.
