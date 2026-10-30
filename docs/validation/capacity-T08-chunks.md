# CAP-T08: atom-chunked node-local segments (falsifier)

Plan task CAP-T08 ([plans/capacity.md](../../plans/capacity.md)), idea 2 of the owner's capacity list of
30 October 2026, through the conductor.

**The idea.** Run the node-local segments of the interactions (every row an atom, nothing across atoms) on atom
chunks:

- the forward saves nothing but the segment's inputs;
- the backward recomputes one chunk at a time.

That cuts the per-atom state the force call keeps, which is about 95 % of its memory at the limit
([capacity-T08-inventory.md](capacity-T08-inventory.md)).

**The falsifier, first:** how much of the saved per-atom memory those segments really hold, and how much of the peak
chunking can reach. Stop if the reachable saving is under about 5 % of the peak.

## Verdict: passes the falsifier, at a high cost in the naive form

In the compiled default plan, chunking both interactions' segments lowers the force call's peak by 0.62 GiB (13 %)
at 21,600 NbCl5 atoms.

- At the 51,840-atom limit that would be about 1.5 GiB, about 10 % of the reserved peak. That is above the 5 % bar.
- The prototype nearly doubles the force call (1.87×): its chunks run outside the compiled graph and break it.
- **Interaction 1 alone is not enough.** It saves 0.47 GiB in eager (8 %), because interaction 1's convolution
  backward then sets the peak. Interaction 0's saved segments are alive there too, so both interactions must be
  chunked.

## Where the peak is

**Tool:** `tools/bench/segment_peaks.py` (new), NbCl5 5×5×12 (21,600 atoms, 572,970 edges), MH-1 hybrid float32,
`recompute_conv_weights=on` (as at the limit), RTX 5060 Ti.

**How it measures.** In the eager tier, markers at each marked module's forward and backward give the peak of every
interval. A `saved_tensors_hooks` pack attributes each saved storage to its module.

**What the backward keeps per module** (eager, per atom):

| Module | Saved, per atom |
| --- | ---: |
| `interaction1.gated_rows` | 51.2 kB |
| `interaction0.gated_rows` | 51.2 kB |
| `product1` | 32.8 kB |
| `product0` | 32.8 kB |
| `interaction1` (the convolution and the rest) | 22.9 kB |
| `interaction0` | 8.2 kB |

The node-local segments (`gated_rows` and the product block) hold 168 of the 199 kB per atom that eager saves.

**The peaks, above the call's start** (eager, no chunking):

| Interval | Peak |
| --- | ---: |
| `interaction1.gated_rows` forward (the call's peak) | 5.70 GiB |
| `product1` backward | 5.53 GiB |
| `interaction1.gated_rows` backward | 5.48 GiB |
| interaction 1's convolution backward | 5.24 GiB |
| interaction 0's backward | 3.35 GiB |

Interaction 1's convolution backward runs after its node-local segments' backward. Their saved tensors are freed by
then, but interaction 0's (1.69 GiB) are still alive.

## The prototype

**How it works.** `segment_peaks.py --chunk MODULE... --chunks K` replaces each named module's forward with
`AtomChunked`, an autograd function:

- **forward:** runs the module on K atom chunks under `no_grad` and saves only its inputs;
- **backward:** recomputes one chunk at a time and calls `autograd.grad` on it.

**How the variants ran.** First a plain call gives the reference forces, then the chunked call is measured. The
compiled variants keep the default compile mode and drop the markers. All of them time 5 more calls (median).

| Variant | Peak above the call's start | Change | Force call | Max force difference |
| --- | ---: | ---: | ---: | ---: |
| eager, none | 5.70 GiB | | 348 ms | |
| eager, interaction 1 (8 chunks) | 5.24 GiB | −0.47 GiB (−8 %) | | 2.4 × 10⁻⁶ eV/Å |
| eager, both interactions (8 chunks) | 4.56 GiB | −1.14 GiB (−20 %) | 442 ms (1.27×) | 3.2 × 10⁻⁶ eV/Å |
| eager, both interactions (4 chunks) | 4.85 GiB | −0.85 GiB (−15 %) | | 3.1 × 10⁻⁶ eV/Å |
| **compiled, none** | **4.79 GiB** | | **210 ms** | |
| **compiled, both interactions (8 chunks)** | **4.16 GiB** | **−0.62 GiB (−13 %)** | **393 ms (1.87×)** | 3.6 × 10⁻⁶ eV/Å |

- **Rounding class, not bitwise.** The GEMMs see fewer rows, so they round differently. The differences are those of
  a float32 call.
- **With 8 chunks the peak moves to interaction 1's convolution backward.** With 4 chunks it stays in the recompute
  of `interaction1.gated_rows`, so more chunks help until the convolution sets the peak.
- **Compiled, the saving is smaller than eager:** 0.62 against 1.14 GiB. Inductor already fuses some saved
  intermediates away (174.5 against 199 kB per atom), and the graph breaks around the chunked modules add their own
  boundary tensors.
- **The time.** The compiled call goes from 210 to 393 ms. Several costs add up:
  - the chunked segments run in eager, without Inductor's fusions;
  - each graph break splits the compiled model;
  - the segments' forward runs again in the backward;
  - the GEMMs run on fewer rows.

## Next step, if the conductor confirms

**The rewrite.** Implement it as a rewrite planned on only when memory requires it (decision 67, like
`recompute_conv_weights`), for both interactions' `gated_rows` and product blocks. Then:

- the gate;
- the A/Bs in both orders at 1,944 and 9,216 atoms;
- new NbCl5 and NaCl limit scans.

**The main design question is the compiled path.** The prototype breaks the graph. A production version should keep
the chunked segment inside Inductor. Two ways:

- compile each chunk's forward, and each chunk's recompute-and-backward, as their own graphs;
- or express the chunk loop so that AOT autograd can trace it.

Either way the cost will stay above `recompute_conv_weights`' 1–4 %, because the segments' GEMMs run twice. Since
it is on only when a cell does not fit otherwise, the cost is paid only where the alternative is not running.

## Records

- `results/cap-t08-chunks-20261030/segments/` holds the tool's records without the plan text, and `jobs.log` the
  jobs. The four runs before the tool's fixes are not included: a missing structures path, and keyword arguments
  not bound.
- **Machine:** Fabi-PC, WSL2, RTX 5060 Ti 16 GB (driver 610.88), `mace_cueq_oeq` (torch 2.14.1+cu130), through
  `tools/ci/local_gpu.py run` reservations. FastMACE `531b6586` (main `fbd8a6ed` plus the tool).
