# OPT6-T02: `chunked_conv_backward`, the streaming backward in L2-sized edge chunks (27 October 2026)

Plan: `plans/opt-6.md`. Falsifier: [opt6-T01.md](opt6-T01.md). Code: FastMACE `08b0d004` on branch
`fastmace/opt6-T01`. Records: `results/opt6-T02-chunked-backward-20261027/` (`local.sh`, `smoke.py`,
`run_child.py`, `local-08b0d004/`).

**Verdict: exact, gate passed. Same-process full-step A/B: NbCl5 −2.0 to −2.3 %, NaNbO3 −3.5 to −3.6 %, NaCl
+0.6 to +1.1 % (slower).**

**Default since 27 October 2026 (owner decision 60, "yes", with a size rule for small cells).** The planner
rule is `on_with_streaming_conv_and_fused_folded_backward`. Inside the op, weights of at most
`UNCHUNKED_L2_MULTIPLE` = 4 L2s run the two kernels once over all edges, as before. With that rule, on the
default commit `2ea19f0d` (`results/opt6-T02-chunked-backward-20261027/default-2ea19f0d/`, same protocol as
below, 07:29–07:49 UTC):

| Cell | a-b b/a | b-a b/a |
| --- | ---: | ---: |
| NaCl | 1.0077 | 1.0087 |
| NbCl5 | **0.9725** | **0.9669** |
| NaNbO3 | **0.9694** | **0.9642** |

NaCl stays about 0.8 % slower although it no longer chunks. So the cost there is not the chunking; its source
is not identified (not profiled). The same CUDA kernels run per edge, and the extra work outside them is a few
small tensors. The NaNbO3 runs overlapped CPU test runs of mine (load average 4–9); its two orders agree.

## What it does

- **New op:** `fastmace.kernels.oeq_streaming.folded_streaming_convolution(a, W, x, sh, …)`.
  - Forward: it builds the per-edge weights `a @ W.T` (folded_linear's GEMM) and runs the streaming
    forward, as before.
  - Backward (`fastmace::streaming_conv_folded_backward`): the unchanged edge-owned kernel runs on chunks of
    `l2_chunk_edges` edges, a quarter of the L2 (1,536 edges for Q = 1280 float32 on the RTX 5060 Ti).
  - Each chunk's weight gradient goes into one of two chunk slots. The folded layer's Triton input gradient
    reads the slot on a side stream while the next chunk's backward runs.
  - The `[E, 1280]` weight gradient is never allocated: 268 MB on NbCl5, 430 MB on NaNbO3.
- **Why two slots and a side stream:** a single stream at 1,536 edges left the small input-gradient grids
  exposed. In OPT6-T01's `local-e` run the backward took 2.83 against 2.10 ms on NbCl5, and 4.58 against
  3.35 ms on NaNbO3.
- **Record:** `chunked_conv_backward` (MH-1 family). It requires `streaming_conv` and `fused_folded_backward`.
  - Exactness `exact`; `deterministic=False`, like `streaming_conv`.
  - Default off; planner rule `never_automatic`.
  - With it, the folded forward hands the folded layer's input `[h c, c]` to the new op. A reassociated first
    interaction keeps its own path.

## Checks

- **CUDA unit test** (`tests/gpu/test_kernels_oeq_streaming_cuda.py`, body `FOLDED_CHUNKED`): `run.log` job
  `check`, passed.
  - It compares against `folded_linear` followed by `streaming_convolution`, with 40 padded edges past every
    row, at chunks of 64, 150 (with `folded_constants` halves), 399 and 4,096 edges.
  - The message, `dsh` and `da` are bitwise equal; `dx` agrees to 1e-5, the order of its atomic sums.
  - A CUDA-graph replay gives the same results.
- **End-to-end smoke** (`smoke.py`, MH-1 float32 default plan against the plan with the record on, rattled
  3×3×3 cells): the force difference is no larger than between two runs of the default plan.

  | Cell | Max \|ΔF\| chunked − default | Max \|ΔF\| default − default | Max \|F\| |
  | --- | ---: | ---: | ---: |
  | NaCl | 2.8e-7 | 2.8e-7 | 0.24 |
  | NbCl5 | 1.4e-6 | 1.6e-6 | 2.5 |
  | NaNbO3 | 2.1e-6 | 2.3e-6 | 3.2 |

- **Force/stress gate** (`tools/force_stress_gate.py --engine fastmace --rewrites chunked_conv_backward=on`,
  three rattled and strained configurations per cell): **passed** on all three. Ratios against the float32
  calculator (E / F / S):
  - NaCl 0.04 / 1.16 / 0.31;
  - NbCl5 0.19 / 1.01 / 0.65;
  - NaNbO3 0.35 / 1.08 / 1.02.
- **CPU suite:** 4,482 passed. The one failure was `test_reproducible.py`'s list of non-deterministic
  records, now updated, and that file passes. The plan-text goldens were regenerated: the new field joins
  "rewrites off … (only an explicit 'on' enables it)".

## Full-step A/B

- **Machine:** local RTX 5060 Ti (sm_120, 36 SMs, 32 MiB L2), WSL2, `~/environments/mace_cueq_oeq`.
- **Source:** the pinned export `~/fastmace-opt6/src-08b0d004`.
- **Method:** `tools/ab_rewrites.py`, MH-1 hybrid float32 NVT at 300 K, 12 blocks of 300 steps.
  - a = the default plan (`chunked_conv_backward=off`), b = `chunked_conv_backward=on`.
  - Build order a-b with ABAB, then b-a with ABBA, one `--quiet` reservation, 27 October 2026 23:38–00:00 UTC.
  - Every run is valid: no compile while timed, no other GPU process.
- **Caveat:** a short NumPy job of mine (about 1.5 minutes, CPU only) overlapped the start of `nbcl5-ab`. That
  block has a stdev of 0.34 ms against 0.17–0.25 ms in `nbcl5-ba`. Its ratio agrees with the other order.

| Cell | a-b (ABAB) b/a | b-a (ABBA) b/a | Medians a → b (ms per step) |
| --- | ---: | ---: | --- |
| NaCl | 1.0057 | 1.0111 | 3.015 → 3.033; 3.082 → 3.117 |
| NbCl5 | **0.9771** | **0.9801** | 22.20 → 21.69; 21.86 → 21.42 |
| NaNbO3 | **0.9635** | **0.9651** | 19.78 → 19.06; 20.02 → 19.32 |

- **Against OPT6-T01's isolated subgraph:**
  - NaNbO3 gains 0.70–0.72 ms per step, against 0.93 ms in the isolated subgraph.
  - NbCl5 gains 0.44–0.51 ms, against 0.61 ms.
  - The rest is lost in the step, probably to the L2 contents the full graph leaves behind. This is not
    profiled.
- **NaCl** has 6,912 real edges in a padded edge bucket. Its `[E, 1280]` weights (35–42 MB) nearly fit the L2
  already. The chunking only adds launches, but the size rule built for the default did not remove the cost (see
  the default table above).
- **Against the plan's T02 bar** (b/a ≤ 0.98 on NbCl5 and NaNbO3 in both orders, NaCl neutral):
  - NaNbO3 passes;
  - NbCl5 passes at the limit (0.977 / 0.980);
  - NaCl is 0.6–1.1 % slower, so it is not neutral.
- **Size rule (built with the default):** chunk only when `E · Q · itemsize > 4 · L2`.

## Not done

- An sm_89 check of the L2 rule (RTX 4000 Ada, 48 MB L2 → 2,432 edges), after the workstation pause.
- The source of NaCl's +0.8 % (not profiled).
