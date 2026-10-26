# Long-term direction: a heterogeneous persistent megakernel (23 October 2026)

Status: **heterogeneous part closed** (owner, 24 October 2026: "Close the part", after OPT4-T07). Recorded on
23 October 2026 after an outside proposal. The persistent-kernel part without tensor-core co-execution stays
a research direction under OPT-T05's stopping rules.

**Gate result (OPT4-T07, 24 October 2026): fail.** The probe fails on the reference GPU (RTX 5060 Ti, sm_120)
and on the A100 (sm_80). The L4 (sm_89) passes only at the threshold, as a no-traffic ceiling. The
heterogeneous part is closed for these GPUs: see "Gate result" below and
`docs/validation/opt4-T07-megakernel-probe.md`.

## The idea

One persistent FastMACE kernel for the whole force call (forward and backward):
- an in-kernel task scheduler with dependency tracking;
- CUDA cores for geometry, spherical harmonics, elementwise work and precision corrections;
- tensor cores for corrected-precision GEMMs and compatible tensor-product work;
- asynchronous loads, prefetching and shared-memory reuse between tasks;
- later perhaps the integrator and the neighbour list as well.

The proposal cites two examples of the scheduling side:
- the Mirage Persistent Kernel (OSDI 2026): up to 1.7× on LLM inference;
- Hazy Research's "No Bubbles": a whole Llama forward pass in one kernel, about 50 % more throughput.
Neither is an equivariant GNN with a backward pass.

## Why it is not a next step

1. **Little to gain from removing boundaries.**
   - NaCl's force call has about 0.15 ms of gaps between kernels out of 2.77 ms (OPT-T09).
   - Round 1 bounded launch and transition savings at 8–16 % on NaCl and 1–2 % on NbCl5 (`plans/opt-1.md`,
     "Starting point"). CUDA graphs already remove the host overhead.
2. **Tensor-core and CUDA-core co-execution did not pay on sm_89 or sm_120**
   (`docs/validation/opt-T01-tensor-cores.md`):
   - two-stream overlap gave 0.77–0.85× on the large shapes (OPT-T01b);
   - 3×TF32 alone is slower than FP32 on those GPUs.
   Only the A100 (sm_80) has the throughput ratio for corrected TF32 (section 5 there), and its accuracy under
   cancellation is 7–14× worse than IEEE (section 7).
3. **No host libraries inside a single kernel:** cuBLAS, cuEquivariance and OpenEquivariance would all need
   device-callable replacements. Our own GEMMs and convolutions have lost to theirs so far (`docs/rewrites.md`).
4. **Generality (SPEC §9.1, the owner's rule):** a megakernel is specific to a model architecture and tuned per
   GPU. FastMACE runs MH-1, OMAT-0, POLAR, OMOL, MACE-OFF and future models on any crystal and GPU, and the
   ceilings above do not show a "massive" gain.
5. **Maintenance and determinism:** every upstream MACE or cuEquivariance change would need porting. Dynamic task
   order must keep fixed-order reductions to stay bitwise in the reproducible mode.

The gains so far came from fusion that cuts memory traffic around the GEMMs: OPT-T02's `gated_rows` and
`segment_gemm`, −9 to −11 % on NbCl5. That is the megakernel's real benefit, delivered one region at a time.

## The gate before any work

**Stage 1, a single-SM co-execution probe (about one day):**
- one kernel keeps a representative FP32 FFMA workload in some warps and independent tensor-core MMA work (TF32
  and BF16) in others;
- it compares the concurrent duration against both isolated durations;
- then it raises register use and shared-memory traffic towards a real GEMM's;
- it runs on sm_89 (L4 or the workstations), sm_120 (the local RTX 5060 Ti, or a G4) and sm_80 (A100).

**Pass rule:** the SIMT side keeps ≥ 90 % of its standalone rate while the MMA side adds enough useful
throughput to pay for an FP32-accurate decomposition. If it fails on sm_89 and sm_120, the heterogeneous part of
the idea is closed for those GPUs.

### Gate result (OPT4-T07, 24 October 2026)

`tools/megakernel_probe.py`, commit `1f724c9`, records in `results/opt4-T07-megakernel-probe-20261024/`. The
probe made the rule's thresholds explicit:
- `keep = simt_both / simt_alone ≥ 0.90`;
- `gain = (simt_both + mma_both / c) / max(simt_best, mma_best / c) ≥ 1.10`, against the better of all warps on
  CUDA cores and all warps on tensor cores;
- `c` = 3 for 3×TF32 and 6 for BF16×6. The `x3` variant does the 3×TF32 split in the kernel and pays for it
  (`c` = 1).

The most realistic variants, a 64×64 tensor-core warp tile (128 accumulators) beside an 8×8 SGEMM thread tile,
4 + 4 warps per SM:

| GPU | keep | gain, 3×TF32 upper bound | gain, `x3` (split measured) | tensor cores alone, FP32-equivalent / CUDA cores | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| RTX 5060 Ti (sm_120) | 0.91–0.92 | 1.07 | 1.04 | 0.42–0.44 | **fail** (gain) |
| L4 (sm_89) | 0.92–0.93 | 1.13 | 1.10 | 0.82–0.85 | pass, at the threshold |
| A100 (sm_80) | 0.58–0.78 | 0.87 | 0.92 | 2.2–2.5 | **fail** (keep and gain) |

- **sm_120:** TF32 tensor-core throughput equals FP32 FFMA throughput (256 FLOP/cycle/SM), so the tensor cores
  cannot pay for three products.
- **A100:** the tensor cores alone (3×TF32) are 2.2× the CUDA cores, and running FFMA warps beside them costs
  both sides.
- **L4:** the ceiling is exactly 10 %. The probe has no global loads, no barriers and no epilogue, and the two
  sides are perfectly balanced. Only the GEMM share of a step would gain. OPT-T01b's real two-stream version was
  0.77–0.85×.

**Decision (owner, 24 October 2026, as proposed):** the heterogeneous part is closed on sm_120 and sm_80. It cannot open on
sm_89 alone, because each later stage needs ≥ 10 % whole-step gain on the reference GPU. For a future GPU whose
TF32 tensor-core rate is several times its FP32 rate, re-run the probe (about 10 s of GPU time) first.

**Round 5 follow-up (`OPT5-T01` and `OPT5-T02`, 26 October 2026, `docs/validation/opt5-T01-scheduler-probe.md`):**
- **Warp-scheduler isolation (`sched_iso`):** pinning SIMT and MMA warps to disjoint warp schedulers (`w % 4`) drops
  `4×4` gain from `1.03–1.05` (`interleaved`) to `0.72` on `sm_120`, because each SM sub-partition owns its own
  private FFMA pipeline and its own private Tensor Core unit (`m_both = 127.5` of `255.2` FLOP/cycle/SM).
  Asymmetric interleaved ratios (`6×2`, `7×1`) reach `keep = 0.97–0.99` with `gain = 1.01–1.03`.
- **DeepSeek-style producer/consumer split (`x3_pipe`):** offloading the `cvt.rna.tf32` + `sub.f32` operand split
  from the MMA warps onto the SIMT warps (via non-blocking PTX `bar.arrive 1` and a 4-stage shared-memory buffer)
  lowers kernel register usage (`205 → 194`, 0 spill) and raises concurrent 3×TF32 MMA throughput by **+35.6 %**
  at `4×4` (`45.2 → 61.3` FLOP/cycle/SM, `mma_share` `0.56 → 0.76`) and **+21.3 %** at `2×6` (`60.1 → 72.9`,
  `mma_share` `0.73 → 0.89`). On `sm_120`, where TF32 MMA rate equals FP32 FFMA rate, the SIMT producer cost still
  makes net `gain = 0.85 < 1.0`. Dense single-SM GEMM co-execution on `sm_120` remains closed.

## What would come after a pass (stages 2–4 of the proposal)

1. A persistent linear-block engine: the irrep GEMMs of one interaction with their elementwise work. This
   overlaps OPT-T05 (regional persistence, `plans/opt-1.md`), which is the existing, smaller version.
2. A device-resident force graph with forward and backward dependencies.
3. Whole-step persistence, only after a persistent force kernel beats the CUDA-graph pipeline.

Each stage needs ≥ 10 % whole-step gain on the reference GPU and on one other production GPU before the next
starts. The objective is the shortest time to forces that pass the force gate, not utilisation.

## Nearer-term work that targets the same time

- OPT-T08's exact folds (species tables, readout tail, product-linear fold): built opt-in, with an estimated
  −6 to −9 % of the force stage. They wait for their GPU A/B after the phase 4 acceptance.
- OPT-T05, regional persistence of the fused atom block, as specified in `plans/opt-1.md`.
- On the A100 only: CUTLASS 3×TF32, pending the A100 production decision and the force-gate test of its
  cancellation behaviour.
