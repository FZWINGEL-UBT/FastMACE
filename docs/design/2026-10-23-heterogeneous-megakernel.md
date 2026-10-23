# Long-term direction: a heterogeneous persistent megakernel (23 October 2026)

Status: **recorded, not planned.** The owner had it recorded on 23 October 2026, after an outside proposal. It
stays a research direction with a cheap gate. It is not a task in any round until that gate passes.

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
