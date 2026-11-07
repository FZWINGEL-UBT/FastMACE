# Research notes

Dated research notes and prompts for outside research agents (megakernels, mixed precision, radial tables, OMAT-0 on the A100).

Columns: file, date added, title, and the report's own verdict line or first sentence.

| File | Added | Title | Verdict / first sentence |
| --- | --- | --- | --- |
| [2026-10-20-megakernel-research](2026-10-20-megakernel-research.md) | 2026-10-20 | Persistent megakernel for the force call: external research report (20 October 2026) | Recommendation: prototype one fused atom-level block first. Do not commit to a complete MACE megakernel yet. Your measurements justify investigating small-GEMM … |
| [2026-10-23-heterogeneous-megakernel](2026-10-23-heterogeneous-megakernel.md) | 2026-10-23 | Long-term direction: a heterogeneous persistent megakernel (23 October 2026) | Status: heterogeneous part closed (owner, 24 October 2026: "Close the part", after OPT4-T07). Recorded on |
| [2026-10-26-fp16x3-and-hybrid-gemm](2026-10-26-fp16x3-and-hybrid-gemm.md) | 2026-10-26 | Exponent-Shifted `3×FP16` Tensor Core GEMMs: Standalone (`sm_80` / `sm_89`) and Fused SIMT+MMA (`sm_120`) | Date: 26 October 2026 (`OPT5-T06` / `OPT5-T08`). |
| [2026-10-26-megakernel-mixed-precision-research-prompt](2026-10-26-megakernel-mixed-precision-research-prompt.md) | 2026-10-26 | Research prompt: further megakernel and mixed-precision opportunities for FastMACE (26 October 2026) | Paste everything below the line into a research-capable model with web access. |
| [2026-10-27-radial-tables-proposal](2026-10-27-radial-tables-proposal.md) | 2026-11-04 | Radial tables: the proposal (27 October 2026) | What this file is. The "external proposal" behind `radial_tables` (`docs/validation/speed/opt6-T04.md:21`, `plans/opt-6.md:26`). |
| [2026-11-04-gemini-research-prompt](2026-11-04-gemini-research-prompt.md) | 2026-11-04 | Research prompt: further speed-ups for MACE-OMAT-0 molecular dynamics in FastMACE (4 November 2026) | *Written for an external research agent. |
| [2026-11-05-new-omat0-a100-paths](2026-11-05-new-omat0-a100-paths.md) | 2026-11-05 | Next-Generation Architectural Paths for MACE-OMAT-0 on NVIDIA A100 (5 November 2026) | Context: The current FastMACE OMAT-0 medium baseline on the A100-SXM4-80GB achieves ~13.6 ms per force call (~2.73–2.95 µs/atom/step) on Benchmark_loup SrTiO3 10×10×10 … |
| [2026-11-07-polar-megakernel](2026-11-07-polar-megakernel.md) | 2026-11-07 | A megakernel for MACE-POLAR-1 on the RTX 5060 Ti: bound and scope (7 November 2026) | Status: design note; no kernel adopted. Wide GEMMs block a complete megakernel on sm_120; the polar remainder overlaps off-polar-alex switches. |
