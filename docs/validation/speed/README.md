# Speed reports

Step-time work on MH-1: optimisation rounds 1–10 (`opt*`), the float64 package (`f64-*`), precision screens (TF32, accurate tensor-core GEMMs, fp32 profile) and the single speed items of 29–31 October. Verdicts and decision numbers are indexed in the [optimisation ledger](../../optimisation-ledger.md).

Columns: file, date added, title, and the report's own verdict line or first sentence.

| File | Added | Title | Verdict / first sentence |
| --- | --- | --- | --- |
| [ab-constants-audit-20261030](ab-constants-audit-20261030.md) | 2026-10-30 | Audit of the `--a-constant` / `--b-constant` A/Bs (30 October 2026) | The segment_gemm tile-order A/B found a bug in `tools/ab_rewrites.py`. |
| [c1-contraction-fusion](c1-contraction-fusion.md) | 2026-10-30 | C1: fusing the symmetric contraction with a neighbouring GEMM (falsifier) | Idea C1 of plans/capacity-ideas-20261030.md, approved by the owner as a falsifier only. |
| [decision66-grouped-launch](decision66-grouped-launch.md) | 2026-10-29 | Decision 66: `grouped_launch` for MH-1 on the RTX 5060 Ti (29 October 2026) | Owner decision 66 (SPEC §18): per-GPU defaults where same-process whole-step A/Bs in both build orders measured a speed-up, by exact device name. |
| [edge-outer-sum-scaling](edge-outer-sum-scaling.md) | 2026-10-31 | edge_outer_sum backward: why it grows faster than the edges (31 October 2026) | Verdict: no fix proposed. |
| [edge-virial-falsifier](edge-virial-falsifier.md) | 2026-10-30 | Edge-virial stress falsifier (30 October 2026) | Verdict: stop. |
| [f64-T01](f64-T01.md) | 2026-10-27 | f64-T01: faster float64 MD (27 October 2026) | Verdict. Three measures together bring a float64 MD step to 0.55 (NaCl) and 0.42 (NbCl5) of the |
| [f64-T04](f64-T04.md) | 2026-10-27 | f64-T04: double-float falsifier for the FP64-bound kernels (27 October 2026) | Verdict: step 1 fails, so step 2 was not run. Correct double-float reaches 1.94× native float64 |
| [fp32-profile-20261028](fp32-profile-20261028.md) | 2026-10-28 | Fine-grained profile of the default float32 MD path (28 October 2026) | Owner request, 28 October 2026: time FastMACE's default float32 MD path, the step and the setup, in as much detail as possible, to guide the next optimisation round. |
| [gated-unpadded](gated-unpadded.md) | 2026-10-31 | gate_epilogue's input-gradient kernel without row padding (31 October 2026) | Verdict: default on the RTX 5060 Ti and the RTX 4000 Ada (decision 66). |
| [o1b-loop-overhead](o1b-loop-overhead.md) | 2026-10-29 | O1b: the `fm.md` / `Simulation` loop against the engine loop, measured again (29 October 2026) | Owner asked whether the loop overhead could be removed. |
| [opt-1](opt-1.md) | 2026-10-21 | Optimisation round 1: summary | > The combined A + B step-time change in §1 (NaCl ≈ −4.2 to −4.7 %, NbCl5 ≈ −13.6 to −14.4 %) is an > estimate, the product of the two separate A/Bs. |
| [opt-5](opt-5.md) | 2026-10-26 | Optimisation round 5: simultaneous CUDA-core and Tensor-core execution (26 October 2026) | Plan: `plans/opt-5.md`. |
| [opt-T01-tensor-cores](opt-T01-tensor-cores.md) | 2026-10-20 | OPT-T01: FP32-accurate tensor-core GEMMs (optimisation round 1) | Plan: `plans/opt-1.md`, OPT-T01. |
| [opt-T02a-gated-rows](opt-T02a-gated-rows.md) | 2026-10-20 | OPT-T02 stage A: `gated_rows` (residual products, gate and `linear_2` on rows) | Result: record `gated_rows` (`RewriteConfig.gated_rows`), opt-in when built and default on since 20 October 2026. |
| [opt-T02b-segment-gemm](opt-T02b-segment-gemm.md) | 2026-10-20 | OPT-T02 stage B: `segment_gemm` (`linear_2` writes its joined output in place) | Result: record `segment_gemm` (`RewriteConfig.segment_gemm`), on top of `gated_rows`. |
| [opt-T04-checkpoint-scan](opt-T04-checkpoint-scan.md) | 2026-10-21 | OPT-T04: checkpoint scan for exact skips | Result: nothing worth building. The learned weights of the ten benchmark models and the five MACE-OFF |
| [opt-T06-template-cache](opt-T06-template-cache.md) | 2026-10-20 | OPT-T06: converted-model template cache | Status: implemented and merged, 21 October 2026; engine-level check on murphy done (run `20261021c`). Plan `plans/opt-1.md`, OPT-T06, branch `fastmace/opt-T06`. Design: |
| [opt-T07-persistent-worker](opt-T07-persistent-worker.md) | 2026-10-21 | OPT-T07: the persistent subprocess worker | Results on murphy's CPUs (`mace_cueq_latest`, `CUDA_VISIBLE_DEVICES=`): |
| [opt-T08-linear](opt-T08-linear.md) | 2026-10-21 | OPT-T08: species tables and folded linears on the GPU (local RTX 5060 Ti) | Optimisation round 1, task OPT-T08 (`docs/design/speed/opt-T08-linear-audit.md`). |
| [opt-def-2](opt-def-2.md) | 2026-10-21 | OPT-DEF-2: three more defaults (owner decision, 21 October 2026) | The owner's words: "Make all 3 defaults". |
| [opt10-T01](opt10-T01.md) | 2026-10-29 | OPT10-T01: the gate's backward folded into `linear_2`'s input-gradient GEMM (29 October 2026) | Verdict: the stop rule is met on the RTX 4000 Ada for NbCl5 and NaNbO3. The fused kernel is 0.86 (NbCl5) and |
| [opt10-T04](opt10-T04.md) | 2026-10-29 | OPT10-T04: evict-first cache hints on the streaming kernels' `w` (29 October 2026) | Verdict: faster in both build orders on the RTX 4000 Ada, exact. NaNbO3 0.980 / 0.980, NbCl5 0.988 / 0.989 |
| [opt10-gate-epilogue](opt10-gate-epilogue.md) | 2026-10-29 | OPT10: the record `gate_epilogue`, opt-in, and its per-GPU default proposal (29 October 2026) | Decision rule: SPEC decision 66 (per-GPU defaults by exact device name, where same-process whole-step A/Bs in both |
| [opt2-T03](opt2-T03.md) | 2026-10-21 | OPT2-T03: the folded input gradient's tiles by a waves rule | Optimisation round 2 (`plans/opt-2.md`), candidate 3 of OPT-T09 (`docs/design/speed/opt-T09-step-breakdown.md`): the folded radial last layer's Triton input gradient … |
| [opt2-T04](opt2-T04.md) | 2026-10-21 | OPT2-T04: NaCl's fixed costs (opt-2) | 21 October 2026. |
| [opt2-T05-split-k](opt2-T05-split-k.md) | 2026-10-21 | OPT2-T05: split-K for small-cell atom products | Result: the candidate is rejected on sm_120. The gates pass, but NaCl does not get faster in |
| [opt3-T03](opt3-T03.md) | 2026-10-21 | OPT3-T03: the integrator stage in four kernels (opt-3) | 21 October 2026. |
| [opt3-T04](opt3-T04.md) | 2026-10-21 | OPT3-T04: NaCl's graph gaps (launch-count reduction); closed, not adopted | Optimisation round 3 (`plans/opt-3.md`). |
| [opt3-T05-layout-audit](opt3-T05-layout-audit.md) | 2026-10-21 | OPT3-T05: layout audit | Plan task OPT3-T05 (`plans/opt-3.md`). |
| [opt4-T01](opt4-T01.md) | 2026-10-24 | OPT4-T01: write-bound Triton kernel for the radial last layer's forward; cut at its first measurement | Optimisation round 4 (`plans/opt-4.md`, §3 candidate 1). |
| [opt4-T02](opt4-T02.md) | 2026-10-24 | OPT4-T02: POLAR-1's small kernels and the float64 scatter (24 October 2026) | Plan task OPT4-T02 (`plans/opt-4.md` §3 candidate 2). |
| [opt4-T03](opt4-T03.md) | 2026-10-24 | OPT4-T03 stage 1: 3×TF32 atom GEMMs, accuracy (optimisation round 4) | Verdict: PASS. |
| [opt4-T04](opt4-T04.md) | 2026-10-24 | OPT4-T04: `grouped_launch` on generic MACE (opt-4 candidate 4) | 24 October 2026. |
| [opt4-T05](opt4-T05.md) | 2026-10-24 | OPT4-T05: setup and capture time of an `fm.md` start | Status: measured, one option built, 24 October 2026; on by default since owner decision 59 (OPT4-T05b, see the last section). Plan `plans/opt-4.md` §3 candidate 5, task … |
| [opt4-T06](opt4-T06.md) | 2026-10-24 | OPT4-T06: the OEq streaming forward on NbCl5 (Nsight Compute verdict) | Verdict: no. Nothing built. The kernel runs at 94 % of DRAM peak (414 GB/s). Its 50 % occupancy |
| [opt4-T07-megakernel-probe](opt4-T07-megakernel-probe.md) | 2026-10-24 | OPT4-T07: single-SM co-execution probe (the megakernel gate) | Result: fail on the reference GPU (RTX 5060 Ti, sm_120) and on the A100 (sm_80). The L4 (sm_89) passes |
| [opt4-T08-precision-sensitivity](opt4-T08-precision-sensitivity.md) | 2026-10-24 | OPT4-T08: per-stage precision sensitivity of the force call | Result: no stage with a real share of the step passes the gate on its own in any 16-bit format (bf16 or |
| [opt5-T01-scheduler-probe](opt5-T01-scheduler-probe.md) | 2026-10-26 | OPT5-T01: Tacker-style scheduler-isolated and asymmetric co-execution probe | Plan: `plans/opt-5.md`, candidate 2 (`OPT5-T01`) and the probe phase of candidate 1 (`OPT5-T02`, `x3_pipe`). |
| [opt6-T01](opt6-T01.md) | 2026-10-26 | OPT6-T01: L2-resident edge chunks for the folded radial layer and the streaming convolution (26 October 2026) | Verdict: exact; NbCl5 passes the falsifier, NaNbO3 misses it narrowly, NaCl gains nothing. |
| [opt6-T02](opt6-T02.md) | 2026-10-27 | OPT6-T02: `chunked_conv_backward`, the streaming backward in L2-sized edge chunks (27 October 2026) | Verdict: exact, gate passed. Same-process full-step A/B: NbCl5 −2.0 to −2.3 %, NaNbO3 −3.5 to −3.6 %, NaCl |
| [opt6-T04](opt6-T04.md) | 2026-10-27 | OPT6-T04: radial tables, the accuracy check and the full-step stub ceiling (27 October 2026) | Verdict: the ceiling passes the stop rule on both large cells. A stand-in for a radial table op takes the |
| [opt6-T05](opt6-T05.md) | 2026-10-27 | OPT6-T05: `radial_tables`, certified Chebyshev tables for the MH-1 radial chain (27 October 2026) | Verdict: certified on every pair and interaction of MH-1 (ratios 0.15–0.30), gate passed, NVE drift unchanged. |
| [opt7-T00-T02](opt7-T00-T02.md) | 2026-10-27 | OPT7-T00 to T02: LunaTP and the streaming convolution backward (27 October 2026) | Verdict: none of the three hypotheses is worth a kernel on this GPU. LunaTP itself is slower on every large |
| [opt8-T01](opt8-T01.md) | 2026-10-27 | OPT8-T01: radial-table output straight into the streaming convolution (27 October 2026) | Verdict: the stop rule is not met. The fused forward and backward are exact. Together they are |
| [opt8-T02](opt8-T02.md) | 2026-10-27 | OPT8-T02/T03/T04: the record `fused_folded_forward`, default on for MH-1 (27 October 2026) | Verdict: exact, gate passed, NVE drift unchanged. |
| [opt9-T01](opt9-T01.md) | 2026-10-29 | OPT9-T01: the radial backward as a dot with tabulated `w'(r)` (28–29 October 2026) | Verdict: the stop rule is not met on the RTX 4000 Ada, so OPT9 stops here. On the A100 the added work is a |
| [opt9-T03](opt9-T03.md) | 2026-10-29 | OPT9-T03/T04/T05: the record `wprime_backward`, default on three GPUs (29 October 2026) | Plan: `plans/opt-9.md` (OPT9-T03 onward). |
| [relax-throughput](relax-throughput.md) | 2026-10-29 | Relaxation throughput: one engine across glass frames (A-T00 to A-T03), batching ceiling (B-T00) | Plan: plans/relax-throughput.md. |
| [segment-gemm-order](segment-gemm-order.md) | 2026-10-30 | segment_gemm tile order (30 October 2026) | Verdict: a row tile's column tiles now run next to each other when the product's A is larger than the L2. The |
| [speed-def](speed-def.md) | 2026-10-31 | Fifth list, items D, E and F, and the rebuild-`a` falsifier (30–31 October 2026) | The fifth list of plans/capacity-ideas-20261030.md, through the conductor, plus the conductor's follow-up "rebuild `a` in the backward instead of saving it". |
| [speed-small](speed-small.md) | 2026-10-30 | Three small speed items: pair repulsion, constexpr D, SH gradient (30 October 2026; none adopted) | The fourth list of plans/capacity-ideas-20261030.md, approved by the owner through the conductor. |
| [step-roofline](step-roofline.md) | 2026-10-31 | The large-cell step's heaviest kernels: roofline and the gated_input_grads tile screen (31 October 2026) | Verdict: STOP. |
| [tc-gemm-screen](tc-gemm-screen.md) | 2026-10-29 | Tensor-core GEMM screen (29–30 October 2026) | Owner request, 29 October 2026: screen whether FP32-accurate tensor-core GEMMs, and tensor-core work running beside FP32 CUDA-core work, give whole-step MD gains on our … |
| [tf32-screen](tf32-screen.md) | 2026-10-29 | TF32 screen for MH-1 MD (29 October 2026) | The owner asked on 29 October 2026 whether FastMACE's MH-1 MD could use TF32 after all, on one idle workstation GPU. |
| [wprime-group-order](wprime-group-order.md) | 2026-10-31 | Group-ordered wprime_backward falsifier (31 October 2026) | Verdict: not GO. Recommend stop. |
