# OMAT-0 medium: approaches beyond kernel tuning

1. Current main is `b9f647ac`; the recorded H200 step is 3,422 µs, so 20% means **684 µs**.
2. I found no additional exact rewrite with a defensible large saving under the existing rules.
3. Two different directions merit CPU research if the owner permits approximation or a changed model.
4. The more practical one is to distil **only interaction 2 into a narrow equivariant operator**, keeping the rest of medium.
5. Its design target is **885–1,185 µs saved**, conditional on accuracy and a 500–800 µs replacement; these are not timing predictions.
6. The checkpoint-preserving alternative is **certified adaptive evaluation**: refine only contributions needed to meet a force-error enclosure.
7. That needs roughly **47–53% cost-weighted work eliminated** from interaction 2, including a 100–200 µs checking allowance.
8. A new CPU probe rules out FP16 weights plus a tiny exact correction: the residual still has rank 64.
9. A third, separately scoped direction can preserve medium's equilibrium distribution using surrogate proposals and exact-energy acceptance; it does not preserve MD dynamics.
10. No GPU, remote job, test suite, production edit, commit or push was performed. Scripts and JSON are beside this report.

## Evidence and limits

This follows [round 7](../omat0-ideas-round7/report.md), rather than reopening its small kernel changes. The complete optimisation ledger was re-audited. Main has not advanced since that report. The working checkout is `D:/Margraf/FastMACE/.claude/worktrees/omat0-ideas-round7`, detached at current main.

Measured reference windows and the newer interaction lab:

| Quantity | H200 | A100 SXM4 80 GB |
|---|---:|---:|
| MD step, mean of the two recorded build-order windows | 3,422.145 µs | 7,089.911 µs |
| 20% threshold | 684.429 µs | 1,417.982 µs |
| Sorted/species-ordered interaction-2 forward, same-process kernel lab | 601.600 µs | Unverified |
| Interaction-2 w′, that lab | 1,083.552 µs | Unverified |
| Sum of those two lab kernels | 1,685.152 µs | Unverified |

Sources: `results/slab-sorted-h200-20261108/h200-order/ab-order-{ab,ba}.json`, `h200-lab/spline_table_bound.json` in the same results folder, and `results/h200-fastpath-20261108/round6-a100/ab-omat0-medium-{ab,ba}-all3.json`. Lab and MD windows are different measurement contexts. The 1,685 µs total is a screening target, not a reconstructed current force-call profile. A100 has no current per-source split; its candidate savings cannot responsibly be estimated from H200 ratios.

The complete replacement budget for a 20% H200 win is consequently **1,685.152 − 684.429 = 1,000.723 µs**, using that lab target. Improving coefficient locality alone cannot get there: its remaining measured envelope is 169.056 µs forward and 23.776 µs backward. That envelope does not bound replacements which remove the tensor-product work itself.

[budgets.py](budgets.py) reproduces every budget below in [budgets.json](budgets.json). The 128 channels, four input components/channel, ten paths and 40 raw output components/channel come from the prior checkpoint census, `results/omat0-breakthroughs-assessment-20261108/codex-report/model.json`, for SHA-256 `d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a`.

## 1. Replace the expensive middle operator, keeping medium's surrounding model

**Mechanism.** Keep medium's first interaction, its density calculation, second symmetric contraction and final readout. Replace only interaction 2's wide message operator with a learned equivariant bottleneck:

`128-channel scalar/vector node features → 32-channel encoder → 32-channel edge convolution → 128-channel projected message`.

Use separate node encoders per input irrep, the same allowed angular paths, and a block decoder producing the original projected message. Fold decoding into the replacement of `linear_1`. Train these new components to reproduce medium's projected messages **and their derivatives**, then refine against the unchanged full model's energies, forces and stress. Forces must remain derivatives of a scalar energy. Teacher labels do not require new DFT, but training and validation remain substantial work.

This differs from replacing medium with OMAT-0 small: medium's first environment representation and nonlinear tail remain intact. It also differs from truncating the SVD of fixed trained weights: the narrow operator is jointly fitted with both encoders and decoders. Full rank of the old weights is evidence against exact equivalence, not against an approximate learned replacement.

**Where.** New experimental interaction module routed from `fastmace/rewrites/generic/modules.py:generic_interaction_forward` (294), at its convolution/projection boundary. The reusable kernel generators are `fastmace/kernels/oeq_streaming/spline_forward.py:render` (228), `SplineForward` (574), and `wprime_backward.py:render` (841), `WPrimeBackward` (1109). Keep the teacher checkpoint and production route unchanged; the result is a distinct, explicitly labelled model artifact.

**Arithmetic.** Channel-linear edge work and logical table width drop to `32/128 = 1/4`; this does not establish quarter-time kernels. For 5,000 atoms, counting each FMA as two FLOPs:

- New input encoder, forward plus reverse: `2 directions × 2 × 5000 × 4 × 128 × 32 = 0.32768 GFLOP`.
- Replacement decoder/projection, forward plus reverse: `2 × 2 × 5000 × 40 × 32 × 128 = 3.2768 GFLOP`.
- The corresponding current wide projection is `2 × 2 × 5000 × 40 × 128 × 128 = 13.1072 GFLOP`.

These are semantic counts, not issued 3×TF32 work or timings. A **500–800 µs complete replacement** is an engineering target to test, including encoder, decoder, backward and any extra packing. Charge all those costs against the old **edge-only** 1,685.152 µs envelope and credit none of the removed wide projection time. Conditional savings are therefore **885–1,185 µs**, or **25.9–34.6%** of the reference step. Failure to reach that target is entirely possible; there is no measured speed or accuracy result for this student. Its unconditional speed expectation is unknown.

**Accuracy / novelty.** Approximate and a changed model: **needs an owner decision**. I do not expect a 32-channel bottleneck to automatically satisfy the 3× float32 yardstick against medium. It needs a scientific teacher-to-student acceptance criterion if that gate fails. Generality requires training/validation across the intended composition and environment coverage; a single-crystal fit would violate the current rules. Closest ledger entries are `rad-low-rank`, `lin-svd` and the breakthrough report's generic distillation alternative. The new proposal is the specific operator boundary, retained teacher components, derivative targets and budget; distillation itself is established work. Hessian-supervised transfer between different force-field architectures is supported by [Amin et al., ICLR 2025](https://arxiv.org/abs/2501.09009), not a speed or accuracy guarantee for this construction.

**Cheapest falsifier.** A small CPU fit of the isolated projected-message map at widths 16/32/64, with held-out directional derivatives in coordinates and input features. If width 32 cannot satisfy the owner-selected accuracy criterion, stop that version before writing kernels. The fit must include short distances, strain and several chemistries; a training loss alone is insufficient. If this progresses to an authorised target-GPU lab, **stop if the complete replacement gain is under 685 µs on H200** for this large-gain objective. On A100 establish its actual target region first; 20% requires 1,418 µs whole-step saving. Effort: roughly **3–5 days for the CPU screen; 15–30 person-days plus training for a qualified prototype**.

## 2. Evaluate only contributions required by a force-error certificate

**Mechanism.** Preserve the checkpoint and construct a hierarchy of edge/path/channel blocks. Setup computes conservative envelopes for radial values/slopes and angular operators. Runtime computes cheap local operand bounds, selects a coarse set of contributions, and refines blocks until an enclosure on **final force and stress error** fits an allocated numerical budget. Ambiguous cases execute the existing full path. The purpose is to omit tensor-product work in both directions, not merely compress the table.

A useful implementation would make decisions over whole channel groups or receiver tiles and compact the remaining work. Masking individual lanes while issuing all instructions gains little. No periodic force reuse or per-crystal lookup table is involved.

**What has to be certified.** Write the second-stage contribution as `E(R)=f(U(R))`, with U including projected messages, density effects and the first-stage features required by skips. Let a tilde denote the cheaper evaluator. With a Hessian bound L for f along the segment between U and Ũ:

`||F − F̃||₂ ≤ ||J_U − J_Ũ||₂ ||∇f(Ũ)||₂ + ||J_U||₂ L ||U − Ũ||₂`.

This follows by adding/subtracting `J_Uᵀ∇f(Ũ)`. A 2-norm bound also bounds each force component, though it may be much too loose. Use local dependency bounds to avoid unnecessary factors of atom count. Include first-interaction derivatives in J_U; for stress include strain coordinates and the volume convention. Neither a small radial error nor a small direct edge-force error bounds the omitted backpropagation through first-stage features.

This equation is a specification for a certificate, **not an implemented cheap certificate**. Computing full Jacobians defeats the purpose. A viable implementation needs inexpensive positive-norm/interval propagation through the actual polynomial contractions and readout. The allowed tolerance must account for baseline error against float64: passing 3× does not supply an unused 3× error allowance. Hard changing masks can also create energy discontinuities; energy/force consistency across refinement boundaries requires explicit treatment.

**Where.** A new dispatch/certificate layer around `generic/modules.py:generic_interaction_forward`; sparse refinement variants of `spline_forward.py:SplineForward` and `wprime_backward.py:WPrimeBackward`. Setup envelopes belong with radial/spline preparation; runtime selection belongs to the force-field state, including rollback. Do not insert this as a bare module-level mutable cache.

**Bound.** If p is the fraction of **time-equivalent work** actually eliminated and C is checking, compaction and refinement overhead, the optimistic linear screen is `saving = p × 1685.152 − C`. Therefore:

| Checking/refinement allowance C | Minimum p for 20% of the MD step |
|---|---:|
| 0 µs | 40.62% |
| 100 µs | 46.55% |
| 200 µs | 52.48% |

As a conditional example only, p=55–70% and C=150 µs gives **777–1,030 µs saved**. No such p, certificate cost or final-error guarantee has been observed. Edge count is not a substitute for cost-weighted skipping, and divergent sparse work can scale worse than this expression. The actual expectation under the strict gate may be zero gain or a slowdown. A100's envelope remains unmeasured.

**Exactness / novelty.** Approximate whenever nonzero contributions are omitted; **needs an owner decision under the current rules**, even if later bounded and accepted by the gate. The checkpoint parameters remain unchanged. This is distinct from the ledger's exact-zero scans, `prec-channel-split`, `prec-precision-screen`, and `rad-low-rank`: those apply a fixed removal or arithmetic change, whereas this method refines according to an error enclosure and falls back. The full 3× force/stress gate remains necessary and does not replace the certificate.

**Cheapest falsifier.** On CPU, build bounds for the actual checkpoint and operand snapshots. Count the work that can be omitted while satisfying the remaining error budget, including propagated feature gradients. **Stop below 53% cost-weighted omission with a 200 µs overhead allowance**, or immediately if a valid bound needs the full expensive contraction/Jacobian. Do not spend GPU time on an uncertainty score that is only empirically correlated with error. A later full-region H200 lab stops below **685 µs net saving**, with all fallback cost included. Effort: **3–7 days for a certificate falsifier; 20–40 person-days if it survives**. This is the most uncertain proposal.

## A different objective: medium's equilibrium distribution without medium forces every step

If the actual goal is equilibrium averages rather than physical MD trajectories, use a cheap model to propose reversible, volume-preserving trajectory segments and accept/reject their endpoints using **medium's Hamiltonian**, `min(1, exp(−β ΔH_medium))`. This is surrogate-assisted hybrid Monte Carlo, with an exact target distribution in the ideal arithmetic/transition-kernel formulation. It does not require the surrogate's forces to match medium at float32 tolerance. [Nagai et al., Physical Review B 102, 041124](https://doi.org/10.1103/PhysRevB.102.041124) establishes the method; its performance on FastMACE is unknown.

The starting hooks are `fastmace/model/forces.py:MACEForceField.energy_only` (1103), `fastmace/mc/evaluator.py:MCEvaluator`, and `fastmace/mc/driver.py:MCDriver.run_block`. This needs a new proposal type and correct momentum, random-stream, rejection and neighbour-state handling. Ordinary Langevin segments cannot simply be put into the deterministic HMC acceptance formula. Freeze the surrogate during sampling or supply a valid adaptation scheme.

Small's **2,290 µs** H200 step is already measured (`results/h200-omat0-small-20261108/README.md`); that measurement is not a matched current-medium A/B. With 32 proposed small-model steps and a **budget assumption**, not an energy-only measurement, of 3,422 µs for the teacher endpoint:

`cost per proposed inner step = 2290 + 3422.145/32 = 2396.942 µs`.

Acceptance would need to exceed **87.6%** to improve the crude cost-per-accepted-path-length proxy by 20%. Actual efficiency must be judged by effective independent samples, not acceptance or purported physical nanoseconds. At 5,000 atoms, accumulated energy mismatch can destroy acceptance. This is an owner decision changing the simulation method, **not an answer to faster medium MD**, and is not included in the ranking above. First falsifier: a small CPU proposal/acceptance experiment; reject if endpoint energy mismatch makes acceptance negligible. An energy-only timing and mixing study are still missing. Rough implementation effort: **10–20 days plus scientific validation**.

## New exact ideas tested and rejected in this pass

These were new questions, even though they did not yield new optimizations. [algebra/residual_rank.py](algebra/residual_rank.py) loads the actual checkpoint in CPU PyTorch, with `CUDA_VISIBLE_DEVICES=` empty; [residual-rank.json](algebra/residual-rank.json) contains the results.

| Candidate | Evidence and bound | Why it stops |
|---|---|---|
| Fast FP16 backbone plus a tiny exact low-rank correction | For I2 W[1280,64], R=W−FP16(W) has rank **64**, s_min/s_max **0.2166**. Rank 62 is needed even to retain 99% of residual squared norm. | Rank-62 correction alone costs `62×(1280+64)=83,328` FMA/edge, versus the original `1280×64=81,920`, before the coarse product. Exact correction needs all 64; the 99% criterion is not gate accuracy. No reduced-work path. Original full rank did not itself settle this residual question; this new probe does for this construction. |
| Dense Hadamard/whitening channel basis that makes the edge operator cheaper | All **128** channel coefficient signatures across 10×64 coefficients are distinct; minimum pair distance **6.596**. | The edge operator is already diagonal in channels. A generic dense Q makes Q D_w Q⁻¹ dense, potentially moving channel work from C to C². Permutations/scalings retain cheap diagonals but remove no work. This is a coefficient-space argument, not proof of independence of all radial functions. |
| Exact correction after a fast student on every MD step | `E_medium = E_student + (E_medium−E_student)` still requires medium to evaluate the second term. | No saving without a cheaper correction representation or omitted work; evaluating both costs more. Occasional correction changes dynamics and is the already-known MTS route. |

I also do not reopen edge-frame/Gaunt, FMM/FFT, FP64 tensor-core adjoints or the whole-edge tape: the existing counting/lab closures still apply. A different producer for tensor-core angular products is a changed premise after SPLINE_W, but no credible total staging/work bound emerged; it is not a new large-gain recommendation.

## Decision

For the **unchanged checkpoint, strict numerical gate and unchanged dynamics**, there is still no evidence-backed large speedup to promise. A failed search is not a mathematical lower-bound proof.

For a larger research bet, I would first test the **32-channel interaction-2 replacement**, if model approximation is acceptable. If preserving the checkpoint is essential while allowing certified numerical approximation, spend a few CPU days on the **adaptive certificate**, with the strict omission threshold above. Both attack the actual dominant work. Neither should be presented as a 2× result or quietly adopted under the rounding-class rules.
