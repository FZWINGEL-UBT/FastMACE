# Optimisation round 7 (plan): the streaming convolution after LunaTP

Owner, 27 October 2026: evaluate LunaTP's ideas (arXiv 2610.18393, "LunaTP: Enabling Efficient and Scalable
Tensor Product for Equivariant MLIPs") on MH-1 and one generic MACE model, on uniform crystals and heterogeneous,
high-degree structures, with neighbour rebuild cost, graph capture and replay, force/stress checks and NVE drift;
bound every benefit by the measured share, `S_step = 1 / ((1 - f) + f / s)`. Proposed order (from an external
review the owner forwarded): **source-owned backward, then path-group traffic reduction, then derivative activity
specialisation**.

Branch for the plan: `fastmace/opt7-plan` (on `fastmace/f64-T01`, which is on `fastmace/opt6-T01`).

**Status, 27 October 2026: T00, T01 (a) and T02 done, no kernel built** ([report](../docs/validation/opt7-T00-T02.md)).
LunaTP is slower on every large cell (backward 1.39–1.53×); the sender atomics cost at most 5 % of the backward; half
of the backward's L2 bytes are not `w`/`dw`, but the receiver-owned remedy loses more parallelism than it saves. The
remaining lever is the `w`/`dw` traffic (idea 2 of the post-radial-tables plan, another session).

## 1. What the convolution costs now

`tools/step_breakdown.py`, float32 production defaults (with `chunked_conv_backward`), 300 timed steps after 300 of
equilibration at 1000 K, 8 profiled force calls; local RTX 5060 Ti, `mace_cueq_oeq`, 27 October 2026. Kernel times
of the vendored OEq streaming kernels (`oeq_projected_forward`, `oeq_projected_backward`) over the MD step's GPU
span. Raw: `results/opt7-plan-20261027/` (`share.sh`, `sharesum.py`, one directory per run).

| Model, cell (atoms) | step (ms) | conv forward | conv backward | rebuild (ms per step) |
| --- | ---: | ---: | ---: | ---: |
| MH-1 + tables, NaCl (216) | 2.78 | 3.3 % | 9.4 % | 0.21 |
| MH-1 + tables, NbCl3O | 5.82 | 4.7 % | 10.4 % | 0.25 |
| MH-1 + tables, NbCl5 (1,944) | 19.21 | 4.7 % | 10.3 % | 0.61 |
| MH-1 + tables, NaNbO3 (1,080) | 15.31 | 7.5 % | 18.6 % | 0.68 |
| OMAT-0 medium, NaCl | 1.71 | 5.3 % | 14.8 % | 0.15 |
| OMAT-0 medium, NbCl3O | 3.32 | 8.4 % | 18.8 % | 0.29 |
| OMAT-0 medium, NbCl5 | 9.99 | 9.1 % | 21.9 % | 0.42 |
| OMAT-0 medium, NaNbO3 | 11.63 | 11.2 % | 24.6 % | 0.80 |

The streaming convolution runs in the second interaction only: the first is reassociated (MH-1 and the generic
family), its node features are species-only. In float64 the convolution is about 6 % of the step (f64-T01).

**Ceilings** (`S_step`):

| Change | MH-1 NbCl5 | MH-1 NaNbO3 | OMAT-0 NbCl5 | OMAT-0 NaNbO3 |
| --- | ---: | ---: | ---: | ---: |
| backward 2× faster | 1.054 | 1.103 | 1.123 | 1.140 |
| backward free (s = ∞) | 1.115 | 1.229 | 1.280 | 1.326 |
| forward + backward 1.5× faster | 1.051 | 1.095 | 1.115 | 1.135 |

So the round is worth up to about 10 % on MH-1's heterogeneous cell and 12–14 % on the generic model, if a kernel
reaches 2×. Uniform small cells (NaCl) gain at most 5 %.

## 2. The three hypotheses against what is known

1. **Source-owned backward.** Today's backward is edge-owned: per edge it computes `dsh` and `dw` and adds the
   sender gradient `dx` atomically. The receiver-owned backward (owner decision 26 October 2026, opt-5 §3.10) was
   4.5–10.6 % *slower* per step: it keeps the receiver loop and still scatters `dx`. A source-owned backward walks a
   CSR by sender (for a symmetric neighbour list a permutation of the receiver CSR, built with the rebuild),
   accumulates `dx` on chip, and gathers `dout` by receiver. It is untested here. Unknowns: how much of the
   backward the `dx` atomics cost, and the cost of the random `dout` gather and of a second CSR per rebuild.
2. **Path-group traffic reduction.** The streaming forward runs at 94 % of DRAM peak on NbCl5 (OPT4-T06): it is
   memory bound, and most of its bytes are the per-edge weights `w [E, 1280]` (288 MB on NbCl5) and, in the
   backward, `dw`. Grouping paths saves reloads of `x`, `sh` and `dout` and accumulator registers, not `w`. The
   `w`/`dw` traffic is what the post-radial-tables plan's fused table → `w` tile → convolution kernel removes
   (`fastmace-after-radial-tables`, idea 2), which **another session owns**. Path grouping is therefore only worth
   it for the non-`w` share of the traffic, which is not measured yet.
3. **Derivative activity specialisation.** The vendored kernels already have activity switches for the forward and
   backward JVPs (`schedule.py:136–186`), used only for double backward, which MD never runs. In MD all three
   first derivatives (`dx`, `dsh`, `dw`) are consumed: the streaming convolution runs only where `x` depends on
   positions, `sh` and `w` always do, and stress uses the same three. What is left: the l = 0 spherical harmonic is
   the constant 1, so its column of `dsh` is computed and discarded. That is about 1/16 of `dsh` (MH-1's 16-wide
   `sh`), a small fraction of the backward. Lowest priority; folded into task T01 as a free change if the kernel is
   touched.

## 3. Tasks

| Task | What | Stop rule |
| --- | --- | --- |
| **OPT7-T00** | LunaTP as it is: install `LunaEquivariance` into a scratch local venv (copy of `mace_cueq_oeq`, dry run first), map MH-1's second-interaction convolution (`uvu`, per-edge weights, MH-1 and OMAT-0 irreps) onto it, and time forward and backward against the vendored streaming kernels on the recorded operands of NbCl5 and NaNbO3 (OPT6-T01's capture method: CUDA-graph replays, L2 flush). Exactness against the streaming kernels. | Neither kernel ≥ 1.3× faster on NaNbO3 for either model: record, and keep T01 only if its own falsifier passes |
| **OPT7-T01** | Source-owned backward falsifier. (a) Ceiling: the edge-owned backward with `dx` written to a per-edge scratch instead of atomically (wrong result, timing only), and Nsight Compute's atomic and DRAM breakdown. (b) A source-owned backward in the vendored template (sender CSR built in the rebuild, `dout` gathered), exact against the edge-owned one, with the l = 0 `dsh` column dropped. Timed per force call in graph replays. | Build only if (a) shows ≥ 25 % of the backward, and (b) reaches ≥ 1.25× on NaNbO3 for MH-1 or OMAT-0 (step ≥ 1.04 by the ceiling) |
| **OPT7-T02** | Traffic split of forward and backward (Nsight Compute, DRAM bytes by tensor: `w`/`dw`, `x`, `sh`, `dout`, outputs) on NbCl5 and NaNbO3. Decides between path grouping here and leaving the `w`/`dw` traffic to idea 2. | Path grouping only if non-`w` traffic ≥ 30 % of the kernel's bytes; otherwise hand the finding to the session that owns idea 2 |
| **OPT7-T03** | Only after T01 or T02 produced a kernel: the full evaluation below, then the owner's default decision | §4's gates |

T00 and T02 are measurements only; T01 (a) is too. Kernels are built only past their stop rules.

## 4. Evaluation protocol for anything built

- **Models:** MH-1 (`mh1-r2scan`, with and without `radial_tables`) and OMAT-0 medium (`omat0-medium`, generic
  family); float32 production plans, and the float64 hybrid plan for any kernel that also has a float64 path.
- **Structures:** uniform: NaCl 3×3×3, NbCl5 3×3×3; heterogeneous and high degree: NaNbO3 3×3×3 (three species,
  highest edge count), NbCl3O 3×3×3, the NaNbOCl4 swap-MC cell (mixed anions), and NaCl at 2000 K (high, varying
  receiver degree). Receiver-degree spread recorded per cell.
- **Neighbour rebuild cost:** the sender CSR must be rebuilt with every list rebuild; report rebuild time per
  rebuild and per step (stage `Neighbour-list rebuild (GPU)`), and rebuild counts, side by side.
- **Graph capture and replay:** capture count and time, replay time of the B graph, no extra captures during
  timing (the A/B tool's validity check), bucket changes across a rebuild.
- **Force/stress gate:** `tools/force_stress_gate.py --engine fastmace --rewrites <record>=on`, including the NPT
  (stress) cells; the kernel must be exact or within the float32 2× rule.
- **NVE drift:** `tools/bench/nve_drift.py`, 20 ps at 1 fs, 300 K, record off and on, NaCl and NbCl5.
- **Speed:** `tools/ab_rewrites.py`, same process, alternating blocks, both build orders, `--quiet` reservation on
  the local GPU; the bound of §1 next to each measured ratio.
- **Generality:** no per-GPU or per-crystal tuning; one Ada spot check when the workstations are back (paused).

## 5. Coordination and order

1. T00 and T01 (a) first: one afternoon of measurement each, no production code.
2. T02 next; its result goes to the owner of the post-radial-tables plan if `w` dominates.
3. T01 (b) only past its stop rule; T03 only for a built kernel.
4. GPU work in FIFO reservations; timed runs alone on the quiet PC. NAS sync stays paused; workstations stay off.
