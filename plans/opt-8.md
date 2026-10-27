# Optimisation round 8: radial-table output straight into the streaming convolution

Owner request, 27 October 2026 (after radial tables, `fastmace-after-radial-tables`): algorithmic changes only; the
fused table → `w` → convolution kernel is the first item.

## Idea

Since `radial_tables` (owner decision 62) the second MH-1 interaction's chain is: table op → `a = [h c, c]`
`[E, 65]` → folded last layer `w = a Wᵀ` (cuBLAS, `W [1280, 65]`) → `w [E, 1280]` in DRAM → receiver-streaming
forward reads it; in the backward the edge-owned streaming kernel reads `w` again and writes `dw [E, 1280]` (in
L2-sized chunks, `chunked_conv_backward`), and the Triton skinny kernel reads `dw` for `da = dw W`. Nsight puts `w`
at 64–67 % of the streaming forward's L2 bytes and `w` + `dw` at 49–51 % of the backward's
(`docs/validation/opt7-T00-T02.md` on `fastmace/opt7-plan`). A fused kernel would build each edge's `w` on chip
from `a` and `W`, use it, and drop it; the backward would build `dw`, fold it into `da` on chip and drop it.

**Cost side.** The fused path does three `[E, 65] × [65, 1280]`-sized products where the chain does two (the
backward must rebuild `w` for `dx` and `dsh`): 7.0 G FMA each on NaNbO3 (84k edges), 4.4 G on NbCl5, in FP32 SIMT
(no TF32; the tensor-core path is a later option, see T02). `W` is 333 KB, more than the shared memory of a block
(about 99 KB on sm_120), so a block keeps a channel slice of `W` (32 channels × 10 paths × 65 = 83 KB) and the
other operands are re-read per slice.

**Sobek's reassociation** (arXiv 2607.18074: accumulate per receiver before the channel products, so no per-edge
weight tensor exists) pays when the per-edge weight comes from a short radial basis. Here the weight comes from a
65-wide `a`: accumulating `Σ_e a[e,k] · TP(x_s, sh_e)` per receiver is 65 × 5120 values per receiver and
65 × 5120 = 333k FMA per edge, against 83k FMA per edge to build the `w` row and use it (the tile approach). It is
4× the arithmetic of the tile approach before any memory is counted, so T01 measures the tile approach; the
reassociation is recorded as rejected on that count.

## Tasks

| Task | What | Stop rule | Status |
| --- | --- | --- | --- |
| **OPT8-T01** | Falsifier, no production code (`tools/bench/opt8_fused_conv.py`). Real operands (`x`, `sh`, senders, receivers, CSR rows, `dout`, `a`, the folded `W`) of one force call of the default float32 hybrid plan. (a) Ceilings: the streaming kernels with `w` and `dw` made register-only (stub), and the `w`/`da` products with their outputs kept in L2. (b) A prototype fused forward (`a`, `W` → message) and fused backward (→ `dx`, `dsh`, `da`), plain FP32 SIMT, against today's chain (cuBLAS GEMM + streaming forward; `chunked_conv_backward` = chunked streaming backward + Triton input gradient on the side stream). CUDA-graph replays with an L2 flush, same process, alternating; exactness against the chain | fused forward + backward ≥ 25 % faster than the chain on NaNbO3 **and** NbCl5; otherwise report in `docs/validation/opt8-T01.md`, commit, push, stop. If (a)'s ceilings already miss 25 %, (b) is not built | **done, not passed**, [report](../docs/validation/opt8-T01.md). The (a) ceilings were not bounds: cuBLAS on L2 chunks is slower than the prototype's own product. (b) is exact (message bitwise). Fused / chain: NbCl5 **0.82**, NaNbO3 **0.83**, NaCl 0.80. The forward halves; the backward is 1.12–1.13× the chain's chunked backward. **Stop**; whether a 4–7 % step gain justifies a record is an owner decision |
| **OPT8-T01b** | Owner follow-up: T01's fused forward that also stores `w [E, 1280]`; today's chunked backward unchanged on it. Forward alone and forward + backward against the chain, same method, 3 rounds; exactness of `w` and the message | forward + backward <= 0.90 of the chain on NbCl5 and NaNbO3 | **done, passed**, [report](../docs/validation/opt8-T01.md#t01b-fused-forward-that-also-stores-w-todays-backward-unchanged-owner-27-october-2026): `w` and the message bitwise equal. Forward 0.66 / 0.65; forward + backward NbCl5 **0.83-0.84**, NaNbO3 **0.83**, NaCl 0.95-0.96. No record or default built (owner); building one is an owner decision |
| **OPT8-T02** | Owner "Yes build it" after T01b: opt-in record `fused_folded_forward` (float32): T01b's fused forward that stores `w`, generated from the streaming schedule, launch rule from the GPU's shared memory and SM count; today's chunked backward unchanged. CPU tests, GPU exactness test against the chain | exact against the chain | **done**, [report](../docs/validation/opt8-T02.md): `w` and message bitwise the chain's (RTX 5060 Ti), gradients equal (dx at atomic-order level) |
| **OPT8-T03** | Gates for the record: force/stress gate (NaCl, NbCl5, NaNbO3), NVE drift 20 ps (NaCl, NbCl5), same-process A/B both build orders (NaCl, NbCl5, NaNbO3), POLAR-1-S and OMOL-0 A/Bs, an sm_89 A/B; report `docs/validation/opt8-T02.md`. Not a default (owner) | gate passes, drift unchanged | **done**: gate passed, drift unchanged; b/a NaNbO3 **0.928 / 0.927**, NbCl5 **0.970 / 0.964**, NaCl 0.991 / 1.000; Colab L4 (sm_89) NaNbO3 0.977, NbCl5 0.991; POLAR-1-S 1.004; OMOL-0 1.035 (slower; its default plan also fails `radial_tables`' self-check); murphy (RTX 4000 Ada) NaNbO3 **0.958 / 0.958**, NbCl5 **0.983 / 0.982**, NaCl **0.980 / 0.983** |
| **OPT8-T04** | Owner "Make it default for mh-1" (decision 64): rule `on_for_mh1_float32_unless_reproducible` | CPU suite | **done**: planner, docs, tests; [report](../docs/validation/opt8-T02.md#opt8-t04-default-for-mh-1-owner-decision-64) |

## Rules

General optimisations only (any crystal, composition and GPU; no per-GPU or per-crystal tables, no shape-keyed
caches), algorithmic only (no timestep change). Local GPU work inside `tools/ci/local_gpu.py run` reservations.
