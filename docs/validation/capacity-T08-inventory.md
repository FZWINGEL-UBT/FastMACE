# CAP-T08 checks: what the force call keeps, and the cheap float32 levers

The conductor's float32 checks of 30 October 2026, asked for by the owner, ahead of CAP-T08
([plans/capacity.md](../../plans/capacity.md)). They change no library default. Machine: Fabi-PC, WSL2, RTX 5060 Ti
16 GB (driver 610.88), `mace_cueq_oeq` (torch 2.14.1+cu130), `tools/ci/local_gpu.py run` reservations with the
idle check. Model MH-1 (`mh1-r2scan`), hybrid backend, float32, default plan unless noted. Raw records:
`results/cap-t08-inventory-20261030/`.

## 1. Saved-tensor inventory

**Tool:** `tools/bench/saved_tensors.py`. It builds the default plan's `fm.Simulation`, runs two steps, then makes one
more force call (forward and the force backward) of the engine's force field and records two things:

- **What the backward keeps.** `torch.autograd.graph.saved_tensors_hooks` sees every tensor an autograd node saves.
  Each is counted once per backing storage, with its dtype, shape, the frame that saved it and its `grad_fn`.
- **What is alive at the call's memory peak.** The caching allocator's history gives the largest live total and the
  blocks alive at that moment, as in `tools/bench/activation_profile.py`.

Shape classes come from the leading dimension:

- per edge: the padded edge count;
- per atom: a multiple of the atom count, which covers irrep-major `[(2l+1)·N, C]` node features;
- fixed: anything else.

`compile_mode=none` (eager) names the modules. The default compiled plan saves through AOT autograd's
`CompiledFunction`, which is what production keeps.

**Sizes.** The extra force call runs beside the captured graphs' pools, so it ran out of memory at 28,800 and 32,400
atoms, and at 21,600 with recompute off. The inventory therefore ran at:

- NbCl5 4×4×8: 9,216 atoms, 245,760 padded edges, 26.5 edges per atom at the 6 Å cutoff;
- NbCl5 5×5×12: 21,600 atoms;
- NaCl 3×3×3: 216 atoms, 32 edges per atom.

Bytes per edge and per atom are the same at 9,216 and 21,600 atoms. The 32k-atom figures below are those rates
times the 32,400-atom cell: 1.07M padded edges.

### What the backward keeps at the peak (default compiled plan)

| Cell | `recompute_conv_weights` | Peak above the model | Saved per edge | Saved per atom | Alive, not saved |
| --- | --- | ---: | ---: | ---: | ---: |
| NbCl5 9,216 | off | 3.22 GiB | 1.26 GiB (5,505 B/edge) | 1.50 GiB (174.5 kB/atom) | 0.46 GiB (53 kB/atom) |
| NbCl5 9,216 | on | 2.74 GiB | 0.09 GiB (385 B/edge) | 1.50 GiB (174.5 kB/atom) | 1.15 GiB (134 kB/atom) |
| NbCl5 21,600 | on | 6.52 GiB | 0.21 GiB (385 B/edge) | 3.51 GiB (174.5 kB/atom) | 2.80 GiB (139 kB/atom) |
| NaCl 216 | off | 0.09 GiB | 0.04 GiB (5,505 B/edge) | 0.04 GiB (174.5 kB/atom) | 0.02 GiB |

Eager keeps more per atom (199 kB per atom: Inductor fuses some of the saved intermediates away) and the same per
edge (5,609 B per edge off, 489 B on).

**Per atom now outweighs per edge.** Even with recompute off, the saved per-atom bytes (174.5 kB/atom) are larger than
the saved per-edge bytes at 26.5 edges per atom (5,505 B × 26.5 = 146 kB/atom). With recompute on, the per-edge term
is 10 kB/atom. On top of the saved per-atom tensors comes the backward's own working set: the feature gradients
alive at the peak, 134-139 kB/atom with recompute on.

### The per-edge holders

| Tensor (compiled plan) | dtype, shape | B/edge | At 32,400 atoms | Saved by | Needed for |
| --- | --- | ---: | ---: | --- | --- |
| **w**, the second interaction's per-edge TP weights | float32 `[E, 1280]` | **5,120 (93 %)** | **5.1 GiB** | `wprime_backward` (`_WPrimeConvolution`, the output of `fused_folded_forward`) | both: dx (feature adjoint) and dsh (coordinate) |
| radial basis `a` | float32 `[E, 65]` | 260 | 0.26 GiB | `radial_table_forward` → `edge_outer_sum` | coordinate (dr through the table) |
| spherical harmonics | float32 `[E, 16]` | 64 | 0.06 GiB | e3nn `SphericalHarmonics` | coordinate and the conv's dx |
| edge vectors, lengths, envelope, indices | `[E, 3]`, `[E, 1]` × 6, int32/int64 | 61 | 0.06 GiB | `get_edge_vectors_and_lengths`, the radial envelope | coordinate |

- **What CAP-T07's recompute already removes.** `recompute_conv_weights` drops w from the saved set: its backward
  rebuilds w per edge chunk with the radial table and `a @ weight.T`. That is all of w and nothing else; the other
  385 B/edge stay.
- **What recompute does not remove.** The forward still writes w edge-wide: `fused_folded_forward` returns it, and it
  is freed after the forward. So w is a forward transient of 5,120 B/edge. It is not at the peak in these cells,
  because the peak moves into the backward. That is why the measured saving at 32,400 atoms was 3,409 of 5,120
  B/edge (capacity-T07.md): the backward's working set of feature gradients becomes the new peak.
- **Nothing else per edge is above 0.5 GiB at 32k atoms.** The rest totals 0.4 GiB.

### The per-atom holders

From the eager attribution at 9,216 atoms; compiled shapes in brackets:

| Holder | dtype, shape | kB/atom | At 32,400 atoms | Saved by | Needed for |
| --- | --- | ---: | ---: | --- | --- |
| Symmetric contraction inputs, two of them | float32 `[N, 8192]` | 65.5 | **2.0 GiB** | cuEquivariance `segmented_polynomial_uniform_1d` (the product basis) | feature adjoint |
| Gated products of the fused linear-gate path | float32 `[N, k, 512]` for k = 1, 3, 5, 7 (compiled `[7N, 512]`, `[5N, 512]`, `[3N, 512]`, two each) | 61-74 | **1.9-2.2 GiB** | `fastmace/rewrites/fusion/modules.py:135` (the `gated_rows`/`segment_gemm` products) | feature adjoint |
| Convolution message and its cat | float32 `[N, 5120]`, `[N, 2048]` | 28.7 | 0.9 GiB | `cutoff_folded_forward` (`rewrites/mh1/modules.py:321`) | feature adjoint |
| Gate activations | float32 `[N, 512]` × 2 (silu), `[N, 1536]` × 2 (sigmoid) | 28.7 | 0.9 GiB | `torch.nn.functional.silu`/`sigmoid` in `fusion/modules.py:123-129` | feature adjoint |
| Linear-layer inputs | float32 `[N, 2048]` × 3 (compiled) | 24 | 0.75 GiB | the block linears (`rewrites/linear`) | feature adjoint |

- **The feature adjoint holds all of them.** None is needed for the coordinate gradient alone. They carry
  dE/d(features) back through each interaction, and that is what reaches the edges.
- **Every one is above 0.5 GiB at 32k atoms,** and the symmetric contraction and gated-product holders are about
  2 GiB each.
- **None can be dropped without recomputing its interaction.** The parameters do not require gradients, so AOT
  autograd already keeps only what dx needs. That makes them CAP-T08's checkpointing target: keep the node state
  between interactions and recompute each interaction in the backward. Item B below is that falsifier.

### Verdict for check 1

- **Per edge.** w is the only holder above 0.5 GiB: 5.1 GiB at 32k atoms, 93 % of the saved per-edge bytes.
  CAP-T07 removes it from the saved set. Rebuilding it inside the backward tile, and never writing it in the
  forward, would remove the forward transient as well and might cost less than CAP-T07's 9-10 %. That is the next
  falsifier (section 2).
- **Per atom.** Four holder groups are each above 0.5 GiB at 32k atoms, about 5.8 GiB together, plus a backward
  working set of about 4.5 GiB. CAP-T08 stays open. Its lever is per-interaction recomputation (item B), not a
  single tensor.

## 2. Rebuilding w inside the wprime backward (falsifier)

**Candidate.** w is never stored:

- the fused forward builds it on chip for the message only (`FusedFoldedForward(store_w=False)`: no `w` store, no
  padding-edge chunks);
- the backward builds it again per batch of edges, from the stored radial basis `a` (`[E, 65]`, kept anyway) and
  the folded weight.

**Stop rules** (conductor): at most 1.08× today's forward plus backward (CAP-T07's cost) makes it a capacity option
ahead of CAP-T07; at most 1.00× makes it a default candidate.

**Kernel** (`fastmace/kernels/oeq_streaming/wrebuild_backward.py`). The backward cannot keep the edge-owned layout
of `wprime_backward`: rebuilding w there would read all of W (1,280 × 65 floats) for every edge. So it uses the
fused forward's slice-owned layout:

- a block owns 32 channels × a part of the paths, with that slice of W in shared memory;
- a warp takes chunks of 64 edges and builds `w[part paths][batch]` with the forward's exact k-ordered `fmaf`
  sequence from the same staged `a`, so w is bitwise the forward's;
- per edge it computes the vendored backward's sums for its slice: `dx` added atomically (as today), and `dsh` and
  `dL/dr` reduced over the warp and added atomically over the slices.

**Measurement** (`tools/bench/wrebuild_bound.py`). The second interaction's real convolution arguments are captured
from one eager force call of the default plan; the parts are timed with CUDA events in 30 alternating rounds
(medians) on the RTX 5060 Ti:

| | NbCl5 4×4×8 (9,216 atoms, 245,760 edges) | NaNbO3 4×4×4 (2,560 atoms, 199,680 edges) |
| --- | ---: | ---: |
| `fused_folded_forward`, w stored (today) | 6.40 ms | 4.75 ms |
| the same without the w store | 4.68 ms | 3.67 ms |
| `wprime_backward`, reads w (today) | 7.43 ms | 5.95 ms |
| CAP-T07's chunked recompute backward | 16.14 ms | 13.95 ms |
| one read of w | 3.18 ms | 2.57 ms |
| `a @ W.T` by cuBLAS, all edges | 5.46 ms | 3.67 ms |
| rebuild backward, the forward's launch rule (2 path parts, 8 edges) | 13.03 ms | 10.39 ms |
| rebuild backward, 1 part, 8 edges | 16.81 ms | 13.61 ms |
| **rebuild backward, 1 part, 4 edges** | **11.42 ms** | **9.18 ms** |
| rebuild backward, 2 parts, 4 edges | 13.16 ms | 10.60 ms |
| **today: forward + backward** | **13.83 ms** | **10.70 ms** |
| CAP-T07: forward + recompute backward | 22.54 ms, **1.63×** | 18.70 ms, **1.75×** |
| optimistic bound: no-store forward + (backward − w read) + cuBLAS GEMM | 14.40 ms, 1.04× | 10.72 ms, 1.00× |
| **candidate: no-store forward + best rebuild backward** | **16.10 ms, 1.16×** | **12.85 ms, 1.20×** |

**Exactness:** dx, dsh and dL/dr agree with today's backward on the stored w to within 1.8–4.3 × 10⁻⁷ of each
quantity's largest value. That is rounding: the same w, another summation order.

**Verdict: stopped by the rule. It is 1.16–1.20× today, above the 1.08 bar.**

- **Where the time goes.** The cost above the bound is the slice split. Each of the four channel slices repeats the
  per-edge warp reduction of 17 values and adds them atomically, where today one 32-lane group owns an edge and
  writes it once.
- **Why the slice split cannot simply go.** Holding all channels in one block would need all of W, 333 KB, in
  shared memory.
- **What it would still give.** It is not a default candidate, but it dominates CAP-T07's mechanism:
  - it costs a third of CAP-T07's extra time (+2.3 ms against +8.7 ms for this subgraph at 9,216 atoms, about
    +2.5 % against +9 % of the step);
  - it also removes the forward's 5,120 B/edge transient.
- **Recommended follow-up, not done here.** Implement `recompute_conv_weights` with it: the forward with
  `store_w=False` plus `WRebuildBackward` (1 part, 4 edges), in place of the chunked cuBLAS recompute. That needs
  the gate (rounding-class), the whole-step A/B in both orders and a new capacity scan.
- **Not tried.** Holding `dsh`/`dr` partials in shared memory per chunk and reducing once per edge; float2 loads.

## 3. CUDA 13 shared-memory spilling and the stack limit (check 2)

**Why it was a candidate.** Under WSL2 the driver reserves a kernel's local memory (its stack frame) for every thread
the GPU can hold, and keeps it for the process's lifetime (capacity-T03.md). CUDA 13's `enable_smem_spilling` PTX
pragma moves register spills to shared memory, which would shrink that reservation.

**Census** (`tools/bench/kernel_resources.py`, NbCl5 3×3×3, default float32 plan, after the build and two steps).
For each NVRTC kernel the plan loaded, it reads:

- ptxas's report for its source, with the options it was compiled with: stack frame, spill stores and loads,
  registers;
- `cuFuncGetAttribute` of the loaded function: local bytes per thread, registers.

The kernels:

| Kernels | Count | Largest stack frame | Spills | Registers |
| --- | ---: | ---: | --- | --- |
| FastMACE streaming convolution (forward, backward, receiver backward, JVPs), interactions 0 and 1 | 10 | 0 B | none | 40-102 |
| `fused_folded_forward`, interactions 0 and 1 | 2 | 0 B | none | 126, 168 |
| `oeq_projected_backward_wprime` | 1 | 0 B | none | 80 |
| loaded functions, `LOCAL_SIZE_BYTES` | 8 | 0 B | | as above |
| OpenEquivariance JIT convolution kernels (forward, backward, double backward, fixups), interactions 0 and 1, with OEq's options (`--use_fast_math`) | 14 | 0 B | none | 4-168 |

The WSL2 reservation they add, local bytes × 55,296 resident threads (36 SMs × 1,536), is therefore 0 B. With the
pragma, the 21 kernels that compiled show the same 0 B stack and the same registers.

**Stack limit.** After setup it is 1,024 B, the CUDA default: nothing raised it. Setting it to 1,024 B again and
running two more steps left the memory outside the allocator at 1,213 MiB before and after: the context and module
images.

**Verdict: stopped. There is no spilling NVRTC kernel in the float32 plan, so the pragma has nothing to move, and
the stack limit is already the default.**

- **Not measured, so not claimed:**
  - The fused forward's and OEq's sources did not compile with the injected pragma; the failing sources are saved
    in `results/cap-t08-inventory-20261030/kernels/`.
  - No A/B or occupancy run was made, since without spills there is no candidate change.
- **What CAP-T03 measured was elsewhere.** Its WSL2 stack reservations came from:
  - OEq's own convolution kernels in float64, used only when rule 12 dropped `streaming_conv` (fixed by decision
    67);
  - Inductor's fused Triton kernels, which reach 5.3 KB, about 0.29 GiB reserved at 55,296 threads. Those are
    Triton-generated and outside this pragma.
- **The workstation step-time check was not run:** with no candidate, there is nothing to time.

## A. Packed FP32 FMA (FFMA2) on sm_120

`tools/bench/ffma2_probe.py` (CPU only) compiles an `fma.rn.f32x2` inline-PTX kernel with NVRTC 13.0 and reads the
SASS with nvdisasm:

| Target | `fma.rn.f32x2` becomes | Two scalar `fmaf` become |
| --- | --- | --- |
| `sm_120` (RTX 5060 Ti) | `FFMA R8, R2, R4, R6` + `FFMA R9, R3, R5, R7` | two `FFMA` |
| `sm_120a` | two `FFMA` | two `FFMA` |
| `sm_100a` (B200 class) | **`FFMA2 R2, R2.F32x2.HI_LO, R4.F32x2.HI_LO, R6.F32x2.HI_LO`** | two `FFMA` |

**Verdict: stopped by its first rule.** ptxas accepts the instruction for sm_120 but splits it into two FFMAs.
Consumer Blackwell has no packed FP32 FMA; only sm_100 does. So there is nothing to time on the RTX 5060 Ti, and
grepping our cubins for FFMA2 is moot.

## B. Interaction-boundary checkpointing (CAP-T08's per-atom side)

**Candidate.** Keep only the node state between the interactions, and rebuild each interaction's intermediates in
the backward. The first falsifier, as asked: in the eager tier, wrap interaction 1 (then interactions 0 and 1) and
its product block in `torch.utils.checkpoint` (non-reentrant, no RNG stash).

**Tool:** `tools/bench/checkpoint_falsifier.py`.

- **What it times.** The wrappers go on the modules' bound forwards, which the MH-1 rewrites replace per instance.
  The captured graphs were taken before any wrapper, so the measured unit is the eager force call (forward and the
  force backward) of the engine's force field, not the step.
- **How.** 8 alternating rounds in one process (medians), recording the call's time and its peak allocation above
  the memory held before it.

**NbCl5 4×4×13** (14,976 atoms, 397,256 edges at the cutoff), RTX 5060 Ti, `compile_mode=none`:

| Variant | Force call | Peak above the call's start | Largest force difference |
| --- | ---: | ---: | ---: |
| none | 629.3 ms | 5.93 GiB | 0 |
| interaction 1 + product 1 checkpointed | 673.0 ms, 1.07× | 6.53 GiB, **1.10×** | 1.0 × 10⁻⁶ eV/Å |
| interactions 0 and 1 + products checkpointed | 640.0 ms, 1.02× | 5.71 GiB, **0.96×** | 1.0 × 10⁻⁶ eV/Å |

**Verdict: stopped by the rule.** The peak falls by 3.6 % at best, and checkpointing one layer raises it by 10 %;
the bar was a 40 % fall.

- **Why the peak barely moves.** MH-1 has two interactions, and interaction 1 holds most of the node state, as
  well as w when recompute is off. The force call's peak is in interaction 1's backward. Checkpointing at the
  interaction boundary rebuilds exactly those intermediates at exactly that moment, beside the gradients already
  alive there, so the peak barely changes.
- **The extra 10 % with one checkpointed layer** is the checkpoint's own copy of the layer's inputs while the
  recomputed intermediates are alive.
- **What would be needed.** A per-atom memory mode would have to split inside interaction 1, keeping fewer of the
  per-atom holders of section 1 (the symmetric contraction's inputs, the gated products). That is a finer
  segmentation than the falsifier, and outside this check.
- **The step-time cost** was small here (1.02-1.07× of an eager force call), but cost does not decide it: memory
  does not fall.

## Summary

| Check | Result |
| --- | --- |
| 1. Saved-tensor inventory | **w is 93 % of the saved per-edge bytes** (5.1 GiB at 32k atoms); CAP-T07 removes it from the saved set. Per atom: 174.5 kB saved plus 134-139 kB of backward working set; four holder groups above 0.5 GiB each at 32k atoms, all for the feature adjoint |
| 2. w rebuilt in the wprime backward | stopped: 1.16-1.20× today's forward plus backward (bar 1.08); dominates CAP-T07's 1.63-1.75× and also removes the forward's w transient. A follow-up could implement `recompute_conv_weights` with it |
| 3. Shared-memory spilling, the stack limit | stopped: no NVRTC kernel in the float32 plan spills or uses stack; the limit is the 1,024 B default |
| A. FFMA2 | stopped: sm_120 splits `fma.rn.f32x2` into two FFMAs; FFMA2 is sm_100 only |
| B. Interaction-boundary checkpointing | stopped: the peak falls 3.6 % at best (bar 40 %) |

No library default changed. The code added here is opt-in only: `FusedFoldedForward(store_w=False)` (default
`True`), `wrebuild_backward.py`, and the tools.
