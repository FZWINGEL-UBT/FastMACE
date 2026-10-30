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
