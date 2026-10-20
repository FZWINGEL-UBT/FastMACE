# OPT-T08: audit of the atom-level linear layers

Optimisation round 1, task OPT-T08 (owner request, 20–21 October 2026). Phase 1 is a CPU-only
audit of the compiled force pass, forward and backward. It looks for exact or round-off-exact
reductions of the atom-level linear layers. These are the per-irrep block linears: 43 % of
NaCl's B stage and 36 % of NbCl5's on MH-1 (2-T20a).

Sources:
- 2-T20a's profiles, `results/2-T20a-step-breakdown-20261020/{NaCl,NbCl5}/` (RTX 5060 Ti, sm_120)
  and `murphy/{NaCl,NbCl5}/` (RTX 4000 Ada, sm_89): `gemms.csv` and `kernel_sequence.csv`, one
  eager force call per crystal;
- MH-1's irreps, read from the checkpoint;
- MACE 0.3.16's forwards (`ScaleShiftMACE.forward`, `RealAgnosticResidualNonLinearInteractionBlock`,
  `EquivariantProductBasisBlock`, the readouts);
- FastMACE's rewritten modules (`BlockLinear`, `ResidualBlockLinear`, `GatedRows`,
  `ProjectedEmbedding`, `cutoff_folded_forward`).

No GPU was used.

## 1. Where the atom-level GEMMs come from (MH-1, hybrid float32 default plan)

The irreps, from `mace-mh-1.model`:

- node features: 512x0e (layer 0 input), then 512x0e+512x1o after layer 0 and 512x0e after
  layer 1;
- edge irreps (after `linear_up`): 128 per l;
- `linear_1`/`linear_res` output: 2048x0e (512 scalars + 3×512 gates) + 512 per l for l = 1, 2, 3;
- 6 heads.

| Module (per interaction) | Products per l (rows × K → N) | Calls (fwd + bwd) |
| --- | --- | --- |
| `node_embedding.linear`, `source_embedding`, `target_embedding` | one-hot `[n, 89] → 512` | 5 fwd |
| `ProjectedEmbedding` (source/target, 1a-T07) | `[n, 512] → 128` + bias | 2 fwd |
| `skip_tp` | int 0: l0 `512 → 512`; int 1: l0 `512 → 512` | fwd + bwd |
| `linear_up` | int 0: l0 `512 → 128`; int 1: l0, l1 `512 → 128` | fwd + bwd |
| `linear_1` + `linear_res` (`merged_residual`) | int 0: l0 `[msg 128 ∣ up 128] → 2048`, l1–l3 `128 → 512`; int 1: l0 `384 → 2048`, l1 `512 → 512`, l2 `384 → 512`, l3 `256 → 512` | fwd + bwd |
| `linear_2` (`gated_rows`, `segment_gemm`) | every l `512 → 512` (rows n, 3n, 5n, 7n) | fwd + bwd |
| `products.0.linear` | l0, l1 `512 → 512` | fwd + bwd |
| `products.1.linear` | l0 `512 → 512` | fwd + bwd |
| `readouts.0.linear` | l0 `512 → 6` | fwd + bwd |
| `readouts.1.linear_1`, `.linear_2` | `512 → 96`, SiLU, `96 → 6` | fwd + bwd |
| backward only | `[n, 512] → 89`: gradient with respect to `node_attrs` | 4 |

Measured per invocation (µs, one force call; local sm_120 / murphy sm_89):

| Product | NaCl local | NaCl murphy | NbCl5 local | NbCl5 murphy |
| --- | ---: | ---: | ---: | ---: |
| l0 `[n, 512] @ [512, 512]` | 19.4 | 17.2 | 87.6 | 105.8 |
| l1 `[3n, 512] @ [512, 512]` | 32.0 | 33.4 | 239.9 | 278.7 |
| l2 `[5n, 512] @ [512, 512]` | 51.3 | – | 370.6 | – |
| l3 `[7n, 512] @ [512, 512]` | 68.5 | – | 472.2 | – |
| `linear_up` l0 `[n, 512] @ [512, 128]` | 7.6 | 9.2 | 33.4 | 33.4 |
| one-hot embedding `[n, 89] @ [89, 512]` | 4.6 | 6.1 | 20.2 | 22.5 |
| `ProjectedEmbedding` addmm `[n, 512] @ [512, 128]` | 19.3 | 9.9 | 32.9 | 35.6 |

**What cannot shrink.**
- `linear_2` is a nonlinearity sandwich: the gate before it, cuEq's symmetric contraction after
  it. Its weights differ per l, and grouping them into one launch was rejected (OPT-T03).
- It costs 2 interactions × (l0 + l1 + l2 + l3) × (fwd + bwd): 684 µs on NaCl (43 % of the atom
  GEMMs) and 4.68 ms on NbCl5 (47 %), on the local card.
- No exact reduction removes it. Only cheaper arithmetic would, which is OPT-T01's ground, and that
  was rejected.

## 2. Candidates, ranked

The saving is GEMM kernel time removed, net of what replaces it. It is given as a share of the B
stage: local 3.39 / 26.98 ms, murphy 3.48 / 31.30 ms, for NaCl / NbCl5. It comes from
`gemms.csv` per invocation; the elementwise gathers that replace products are left out. The
estimator is in `results/opt-T08-linear-audit/estimate.py` (it reads only 2-T20a's files).

| # | Candidate | Local NaCl | Local NbCl5 | murphy NaCl | murphy NbCl5 | Class | Needs |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| C1 | Species tables for every species-only atom-level product | **4.1 %** (140 µs) | **2.1 %** (573 µs) | 3.8 % | 2.1 % | rounding (embedding gather: bitwise) | fixed head; `first_inputs_species_only` for the layer-0 part |
| C3 | Fold `products.0.linear` into its linear consumers | **3.0 %** (103 µs) | **2.4 %** (655 µs) | 2.9 % | 2.5 % | rounding | C1 (the layer-0 skip term becomes a table) |
| C2 | Fold the last layer's linear tail into the readout, one head | **2.0 %** (69 µs) | **1.2 %** (334 µs) | 1.6 % | 1.2 % | rounding | fixed head |
| C6 | No gradient with respect to `node_attrs` | 0.4 % (14 µs) | 0.2 % (53 µs) | 0.4 % | – | bitwise | nothing (most of it also goes with C1) |
| C4 | One product for `skip_tp` and `linear_up` (same input) in layer 1 | ≈ 0.3 % | ≈ 0.1 % | | | rounding (forward bitwise on the CPU) | absorbed by C2 and C3 |
| C5 | No `node_feats` descriptor concatenation | ≈ 0.1 % | ≈ 0.2 % | | | bitwise | absorbed by C3 |
| C7 | Readout columns of the unused heads | < 0.1 % | < 0.1 % | | | bitwise | absorbed by C2 |

All three together (C1 + C2 + C3) remove about **9 %** of NaCl's B stage and **5.7 %** of
NbCl5's on the local card (8.3 % and 5.8 % on murphy). They are independent arithmetic, so the
shares add, up to the gathers each adds.

Every candidate keeps every model input. None uses a per-GPU or per-shape table: the tables in C1
are functions of the model's own weights, indexed by species.

### C1: species tables (species-only atom-level work, recomputed every step)

**Where.** `node_attrs` is one-hot and never changes in MD; the plan fixes the species
(`fixed_species_superset`). These products and their backward run every step, although their
values depend only on each atom's species:
- `node_embedding.linear`, `[n, 89] → 512`;
- each interaction's `source_embedding`/`target_embedding` and their `ProjectedEmbedding`
  projections, `89 → 512 → 128`;
- in interaction 0, whose input is the embedding (`first_inputs_species_only`, already relied on by
  `reassociate_first`):
  - `skip_tp`, l0 `512 → 512`;
  - `linear_up`, `512 → 128`;
  - the `linear_res` half of the merged l0 residual product, K = 128 of 256.

**The rewrite.** At apply time, evaluate each of them once on the identity rows (one row per
element of the model). Keep the results as tables `[n_elements, width]`. In the forward, gather
the rows by species index (`node_attrs.argmax(-1)`, which cuEq's symmetric contraction already
computes). The products and their backward disappear from the step; one gather per table replaces
them.

**Saving.**

| | Local NaCl | Local NbCl5 | murphy NaCl | murphy NbCl5 |
| --- | ---: | ---: | ---: | ---: |
| GEMM time removed | 140 µs | 573 µs | 131 µs | 647 µs |
| Share of B | 4.1 % | 2.1 % | 3.8 % | 2.1 % |

The cat of `[node_feats, node_attrs]` and the 89-wide gradient (C6) also go away.

**Exactness.**
- The embedding gather is bitwise: a one-hot GEMM sums exact zeros.
- The projected and layer-0 tables are rounding class: a GEMM with M = 89 rows instead of n rows
  can sum in another order.

**Generality.**
- Any MACE: every model embeds species with a one-hot linear layer.
- The layer-0 part needs the trait `first_inputs_species_only`. OMOL-0's joint charge/spin
  embedding is constant per run but not per species, so it would need per-run tables (left out).
- Species swaps (phase 3 MC) only change the gathered index, not the tables.

### C3: fold `products.0.linear` into its consumers

**Where.** Layer 0's output is `x1 = SC0 @ W_p + sc0` (SC0: symmetric contraction; `W_p`:
`products.0.linear`, l0 and l1 `512 → 512`). Everything that reads `x1` is linear:
- interaction 1's `skip_tp` and `linear_up`;
- `readouts.0.linear`;
- MACE's `node_feats` output concatenation, which FastMACE never reads (C5).

**The rewrite.** Each consumer product `x1 @ W_c` becomes `SC0 @ (W_p W_c) + table(sc0 @ W_c)`.
`sc0` is species-only after C1, so its term is a species table.

**Saving.** The `products.0.linear` products leave the step, forward and backward. The consumers
keep their widths.

| | Local NaCl | Local NbCl5 | murphy NaCl | murphy NbCl5 |
| --- | ---: | ---: | ---: | ---: |
| GEMM time removed | 103 µs | 655 µs | 101 µs | 769 µs |
| Share of B | 3.0 % | 2.4 % | 2.9 % | 2.5 % |

**Exactness.** Rounding: the folded weights are computed in float64 at apply time and stored in the
model's dtype. `(x W_p) W_c` and `x (W_p W_c)` differ by round-off.

**Generality.** Any MACE whose layer-0 skip term is species-only: the residual, density and
nonlinear interactions all have `sc = skip_tp(node_feats)` on the embedding. Without C1 the fold
adds a product of the same size as the one it removes, so it saves nothing. That makes C1 a
prerequisite.

### C2: the last layer's linear tail and the readout head

**Where.** The last layer's output `x2 = SC1 @ W_p1 + skip_tp1(x1)` is read only by the nonlinear
readout. That readout's first layer is linear: `linear_1`, `512 → 96` (6 heads × 16, MACE masks the
other heads after the SiLU).

**The rewrite.** For the plan's one head, `h = SC1 @ (W_p1 W_r1[:, head]) + x1 @ (W_sk1 W_r1[:, head]) + b`,
both `512 → 16`. This removes:
- `products.1.linear`, l0 `512 → 512`;
- layer 1's `skip_tp`, l0 `512 → 512`;
- the five unused heads' readout columns.

**Saving.** Four l0 `512²` products and the `512 → 96` readout, forward and backward, are replaced
by four `512 → 16` products.

| | Local NaCl | Local NbCl5 | murphy NaCl | murphy NbCl5 |
| --- | ---: | ---: | ---: | ---: |
| GEMM time removed (net) | 69 µs | 334 µs | 54 µs | 377 µs |
| Share of B | 2.0 % | 1.2 % | 1.6 % | 1.2 % |

**Exactness.** Rounding (folded weights).

The estimate assumes the 16-column head slice. The built record keeps all 96 columns, so it saves
about 10 µs less on NaCl and 40 µs less on NbCl5.

**Generality.** The MH-1 class (MH-1, OMOL-0): there the last `skip_tp` is a linear layer. The
residual, density and OFF families use an element-dependent skip, so they are refused (section 3).
The built record keeps all heads' columns (`512 → 96`), so it needs no fixed head. The head slice
(`512 → 16`) is left for later.

### Smaller items

- **C6: the gradient with respect to `node_attrs`.**
  - What it is: MACE's `forward` calls `data["node_attrs"].requires_grad_(True)`. The compiled
    backward therefore computes the one-hot input's gradient through the embeddings and the cat
    `[node_feats, node_attrs]`: four `[n, 512] → 89` products, 14 µs on NaCl and 53 µs on NbCl5
    (0.2–0.4 %). Forces never read it.
  - The fix: one more source edit in FastMACE's pinned float64 forward (`fastmace.model.energy`,
    which already edits MACE's `forward`). It is bitwise.
  - Below 1 % on its own; it also comes with C1. Build it alongside C1, as a separate switch.
- **C4: one product for `skip_tp` and `linear_up` in layer 1.**
  - They read the same input, so the weights could be concatenated along N. Expected about 10 µs on
    NaCl and 30 µs on NbCl5 (< 0.5 %).
  - C2 and C3 replace both products anyway, so it is not built.
- **C5: the `node_feats` descriptor concatenation.** `torch.cat(node_feats_list)`, which Inductor
  fuses into the product's add as a second output. Gone with C3.
- **C7: the unused heads of `readouts.0.linear` (`512 → 6`).** Under 5 µs; part of C3's consumer
  fold.

### Checked and not candidates

- **Folding `linear_res` into `linear_up`:** `x (W_up W_res)` has K = 512 instead of the merged
  product's 128. More arithmetic, not less.
- **Folding `products.*.linear` into the symmetric contraction:** the contraction is channel-wise
  and the linear mixes channels, so they do not commute.
- **Folding `linear_2` into the gate or the contraction:** nonlinear on both sides.
- **The merged residual products:** already one K-concatenated product per output segment
  (`merged_residual`). The layer-1 l0 product (`384 → 2048`) has no shared input with another
  product.
- **Backward equivalents:** the folds above change the forward products. Their backward products
  shrink the same way through autograd. No backward-only merge was found beyond C6.

## 3. Which of the 19 labels each record applies to

Checked by each record's applicability test, on tiny models of every family and on stand-ins of the
19 labels (`tests/cpu/test_rewrites_species.py`).

| Labels | `species_tables` (C1) | `fold_product_linear` (C3) | `fold_readout_tail` (C2) | Why not |
| --- | --- | --- | --- | --- |
| `mh1-omat`, `mh1-mp`, `mh1-r2scan` | yes | yes | yes | |
| `omol0-xl` | no | no | yes | The joint charge/spin embedding is added to layer 0's input, which is then not species-only. The tail does not depend on it |
| `polar1-s`, `-m`, `-l` | no | no | no | PolarMACE: external fields and its own forward; only `ScaleShiftMACE.forward` is pinned |
| `mp0-*`, `mp0b3-medium`, `mpa0-medium`, `omat0-medium`, `matpes-*`, `off23-*`, `off24-medium` | no | no | no | Their interactions' `skip_tp` is element-dependent, a tensor product of the features with `node_attrs`, not a linear layer. C1 and C3 also need the MH-1 class's pinned forward. Folding the tail there would add a product, not remove one |

So the gains apply to the MH-1 labels: the benchmark model, NaCl and NbCl5.
- The residual, density and OFF families could in principle get C1 and C2 through their element-dependent
  skip: per species, it is a linear layer, which could be tabulated or composed per species.
- That is not built.

## 4. Phase 2: built (CPU), GPU pending

Three opt-in records in `fastmace/rewrites/species/`:
- `species_tables`, `fold_readout_tail` and `fold_product_linear`;
- each is a module replacement, with no forward edits;
- each is `RewriteConfig` "auto" = off (`never_automatic`), deterministic, and self-checked against what it
  replaced. A perturbed table or fold fails the check;
- each pins `ScaleShiftMACE.forward` and the MH-1-class interaction forward.

| Record | Class | CPU evidence (tiny MH-1, float64, `tests/cpu/test_rewrites_species.py`) |
| --- | --- | --- |
| `species_tables` (C1) | rounding by declaration; **bitwise on the CPU** | Energy and forces are bitwise equal to the default. A one-hot row times a weight is the weight row, and the tables are the same modules' outputs on the identity rows. On the GPU a GEMM over 89 rows may sum in another order than over n rows; the GPU check decides |
| `fold_product_linear` (C3) | rounding | Energy within 1e-12 and forces within 1e-10 relative of the default, with and without C2 |
| `fold_readout_tail` (C2) | rounding | The same, on MH-1 and OMOL-0 |

**Species superset and phase 3.**
- The tables have one row per element of the model (89 on MH-1), which covers every possible
  `species=` superset.
- They are gathered by species index at every evaluation, and no atom-to-species assignment is
  stored. Under C3, layer 0's output carries the one-hot itself.
- A test changes one atom's species between two evaluations of the same rewritten model. The result
  follows the unrewritten model, and C1 alone stays bitwise.

**C6 (the `node_attrs` gradient) is not built separately.** Under C1 the tables take no gradient, so the
four 89-wide backward products disappear. Only the cat `[node_feats, node_attrs]` and MACE's E0 lookup
`[n, 89] → 6` remain on that path, both a few µs.

**A CPU `Simulation`** with all three records follows the default plan within 1e-12 Å over 6 Langevin
steps.

**GPU (after the acceptance, on the conductor's go):** `results/opt-T08-linear-audit/run.sh`:
- the first force on and off (plain cuEq float64, hybrid float32);
- the force gate with all three, float32 target;
- the A/B of all three in both orders, NaCl and NbCl5, on both GPUs;
- attribution runs of C1, C1 + C3 and C2.
