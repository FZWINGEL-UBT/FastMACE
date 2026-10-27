# OPT6-T04: radial tables, the accuracy check and the full-step stub ceiling (27 October 2026)

Plan: `plans/opt-6.md`. Records: `results/opt6-T04-radial-stub-20261027/`.
- `table-check/`: the proposal's NumPy scripts, this check's `mh1.json` and `verify_against_mace.py`.
- `local-afdef58c/`: the stub A/Bs.
- Tool: `tools/bench/opt6_radial_stub.py`.

**Verdict: the ceiling passes the stop rule on both large cells.** A stand-in for a radial table op takes the
force call from:

| Cell | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
| NbCl5 | 20.8 | 19.3 | −7.2 % |
| NaNbO3 | 18.8 | 15.8 | −16 % |
| NaCl | 2.69 | 2.41 | −10.3 % |

These are upper bounds; the table's own arithmetic is not in the stub. The accuracy premise reproduces on our
checkpoint: the float32 table error is 0.13–0.57 of the float32 network's. Building the table op is the next
step (OPT6-T05).

## The idea (owner, 27 October 2026, from an external proposal)

**The mapping.** In every factored MH-1 interaction, for a fixed (sender, receiver) species pair, everything
from the edge length `r` to the folded last layer's input `[h c, c]` is a function of `r` alone:
- the Bessel basis and the Agnesi transform;
- the gathered species embeddings;
- the joint first layer and the hidden LayerNorm/SiLU layers;
- the cutoff.

The same holds for the edge density.

**The table.** Piecewise Chebyshev series, fitted once in float64 per pair and interaction, could replace
that whole chain. The folded last layer, the streaming convolution and everything after stay as they are.

**Why this is not the rejected screen.** The research engine rejected "tabulated radial networks" on 15
October as "approximate" (`docs/rewrites.md`). That reason is answered if the table evaluated in float32 is
closer to the float64 network than the float32 network is.

## Accuracy premise, reproduced

- **Re-implementation check:** `verify_against_mace.py` compares the proposal's NumPy version of MACE's radial
  code (`radial_cheb.Radial`, checkpoint read without torch) with mace-torch 0.3.16 itself, float64, CPU.
  - It checks the per-edge TP weights `w(r)` of both interactions of `mace-mh-1.model` (the local file) for
    the pairs Na–Cl, Nb–O, O–O and Nb–Nb at 400 distances in [0.6, 6) Å.
  - The worst relative difference is **1.1e-14**, so the re-implementation is MACE's arithmetic.
- **Table test:** `radial_adaptive.py --model ~/models/mace-mh-1.model`. It uses degree-12 Chebyshev pieces on
  [0.6, 6] Å, bisected until the float64 truncation error is below 2e-7. It compares value and d/dr at the w
  level against float64 truth.
  - All 24 cases pass (12 ordered pairs of O, Na, Cl, Nb × 2 interactions).
  - Table error ÷ float32 network error, worst of value and derivative: max **0.13–0.57**, RMS 0.26–0.79.
  - Size: 17–25 pieces, 56–83 KiB per pair and interaction.
- **Limit of this check:** the float32 network here is NumPy float32, not the GPU's Inductor kernels. The force
  gate on the GPU decides, against the float32 calculator.

## Stub ceiling

**What the stub does.** Side b runs the default plan with each folded interaction's radial section replaced by
the table op's own traffic:
- forward: `[h c, c] = [c · row, c]` written as `[E, 65]` from the cutoff `c`, and the edge density `c / 2`;
- backward: autograd reads the `[E, 65]` gradient and reduces it per edge.

Inductor then drops the unused Bessel and Agnesi terms and species embeddings. The stub forward is the live
`cutoff_folded_forward` with its radial lines replaced; the tool asserts the text, so it stops if that
function changes. Both sides build with `check=False`, since side b's forces are wrong.

**Why force calls, not MD steps.** Timing is per **force call**: the engine's force stage replayed as one
captured graph at the built configuration. An MD-mode smoke on `8c1c2e68` ran side b 2.46× slower: wrong forces
heat the cell and grow its edges. That record is `local-afdef58c/smoke-md-8c1c2e68.json`.

**Run conditions:**
- local RTX 5060 Ti (sm_120), `~/environments/mace_cueq_oeq`, FastMACE `afdef58c` (pinned export), MH-1 float32
  default plan;
- 12 blocks of 300 force calls, both build orders (ABAB, ABBA), one `--quiet` reservation,
  00:10–00:28 UTC on 27 October 2026;
- all runs valid: no compile while timed, no other GPU process.

| Cell | a-b: a → b (ms per force call), b/a | b-a: b/a | Saving | Share of the MD step* |
| --- | --- | ---: | ---: | ---: |
| NaCl | 2.695 → 2.417, **0.897** | 0.898 | 0.28 ms | ≈ 9 % |
| NbCl5 | 20.88 → 19.37, **0.928** | 0.928 | 1.50 ms | ≈ 7 % |
| NaNbO3 | 18.82 → 15.80, **0.839** | 0.843 | 2.95–3.02 ms | ≈ 16 % |

\* Against opt-4's step times (NaCl 3.08, NbCl5 21.9, NaNbO3 18.4 ms): an inference. The force call is most of
the step.

**What the ceiling leaves out:**
- the table op's per-edge Chebyshev evaluation: about 13 coefficients × 65 outputs, value and derivative, from
  L2-resident coefficients; about 0.1–0.2 G FMAs per force call on NaNbO3;
- the piece lookup.

Neither moves the `[E, 65]` traffic that the stub already pays.

**Against the research engine's 15 October ceiling** (2.3–4.8 %): that one removed only the hidden layers. This
one also removes the joint first layer, the basis, the species gathers and the density network, which the
table replaces as well.

## Next (OPT6-T05)

1. **Setup:** fit per interaction and ordered species pair of the plan's species superset (MC included), in
   float64.
   - The table's lower end should sit below every reachable distance, for example 0.3 Å, rather than a
     mid-step fallback inside a CUDA graph.
   - Self-check on a dense grid against the float64 network; refuse if the float32 table is worse than the
     float32 network.
2. **Runtime:** one Triton op `r → [h c, c]` and the edge density. Its backward recomputes `g'(r)`, so it saves
   no activations.
3. **Class and gate:** a new exactness class, "bounded approximation (below float32 rounding)", is the owner's
   decision. Then the force gate on the three cells, NVE drift, and the full-step A/B. It composes with
   `chunked_conv_backward`: the table feeds that op's `[h c, c]`.
