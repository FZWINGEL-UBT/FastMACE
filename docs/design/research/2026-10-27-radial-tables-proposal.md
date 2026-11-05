# Radial tables: the proposal (27 October 2026)

**What this file is.** The "external proposal" behind `radial_tables` (`docs/validation/speed/opt6-T04.md:21`,
`plans/opt-6.md:26`).
- **Source:** a Claude research session that the owner ran on another account.
- **Its brief:** read FastMACE's optimisation documents and find more opportunities.
- **Its NumPy scripts:** `results/opt6-T04-radial-stub-20261027/table-check/` (`radial_cheb.py`,
  `radial_adaptive.py`, `radial_generic.py`, `ckpt.py`).
- **Recovered:** by the owner on 4 November 2026.

**Provenance findings:**
- **The session found the idea itself.** It calls it "The most radical lever I found". It cites no outside source for
  the tables. Its two sources (Fass et al. 2018; Leimkuhler & Matthews) are for its untested idea 2, the timestep.
- **The certificate came from it too:** the float32 table must beat the float32 network against float64. It proposed
  "Self-check them on a dense grid and refuse if the ratio exceeds 1". Decision 61 later made this the `certified`
  class.
- **It tested the outside spline proposal and rejected it.** In its second answer it ran an outside report's
  512-knot cubic Hermite spline (the method Symmetrix-XL also uses) on MH-1. Values passed. The r-derivatives were
  "1.4–34× worse than the float32 network", and the tables overflowed the L2.

The chat is kept below unchanged as a record. Its numbers are its own estimates and NumPy results, not GPU
measurements; [opt6-T04](../../validation/speed/opt6-T04.md) and [opt6-T05](../../validation/speed/opt6-T05.md) hold ours.

## The chat

```text
User: I am working on FastMace: https://github.com/FZWINGEL-UBT/FastMACE can you look at the docs at all of the
optimizations I have already tried and try to research more optimization opportunities?

Claude: I'll look for levers at the level of the model's math and physics rather than kernels, and test the most
promising ones numerically on a real MACE checkpoint where I can.

The most radical lever I found is one I could test here: replace MACE's radial networks with certified
per-species-pair polynomial tables. On MH-1 the tables are more accurate in float32 than the network they replace,
and they remove about 8–15 % of the step. Three other ideas are untested.

1. Certified radial tables (measured)
The insight. For a fixed (sender, receiver) species pair, everything between the edge length r and the folded last
layer's input [h·c, c] is a function of r alone. That covers the Bessel basis, the Agnesi transform, the species
embeddings, three LayerNorm/SiLU layers and the cutoff. A smooth function of one variable can be replaced by
piecewise Chebyshev series, fitted once in float64. The density network collapses the same way.

The research engine rejected tabulated radials on 15 October as "approximate". The new evidence: the float32 network
is itself the approximate one. In a NumPy float32 emulation of your arithmetic, it is only accurate to 5e-6–2.4e-4
relative, losing 7–12 bits. A table evaluated in float32 beats that, so it is a rounding-class rewrite like
segment_gemm.

What I ran. PyPI is blocked in this sandbox, so I parsed the checkpoints with a stub unpickler and re-implemented
MACE 0.3.16's radial code in NumPy. Every result compares both the float32 table and the float32 network against
float64 truth, at the per-edge TP weights w and their r-derivatives, which is what forces see.

Model | Cases | Table, r ∈ [0.6, 6] Å, degree 12, adaptive | Table error ÷ float32 network error (worst of value and d/dr)
MH-1 | 12 ordered pairs of O, Na, Cl, Nb × 2 interactions | 17–25 pieces, 56–83 KiB per pair and interaction | max 0.11–0.63, RMS 0.25–0.76; all 24 pass
MP-0 medium | species-agnostic, 1 table per interaction | 65–80 pieces, 211–260 KiB | 0.47–0.96; pass, but use a tighter tolerance for margin

NaNbO3 needs 9 pairs × 2 interactions ≈ 1.3 MiB of tables, which stays resident in the 32 MB L2.

Ceiling, from OPT-T10's breakdown:
                                                                     NaCl     NbCl5    NaNbO3
Replaced: radial hidden GEMMs and elementwise, forward and backward  0.30 ms  1.80 ms  3.08 ms
Table kernel (my estimate: [E, 65] write and read, L2 coefficient
reads)                                                               ~0.03    ~0.15    ~0.25
Net saving                                                           ≈ 9 %    ≈ 8 %    ≈ 15 %

Part of the 2–4 % "edge geometry, radial basis" share also goes, since the Bessel and Agnesi steps disappear.

How to build it:
Setup: fit the tables per interaction and per species pair, including the MC superset, in about a second.
Self-check them on a dense grid and refuse if the ratio exceeds 1.
Runtime: one Triton op maps r to [E, 65] plus the edge density. Its backward is dL/dr = Σ dL/dg · g′(r),
recomputed, so no saved activations. The folded last layer and OEq stay unchanged.
Short-range guard: set a flag if any r < r_floor and fall back for that step. ZBL dominates there anyway; the only
steep feature I found was Nb–Nb near 1.06 Å.
Because radial stages were the most precision-sensitive part in OPT4-T08, the table may lower your float32 force
error. The gate will show that.

Your falsifier:
Microbenchmark the op, forward plus backward. It must come in at ≤ 0.05 / 0.25 / 0.35 ms at E = 8,192 / 56,320 /
82,688.
Then run the force gate on all three crystals and NVE drift. Knot discontinuities are about 1e-8 relative.
Then the A/B.

2. Same configurational accuracy at 2–3× the timestep (untested)
FastMACE's NVT default is ASE's Langevin scheme, but you already have BAOAB. On a water cluster, Fass et al. found
BAOAB at about 6 fs had the configuration-space error other splittings showed at about 2 fs, while warning that bias
can matter below the stability limit. Leimkuhler and Matthews report the same low-bias property.

For static averages (energies, RDFs, lattice constants), that is a 2–3× cut in force calls per sampled
configuration. It needs no kernel work, though a fused fixed-cell BAOAB stage would keep the per-step speed.

Falsifier: run both integrators at 0.5, 1, 2, 3 and 4 fs on NaCl and NbCl5 at 300 K, and keep the largest dt whose
observables match the 0.5 fs reference within error. This doesn't help dynamics such as diffusion or quench rates.

3. Multiple time stepping split by interaction layer, with no training (untested, speculative)
MACE returns energy contributions per interaction. Use −∇(E_ZBL + E⁽¹⁾) as the fast force every step and correct
with −∇E⁽²⁾ every n steps. That is a 1.4–1.7× speed-up at n = 3–4, assuming the layer-1 path costs 0.3–0.4 of the
full call.

It may fail: E⁽²⁾ isn't trained to be slow, and RESPA needs it to be.

Falsifier: from a 2 ps trajectory, stop if F⁽²⁾ has more than about 1 % of the total force's spectral power above
1/(2n·dt), or if NVE drift at n = 2 exceeds 2× the reference.

4. The GPU's operating point (zero code)
Your profile shows the card pinned at its 180 W cap. An undervolted V/f curve on the Windows host often sustains
higher clocks at the same cap, and a memory offset helps the DRAM-bound edge side. Measure it with ab_step_time.
Your reproducible mode's bitwise trajectory test doubles as a stability check for silent errors.

Considered and dropped
Per-pair low-rank last layer: measured. It needs rank 48 of 65 on [0.6, 6] Å, so ≤ 2 %, and never materialising
[E, 1280] (OPT2-T01) stays closed.
Fitting the tables in the Agnesi variable t instead of r: measured, no better.
Parallel-in-time integration: your fit's fixed cost is only 0.32 ms, so the GPU is already saturated by per-size
arithmetic.
Strassen: K = 512 is too small, and its error bound is norm-wise.
eSCN/SO(2) rotation and Gaunt products: the convolution is memory-bound, and Gaunt products don't reproduce per-path
weights.
RT cores for the neighbour search: the build is about 1 % of the step after overlap_rebuild.
CPU offload: PCIe latency is too large against a 3 ms step.

The attached NumPy scripts reproduce the table test without torch: python -I radial_adaptive.py --model
mace-mh-1.model, and radial_generic.py with the MP-0 checkpoint path. Next step, if you want it: I prototype the
Triton table op and the setup fit as an opt-in rewrite record on a branch.

Sources:
Fass et al. 2018, quantifying configuration-sampling error in Langevin simulations
Leimkuhler & Matthews, robust and efficient configurational sampling via Langevin dynamics (arXiv 1304.3269)

[Attachments: radial_adaptive.py, radial_cheb.py, ckpt.py, radial_generic.py, radial_adaptive_0p6 (JSON),
Pasted text.txt (an outside research report, "Algorithmic Optimization Opportunities for FastMACE")]

User: What do you think about this report does it help?

Claude: It doesn't help much. The report is written against stock, unoptimised MACE, and for million-atom capacity.
Most of it is already inside FastMACE, was measured and rejected in your rounds, or targets scale you've ruled out of
scope. Its one relevant idea, tabulating the radial network, is the same lever I found, but its specific design
fails the accuracy test.

Its premise doesn't match FastMACE. It assumes host dispatch dominates, FP64 dgemm hotspots, materialised edge
gathers and no graph capture. Your step is a captured FP32 graph whose kernels run close to the FP32 and DRAM limits,
with 100–150 µs of gaps per replay.

Report item | Status in FastMACE
CUDA graphs, padded edge buckets, GPU skin list, device-side rebuild check | Core of the engine since phase 1
cuEq + OEq hybrid routing, streaming convolution | Your default backend
Species slicing and gather instead of 89-wide one-hot | Done: species_tables and OPT-T08
FP64 or compensated energy sum | Done: float64 energies
Morton atom order | Rejected: no gain
BF16/FP16 linears, TF32 | Rejected: fails your gate by 2–600× (OPT4-T08); TF32 runs at the FP32 rate on sm_120 anyway
SO(2) rotation, spherical-grid products, pruning ℓ | Either change the model or attack FLOPs in a memory-bound convolution
sphericart solid harmonics | Inside the 2–4 % "edge geometry" share, so ≤ 1–2 % at best
Three-pass recomputation (mace-million) | Trades time for memory to reach millions of atoms; slower on 216–1,944-atom cells, and your README routes > 100k atoms elsewhere
Distilled student with teacher repair | Not exact; it belongs in the same "approximate tier" as distilled MTS
Adjoint derivative for QEq solves | I found no linear solve in your long-range code, so it likely doesn't apply to POLAR-1 as you run it

The radial spline: right idea, wrong design. I tested its exact proposal, a 512-knot cubic Hermite spline of the
weights on (0, 6] Å, on MH-1 with the same harness as before.

Values pass. Its truncation error is 0.0–0.5× the float32 network's.
r-derivatives, which become forces, fail. They are 1.4–34× worse than the float32 network, even evaluated in
float64, which is the most favourable case. Nb–Nb and O–Nb are the worst.
"Exact to 10⁻¹³ eV/Å" is therefore false. A cubic spline at that spacing is nowhere near machine precision on this
network.
It also tabulates the wrong tensor. Storing the 1,280-wide weights with derivatives costs about 5 MB per pair per
interaction, around 66 MB for NaNbO3, which overflows the 32 MB L2. Each edge would also read about 20 KB of table.

The design from my last reply avoids both problems. It tabulates the 65-wide input of the folded last layer with
adaptive degree-12 pieces, about 70 KiB per pair. It beat the float32 network on values and derivatives in all 24
MH-1 cases, and in both MP-0 cases. The report does confirm that the direction is a natural one; it just shows why
the accuracy has to be certified per checkpoint rather than assumed.

Reliability. Several specifics look unverified or invented: a 1.5–2.5× embedding speed-up, the 10⁻¹³ claim, and a
"piwheels" FastMACE entry. Its speed-ups compare against e3nn or eager baselines, not against a graph-captured engine
like yours. I'd treat it as a literature list, not a plan.

So it changes nothing in the ranking. The certified radial table is still the lever to prototype first.
```
