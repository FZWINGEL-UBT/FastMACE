# OPT6-T05: `radial_tables`, certified Chebyshev tables for the MH-1 radial chain (27 October 2026)

Plan: `plans/opt-6.md`. Ceiling: [opt6-T04.md](opt6-T04.md). Code: FastMACE branch `fastmace/opt6-T01`, run on
the pinned export `68fdd7d1` (smoke on `91c1822b`, the same library code). Records:
`results/opt6-T05-radial-tables-20261027/` (`local.sh`, `smoke.py`, `smoke-91c1822b.log`, `local-68fdd7d1/`).

**Verdict: certified on every pair and interaction of MH-1 (ratios 0.15–0.30), gate passed, NVE drift unchanged.
Same-process full-step A/B, on top of the new defaults (with `chunked_conv_backward`), on the local RTX 5060 Ti:**

| Cell | b/a, a built first | b/a, b built first | Medians, ms per MD step |
| --- | ---: | ---: | --- |
| NaCl | **0.901** | **0.898** | 2.99 → 2.70 |
| NbCl5 | **0.915** | **0.916** | 21.39 → 19.57 |
| NaNbO3 | **0.811** | **0.807** | 19.47 → 15.78 |

The record is opt-in (`never_automatic`). Making it a default is the owner's decision.

## What it does

`fastmace.rewrites.radial`, record `radial_tables`, exactness `certified` (SPEC §6.1, owner decision 61).

- **Fit.** Per interaction and ordered species pair of the plan's superset, the chain `r → [h c, c]` and the
  edge density are fitted in float64 on the CPU from copies of the model's own modules (its float32 weights).
  - The chain covers Bessel, Agnesi, the projected embeddings, the first layers, the hidden layers and the
    cutoff.
  - The pieces are degree 10 over [0.3 Å, r_max), dyadically refined until values **and r-derivatives** are
    within 1e-7 of their local scale.
- **Certificate.** On 12,000 lengths, the float32 table and the float32 network chain are both compared with
  the float64 chain, and the table's error is divided by the network's. The table is evaluated by the kernels'
  own arithmetic: forward, and the backward with one-hot gradients.
  - Compared quantities: `w = [h c, c] W^T`, the edge density, and both their r-derivatives.
  - Regions: the whole domain and [1 Å, r_max).
  - The self-check fails, and rule 13 turns the record off, if any ratio exceeds 1.
- **Run time.**
  - One Triton op maps r and the pair index to `[E, 65]` and the density. The piece comes from one bin
    gather, and Clenshaw sums the series.
  - Beyond the cutoff (the padded edges too) the outputs are exactly zero. Below 0.3 Å the table is evaluated
    at 0.3 Å with a zero slope.
  - The backward recomputes the derivative series, so it saves nothing but r and the pair.
  - `LengthsEmbedding` takes the radial embedding's place and hands the interactions the lengths.
- **Planner.** `never_automatic`; refused off float32 (`radial-tables-needs-float32`).

### The first fit only bounded values

The fit first held values only. On the tiny CPU model the derivative error then made the certificate fail
(ratio 1.03–1.07, the derivative of `w`). Holding the derivative series to the same tolerance fixed it: the
tiny model's ratios fell to 0.06–0.13 at degrees 8–12. Degree 10 was chosen (36 pieces on the tiny model;
degree 12 gives the same ratios with 24).

## Results

**Certificates** (the applied detail of `smoke.py`; pieces, pairs and worst ratio per interaction):

| Cell | Interaction 0 | Interaction 1 |
| --- | --- | --- |
| NaCl | 29 pieces, 4 pairs, **0.19** | 47 pieces, 4 pairs, **0.15** |
| NbCl5 | 32 pieces, 4 pairs, **0.16** | 53 pieces, 4 pairs, **0.15** |
| NaNbO3 | 164 pieces, 9 pairs, **0.30** | 65 pieces, 9 pairs, **0.18** |

**Forces against the float64 plan** (`smoke.py`, one rattled 3×3×3 cell). The tables are as accurate as the
network, which the certificate also says at the radial level:

| Cell | Default, max \|ΔF\| (eV/Å) | Tables, max \|ΔF\| | Default RMS | Tables RMS |
| --- | ---: | ---: | ---: | ---: |
| NaCl | 2.56e-6 | 2.20e-6 | 8.2e-7 | 6.9e-7 |
| NbCl5 | 7.02e-5 | 6.70e-5 | 9.5e-6 | 9.6e-6 |
| NaNbO3 | 9.56e-5 | 9.77e-5 | 1.12e-5 | 1.09e-5 |

**Force/stress gate** (`tools/force_stress_gate.py --engine fastmace --rewrites radial_tables=on`): **passed**.
Ratios against the float32 calculator (E / F / S):
- NaCl 0.08 / 0.98 / 0.36;
- NbCl5 0.11 / 1.05 / 0.41;
- NaNbO3 0.32 / 1.00 / 0.60.

**NVE drift** (`tools/bench/nve_drift.py`, 20 ps at 1 fs, 300 K, the default plan with the record off and on).
The RMS and range are unchanged; the drifts stay at the level of 1e-6 meV/atom/ps:

| Cell | Off: drift (meV/atom/ps), RMS, range | On |
| --- | --- | --- |
| NaCl | 4.9e-7, 2.7e-4, 6.4e-3 | 2.1e-6, 2.7e-4, 7.0e-3 |
| NbCl5 | −1.3e-6, 3.8e-4, 1.25e-2 | 1.7e-6, 3.8e-4, 1.27e-2 |

**A/B** (`tools/ab_rewrites.py`, MH-1 hybrid float32 NVT at 300 K, 12 blocks of 300 steps, both build orders,
one `--quiet` reservation from 09:14 to 09:35 UTC; every run valid): see the verdict table.
- The gain is larger than the stub ceiling on NaNbO3 (−19 % per step against −16 % per force call). Two
  differences could explain it; this is an inference, since neither was measured:
  - this A/B's default includes `chunked_conv_backward`;
  - the stub kept the radial embedding's cutoff term and the `[E, 65]` concatenation.
- Setup: side b's build took 24–36 s against 23–36 s for side a (including the fit and the certificate; one
  process, cache state varies).

**Tests:**
- `tests/cpu/test_radial_tables.py` (tiny MH-1 model, CPU):
  - the record's fields;
  - certificate and self-check pass;
  - forces within 1e-4 of the network and no further from float64;
  - zeros beyond the cutoff;
  - gradient against finite differences;
  - float64 refused, by the record and by the planner.
- `tests/gpu/test_radial_table_cuda.py`: the Triton kernels against `evaluate` (values, backward, the domain's
  ends, a CUDA-graph replay).

## Limits and not done

- **Below 0.3 Å** the tables are evaluated at 0.3 Å. No MD edge gets close (ZBL), but an edge there would see
  the network's value at 0.3 Å with zero slope, not the network.
- **Scope:** the MH-1 class only, where every interaction is factored. The generic families' radial networks
  (MP-0, OMAT-0 and others) would need their own record.
- **Not measured:**
  - an sm_89 A/B (workstations paused);
  - an MC run with a species superset larger than the structure's (the tables cover the superset's pairs);
  - the owner's default decision.
- **`tools/bench/opt6_radial_stub.py`** (OPT6-T04's ceiling tool) asserts the text of the old folded forward.
  It stops since this change, as designed.
