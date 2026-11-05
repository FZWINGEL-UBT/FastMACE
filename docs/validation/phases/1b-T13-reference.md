# 1b-T13: reference records and corrected reference calculators

Plan task 1b-T13 (SPEC §12.1, §8.4), 19 October 2026. Code: `fastmace/reference/records.py`,
`fastmace/reference/calculators.py`; FastMACE commit `ae96ba5`.

## What the record states

One `ReferenceRecord` per (MACE version, model family); today only MACE 0.3.16 is qualified.
Its id is `fastmace.reference/<version>/<family>/<digest>`, a SHA-256 over every field; it goes into
`ForceFieldIdentity.reference_record_id` (§10.6) and provenance (§13).

- **Stock:** MACE's `MACECalculator`, cuEquivariance, float64 reference and float32 yardstick. As
  1a-T19's references: MACE's own cuEq conversion with the convolution unfused, and the one-graph
  strain (MACE PR 1498).
- **Upstream fix:** the PolarMACE stress, MACE issue 1642 (fixed upstream in PR 1718). Every
  PolarMACE record for 0.3.16 carries it, and a polar model is refused with a record that lacks it.
  Validated by finite differences on NaCl 3×3×3 only. On NbCl5 3×3×3 the corrected stress still
  differs by 4.4e-4 eV/Å³ (1a-T19); that is open for phase 3b.
- **Optional physics:** PR 1762 (non-stock label); D3 is declared and refused until phase 3b.
- **Integrator correction:** full-cell BAOAB drag, with 1b-T09's `QUALIFICATION`.
- **Numerical treatment:** the float32 yardstick's energy is reduced into float64 with the
  checkpoint's E0 table; its forces and stress are MACE's float32 values, bit for bit. For
  PolarMACE the yardstick runs the corrected long-range path, so the electrostatic energy is summed
  per k in float64 (MACE's own code would give only the float32 total, upcast).

The calculators apply the corrections in the safe order and check before every evaluation that
each one is the active code path (`ReferenceOrderError` otherwise). This addresses the 1a-T19
finding: a forward stored on the model before the polar fix was installed made the references run
MACE 0.3.16's uncorrected stress, and nothing noticed.

## CPU (tiny models)

`tests/cpu/test_reference_policy.py`, 27 tests. They pass in `mace_cueq_latest` with CUDA hidden.
In `fastmace_cpu`, which has no `graph_longrange`, the polar tests skip. Results:

- energy, forces and stress equal stock MACE within 1e-12 for the mh1, residual, density, off and
  omol families;
- PolarMACE: energy within 1e-12 relative, forces within 2.9e-13 eV/Å (another k-summation order);
  the stress agrees with finite differences of the reference's own energy, and MACE 0.3.16's own
  stress does not;
- the yardstick's forces and stress equal stock float32 bit for bit;
- the yardstick's energy error against float64 is below half the stock float32 error.

## GPU: POLAR-1-S against 1a-T19's references

| Item | Value |
| --- | --- |
| Machine, GPU | murphy, RTX 4000 Ada 20 GB, driver 595.58.03 (idle check: no compute process) |
| Environment | NAS `mace_cueq_latest`: torch 2.14.1+cu130, cuequivariance 0.12.0, mace-torch 0.3.16, graph-longrange 0.4.0 |
| Backend, precision | cuEq (MACE conversion, conv fusion off), float64, one-graph strain on |
| Model | MACE-POLAR-1-S (SHA-256 from the 1a-T19 run's metadata) |
| Commits | FastMACE `ae96ba5`; baseline: 1a-T19's references (leto, research engine export 1a-T18, FastMACE `e05dd55`) |
| Structures | NaCl 3×3×3 (216 atoms), 3 hydrostatic and 3 full-cell configurations of 1a-T19 |
| Record | `fastmace.reference/0.3.16/polar/8e216bde0c3515e9` |
| Energy, worst relative difference | 0 (bitwise equal) |
| Forces, worst \|Δ\| | 5.1e-15 eV/Å |
| Stress, worst \|Δ\| | 1.3e-16 eV/Å³ |
| Finite-difference self-test (hydrostatic-0) | 6.93e-9 eV/Å³ (tolerance 1e-6; 1a-T19: 6.93e-9) |

The check passes the 1e-12 limit. Both runs used a fresh reference process. JSON:
`results/1b-T13-murphy-20261019/`.
