# HYBRID and GEOM on the H200 (11 November 2026)

Two switches that were default on the A100 but never measured on the H200: `tf32x3.HYBRID` (CUTLASS 3×TF32 entries
with TF32 big·big plus one FP16 cross mma; sm_90 cubin) and `spline_forward.GEOM` (the SPLINE_W forward's shared CG
geometry sums). One Modal container, one pack (`tools/modal/h200_hybrid_geom.sh`), approved beyond the month's
Modal cap by the owner.

**Verdict: both adopted on the H200** (decision 66: rounding switches, faster in both build orders on both
structures, gate passes). `HYBRID_GPUS` and `GEOM_GPUS` now include `NVIDIA H200`.

## Run

- Modal H200 (driver 610.57.04, CUDA 13.3), profile `sixth`, app ap-BP79lByjlWkTznN2zAndhK, source `daf1ab5b`
  (main 4d444f9b plus the pack script).
- 810 s in the container. **Cost: $1.19 estimated, $1.24 metered** (ledger row, limit raised to $34 with the
  owner override; the worst case was $33.38).
- Records in `run/`: `run.log`, the A/B JSON and logs, the gate logs (per-structure summaries) and metadata,
  `modal.json`.
- CPU check before the launch: `pack_check.py` → `pack_check.json` (arguments, constants, GEOM's sm_90 compile, the
  HYBRID sm_90 cubin's entries, the slim tarball's contents).

## MD-step A/B

OMAT-0 medium, float32, a = the H200's default plan, b = one switch on. 1 fs, skin 0.5, warm-up 200, 6 blocks of
256 steps per side; one `tools/ab_rewrites.py --pack` process per build order (ab: abab, ba: abba).

| b/a | SrTiO3 10×10×10 ab | ba | NbCl5 3×3×3 ab | ba |
| --- | --- | --- | --- | --- |
| HYBRID | 0.9909 (3.3078 → 3.2777 ms) | 0.9940 (3.3107 → 3.2907 ms) | 0.9892 (1.0550 → 1.0436 ms) | 0.9885 (1.0574 → 1.0452 ms) |
| GEOM | 0.9857 (3.3265 → 3.2790 ms) | 0.9890 (3.3208 → 3.2844 ms) | 0.9891 (1.0584 → 1.0469 ms) | 0.9860 (1.0573 → 1.0425 ms) |

Both are faster in every order and structure. The two were measured separately against the defaults, not together;
they act in different kernels (CUTLASS GEMMs, the interaction 2 forward).

## Force/stress gate

`tools/force_stress_gate.py`, ratio to the ASE float32 error, limit 3× (energy: 10 meV floor). Every quantity passes.
The default plan's own gate was left out to fit the credit; decision 66 needs only the switch's gate.

| side | SrTiO3 3×3×3 F / S | NaCl F / S | NbCl5 F / S | energy ratios (floor 10 meV) |
| --- | --- | --- | --- | --- |
| HYBRID | 0.92 / 0.41 | 0.76 / 0.94 | 0.95 / 1.30 | 6.1 / 0.48 / 2.3, all under the floor |
| GEOM | 0.83 / 0.42 | 0.85 / 0.83 | 0.99 / 1.04 | 5.6 / 0.39 / 2.2, all under the floor |

## GEOM decision note

The compact-angular-products idea was to get a census only if GEOM were slower on the H200. GEOM is faster there
(0.986–0.989), so the idea is **not eligible** and is closed in the ledger. For the record, GEOM's sm_90 compile of
the H200's forward form (`pack_check.json`): 80 registers, 5 spill ops, edge loop 348 → 317 instructions; the spill
does not cost it the A/B.

## Launch notes

- Run the Modal client from a folder holding `tools/modal/` (extract it from the slim tar next to `src.tar`).
- `MODAL_IMAGE_BUILDER_VERSION=2025.06` for a workspace on image builder 2026.10, which rejects
  `python_version="3.10"` (the image rebuilds once).
- Past the month's cap the guard needs `--limit-usd` above the worst case together with `--owner-override`.
