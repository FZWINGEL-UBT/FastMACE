# w′ late pullback: three outside-review backward designs against the H200 default (12 Nov 2026)

Brief (conductor, 12 November 2026): screen three new w′ backward designs of the outside review against today's H200
default, `skip+hoist` (main ≥ `c2ac5f53`; H200 lab about 1,014 µs, MD step about 3.32 ms on SrTiO3 10³). Kernel:
`oeq_projected_backward_wprime`, OMAT-0 medium, interaction 2 (128x0e+128x1o ⊗ 0e+1o+2e+3o, 10 uvu paths).
Branch `fastmace/wprime-late-pullback`.

| Design | What | Result | Verdict |
|---|---|---|---|
| 1 | Four-component backward: per lane only G = Σ w ∇φ (3) and R = Σ w′ φ, geometry once per edge | CPU algebra exact; sm_90 census 0.901 of skip+hoist at 56 regs (stop rule: ≤ 0.933 at ≤ 96) | **passes the census; GPU checks below** |
| 2 | Per edge degree l spherical or direct (16 masks) | every hybrid dominated by all-direct | closed |
| 3 | Irrep-first loop nest (one harmonic block live), compile-only | 64 regs, but +31 % instructions | rejected |

## Prior art

- N1 Cartesian backward (ledger `cx-n1-cartesian-wprime`, [cartesian-backward-20261106](../cartesian-backward-20261106/README.md),
  `tools/lab_cartesian.py`): Cartesian harmonic adjoints q = M g with the radial/angular force assembled per path and
  channel, `(w/r)(∇f − l f n)` with the Euler term. H200 kernel 1.022, with its 57 µs converter 1.060.
- [wprime-h200-20261111](../wprime-h200-20261111/README.md): `census_sass.py`, `render_sources.py`, the sources.

## Design 1: four-component backward with delayed geometry differentiation

Per path (input degree a ∈ {0, 1}, edge degree l, output degree L), with q = M g (N1's conversion) and h = H_L(q; n):

| form | φ (on the unit sphere) | ∇φ | dx |
|---|---|---|---|
| a = 0 | x₀ h | x₀ ∇h | w h |
| a = 1, L = l + 1 | x·∇h | H x | w ∇h |
| a = 1, L = l − 1 | A h − B / (2L + 1), A = x·n, B = x·∇h | x h + A ∇h − H x / (2L + 1) | w (n h − ∇h / (2L + 1)) |

The last row is the reviewer's unit-sphere extension (1⊗1→0, 1⊗2→1, 1⊗3→2): it drops the 2nB term parallel to n
and N1's per-path Euler term, both of which the projector removes. Per lane the kernel sums G = Σ w ∇φ and
R = Σ (dw/dt) φ over the paths and its channels; after the channel reduction, once per edge,
dL/dv = n R sc + (G − n (n·G)) / r.

**Production form (drop-in).** `fastmace/kernels/oeq_streaming/wprime_pullback.py`, switch
`wprime_backward.LATE_PULLBACK` (off; `LATE_PULLBACK_GPUS` empty). A post-render transform like `wprime_tuning`.
It writes G/√3 into dsh's l = 1 slots and R·sc into dr. edge_sh3's backward maps an l = 1 slot gradient d to
√3 (I − n nᵀ) d / r, so the kernel keeps today's outputs (dsh [E, 16], dr [E]) and the autograd graph. This
avoids EDGE_FORCE's plumbing (rejected 11 Nov: AOT zero gradients, interaction 0 keeps the sh backward). The STAGE_DOUT
prologue stages q = M g instead of g, one thread per channel (the M are sparse: 44 FP per thread per run).
The conversion is fitted at build time in float64 against e3nn and the schedule's coefficients, divided by FOLD_ALPHA's
alpha where the table carries it.

### Step 1: CPU algebra check (`algebra_check.py` → `algebra_check.json`)

All ten paths × 128 channels, 64 edges, nonconstant cubic radial weights per (path, channel), against the direct CG
contraction with autograd in float64 (dL/dv in the sh frame, dL/dx):

| | dL/dv max rel | dL/dv median edge | dx max rel (0e / 1o) |
|---|---:|---:|---:|
| design 1, float64 | 6.6e-16 | 5.5e-16 | 4.6e-16 / 4.6e-16 |
| design 1, float32 | 1.6e-7 | 1.7e-7 | 1.3e-7 / 2.1e-7 |
| direct contraction, float32 (baseline) | 1.6e-7 | 2.4e-7 | 1.3e-7 / 2.1e-7 |

N1's conversions fit to ≤ 6.1e-16. The generated CUDA itself was also checked: `emulate_check.py` (lab source
`d1-conv.cu`) and `emulate2.py` (production transform, every mask, `emulate2.jsonl`). Both translate the prologue
and channel-loop statements to Python and run them in float64. Errors are 1.4e-9 to 1.7e-9, the rounding of the
9-digit literals of M. The unmodified kernel emulates to 2e-13. The A100 render (STAGE_EDGES) emulates to 1.6e-9
(`a100_check.json`).

### Step 2: offline census (`gen_sources.py`, `census_sass.py`, `census_prologue.py`)

Per edge and lane: the channel pass × 4 plus the rest of the edge loop (census_sass), plus the STAGE_DOUT prologue,
which census_sass leaves out. The prologue is P × runs per block / 8: P is its per-thread instruction count with
the copy loops weighted by their trip counts, and there are 1.41 receiver runs per 32-edge block for SrTiO3 10³ at
6 Å (`runs_per_block.json`, perfect lattice, 370,000 edges).

| source | sm_90 regs | sm_90 channel pass | sm_90 edge loop | + prologue | **sm_90 total** | ratio | sm_80 regs | sm_80 total | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| skip+hoist (today's H200 default) | 96 | 423 | 1,990 | 36 | **2,026** | 1.000 | 128 | 2,010 | 1.000 |
| design 1, lab form (`d1-conv`) | 56 | 373 | 1,780 | 47 | **1,827** | 0.902 | 64 | 1,802 | 0.897 |
| design 1, production (`mask-DDDD`) | 56 | 372 | 1,779 | 47 | **1,826** | 0.901 | 64 | 1,799 | 0.895 |
| N1 control on today's head/tail, in-prologue conversion (`n1-conv`) | 56 | 529 | 2,359 | 47 | 2,406 | 1.187 | 69 | 2,371 | 1.180 |
| N1 control with the plain copy (`n1-copy`; converter kernel extra) | 56 | 529 | 2,360 | 40 | 2,400 + converter | | 66 | 2,359 + converter | |

- No variant spills. Shared memory is unchanged (21,760 B on sm_90).
- FP per edge: 1,438 → 1,268. Moving the per-path force assembly out of the channel loop saves 156 instructions per
  channel pass against N1 (529 → 373), about 620 per edge. The reviewer estimated about 200.
- The conversion costs about 11 instructions per edge over the plain copy (the M are mostly diagonal).
- **Stop rule** (≥ 6.7 % below skip+hoist on sm_90, register class ≤ 96, conversion included): 0.901 at 56
  registers. **Passes.**
- The A100 plan (STAGE_EDGES, no skip/hoist; `census_a100.py`): 2,087 → 1,890 (0.906), 79 → 64 registers.

### Step 3: GPU checks

(in progress: the local 5060 Ti with the H200 plan alias for the kernel lab and the 3× float32 gate, and Alex A100
job 4988692 for the lab, gate and MD-step A/B in both orders. Results go here.)

## Design 2: per-degree hybrids (`masks.py`, `census_masks.py` → `census_masks.json`)

`wprime_pullback.apply(..., direct=...)` keeps the spherical blocks of the other degrees. Their dsh slots and the
direct G meet in the l = 1 slots before the butterfly. Mask bits l = 0..3 (D direct, s spherical); totals include
the prologue:

| mask | sm_90 regs | sm_90 total | sm_80 regs | sm_80 total |
|---|---:|---:|---:|---:|
| ssss (= skip+hoist) | 96 | 2,026 | 128 | 2,010 |
| Dsss | 96 | 1,997 | 128 | 1,976 |
| sDss | 96 | 2,010 | 128 | 1,991 |
| DDss | 96 | 1,957 | 128 | 1,939 |
| ssDs | 90 | 1,962 | 102 | 1,943 |
| DsDs | 86 | 1,910 | 96 | 1,893 |
| sDDs | 80 | 1,908 | 96 | 1,892 |
| DDDs | 72 | 1,854 | 96 | 1,836 |
| sssD | 80 | 2,044 | 96 | 2,035 |
| DssD | 80 | 2,001 | 96 | 1,976 |
| sDsD | 72 | 1,998 | 80 | 1,983 |
| DDsD | 64 | 1,945 | 79 | 1,930 |
| ssDD | 64 | 1,938 | 80 | 1,916 |
| DsDD | 67 | 1,888 | 75 | 1,865 |
| sDDD | 56 | 1,886 | 71 | 1,863 |
| **DDDD** | **56** | **1,826** | **64** | **1,799** |

All-direct has the fewest instructions and the fewest registers on both architectures, so every hybrid is
dominated and design 2 closes. The mixed prologue (some paths converted, some copied) costs the same as the full
conversion. All 16 masks emulate correctly (`emulate2.jsonl`).

## Design 3: irrep-first loop nest, compile-only (`design3.py`, `census_d3.py` → `census_d3.json`)

Today's spherical kernel with the channel loop split by edge degree. Each group has its own accumulators and dx
atomics (hoisted pointer), and its 2l + 1 slots are reduced and stored right after its loop. Counted the same way
for both (every FFMA loop × 4 + the rest of the edge loop):

| | sm_90 regs | sm_90 per edge | sm_80 regs | sm_80 per edge |
|---|---:|---:|---:|---:|
| skip+hoist | 96 | 1,972 | 128 | 1,976 |
| irrep-first | 64 | 2,576 (+30.6 %) | 72 | 2,596 (+31.4 %) |

Channel loops 60 + 136 + 198 + 150 = 544 per pass against 423: x reloads per group, 16 dx atomics per pass instead
of 4, the per-group loop overhead, and 16 five-step slot reductions instead of the butterfly. It reaches ≤ 80
registers, but the increase is not modest. The screening model t ≈ (I_new/I_old) × (η_old/η_new) gives at least
1.306 × 0.77 = 1.006 even at η_new = 1. **Rejected.**

## Files

- `common.py` (schedule, paths, conversions), `run_wsl.sh` (CPU, mace_cueq_oeq).
- Step 1: `algebra_check.py/.json`; `emulate_check.py/.json` (lab source), `emulate2.py` + `emulate2.jsonl`
  (production masks), `a100_check.py/.json`.
- Step 2: `gen_sources.py` → `sources/{skip+hoist,d1-conv,d1-copy,n1-conv,n1-copy}.cu`; `census_sm90.json`,
  `census_sm80.json` (census_sass), `census_prologue.py` → `census_prologue_sm{90,80}.json`; `runs_per_block.py/.json`;
  `census_a100.py/.json` (`sources/a100-*.cu`).
- Design 2: `masks.py` → `sources/mask-*.cu` (only `mask-DDDD.cu` committed; the script regenerates the rest),
  `census_masks.py/.json`.
- Design 3: `design3.py` → `sources/d3-irrepfirst.cu`, `census_d3.py/.json`.
- `sass/`: `skip+hoist.sm90.sass`, `mask-DDDD.sm90.sass` (the others are regenerated by the census scripts).
- `jobs/alex_lp.sh`: the Alex A100 job.
