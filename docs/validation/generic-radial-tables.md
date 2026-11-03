# Generic radial tables: MACE-OMAT-0 medium

`generic_radial_tables` extends MH-1's certified radial tables (`radial_tables`, decision 61) to the generic MACE
family: MACE-MP-0, MACE-OMAT-0, and the density blocks. A profile of a compiled tiled call on 975,560 SrTiO3 atoms
put the radial MLP's products and SiLU kernels at about 5.6 s of 19-20 s, about 29 %
(`results/tiled-omat-20261102/radial_profile.py`).

**Setup**
- Ada: NVIDIA RTX 4000 Ada (20 GB) on murphy and leto, the `mace_cueq_oeq` NAS environment.
- 5060 Ti: the local PC's RTX 5060 Ti (16 GB) in WSL2, `~/environments/mace_cueq_oeq`, through a local GPU
  reservation.
- Model: `omat0-medium` (`mace-omat-0-medium.model`); hybrid backend, float32 forces.
- FastMACE at 91d69c4e (workstations) and 5f073d22 (local). Those commits differ only in result scripts.
- Scripts and runs: `results/generic-radial-20261102/`.

## 1. What the rewrite does

- **One table per interaction, and per species pair** when the radial embedding is species-dependent. OMAT-0's
  `AgnesiTransform` takes the sender's and receiver's species, so its tables are per pair (9 for SrTiO3, 4 for NaCl).
  Models without a distance transform get one table per interaction. No table is keyed by a shape or a structure.
- **What a table holds:** the radial chain from the edge length to `[h c, c, density]`, where:
  - `h` is the radial MLP's last hidden layer;
  - `c` is the cutoff (ones where the embedding applies the cutoff itself, as OMAT-0's does);
  - `density` is the density blocks' edge density (zeros for residual blocks).
- **Where it plugs in:** the interaction's `[h c, c] @ [W, 0]` stays a GEMM on the table's output, which
  `generic_fused_radial_backward` already folds. So the rewrite requires that rewrite.
- **Fit and certificate:** MH-1's machinery (`fastmace/rewrites/radial`), unchanged:
  - piecewise Chebyshev fits against the float64 chain, with the derivative;
  - the same certificate, which must be at most 1;
  - the same table cache and the same Triton `radial_table` kernel.
  - One change in `fit`: a column that is identically zero or constant (the density of a residual block, the cutoff
    column where it is all ones) has a zero error floor. Its error is now taken as zero instead of the NaN that
    0/0 gave. This changes the cache keys of MH-1's tables (the module's source hash), not their values.
- **Record:** `generic_radial_tables`, on the `radial_tables` config field.
  - Class: certified.
  - Default: `auto`, on only on the GPUs of decision 66 (section 6).
  - Refused for float64 plans and for `reproducible=True`.
  - Requires `generic_fused_radial_backward`.
  - Workload constraint: a fixed species superset.
- **Self-check:** the certificate (worst at most 1), then the tables against the **float64** chain on the CPU, at the
  float32 threshold (1e-4 relative to the largest value), for `w = [h c, c] @ [W, 0]`, the density and `dr`.
  - The first version compared against the float32 network and failed on OMAT-0 (2.0e-4 > 1e-4).
  - The probe (`probe.py`, `runs/murphy-grt-probe`) showed why. Near OMAT-0's cutoff (5-6 Å) the float32
    **network** is 5.5e-5 of the largest `w` away from float64, while the table is 3.6e-8 away. The failure was the
    network's own rounding, not the table's.

  | r band (Å) | Float32 network vs float64 | Table vs float64 |
  | --- | ---: | ---: |
  | 0.3-1.0 | 2.7e-6 | 3.9e-7 |
  | 1.0-5.0 | 1.0e-7 to 2.6e-7 | 1.2e-8 to 6.1e-8 |
  | 5.0-6.0 (interaction 0) | 5.5e-5 | 3.6e-8 |
  | 5.0-6.0 (interaction 1) | 3.0e-6 | 1.4e-8 |

- **Memory (rule 12):** `memory_terms.generic_radial_tables` takes off what the network's backward keeps per edge
  (every hidden layer's input and its activation's input). The table's output is what the folded path keeps anyway,
  so nothing is added per edge.

## 2. Force and stress gate (murphy, `gate.py`, `runs/murphy-grt-gate2`)

Rattled (0.05 Å) and strained (1 %) supercells. Each structure is built with the tables twice: once fitting them
into a fresh cache, then from that cache.

| Structure | Atoms | Largest force | Tables vs network: force | Tables vs float64: force | Network vs float64: force | Energy (meV/atom), all pairs | Stress, all pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SrTiO3 6×6×6 | 1,080 | 4.30 | 3.5e-5 | 6.4e-5 | 5.6e-5 | ≤ 4.4e-4 | ≤ 2.3e-7 |
| NaCl 4×4×4 | 512 | 0.56 | 4.0e-6 | 3.9e-6 | 4.8e-6 | ≤ 1.4e-4 | ≤ 2e-8 |

- **The tables are as close to float64 as the float32 network is.**
- **Tables built:**
  - SrTiO3: 162 and 175 pieces over 9 pairs, certificates 0.31 and 0.34.
  - NaCl: 84 and 49 pieces over 4 pairs, certificates 0.30 and 0.30.
- **Cached tables give the same forces** as freshly fitted ones (2.6e-6 and 1.5e-7 eV/Å; the builds compile anew).

## 3. Setup time

| Structure | First build, fitting the tables | Build from the cache | Fit's share |
| --- | ---: | ---: | ---: |
| SrTiO3 (9 pairs) | 46.3 s | 10.7 s | about 35 s |
| NaCl (4 pairs) | 52.7 s | 19.6 s | about 33 s |

The fit runs once per model and species set. The tables are cached under `FASTMACE_CACHE`.

## 4. Speed

**Whole system:** `tools/ab_rewrites.py`, the MD step of the default NVT plan, side b `radial_tables=on`, in one
process. Both build orders (`ab` with pattern `abab`, `ba` with `abba`), 12 blocks of 300 steps.

| GPU | Structure | Atoms | Untabled step | Tabled step | b/a (ab) | b/a (ba) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX 4000 Ada (murphy) | NaCl 10×10×10 | 8,000 | 53.96 ms | 46.62 ms | 0.864 | 0.862 |
| RTX 4000 Ada (murphy) | SrTiO3 11×11×11 | 6,655 | 79.20 ms | 66.94 ms | 0.845 | 0.844 |
| RTX 5060 Ti (local) | NaCl 10×10×10 | 8,000 | 49.51 ms | 42.03 ms | 0.849 | 0.848 |
| RTX 5060 Ti (local) | SrTiO3 11×11×11 | 6,655 | 72.57 ms | 59.61 ms | 0.821 | 0.821 |

Every A/B's gate passed (`valid`).

**Tiled:** `results/tiled-speed-20261101/ab_arms.py`, compiled tiles with the budget, without and with the tables,
in one process. Both arm orders, 4 alternating blocks of 1 call each.

| GPU | Atoms | Untabled call | Tabled call | Tabled/untabled (untabled first) | (tabled first) | Force diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 4000 Ada (murphy) | 975,560 (58×58×58) | 20.21 s | 17.40 s | 0.861 | 0.862 | 7.0e-5 |
| RTX 5060 Ti (local) | 320,000 (40×40×40) | 7.62 s | 6.72-6.96 s | 0.912 | 0.882 | 7.0e-5 |

- **The tables save 14-18 % of an MD step on both GPUs**, and 12-14 % of a tiled call.
- **The tiled calls' energies agree within 2.9e-4 meV/atom**, and their stresses within 2.1e-6.
- **Both arms chose the same receiver cap** (16,384).
- **The local tiled A/B is noisier in the tabled arm.** Between the two orders, the untabled median moved 0.1 %
  (7.63 and 7.62 s) and the tabled one 3.5 % (6.96 and 6.72 s).

## 5. NVE drift (leto, `tools/bench/nve_drift.py`)

NaCl 3×3×3 (216 atoms), 300 K, 10 ps at 1 fs, default plan; runs `runs/leto-grt-nve-NaCl-tables-{off,on}`.

| Tables | Drift (meV/atom/ps) | RMS about the line (meV/atom) | Range (meV/atom) |
| --- | ---: | ---: | ---: |
| Off | 9.2e-7 | 3.07e-4 | 6.5e-3 |
| On | -4.2e-6 | 3.08e-4 | 6.6e-3 |

Both drifts are negligible, and the fluctuations are the same.

## 6. Status

- **Default on the measured GPUs (owner, 2 November 2026: "Yes make it default"; decision 66).** For the generic
  family, `radial_tables="auto"` is on only on the "NVIDIA GeForce RTX 5060 Ti", the
  "NVIDIA RTX 4000 Ada Generation" and, since 3 November 2026 (owner, after Benchmark_loup, `loup-a100.md`), the
  "NVIDIA A100 80GB PCIe" and "NVIDIA A100-SXM4-80GB" (`planner.fixed.DECISION_66_GENERIC_RADIAL_TABLES_GPUS`, exact
  device names), and off on every other GPU. `radial_tables="on"` turns them on anywhere.
- Float32 plans only. In float64 plans "auto" keeps the radial chain and an explicit "on" is refused; the same for
  `reproducible=True`.
- The table cache under `FASTMACE_CACHE` covers the setup cost after the first fit (section 3).
- Tests: `tests/cpu/test_generic_radial_tables.py`, and the capability rows in `tests/cpu/test_capabilities.py`.
