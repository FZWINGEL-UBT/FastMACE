
### rtx4000ada


#### SrTiO3 10x10x10, 5,000 atoms

| setup | boundary | N | trials | median | min–max | force RMSE (meV/Å) | energy err (meV/atom) | T start→end (K) | nbr builds | peak GPU mem (MiB) |
|---|---|---:|---:|---:|---|---:|---:|---|---:|---|
| FastMACE (ASE, compiled, skin 0.5, fp32 energy) | ASE call (us/atom) | 5000 | 3 | 3.91 | 3.91–3.91 | 0.0055 | 0.0008 | static | n/r | 1961 (torch peak) |
| Symmetrix-XL (ASE) | ASE call (us/atom) | 5000 | 3 | 8.03 | 8.00–8.04 | 0.0167 | 0.0180 | static | n/r | 906 (nvidia-smi after) |
| Symmetrix-XL (LAMMPS, static) | LAMMPS loop (us/atom/step) | 5000 | 3 | 8.10 | 8.10–8.16 | 0.0167 | 0.0180 | static | 0 | n/r |
| MACE + cuEquivariance (ASE; also the reference code) | ASE call (us/atom) | 5000 | 3 | 51.88 | 51.80–51.92 | reference | reference | static | n/r | 6324 (torch peak) |
| Symmetrix-XL (LAMMPS, NVE MD) | MD step (us/atom/step) | 5000 | 3 | 8.15 | 8.06–8.18 | 0.0177 | 0.0243 | 3.0→7.9 | 0 | n/r |
| FastMACE own MD engine (NVE) | MD step (us/atom/step) | 5000 | 3 | 3.19 | 3.18–3.20 | 0.0056 | 0.0210 | n/r | n/r | 1961 (torch peak) |
| Symmetrix-XL (LAMMPS, Langevin NVT 300 K) | MD step (us/atom/step) | 5000 | 3 | 8.26 | 8.23–8.28 | 0.0174 | 0.0280 | 8.3→25.4 | 0 | n/r |
| FastMACE own MD engine (NVT Langevin 300 K) | MD step (us/atom/step) | 5000 | 3 | 3.10 | 3.10–3.11 | 0.0056 | 0.0148 | n/r | n/r | 1961 (torch peak) |

#### Cu(111)-H2O, OMAT-0, 1x1 (272 atoms)

| setup | boundary | N | trials | median | min–max | force RMSE (meV/Å) | energy err (meV/atom) | T start→end (K) | nbr builds | peak GPU mem (MiB) |
|---|---|---:|---:|---:|---|---:|---:|---|---:|---|
| FastMACE (ASE, compiled, skin 0.5, fp32 energy) | ASE call (us/atom) | 272 | 3 | 21.57 | 20.54–21.57 | 0.0055 | 0.0004 | static | n/r | 2091 (torch peak) |
| Symmetrix-XL (ASE) | ASE call (us/atom) | 272 | 3 | 12.48 | 12.48–12.51 | 0.0609 | 0.0008 | static | n/r | 526 (nvidia-smi after) |
| Symmetrix-XL (LAMMPS, static) | LAMMPS loop (us/atom/step) | 272 | 3 | 11.68 | 11.62–11.68 | 0.0610 | 0.0008 | static | 0 | n/r |
| MACE + cuEquivariance (ASE; also the reference code) | ASE call (us/atom) | 272 | 3 | 119.48 | 110.79–120.93 | reference | reference | static | n/r | 415 (torch peak) |
| Symmetrix-XL (LAMMPS, NVE MD) | MD step (us/atom/step) | – | 0 of 3 | (11.92) rejected | 11.92–11.95 | – | – | 45.6→113.4 | 1 | n/r |
| FastMACE own MD engine (NVE) | MD step (us/atom/step) | 272 | 3 | 6.04 | 5.74–6.05 | 0.0052 | 0.0003 | n/r | n/r | 2091 (torch peak) |
| Symmetrix-XL (LAMMPS, Langevin NVT 300 K) | MD step (us/atom/step) | – | 0 of 3 | (11.89) rejected | 11.89–11.90 | – | – | 47.9→121.8 | 1 | n/r |
| FastMACE own MD engine (NVT Langevin 300 K) | MD step (us/atom/step) | 272 | 3 | 5.77 | 5.77–5.78 | 0.0051 | 0.0003 | n/r | n/r | 2091 (torch peak) |

#### Cu(111)-H2O, OMAT-0, 2x2 (1088 atoms)

| setup | boundary | N | trials | median | min–max | force RMSE (meV/Å) | energy err (meV/atom) | T start→end (K) | nbr builds | peak GPU mem (MiB) |
|---|---|---:|---:|---:|---|---:|---:|---|---:|---|
| FastMACE (ASE, compiled, skin 0.5, fp32 energy) | ASE call (us/atom) | 1088 | 3 | 6.49 | 6.45–6.51 | 0.0056 | 0.0072 | static | n/r | 2092 (torch peak) |
| Symmetrix-XL (ASE) | ASE call (us/atom) | 1088 | 3 | 9.09 | 9.09–9.11 | 0.0619 | 0.0001 | static | n/r | 602 (nvidia-smi after) |
| Symmetrix-XL (LAMMPS, static) | LAMMPS loop (us/atom/step) | 1088 | 3 | 8.69 | 8.66–8.81 | 0.0620 | 0.0001 | static | 0 | n/r |
| MACE + cuEquivariance (ASE; also the reference code) | ASE call (us/atom) | 1088 | 3 | 56.43 | 56.39–56.67 | reference | reference | static | n/r | 1484 (torch peak) |
| Symmetrix-XL (LAMMPS, NVE MD) | MD step (us/atom/step) | – | 0 of 3 | (9.06) rejected | 8.82–9.11 | – | – | 45.5→113.1 | 1 | n/r |
| FastMACE own MD engine (NVE) | MD step (us/atom/step) | 1088 | 3 | 3.71 | 3.70–3.72 | 0.0058 | 0.0028 | n/r | n/r | 2092 (torch peak) |
| Symmetrix-XL (LAMMPS, Langevin NVT 300 K) | MD step (us/atom/step) | – | 0 of 3 | (8.92) rejected | 8.81–8.94 | – | – | 47.6→120.5 | 1 | n/r |
| FastMACE own MD engine (NVT Langevin 300 K) | MD step (us/atom/step) | 1088 | 3 | 3.63 | 3.63–3.64 | 0.0055 | 0.0016 | n/r | n/r | 2092 (torch peak) |

#### Cu(111)-H2O, OMAT-0, 3x3 (2448 atoms)

| setup | boundary | N | trials | median | min–max | force RMSE (meV/Å) | energy err (meV/atom) | T start→end (K) | nbr builds | peak GPU mem (MiB) |
|---|---|---:|---:|---:|---|---:|---:|---|---:|---|
| FastMACE (ASE, compiled, skin 0.5, fp32 energy) | ASE call (us/atom) | 2448 | 3 | 4.55 | 4.44–4.56 | 0.0055 | 0.0016 | static | n/r | 2093 (torch peak) |
| Symmetrix-XL (ASE) | ASE call (us/atom) | 2448 | 3 | 8.52 | 8.52–8.55 | 0.0625 | 0.0045 | static | n/r | 708 (nvidia-smi after) |
| Symmetrix-XL (LAMMPS, static) | LAMMPS loop (us/atom/step) | 2448 | 3 | 8.55 | 8.53–8.56 | 0.0625 | 0.0045 | static | 0 | n/r |
| MACE + cuEquivariance (ASE; also the reference code) | ASE call (us/atom) | 2448 | 3 | 53.72 | 53.63–54.27 | reference | reference | static | n/r | 3270 (torch peak) |
| Symmetrix-XL (LAMMPS, NVE MD) | MD step (us/atom/step) | – | 0 of 3 | (8.65) rejected | 8.62–8.66 | – | – | 45.5→113.0 | 1 | n/r |
| FastMACE own MD engine (NVE) | MD step (us/atom/step) | 2448 | 3 | 3.56 | 3.55–3.58 | 0.0061 | 0.0019 | n/r | n/r | 2093 (torch peak) |
| Symmetrix-XL (LAMMPS, Langevin NVT 300 K) | MD step (us/atom/step) | – | 0 of 3 | (8.62) rejected | 8.62–8.70 | – | – | 47.7→120.6 | 1 | n/r |
| FastMACE own MD engine (NVT Langevin 300 K) | MD step (us/atom/step) | 2448 | 3 | 3.50 | 3.50–3.51 | 0.0055 | 0.0041 | n/r | n/r | 2093 (torch peak) |

#### Cu(111)-H2O, OMAT-0, 4x4 (4352 atoms)

| setup | boundary | N | trials | median | min–max | force RMSE (meV/Å) | energy err (meV/atom) | T start→end (K) | nbr builds | peak GPU mem (MiB) |
|---|---|---:|---:|---:|---|---:|---:|---|---:|---|
| FastMACE (ASE, compiled, skin 0.5, fp32 energy) | ASE call (us/atom) | 4352 | 3 | 4.36 | 4.35–4.37 | 0.0055 | 0.0009 | static | n/r | 2095 (torch peak) |
| Symmetrix-XL (ASE) | ASE call (us/atom) | 4352 | 3 | 8.78 | 8.78–8.80 | 0.0638 | 0.0143 | static | n/r | 864 (nvidia-smi after) |
| Symmetrix-XL (LAMMPS, static) | LAMMPS loop (us/atom/step) | 4352 | 3 | 8.95 | 8.92–8.96 | 0.0638 | 0.0143 | static | 0 | n/r |
| MACE + cuEquivariance (ASE; also the reference code) | ASE call (us/atom) | 4352 | 3 | 55.36 | 55.26–55.63 | reference | reference | static | n/r | 5773 (torch peak) |
| Symmetrix-XL (LAMMPS, NVE MD) | MD step (us/atom/step) | – | 0 of 3 | (8.95) rejected | 8.94–9.02 | – | – | 45.5→113.0 | 1 | n/r |
| FastMACE own MD engine (NVE) | MD step (us/atom/step) | 4352 | 3 | 3.70 | 3.68–3.70 | 0.0055 | 0.0066 | n/r | n/r | 2095 (torch peak) |
| Symmetrix-XL (LAMMPS, Langevin NVT 300 K) | MD step (us/atom/step) | – | 0 of 3 | (9.01) rejected | 8.92–9.01 | – | – | 47.6→121.2 | 1 | n/r |
| FastMACE own MD engine (NVT Langevin 300 K) | MD step (us/atom/step) | 4352 | 3 | 3.57 | 3.57–3.58 | 0.0056 | 0.0065 | n/r | n/r | 2095 (torch peak) |

#### Cu(111)-H2O, OMAT-0, 4x5 (5440 atoms)

| setup | boundary | N | trials | median | min–max | force RMSE (meV/Å) | energy err (meV/atom) | T start→end (K) | nbr builds | peak GPU mem (MiB) |
|---|---|---:|---:|---:|---|---:|---:|---|---:|---|
| FastMACE (ASE, compiled, skin 0.5, fp32 energy) | ASE call (us/atom) | 5440 | 3 | 4.43 | 4.43–4.43 | 0.0055 | 0.0036 | static | n/r | 2096 (torch peak) |
| Symmetrix-XL (ASE) | ASE call (us/atom) | 5440 | 3 | 8.92 | 8.88–8.92 | 0.0644 | 0.0200 | static | n/r | 954 (nvidia-smi after) |
| Symmetrix-XL (LAMMPS, static) | LAMMPS loop (us/atom/step) | 5440 | 3 | 8.92 | 8.91–8.94 | 0.0644 | 0.0200 | static | 0 | n/r |
| MACE + cuEquivariance (ASE; also the reference code) | ASE call (us/atom) | 5440 | 3 | 55.10 | 54.92–55.14 | reference | reference | static | n/r | 7194 (torch peak) |
| Symmetrix-XL (LAMMPS, NVE MD) | MD step (us/atom/step) | – | 0 of 3 | (8.94) rejected | 8.92–8.96 | – | – | 45.5→113.0 | 1 | n/r |
| FastMACE own MD engine (NVE) | MD step (us/atom/step) | 5440 | 3 | 3.74 | 3.72–3.77 | 0.0059 | 0.0117 | n/r | n/r | 2096 (torch peak) |
| Symmetrix-XL (LAMMPS, Langevin NVT 300 K) | MD step (us/atom/step) | – | 0 of 3 | (8.97) rejected | 8.95–8.99 | – | – | 47.6→121.2 | 1 | n/r |
| FastMACE own MD engine (NVT Langevin 300 K) | MD step (us/atom/step) | 5440 | 3 | 3.62 | 3.61–3.64 | 0.0057 | 0.0218 | n/r | n/r | 2096 (torch peak) |

#### Cu(111)-H2O, OMAT-0, 6x6 (9792 atoms)

| setup | boundary | N | trials | median | min–max | force RMSE (meV/Å) | energy err (meV/atom) | T start→end (K) | nbr builds | peak GPU mem (MiB) |
|---|---|---:|---:|---:|---|---:|---:|---|---:|---|
| FastMACE (ASE, compiled, skin 0.5, fp32 energy) | ASE call (us/atom) | 9792 | 3 | 4.55 | 4.54–4.56 | 0.0055 | 0.0004 | static | n/r | 2099 (torch peak) |
| Symmetrix-XL (ASE) | ASE call (us/atom) | 9792 | 3 | 8.61 | 8.60–8.63 | 0.0658 | 0.0376 | static | n/r | 1254 (nvidia-smi after) |
| Symmetrix-XL (LAMMPS, static) | LAMMPS loop (us/atom/step) | 9792 | 3 | 8.58 | 8.55–8.60 | 0.0658 | 0.0376 | static | 0 | n/r |
| MACE + cuEquivariance (ASE; also the reference code) | ASE call (us/atom) | 9792 | 3 | 64.32 | 63.50–64.92 | reference | reference | static | n/r | 12906 (torch peak) |
| Symmetrix-XL (LAMMPS, NVE MD) | MD step (us/atom/step) | – | 0 of 3 | (8.70) rejected | 8.69–8.72 | – | – | 45.5→113.0 | 1 | n/r |
| FastMACE own MD engine (NVE) | MD step (us/atom/step) | 9792 | 3 | 3.78 | 3.78–3.79 | 0.0057 | 0.0226 | n/r | n/r | 2099 (torch peak) |
| Symmetrix-XL (LAMMPS, Langevin NVT 300 K) | MD step (us/atom/step) | – | 0 of 3 | (8.71) rejected | 8.70–8.72 | – | – | 47.6→121.2 | 1 | n/r |
| FastMACE own MD engine (NVT Langevin 300 K) | MD step (us/atom/step) | 9792 | 3 | 3.63 | 3.63–3.63 | 0.0059 | 0.0126 | n/r | n/r | 2099 (torch peak) |

#### Cu(111)-H2O, OMAT-0, 8x8 (17408 atoms)

| setup | boundary | N | trials | median | min–max | force RMSE (meV/Å) | energy err (meV/atom) | T start→end (K) | nbr builds | peak GPU mem (MiB) |
|---|---|---:|---:|---:|---|---:|---:|---|---:|---|
| FastMACE (ASE, compiled, skin 0.5, fp32 energy) | ASE call (us/atom) | 17408 | 3 | 4.30 | 4.29–4.33 | – | – | static | n/r | 2105 (torch peak) |
| Symmetrix-XL (ASE) | ASE call (us/atom) | 17408 | 3 | 8.89 | 8.87–8.90 | – | – | static | n/r | 1706 (nvidia-smi after) |
| Symmetrix-XL (LAMMPS, static) | LAMMPS loop (us/atom/step) | 17408 | 3 | 8.75 | 8.72–8.77 | – | – | static | 0 | n/r |
| MACE + cuEquivariance (ASE; also the reference code) | ASE call (us/atom) | – | 0 | – | ––– | reference | reference | static | n/r | n/r |
| Symmetrix-XL (LAMMPS, NVE MD) | MD step (us/atom/step) | – | 0 of 3 | (8.89) rejected | 8.88–8.90 | – | – | 45.5→113.0 | 1 | n/r |
| FastMACE own MD engine (NVE) | MD step (us/atom/step) | 17408 | 3 | 3.89 | 3.88–3.89 | – | – | n/r | n/r | 2105 (torch peak) |
| Symmetrix-XL (LAMMPS, Langevin NVT 300 K) | MD step (us/atom/step) | – | 0 of 3 | (8.90) rejected | 8.89–8.92 | – | – | 47.7→121.3 | 1 | n/r |
| FastMACE own MD engine (NVT Langevin 300 K) | MD step (us/atom/step) | 17408 | 3 | 3.68 | 3.68–3.68 | – | – | n/r | n/r | 2105 (torch peak) |

Failures / skips:

- MACE + cuEquivariance ASE: trial 1 exit 1 (torch.OutOfMemoryError: CUDA out of memory)
- MACE + cuEquivariance ASE: trial 2 exit 1 (torch.OutOfMemoryError: CUDA out of memory)
- MACE + cuEquivariance ASE: trial 3 exit 1 (torch.OutOfMemoryError: CUDA out of memory)

#### Cu(111)-H2O, OMAT-0, 12x12 (39168 atoms)

| setup | boundary | N | trials | median | min–max | force RMSE (meV/Å) | energy err (meV/atom) | T start→end (K) | nbr builds | peak GPU mem (MiB) |
|---|---|---:|---:|---:|---|---:|---:|---|---:|---|
| FastMACE (ASE, compiled, skin 0.5, fp32 energy) | ASE call (us/atom) | 39168 | 3 | 4.36 | 4.36–4.37 | – | – | static | n/r | 2620 (torch peak) |
| Symmetrix-XL (ASE) | ASE call (us/atom) | 39168 | 3 | 9.04 | 9.03–9.06 | – | – | static | n/r | 2962 (nvidia-smi after) |
| Symmetrix-XL (LAMMPS, static) | LAMMPS loop (us/atom/step) | 39168 | 3 | 8.89 | 8.86–8.91 | – | – | static | 0 | n/r |
| MACE + cuEquivariance (ASE; also the reference code) | ASE call (us/atom) | – | 0 | – | ––– | reference | reference | static | n/r | n/r |
| Symmetrix-XL (LAMMPS, NVE MD) | MD step (us/atom/step) | – | 0 of 3 | (8.96) rejected | 8.93–8.98 | – | – | 45.5→113.0 | 1 | n/r |
| FastMACE own MD engine (NVE) | MD step (us/atom/step) | 39168 | 3 | 3.95 | 3.95–3.95 | – | – | n/r | n/r | 2610 (torch peak) |
| Symmetrix-XL (LAMMPS, Langevin NVT 300 K) | MD step (us/atom/step) | – | 0 of 3 | (9.00) rejected | 8.96–9.01 | – | – | 47.7→121.3 | 1 | n/r |
| FastMACE own MD engine (NVT Langevin 300 K) | MD step (us/atom/step) | 39168 | 3 | 3.59 | 3.58–3.60 | – | – | n/r | n/r | 2684 (torch peak) |

Failures / skips:

- MACE + cuEquivariance ASE: trial 1 exit 1 (torch.OutOfMemoryError: CUDA out of memory)
- MACE + cuEquivariance ASE: trial 2 exit 1 (torch.OutOfMemoryError: CUDA out of memory)
- MACE + cuEquivariance ASE: trial 3 exit 1 (torch.OutOfMemoryError: CUDA out of memory)

#### Cu(111)-H2O, OMAT-0, 20x20 (108800 atoms)

| setup | boundary | N | trials | median | min–max | force RMSE (meV/Å) | energy err (meV/atom) | T start→end (K) | nbr builds | peak GPU mem (MiB) |
|---|---|---:|---:|---:|---|---:|---:|---|---:|---|
| FastMACE (ASE, compiled, skin 0.5, fp32 energy) | ASE call (us/atom) | 108800 | 3 | 4.49 | 4.49–4.49 | – | – | static | n/r | 6361 (torch peak) |
| Symmetrix-XL (ASE) | ASE call (us/atom) | 108800 | 3 | 9.31 | 9.29–9.32 | – | – | static | n/r | 7002 (nvidia-smi after) |
| Symmetrix-XL (LAMMPS, static) | LAMMPS loop (us/atom/step) | 108800 | 3 | 9.17 | 9.11–9.18 | – | – | static | 0 | n/r |
| MACE + cuEquivariance (ASE; also the reference code) | ASE call (us/atom) | – | 0 | – | ––– | reference | reference | static | n/r | n/r |
| Symmetrix-XL (LAMMPS, NVE MD) | MD step (us/atom/step) | – | 0 of 3 | (9.30) rejected | 9.25–9.32 | – | – | 45.5→113.0 | 1 | n/r |
| FastMACE own MD engine (NVE) | MD step (us/atom/step) | 108800 | 3 | 4.06 | 4.06–4.07 | – | – | n/r | n/r | 6291 (torch peak) |
| Symmetrix-XL (LAMMPS, Langevin NVT 300 K) | MD step (us/atom/step) | – | 0 of 3 | (9.34) rejected | 9.31–9.34 | – | – | 47.7→121.2 | 1 | n/r |
| FastMACE own MD engine (NVT Langevin 300 K) | MD step (us/atom/step) | 108800 | 3 | 3.69 | 3.69–3.69 | – | – | n/r | n/r | 6368 (torch peak) |

Failures / skips:

- MACE + cuEquivariance ASE: trial 1 exit 1 (torch.OutOfMemoryError: CUDA out of memory)
- MACE + cuEquivariance ASE: trial 2 exit 1 (torch.OutOfMemoryError: CUDA out of memory)
- MACE + cuEquivariance ASE: trial 3 exit 1 (torch.OutOfMemoryError: CUDA out of memory)
