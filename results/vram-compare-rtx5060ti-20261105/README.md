# GPU memory: FastMACE against Symmetrix-XL on 5,000 atoms (RTX 5060 Ti, 5 November 2026)

`tools/bench/vram_compare.py` on the local RTX 5060 Ti 16 GB (WSL2, driver 610.88): MACE-OMAT-0 medium, float32,
SrTiO3 10x10x10 (5,000 atoms, the Benchmark_loup cell `results/loup-a100-20261103/SrTiO3.cif`), Langevin 300 K, 1 fs,
friction 0.002 per fs, skin 0.5 Å, Maxwell-Boltzmann start (seed 20261105); 100 steps, then 300 timed. GPU memory is
`nvidia-smi` used memory sampled every 0.1 s, peak minus the idle baseline right before (5,365 MiB, the desktop), so
it is the whole process: CUDA context, libraries, model, buffers. One run each, one after the other.

| Code | Peak GPU memory over baseline | PyTorch peak allocated / reserved | Step | Per atom and step |
|---|---:|---:|---:|---:|
| Symmetrix-XL 0.1.1, LAMMPS (Kokkos CUDA, `symmetrix/mace/float32/kk`, `streamed_edges direct profile speed`, `SYMMETRIX_A1_CUBLAS=0`) | **767 MiB** | n/a | 50.8 ms | 10.2 µs |
| FastMACE 1.0.0 (round5 branch, a88fa6f0 plus draw-ahead 349f03e3 and budget cb6c81ce; default plan for this GPU, hybrid float32) | **1,381 MiB** | 1,174 / 1,218 MiB | 24.4 ms | 4.9 µs |

FastMACE needs 1.8x the memory and is 2.1x faster per step on this cell. About 1.2 GB of its 1.38 GB is PyTorch's
pool (the CUDA graphs' buffers, the 61 MB drawn-ahead noise batch, the w' derivative table); the rest is the CUDA
context and libraries. Symmetrix-XL's LAMMPS build is the one of the 23 October rematch (ResearchModule
`Experiments/Langevin_Thermostat/studies/symmetrix_xl/README.md`); its input is `sx-work/in.md`, its log
`sx-work/log.lammps`. Not run on an A100 (no Modal credits spent on this comparison, owner).
