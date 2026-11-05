# NaCl 216 atoms: FastMACE against public Symmetrix-XL (RTX 5060 Ti, 5 November 2026)

`tools/bench/vram_compare.py` on the local RTX 5060 Ti 16 GB (WSL2, driver 610.88), desktop idle: MACE-OMAT-0 medium,
float32, NaCl 3x3x3 (216 atoms, `ResearchModule/Structures/Benchmark/R2SCAN/NaCl.cif`), Langevin 300 K, 1 fs,
friction 0.002 per fs, skin 0.5 Å, Maxwell-Boltzmann start (seed 20261105); 500 steps, then 3,000 timed. Three rounds,
the two codes alternating (Symmetrix-XL first in each).

- **FastMACE** 1.0.0, branch `fastmace/round5` at `b03fc94e` (main 0d581bdd plus tools; the RTX 5060 Ti's default
  plan, hybrid float32), environment `mace_cueq_oeq`.
- **Symmetrix-XL 0.1.1**: LAMMPS (Kokkos CUDA, `symmetrix/mace/float32/kk`, `no_domain_decomposition streamed_edges
  direct profile speed`, `SYMMETRIX_A1_CUBLAS=0`), the build of the 23 October rematch (ResearchModule
  `Experiments/Langevin_Thermostat/studies/symmetrix_xl/README.md`); inputs and logs in `sx-work-*`.

| Round | Symmetrix-XL ms/step | FastMACE ms/step | Ratio |
|---|---:|---:|---:|
| 1 | 3.553 | 1.100 | 3.23 |
| 2 | 3.575 | 1.182 | 3.03 |
| 3 | 3.614 | 1.156 | 3.13 |

Per atom and step: Symmetrix-XL 16.4–16.7 µs, FastMACE 5.1–5.5 µs; FastMACE is 3.0–3.2x faster. On 23 October the
same comparison (with the research-era FastMACE, 1.62–1.65 ms) gave 2.0x. GPU memory over the idle baseline:
Symmetrix-XL 455 MiB, FastMACE 713–727 MiB (PyTorch 483 MiB allocated, 540 MiB reserved).
