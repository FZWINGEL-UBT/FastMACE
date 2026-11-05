# Round 5: A100 profile follow-ups (5 November 2026)

The A100 profile at `5b6c5550` is in `results/omat0-a100-profile-5b6c5550/`: Benchmark_loup at 2.38 µs per atom and step (NVE), with an 11.72 ms eager force call. The plan section is in `plans/ideas-20261105-round2.md`, "Round 5".

All runs use MACE-OMAT-0 medium, float32 and the default plan.

## Outcomes

| Idea | Switch | GPU | Result | Verdict |
|---|---|---|---|---|
| Edge spherical harmonics in one Triton kernel each way | `generic.modules.EDGE_SH` | RTX 5060 Ti | MD step SrTiO3 0.985 / 0.961, NbCl5 0.990 / 0.993 | **default** |
| | | RTX 4000 Ada (leto) | 0.989 / 0.987, 0.998 / 0.997 | **default** |
| | | A100 80GB PCIe (Modal) | 0.983 / 0.985, 0.995 / 0.988 | **default (every A100)** |
| Species skip as a grouped GEMM on the A100 (Triton) | `SPECIES_GEMM` | A100-SXM4-80GB | SrTiO3 0.996 / 0.994, NbCl5 1.026 / 1.027 | rejected |
| The same through CUTLASS's gathered entry | `+ species_gemm.CUTLASS` | A100 80GB PCIe | SrTiO3 0.979 / 0.980, NbCl5 1.053 / 1.058 | rejected |
| w from the value table inside the forward | lab `simt-slab-real` | A100-SXM4-80GB | 1.333 of the default forward (3,192 µs) | rejected |
| | | RTX 5060 Ti | 1.356 | rejected |

The table-in-forward rows are the lab bound. `simt-slab3` keeps the table resident in cache and reaches 0.988 on the A100 and 0.820 on the RTX 5060 Ti. `simt-ns-no-gemm`, the forward with no product at all, reaches 0.603 on the A100.

### Force/stress gate

The gate ran with NaCl, NbCl5 and SrTiO3 3×3×3.
- **EDGE_SH:** passes on the RTX 5060 Ti and the RTX 4000 Ada. On the A100 it passes except NbCl5 stress at 1.58e-7. That is the A100's accepted exception, which was 1.57e-7 before this change.
- **SPECIES_GEMM:** the same picture on the A100 (NbCl5 stress 1.53e-7).

### EDGE_SH accuracy

Against e3nn in float64 (`tests/gpu/test_edge_sh_cuda.py`, `tools/edge_sh_lab.py`):
- values: 5.5e-7 (Inductor's e3nn: 5.2e-7);
- vector gradient: 1.35e-7 (the same as Inductor's).

## Runs

The run records are in `runs/`; source snapshots and caches are left out.

| Directory | Commit | What |
|---|---|---|
| `a100-slab/` | `25e5f5f2` | Modal, A100-SXM4 |
| `rtx5060ti-slab/` | `25e5f5f2` | local |
| `a100-species/` | `5b6c5550` | Modal, A100-SXM4 |
| `a100-species-cutlass/` | `5b6c5550` | Modal, A100 PCIe |
| `a100-edgesh/` | `7da10b37` | Modal, A100 PCIe |
| `rtx5060ti-edgesh/` | `25e5f5f2` | local, plus the uncommitted EDGE_SH change that `7da10b37` then committed |
| `rtx4000ada-edgesh/` | `7da10b37` | leto, `mace_cueq_oeq` |
