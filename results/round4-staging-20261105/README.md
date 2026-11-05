# Round 4: the fused forward's staging (5 November 2026)

OMAT-0 medium, float32, Benchmark_loup SrTiO3 10×10×10 (5,000 atoms) and NbCl5. Plan and discussion:
[round 4](../../plans/ideas-20261105-round2.md#round-4-the-fused-forwards-staging-5-november-2026); ledger entries
`omat0-ff-vector-staging` (adopted), `omat0-ff-prefetch`, `omat0-ff-hints`, `omat0-atom-order`, `omat0-fp16x3-gemm`,
`omat0-radial-table-shapes`, `omat0-tp-factor`, `omat0-table-in-forward` (rejected).

## Outcome

Adopted per GPU (decision 66), on branch `fastmace/round4-staging`:

- `fused_forward.VEC_SH`: on every A100 and the RTX 5060 Ti.
- `VEC_A` and `TC_WFRAG`: on every A100.

MD step b/a, both build orders:

| GPU | Switches | SrTiO3 | NbCl5 |
|---|---|---|---|
| A100-SXM4-80GB (Modal) | VEC_A + VEC_SH + TC_WFRAG | 0.960 / 0.954 | 0.970 / 0.969 |
| A100-SXM4-80GB (Modal) | VEC_A + VEC_SH | 0.973 / 0.976 | 0.976 / 0.981 |
| RTX 5060 Ti (local) | VEC_SH | 0.977 / 0.977 | 0.984 / 0.984 |

## Records (`runs/`)

Lab runs use `tools/omat0_kernel_lab.py`; A/Bs use `tools/modal/round2_ab.sh`. Each Modal directory has
`modal.json`, `gpu.txt` and `pip-freeze.txt`.

| Directory | GPU, provider | What |
|---|---|---|
| `round4-a100-fwd` | A100-SXM4, Modal | forward attribution falsifiers |
| `round4-a100-prefetch`, `round4-a100-xprefetch` | A100, Modal | cp.async prefetch (rejected) |
| `round4-a100-vec`, `round4-a100-vec2` | A100, Modal | sh4, tc-a4 lab |
| `round4-a100-xs` | A100, Modal | x shared tile (does not launch), tc-a4, F-no-xload |
| `round4-a100-hints` | A100, Modal | cache hints (rejected) |
| `round4-a100-wfrag` | A100, Modal | tc-wfrag lab |
| `round4-a100-veca` | A100-SXM4, Modal | GPU test, gate and MD A/B: VEC_A + VEC_SH |
| `round4-a100-wfrag-ab` | A100-SXM4, Modal | GPU test, gate and MD A/B: + TC_WFRAG |
| `round4-a100-fp16x3` | A100, Modal | `tools/fp16x3_gemm_bench.py` (rejected on accuracy) |
| `round4-vecsh-5060ti` | RTX 5060 Ti, local | MD A/B VEC_SH |
| `round4-orders` | RTX 5060 Ti, local | atom order lab (`orders/`) |
| other `round4-*` | RTX 5060 Ti, local | lab screens (prefetch, sh4, hints, wfrag, radial table) |
| `leto/` | RTX 4000 Ada, leto | lab screens (prefetch, vec, a4); source snapshots and caches not kept |

Environments:

- Local and leto: `mace_cueq_oeq`.
- Modal: FastMACE's image (`tools/modal/`).
