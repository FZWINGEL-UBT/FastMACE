# Round 2 screens and A/Bs (5 November 2026)

Records of [plans/ideas-20261105-round2.md](../../plans/ideas-20261105-round2.md), OMAT-0 medium, Benchmark_loup
(SrTiO3 10×10×10, 5,000 atoms) and NbCl5, float32. Results and verdicts are in the plan's Results section and the
[optimisation ledger](../../docs/optimisation-ledger.md) (`omat0-fused-first`, `omat0-horner`, `omat0-eos-walk`,
`omat0-fp16-tc`).

| Directory | GPU, provider | Commit | What |
|---|---|---|---|
| `round2-a100-lab`, `round2-a100-fwd` | A100-SXM4-80GB, Modal | `7a0fbce8`, `a344f471` | kernel lab: 3×FP16 forward, w′ variants, outer-sum walk |
| `round2-a100-ab` | A100-SXM4-80GB, Modal | `38eae731` | lab (8 / 10 warps, walk sweep, walk-dr); gate + A/B HORNER + SCALAR_PATHS, BACKWARD_WALK |
| `round2-a100-fused2` | A100-SXM4-80GB, Modal | `b4f37680` | gate + A/B FUSED_FIRST_BACKWARD, + TABLED_FORWARD |
| `round2-ab-5060ti` | RTX 5060 Ti, local WSL | working tree on `d8c677e1` (the Horner code committed as `38eae731`) | gate + A/B FACTORED without BLOCK_LDG, HORNER, 80 registers |
| `round2-fused-5060ti`, `round2-fused-leto-b70bcad9` | RTX 5060 Ti; RTX 4000 Ada (leto) | `b70bcad9` | fused A/Bs that measured nothing: the fused branch never ran (see the plan) |
| `round2-fused2-5060ti`, `round2-fused-leto-b4f37680` | RTX 5060 Ti; RTX 4000 Ada (leto) | `b4f37680` (local: the working tree, which had `03125b0c`'s 5060 Ti w′ defaults on both sides from the first A/B on) | gate + A/B FUSED_FIRST_BACKWARD, + TABLED_FORWARD |
| `round2-ab-leto-38eae731` | RTX 4000 Ada (leto) | `38eae731` | gate + A/B HORNER; outer-sum lab |
| `round2-lab-leto-2dc9b38d` | RTX 4000 Ada (leto) | `2dc9b38d` | w′ kernel lab |
| `round2-fp16-local*`, `round2-horner-default-local`, `round2-eos-5060`, `round2-ptxas` | RTX 5060 Ti, local WSL | `a344f471`–`38eae731` | kernel labs, `ptxas` register counts |
| `omat0-a100-profile-2e9a3878` | A100 80GB PCIe, Modal | `2e9a3878` | profile with every round-2 default (benchmark and breakdown; the kernel-source step failed in the tool) |
| `round2-tslab-5060` | RTX 5060 Ti, local WSL | working tree on `5ba24365` | `tools/tabled_sum_lab.py`: chunk and warp sweep |
| `round2-degree3.json`, `round2-degree4.json` | CPU | `a344f471` | degree 3 / 4 share of the edges (`tools/degree3_share.py`) |

Environments: local WSL `~/environments/mace_cueq_oeq`, leto NAS `mace_cueq_oeq` (idle check before every job),
Modal image of `tools/modal/fastmace_modal.py` (profile `second`). Modal spend: $1.51 (`38eae731` job) and $1.15
(`b4f37680` job), from the ledger lines in the `*.client.log` files.
