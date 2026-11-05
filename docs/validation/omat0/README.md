# OMAT-0 reports

OMAT-0 (generic family) speed and capacity, 2–5 November 2026, mostly on the A100. How the parts fit together: [current/omat0-a100.md](../../current/omat0-a100.md); what is on per GPU: [current/defaults-by-gpu.md](../../current/defaults-by-gpu.md).

Columns: file, date added, title, and the report's own verdict line or first sentence.

| File | Added | Title | Verdict / first sentence |
| --- | --- | --- | --- |
| [bf16x3-facto](bf16x3-facto.md) | 2026-11-03 | BF16×3 atom GEMMs (FACTO, arXiv 2611.03276): accuracy and speed (3 November 2026) | Result: rejected on all three GPUs measured. It stays within the force gate but is slower than the default |
| [cutlass-tf32x3](cutlass-tf32x3.md) | 2026-11-03 | CUTLASS 3×TF32 GEMMs on the A100 (3 November 2026) | Owner request: "build it for a100", after `tf32x3_gemms` on Triton measured slower than IEEE on the A100 (NaCl 3.1 → 4.2 ms) while a standalone CUTLASS 3×TF32 benchmark … |
| [generic-fold-product](generic-fold-product.md) | 2026-11-05 | The C3 fold for the generic family: `generic_fold_product_linear` (4–5 November 2026) | Verdict: default on the RTX 4000 Ada and every A100 (decision 66, `planner.fixed.DECISION_66_GENERIC_FOLD_PRODUCT_GPUS`). |
| [generic-radial-tables](generic-radial-tables.md) | 2026-11-02 | Generic radial tables: MACE-OMAT-0 medium | `generic_radial_tables` extends MH-1's certified radial tables (`radial_tables`, decision 61) to the generic MACE family: MACE-MP-0, MACE-OMAT-0, and the density blocks. |
| [generic-wprime](generic-wprime.md) | 2026-11-03 | Generic fused forward and wprime_backward for OMAT-0 (3 November 2026) | Owner request: "Yes, build it", after the opt-9 results (MH-1's `wprime_backward` 11–19 % on the A100) raised the question whether the generic folded path has it. |
| [loup-a100](loup-a100.md) | 2026-11-03 | Benchmark_loup on the A100: why Symmetrix-XL's MD step was faster (3 November 2026) | Background. MargrafGroup's Benchmark_loup (private repository, commit `89ab3d8`) compared complete MD steps on an NVIDIA A100 80GB PCIe: - Setup: MACE-OMAT-0 medium, … |
| [omat0-capacity](omat0-capacity.md) | 2026-11-02 | MACE-OMAT-0 medium: the generic folded path and its memory rescue (2 November 2026) | decision 51's 10 % margin. |
| [omat0-tp-backward](omat0-tp-backward.md) | 2026-11-04 | A model-specialised tensor-product backward for OMAT-0 (4 November 2026) | Verdict: adopt `wprime_backward.FACTORED`. Default on the RTX 4000 Ada and every A100 (`FACTORED_GPUS`, owner, |
