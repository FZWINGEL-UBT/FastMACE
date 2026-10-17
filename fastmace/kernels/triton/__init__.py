"""Triton kernels of the exact rewrites (SPEC §5 ``kernels/triton``).

- ``edge_outer_sum``: per-slot sums of edge outer products with a fused
  backward, and the run-segmented forward (reassociated first interaction,
  segmented first sum).
- ``skinny_linear``: the folded radial last layer with a Triton input
  gradient (fused folded backward).
- ``fused_radial``: the fused LayerNorm/SiLU/Linear hidden block of the radial
  networks (off by default; kept for CLI parity).
- ``inplace_linear``: the in-place block-diagonal linear layer (opt-in only,
  SPEC §6.2; strided-batched GEMMs, no Triton).
- ``cache``: where Triton and Inductor keep their caches (under
  ``fastmace._cache``).

Every kernel is built on first use; importing a module here imports torch but
not triton. On the CPU, or for dtypes and shapes a kernel does not cover, the
custom ops run the same formulas in PyTorch. Kernel configurations are fixed
for every crystal and GPU (no autotuning, no shape-keyed tables).
"""
