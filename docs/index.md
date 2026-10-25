# FastMACE

FastMACE runs long molecular-dynamics trajectories with pretrained MACE
models on one NVIDIA GPU, as fast as we have measured any tool do it, and
exact to MACE within float32 rounding.

- **Fast:** CUDA graphs over the whole step, exact model rewrites and a
  cuEquivariance/OpenEquivariance hybrid backend. On an RTX 5060 Ti, MH-1
  takes 2.90 ms per step on a 216-atom NaCl cell.
- **Checked:** every rewrite is verified against MACE when it is applied,
  and an optional gate checks the fast path on your own structure.
- **Complete:** NVT, NPT and NVE integrators, GPU relaxations, species swaps
  at graph speed, D3 dispersion, PolarMACE and full-state restarts.

## Where to start

- [Install](install.md), then [FastMACE in five minutes](quickstart.md).
- Have an ASE script? [Port it](guide/ase.md) by changing one import.
- Look up a function in the [API reference](api/index.md).
- Something fails? See [FAQ and troubleshooting](faq.md).

## Further reading in the repository

These records are kept in the repository and are not part of this site:

- [Specification (SPEC.md)](https://github.com/FZWINGEL-UBT/FastMACE/blob/main/SPEC.md)
- [Phase plans](https://github.com/FZWINGEL-UBT/FastMACE/tree/main/plans)
- [Design notes](https://github.com/FZWINGEL-UBT/FastMACE/tree/main/docs/design),
  including the [positioning against other tools](https://github.com/FZWINGEL-UBT/FastMACE/blob/main/docs/design/2026-10-positioning.md)
- [Validation records](https://github.com/FZWINGEL-UBT/FastMACE/tree/main/docs/validation)
