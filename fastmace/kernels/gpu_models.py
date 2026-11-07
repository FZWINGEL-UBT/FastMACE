"""GPU model sets of the per-GPU defaults (decision 66), matched by CUDA device name and GPU family.

Owner, 4 November 2026: "turn on a100 speedups no matter which version of it it runs on". Owner, 7 November 2026,
generalising it: "if its tested on one variant of a gpu it counts for all gpus (unless the variants vary significantly
in performance)". An entry in a ``GPUModels`` set therefore stands for every variant of its family:

- A100: SXM4 and PCIe, 40 and 80 GB. The same GA100 chip with 108 SMs and compute capability 8.0; clocks, power and
  memory bandwidth differ.
- H200: the SXM card ("NVIDIA H200") and the H200 NVL. The same GH100 chip with 132 SMs and 141 GB of HBM3e at
  4.8 TB/s; the NVL card has a lower power limit.

The H100 variants are not a family: the PCIe card has 114 SMs and HBM2e at 2.0 TB/s against the SXM card's 132 SMs and
HBM3 at 3.35 TB/s, a significant difference in performance. Every other entry is an exact device name.
"""

from __future__ import annotations

import re

_A100 = re.compile(r"NVIDIA A100([- ]|$)")
_H200 = re.compile(r"NVIDIA H200([- ]|$)")


def is_a100(name):
    """Whether a CUDA device name is an A100 model ("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", ...)."""
    return isinstance(name, str) and _A100.match(name) is not None


def family(name):
    """The decision-66 family of a CUDA device name: "A100", "H200", or the name itself."""
    if not isinstance(name, str):
        return name
    if _A100.match(name):
        return "A100"
    if _H200.match(name):
        return "H200"
    return name


class GPUModels(tuple):
    """Device names; ``name in models`` holds for every variant of a listed GPU's family (A100, H200)."""

    def __contains__(self, name):
        return tuple.__contains__(self, name) or any(family(name) == family(n) for n in self)

    def __add__(self, other):
        return GPUModels(tuple(self) + tuple(other))


def lookup(table, name, default=None):
    """``table[name]`` for a per-GPU dict, matching by family as ``GPUModels`` does (an exact name wins)."""
    if name in table:
        return table[name]
    key = family(name)
    for listed, value in table.items():
        if family(listed) == key:
            return value
    return default


__all__ = ["GPUModels", "family", "is_a100", "lookup"]
