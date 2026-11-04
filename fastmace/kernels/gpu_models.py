"""GPU model sets of the per-GPU defaults (decision 66), matched by CUDA device name.

Owner, 4 November 2026: "turn on a100 speedups no matter which version of it it runs on". An A100 entry in a
``GPUModels`` set stands for every A100 model (SXM4 and PCIe, 40 and 80 GB: the same GA100 chip with 108 SMs and
compute capability 8.0; clocks, power and memory bandwidth differ). Every other entry is an exact device name.
"""

from __future__ import annotations

import re

_A100 = re.compile(r"NVIDIA A100([- ]|$)")


def is_a100(name):
    """Whether a CUDA device name is an A100 model ("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", ...)."""
    return isinstance(name, str) and _A100.match(name) is not None


class GPUModels(tuple):
    """Device names; ``name in models`` also holds for every A100 model when one A100 is listed."""

    def __contains__(self, name):
        return tuple.__contains__(self, name) or (is_a100(name) and any(is_a100(n) for n in self))

    def __add__(self, other):
        return GPUModels(tuple(self) + tuple(other))


__all__ = ["GPUModels", "is_a100"]
