"""Peak tensor memory of a CPU computation (plan task 5-T03): rule 12's estimate checked on the tiny models.

The CPU allocator keeps no peak statistics, so ``PeakTracker`` counts storages itself. It is a dispatch
mode: every storage an operator returns is added once, and a storage is dropped when its weak reference
expires (autograd's saved tensors keep theirs alive, as on the GPU). ``peak`` is the largest sum of live
bytes seen since entering, over the storages created inside; tensors made before entering (the inputs,
the weights) are not counted.
"""

from __future__ import annotations

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves


class PeakTracker(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.live: dict[int, tuple[StorageWeakRef, int]] = {}
        self.current = 0
        self.peak = 0

    def _sweep(self):
        for key in [key for key, (ref, _) in self.live.items() if ref.expired()]:
            self.current -= self.live.pop(key)[1]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self._sweep()
        for leaf in tree_leaves(out):
            if isinstance(leaf, torch.Tensor) and leaf.layout == torch.strided:
                storage = leaf.untyped_storage()
                key = storage.data_ptr()
                if storage.nbytes() and key not in self.live:
                    self.live[key] = (StorageWeakRef(storage), storage.nbytes())
                    self.current += storage.nbytes()
        self.peak = max(self.peak, self.current)
        return out


def peak_bytes(fn, *args, **kwargs) -> int:
    """The peak live bytes of the storages ``fn(*args, **kwargs)`` creates."""
    with PeakTracker() as tracker:
        fn(*args, **kwargs)
    return tracker.peak
