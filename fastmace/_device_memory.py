"""Release cached device memory and wait until the driver has it back (plans/capacity.md, CAP-T03).

``torch.cuda.empty_cache()`` returns PyTorch's cached blocks to the driver. Under WSL2 (the WDDM driver) the free
memory takes them back about 0.7 s later: a 25,200-atom NbCl5 build released 12 GiB at a time and
``cudaMemGetInfo`` showed it free only after that delay (docs/validation/capacity-T03.md). An allocation in that
window, such as a graph capture right after its ``empty_cache``, can fail near a full card with "CUDA driver
error: device not ready / Failed to create GPU mapping" and leave a sticky error. ``release_cached_memory`` empties
the cache and polls the free memory until the released bytes are back, at most ``timeout_s``. Where the driver
frees at once (Linux), the first poll already sees them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

# Within this of the expected free memory counts as settled: other processes (the desktop) allocate meanwhile.
SLACK_BYTES = 64 * 2**20
POLL_S = 0.02
TIMEOUT_S = 5.0


@dataclass(frozen=True)
class Released:
    """What ``release_cached_memory`` did: the bytes the cache gave back, whether the driver showed them free
    before the timeout, and the seconds it waited."""

    released: int
    settled: bool
    seconds: float


def _cuda():
    import torch

    return torch.cuda


def release_cached_memory(device, timeout_s: float = TIMEOUT_S) -> Released:
    """``torch.cuda.empty_cache()`` on a CUDA ``device``, then wait until its free memory has grown by the bytes
    released (less ``SLACK_BYTES``), polling every ``POLL_S``, at most ``timeout_s``. Nothing off CUDA."""
    if not str(device).startswith("cuda"):
        return Released(0, True, 0.0)
    cuda = _cuda()
    reserved = cuda.memory_reserved(device)
    free_before = cuda.mem_get_info(device)[0] if reserved else 0
    cuda.empty_cache()
    released = reserved - cuda.memory_reserved(device)
    if released <= 0:
        return Released(0, True, 0.0)
    target = free_before + released - SLACK_BYTES
    start = time.perf_counter()
    while True:
        if cuda.mem_get_info(device)[0] >= target:
            return Released(int(released), True, time.perf_counter() - start)
        waited = time.perf_counter() - start
        if waited >= timeout_s:
            return Released(int(released), False, waited)
        time.sleep(POLL_S)
