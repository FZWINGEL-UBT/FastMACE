"""Releasing cached device memory and waiting for the driver (plans/capacity.md, CAP-T03; fastmace._device_memory).

Under WSL2 the driver frees the blocks ``torch.cuda.empty_cache()`` unmaps about 0.7 s later
(docs/validation/capacity-T03.md): an allocation made at once can fail ("device not ready / Failed to create GPU
mapping") near a full card. ``release_cached_memory`` empties the cache and polls the free memory until the released
bytes are back, with a timeout. The CUDA calls are faked here.
"""

import pytest

from fastmace import _device_memory as dm


class FakeCuda:
    """``memory_reserved`` drops by ``released`` at ``empty_cache``; ``mem_get_info``'s free memory takes them back
    after ``lag`` polls."""

    def __init__(self, released, lag, free=1000):
        self.reserved, self.released, self.lag, self.free, self.polls = 10**10, released, lag, free, 0
        self.returned = released  # what the free memory takes back
        self.emptied = False

    def memory_reserved(self, device=None):
        return self.reserved

    def empty_cache(self):
        self.reserved -= self.released
        self.emptied = True

    def mem_get_info(self, device=None):
        if self.emptied:
            self.polls += 1
        back = self.returned if self.emptied and self.polls > self.lag else 0
        return self.free + back, 10_000


@pytest.fixture
def fake(monkeypatch):
    def install(**kwargs):
        cuda = FakeCuda(**kwargs)
        monkeypatch.setattr(dm, "_cuda", lambda: cuda)
        monkeypatch.setattr(dm.time, "sleep", lambda seconds: None)
        return cuda
    return install


def test_waits_until_the_released_bytes_are_free(fake):
    cuda = fake(released=dm.SLACK_BYTES * 4, lag=5)
    result = dm.release_cached_memory("cuda:0", timeout_s=1.0)
    assert cuda.emptied and result.released == dm.SLACK_BYTES * 4 and result.settled
    assert cuda.polls == 6  # five polls still without the released bytes, the sixth with them


def test_returns_at_once_when_nothing_was_cached(fake):
    cuda = fake(released=0, lag=5)
    result = dm.release_cached_memory("cuda:0")
    assert result.released == 0 and result.settled and cuda.polls == 0


def test_gives_up_after_the_timeout(fake, monkeypatch):
    clock = iter(range(0, 1000))
    monkeypatch.setattr(dm.time, "perf_counter", lambda: next(clock) * 0.1)
    fake(released=dm.SLACK_BYTES * 4, lag=10**6)
    result = dm.release_cached_memory("cuda:0", timeout_s=0.5)
    assert not result.settled and result.seconds >= 0.5


def test_a_small_shortfall_counts_as_settled(fake):
    """Others may allocate meanwhile: within SLACK_BYTES of the target is enough."""
    cuda = fake(released=dm.SLACK_BYTES * 4, lag=0)
    cuda.returned = cuda.released - dm.SLACK_BYTES // 2
    assert dm.release_cached_memory("cuda:0", timeout_s=0.0).settled
    cuda = fake(released=dm.SLACK_BYTES * 4, lag=0)
    cuda.returned = cuda.released - 2 * dm.SLACK_BYTES
    assert not dm.release_cached_memory("cuda:0", timeout_s=0.0).settled


def test_off_cuda_it_does_nothing(monkeypatch):
    monkeypatch.setattr(dm, "_cuda", lambda: pytest.fail("no CUDA call off CUDA"))
    result = dm.release_cached_memory("cpu")
    assert result.released == 0 and result.settled
