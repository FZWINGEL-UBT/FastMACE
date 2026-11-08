"""``fastmace.kernels.cuda.kspace`` on the host: ``cubin_for`` (a GPU loads the cubin of its major version with the
highest minor version not above its own, so sm_86 takes sm_80 and sm_121 takes sm_120; every listed cubin is
shipped) and ``_plan`` (the grid and the splits of the pair sums)."""

from fastmace.kernels.cuda import kspace


def test_each_capability_takes_the_nearest_lower_cubin_of_its_major():
    assert kspace.cubin_for((8, 0)).name == "kspace_megakernel_sm80.cubin"
    assert kspace.cubin_for((8, 6)).name == "kspace_megakernel_sm80.cubin"
    assert kspace.cubin_for((8, 7)).name == "kspace_megakernel_sm80.cubin"
    assert kspace.cubin_for((8, 9)).name == "kspace_megakernel_sm89.cubin"
    assert kspace.cubin_for((9, 0)).name == "kspace_megakernel_sm90.cubin"
    assert kspace.cubin_for((12, 0)).name == "kspace_megakernel_sm120.cubin"
    assert kspace.cubin_for((12, 1)).name == "kspace_megakernel_sm120.cubin"


def test_no_cubin_below_sm80_or_across_majors():
    assert kspace.cubin_for((7, 5)) is None
    assert kspace.cubin_for((10, 0)) is None


def test_every_cubin_is_shipped():
    assert all(path.is_file() for path in kspace.CUBINS.values())


class _Kernels:
    warps, rows = 8, 32

    def __init__(self, resident):
        self.resident = resident

    def grid(self, name):
        return self.resident


def test_large_sums_take_the_resident_grid():
    """NbCl5 (2989 k-vectors, 1944 atoms) on 48 SMs x 2 blocks: every block busy, as before the planner."""
    grid, ((s1, c1), (s2, c2)) = kspace._plan(_Kernels(96), "e", (94, 1944), (61, 2989))
    assert grid == 96
    assert (s1, c1) == (3, 648) and (s2, c2) == (4, 748)


def test_small_sums_take_fewer_blocks_with_longer_chunks():
    """NaCl (243 k-vectors, 216 atoms): fewer blocks, each warp at least MIN_CHUNK_STEPS steps per item."""
    resident = 96
    grid, plans = kspace._plan(_Kernels(resident), "e", (8, 216), (7, 243))
    assert grid < resident
    min_chunk = kspace.MIN_CHUNK_STEPS * _Kernels.warps
    for (tiles, length), (splits, chunk) in zip(((8, 216), (7, 243)), plans):
        assert chunk >= min(min_chunk, length) and splits * chunk >= length > (splits - 1) * chunk
        assert tiles * splits >= kspace.ITEMS_PER_BLOCK * grid or chunk < 2 * min_chunk


def test_tiny_sums_take_one_block():
    grid, ((splits, chunk),) = kspace._plan(_Kernels(96), "e", (1, 5))
    assert (grid, splits, chunk) == (1, 1, 5)
