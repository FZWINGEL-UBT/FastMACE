"""``fastmace.kernels.cuda.kspace.cubin_for``: a GPU loads the cubin of its major version with the highest minor
version not above its own (SASS compatibility), so sm_86 (RTX A4000) takes sm_80 and sm_121 takes sm_120; every
listed cubin is shipped."""

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
