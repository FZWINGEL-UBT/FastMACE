"""step-layout-cleanup (12 November 2026) on CUDA: the species products read and write strided views instead of
copies (bitwise the copies' products), the composed first interaction's joined layout compiles without a layout
copy around the products, and the bound species plan keeps the sort out of the compiled step."""
import pytest
import torch

from fastmace.kernels.triton import species_gemm as sg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SIZES = [1, 3, 5, 7]


def _operands(seed=0, atoms=301, k=96, n=128, species=3):
    g = torch.Generator(device="cuda").manual_seed(seed)
    index = torch.randint(0, species, (atoms,), generator=g, device="cuda")
    x = torch.randn(atoms * sum(SIZES), k, generator=g, device="cuda")
    ws = [torch.randn(species, k, n, generator=g, device="cuda") for _ in SIZES]
    w_ts = [w.transpose(1, 2).contiguous() for w in ws]
    return index, x, ws, w_ts


def _per_path(x, ws, w_ts, index, precision):
    """The old route: each path's rows copied out, ``species_mm``, joined by ``torch.cat``."""
    order, counts = sg.species_plan(index, ws[0].shape[0])
    views = sg._rows_views(x, SIZES)
    return torch.cat([sg.species_mm(v.contiguous(), w, w_t, index, order, counts, precision)
                      for v, w, w_t in zip(views, ws, w_ts)], dim=1)


def _rows(x, ws, w_ts, index, precision, plan=None):
    order, counts = sg.species_plan(index, ws[0].shape[0]) if plan is None else plan
    _, scatters = sg.joined_rows(order, SIZES)
    return sg.species_mm_rows(x, SIZES, ws, w_ts, index, order, counts, scatters, precision)


@pytest.mark.parametrize("precision", ["ieee", "tf32x3"])
@pytest.mark.parametrize("compiled", [False, True])
def test_rows_read_in_place_equal_the_copied_paths(precision, compiled):
    index, x, ws, w_ts = _operands()
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    rows = torch.compile(_rows, fullgraph=True) if compiled else _rows
    got = rows(a, ws, w_ts, index, precision)
    want = _per_path(b, ws, w_ts, index, precision)
    assert torch.equal(got, want)
    up = torch.randn_like(got)
    (ga,), (gb,) = torch.autograd.grad(got, [a], up), torch.autograd.grad(want, [b], up)
    assert torch.equal(ga, gb)


def _kernels(fn, *args):
    from torch._inductor.utils import run_and_get_code

    torch._dynamo.reset()
    _, codes = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
    return "\n".join(codes)


def test_the_joined_layout_compiles_without_layout_copies():
    """The old route compiles a copy per path and a cat (the Ada attribution's ``clone_slice_species_mm_transpose``
    and ``cat_view`` kernels at generic/modules.py:211); the joined layout compiles none, and with the plan bound
    (the force field's, outside the step) no sort either."""
    index, x, ws, w_ts = _operands()
    plan = sg.species_plan(index, ws[0].shape[0])

    def old(x):
        nodes, total = index.shape[0], sum(SIZES)
        summed = x.view(nodes, total, -1)
        lefts = [summed[:, s:s + size].transpose(0, 1).transpose(0, 1).contiguous()
                 for s, size in zip([0, 1, 4, 9], SIZES)]
        blocks = [sg.species_mm(left, w, w_t, index, plan[0], plan[1], "tf32x3").reshape(nodes, -1)
                  for left, w, w_t in zip(lefts, ws, w_ts)]
        return torch.cat(blocks, dim=1)

    def new(x):
        return _rows(x, ws, w_ts, index, "tf32x3", plan).reshape(index.shape[0], -1)

    old_code = _kernels(old, x)
    assert "cat" in old_code  # the probe sees the copies where they are
    new_code = _kernels(new, x)
    assert "triton_poi_fused_cat" not in new_code and "clone" not in new_code
    assert "sort" not in new_code
    assert torch.equal(old(x), new(x))


def test_the_family_sets_load_the_sm80_cubin_on_another_8x(monkeypatch):
    """Item 5 (CUTLASS on the RTX 4000 Ada, compute capability 8.9): ``FAMILY_CAPABILITIES`` or ``FAMILY_GPUS`` lets
    an 8.x device without its own cubin load the sm_80 one; with both empty it does not."""
    from fastmace.kernels.cutlass import tf32x3

    capability = tuple(torch.cuda.get_device_capability())
    if capability[0] != 8 or capability in tf32x3.CUBINS:
        pytest.skip("needs an 8.x device without its own cubin")
    monkeypatch.setattr(tf32x3, "FAMILY_GPUS", ())
    assert not tf32x3.available("cuda")
    monkeypatch.setattr(tf32x3, "FAMILY_CAPABILITIES", (capability,))
    assert tf32x3.available("cuda")
    monkeypatch.setattr(tf32x3, "FAMILY_CAPABILITIES", ())
    monkeypatch.setattr(tf32x3, "FAMILY_GPUS", (torch.cuda.get_device_name(),))
    assert tf32x3.available("cuda")


def test_the_ada_runs_its_products_on_cutlass_by_default():
    """The RTX 4000 Ada's default after item 5: the CUTLASS backend (by model, the sm_80 cubin) with LINEAR_DIV."""
    from fastmace.kernels.cutlass import linear_div
    from fastmace.kernels.triton import tf32x3_gemm

    name = torch.cuda.get_device_name()
    if name != "NVIDIA RTX 4000 Ada Generation":
        pytest.skip("the RTX 4000 Ada's default")
    assert tf32x3_gemm.backend("cuda") == "cutlass"
    assert linear_div.linear_div_default(name)
