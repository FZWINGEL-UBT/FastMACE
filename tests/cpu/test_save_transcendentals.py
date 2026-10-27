"""save_transcendentals (ideas-20261027 D3) without a GPU: the record qualifies on the pinned torch, sets one
compile option, and a model compiled with its partitioner gives the eager gradients."""

import pytest
import torch

from fastmace.rewrites import inductor
from fastmace.rewrites import registry as reg


def test_the_record_is_an_exact_opt_in_compile_option():
    record = inductor.RECORDS["save_transcendentals"]
    assert record.exactness == "exact" and record.default == "off" and record.deterministic
    if torch.__version__.split("+")[0] == "2.14.1":
        assert reg.qualify(record).supported
    model = torch.nn.Linear(3, 3)
    assert inductor._apply_save(model, None) == ["custom_partitioner_fn"]
    option = inductor.inductor_options(model)["custom_partitioner_fn"]
    assert option.uuid() == inductor.SAVE_TRANSCENDENTALS_UUID
    assert inductor._save_selfcheck(None, model, None).passed
    with pytest.raises(reg.RewriteError, match="another record"):
        inductor._apply_save(model, None)


def test_the_partitioner_keeps_the_gradients_and_restores_the_op_list():
    from torch._functorch import partitioners

    original = partitioners.get_default_op_list

    def energy(x, w):
        h = torch.nn.functional.silu(x @ w)
        return (torch.tanh(h) * torch.exp(-h) + torch.sigmoid(h)).sum()

    x = torch.randn(32, 8, dtype=torch.float64, requires_grad=True)
    w = torch.randn(8, 8, dtype=torch.float64)
    want = torch.autograd.grad(energy(x, w), x)[0]
    compiled = torch.compile(energy, options={"custom_partitioner_fn": inductor.save_transcendentals_partitioner()})
    got = torch.autograd.grad(compiled(x, w), x)[0]
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    assert partitioners.get_default_op_list is original
