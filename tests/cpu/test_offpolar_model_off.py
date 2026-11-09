"""Per-model exclusions of three A100 defaults for MACE-OFF and POLAR-1 (decision 66; branch
fastmace/off-polar-a100-r2, 9 November 2026, results/off-polar-a100-r2-20261109): ``SORT_BY_LENGTH`` off for
OFF23-small and OFF23-large, ``STAGE_DOUT`` and ``DEGREE_PARTS`` off for POLAR-1-M, each on the A100 family, and the
model label's way to each switch (the neighbour list, the POLAR w' attach, the MH-1 fused forward)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastmace.config import NeighbourConfig, RewriteConfig
from fastmace.kernels.oeq_streaming import fused_forward, wprime_backward
from fastmace.neighbors import skin

A100 = "NVIDIA A100-SXM4-80GB"
A100_PCIE = "NVIDIA A100 80GB PCIe"
H200 = "NVIDIA H200"


@pytest.fixture
def gpu(monkeypatch):
    """Pretend the CUDA device is the named GPU."""
    def use(name):
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: name)
    return use


def test_the_maps_list_the_measured_models_on_the_a100_family():
    assert set(skin.SORT_BY_LENGTH_MODEL_OFF_GPUS) == {"off23-small", "off23-large"}
    assert set(wprime_backward.STAGE_DOUT_MODEL_OFF_GPUS) == {"polar1-m"}
    assert set(fused_forward.DEGREE_PARTS_MODEL_OFF_GPUS) == {"polar1-m"}
    for table in (skin.SORT_BY_LENGTH_MODEL_OFF_GPUS, wprime_backward.STAGE_DOUT_MODEL_OFF_GPUS,
                  fused_forward.DEGREE_PARTS_MODEL_OFF_GPUS):
        for gpus in table.values():
            assert A100 in gpus and A100_PCIE in gpus and H200 not in gpus


@pytest.mark.parametrize("label", ["off23-small", "off23-large"])
def test_sort_by_length_is_off_for_the_listed_off_models_on_the_a100(gpu, label):
    gpu(A100)
    assert not skin.sort_by_length_default("cuda", label)
    assert skin.sort_by_length_default("cuda", "off23-medium")
    assert skin.sort_by_length_default("cuda", "polar1-m")
    assert skin.sort_by_length_default("cuda")
    gpu(H200)
    assert skin.sort_by_length_default("cuda", label)


def test_the_sort_constant_overrides_the_model_map(gpu, monkeypatch):
    gpu(A100)
    monkeypatch.setattr(skin, "SORT_BY_LENGTH", True)
    assert skin.sort_by_length_default("cuda", "off23-small")


def test_stage_dout_and_degree_parts_are_off_for_polar1_m_on_the_a100(gpu, monkeypatch):
    gpu(A100)
    assert not wprime_backward.stage_dout_default("cuda", "polar1-m")
    assert wprime_backward.stage_dout_default("cuda", "polar1-s")
    assert wprime_backward.stage_dout_default("cuda")
    assert not fused_forward.degree_parts_default(A100, "polar1-m")
    assert fused_forward.degree_parts_default(A100, "omat0-medium")
    assert fused_forward.degree_parts_default(A100)
    assert fused_forward.degree_parts_default(H200, "polar1-m")
    gpu(H200)
    assert wprime_backward.stage_dout_default("cuda", "polar1-m")
    monkeypatch.setattr(wprime_backward, "STAGE_DOUT", True)
    monkeypatch.setattr(fused_forward, "DEGREE_PARTS", True)
    gpu(A100)
    assert wprime_backward.stage_dout_default("cuda", "polar1-m")
    assert fused_forward.degree_parts_default(A100, "polar1-m")


def test_the_neighbour_list_passes_its_label_to_the_sort_default(monkeypatch):
    seen = []
    monkeypatch.setattr(skin, "sort_by_length_default", lambda device, model_label=None: seen.append(model_label))
    config = NeighbourConfig(sort_receivers="on", gpu_neighbour_list="off")
    skin.SkinNeighbourList(np.eye(3) * 10.0, 5.0, "cpu", torch.float64, config, model_label="off23-small")
    assert seen == ["off23-small"]


def test_the_wprime_and_fused_forward_records_pass_the_label(monkeypatch):
    from fastmace.rewrites import mh1, radial
    from fastmace.rewrites.registry import Context

    seen = []
    monkeypatch.setattr(radial, "attach_wprime", lambda model, model_label=None: seen.append(("w", model_label)))
    monkeypatch.setattr(mh1, "fused_folded_forwards",
                        lambda model, model_label=None: seen.append(("f", model_label)))
    context = Context(config=RewriteConfig(), inputs={"model_label": "polar1-m"})
    radial.RECORDS["wprime_backward"].apply(SimpleNamespace(), context)
    mh1.RECORDS["fused_folded_forward"].apply(SimpleNamespace(), context)
    assert seen == [("w", "polar1-m"), ("f", "polar1-m")]
