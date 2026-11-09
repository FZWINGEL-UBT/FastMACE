"""``RewriteConfig.cueq_projected``: the planner's per-model exclusion of ``CUEQ_PROJECTED`` (decision 66,
``planner.fixed.DECISION_66_CUEQ_PROJECTED_EXCLUDED``; branch fastmace/off-polar-a100, 8 November 2026)."""

from types import SimpleNamespace

import pytest

from fastmace.config import RewriteConfig
from fastmace.planner import fixed
from fastmace.rewrites import cueq_projection

A100 = "NVIDIA A100-SXM4-80GB"


class Book:
    def __init__(self):
        self.decisions = []

    def decide(self, item, choice, reason):
        self.decisions.append((item, choice, reason))


def decide(asked, tf32x3, label, gpu):
    book = Book()
    choice = fixed.FixedPlanner._cueq_projected(book, asked, tf32x3, label, SimpleNamespace(name=gpu))
    (item, recorded, reason), = book.decisions
    assert item == "RewriteConfig.cueq_projected" and recorded == choice
    return choice, reason


def test_the_field_defaults_to_auto_with_its_rule():
    assert RewriteConfig().cueq_projected == "auto"
    assert fixed.AUTO_RULES["RewriteConfig.cueq_projected"] == (
        "on_with_tf32x3_gemms_on_listed_gpus_unless_model_excluded_decision_66")


def test_the_excluded_models_are_off23_large_and_polar1_s_and_m_on_the_a100_family():
    assert set(fixed.DECISION_66_CUEQ_PROJECTED_EXCLUDED) == {"off23-large", "polar1-s", "polar1-m"}
    for gpus in fixed.DECISION_66_CUEQ_PROJECTED_EXCLUDED.values():
        assert A100 in gpus and "NVIDIA A100 80GB PCIe" in gpus and "NVIDIA H200" not in gpus


@pytest.mark.parametrize("label", ["off23-large", "polar1-s", "polar1-m"])
def test_an_excluded_model_is_off_on_the_a100_with_its_reason(label):
    choice, reason = decide("auto", "on", label, A100)
    assert choice == "off"
    assert reason.startswith("auto (on_with_tf32x3_gemms_on_listed_gpus_unless_model_excluded_decision_66)")
    assert f"{label} measured not faster on {A100}, decision 66" in reason


@pytest.mark.parametrize("label", ["off23-small", "off23-medium", "off23b-medium", "off24-medium", "polar1-l",
                                   "omat0-medium"])
def test_other_models_are_on_with_tf32x3_on_the_a100(label):
    assert decide("auto", "on", label, A100)[0] == "on"


def test_an_excluded_model_stays_on_where_it_is_not_excluded():
    assert decide("auto", "on", "off23-large", "NVIDIA H200")[0] == "on"


def test_off_without_tf32x3_or_on_an_unlisted_gpu():
    assert decide("auto", "off", "off23-small", A100) == ("off", "auto (on_with_tf32x3_gemms_on_listed_gpus_unless_"
                                                              "model_excluded_decision_66): no effect: tf32x3_gemms is "
                                                              "off")
    assert decide("auto", "on", "off23-small", "NVIDIA GeForce RTX 5060 Ti")[0] == "off"
    assert decide("auto", "on", "off23-small", None)[0] == "off"


@pytest.mark.parametrize("asked", ["on", "off"])
def test_an_explicit_choice_is_kept(asked):
    assert decide(asked, "on", "off23-large", A100)[0] == asked


def test_the_config_value_overrides_the_gpu_list_and_the_constant_overrides_both(monkeypatch):
    assert cueq_projection.cueq_projected_default(A100)  # auto: the GPU list
    assert not cueq_projection.cueq_projected_default(A100, "off")
    assert cueq_projection.cueq_projected_default("NVIDIA GeForce RTX 5060 Ti", "on")
    monkeypatch.setattr(cueq_projection, "CUEQ_PROJECTED", True)  # an A/B side
    assert cueq_projection.cueq_projected_default(A100, "off")
    monkeypatch.setattr(cueq_projection, "CUEQ_PROJECTED", False)
    assert not cueq_projection.cueq_projected_default(A100, "on")


def test_precision_passes_the_config_to_the_route(monkeypatch):
    from fastmace.rewrites import precision
    from fastmace.rewrites.registry import Context

    seen = []
    monkeypatch.setattr(precision, "route", lambda model, cueq_projected="auto": seen.append(cueq_projected))
    precision._apply(object(), Context(config=RewriteConfig(cueq_projected="off")))
    assert seen == ["off"]
