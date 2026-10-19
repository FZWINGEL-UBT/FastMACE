"""The ``"d3bj"`` expansion of ``fm.D3`` (plan task 1b-T20; SPEC §7.7 "Dispersion", §11.2).

``dispersion="d3bj"`` expands to the complete record from the head's training functional in the
label table, with the §11.2 defaults. It raises ``ValueError`` naming the reason when the training
reference already contains dispersion (MACE-OFF, OMOL-0, MH-1's ``omol`` and ``spice_wB97M``
heads) and when the functional is unknown (POLAR-1, a path or URL). An explicit ``fm.D3`` is used
as given; a functional other than the training one is a warning. Torch-free.
"""

from __future__ import annotations

import pytest

from fastmace.longrange import d3
from fastmace.longrange.d3 import D3
from fastmace.model import labels

# The parameter set every label's "d3bj" expands to, or the reason it is refused (a word of the
# ValueError). Written out by hand from SPEC §7.7, so a label-table change fails here.
EXPECTED = {
    "mp0-medium": "pbe", "mp0-large": "pbe", "mp0b3-medium": "pbe", "mpa0-medium": "pbe", "omat0-medium": "pbe",
    "matpes-pbe": "pbe", "matpes-r2scan": "r2scan", "mh1-omat": "pbe", "mh1-mp": "pbe", "mh1-r2scan": "r2scan",
    "polar1-s": "unknown", "polar1-m": "unknown", "polar1-l": "unknown",
    "omol0-xl": "VV10",
    "off23-small": "D3(BJ)", "off23-medium": "D3(BJ)", "off23-large": "D3(BJ)", "off23b-medium": "D3(BJ)",
    "off24-medium": "D3(BJ)",
}


def test_every_label_is_listed():
    assert set(EXPECTED) == set(labels.LABELS)


@pytest.mark.parametrize("label", sorted(EXPECTED))
def test_d3bj_expansion_per_label(label):
    entry = labels.LABELS[label]
    expected = EXPECTED[label]
    if expected in d3.D3BJ_PARAMETER_SETS:
        record, warnings = d3.expand("d3bj", functional=entry.functional, model=label)
        assert record == D3(functional=expected)
        assert warnings == ()
    else:
        with pytest.raises(ValueError, match=expected.replace("(", r"\(").replace(")", r"\)")) as raised:
            d3.expand("d3bj", functional=entry.functional, model=label)
        assert label in str(raised.value)
        assert "fm.D3" in str(raised.value) or "twice" in str(raised.value)


def test_expansion_is_the_spec_record():
    record, _ = d3.expand("d3bj", functional="r2scan", model="mh1-r2scan")
    assert record.damping == "bj"
    assert (record.pair_cutoff_bohr, record.cn_cutoff_bohr) == (40.0, 40.0)
    assert (record.three_body, record.smoothing, record.precision) == (False, "none", "float64")
    assert (record.autoang, record.autoev) == (0.52917726, 27.21138505)


@pytest.mark.parametrize("head,word", [("omol", "VV10"), ("spice_wB97M", "D3(BJ)"), ("matpes_r2scan", None),
                                       ("oc20_usemppbe", None)])
def test_mh1_heads(head, word):
    entry = labels.find("mh1", head=head)
    if word is None:
        record, _ = d3.expand("d3bj", functional=entry.functional, model=f"mh1 head {head}")
        assert record.functional == entry.functional
    else:
        with pytest.raises(ValueError, match="twice"):
            d3.expand("d3bj", functional=entry.functional, model=f"mh1 head {head}")


def test_unknown_functional_needs_an_explicit_record():
    with pytest.raises(ValueError, match=r"unknown.*fm\.D3\(functional="):
        d3.expand("d3bj", functional=None, model="/models/my.model")


def test_functional_without_a_parameter_set_is_refused():
    with pytest.raises(ValueError, match="b3lyp"):
        d3.expand("d3bj", functional="b3lyp", model="x")


def test_explicit_record_is_used_as_given():
    record = D3(functional="pbe", three_body=True)
    assert d3.expand(record, functional="pbe", model="mp0-medium") == (record, ())
    # Even where "d3bj" would refuse: an explicit record is the user's choice.
    assert d3.expand(record, functional=None, model="/models/my.model") == (record, ())


def test_explicit_record_with_another_functional_warns():
    record = D3(functional="pbe")
    got, warnings = d3.expand(record, functional="r2scan", model="mh1-r2scan")
    assert got is record
    assert len(warnings) == 1 and "pbe" in warnings[0] and "r2scan" in warnings[0]


def test_none_and_bad_values():
    assert d3.expand(None, functional="pbe", model="x") == (None, ())
    with pytest.raises(ValueError, match="d3bj"):
        d3.expand("d3", functional="pbe", model="x")
    with pytest.raises(TypeError, match="dispersion"):
        d3.expand(3, functional="pbe", model="x")


def test_record_text_names_every_field():
    text = d3.describe(D3(functional="r2scan"))
    for field in ("functional=r2scan", "damping=bj", "pair_cutoff_bohr=40", "cn_cutoff_bohr=40", "three_body=False",
                  "smoothing=none", "precision=float64", "autoang=0.52917726", "autoev=27.21138505"):
        assert field in text
