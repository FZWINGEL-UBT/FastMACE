"""The plan corpus of phase 5's acceptance (plan task 5-T11; ``tools/plan_corpus.py``, conductor's choice 7).

- The acceptance set: 19 labels x six workloads x NaCl, NbCl5 x three backends, plus the low-memory cases,
  each with a stable unique id.
- A case plans (or is refused, with its rule and code) from the stand-ins, without a checkpoint or a GPU.
- ``compare``: a dump against itself is identical; a planted size change is ``different``; a rewrite
  change carried by an explaining code is ``explained``, without one ``different``; a refusal's reasons in
  another order are identical; decisions only the new side has are listed, not judged.
- ``tools/plan_corpus_dims.json`` holds the sizes of all 19 checkpoints, read from the released files.
- ``tools/acceptance/phase5_local.sh`` parses; a job that waits for the freeze exits 3, one without its
  baseline 2.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fastmace.model import labels

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


corpus = _load("plan_corpus")
corpus_labels = _load("plan_corpus_labels")

SMALL = [corpus.Case("mh1-r2scan", "nvt", "NaCl"), corpus.Case("mh1-r2scan", "npt-d3", "NbCl5", "cueq"),
         corpus.Case("polar1-s", "relax-cell", "NaCl"), corpus.Case("off23-small", "mc-swap", "NaCl", "hybrid"),
         corpus.Case("omol0-xl", "npt", "NbCl5", "auto", "low-memory")]


@pytest.fixture
def small_set(monkeypatch):
    monkeypatch.setitem(corpus.SETS, "small", lambda: list(SMALL))
    return "small"


@pytest.fixture(scope="module")
def small_dump():
    return {"package": "fastmace", "set": "small",
            "cases": {case.id: corpus.plan_case("fastmace", case) for case in SMALL}}


def test_acceptance_set_ids_are_unique_and_complete():
    cases = corpus.acceptance_cases()
    ids = [case.id for case in cases]
    assert len(ids) == len(set(ids)) == 19 * 6 * 2 * 3 + 4 * 2 * 2
    assert {case.label for case in cases} == set(labels.LABELS)
    assert corpus.Case("mh1-r2scan", "nvt", "NaCl").id == "mh1-r2scan/nvt/NaCl/auto/rtx5060ti/float32"


def test_labels_module_is_the_planner_tests_copy():
    from fastmace.model import capabilities

    assert set(corpus_labels.LABELS) == set(labels.LABELS)

    assert capabilities.model_traits(corpus_labels.stand_in("polar1-s")).polar


def test_dims_of_every_checkpoint_were_captured_from_the_released_files():
    for label, entry in labels.LABELS.items():
        dims = corpus_labels.dims(label)
        assert dims is not None, label
        assert corpus_labels.checkpoint_sha256(label) == entry.checkpoint.sha256, label
        assert dims["num_interactions"] == len(corpus_labels.LABEL_FACTS[label][1]), label
        assert dims["model_class"] == corpus_labels.LABEL_FACTS[label][0], label
        assert all(block["conv_tp_irreps_out_dim"] for block in dims["interactions"]), label


def test_cases_plan_or_refuse_with_their_rule(small_dump):
    plans = small_dump["cases"]
    nvt = plans["mh1-r2scan/nvt/NaCl/auto/rtx5060ti/float32"]
    assert nvt["refused"] is None and nvt["tier"] == "graph"
    assert nvt["configs"]["engine"]["backend"] == "hybrid"
    assert nvt["superset"] == [11, 17]
    polar = plans["polar1-s/relax-cell/NaCl/auto/rtx5060ti/float32"]
    assert polar["refused"]["reasons"] == [[1, "polar-variable-cell-optimizer"]]
    d3 = plans["mh1-r2scan/npt-d3/NbCl5/cueq/rtx5060ti/float32"]
    assert d3["configs"]["engine"]["backend"] == "cueq" and d3["configs"]["neighbours"]["d3_skin"] is not None
    json.dumps(small_dump)  # canonical JSON


def test_a_dump_against_itself_is_identical(small_dump):
    report = corpus.compare(small_dump, copy.deepcopy(small_dump))
    assert report["counts"] == {"identical": len(SMALL)}


def test_planted_differences_are_classified(small_dump):
    case_id = "mh1-r2scan/nvt/NaCl/auto/rtx5060ti/float32"
    new = copy.deepcopy(small_dump)
    entry = new["cases"][case_id]
    entry["configs"]["neighbours"]["skin_A"] = 99.0
    assert corpus.compare(small_dump, new)["rows"][case_id]["verdict"] == "different"

    new = copy.deepcopy(small_dump)
    entry = new["cases"][case_id]
    entry["configs"]["rewrites"]["planted"] = "off"
    assert corpus.compare(small_dump, new)["rows"][case_id]["verdict"] == "different"
    entry["decisions"]["memory"] = "drop planted"
    entry["decision_codes"]["memory"] = [12, "rule-12-planted"]
    row = corpus.compare(small_dump, new, frozenset({"rule-12-planted"}))["rows"][case_id]
    assert row["verdict"] == "explained" and row["by"] == ["memory"] and row["added_decisions"] == ["memory"]

    # An explaining code does not excuse a change outside the rewrite set.
    entry["tier"] = "compiled"
    assert corpus.compare(small_dump, new, frozenset({"rule-12-planted"}))["rows"][case_id]["verdict"] == "different"


def test_added_decisions_and_texts_are_not_judged(small_dump):
    case_id = "mh1-r2scan/nvt/NaCl/auto/rtx5060ti/float32"
    new = copy.deepcopy(small_dump)
    entry = new["cases"][case_id]
    entry["decisions"]["warn.model_structure"] = "warn"
    entry["decision_codes"]["warn.model_structure"] = [15, "model-structure-warning"]
    entry["texts"] = {key: "reworded" for key in entry["texts"]}
    entry["phase5"]["predicted_memory_bytes"] = 123
    row = corpus.compare(small_dump, new)["rows"][case_id]
    assert row == {"verdict": "identical", "added_decisions": ["warn.model_structure"]}


def test_refusal_reasons_compare_as_a_multiset(small_dump):
    case_id = "polar1-s/relax-cell/NaCl/auto/rtx5060ti/float32"
    old, new = copy.deepcopy(small_dump), copy.deepcopy(small_dump)
    old["cases"][case_id]["refused"]["reasons"] = [[1, "a"], [2, "b"]]
    new["cases"][case_id]["refused"]["reasons"] = [[2, "b"], [1, "a"]]
    assert corpus.compare(old, new)["rows"][case_id]["verdict"] == "identical"
    new["cases"][case_id]["refused"]["reasons"] = [[2, "b"]]
    assert corpus.compare(old, new)["rows"][case_id]["verdict"] == "different"


def test_explaining_codes_come_from_the_new_packages_rule_table():
    codes = corpus.explaining_codes("fastmace", [12], ["extra"])
    assert "rule-12-memory" in codes and "extra" in codes and "model-structure-warning" not in codes


def test_cli_dump_and_compare(small_set, tmp_path, capsys):
    out = tmp_path / "a.json"
    assert corpus.main(["dump", "--set", small_set, "--out", str(out)]) == 0
    dumped = json.loads(out.read_text())
    assert dumped["count"] == len(SMALL) and dumped["package"] == "fastmace"
    assert corpus.main(["compare", str(out), str(out), "--explain-rules", "12", "--out",
                        str(tmp_path / "r.json")]) == 0
    changed = copy.deepcopy(dumped)
    next(iter(changed["cases"].values()))["tier"] = "eager"
    other = tmp_path / "b.json"
    other.write_text(json.dumps(changed))
    assert corpus.main(["compare", str(out), str(other)]) == 1
    assert "different" in capsys.readouterr().out


def test_a_renamed_exports_package_name_reads_back_as_fastmace():
    entry = {"configs": {"integrator": {"dof": "fastmace_4"}}, "texts": {"x": "fastmace_4.planner said so"}}
    assert corpus._unrenamed(entry, "fastmace_4") == {"configs": {"integrator": {"dof": "fastmace"}},
                                                      "texts": {"x": "fastmace.planner said so"}}
    assert corpus._unrenamed({"a": "fastmace_40"}, "fastmace_4") == {"a": "fastmace_40"}


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_phase_5_job_script_parses_and_refuses_what_is_not_ready(tmp_path):
    script = ROOT / "tools" / "acceptance" / "phase5_local.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    unknown = subprocess.run(["bash", str(script), str(tmp_path), "no-such-job"], env=env, capture_output=True)
    assert unknown.returncode == 2
    pending = subprocess.run(["bash", str(script), str(tmp_path), "adapt-storm"], env=env, capture_output=True)
    assert pending.returncode == 3 and b"filled in at the freeze" in pending.stderr
    baseline = subprocess.run(["bash", str(script), str(tmp_path), "corpus"], env=env, capture_output=True)
    assert baseline.returncode == 2 and b"FASTMACE_4_COMMIT" in baseline.stderr
