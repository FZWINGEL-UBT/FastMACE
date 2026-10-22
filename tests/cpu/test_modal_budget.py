"""MODAL-T02: the monthly Modal budget guard (tools/modal/budget.py), with the billing calls faked."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODAL_DIR = ROOT / "tools" / "modal"


def _load_budget():
    spec = importlib.util.spec_from_file_location("fastmace_modal_budget", MODAL_DIR / "budget.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up
    spec.loader.exec_module(module)
    return module


budget = _load_budget()

# Modal's rate table as `modal billing rates` returns it (strings), October 2026.
RATES = {"gpu_hour_cost_l4": "0.80", "gpu_hour_cost_a100_40gb": "2.10", "gpu_hour_cost_a100_80gb": "2.50",
         "gpu_hour_cost_h100": "3.95", "cpu_hour_cost": "0.04730", "mem_gib_hour_cost": "0.00800"}
JOB = dict(gpu="L4", timeout_s=4200, cpu_limit=8.0, memory_limit_gib=32.0)


def _check(metered, rates=RATES, **kwargs):
    job = {**JOB, **{k: kwargs.pop(k) for k in list(kwargs) if k in JOB}}
    read_metered = metered if callable(metered) else (lambda: metered)
    read_rates = rates if callable(rates) else (lambda: rates)
    return budget.check(read_metered, read_rates, job["gpu"], job["timeout_s"], job["cpu_limit"],
                        job["memory_limit_gib"], **kwargs)


def _worst():
    return budget.worst_case(RATES, JOB["gpu"], JOB["timeout_s"], JOB["cpu_limit"], JOB["memory_limit_gib"])


@pytest.fixture(autouse=True)
def _no_env_limit(monkeypatch):
    monkeypatch.delenv("FASTMACE_MODAL_LIMIT_USD", raising=False)


def test_cap_is_30_with_a_3_reserve():
    assert budget.CAP_USD == 30 and budget.RESERVE_USD == 3 and budget.LIMIT_USD == 27


def test_worst_case_counts_gpu_cpu_memory_and_start():
    # (4200 + 300) s at 0.80 + 8 x 0.0473 + 32 x 0.008 = 1.4344 USD/h
    assert _worst() == Decimal("1.7930")
    assert budget.START_S == 300


def test_under_the_cap_is_allowed():
    decision = _check("3.65")
    assert decision.allowed
    assert decision.metered_usd == Decimal("3.65") and decision.worst_case_usd == _worst()
    assert "$3.65" in decision.message and "$27.00" in decision.message


def test_exactly_at_the_cap_is_allowed():
    decision = _check(str(budget.LIMIT_USD - _worst()))
    assert decision.allowed


def test_over_the_cap_is_refused_naming_the_numbers():
    metered = budget.LIMIT_USD - _worst() + Decimal("0.01")
    with pytest.raises(budget.BudgetRefused) as refused:
        _check(str(metered))
    text = str(refused.value)
    assert f"${metered:.2f}" in text and f"${_worst():.2f}" in text and "$27.00" in text


@pytest.mark.parametrize("read_metered", [
    lambda: (_ for _ in ()).throw(ConnectionError("no network")),
    lambda: None,
    lambda: "n/a",
    lambda: "NaN",
])
def test_unreadable_metered_cost_refuses(read_metered):
    with pytest.raises(budget.BudgetRefused, match="refused"):
        _check(read_metered)


@pytest.mark.parametrize("rates", [
    lambda: (_ for _ in ()).throw(PermissionError("no billing access")),
    {k: v for k, v in RATES.items() if k != "cpu_hour_cost"},
    {**RATES, "gpu_hour_cost_l4": "?"},
])
def test_unreadable_rates_refuse(rates):
    with pytest.raises(budget.BudgetRefused, match="refused"):
        _check("1.00", rates=rates)


def test_unknown_gpu_refuses():
    with pytest.raises(budget.BudgetRefused, match="no rate"):
        _check("1.00", gpu="B200")


@pytest.mark.parametrize("timeout_s", [0, None, -5])
def test_a_job_without_a_timeout_is_refused(timeout_s):
    with pytest.raises(budget.BudgetRefused, match="timeout"):
        _check("1.00", timeout_s=timeout_s)


def test_a_lower_limit_is_allowed_and_applied(monkeypatch):
    with pytest.raises(budget.BudgetRefused):
        _check("1.00", limit="2.50")
    assert _check("1.00", limit="5").limit_usd == 5
    monkeypatch.setenv("FASTMACE_MODAL_LIMIT_USD", "2.50")
    with pytest.raises(budget.BudgetRefused):
        _check("1.00")


def test_a_higher_limit_needs_an_owner_override():
    with pytest.raises(budget.BudgetRefused, match="owner-override"):
        _check("26.00", limit="40")
    with pytest.raises(budget.BudgetRefused, match="owner-override"):
        _check("26.00", limit="40", override="   ")
    decision = _check("26.00", limit="40", override="owner: thesis deadline")
    assert decision.limit_usd == 40 and decision.override == "owner: thesis deadline"
    assert "owner override: owner: thesis deadline" in decision.message


def test_negative_limit_refuses():
    with pytest.raises(budget.BudgetRefused):
        _check("0", limit="-1")


def test_jobs_still_running_count_as_spent(tmp_path):
    in_flight = tmp_path / "in-flight.json"
    first = _check("20.00", in_flight=in_flight)
    with budget.holding(in_flight, first):
        assert budget.pending(in_flight) == _worst()
        second = _check("20.00", in_flight=in_flight)  # 20 + 1.79 + 1.79 fits
        with budget.holding(in_flight, second):
            with pytest.raises(budget.BudgetRefused, match="still running"):
                _check(str(budget.LIMIT_USD - 2 * _worst()), in_flight=in_flight)
    assert budget.pending(in_flight) == 0


def test_expired_in_flight_entries_are_dropped(tmp_path):
    in_flight = tmp_path / "in-flight.json"
    in_flight.write_text(json.dumps({"old": dict(worst_case_usd="9", pid=1, expires=time.time() - 1)}))
    assert budget.pending(in_flight) == 0


def test_ledger_records_the_delta(tmp_path):
    ledger = tmp_path / "results" / "modal-ledger.csv"
    decision = _check("3.65")
    budget.record(ledger, decision, commit="abc123", gpu="L4", timeout_s=4200, wall_s=612.34,
                  estimated_cost=Decimal("0.2440"), metered_after=Decimal("3.93"), exit_code=0)
    budget.record(ledger, decision, commit="abc123", gpu="L4", timeout_s=4200, wall_s=10,
                  estimated_cost=Decimal("0.0040"), metered_after=None, exit_code="error")
    rows = list(csv.DictReader(ledger.open()))
    assert len(rows) == 2 and tuple(rows[0]) == budget.LEDGER_FIELDS
    assert rows[0]["commit"] == "abc123" and rows[0]["gpu"] == "L4" and rows[0]["wall_s"] == "612.3"
    assert rows[0]["metered_before_usd"] == "3.6500" and rows[0]["metered_after_usd"] == "3.9300"
    assert rows[0]["metered_delta_usd"] == "0.2800" and rows[0]["estimated_cost_usd"] == "0.2440"
    assert rows[1]["metered_after_usd"] == "" and rows[1]["exit"] == "error"


def _keywords(call):
    return {k.arg: k.value for k in call.keywords if k.arg}


def test_every_modal_function_has_a_timeout_no_retries_and_short_idle():
    tree = ast.parse((MODAL_DIR / "fastmace_modal.py").read_text())
    functions = [d for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) for d in node.decorator_list
                 if isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "function"]
    assert len(functions) >= 2
    for decorator in functions:
        keywords = _keywords(decorator)
        assert "timeout" in keywords, ast.unparse(decorator)
        assert ast.unparse(keywords.get("retries", ast.Constant(None))) == "0", ast.unparse(decorator)
        assert ast.unparse(keywords.get("scaledown_window", ast.Constant(None))) == "2", ast.unparse(decorator)
        assert {"cpu", "memory"} <= set(keywords), ast.unparse(decorator)  # the worst case uses these limits
    options = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and getattr(node.func, "attr", "") == "with_options"]
    assert options
    for call in options:
        keywords = _keywords(call)
        assert "timeout" in keywords and ast.unparse(keywords.get("retries", ast.Constant(None))) == "0"


def test_every_remote_call_is_behind_the_guard():
    """``.remote(`` appears only inside a ``launch`` closure handed to ``_guarded``."""
    tree = ast.parse((MODAL_DIR / "fastmace_modal.py").read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name == "launch":
            continue
        inner = [n for n in node.body if not (isinstance(n, ast.FunctionDef) and n.name == "launch")]
        for stmt in inner:
            for call in ast.walk(stmt):
                if isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "remote":
                    pytest.fail(f"{node.name} calls .remote outside a guarded launch")
    launchers = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and any(isinstance(c, ast.FunctionDef) and c.name == "launch" for c in n.body)]
    assert {n.name for n in launchers} == {"upload", "listing", "_job"}
    for launcher in launchers:
        assert any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_guarded" for c in ast.walk(launcher))
