"""The monthly Modal budget guard (MODAL-T02; owner: "on modal make sure to in total not use more than the 30$ credit
i get for free per month").

Pure Python, no Modal import, so the CPU tests can fake the billing calls. ``fastmace_modal.py`` calls
``check`` before every launch and ``record`` after it.

- **Cap:** ``CAP_USD`` = $30 per calendar month (Modal's free credit) with a ``RESERVE_USD`` = $3 reserve. A job
  is refused when this month's metered cost plus the job's worst case exceeds ``LIMIT_USD`` = $27.
- **Lower limits:** ``limit`` may be set lower (``FASTMACE_MODAL_LIMIT_USD`` or ``--limit-usd``) and never higher,
  except with an explicit owner override (``--owner-override "who: why"``). The override is written to the
  ledger with the job.
- **Worst case:** (function timeout + ``START_S``) × (GPU rate + CPU limit × CPU rate + memory limit × memory
  rate), from Modal's own rate table (``modal billing rates``). The CPU and memory limits are the job function's
  hard limits, so the bound holds.
- **Jobs still running:** the worst cases of jobs launched from this checkout that have not finished yet
  (``in_flight``, a JSON file; ``holding`` adds and removes the entry) count as spent, so parallel launches cannot
  each fit on their own and overrun together. An entry expires at its own worst-case end.
- **Unreadable billing:** a summary or rate table that cannot be read or parsed refuses the job.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

CAP_USD = Decimal("30")
RESERVE_USD = Decimal("3")
LIMIT_USD = CAP_USD - RESERVE_USD
START_S = 300  # container start, image pull and the volume commit, billed like the job (generous)
GPU_RATE_KEYS = {"L4": "gpu_hour_cost_l4", "A100": "gpu_hour_cost_a100_40gb", "A100-80GB": "gpu_hour_cost_a100_80gb",
                 "H100": "gpu_hour_cost_h100", None: None}
LEDGER_FIELDS = ("date_utc", "commit", "gpu", "timeout_s", "wall_s", "estimated_cost_usd", "worst_case_usd",
                 "metered_before_usd", "metered_after_usd", "metered_delta_usd", "limit_usd", "override", "exit")


class BudgetRefused(RuntimeError):
    """A job the monthly budget does not allow (or whose cost cannot be checked)."""


@dataclass(frozen=True)
class Decision:
    allowed: bool
    metered_usd: Decimal
    worst_case_usd: Decimal
    pending_usd: Decimal
    limit_usd: Decimal
    override: str
    message: str
    worst_case_s: int = 0


def _decimal(value, what):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise BudgetRefused(f"refused: {what} is not readable ({value!r})") from error
    if not number.is_finite():
        raise BudgetRefused(f"refused: {what} is not a finite number ({value!r})")
    return number


def resolve_limit(requested=None, override="", environ=os.environ):
    """The month's limit in USD: ``LIMIT_USD`` unless set lower; higher only with an owner override."""
    raw = requested if requested not in (None, "") else environ.get("FASTMACE_MODAL_LIMIT_USD", "")
    if raw in (None, ""):
        return LIMIT_USD
    limit = _decimal(raw, "the requested limit")
    if limit < 0:
        raise BudgetRefused(f"refused: a negative limit ({limit})")
    if limit > LIMIT_USD and not override.strip():
        raise BudgetRefused(f"refused: limit ${limit} is above ${LIMIT_USD} (${CAP_USD} cap - ${RESERVE_USD} "
                            "reserve); raising it needs --owner-override \"who: why\"")
    return limit


def _per_hour(rates, gpu, cpu_limit, memory_limit_gib):
    try:
        gpu_key = GPU_RATE_KEYS[gpu]
    except KeyError as error:
        raise BudgetRefused(f"refused: no rate for GPU {gpu!r}") from error

    def rate(key):
        if key is None:
            return Decimal(0)
        try:
            value = rates.get(key) if hasattr(rates, "get") else rates[key]
        except Exception as error:  # noqa: BLE001  (any failure to read the table refuses the job)
            raise BudgetRefused(f"refused: the rate table cannot be read ({error})") from error
        if value is None:
            raise BudgetRefused(f"refused: the rate table has no {key}")
        return _decimal(value, key)

    return (rate(gpu_key) + Decimal(str(cpu_limit)) * rate("cpu_hour_cost")
            + Decimal(str(memory_limit_gib)) * rate("mem_gib_hour_cost"))


def cost(rates, gpu, seconds, cpu_limit, memory_limit_gib):
    """USD for ``seconds`` of one container at its limits (the ledger's estimate of a finished job)."""
    hours = Decimal(str(round(float(seconds), 3))) / Decimal(3600)
    return (hours * _per_hour(rates, gpu, cpu_limit, memory_limit_gib)).quantize(Decimal("0.0001"))


def worst_case(rates, gpu, timeout_s, cpu_limit, memory_limit_gib):
    """USD for a job that runs to its timeout plus the start allowance, at Modal's rate table ``rates``."""
    if not timeout_s or timeout_s <= 0:
        raise BudgetRefused("refused: every Modal function needs a positive timeout")
    return cost(rates, gpu, int(timeout_s) + START_S, cpu_limit, memory_limit_gib)


@contextlib.contextmanager
def _locked(path):
    """``path``'s entries under an exclusive lock (``fcntl`` where available), written back on exit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(path.suffix + ".lock"), "a") as lock:
        try:
            import fcntl
        except ImportError:  # Windows: no lock, the CPU tests only
            fcntl = None
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            entries = json.loads(path.read_text()) if path.exists() and path.read_text().strip() else {}
        except ValueError as error:
            raise BudgetRefused(f"refused: {path} cannot be read ({error}); check it and delete it") from error
        now = time.time()
        entries = {key: entry for key, entry in entries.items() if float(entry["expires"]) > now}
        yield entries
        path.write_text(json.dumps(entries, indent=1) + "\n")


def pending(in_flight):
    """The summed worst case of the jobs in ``in_flight`` that may still be running."""
    if in_flight is None:
        return Decimal(0)
    with _locked(in_flight) as entries:
        return sum((Decimal(entry["worst_case_usd"]) for entry in entries.values()), Decimal(0))


@contextlib.contextmanager
def holding(in_flight, decision):
    """Count ``decision``'s worst case as spent while the job runs (until it ends or its worst case expires)."""
    key = uuid.uuid4().hex
    with _locked(in_flight) as entries:
        entries[key] = dict(worst_case_usd=str(decision.worst_case_usd), pid=os.getpid(),
                            expires=time.time() + decision.worst_case_s)
    try:
        yield key
    finally:
        with _locked(in_flight) as entries:
            entries.pop(key, None)


def check(read_metered, read_rates, gpu, timeout_s, cpu_limit, memory_limit_gib, limit=None, override="",
          in_flight=None):
    """Refuse (``BudgetRefused``) unless this month's metered cost, the worst case of the jobs still running
    (``in_flight``) and this job's worst case together fit the limit."""
    limit_usd = resolve_limit(limit, override)
    try:
        metered = _decimal(read_metered(), "this month's metered cost")
        rates = read_rates()
    except BudgetRefused:
        raise
    except Exception as error:  # noqa: BLE001  (unreadable billing refuses)
        raise BudgetRefused(f"refused: Modal billing cannot be read ({type(error).__name__}: {error})") from error
    worst = worst_case(rates, gpu, timeout_s, cpu_limit, memory_limit_gib)
    running = pending(in_flight)
    total = metered + running + worst
    message = (f"Modal budget: metered this month ${metered:.2f} + still running ${running:.2f} "
               f"+ this job's worst case ${worst:.2f} "
               f"({gpu or 'CPU'}, {timeout_s} s + {START_S} s start) = ${total:.2f}, limit ${limit_usd:.2f} "
               f"(cap ${CAP_USD}, reserve ${RESERVE_USD}){'; owner override: ' + override if override else ''}")
    if total > limit_usd:
        raise BudgetRefused("refused: " + message)
    return Decision(True, metered, worst, running, limit_usd, override, message, int(timeout_s) + START_S)


def record(ledger, decision, *, commit, gpu, timeout_s, wall_s, estimated_cost, metered_after, exit_code):
    """Append the job to the month ledger (CSV; the header is written once)."""
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    new = not ledger.exists()
    after = None if metered_after is None else Decimal(str(metered_after))
    row = dict(date_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), commit=commit, gpu=gpu or "CPU",
               timeout_s=timeout_s, wall_s=round(float(wall_s), 1), estimated_cost_usd=f"{float(estimated_cost):.4f}",
               worst_case_usd=f"{decision.worst_case_usd:.4f}", metered_before_usd=f"{decision.metered_usd:.4f}",
               metered_after_usd="" if after is None else f"{after:.4f}",
               metered_delta_usd="" if after is None else f"{after - decision.metered_usd:.4f}",
               limit_usd=f"{decision.limit_usd:.2f}", override=decision.override, exit=exit_code)
    with ledger.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEDGER_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)
    return row


__all__ = ["BudgetRefused", "CAP_USD", "Decision", "LIMIT_USD", "RESERVE_USD", "START_S", "check", "cost",
           "holding", "pending", "record", "resolve_limit", "worst_case"]
