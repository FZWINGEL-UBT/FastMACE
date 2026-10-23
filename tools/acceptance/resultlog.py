"""A pytest plugin that keeps per-test results when an acceptance run is stopped (3b-T10).

Phase 3's partial local acceptance tier lost its junit file: stopping the job's shell ended pytest before
``pytest_sessionfinish`` wrote it (docs/validation/phase-3.md, open finding 7). This plugin does two things:

- ``--result-log PATH`` appends one JSON line per finished test phase that decides the outcome (node id,
  outcome, phase, duration, UTC time, the failure text's last line), flushed and fsynced at once, so even a
  killed run keeps every finished test.
- At configure time it makes SIGINT raise ``KeyboardInterrupt`` again (a non-interactive bash starts
  background jobs with SIGINT ignored, and Python then leaves it ignored) and turns SIGTERM and SIGHUP into
  ``KeyboardInterrupt`` too. pytest then ends the session normally, and ``--junitxml`` is written.

Load it with ``PYTHONPATH=<checkout>/tools/acceptance pytest -p resultlog --result-log PATH ...``;
``tools/acceptance/stoppable_pytest.sh`` runs pytest this way and forwards its own signals as SIGINT.
"""

from __future__ import annotations

import json
import os
import signal
from datetime import datetime, timezone

STOP_SIGNALS = ("SIGINT", "SIGTERM", "SIGHUP")


def pytest_addoption(parser):
    parser.addoption("--result-log", default=None, metavar="PATH",
                     help="Append one JSON line per finished test (resultlog plugin, 3b-T10)")


def _interrupt(signum, frame):
    raise KeyboardInterrupt(f"stopped by {signal.Signals(signum).name}")


def pytest_configure(config):
    for name in STOP_SIGNALS:
        number = getattr(signal, name, None)
        if number is not None:
            signal.signal(number, signal.default_int_handler if name == "SIGINT" else _interrupt)
    path = config.getoption("--result-log")
    if path:
        config._resultlog = ResultLog(path)
        config.pluginmanager.register(config._resultlog, "resultlog-writer")


def pytest_unconfigure(config):
    writer = getattr(config, "_resultlog", None)
    if writer is not None:
        writer.close()
        config.pluginmanager.unregister(writer)


class ResultLog:
    """The JSON-lines writer: one line per test outcome, written as soon as it is known."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.file = open(path, "a", encoding="utf-8")
        self.running = None  # the node id of the test in progress
        self.write({"event": "session_start", "pid": os.getpid()})

    def pytest_runtest_logstart(self, nodeid, location):
        self.running = nodeid

    def pytest_runtest_logfinish(self, nodeid, location):
        self.running = None

    def pytest_keyboard_interrupt(self, excinfo):
        # junit records the interrupted test only as an unnamed <testcase/>; this line names it.
        self.write({"event": "interrupted", "nodeid": self.running, "message": str(excinfo.value)})

    def write(self, record):
        record = {"time_utc": datetime.now(timezone.utc).isoformat(), **record}
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())

    def pytest_runtest_logreport(self, report):
        # The call phase decides a test; setup decides it when it fails or skips; teardown only when it fails.
        if report.when == "call" or (report.when == "setup" and not report.passed) or (
                report.when == "teardown" and report.failed):
            message = None
            if not report.passed:
                text = str(report.longrepr) if report.longrepr is not None else ""
                message = text.strip().splitlines()[-1] if text.strip() else None
            self.write({"event": "test", "nodeid": report.nodeid, "when": report.when, "outcome": report.outcome,
                        "duration_s": round(report.duration, 3), "message": message})

    def pytest_sessionfinish(self, session, exitstatus):
        self.write({"event": "session_finish", "exitstatus": int(exitstatus)})

    def close(self):
        if not self.file.closed:
            self.file.close()
