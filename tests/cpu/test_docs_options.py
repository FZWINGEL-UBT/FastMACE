"""Every option of every config record has a row in the option reference (docs/reference/)."""

import dataclasses
import re
from pathlib import Path

import pytest

from fastmace import config
from fastmace.planner.workload import Workload

ROOT = Path(__file__).resolve().parents[2]

# Record -> the reference page that documents its fields, one table row each: "| `field` | ...".
PAGES = {
    config.RewriteConfig: "rewrites.md",
    config.PrecisionConfig: "rewrites.md",
    config.EngineConfig: "engine.md",
    config.NeighbourConfig: "engine.md",
    config.FallbackConfig: "planner.md",
    config.GateConfig: "planner.md",
    config.RunnerConfig: "planner.md",
    config.AdaptationConfig: "planner.md",
    config.CalibrationConfig: "planner.md",
    Workload: "planner.md",
    config.IntegratorConfig: "integrators.md",
    config.OptimizerConfig: "integrators.md",
    config.MCConfig: "mc.md",
    config.PolarConfig: "models.md",
}


def test_every_config_record_has_a_page():
    records = {value for value in vars(config).values()
               if isinstance(value, type) and dataclasses.is_dataclass(value) and value.__module__ == config.__name__}
    assert records - set(PAGES) == set(), "add the new record to PAGES and document it in docs/reference/"


@pytest.mark.parametrize("record", list(PAGES), ids=lambda record: record.__name__)
def test_every_field_has_a_row(record):
    text = (ROOT / "docs" / "reference" / PAGES[record]).read_text(encoding="utf-8")
    missing = [field.name for field in dataclasses.fields(record)
               if not re.search(rf"^\|\s*`{re.escape(field.name)}`\s*\|", text, re.MULTILINE)]
    assert missing == [], f"{record.__name__} fields without a row in docs/reference/{PAGES[record]}: {missing}"
