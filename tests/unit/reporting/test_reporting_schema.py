"""Regression tests for the frozen Week 2 JSON Schemas."""

from pathlib import Path

from experiment_failure_investigator.reporting.schema import render_frozen_schemas

REPOSITORY_ROOT = Path(__file__).parents[3]
SCHEMA_DIRECTORY = REPOSITORY_ROOT / "docs" / "schemas"


def test_committed_week_2_schemas_are_current() -> None:
    for filename, rendered in render_frozen_schemas().items():
        assert (SCHEMA_DIRECTORY / filename).read_text(encoding="utf-8") == rendered
