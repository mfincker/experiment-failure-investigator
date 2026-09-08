"""Regression tests for frozen Week 3 agent JSON Schemas."""

from pathlib import Path

from experiment_failure_investigator.agent.schema import render_agent_schemas

REPOSITORY_ROOT = Path(__file__).parents[3]
SCHEMA_DIRECTORY = REPOSITORY_ROOT / "docs" / "schemas"


def test_committed_agent_schemas_are_current() -> None:
    for filename, rendered in render_agent_schemas().items():
        assert (SCHEMA_DIRECTORY / filename).read_text(encoding="utf-8") == rendered
