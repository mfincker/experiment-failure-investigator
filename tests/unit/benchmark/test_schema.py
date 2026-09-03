"""Tests for deterministic case-manifest JSON Schema generation."""

from pathlib import Path

from experiment_failure_investigator.benchmark.schema import render_manifest_schema

REPOSITORY_ROOT = Path(__file__).parents[3]
COMMITTED_SCHEMA = REPOSITORY_ROOT / "docs" / "case_manifest.schema.json"


def test_committed_manifest_schema_is_current() -> None:
    assert COMMITTED_SCHEMA.read_text(encoding="utf-8") == render_manifest_schema()
