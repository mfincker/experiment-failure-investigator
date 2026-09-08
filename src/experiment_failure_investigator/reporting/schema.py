"""Render the frozen investigator and deterministic-report JSON Schemas."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from experiment_failure_investigator.analysis.contracts import InvestigatorCase
from experiment_failure_investigator.analysis.results import ScientificToolResult
from experiment_failure_investigator.reporting.baseline import BaselineReport

SCHEMA_MODELS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("investigator_case.schema.json", InvestigatorCase),
    ("scientific_tool_result.schema.json", ScientificToolResult),
    ("baseline_report.schema.json", BaselineReport),
)


def render_model_schema(model: type[BaseModel]) -> str:
    """Return one stable, newline-terminated Pydantic JSON Schema."""
    return json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"


def render_frozen_schemas() -> dict[str, str]:
    """Return every frozen Week 2 schema keyed by its committed filename."""
    return {name: render_model_schema(model) for name, model in SCHEMA_MODELS}


def write_frozen_schemas(output_directory: Path) -> tuple[Path, ...]:
    """Write every frozen Week 2 schema to one directory."""
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, contents in render_frozen_schemas().items():
        path = output_directory / filename
        path.write_text(contents, encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def main(argv: Sequence[str] | None = None) -> None:
    """Write the frozen schemas to a requested directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("docs/schemas"),
    )
    args = parser.parse_args(argv)
    for path in write_frozen_schemas(args.output):
        print(path)


if __name__ == "__main__":
    main()
