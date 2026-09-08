"""Render the frozen Week 3 investigator output and trace JSON Schemas."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from experiment_failure_investigator.agent.contracts import InvestigatorOutput
from experiment_failure_investigator.agent.trace import InvestigationTrace

SCHEMA_MODELS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("investigator_output.schema.json", InvestigatorOutput),
    ("investigation_trace.schema.json", InvestigationTrace),
)


def render_agent_schemas() -> dict[str, str]:
    """Return every Week 3 schema keyed by its committed filename."""
    return {
        filename: json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        for filename, model in SCHEMA_MODELS
    }


def write_agent_schemas(output_directory: Path) -> tuple[Path, ...]:
    """Write every Week 3 schema to one directory."""
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, contents in render_agent_schemas().items():
        path = output_directory / filename
        path.write_text(contents, encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def main(argv: Sequence[str] | None = None) -> None:
    """Write the Week 3 schemas to a requested directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("docs/schemas"),
    )
    args = parser.parse_args(argv)
    for path in write_agent_schemas(args.output):
        print(path)


if __name__ == "__main__":
    main()
