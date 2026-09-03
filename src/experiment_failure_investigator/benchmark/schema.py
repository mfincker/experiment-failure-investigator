"""Render the evaluation-only case manifest as JSON Schema."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from experiment_failure_investigator.benchmark.models import CaseManifest


def render_manifest_schema() -> str:
    """Return the canonical, newline-terminated manifest JSON Schema."""
    return json.dumps(CaseManifest.model_json_schema(), indent=2, sort_keys=True) + "\n"


def write_manifest_schema(output: Path) -> None:
    """Write the canonical manifest schema to ``output``."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_manifest_schema(), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    """Write the schema to a requested repository-relative destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("docs/case_manifest.schema.json"),
    )
    args = parser.parse_args(argv)
    write_manifest_schema(args.output)


if __name__ == "__main__":
    main()
