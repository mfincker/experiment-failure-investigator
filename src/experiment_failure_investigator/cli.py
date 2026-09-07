"""Command-line interface for the experiment failure investigator."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from experiment_failure_investigator.benchmark.registry import (
    generate_registered_cases,
    validate_registered_cases,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser used by current and future subcommands."""
    parser = argparse.ArgumentParser(
        prog="experiment-failure-investigator",
        description=(
            "Generate, validate, and investigate synthetic biological assay failures."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('experiment-failure-investigator')}",
    )
    subparsers = parser.add_subparsers(dest="command")
    generate_parser = subparsers.add_parser(
        "generate-cases",
        help="Generate the version-controlled Week 1 benchmark cases.",
    )
    generate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("cases"),
        help="Output directory (default: cases).",
    )
    generate_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing registered case directories.",
    )
    validate_parser = subparsers.add_parser(
        "validate-cases",
        help="Validate a generated Week 1 benchmark directory.",
    )
    validate_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("cases"),
        help="Case directory to validate (default: cases).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate-cases":
        cases = generate_registered_cases(args.output, force=args.force)
        print(f"Generated and validated {len(cases)} cases in {args.output}")
    elif args.command == "validate-cases":
        cases = validate_registered_cases(args.path)
        print(f"Validated {len(cases)} cases in {args.path}")
