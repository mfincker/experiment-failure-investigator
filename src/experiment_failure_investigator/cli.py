"""Command-line interface for the experiment failure investigator."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.metadata import version


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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line interface."""
    parser = build_parser()
    parser.parse_args(argv)
