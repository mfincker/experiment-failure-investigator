"""Command-line interface for the experiment failure investigator."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from experiment_failure_investigator.benchmark.adapter import load_investigator_case
from experiment_failure_investigator.benchmark.registry import (
    generate_registered_cases,
    validate_registered_cases,
)
from experiment_failure_investigator.reporting import (
    BatchQcSummary,
    build_baseline_report,
    summarize_case_report,
    write_baseline_report,
    write_batch_summary,
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
    qc_parser = subparsers.add_parser(
        "qc",
        help="Run deterministic QC for one public benchmark case.",
    )
    qc_parser.add_argument("path", type=Path, help="Case directory to investigate.")
    qc_parser.add_argument(
        "--output",
        type=Path,
        help="Report directory (default: reports/<opaque-case-id>).",
    )
    qc_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing report directory.",
    )
    batch_parser = subparsers.add_parser(
        "qc-batch",
        help="Run deterministic QC for every case below a directory.",
    )
    batch_parser.add_argument("path", type=Path, help="Directory containing cases.")
    batch_parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports"),
        help="Batch report directory (default: reports).",
    )
    batch_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite reports in an existing batch output directory.",
    )
    return parser


def _require_output_available(output: Path, *, force: bool) -> None:
    if output.exists() and not output.is_dir():
        raise ValueError(f"output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(
            f"output directory is not empty: {output}; pass --force to overwrite reports"
        )


def _reject_output_inside_input(input_directory: Path, output: Path) -> None:
    input_root = input_directory.resolve()
    if output.resolve().is_relative_to(input_root):
        raise ValueError("report output must not be inside the input case directory")


def _run_single_qc(
    case_path: Path,
    output: Path | None,
    *,
    force: bool,
) -> tuple[Path, Path]:
    case = load_investigator_case(case_path)
    report_directory = output or Path("reports") / case.case_id
    _reject_output_inside_input(case_path, report_directory)
    _require_output_available(report_directory, force=force)
    report = build_baseline_report(case, report_directory)
    return write_baseline_report(report, report_directory)


def _case_directories(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        raise ValueError(f"case root is not a directory: {root}")
    cases = tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if path.is_dir() and (path / "manifest.json").is_file()
            ),
            key=lambda path: path.name,
        )
    )
    if not cases:
        raise ValueError(f"no benchmark case directories found in: {root}")
    return cases


def _run_batch_qc(
    case_root: Path,
    output: Path,
    *,
    force: bool,
) -> tuple[BatchQcSummary, Path, Path]:
    case_paths = _case_directories(case_root)
    _reject_output_inside_input(case_root, output)
    _require_output_available(output, force=force)
    loaded = [
        (case_path, load_investigator_case(case_path)) for case_path in case_paths
    ]
    summaries = []
    for case_path, case in loaded:
        report_directory = output / case.case_id
        report = build_baseline_report(case, report_directory)
        json_path, markdown_path = write_baseline_report(report, report_directory)
        summaries.append(
            summarize_case_report(
                report,
                case_path=case_path,
                report_json=json_path.relative_to(output),
                report_markdown=markdown_path.relative_to(output),
            )
        )
    summary = BatchQcSummary(cases=tuple(summaries))
    json_path, markdown_path = write_batch_summary(summary, output)
    return summary, json_path, markdown_path


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
    elif args.command == "qc":
        try:
            json_path, markdown_path = _run_single_qc(
                args.path,
                args.output,
                force=args.force,
            )
        except (FileExistsError, FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        print(f"Wrote deterministic QC report to {json_path} and {markdown_path}")
    elif args.command == "qc-batch":
        try:
            summary, json_path, markdown_path = _run_batch_qc(
                args.path,
                args.output,
                force=args.force,
            )
        except (FileExistsError, FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Wrote {len(summary.cases)} deterministic QC reports and batch summaries "
            f"to {json_path} and {markdown_path}"
        )
