"""Command-line interface for the experiment failure investigator."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from pydantic_ai.models import Model

from experiment_failure_investigator.agent.config import (
    AgentRuntimeConfig,
    build_ollama_model,
    load_runtime_config,
)
from experiment_failure_investigator.agent.controller import (
    InvestigationRun,
    run_investigation,
)
from experiment_failure_investigator.agent.trace import RunStatus
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
    run_parser = subparsers.add_parser(
        "run",
        help="Run one bounded investigation through Pydantic AI and Ollama.",
    )
    run_parser.add_argument("path", type=Path, help="Case directory to investigate.")
    run_parser.add_argument(
        "--output",
        type=Path,
        help="Run directory (default: runs/<opaque-case-id>).",
    )
    run_parser.add_argument(
        "--model",
        dest="model_tag",
        help="Installed Ollama model tag (default: EFI_MODEL_TAG or qwen2.5:7b).",
    )
    run_parser.add_argument(
        "--ollama-base-url",
        help=(
            "Local Ollama OpenAI-compatible URL "
            "(default: EFI_OLLAMA_BASE_URL or http://localhost:11434/v1)."
        ),
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


def _runtime_config_with_overrides(
    *,
    model_tag: str | None,
    ollama_base_url: str | None,
) -> AgentRuntimeConfig:
    """Load runtime settings and apply the two user-facing CLI overrides."""
    values = load_runtime_config().model_dump(mode="python")
    if model_tag is not None:
        values["model_tag"] = model_tag
    if ollama_base_url is not None:
        values["ollama_base_url"] = ollama_base_url
    return AgentRuntimeConfig.model_validate(values)


async def _run_single_investigation(
    case_path: Path,
    output: Path | None,
    *,
    config: AgentRuntimeConfig,
    model: Model | None = None,
) -> InvestigationRun:
    """Run one case through the existing controller with a safe output path."""
    case = load_investigator_case(case_path)
    run_directory = output or Path("runs") / case.case_id
    _reject_output_inside_input(case_path, run_directory)
    if run_directory.exists():
        raise FileExistsError(
            f"run output already exists: {run_directory}; choose a new directory"
        )
    selected_model = build_ollama_model(config) if model is None else model
    return await run_investigation(
        case_path,
        run_directory,
        model=selected_model,
        config=config,
    )


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
    elif args.command == "run":
        try:
            config = _runtime_config_with_overrides(
                model_tag=args.model_tag,
                ollama_base_url=args.ollama_base_url,
            )
            run = asyncio.run(
                _run_single_investigation(
                    args.path,
                    args.output,
                    config=config,
                )
            )
        except (FileExistsError, FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        if run.trace.status is RunStatus.FAILED:
            failure = run.trace.failure
            if failure is None:
                parser.error(
                    f"investigation failed without a failure record; trace: "
                    f"{run.artifacts.trace_json}"
                )
            parser.error(
                f"investigation failed ({failure.code.value}): "
                f"{failure.detail}; trace: {run.artifacts.trace_json}"
            )
        investigation_path = run.artifacts.investigation_json
        if investigation_path is None:
            parser.error(
                "investigation succeeded without writing its validated output; "
                f"trace: {run.artifacts.trace_json}"
            )
        print(
            "Wrote validated investigation to "
            f"{investigation_path} and trace to "
            f"{run.artifacts.trace_json}"
        )
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
