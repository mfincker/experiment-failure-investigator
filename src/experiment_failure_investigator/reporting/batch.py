"""Compact, label-free summaries for deterministic batch QC runs."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator

from experiment_failure_investigator.analysis.results import ToolStatus, canonical_json
from experiment_failure_investigator.benchmark.models import StrictModel
from experiment_failure_investigator.reporting.baseline import BaselineReport

BATCH_SUMMARY_VERSION = "1.0.0"


class ToolStatusSummary(StrictModel):
    """Distinct statuses returned by one tool across its analysis scopes."""

    tool_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    statuses: tuple[ToolStatus, ...] = Field(min_length=1)

    @field_validator("statuses")
    @classmethod
    def canonicalize_statuses(
        cls, values: tuple[ToolStatus, ...]
    ) -> tuple[ToolStatus, ...]:
        return tuple(sorted(set(values), key=lambda status: status.value))


class BatchCaseSummary(StrictModel):
    """Operational pointers and aggregate counts for one public-data report."""

    case_path: str = Field(min_length=1)
    case_id: str = Field(pattern=r"^case_[0-9a-f]{16}$")
    applicable_tools: tuple[str, ...]
    tool_statuses: tuple[ToolStatusSummary, ...]
    warning_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    report_json: str = Field(min_length=1)
    report_markdown: str = Field(min_length=1)

    @field_validator("applicable_tools")
    @classmethod
    def canonicalize_tools(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(set(values)))

    @field_validator("tool_statuses")
    @classmethod
    def canonicalize_tool_statuses(
        cls, values: tuple[ToolStatusSummary, ...]
    ) -> tuple[ToolStatusSummary, ...]:
        names = [value.tool_name for value in values]
        if len(names) != len(set(names)):
            raise ValueError("tool status summaries must have unique tool names")
        return tuple(sorted(values, key=lambda value: value.tool_name))


class BatchQcSummary(StrictModel):
    """Deterministic summary of a set of investigator-safe QC reports."""

    summary_version: str = Field(
        default=BATCH_SUMMARY_VERSION,
        pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$",
    )
    cases: tuple[BatchCaseSummary, ...] = Field(min_length=1)

    @field_validator("cases")
    @classmethod
    def canonicalize_cases(
        cls, values: tuple[BatchCaseSummary, ...]
    ) -> tuple[BatchCaseSummary, ...]:
        ids = [value.case_id for value in values]
        if len(ids) != len(set(ids)):
            raise ValueError("batch summary case IDs must be unique")
        return tuple(sorted(values, key=lambda value: value.case_id))


def summarize_case_report(
    report: BaselineReport,
    *,
    case_path: Path,
    report_json: Path,
    report_markdown: Path,
) -> BatchCaseSummary:
    """Build a compact summary without consulting evaluation-only metadata."""
    statuses_by_tool: dict[str, set[ToolStatus]] = {}
    for result in report.tool_results:
        statuses_by_tool.setdefault(result.tool_name, set()).add(result.status)
    applicable = {
        result.tool_name
        for result in report.tool_results
        if result.status is not ToolStatus.NOT_APPLICABLE
    }
    return BatchCaseSummary(
        case_path=case_path.as_posix(),
        case_id=report.case_id,
        applicable_tools=tuple(applicable),
        tool_statuses=tuple(
            ToolStatusSummary(tool_name=name, statuses=tuple(statuses))
            for name, statuses in statuses_by_tool.items()
        ),
        warning_count=sum(len(result.warnings) for result in report.tool_results),
        evidence_count=len(report.evidence_index),
        report_json=report_json.as_posix(),
        report_markdown=report_markdown.as_posix(),
    )


def render_batch_markdown(summary: BatchQcSummary) -> str:
    """Render a compact operational index for human baseline review."""
    lines = [
        "# Deterministic QC batch summary",
        "",
        "| Case ID | Source path | Applicable tools | Statuses | Warnings | Evidence | Reports |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for case in summary.cases:
        statuses = ", ".join(
            f"{item.tool_name}: {'/'.join(status.value for status in item.statuses)}"
            for item in case.tool_statuses
        )
        reports = f"[JSON]({case.report_json}) / [Markdown]({case.report_markdown})"
        lines.append(
            f"| `{case.case_id}` | `{case.case_path}` | "
            f"{len(case.applicable_tools)} | {statuses} | {case.warning_count} | "
            f"{case.evidence_count} | {reports} |"
        )
    lines.extend(
        [
            "",
            "This index is generated only from public investigator inputs and report results.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_batch_summary(
    summary: BatchQcSummary,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Write canonical JSON and Markdown batch summaries."""
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "batch_summary.json"
    markdown_path = output_directory / "batch_summary.md"
    json_path.write_text(canonical_json(summary), encoding="utf-8")
    markdown_path.write_text(render_batch_markdown(summary), encoding="utf-8")
    return json_path, markdown_path
