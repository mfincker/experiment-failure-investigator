"""Deterministic baseline orchestration, heuristics, and report rendering."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import Field, StringConstraints, field_validator, model_validator

from experiment_failure_investigator.analysis.batches import compare_batches
from experiment_failure_investigator.analysis.contracts import (
    DesignCapabilities,
    InvestigatorCase,
)
from experiment_failure_investigator.analysis.controls import summarize_controls
from experiment_failure_investigator.analysis.dose_response import fit_dose_response
from experiment_failure_investigator.analysis.heatmaps import generate_diagnostic_plot
from experiment_failure_investigator.analysis.missingness import inspect_missingness
from experiment_failure_investigator.analysis.replicates import (
    calculate_replicate_variability,
)
from experiment_failure_investigator.analysis.results import (
    EvidenceId,
    EvidenceRecord,
    ScientificToolResult,
    canonical_json,
)
from experiment_failure_investigator.analysis.spatial import detect_spatial_effects
from experiment_failure_investigator.benchmark.models import StrictModel

REPORT_VERSION = "1.0.0"
HEURISTIC_VERSION = "1.0.0"
FindingId = Annotated[str, StringConstraints(pattern=r"^fd_[0-9a-f]{20}$")]


class FindingSeverity(StrEnum):
    """Kept as constants so report JSON remains simple and stable."""

    REVIEW = "review"


class FindingCategory(StrEnum):
    """Stable diagnostic families used by the synthetic baseline."""

    DATA_COMPLETENESS = "data_completeness"
    CONTROL_SEPARATION = "control_separation"
    LAYOUT_IDENTIFIABILITY = "layout_identifiability"
    EDGE_ASSOCIATION = "edge_association"
    POSITION_TREND = "position_trend"
    LOCALIZED_PATTERN = "localized_pattern"
    PLATE_RESPONSE = "plate_response"
    TREATMENT_RESPONSE = "treatment_response"


class HeuristicConfiguration(StrictModel):
    """Versioned thresholds used only for the frozen synthetic benchmark."""

    version: str = Field(default=HEURISTIC_VERSION, pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    weak_control_z_prime_maximum: float = Field(default=0.0, allow_inf_nan=False)
    edge_standardized_difference_minimum: float = Field(
        default=0.75, gt=0, allow_inf_nan=False
    )
    position_correlation_minimum: float = Field(
        default=0.45, gt=0, le=1, allow_inf_nan=False
    )
    localized_run_minimum: int = Field(default=2, ge=1)
    plate_ratio_lower: float = Field(default=0.85, gt=0, allow_inf_nan=False)
    plate_ratio_upper: float = Field(default=1.15, gt=0, allow_inf_nan=False)
    relative_treatment_range_maximum: float = Field(
        default=0.5, gt=0, lt=1, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def validate_ratio_interval(self) -> HeuristicConfiguration:
        if self.plate_ratio_lower >= self.plate_ratio_upper:
            raise ValueError("plate ratio bounds must be increasing")
        return self


class BaselineFinding(StrictModel):
    """Neutral interpretation whose claims are linked to numeric evidence."""

    finding_id: FindingId
    category: FindingCategory
    severity: FindingSeverity = FindingSeverity.REVIEW
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    evidence_ids: tuple[EvidenceId, ...] = Field(min_length=1)

    @field_validator("title", "summary")
    @classmethod
    def reject_uncited_numeric_narrative(cls, value: str) -> str:
        if re.search(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?", value):
            raise ValueError(
                "finding narrative must not embed numbers; cite evidence records"
            )
        return value.strip()

    @field_validator("evidence_ids")
    @classmethod
    def canonicalize_evidence_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("finding evidence IDs must be unique")
        return tuple(sorted(values))


class BaselineReport(StrictModel):
    """Ground-truth-free deterministic QC report."""

    report_version: str = Field(
        default=REPORT_VERSION, pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$"
    )
    heuristic_configuration: HeuristicConfiguration
    case_id: str = Field(pattern=r"^case_[0-9a-f]{16}$")
    capabilities: DesignCapabilities
    tool_results: tuple[ScientificToolResult, ...]
    findings: tuple[BaselineFinding, ...]
    evidence_index: tuple[EvidenceRecord, ...]
    limitations: tuple[str, ...]
    suggested_followups: tuple[str, ...]

    @model_validator(mode="after")
    def validate_evidence_graph(self) -> BaselineReport:
        result_records = tuple(
            record for result in self.tool_results for record in result.evidence
        )
        result_by_id = {record.evidence_id: record for record in result_records}
        index_by_id = {record.evidence_id: record for record in self.evidence_index}
        if len(result_by_id) != len(result_records):
            raise ValueError("tool results contain duplicate evidence IDs")
        if index_by_id != result_by_id:
            raise ValueError("evidence index must exactly match tool-result evidence")
        for finding in self.findings:
            if not set(finding.evidence_ids) <= set(index_by_id):
                raise ValueError("finding references evidence outside the report")
        return self

    @field_validator("findings")
    @classmethod
    def canonicalize_findings(
        cls, values: tuple[BaselineFinding, ...]
    ) -> tuple[BaselineFinding, ...]:
        ids = [finding.finding_id for finding in values]
        if len(ids) != len(set(ids)):
            raise ValueError("finding IDs must be unique")
        return tuple(
            sorted(values, key=lambda finding: (finding.category.value, finding.finding_id))
        )

    @field_validator("evidence_index")
    @classmethod
    def canonicalize_evidence(
        cls, values: tuple[EvidenceRecord, ...]
    ) -> tuple[EvidenceRecord, ...]:
        return tuple(sorted(values, key=lambda record: record.evidence_id))

    @field_validator("limitations", "suggested_followups")
    @classmethod
    def canonicalize_text(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("report text entries must not be blank")
        return tuple(sorted(set(values)))


def _finding(
    category: FindingCategory,
    title: str,
    summary: str,
    records: list[EvidenceRecord],
    config: HeuristicConfiguration,
) -> BaselineFinding:
    evidence_ids = tuple(sorted(record.evidence_id for record in records))
    identity = canonical_json(
        {
            "category": category,
            "evidence_ids": list(evidence_ids),
            "heuristic_configuration": config.model_dump(mode="json"),
        }
    )
    finding_id = "fd_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return BaselineFinding(
        finding_id=finding_id,
        category=category,
        title=title,
        summary=summary,
        evidence_ids=evidence_ids,
    )


def _numeric(record: EvidenceRecord) -> float | None:
    if isinstance(record.value, bool) or not isinstance(record.value, int | float):
        return None
    return float(record.value)


def _records_by_metric(
    results: tuple[ScientificToolResult, ...],
) -> dict[str, list[EvidenceRecord]]:
    grouped: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for result in results:
        for record in result.evidence:
            grouped[record.metric_name].append(record)
    return grouped


def _derive_findings(
    results: tuple[ScientificToolResult, ...],
    config: HeuristicConfiguration,
) -> tuple[BaselineFinding, ...]:
    metrics = _records_by_metric(results)
    findings: list[BaselineFinding] = []

    incomplete = [
        record
        for name in (
            "missingness.missing_measurement_count",
            "missingness.null_measurement_count",
            "missingness.nonfinite_measurement_count",
            "missingness.missing_design_annotation_count",
        )
        for record in metrics[name]
        if (_numeric(record) or 0.0) > 0
    ]
    if incomplete:
        findings.append(
            _finding(
                FindingCategory.DATA_COMPLETENESS,
                "Incomplete assay inputs",
                "Some non-empty wells lack usable measurements or design annotations.",
                incomplete,
                config,
            )
        )

    for record in metrics["controls.z_prime"]:
        value = _numeric(record)
        if value is not None and value < config.weak_control_z_prime_maximum:
            findings.append(
                _finding(
                    FindingCategory.CONTROL_SEPARATION,
                    "Limited control separation",
                    "Control distributions show limited separation under the synthetic benchmark heuristic.",
                    [record],
                    config,
                )
            )

    for result in results:
        matching_warnings = [
            warning
            for warning in result.warnings
            if warning.code == "condition_position_nonidentifiability"
        ]
        for warning in matching_warnings:
            records = [
                record
                for record in result.evidence
                if record.evidence_id in warning.evidence_ids
            ]
            findings.append(
                _finding(
                    FindingCategory.LAYOUT_IDENTIFIABILITY,
                    "Condition and position are confounded",
                    "The observed layout cannot separate some condition effects from position effects.",
                    records,
                    config,
                )
            )

    for record in metrics["spatial.edge_standardized_mean_difference"]:
        value = _numeric(record)
        if value is not None and abs(value) >= config.edge_standardized_difference_minimum:
            findings.append(
                _finding(
                    FindingCategory.EDGE_ASSOCIATION,
                    "Residuals differ between edge and interior wells",
                    "Condition-centered residuals are associated with edge versus interior position.",
                    [record],
                    config,
                )
            )

    trend_records = [
        record
        for name in ("spatial.row_correlation", "spatial.column_correlation")
        for record in metrics[name]
        if (_numeric(record) is not None)
        and abs(_numeric(record) or 0.0) >= config.position_correlation_minimum
    ]
    for record in trend_records:
        findings.append(
            _finding(
                FindingCategory.POSITION_TREND,
                "Residuals vary monotonically across a plate axis",
                "Condition-centered residuals have a notable positional association without implying dispense order.",
                [record],
                config,
            )
        )

    for record in metrics["spatial.largest_adjacent_extreme_run_size"]:
        value = _numeric(record)
        if value is not None and value >= config.localized_run_minimum:
            companion = [
                candidate
                for candidate in metrics[
                    "spatial.largest_adjacent_extreme_run_well_ids"
                ]
                if candidate.scope.plate_ids == record.scope.plate_ids
            ]
            findings.append(
                _finding(
                    FindingCategory.LOCALIZED_PATTERN,
                    "Adjacent extreme residuals are localized",
                    "A spatially adjacent residual pattern merits review without identifying a physical cause.",
                    [record, *companion],
                    config,
                )
            )

    plate_ratio_records: dict[tuple[str, ...], list[EvidenceRecord]] = defaultdict(list)
    for metric_name in (
        "batches.control_window_ratio",
        "batches.response_range_ratio",
    ):
        for record in metrics[metric_name]:
            value = _numeric(record)
            if value is not None and not (
                config.plate_ratio_lower <= value <= config.plate_ratio_upper
            ):
                plate_ratio_records[record.scope.plate_ids].append(record)
    for records in plate_ratio_records.values():
        findings.append(
            _finding(
                FindingCategory.PLATE_RESPONSE,
                "Matched plate response ranges differ",
                "Like-for-like conditions show a plate-level response-range difference.",
                records,
                config,
            )
        )

    ranges_by_plate: dict[tuple[str, ...], list[EvidenceRecord]] = defaultdict(list)
    for record in metrics["dose_response.observed_response_range"]:
        if record.scope.treatments:
            ranges_by_plate[record.scope.plate_ids].append(record)
    for records in ranges_by_plate.values():
        if len(records) < 2:
            continue
        maximum_record = max(records, key=lambda record: _numeric(record) or 0.0)
        maximum = _numeric(maximum_record) or 0.0
        if maximum <= 0:
            continue
        for record in records:
            value = _numeric(record) or 0.0
            if value / maximum < config.relative_treatment_range_maximum:
                findings.append(
                    _finding(
                        FindingCategory.TREATMENT_RESPONSE,
                        "Treatment response spans differ",
                        "One treatment has substantially less observed response variation than another on the same plate.",
                        [record, maximum_record],
                        config,
                    )
                )
    return tuple(findings)


def _run_tools(
    case: InvestigatorCase,
    artifact_directory: Path,
) -> tuple[ScientificToolResult, ...]:
    """Run tools in a documented order without model-driven selection."""
    return (
        inspect_missingness(case),
        *summarize_controls(case),
        *calculate_replicate_variability(case),
        *fit_dose_response(case),
        *detect_spatial_effects(case),
        *compare_batches(case),
        generate_diagnostic_plot(
            case,
            artifact_directory,
            kind="raw_signal",
            filename="raw_signal.png",
        ),
        generate_diagnostic_plot(
            case,
            artifact_directory,
            kind="condition_residual",
            filename="condition_residual.png",
        ),
        generate_diagnostic_plot(
            case,
            artifact_directory,
            kind="control_distribution",
            filename="control_distribution.png",
        ),
        generate_diagnostic_plot(
            case,
            artifact_directory,
            kind="dose_response",
            filename="dose_response.png",
        ),
    )


def build_baseline_report(
    case: InvestigatorCase,
    artifact_directory: Path,
    *,
    heuristic_configuration: HeuristicConfiguration | None = None,
) -> BaselineReport:
    """Build one deterministic report from public data and tool evidence."""
    config = heuristic_configuration or HeuristicConfiguration()
    results = _run_tools(case, artifact_directory)
    evidence = tuple(record for result in results for record in result.evidence)
    limitations = tuple(
        {
            *case.design.limitations,
            *(limitation for result in results for limitation in result.limitations),
            "Heuristic thresholds are calibrated only for the synthetic benchmark and are not universal laboratory QC criteria.",
        }
    )
    findings = _derive_findings(results, config)
    categories = {finding.category for finding in findings}
    followups = {
        FindingCategory.DATA_COMPLETENESS: "Resolve missing measurements and design annotations before interpretation.",
        FindingCategory.CONTROL_SEPARATION: "Review control identities, preparation, and assay-specific acceptance criteria.",
        FindingCategory.LAYOUT_IDENTIFIABILITY: "Repeat with conditions distributed across spatial regions.",
        FindingCategory.EDGE_ASSOCIATION: "Review plate handling and repeat with edge-balanced conditions.",
        FindingCategory.POSITION_TREND: "Review time, temperature, and available liquid-handling records.",
        FindingCategory.LOCALIZED_PATTERN: "Inspect neighboring wells and available liquid-handling records.",
        FindingCategory.PLATE_RESPONSE: "Review plate-level metadata and compare matched conditions in a repeat run.",
        FindingCategory.TREATMENT_RESPONSE: "Verify treatment preparation and dose annotations before interpreting biology.",
    }
    return BaselineReport(
        heuristic_configuration=config,
        case_id=case.case_id,
        capabilities=case.design.capabilities,
        tool_results=results,
        findings=findings,
        evidence_index=evidence,
        limitations=limitations,
        suggested_followups=tuple(
            followups[category] for category in sorted(categories)
        ),
    )


def _format_value(record: EvidenceRecord) -> str:
    value = json.dumps(record.value, ensure_ascii=False, sort_keys=True)
    return value if record.unit is None else f"{value} {record.unit}"


def render_baseline_markdown(report: BaselineReport) -> str:
    """Render a concise report whose numeric values live in the evidence index."""
    lines = [
        "# Deterministic assay QC report",
        "",
        f"Case: `{report.case_id}`",
        "",
        "## Findings",
        "",
    ]
    if report.findings:
        for finding in report.findings:
            citations = ", ".join(f"`{item}`" for item in finding.evidence_ids)
            lines.extend(
                [
                    f"### {finding.title}",
                    "",
                    finding.summary,
                    "",
                    f"Evidence: {citations}",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "No configured synthetic-benchmark heuristic produced a finding.",
                "",
            ]
        )
    lines.extend(["## Tool results", "", "| Tool | Status |", "|---|---|"])
    tool_statuses = sorted(
        {(result.tool_name, result.status.value) for result in report.tool_results}
    )
    lines.extend(
        f"| `{tool_name}` | `{status}` |" for tool_name, status in tool_statuses
    )
    lines.extend(["", "## Evidence index", ""])
    warnings = [
        warning for result in report.tool_results for warning in result.warnings
    ]
    cited_ids = {
        evidence_id
        for finding in report.findings
        for evidence_id in finding.evidence_ids
    } | {
        evidence_id for warning in warnings for evidence_id in warning.evidence_ids
    }
    cited_records = [
        record for record in report.evidence_index if record.evidence_id in cited_ids
    ]
    if cited_records:
        for record in cited_records:
            lines.append(
                f"- `{record.evidence_id}` — `{record.metric_name}`: "
                f"{_format_value(record)}"
            )
    else:
        lines.append("No evidence records are cited by findings or warnings.")
    lines.extend(
        [
            "",
            "The canonical JSON report contains the complete evidence index and tool results.",
        ]
    )
    attachments = [
        attachment
        for result in report.tool_results
        for attachment in result.attachments
    ]
    if attachments:
        lines.extend(["", "## Plot attachments", ""])
        lines.extend(
            f"- `{attachment.name}`: `{attachment.path}` "
            f"(SHA-256 `{attachment.sha256}`)"
            for attachment in attachments
        )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            citations = ", ".join(f"`{item}`" for item in warning.evidence_ids)
            suffix = "" if not citations else f" Evidence: {citations}."
            lines.append(f"- `{warning.code}`: {warning.message}{suffix}")
    if report.suggested_followups:
        lines.extend(["", "## Suggested follow-ups", ""])
        lines.extend(f"- {item}" for item in report.suggested_followups)
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.limitations)
    return "\n".join(lines) + "\n"


def write_baseline_report(
    report: BaselineReport,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Write canonical JSON and Markdown representations of one report."""
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "report.json"
    markdown_path = output_directory / "report.md"
    json_path.write_text(canonical_json(report), encoding="utf-8")
    markdown_path.write_text(render_baseline_markdown(report), encoding="utf-8")
    return json_path, markdown_path
