"""Compact, read-only access to deterministic public assay evidence."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from typing import Annotated

from pydantic import Field, StringConstraints

from experiment_failure_investigator.agent.contracts import AgentStrictModel
from experiment_failure_investigator.analysis.contracts import (
    DesignCapabilities,
    InvestigatorCase,
)
from experiment_failure_investigator.analysis.results import (
    EvidenceRecord,
    ResultWarning,
    ScientificToolResult,
    ToolStatus,
)
from experiment_failure_investigator.benchmark.models import (
    AssayType,
    PlateFormat,
    SignalDirection,
    WellRole,
)
from experiment_failure_investigator.reporting.baseline import (
    BaselineFinding,
    BaselineReport,
)

BRIEFING_VERSION = "1.0.0"
DEFAULT_MAX_EVIDENCE_RECORDS = 25
HARD_MAX_EVIDENCE_RECORDS = 50
MetricPrefix = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$"),
]

# This explicit boundary must be reviewed whenever the deterministic layer grows.
ALLOWED_DIAGNOSTIC_TOOLS = frozenset(
    {
        "calculate_replicate_variability",
        "compare_batches",
        "detect_spatial_effects",
        "fit_dose_response",
        "generate_diagnostic_plot",
        "inspect_missingness",
        "summarize_controls",
    }
)


class PlateBriefing(AgentStrictModel):
    """Compact public context for one observed plate."""

    plate_id: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    operator_label: str | None = None
    run_date: date | None = None
    instrument_label: str | None = None
    plate_format: PlateFormat
    observed_well_count: int = Field(ge=0)
    expected_well_count: int = Field(ge=1)
    well_role_counts: dict[WellRole, int]
    missing_well_count: int = Field(ge=0)
    unusable_nonempty_measurement_count: int = Field(ge=0)
    empty_well_count: int = Field(ge=0)
    missing_design_annotation_count: int = Field(ge=0)
    geometry_inference_warning: str | None = None


class TreatmentBriefing(AgentStrictModel):
    """Observed public dose series for one treatment and unit."""

    treatment: str = Field(min_length=1)
    dose_unit: str | None = None
    doses: tuple[float, ...]


class DiagnosticResultCatalogEntry(AgentStrictModel):
    """Non-tabular description of evidence available from one tool."""

    tool_name: str
    tool_versions: tuple[str, ...]
    statuses: tuple[ToolStatus, ...]
    status_reasons: tuple[str, ...]
    invocation_count: int = Field(ge=1)
    evidence_count: int = Field(ge=0)
    plate_ids: tuple[str, ...]
    well_roles: tuple[WellRole, ...]
    treatments: tuple[str, ...]
    metric_prefixes: tuple[MetricPrefix, ...]
    warnings: tuple[ResultWarning, ...]
    limitations: tuple[str, ...]
    attachment_names: tuple[str, ...]


class DiagnosticResultList(AgentStrictModel):
    """Typed response for the diagnostic catalog tool."""

    case_id: str = Field(pattern=r"^case_[0-9a-f]{16}$")
    results: tuple[DiagnosticResultCatalogEntry, ...]


class AgentBriefing(AgentStrictModel):
    """Compact public case context suitable for a model prompt."""

    briefing_version: str = Field(
        default=BRIEFING_VERSION,
        pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$",
    )
    case_id: str = Field(pattern=r"^case_[0-9a-f]{16}$")
    assay_type: AssayType
    signal_direction: SignalDirection
    problem_statement: str = Field(min_length=1)
    protocol: str = Field(min_length=1)
    plates: tuple[PlateBriefing, ...]
    treatments: tuple[TreatmentBriefing, ...]
    capabilities: DesignCapabilities
    findings: tuple[BaselineFinding, ...]
    diagnostic_results: tuple[DiagnosticResultCatalogEntry, ...]
    limitations: tuple[str, ...]


class EvidenceFilters(AgentStrictModel):
    """Filters applied to an evidence page."""

    metric_prefix: MetricPrefix | None = None
    plate_id: str | None = None
    treatment: str | None = None


class EvidencePage(AgentStrictModel):
    """Bounded, deterministic evidence returned for one diagnostic tool."""

    case_id: str = Field(pattern=r"^case_[0-9a-f]{16}$")
    tool_name: str
    filters: EvidenceFilters
    total_tool_evidence_count: int = Field(ge=0)
    matched_count: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=HARD_MAX_EVIDENCE_RECORDS)
    returned_count: int = Field(ge=0)
    has_more: bool
    evidence: tuple[EvidenceRecord, ...]
    statuses: tuple[ToolStatus, ...]
    status_reasons: tuple[str, ...]
    warnings: tuple[ResultWarning, ...]
    limitations: tuple[str, ...]


class ResolvedEvidence(AgentStrictModel):
    """Exact, bounded records resolved from model-provided evidence IDs."""

    case_id: str = Field(pattern=r"^case_[0-9a-f]{16}$")
    evidence: tuple[EvidenceRecord, ...]


def _ensure_public_pair(case: InvestigatorCase, baseline: BaselineReport) -> None:
    """Verify that the public case and baseline describe the same analysis."""
    if case.case_id != baseline.case_id:
        raise ValueError("investigator case ID does not match the baseline")
    if case.design.capabilities != baseline.capabilities:
        raise ValueError("investigator capabilities do not match the baseline")


def _selected_results(
    baseline: BaselineReport,
    tool_name: str,
) -> tuple[ScientificToolResult, ...]:
    """Return all baseline results for one allowlisted diagnostic tool."""
    if tool_name not in ALLOWED_DIAGNOSTIC_TOOLS:
        raise ValueError(f"diagnostic tool is not allowlisted: {tool_name!r}")
    selected = tuple(
        result for result in baseline.tool_results if result.tool_name == tool_name
    )
    if not selected:
        raise ValueError(
            f"diagnostic tool is unavailable in this baseline: {tool_name!r}"
        )
    return selected


def _metric_prefixes(records: tuple[EvidenceRecord, ...]) -> tuple[str, ...]:
    """Derive the canonical searchable prefixes from evidence metric names."""
    prefixes: set[str] = set()
    for record in records:
        parts = record.metric_name.split(".")
        prefixes.update(".".join(parts[:end]) for end in range(1, len(parts) + 1))
    return tuple(sorted(prefixes))


def _unique_warnings(
    results: tuple[ScientificToolResult, ...],
) -> tuple[ResultWarning, ...]:
    """Deduplicate and canonically order warnings from several tool results."""
    warnings = {
        (warning.code, warning.message, warning.evidence_ids): warning
        for result in results
        for warning in result.warnings
    }
    return tuple(warnings[key] for key in sorted(warnings))


def _catalog_entry(
    tool_name: str,
    results: tuple[ScientificToolResult, ...],
) -> DiagnosticResultCatalogEntry:
    """Summarize one diagnostic family without including evidence values."""
    evidence = tuple(record for result in results for record in result.evidence)
    scopes = tuple(result.scope for result in results) + tuple(
        record.scope for record in evidence
    )
    return DiagnosticResultCatalogEntry(
        tool_name=tool_name,
        tool_versions=tuple(sorted({result.tool_version for result in results})),
        statuses=tuple(sorted({result.status for result in results}, key=str)),
        status_reasons=tuple(
            sorted(
                {
                    result.status_reason
                    for result in results
                    if result.status_reason is not None
                }
            )
        ),
        invocation_count=len(results),
        evidence_count=len(evidence),
        plate_ids=tuple(sorted({item for scope in scopes for item in scope.plate_ids})),
        well_roles=tuple(
            sorted(
                {item for scope in scopes for item in scope.well_roles},
                key=lambda role: role.value,
            )
        ),
        treatments=tuple(
            sorted({item for scope in scopes for item in scope.treatments})
        ),
        metric_prefixes=_metric_prefixes(evidence),
        warnings=_unique_warnings(results),
        limitations=tuple(
            sorted({item for result in results for item in result.limitations})
        ),
        attachment_names=tuple(
            sorted(
                {
                    attachment.name
                    for result in results
                    for attachment in result.attachments
                }
            )
        ),
    )


def list_diagnostic_results(baseline: BaselineReport) -> DiagnosticResultList:
    """List only allowlisted diagnostic result families in canonical order."""
    present = {result.tool_name for result in baseline.tool_results}
    unexpected = sorted(present - ALLOWED_DIAGNOSTIC_TOOLS)
    if unexpected:
        raise ValueError(
            f"baseline contains non-allowlisted diagnostic tools: {unexpected}"
        )
    grouped: dict[str, list[ScientificToolResult]] = defaultdict(list)
    for result in baseline.tool_results:
        grouped[result.tool_name].append(result)
    entries = tuple(
        _catalog_entry(tool_name, tuple(grouped[tool_name]))
        for tool_name in sorted(grouped)
    )
    return DiagnosticResultList(case_id=baseline.case_id, results=entries)


def build_agent_briefing(
    case: InvestigatorCase,
    baseline: BaselineReport,
) -> AgentBriefing:
    """Project public inputs and deterministic summaries into compact context."""
    _ensure_public_pair(case, baseline)
    plate_metadata = {plate.plate_id: plate for plate in case.metadata.plates}
    missing_metadata = sorted(
        {plate.plate_id for plate in case.design.plates} - set(plate_metadata)
    )
    if missing_metadata:
        raise ValueError(f"design plates lack public metadata: {missing_metadata}")
    plates = tuple(
        PlateBriefing(
            plate_id=plate.plate_id,
            batch_id=plate_metadata[plate.plate_id].batch_id,
            operator_label=plate_metadata[plate.plate_id].operator_label,
            run_date=plate_metadata[plate.plate_id].run_date,
            instrument_label=plate_metadata[plate.plate_id].instrument_label,
            plate_format=plate.plate_format,
            observed_well_count=plate.observed_well_count,
            expected_well_count=plate.expected_well_count,
            well_role_counts=plate.role_counts,
            missing_well_count=len(plate.missing_wells),
            unusable_nonempty_measurement_count=(
                plate.missing_measurement_count
                + plate.null_measurement_count
                + plate.nonfinite_measurement_count
            ),
            empty_well_count=plate.empty_well_count,
            missing_design_annotation_count=plate.missing_design_annotation_count,
            geometry_inference_warning=plate.geometry_inference_warning,
        )
        for plate in sorted(case.design.plates, key=lambda item: item.plate_id)
    )
    treatments = tuple(
        TreatmentBriefing(
            treatment=series.treatment,
            dose_unit=series.dose_unit,
            doses=series.doses,
        )
        for series in sorted(
            case.design.treatment_dose_series,
            key=lambda item: (item.treatment, item.dose_unit or ""),
        )
    )
    catalog = list_diagnostic_results(baseline)
    return AgentBriefing(
        case_id=case.case_id,
        assay_type=case.metadata.assay_type,
        signal_direction=case.metadata.signal_direction,
        problem_statement=case.problem_statement,
        protocol=case.protocol,
        plates=plates,
        treatments=treatments,
        capabilities=case.design.capabilities,
        findings=tuple(
            sorted(
                baseline.findings,
                key=lambda item: (item.category.value, item.finding_id),
            )
        ),
        diagnostic_results=catalog.results,
        limitations=tuple(sorted(set(baseline.limitations))),
    )


def _validate_maximum(value: int, *, name: str) -> None:
    """Validate a configurable bound against the application hard limit."""
    if value < 1 or value > HARD_MAX_EVIDENCE_RECORDS:
        raise ValueError(f"{name} must be between one and {HARD_MAX_EVIDENCE_RECORDS}")


def _filter_value(value: str | None, *, name: str) -> str | None:
    """Trim an optional text filter and reject blank values."""
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be blank")
    return normalized


def inspect_diagnostic_result(
    baseline: BaselineReport,
    tool_name: str,
    *,
    metric_prefix: str | None = None,
    plate_id: str | None = None,
    treatment: str | None = None,
    offset: int = 0,
    limit: int = DEFAULT_MAX_EVIDENCE_RECORDS,
    maximum_page_size: int = DEFAULT_MAX_EVIDENCE_RECORDS,
) -> EvidencePage:
    """Return one filtered page from an allowlisted frozen diagnostic result."""
    _validate_maximum(maximum_page_size, name="maximum page size")
    if offset < 0:
        raise ValueError("offset must not be negative")
    if limit < 1 or limit > maximum_page_size:
        raise ValueError(f"limit must be between one and {maximum_page_size}")
    selected = _selected_results(baseline, tool_name)
    catalog = _catalog_entry(tool_name, selected)
    normalized_prefix = _filter_value(metric_prefix, name="metric prefix")
    normalized_plate = _filter_value(plate_id, name="plate ID")
    normalized_treatment = _filter_value(treatment, name="treatment")
    if normalized_prefix is not None:
        if re.fullmatch(
            r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*", normalized_prefix
        ) is None:
            raise ValueError("metric prefix has an invalid format")
        if normalized_prefix not in catalog.metric_prefixes:
            raise ValueError(f"metric prefix is unavailable for {tool_name!r}")
    if normalized_plate is not None and normalized_plate not in catalog.plate_ids:
        raise ValueError(f"plate scope is unavailable for {tool_name!r}")
    if (
        normalized_treatment is not None
        and normalized_treatment not in catalog.treatments
    ):
        raise ValueError(f"treatment scope is unavailable for {tool_name!r}")

    records = tuple(
        sorted(
            (record for result in selected for record in result.evidence),
            key=lambda record: record.evidence_id,
        )
    )
    matches = tuple(
        record
        for record in records
        if (
            normalized_prefix is None
            or record.metric_name == normalized_prefix
            or record.metric_name.startswith(normalized_prefix + ".")
        )
        and (normalized_plate is None or normalized_plate in record.scope.plate_ids)
        and (
            normalized_treatment is None
            or normalized_treatment in record.scope.treatments
        )
    )
    page = matches[offset : offset + limit]
    return EvidencePage(
        case_id=baseline.case_id,
        tool_name=tool_name,
        filters=EvidenceFilters(
            metric_prefix=normalized_prefix,
            plate_id=normalized_plate,
            treatment=normalized_treatment,
        ),
        total_tool_evidence_count=len(records),
        matched_count=len(matches),
        offset=offset,
        limit=limit,
        returned_count=len(page),
        has_more=offset + len(page) < len(matches),
        evidence=page,
        statuses=catalog.statuses,
        status_reasons=catalog.status_reasons,
        warnings=catalog.warnings,
        limitations=catalog.limitations,
    )


def resolve_evidence(
    baseline: BaselineReport,
    evidence_ids: tuple[str, ...],
    *,
    maximum_ids: int = 10,
) -> ResolvedEvidence:
    """Resolve a small exact set of evidence IDs without exposing the full index."""
    _validate_maximum(maximum_ids, name="maximum evidence ID count")
    if not evidence_ids:
        raise ValueError("at least one evidence ID is required")
    if len(evidence_ids) > maximum_ids:
        raise ValueError(f"at most {maximum_ids} evidence IDs may be resolved")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("evidence IDs must be unique")
    invalid = sorted(
        item for item in evidence_ids if re.fullmatch(r"ev_[0-9a-f]{20}", item) is None
    )
    if invalid:
        raise ValueError(f"invalid evidence IDs: {invalid}")
    by_id = {record.evidence_id: record for record in baseline.evidence_index}
    unknown = sorted(set(evidence_ids) - set(by_id))
    if unknown:
        raise ValueError(f"unknown evidence IDs: {unknown}")
    return ResolvedEvidence(
        case_id=baseline.case_id,
        evidence=tuple(by_id[item] for item in sorted(evidence_ids)),
    )
