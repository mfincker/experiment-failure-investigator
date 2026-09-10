"""Compact, read-only access to deterministic public assay evidence."""

from __future__ import annotations

import re
from collections import defaultdict

from pydantic import Field, JsonValue

from experiment_failure_investigator.agent.contracts import AgentStrictModel
from experiment_failure_investigator.analysis.contracts import InvestigatorCase
from experiment_failure_investigator.analysis.results import (
    EvidenceRecord,
    ResultWarning,
    ScientificToolResult,
    ToolStatus,
)
from experiment_failure_investigator.reporting.baseline import BaselineReport

BRIEFING_VERSION = "1.0.0"
DEFAULT_MAX_EVIDENCE_RECORDS = 25
HARD_MAX_EVIDENCE_RECORDS = 50
JsonObject = dict[str, JsonValue]

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


class EvidencePage(AgentStrictModel):
    """Bounded, deterministic evidence returned for one diagnostic tool."""

    case_id: str = Field(pattern=r"^case_[0-9a-f]{16}$")
    tool_name: str
    filters: dict[str, str | None]
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
) -> JsonObject:
    """Summarize one diagnostic family without including evidence values."""
    evidence = tuple(record for result in results for record in result.evidence)
    scopes = tuple(result.scope for result in results) + tuple(
        record.scope for record in evidence
    )
    return {
        "tool_name": tool_name,
        "tool_versions": list(
            sorted({result.tool_version for result in results})
        ),
        "statuses": [
            status.value
            for status in sorted({result.status for result in results}, key=str)
        ],
        "status_reasons": list(
            sorted(
                {
                    result.status_reason
                    for result in results
                    if result.status_reason is not None
                }
            )
        ),
        "invocation_count": len(results),
        "evidence_count": len(evidence),
        "plate_ids": list(
            sorted({item for scope in scopes for item in scope.plate_ids})
        ),
        "well_roles": [
            role.value
            for role in sorted(
                {item for scope in scopes for item in scope.well_roles},
                key=lambda role: role.value,
            )
        ],
        "treatments": list(
            sorted({item for scope in scopes for item in scope.treatments})
        ),
        "metric_prefixes": list(_metric_prefixes(evidence)),
        "warnings": [
            warning.model_dump(mode="json") for warning in _unique_warnings(results)
        ],
        "limitations": list(
            sorted({item for result in results for item in result.limitations})
        ),
        "attachment_names": list(
            sorted(
                {
                    attachment.name
                    for result in results
                    for attachment in result.attachments
                }
            )
        ),
    }


def _catalog_entries(baseline: BaselineReport) -> list[JsonObject]:
    """Build the canonical model-visible catalog from a validated baseline."""
    present = {result.tool_name for result in baseline.tool_results}
    unexpected = sorted(present - ALLOWED_DIAGNOSTIC_TOOLS)
    if unexpected:
        raise ValueError(
            f"baseline contains non-allowlisted diagnostic tools: {unexpected}"
        )
    grouped: dict[str, list[ScientificToolResult]] = defaultdict(list)
    for result in baseline.tool_results:
        grouped[result.tool_name].append(result)
    return [
        _catalog_entry(tool_name, tuple(grouped[tool_name]))
        for tool_name in sorted(grouped)
    ]


def list_diagnostic_results(baseline: BaselineReport) -> JsonObject:
    """List only allowlisted diagnostic result families in canonical order."""
    return {
        "case_id": baseline.case_id,
        "results": _catalog_entries(baseline),
    }


def build_agent_briefing(
    case: InvestigatorCase,
    baseline: BaselineReport,
) -> JsonObject:
    """Project public inputs and deterministic summaries into compact context."""
    _ensure_public_pair(case, baseline)
    plate_metadata = {plate.plate_id: plate for plate in case.metadata.plates}
    missing_metadata = sorted(
        {plate.plate_id for plate in case.design.plates} - set(plate_metadata)
    )
    if missing_metadata:
        raise ValueError(f"design plates lack public metadata: {missing_metadata}")
    plates: list[JsonValue] = [
        {
            "plate_id": plate.plate_id,
            "batch_id": plate_metadata[plate.plate_id].batch_id,
            "operator_label": plate_metadata[plate.plate_id].operator_label,
            "run_date": (
                plate_metadata[plate.plate_id].run_date.isoformat()
                if plate_metadata[plate.plate_id].run_date is not None
                else None
            ),
            "instrument_label": plate_metadata[plate.plate_id].instrument_label,
            "plate_format": plate.plate_format.value,
            "observed_well_count": plate.observed_well_count,
            "expected_well_count": plate.expected_well_count,
            "well_role_counts": {
                role.value: count for role, count in plate.role_counts.items()
            },
            "missing_well_count": len(plate.missing_wells),
            "unusable_nonempty_measurement_count": (
                plate.missing_measurement_count
                + plate.null_measurement_count
                + plate.nonfinite_measurement_count
            ),
            "empty_well_count": plate.empty_well_count,
            "missing_design_annotation_count": plate.missing_design_annotation_count,
            "geometry_inference_warning": plate.geometry_inference_warning,
        }
        for plate in sorted(case.design.plates, key=lambda item: item.plate_id)
    ]
    treatments: list[JsonValue] = [
        {
            "treatment": series.treatment,
            "dose_unit": series.dose_unit,
            "doses": list(series.doses),
        }
        for series in sorted(
            case.design.treatment_dose_series,
            key=lambda item: (item.treatment, item.dose_unit or ""),
        )
    ]
    return {
        "briefing_version": BRIEFING_VERSION,
        "case_id": case.case_id,
        "assay_type": case.metadata.assay_type.value,
        "signal_direction": case.metadata.signal_direction.value,
        "problem_statement": case.problem_statement,
        "protocol": case.protocol,
        "plates": plates,
        "treatments": treatments,
        "capabilities": case.design.capabilities.model_dump(mode="json"),
        "findings": [
            finding.model_dump(mode="json")
            for finding in sorted(
                baseline.findings,
                key=lambda item: (item.category.value, item.finding_id),
            )
        ],
        "diagnostic_results": _catalog_entries(baseline),
        "limitations": list(sorted(set(baseline.limitations))),
    }


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
    records = tuple(
        sorted(
            (record for result in selected for record in result.evidence),
            key=lambda record: record.evidence_id,
        )
    )
    scopes = tuple(result.scope for result in selected) + tuple(
        record.scope for record in records
    )
    available_plate_ids = {item for scope in scopes for item in scope.plate_ids}
    available_treatments = {
        item for scope in scopes for item in scope.treatments
    }
    normalized_prefix = _filter_value(metric_prefix, name="metric prefix")
    normalized_plate = _filter_value(plate_id, name="plate ID")
    normalized_treatment = _filter_value(treatment, name="treatment")
    if normalized_prefix is not None:
        if re.fullmatch(
            r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*", normalized_prefix
        ) is None:
            raise ValueError("metric prefix has an invalid format")
        if normalized_prefix not in _metric_prefixes(records):
            raise ValueError(f"metric prefix is unavailable for {tool_name!r}")
    if normalized_plate is not None and normalized_plate not in available_plate_ids:
        raise ValueError(f"plate scope is unavailable for {tool_name!r}")
    if (
        normalized_treatment is not None
        and normalized_treatment not in available_treatments
    ):
        raise ValueError(f"treatment scope is unavailable for {tool_name!r}")
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
        filters={
            "metric_prefix": normalized_prefix,
            "plate_id": normalized_plate,
            "treatment": normalized_treatment,
        },
        total_tool_evidence_count=len(records),
        matched_count=len(matches),
        offset=offset,
        limit=limit,
        returned_count=len(page),
        has_more=offset + len(page) < len(matches),
        evidence=page,
        statuses=tuple(sorted({result.status for result in selected}, key=str)),
        status_reasons=tuple(
            sorted(
                {
                    result.status_reason
                    for result in selected
                    if result.status_reason is not None
                }
            )
        ),
        warnings=_unique_warnings(selected),
        limitations=tuple(
            sorted({item for result in selected for item in result.limitations})
        ),
    )


def resolve_evidence(
    baseline: BaselineReport,
    evidence_ids: tuple[str, ...],
    *,
    maximum_ids: int = 10,
) -> JsonObject:
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
    return {
        "case_id": baseline.case_id,
        "evidence": [
            by_id[item].model_dump(mode="json") for item in sorted(evidence_ids)
        ],
    }
