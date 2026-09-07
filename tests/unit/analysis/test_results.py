"""Tests for stable scientific result and evidence contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import JsonValue, ValidationError

from experiment_failure_investigator.analysis.contracts import PublicArtifactHashes
from experiment_failure_investigator.analysis.results import (
    ArtifactAttachment,
    EvidenceRecord,
    EvidenceScope,
    ResultWarning,
    RuntimeTelemetry,
    ScientificProvenance,
    ScientificToolResult,
    ToolExecution,
    ToolStatus,
    build_evidence_id,
    build_evidence_record,
    canonical_json,
)

CASE_ID = "case_0123456789abcdef"
VALID_HASH = "0" * 64
HASHES = PublicArtifactHashes(
    measurements=VALID_HASH,
    plate_map=VALID_HASH,
    metadata=VALID_HASH,
    protocol=VALID_HASH,
    problem_statement=VALID_HASH,
)
PROVENANCE = ScientificProvenance(
    case_id=CASE_ID,
    public_artifact_hashes=HASHES,
)
PARAMETERS: dict[str, JsonValue] = {"estimator": "mean", "trim_fraction": 0.0}


def _scope(**overrides: Any) -> EvidenceScope:
    values: dict[str, Any] = {
        "case_id": CASE_ID,
        "plate_ids": ("plate_01",),
        "well_roles": ("negative_control",),
    }
    values.update(overrides)
    return EvidenceScope(**values)


def _evidence(
    metric_name: str = "controls.mean",
    *,
    scope: EvidenceScope | None = None,
    parameters: dict[str, JsonValue] | None = None,
) -> EvidenceRecord:
    return build_evidence_record(
        tool_name="summarize_controls",
        tool_version="1.0.0",
        parameters=PARAMETERS if parameters is None else parameters,
        metric_name=metric_name,
        value=0.98,
        unit="raw_signal",
        scope=_scope() if scope is None else scope,
        sample_count=8,
        description="Mean negative-control signal on plate 01.",
    )


def _result(
    status: ToolStatus = ToolStatus.SUCCESS,
    *,
    evidence: tuple[EvidenceRecord, ...] | None = None,
    status_reason: str | None = None,
) -> ScientificToolResult:
    return ScientificToolResult(
        status=status,
        status_reason=status_reason,
        tool_name="summarize_controls",
        tool_version="1.0.0",
        parameters=PARAMETERS,
        scope=_scope(),
        evidence=(_evidence(),) if evidence is None else evidence,
        provenance=PROVENANCE,
    )


def test_successful_result_has_canonical_json_and_traceable_evidence() -> None:
    result = _result()
    serialized = canonical_json(result)

    assert serialized.endswith("\n")
    assert '": ' not in serialized
    assert '", ' not in serialized
    assert result.evidence[0].evidence_id in serialized
    assert canonical_json(result) == serialized


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (ToolStatus.NOT_APPLICABLE, "The design has no positive controls."),
        (ToolStatus.INSUFFICIENT_DATA, "Only one positive control is observed."),
        (ToolStatus.ERROR, "The requested estimator is invalid."),
    ],
)
def test_non_success_statuses_require_explicit_reasons(
    status: ToolStatus,
    reason: str,
) -> None:
    result = _result(status, evidence=(), status_reason=reason)

    assert result.status is status
    assert result.status_reason == reason


def test_insufficient_data_can_preserve_partial_evidence() -> None:
    result = _result(
        ToolStatus.INSUFFICIENT_DATA,
        evidence=(_evidence(),),
        status_reason="A mean is available, but variability needs two observations.",
    )

    assert len(result.evidence) == 1


def test_not_applicable_and_error_cannot_claim_scientific_evidence() -> None:
    for status in (ToolStatus.NOT_APPLICABLE, ToolStatus.ERROR):
        with pytest.raises(ValidationError, match="must not contain evidence"):
            _result(status, status_reason="The tool cannot produce a result.")


def test_non_success_requires_reason_and_success_rejects_reason() -> None:
    with pytest.raises(ValidationError, match="require a status reason"):
        _result(ToolStatus.INSUFFICIENT_DATA, evidence=())
    with pytest.raises(ValidationError, match="must not include a status reason"):
        _result(status_reason="No reason belongs on success.")


def test_evidence_id_ignores_mapping_and_scope_input_order() -> None:
    first_scope = _scope(
        plate_ids=("plate_02", "plate_01"),
        well_ids=("B02", "A01"),
        dimensions={"axis": "row", "bounds": {"high": 3, "low": 1}},
    )
    second_scope = _scope(
        plate_ids=("plate_01", "plate_02"),
        well_ids=("A01", "B02"),
        dimensions={"bounds": {"low": 1, "high": 3}, "axis": "row"},
    )
    first_parameters: dict[str, JsonValue] = {"trim": 0.1, "method": "mean"}
    second_parameters: dict[str, JsonValue] = {"method": "mean", "trim": 0.1}

    first = build_evidence_id(
        tool_name="summarize_controls",
        tool_version="1.0.0",
        parameters=first_parameters,
        metric_name="controls.mean",
        scope=first_scope,
    )
    second = build_evidence_id(
        tool_name="summarize_controls",
        tool_version="1.0.0",
        parameters=second_parameters,
        metric_name="controls.mean",
        scope=second_scope,
    )

    assert first == second


def test_evidence_identity_changes_with_scientific_scope_or_tool_version() -> None:
    identities = {
        build_evidence_id(
            tool_name=tool_name,
            tool_version=version,
            parameters=PARAMETERS,
            metric_name=metric,
            scope=scope,
        )
        for tool_name, version, metric, scope in (
            ("summarize_controls", "1.0.0", "controls.mean", _scope()),
            (
                "summarize_controls",
                "1.0.0",
                "controls.mean",
                _scope(plate_ids=("plate_02",)),
            ),
            (
                "summarize_controls",
                "1.0.0",
                "controls.mean",
                _scope(treatments=("test_treatment",)),
            ),
            ("summarize_controls", "1.0.0", "controls.median", _scope()),
            ("summarize_controls", "1.1.0", "controls.mean", _scope()),
            ("inspect_missingness", "1.0.0", "controls.mean", _scope()),
        )
    }

    assert len(identities) == 6


def test_tool_result_rejects_tampered_evidence_identity() -> None:
    evidence = _evidence().model_copy(update={"evidence_id": "ev_" + "f" * 20})

    with pytest.raises(ValidationError, match="canonical identity"):
        _result(evidence=(evidence,))


def test_result_order_is_canonicalized() -> None:
    mean = _evidence("controls.mean")
    median = _evidence("controls.median")
    first = ScientificToolResult(
        status=ToolStatus.SUCCESS,
        tool_name="summarize_controls",
        tool_version="1.0.0",
        parameters=PARAMETERS,
        scope=_scope(),
        evidence=(mean, median),
        warnings=(
            ResultWarning(code="small_group", message="Group is small."),
            ResultWarning(code="assay_specific", message="Interpret in context."),
        ),
        limitations=("Second limitation.", "First limitation."),
        provenance=PROVENANCE,
    )
    second = ScientificToolResult(
        status=ToolStatus.SUCCESS,
        tool_name="summarize_controls",
        tool_version="1.0.0",
        parameters={"trim_fraction": 0.0, "estimator": "mean"},
        scope=_scope(),
        evidence=(median, mean),
        warnings=tuple(reversed(first.warnings)),
        limitations=tuple(reversed(first.limitations)),
        provenance=PROVENANCE,
    )

    assert first == second
    assert canonical_json(first) == canonical_json(second)


def test_artifact_attachments_are_canonical_and_not_numeric_evidence() -> None:
    first = ArtifactAttachment(
        name="raw_signal",
        path="reports/raw.png",
        sha256=VALID_HASH,
        media_type="image/png",
        description="Public-data plate heatmap.",
    )
    second = ArtifactAttachment(
        name="condition_residual",
        path="reports/residual.png",
        sha256="1" * 64,
        media_type="image/png",
        description="Public-data residual heatmap.",
    )
    result = ScientificToolResult(
        status=ToolStatus.SUCCESS,
        tool_name="generate_plate_heatmap",
        tool_version="1.0.0",
        scope=_scope(),
        attachments=(first, second),
        provenance=PROVENANCE,
    )

    assert result.evidence == ()
    assert tuple(item.name for item in result.attachments) == (
        "condition_residual",
        "raw_signal",
    )


def test_not_applicable_result_rejects_attachments() -> None:
    with pytest.raises(ValidationError, match="evidence or attachments"):
        ScientificToolResult(
            status=ToolStatus.NOT_APPLICABLE,
            status_reason="No plate is available.",
            tool_name="generate_plate_heatmap",
            tool_version="1.0.0",
            scope=_scope(),
            attachments=(
                ArtifactAttachment(
                    name="raw_signal",
                    path="reports/raw.png",
                    sha256=VALID_HASH,
                    media_type="image/png",
                    description="Unexpected artifact.",
                ),
            ),
            provenance=PROVENANCE,
        )


def test_runtime_telemetry_does_not_change_scientific_output() -> None:
    result = _result()
    first = ToolExecution(
        result=result,
        telemetry=RuntimeTelemetry(
            execution_id="run_01",
            started_at=datetime(2026, 9, 7, 9, 0, tzinfo=UTC),
            duration_ms=12.5,
        ),
    )
    second = ToolExecution(
        result=result,
        telemetry=RuntimeTelemetry(
            execution_id="run_02",
            started_at=datetime(2026, 9, 7, 9, 0, tzinfo=UTC)
            + timedelta(minutes=5),
            duration_ms=30.0,
        ),
    )

    assert first.telemetry != second.telemetry
    assert first.result == second.result
    assert canonical_json(first.result) == canonical_json(second.result)


@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), {"nested": [1.0, float("-inf")]}],
)
def test_nonfinite_evidence_and_parameters_are_rejected(
    invalid_value: JsonValue,
) -> None:
    with pytest.raises(ValidationError, match="non-finite"):
        EvidenceRecord(
            evidence_id="ev_" + "0" * 20,
            metric_name="controls.mean",
            value=invalid_value,
            scope=_scope(),
            description="Invalid evidence.",
        )
    with pytest.raises(ValidationError, match="non-finite"):
        ScientificToolResult(
            status=ToolStatus.SUCCESS,
            tool_name="summarize_controls",
            tool_version="1.0.0",
            parameters={"invalid": invalid_value},
            scope=_scope(),
            provenance=PROVENANCE,
        )


def test_unknown_fields_and_naive_runtime_timestamps_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EvidenceScope.model_validate({"case_id": CASE_ID, "secret": "value"})
    with pytest.raises(ValidationError, match="timezone"):
        RuntimeTelemetry(
            execution_id="run_01",
            started_at=datetime(2026, 9, 7, 9, 0),
            duration_ms=1.0,
        )


def test_warning_cannot_reference_evidence_from_another_result() -> None:
    with pytest.raises(ValidationError, match="outside this result"):
        ScientificToolResult(
            status=ToolStatus.SUCCESS,
            tool_name="summarize_controls",
            tool_version="1.0.0",
            parameters=PARAMETERS,
            scope=_scope(),
            evidence=(_evidence(),),
            warnings=(
                ResultWarning(
                    code="unrelated",
                    message="References unavailable evidence.",
                    evidence_ids=("ev_" + "f" * 20,),
                ),
            ),
            provenance=PROVENANCE,
        )
