"""Tests for deterministic missingness diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment_failure_investigator.analysis.contracts import PublicArtifactHashes
from experiment_failure_investigator.analysis.design import build_investigator_case
from experiment_failure_investigator.analysis.missingness import inspect_missingness
from experiment_failure_investigator.analysis.results import EvidenceRecord, ToolStatus
from experiment_failure_investigator.benchmark.models import WellRole
from experiment_failure_investigator.benchmark.serialization import load_case

VALID_HASH = "0" * 64
HASHES = PublicArtifactHashes(
    measurements=VALID_HASH,
    plate_map=VALID_HASH,
    metadata=VALID_HASH,
    protocol=VALID_HASH,
    problem_statement=VALID_HASH,
)


def _record(
    evidence: tuple[EvidenceRecord, ...],
    metric_name: str,
    *,
    role: WellRole | None = None,
) -> EvidenceRecord:
    matches = [
        record
        for record in evidence
        if record.metric_name == metric_name
        and (role is None or record.scope.well_roles == (role,))
    ]
    assert len(matches) == 1
    return matches[0]


def test_complete_case_reports_zero_missingness_without_warning() -> None:
    loaded = load_case(Path("cases/true_non_response_obvious"))
    case = build_investigator_case(
        measurements=loaded.measurements,
        plate_map=loaded.plate_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )

    result = inspect_missingness(case)

    assert result.status is ToolStatus.SUCCESS
    assert result.warnings == ()
    unmapped = _record(result.evidence, "missingness.unmapped_well_count")
    assert unmapped.value == 0
    assert unmapped.sample_count == 96
    assert (
        _record(result.evidence, "missingness.missing_measurement_count").value
        == 0
    )


def test_missingness_is_separated_by_plate_and_role() -> None:
    loaded = load_case(Path("cases/true_non_response_obvious"))
    measurements = loaded.measurements.copy(deep=True)
    plate_map = loaded.plate_map.copy(deep=True)

    negative_well = plate_map.loc[
        plate_map["well_role"] == "negative_control", "well"
    ].iloc[0]
    positive_well = plate_map.loc[
        plate_map["well_role"] == "positive_control", "well"
    ].iloc[0]
    treatment_indices = plate_map.index[plate_map["well_role"] == "treatment"]
    nonfinite_well = plate_map.loc[treatment_indices[0], "well"]
    plate_map.loc[treatment_indices[1], "replicate"] = np.nan

    measurements = measurements[measurements["well"] != negative_well].copy()
    measurements.loc[measurements["well"] == positive_well, "raw_signal"] = np.nan
    measurements.loc[measurements["well"] == nonfinite_well, "raw_signal"] = np.inf
    case = build_investigator_case(
        measurements=measurements,
        plate_map=plate_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )

    result = inspect_missingness(case)

    assert result.status is ToolStatus.SUCCESS
    assert len(result.warnings) == 1
    assert (
        _record(
            result.evidence,
            "missingness.role_missing_measurement_count",
            role=WellRole.NEGATIVE_CONTROL,
        ).value
        == 1
    )
    assert (
        _record(
            result.evidence,
            "missingness.role_null_measurement_count",
            role=WellRole.POSITIVE_CONTROL,
        ).value
        == 1
    )
    assert (
        _record(
            result.evidence,
            "missingness.role_nonfinite_measurement_count",
            role=WellRole.TREATMENT,
        ).value
        == 1
    )
    assert (
        _record(
            result.evidence,
            "missingness.role_missing_design_annotation_count",
            role=WellRole.TREATMENT,
        ).value
        == 1
    )
    warning_ids = set(result.warnings[0].evidence_ids)
    assert warning_ids
    assert warning_ids <= {record.evidence_id for record in result.evidence}


def test_two_plate_missingness_has_distinct_plate_scopes() -> None:
    loaded = load_case(Path("cases/batch_shift_obvious"))
    case = build_investigator_case(
        measurements=loaded.measurements,
        plate_map=loaded.plate_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )

    result = inspect_missingness(case)
    expected_counts = [
        record
        for record in result.evidence
        if record.metric_name == "missingness.expected_well_count"
    ]

    assert len(expected_counts) == 2
    assert {record.scope.plate_ids for record in expected_counts} == {
        ("plate_01",),
        ("plate_02",),
    }
