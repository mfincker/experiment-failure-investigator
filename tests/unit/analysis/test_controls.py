"""Tests for deterministic control summaries and assay-window statistics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiment_failure_investigator.analysis.contracts import (
    InvestigatorCase,
    PublicArtifactHashes,
)
from experiment_failure_investigator.analysis.controls import summarize_controls
from experiment_failure_investigator.analysis.design import build_investigator_case
from experiment_failure_investigator.analysis.results import (
    EvidenceRecord,
    ScientificToolResult,
    ToolStatus,
)
from experiment_failure_investigator.benchmark.serialization import load_case

VALID_HASH = "0" * 64
HASHES = PublicArtifactHashes(
    measurements=VALID_HASH,
    plate_map=VALID_HASH,
    metadata=VALID_HASH,
    protocol=VALID_HASH,
    problem_statement=VALID_HASH,
)


def _control_case(
    negative: list[float | None],
    positive: list[float | None],
) -> InvestigatorCase:
    loaded = load_case(Path("cases/true_non_response_obvious"))
    plate_map = loaded.plate_map
    selected_negative = plate_map[plate_map["well_role"] == "negative_control"].head(
        len(negative)
    )
    selected_positive = plate_map[plate_map["well_role"] == "positive_control"].head(
        len(positive)
    )
    selected_map = np.concatenate(
        [
            selected_negative.index.to_numpy(),
            selected_positive.index.to_numpy(),
            plate_map.index[plate_map["well_role"] == "treatment"].to_numpy(),
        ]
    )
    reduced_map = plate_map.loc[selected_map].copy()
    retained_wells = set(reduced_map["well"])
    measurements = loaded.measurements[
        loaded.measurements["well"].isin(retained_wells)
    ].copy()
    for row, value in zip(
        selected_negative.itertuples(index=False), negative, strict=True
    ):
        measurements.loc[measurements["well"] == row.well, "raw_signal"] = value
    for row, value in zip(
        selected_positive.itertuples(index=False), positive, strict=True
    ):
        measurements.loc[measurements["well"] == row.well, "raw_signal"] = value
    return build_investigator_case(
        measurements=measurements,
        plate_map=reduced_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )


def _component(
    results: tuple[ScientificToolResult, ...],
    component: str,
    *,
    role: str | None = None,
) -> ScientificToolResult:
    matches = [
        result
        for result in results
        if result.parameters["component"] == component
        and (role is None or result.parameters.get("role") == role)
    ]
    assert len(matches) == 1
    return matches[0]


def _metric(result: ScientificToolResult, metric_name: str) -> EvidenceRecord:
    matches = [
        evidence for evidence in result.evidence if evidence.metric_name == metric_name
    ]
    assert len(matches) == 1
    return matches[0]


def test_control_statistics_match_hand_calculation() -> None:
    case = _control_case(negative=[0.9, 1.1], positive=[0.1, 0.3])
    results = summarize_controls(case)
    negative = _component(results, "role_summary", role="negative_control")
    separation = _component(results, "control_separation")

    assert negative.status is ToolStatus.SUCCESS
    assert _metric(negative, "controls.mean").value == pytest.approx(1.0)
    assert _metric(negative, "controls.median").value == pytest.approx(1.0)
    assert _metric(
        negative, "controls.sample_standard_deviation"
    ).value == pytest.approx(0.14142135623730953)
    assert _metric(
        negative, "controls.median_absolute_deviation"
    ).value == pytest.approx(0.1)
    assert _metric(separation, "controls.signed_separation").value == pytest.approx(
        0.8
    )
    assert _metric(separation, "controls.z_prime").value == pytest.approx(
        -0.060660171779821415
    )


def test_absent_positive_controls_are_not_applicable() -> None:
    results = summarize_controls(_control_case(negative=[0.9, 1.1], positive=[]))

    assert _component(
        results, "role_summary", role="negative_control"
    ).status is ToolStatus.SUCCESS
    assert _component(
        results, "role_summary", role="positive_control"
    ).status is ToolStatus.NOT_APPLICABLE
    assert _component(
        results, "control_separation"
    ).status is ToolStatus.NOT_APPLICABLE


def test_singleton_control_retains_levels_but_not_variability() -> None:
    results = summarize_controls(_control_case(negative=[1.0], positive=[0.1, 0.3]))
    negative = _component(results, "role_summary", role="negative_control")
    separation = _component(results, "control_separation")

    assert negative.status is ToolStatus.INSUFFICIENT_DATA
    assert _metric(negative, "controls.mean").value == 1.0
    assert not any(
        evidence.metric_name == "controls.sample_standard_deviation"
        for evidence in negative.evidence
    )
    assert separation.status is ToolStatus.INSUFFICIENT_DATA
    assert _metric(separation, "controls.absolute_separation").value == pytest.approx(
        0.8
    )
    assert not any(
        evidence.metric_name == "controls.z_prime"
        for evidence in separation.evidence
    )


def test_zero_control_window_is_structured_insufficient_data() -> None:
    results = summarize_controls(
        _control_case(negative=[1.0, 1.0], positive=[1.0, 1.0])
    )
    separation = _component(results, "control_separation")

    assert separation.status is ToolStatus.INSUFFICIENT_DATA
    assert _metric(separation, "controls.absolute_separation").value == 0.0
    assert separation.warnings[0].code == "zero_control_window"
    assert not any(
        evidence.metric_name == "controls.z_prime"
        for evidence in separation.evidence
    )


def test_robust_dispersion_is_not_driven_by_one_outlier() -> None:
    results = summarize_controls(
        _control_case(negative=[1.0, 1.0, 1.0, 10.0], positive=[0.1, 0.2])
    )
    negative = _component(results, "role_summary", role="negative_control")

    assert _metric(negative, "controls.mean").value == pytest.approx(3.25)
    assert _metric(negative, "controls.median").value == pytest.approx(1.0)
    assert _metric(
        negative, "controls.sample_standard_deviation"
    ).value == pytest.approx(4.5)
    assert _metric(negative, "controls.median_absolute_deviation").value == 0.0


def test_missing_control_value_is_excluded_and_reported() -> None:
    results = summarize_controls(
        _control_case(negative=[1.0, None, 1.1], positive=[0.1, 0.2])
    )
    negative = _component(results, "role_summary", role="negative_control")

    assert negative.status is ToolStatus.SUCCESS
    assert _metric(negative, "controls.mapped_well_count").value == 3
    assert _metric(negative, "controls.observed_signal_count").value == 2
    assert negative.warnings[0].code == "missing_control_signals"


def test_two_plate_case_returns_independent_plate_results() -> None:
    from experiment_failure_investigator.benchmark.adapter import load_investigator_case

    case = load_investigator_case(Path("cases/batch_shift_obvious"))
    results = summarize_controls(case)

    assert len(results) == 6
    assert {result.scope.plate_ids for result in results} == {
        ("plate_01",),
        ("plate_02",),
    }
    assert all(result.status is ToolStatus.SUCCESS for result in results)


def test_weak_control_case_has_less_separation_than_healthy_controls() -> None:
    from experiment_failure_investigator.benchmark.adapter import load_investigator_case

    healthy = summarize_controls(
        load_investigator_case(Path("cases/true_non_response_obvious"))
    )
    weak = summarize_controls(
        load_investigator_case(Path("cases/weak_controls_obvious"))
    )
    healthy_separation = _metric(
        _component(healthy, "control_separation"), "controls.absolute_separation"
    ).value
    weak_separation = _metric(
        _component(weak, "control_separation"), "controls.absolute_separation"
    ).value

    assert isinstance(healthy_separation, float)
    assert isinstance(weak_separation, float)
    assert weak_separation < healthy_separation


def test_denominator_epsilon_must_be_positive_and_finite() -> None:
    case = _control_case(negative=[1.0, 1.1], positive=[0.1, 0.2])

    for value in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            summarize_controls(case, denominator_epsilon=value)
