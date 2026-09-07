"""Tests for within-condition replicate variability summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment_failure_investigator.analysis.contracts import (
    InvestigatorCase,
    PublicArtifactHashes,
)
from experiment_failure_investigator.analysis.design import build_investigator_case
from experiment_failure_investigator.analysis.replicates import (
    calculate_replicate_variability,
)
from experiment_failure_investigator.analysis.results import (
    EvidenceRecord,
    ScientificToolResult,
    ToolStatus,
)
from experiment_failure_investigator.benchmark.adapter import load_investigator_case
from experiment_failure_investigator.benchmark.serialization import load_case

VALID_HASH = "0" * 64
HASHES = PublicArtifactHashes(
    measurements=VALID_HASH,
    plate_map=VALID_HASH,
    metadata=VALID_HASH,
    protocol=VALID_HASH,
    problem_statement=VALID_HASH,
)


def _metric(result: ScientificToolResult, name: str) -> EvidenceRecord:
    matches = [record for record in result.evidence if record.metric_name == name]
    assert len(matches) == 1
    return matches[0]


def _single_condition_case(values: list[float | None]) -> InvestigatorCase:
    loaded = load_case(Path("cases/true_non_response_obvious"))
    plate_map = loaded.plate_map
    condition = plate_map[
        (plate_map["treatment"] == "test_treatment") & (plate_map["dose"] == 0.3)
    ].head(len(values))
    controls = plate_map[plate_map["well_role"] != "treatment"]
    selected_indices = np.concatenate(
        [controls.index.to_numpy(), condition.index.to_numpy()]
    )
    reduced_map = plate_map.loc[selected_indices].copy()
    retained = set(reduced_map["well"])
    measurements = loaded.measurements[
        loaded.measurements["well"].isin(retained)
    ].copy()
    for row, value in zip(condition.itertuples(index=False), values, strict=True):
        measurements.loc[measurements["well"] == row.well, "raw_signal"] = value
    return build_investigator_case(
        measurements=measurements,
        plate_map=reduced_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )


def test_hand_computed_replicate_statistics_and_distribution() -> None:
    results = calculate_replicate_variability(_single_condition_case([1.0, 2.0, 3.0]))
    condition = next(
        result for result in results if result.parameters["component"] == "condition"
    )
    distribution = next(
        result
        for result in results
        if result.parameters["component"] == "plate_distribution"
    )

    assert condition.status is ToolStatus.SUCCESS
    assert _metric(condition, "replicates.mean").value == 2.0
    assert _metric(condition, "replicates.median").value == 2.0
    assert _metric(condition, "replicates.sample_standard_deviation").value == 1.0
    assert _metric(condition, "replicates.median_absolute_deviation").value == 1.0
    assert distribution.status is ToolStatus.SUCCESS
    assert _metric(
        distribution, "replicates.conditions_with_variability_count"
    ).value == 1


def test_single_finite_replicate_is_structured_insufficient_data() -> None:
    results = calculate_replicate_variability(_single_condition_case([1.0, None]))
    condition = next(
        result for result in results if result.parameters["component"] == "condition"
    )
    distribution = next(
        result
        for result in results
        if result.parameters["component"] == "plate_distribution"
    )

    assert condition.status is ToolStatus.INSUFFICIENT_DATA
    assert _metric(condition, "replicates.mapped_well_count").value == 2
    assert _metric(condition, "replicates.observed_signal_count").value == 1
    assert _metric(condition, "replicates.mean").value == 1.0
    assert condition.warnings[0].code == "missing_replicate_signals"
    assert distribution.status is ToolStatus.INSUFFICIENT_DATA


def test_no_annotated_treatments_is_not_applicable() -> None:
    loaded = load_case(Path("cases/true_non_response_obvious"))
    plate_map = loaded.plate_map[loaded.plate_map["well_role"] != "treatment"].copy()
    retained = set(plate_map["well"])
    measurements = loaded.measurements[
        loaded.measurements["well"].isin(retained)
    ].copy()
    case = build_investigator_case(
        measurements=measurements,
        plate_map=plate_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )

    results = calculate_replicate_variability(case)

    assert len(results) == 1
    assert results[0].status is ToolStatus.NOT_APPLICABLE


def test_unequal_replicate_counts_are_reported_without_assuming_five() -> None:
    loaded = load_case(Path("cases/true_non_response_obvious"))
    plate_map = loaded.plate_map.copy(deep=True)
    measurements = loaded.measurements.copy(deep=True)
    group = plate_map[
        (plate_map["treatment"] == "test_treatment") & (plate_map["dose"] == 0.3)
    ]
    measurements = measurements[measurements["well"] != group.iloc[0]["well"]].copy()
    case = build_investigator_case(
        measurements=measurements,
        plate_map=plate_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )

    results = calculate_replicate_variability(case)
    target = next(
        result
        for result in results
        if result.parameters["component"] == "condition"
        and result.evidence[0].scope.treatments == ("test_treatment",)
        and result.evidence[0].scope.doses == (0.3,)
    )

    assert target.status is ToolStatus.SUCCESS
    assert _metric(target, "replicates.mapped_well_count").value == 5
    assert _metric(target, "replicates.observed_signal_count").value == 4
    assert target.warnings[0].code == "missing_replicate_signals"


def test_two_plate_results_remain_plate_specific() -> None:
    case = load_investigator_case(Path("cases/batch_shift_obvious"))
    results = calculate_replicate_variability(case)

    condition_results = [
        result for result in results if result.parameters["component"] == "condition"
    ]
    distribution_results = [
        result
        for result in results
        if result.parameters["component"] == "plate_distribution"
    ]
    assert len(condition_results) == 32
    assert len(distribution_results) == 2
    assert {result.scope.plate_ids for result in distribution_results} == {
        ("plate_01",),
        ("plate_02",),
    }


def test_replicate_results_are_invariant_to_source_row_order() -> None:
    loaded = load_case(Path("cases/true_non_response_obvious"))
    original = build_investigator_case(
        measurements=loaded.measurements,
        plate_map=loaded.plate_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )
    reordered = build_investigator_case(
        measurements=loaded.measurements.sample(frac=1, random_state=3),
        plate_map=loaded.plate_map.sample(frac=1, random_state=5),
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )

    assert calculate_replicate_variability(original) == calculate_replicate_variability(
        reordered
    )
