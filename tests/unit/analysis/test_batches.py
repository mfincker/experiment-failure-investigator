"""Tests for matched, investigator-safe plate comparisons."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiment_failure_investigator.analysis.batches import compare_batches
from experiment_failure_investigator.analysis.contracts import PublicArtifactHashes
from experiment_failure_investigator.analysis.design import build_investigator_case
from experiment_failure_investigator.analysis.results import ToolStatus
from experiment_failure_investigator.benchmark.adapter import load_investigator_case
from experiment_failure_investigator.benchmark.serialization import load_case

ZERO_HASH = "0" * 64
HASHES = PublicArtifactHashes(
    measurements=ZERO_HASH,
    plate_map=ZERO_HASH,
    metadata=ZERO_HASH,
    protocol=ZERO_HASH,
    problem_statement=ZERO_HASH,
)


def _metrics(case_name: str) -> dict[str, float]:
    result = compare_batches(
        load_investigator_case(Path("cases") / case_name)
    )[0]
    assert result.status is ToolStatus.SUCCESS
    return {
        record.metric_name: float(record.value)
        for record in result.evidence
        if record.metric_name
        in {
            "batches.negative_control_anchor_difference",
            "batches.control_window_ratio",
            "batches.response_range_ratio",
        }
    }


@pytest.mark.parametrize("variant", ["obvious", "noisy"])
def test_batch_cases_show_reduced_response_with_anchored_negative_controls(
    variant: str,
) -> None:
    metrics = _metrics(f"batch_shift_{variant}")

    assert abs(metrics["batches.negative_control_anchor_difference"]) < 0.03
    assert 0.5 < metrics["batches.control_window_ratio"] < 0.9
    assert 0.5 < metrics["batches.response_range_ratio"] < 0.9


def test_identical_complete_replicate_plates_have_zero_differences() -> None:
    loaded = load_case(Path("cases/batch_shift_obvious"))
    measurements = loaded.measurements.copy()
    first = measurements[measurements["plate_id"] == "plate_01"].set_index("well")
    second_mask = measurements["plate_id"] == "plate_02"
    measurements.loc[second_mask, "raw_signal"] = measurements.loc[
        second_mask, "well"
    ].map(first["raw_signal"])
    case = build_investigator_case(
        measurements=measurements,
        plate_map=loaded.plate_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )

    result = compare_batches(case)[0]
    differences = [
        float(record.value)
        for record in result.evidence
        if record.metric_name == "batches.condition_mean_difference"
    ]
    ratios = [
        float(record.value)
        for record in result.evidence
        if record.metric_name
        in {"batches.control_window_ratio", "batches.response_range_ratio"}
    ]

    assert differences
    assert differences == pytest.approx([0.0] * len(differences))
    assert ratios == pytest.approx([1.0, 1.0])


def test_split_conditions_across_plates_fail_closed() -> None:
    loaded = load_case(Path("cases/batch_shift_obvious"))
    plate_map = loaded.plate_map
    keep = ~(
        ((plate_map["plate_id"] == "plate_01") & (plate_map["treatment"] == "test_treatment"))
        | ((plate_map["plate_id"] == "plate_02") & (plate_map["treatment"] == "reference_treatment"))
    )
    split_map = plate_map.loc[keep].copy()
    keys = pd.MultiIndex.from_frame(split_map[["plate_id", "well"]])
    measurements = (
        loaded.measurements.set_index(["plate_id", "well"]).loc[keys].reset_index()
    )
    case = build_investigator_case(
        measurements=measurements,
        plate_map=split_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )

    result = compare_batches(case)[0]

    assert result.status is ToolStatus.INSUFFICIENT_DATA
    assert not result.evidence
    assert "complete replicates" in result.status_reason


def test_single_plate_is_not_applicable() -> None:
    result = compare_batches(
        load_investigator_case(Path("cases/edge_effect_obvious"))
    )[0]

    assert result.status is ToolStatus.NOT_APPLICABLE
    assert not result.evidence


def test_plate_and_csv_order_do_not_change_results() -> None:
    loaded = load_case(Path("cases/batch_shift_obvious"))
    original = build_investigator_case(
        measurements=loaded.measurements,
        plate_map=loaded.plate_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )
    reordered_metadata = loaded.metadata.model_copy(
        update={"plates": list(reversed(loaded.metadata.plates))}
    )
    reordered = build_investigator_case(
        measurements=loaded.measurements.sample(frac=1, random_state=13),
        plate_map=loaded.plate_map.sample(frac=1, random_state=17),
        metadata=reordered_metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )

    assert compare_batches(reordered) == compare_batches(original)


def test_denominator_guard_and_same_batch_status() -> None:
    case = load_investigator_case(Path("cases/batch_shift_obvious"))
    with pytest.raises(ValueError, match="finite and positive"):
        compare_batches(case, denominator_epsilon=0)

    same_batch_metadata = case.metadata.model_copy(
        update={
            "plates": tuple(
                plate.model_copy(update={"batch_id": "one_batch"})
                for plate in case.metadata.plates
            )
        }
    )
    same_batch_case = case.model_copy(update={"metadata": same_batch_metadata})
    result = compare_batches(same_batch_case)[0]
    assert result.status is ToolStatus.NOT_APPLICABLE
