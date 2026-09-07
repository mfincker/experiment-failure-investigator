"""Tests for geometry-aware residual and layout diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment_failure_investigator.analysis.contracts import PublicArtifactHashes
from experiment_failure_investigator.analysis.design import build_investigator_case
from experiment_failure_investigator.analysis.results import (
    EvidenceRecord,
    ScientificToolResult,
    ToolStatus,
)
from experiment_failure_investigator.analysis.spatial import (
    condition_centered_residuals,
    detect_spatial_effects,
)
from experiment_failure_investigator.benchmark.adapter import load_investigator_case
from experiment_failure_investigator.benchmark.layouts import enumerate_wells
from experiment_failure_investigator.benchmark.models import (
    AssayType,
    CaseMetadata,
    PlateFormat,
    PlateMetadata,
    SignalDirection,
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


def _result(case_name: str) -> ScientificToolResult:
    case = load_investigator_case(Path("cases") / case_name)
    results = detect_spatial_effects(case)
    assert len(results) == 1
    return results[0]


def _metric(result: ScientificToolResult, name: str) -> EvidenceRecord:
    matches = [record for record in result.evidence if record.metric_name == name]
    assert len(matches) == 1
    return matches[0]


def _numeric_metric(result: ScientificToolResult, name: str) -> float:
    value = _metric(result, name).value
    assert isinstance(value, int | float)
    return float(value)


def test_condition_residuals_are_median_centered_within_public_groups() -> None:
    case = load_investigator_case(Path("cases/edge_effect_obvious"))
    residuals = condition_centered_residuals(case)
    frame = pd.DataFrame(record.model_dump(mode="json") for record in residuals)

    medians = frame.groupby(
        ["plate_id", "well_role", "treatment", "dose", "dose_unit"],
        dropna=False,
    )["condition_residual"].median()

    assert len(residuals) == 96
    assert all(abs(value) < 1e-12 for value in medians)


def test_edge_effect_strength_tracks_obvious_noisy_and_non_edge_cases() -> None:
    obvious = _numeric_metric(_result("edge_effect_obvious"), "spatial.edge_minus_interior")
    noisy = _numeric_metric(_result("edge_effect_noisy"), "spatial.edge_minus_interior")
    non_edge = abs(
        _numeric_metric(
            _result("true_non_response_obvious"),
            "spatial.edge_minus_interior",
        )
    )

    assert obvious > noisy > non_edge


def test_progressive_drift_has_row_trend_without_dispense_order() -> None:
    obvious = _numeric_metric(
        _result("pipetting_drift_obvious"), "spatial.row_correlation"
    )
    noisy = _numeric_metric(
        _result("pipetting_drift_noisy"), "spatial.row_correlation"
    )
    comparison = abs(
        _numeric_metric(
            _result("true_non_response_obvious"),
            "spatial.row_correlation",
        )
    )

    assert obvious > noisy > comparison
    assert "dispense order" in " ".join(
        _result("pipetting_drift_obvious").limitations
    )


def test_transient_pattern_is_localized_without_claiming_a_tip_cause() -> None:
    obvious = _result("transient_tip_clog_obvious")
    noisy = _result("transient_tip_clog_noisy")
    comparison = _result("true_non_response_obvious")

    assert _numeric_metric(
        obvious, "spatial.largest_adjacent_extreme_run_size"
    ) == 4
    assert _numeric_metric(
        noisy, "spatial.largest_adjacent_extreme_run_size"
    ) == 2
    assert _numeric_metric(
        comparison, "spatial.largest_adjacent_extreme_run_size"
    ) == 0
    assert "do not establish tip identity" in " ".join(obvious.limitations)


def test_layout_confounding_produces_nonidentifiability_warning() -> None:
    for variant in ("obvious", "noisy"):
        result = _result(f"layout_confounding_{variant}")

        assert result.status is ToolStatus.SUCCESS
        assert _numeric_metric(result, "spatial.single_row_condition_count") > 0
        assert "condition_position_nonidentifiability" in {
            warning.code for warning in result.warnings
        }

    baseline = _result("true_non_response_obvious")
    assert _numeric_metric(baseline, "spatial.single_row_condition_count") == 0
    assert "condition_position_nonidentifiability" not in {
        warning.code for warning in baseline.warnings
    }


def test_spatial_diagnostics_support_384_well_geometry() -> None:
    wells = enumerate_wells(PlateFormat.WELLS_384)
    plate_map = wells[["well", "row", "column"]].copy()
    plate_map.insert(0, "plate_id", "plate_384")
    plate_map["sample_id"] = [f"sample_{index:03d}" for index in range(384)]
    plate_map["well_role"] = "treatment"
    plate_map["treatment"] = "test_treatment"
    plate_map["dose"] = 1.0
    plate_map["dose_unit"] = "uM"
    plate_map["replicate"] = list(range(1, 385))
    measurements = plate_map[["plate_id", "well", "row"]].copy()
    measurements.insert(0, "case_id", "spatial_fixture")
    measurements["raw_signal"] = measurements["row"].map(
        lambda row: 1.0 + (ord(row) - ord("A")) * 0.01
    )
    measurements = measurements.drop(columns="row")
    metadata = CaseMetadata(
        case_id="spatial_fixture",
        assay_type=AssayType.CELL_VIABILITY_ENDPOINT,
        signal_direction=SignalDirection.LOWER_IS_STRONGER,
        layout_fingerprint=VALID_HASH,
        plates=[PlateMetadata(plate_id="plate_384", batch_id="batch_01")],
    )
    case = build_investigator_case(
        measurements=measurements,
        plate_map=plate_map,
        metadata=metadata,
        protocol="Synthetic 384-well protocol.",
        problem_statement="Inspect spatial structure.",
        public_artifact_hashes=HASHES,
    )

    result = detect_spatial_effects(case)[0]

    assert case.design.plates[0].plate_format is PlateFormat.WELLS_384
    assert _numeric_metric(result, "spatial.row_correlation") > 0.99


def test_spatial_results_are_invariant_to_source_row_order() -> None:
    loaded = load_case(Path("cases/transient_tip_clog_noisy"))
    original = build_investigator_case(
        measurements=loaded.measurements,
        plate_map=loaded.plate_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )
    reordered = build_investigator_case(
        measurements=loaded.measurements.sample(frac=1, random_state=13),
        plate_map=loaded.plate_map.sample(frac=1, random_state=17),
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )

    assert detect_spatial_effects(original) == detect_spatial_effects(reordered)
