"""Tests for investigator-safe loading and observed design derivation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiment_failure_investigator.analysis.contracts import PublicArtifactHashes
from experiment_failure_investigator.analysis.design import build_investigator_case
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

ZERO_HASH = "0" * 64
TEST_HASHES = PublicArtifactHashes(
    measurements=ZERO_HASH,
    plate_map=ZERO_HASH,
    metadata=ZERO_HASH,
    protocol=ZERO_HASH,
    problem_statement=ZERO_HASH,
)


def _build_from_loaded(
    case_directory: Path,
    *,
    measurements: pd.DataFrame | None = None,
    plate_map: pd.DataFrame | None = None,
):
    loaded = load_case(case_directory)
    return build_investigator_case(
        measurements=loaded.measurements if measurements is None else measurements,
        plate_map=loaded.plate_map if plate_map is None else plate_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=TEST_HASHES,
    )


def test_all_frozen_benchmark_cases_load_without_private_truth() -> None:
    case_directories = sorted(
        path for path in Path("cases").iterdir() if (path / "manifest.json").is_file()
    )

    assert len(case_directories) == 14
    opaque_ids: set[str] = set()
    for case_directory in case_directories:
        case = load_investigator_case(case_directory)
        serialized = case.model_dump_json()
        source_case_id = case_directory.name

        assert case.case_id.startswith("case_")
        opaque_ids.add(case.case_id)
        assert source_case_id not in serialized
        assert "planted_failure_mode" not in serialized
        assert "ground_truth" not in serialized
        assert "injection_parameters" not in serialized
        assert "child_seeds" not in serialized
        assert "layout_fingerprint" not in serialized
        assert "root_seed" not in serialized
        assert "expected_signal" not in serialized
        assert "injected_effect" not in serialized
    assert len(opaque_ids) == 14


def test_baseline_design_summary_is_derived_from_public_rows() -> None:
    case = load_investigator_case(Path("cases/edge_effect_obvious"))

    assert case.design.plate_count == 1
    assert case.design.plates[0].plate_format is PlateFormat.WELLS_96
    assert case.design.plates[0].observed_well_count == 96
    assert case.design.plates[0].missing_wells == ()
    assert case.design.treatments == ("reference_treatment", "test_treatment")
    assert case.design.capabilities.has_negative_controls
    assert case.design.capabilities.has_positive_controls
    assert case.design.capabilities.has_replicates
    assert case.design.capabilities.has_dose_series
    assert not case.design.capabilities.has_multiple_plates
    assert case.design.capabilities.plates_are_complete_design_replicates is None
    assert not case.design.capabilities.supports_cross_plate_analysis


def test_design_summary_is_invariant_to_public_csv_row_order() -> None:
    case_directory = Path("cases/edge_effect_obvious")
    loaded = load_case(case_directory)
    original = _build_from_loaded(case_directory)
    reordered = _build_from_loaded(
        case_directory,
        measurements=loaded.measurements.sample(frac=1, random_state=17),
        plate_map=loaded.plate_map.sample(frac=1, random_state=29),
    )

    assert reordered.design == original.design
    assert reordered.measurements == original.measurements
    assert reordered.plate_map == original.plate_map


def test_irregular_design_preserves_missingness_and_observed_capabilities() -> None:
    case_directory = Path("cases/edge_effect_obvious")
    loaded = load_case(case_directory)
    plate_map = loaded.plate_map
    selected = pd.concat(
        [
            plate_map[plate_map["well_role"] == "negative_control"].head(2),
            plate_map[plate_map["well_role"] == "positive_control"].head(1),
            plate_map[
                (plate_map["well_role"] == "treatment")
                & (plate_map["treatment"] == "test_treatment")
                & (plate_map["dose"].isin([0.003, 0.01]))
            ].groupby("dose", sort=True).head(2),
        ],
        ignore_index=True,
    )
    second_dose = selected.index[
        (selected["well_role"] == "treatment") & (selected["dose"] == 0.01)
    ]
    selected = selected.drop(second_dose[-1]).reset_index(drop=True)
    selected.loc[selected["well_role"] == "positive_control", "replicate"] = np.nan

    keys = pd.MultiIndex.from_frame(selected[["plate_id", "well"]])
    measurements = loaded.measurements.set_index(["plate_id", "well"]).loc[keys]
    measurements = measurements.reset_index()
    measurements.loc[0, "raw_signal"] = np.nan
    measurements.loc[1, "raw_signal"] = np.inf
    measurements = measurements.iloc[:-1].copy()

    case = _build_from_loaded(
        case_directory,
        measurements=measurements,
        plate_map=selected,
    )

    plate = case.design.plates[0]
    assert plate.role_counts == {
        "negative_control": 2,
        "positive_control": 1,
        "treatment": 3,
    }
    assert plate.missing_measurement_count == 1
    assert plate.null_measurement_count == 1
    assert plate.nonfinite_measurement_count == 1
    assert plate.observed_well_count == 6
    assert len(plate.missing_wells) == 90
    assert case.design.treatments == ("test_treatment",)
    assert case.design.replicate_count_distribution == {1: 2, 2: 2}
    assert case.design.capabilities.has_dose_series


def test_full_384_well_geometry_is_inferred_from_coordinates() -> None:
    wells = enumerate_wells(PlateFormat.WELLS_384)
    plate_map = wells[["well", "row", "column"]].copy()
    plate_map.insert(0, "plate_id", "plate_384")
    plate_map["sample_id"] = [f"empty_{index:03d}" for index in range(1, 385)]
    plate_map["well_role"] = "empty"
    plate_map["treatment"] = None
    plate_map["dose"] = None
    plate_map["dose_unit"] = None
    plate_map["replicate"] = None
    measurements = plate_map[["plate_id", "well"]].copy()
    measurements.insert(0, "case_id", "geometry_fixture")
    measurements["raw_signal"] = 0.0
    metadata = CaseMetadata(
        case_id="geometry_fixture",
        assay_type=AssayType.CELL_VIABILITY_ENDPOINT,
        signal_direction=SignalDirection.LOWER_IS_STRONGER,
        layout_fingerprint=ZERO_HASH,
        plates=[PlateMetadata(plate_id="plate_384", batch_id="batch_01")],
    )

    case = build_investigator_case(
        measurements=measurements,
        plate_map=plate_map,
        metadata=metadata,
        protocol="Synthetic protocol",
        problem_statement="Inspect the assay.",
        public_artifact_hashes=TEST_HASHES,
    )

    assert case.design.plates[0].plate_format is PlateFormat.WELLS_384
    assert case.design.plates[0].expected_well_count == 384
    assert case.design.plates[0].missing_wells == ()


def test_complete_replicate_plates_enable_cross_plate_analysis() -> None:
    case = load_investigator_case(Path("cases/batch_shift_obvious"))

    assert case.design.capabilities.has_multiple_plates
    assert case.design.capabilities.plates_are_complete_design_replicates
    assert case.design.capabilities.supports_cross_plate_analysis
    assert case.design.limitations == ()


def test_conditions_split_across_plates_fail_closed_for_cross_plate_analysis() -> None:
    case_directory = Path("cases/batch_shift_obvious")
    loaded = load_case(case_directory)
    plate_map = loaded.plate_map
    split_map = plate_map[
        ~(
            (
                (plate_map["plate_id"] == "plate_01")
                & (plate_map["treatment"] == "test_treatment")
            )
            | (
                (plate_map["plate_id"] == "plate_02")
                & (plate_map["treatment"] == "reference_treatment")
            )
        )
    ].copy()
    retained_keys = pd.MultiIndex.from_frame(split_map[["plate_id", "well"]])
    split_measurements = (
        loaded.measurements.set_index(["plate_id", "well"])
        .loc[retained_keys]
        .reset_index()
    )

    case = _build_from_loaded(
        case_directory,
        measurements=split_measurements,
        plate_map=split_map,
    )

    assert case.design.capabilities.has_multiple_plates
    assert case.design.capabilities.plates_are_complete_design_replicates is False
    assert not case.design.capabilities.supports_cross_plate_analysis
    assert "unsupported" in " ".join(case.design.limitations).lower()


def test_serialized_contract_contains_only_json_values() -> None:
    case = load_investigator_case(Path("cases/batch_shift_obvious"))

    serialized = case.model_dump_json()

    assert json.loads(serialized)["case_id"] == case.case_id


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_map_key", "duplicate plate/well"),
        ("bad_coordinate", "disagrees with row/column"),
        ("bad_role", "unsupported well role"),
        ("unexpected_measurement", "absent from the plate map"),
        ("metadata_mismatch", "metadata plate references"),
    ],
)
def test_invalid_public_structure_is_rejected(
    mutation: str,
    message: str,
) -> None:
    case_directory = Path("cases/edge_effect_obvious")
    loaded = load_case(case_directory)
    measurements = loaded.measurements.copy(deep=True)
    plate_map = loaded.plate_map.copy(deep=True)
    metadata = loaded.metadata

    if mutation == "duplicate_map_key":
        plate_map.loc[1, ["plate_id", "well"]] = plate_map.loc[
            0, ["plate_id", "well"]
        ].to_numpy()
    elif mutation == "bad_coordinate":
        plate_map.loc[0, "column"] = 12 if plate_map.loc[0, "column"] != 12 else 11
    elif mutation == "bad_role":
        plate_map.loc[0, "well_role"] = "unsupported"
    elif mutation == "unexpected_measurement":
        measurements.loc[0, "well"] = "Z99"
    else:
        metadata = metadata.model_copy(
            update={
                "plates": [
                    metadata.plates[0].model_copy(update={"plate_id": "other_plate"})
                ]
            }
        )

    with pytest.raises(ValueError, match=message):
        build_investigator_case(
            measurements=measurements,
            plate_map=plate_map,
            metadata=metadata,
            protocol=loaded.protocol,
            problem_statement=loaded.problem_statement,
            public_artifact_hashes=TEST_HASHES,
        )
