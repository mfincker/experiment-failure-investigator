"""Tests for benchmark configuration and manifest contracts."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from experiment_failure_investigator.benchmark.models import (
    ArtifactReference,
    AssayType,
    CaseFiles,
    CaseManifest,
    CaseVariant,
    FailureMode,
    GeneratorConfig,
    GroundTruth,
    PlateFormat,
    SignalDirection,
    WellCoordinate,
)

VALID_HASH = "0" * 64


def artifact(path: str) -> ArtifactReference:
    return ArtifactReference(path=path, sha256=VALID_HASH)


def valid_config(**overrides: Any) -> GeneratorConfig:
    values: dict[str, Any] = {
        "case_id": "edge_effect_obvious",
        "case_variant": CaseVariant.OBVIOUS,
        "failure_mode": FailureMode.EDGE_EFFECT,
        "root_seed": 42,
        "noise_sd": 0.03,
    }
    values.update(overrides)
    return GeneratorConfig(**values)


def valid_manifest(**overrides: Any) -> CaseManifest:
    generation = overrides.pop("generation", valid_config())
    values: dict[str, Any] = {
        "case_id": generation.case_id,
        "case_variant": generation.case_variant,
        "root_seed": generation.root_seed,
        "assay_type": generation.assay_type,
        "signal_direction": generation.signal_direction,
        "planted_failure_mode": generation.failure_mode,
        "generation": generation,
        "ground_truth": GroundTruth(
            mechanism=generation.failure_mode,
            child_seeds={"layout": 1, "baseline_noise": 2, "injection": 3},
            injection_parameters={"edge_shift": 0.2},
        ),
        "expected_discriminating_evidence": [
            "Edge wells differ from condition-matched interior wells."
        ],
        "plausible_confounders": ["Condition placement at the plate boundary."],
        "files": CaseFiles(
            measurements=artifact("measurements.csv"),
            plate_map=artifact("plate_map.csv"),
            metadata=artifact("metadata.json"),
            protocol=artifact("protocol.md"),
            problem_statement=artifact("problem_statement.txt"),
        ),
    }
    values.update(overrides)
    return CaseManifest(**values)


def test_minimal_valid_manifest() -> None:
    manifest = valid_manifest()

    assert manifest.case_id == "edge_effect_obvious"
    assert manifest.assay_type is AssayType.CELL_VIABILITY_ENDPOINT
    assert manifest.signal_direction is SignalDirection.LOWER_IS_STRONGER
    assert manifest.generation.negative_control_count == 8
    assert manifest.generation.positive_control_count == 8


def test_unknown_failure_mode_is_rejected() -> None:
    with pytest.raises(ValidationError, match="failure_mode"):
        valid_config(failure_mode="instrument_gremlins")


@pytest.mark.parametrize(
    ("row", "column"),
    [("1", 1), ("AA", 1), ("A", 0), ("A", 100)],
)
def test_invalid_well_is_rejected(row: str, column: int) -> None:
    with pytest.raises(ValidationError):
        WellCoordinate(row=row, column=column)


def test_well_is_normalized_to_canonical_identifier() -> None:
    assert WellCoordinate(row="b", column=3).well == "B03"


def test_well_syntax_does_not_assume_a_96_well_plate() -> None:
    assert WellCoordinate(row="P", column=24).well == "P24"
    assert PlateFormat.WELLS_384.capacity == 384


def test_negative_noise_is_rejected() -> None:
    with pytest.raises(ValidationError, match="noise_sd"):
        valid_config(noise_sd=-0.01)


def test_manifest_artifact_requires_hash() -> None:
    with pytest.raises(ValidationError, match="sha256"):
        ArtifactReference.model_validate({"path": "measurements.csv"})


@pytest.mark.parametrize(
    "path",
    ["/tmp/measurements.csv", "../measurements.csv", "nested\\measurements.csv"],
)
def test_artifact_path_must_stay_inside_case_directory(path: str) -> None:
    with pytest.raises(ValidationError, match="artifact path"):
        ArtifactReference(path=path, sha256=VALID_HASH)


def test_plate_design_must_fill_96_wells() -> None:
    with pytest.raises(ValidationError, match="fill exactly 96 wells"):
        valid_config(replicates_per_condition=4)


def test_generator_contract_allows_different_control_and_blank_counts() -> None:
    config = valid_config(
        negative_control_count=12,
        positive_control_count=12,
        blank_count=8,
        replicates_per_condition=4,
    )

    assert config.negative_control_count == 12
    assert config.blank_count == 8


def test_generator_contract_does_not_require_reference_treatment() -> None:
    config = valid_config(
        treatments=["test_treatment"],
        replicates_per_condition=10,
        curves={"test_treatment": {"ic50": 1.0}},
    )

    assert config.treatments == ["test_treatment"]


def test_generator_contract_supports_384_well_capacity() -> None:
    config = valid_config(
        plate_format=PlateFormat.WELLS_384,
        negative_control_count=16,
        positive_control_count=16,
        blank_count=32,
        replicates_per_condition=20,
    )

    assert config.plate_format.capacity == 384


def test_doses_must_be_positive_unique_and_sorted() -> None:
    for doses in ([0.0] * 8, [0.1, 0.1], [1.0, 0.1]):
        with pytest.raises(ValidationError, match="doses"):
            valid_config(doses_micromolar=doses)


def test_batch_shift_requires_multiple_plates() -> None:
    with pytest.raises(ValidationError, match="at least two plates"):
        valid_config(failure_mode=FailureMode.BATCH_SHIFT)

    config = valid_config(
        case_id="batch_shift_obvious",
        failure_mode=FailureMode.BATCH_SHIFT,
        plate_count=2,
    )
    assert config.plate_count == 2


def test_manifest_rejects_inconsistent_nested_values() -> None:
    with pytest.raises(ValidationError, match="case_id"):
        valid_manifest(case_id="different_case")


def test_unknown_fields_fail_closed() -> None:
    values = valid_config().model_dump()
    values["unexpected"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GeneratorConfig.model_validate(values)
