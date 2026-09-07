"""Tests for deterministic, validated, atomic benchmark case persistence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiment_failure_investigator.benchmark.injectors import (
    EdgeEffectParameters,
    inject_edge_effect,
)
from experiment_failure_investigator.benchmark.layouts import build_balanced_layout
from experiment_failure_investigator.benchmark.models import (
    CaseVariant,
    FailureMode,
    GeneratorConfig,
)
from experiment_failure_investigator.benchmark.serialization import (
    MANIFEST_FILENAME,
    CasePayload,
    build_case_metadata,
    layout_fingerprint,
    load_case,
    sha256_file,
    validate_case_content,
    write_case,
)
from experiment_failure_investigator.benchmark.signals import generate_clean_assay


@pytest.fixture
def payload() -> CasePayload:
    config = GeneratorConfig(
        case_id="edge_effect_obvious",
        case_variant=CaseVariant.OBVIOUS,
        failure_mode=FailureMode.EDGE_EFFECT,
        root_seed=42,
        noise_sd=0.03,
    )
    clean = generate_clean_assay(build_balanced_layout(config), config)
    injected = inject_edge_effect(
        clean,
        EdgeEffectParameters(magnitude=0.25, sides=("top",)),
    )
    metadata = build_case_metadata(
        config,
        injected.plate_map,
    )
    return CasePayload(
        config=config,
        assay=injected,
        metadata=metadata,
        protocol="# Protocol\n\nSynthetic endpoint viability assay.\n",
        problem_statement="A spatial pattern may be present.\n",
        expected_discriminating_evidence=(
            "A condition-adjusted difference on the selected plate boundary.",
        ),
        plausible_confounders=("Condition placement near the boundary.",),
    )


def test_case_round_trips_with_typed_manifest_and_verified_hashes(
    tmp_path: Path,
    payload: CasePayload,
) -> None:
    case_directory = write_case(payload, tmp_path)
    loaded = load_case(case_directory)

    assert loaded.manifest.case_id == payload.config.case_id
    assert loaded.manifest.generation == payload.config
    assert loaded.metadata == payload.metadata
    assert loaded.protocol == payload.protocol
    assert loaded.problem_statement == payload.problem_statement
    assert list(loaded.measurements.columns) == list(payload.assay.measurements.columns)
    assert list(loaded.plate_map.columns) == list(payload.assay.plate_map.columns)
    assert "expected_signal" not in loaded.measurements
    assert "injected_effect" not in loaded.measurements

    serialized_files = loaded.manifest.files.model_dump()
    references = [
        reference
        for name, reference in serialized_files.items()
        if name != "plots"
    ]
    for reference in references:
        assert sha256_file(case_directory / reference["path"]) == reference["sha256"]
    assert MANIFEST_FILENAME not in {
        reference["path"] for reference in references
    }


def test_serialization_is_byte_reproducible(
    tmp_path: Path,
    payload: CasePayload,
) -> None:
    first = write_case(payload, tmp_path / "first")
    second = write_case(payload, tmp_path / "second")

    first_files = {path.name: path.read_bytes() for path in first.iterdir()}
    second_files = {path.name: path.read_bytes() for path in second.iterdir()}
    assert first_files == second_files


def test_layout_fingerprint_ignores_order_ids_and_replicate_labels(
    payload: CasePayload,
) -> None:
    original = payload.assay.plate_map
    equivalent = original.sample(frac=1, random_state=7).copy()
    equivalent["plate_id"] = "renamed_plate"
    equivalent["sample_id"] = "changed"
    equivalent["replicate"] = 99

    assert layout_fingerprint(original, payload.config) == layout_fingerprint(
        equivalent,
        payload.config,
    )


def test_layout_fingerprint_changes_with_semantic_assignment(
    payload: CasePayload,
) -> None:
    original = payload.assay.plate_map
    changed = original.copy(deep=True)
    first, second = changed.index[:2]
    semantic_columns = ["well_role", "treatment", "dose"]
    changed.loc[[first, second], semantic_columns] = changed.loc[
        [second, first], semantic_columns
    ].to_numpy()

    assert layout_fingerprint(original, payload.config) != layout_fingerprint(
        changed,
        payload.config,
    )


def test_plot_artifacts_are_hashed_and_validated(
    tmp_path: Path,
    payload: CasePayload,
) -> None:
    def write_plot(plot_directory: Path) -> dict[str, Path]:
        output = plot_directory / "inspection.png"
        output.write_bytes(b"synthetic-png")
        return {"inspection": output}

    case_directory = write_case(payload, tmp_path, plot_writer=write_plot)
    loaded = load_case(case_directory)

    plot = loaded.manifest.files.plots["inspection"]
    assert plot.path == "plots/inspection.png"
    assert sha256_file(case_directory / plot.path) == plot.sha256


def test_existing_case_requires_force_and_force_replaces_atomically(
    tmp_path: Path,
    payload: CasePayload,
) -> None:
    case_directory = write_case(payload, tmp_path)
    with pytest.raises(FileExistsError, match="already exists"):
        write_case(payload, tmp_path)

    replacement = replace(
        payload,
        problem_statement="A revised synthetic observation.\n",
    )
    replaced_directory = write_case(replacement, tmp_path, force=True)

    assert replaced_directory == case_directory
    assert (
        load_case(replaced_directory).problem_statement
        == replacement.problem_statement
    )
    assert not list(tmp_path.glob(".*.staging-*"))
    assert not list(tmp_path.glob(".*.backup-*"))


def test_hash_corruption_is_rejected(
    tmp_path: Path,
    payload: CasePayload,
) -> None:
    case_directory = write_case(payload, tmp_path)
    measurements = case_directory / "measurements.csv"
    measurements.write_bytes(measurements.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="artifact hash mismatch: measurements.csv"):
        load_case(case_directory)


def test_artifact_symlink_cannot_escape_case_directory(
    tmp_path: Path,
    payload: CasePayload,
) -> None:
    case_directory = write_case(payload, tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_text("not case data\n", encoding="utf-8")
    measurements = case_directory / "measurements.csv"
    measurements.unlink()
    measurements.symlink_to(outside)

    with pytest.raises(ValueError, match="escapes the case directory"):
        load_case(case_directory)


@pytest.mark.parametrize(
    "mutation",
    ["duplicate_key", "missing_key", "bad_role", "bad_dose", "nonfinite"],
)
def test_invalid_case_content_is_rejected_before_writing(
    tmp_path: Path,
    payload: CasePayload,
    mutation: str,
) -> None:
    measurements = payload.assay.measurements.copy(deep=True)
    plate_map = payload.assay.plate_map.copy(deep=True)
    if mutation == "duplicate_key":
        measurements.loc[1, ["plate_id", "well"]] = measurements.loc[
            0, ["plate_id", "well"]
        ].to_numpy()
    elif mutation == "missing_key":
        measurements = measurements.iloc[:-1].copy()
    elif mutation == "bad_role":
        plate_map.loc[0, "well_role"] = "unknown_role"
    elif mutation == "bad_dose":
        treatment_index = plate_map.index[plate_map["well_role"] == "treatment"][0]
        plate_map.loc[treatment_index, "dose"] = -1.0
    else:
        measurements.loc[0, "raw_signal"] = np.inf
    invalid_assay = replace(
        payload.assay,
        measurements=measurements,
        plate_map=plate_map,
    )

    with pytest.raises(ValueError):
        write_case(replace(payload, assay=invalid_assay), tmp_path)
    assert not (tmp_path / payload.config.case_id).exists()
    assert not list(tmp_path.glob(".*.staging-*"))


def test_metadata_plate_references_must_match(
    payload: CasePayload,
) -> None:
    invalid_metadata = payload.metadata.model_copy(
        update={
            "plates": [
                payload.metadata.plates[0].model_copy(
                    update={"plate_id": "unknown_plate"}
                )
            ]
        }
    )

    with pytest.raises(ValueError, match="metadata plate references"):
        validate_case_content(
            payload.assay.measurements,
            payload.assay.plate_map,
            invalid_metadata,
            payload.config,
            payload.protocol,
            payload.problem_statement,
        )


def test_private_columns_cannot_leak_into_public_tables(
    payload: CasePayload,
) -> None:
    measurements = payload.assay.measurements.assign(injected_effect=0.25)

    with pytest.raises(ValueError, match="unexpected injected_effect"):
        validate_case_content(
            measurements,
            payload.assay.plate_map,
            payload.metadata,
            payload.config,
            payload.protocol,
            payload.problem_statement,
        )
