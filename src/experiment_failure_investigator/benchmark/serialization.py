"""Deterministic, validated persistence for synthetic benchmark cases."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import JsonValue

from experiment_failure_investigator.benchmark.injectors import InjectedAssayResult
from experiment_failure_investigator.benchmark.layouts import (
    PLATE_MAP_COLUMNS,
    enumerate_wells,
)
from experiment_failure_investigator.benchmark.models import (
    ArtifactReference,
    CaseFiles,
    CaseManifest,
    CaseMetadata,
    GeneratorConfig,
    GroundTruth,
    PlateMetadata,
    SimulatedTraversal,
    WellCoordinate,
    WellRole,
)
from experiment_failure_investigator.benchmark.signals import MEASUREMENT_COLUMNS

MANIFEST_FILENAME = "manifest.json"
PUBLIC_FILENAMES = {
    "measurements": "measurements.csv",
    "plate_map": "plate_map.csv",
    "metadata": "metadata.json",
    "protocol": "protocol.md",
    "problem_statement": "problem_statement.txt",
}
PlotWriter = Callable[[Path], Mapping[str, Path]]


@dataclass(frozen=True)
class CasePayload:
    """All inputs required to validate and serialize one injected case."""

    config: GeneratorConfig
    assay: InjectedAssayResult
    metadata: CaseMetadata
    protocol: str
    problem_statement: str
    expected_discriminating_evidence: tuple[str, ...]
    plausible_confounders: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoadedCase:
    """A validated case loaded from its serialized representation."""

    directory: Path
    manifest: CaseManifest
    measurements: pd.DataFrame
    plate_map: pd.DataFrame
    metadata: CaseMetadata
    protocol: str
    problem_statement: str


def build_case_metadata(
    config: GeneratorConfig,
    plate_map: pd.DataFrame,
) -> CaseMetadata:
    """Build minimal public metadata without inventing optional provenance."""
    plate_ids = sorted(str(value) for value in plate_map["plate_id"].unique())
    return CaseMetadata(
        case_id=config.case_id,
        assay_type=config.assay_type,
        signal_direction=config.signal_direction,
        layout_fingerprint=layout_fingerprint(plate_map, config),
        plates=[
            PlateMetadata(plate_id=plate_id, batch_id=plate_id)
            for plate_id in sorted(plate_ids)
        ],
    )


def layout_fingerprint(
    plate_map: pd.DataFrame,
    config: GeneratorConfig,
) -> str:
    """Fingerprint semantic well assignments independently of IDs and row order."""
    required = {"plate_id", "well", "well_role", "treatment", "dose"}
    missing = sorted(required - set(plate_map.columns))
    if missing:
        raise ValueError(
            "layout fingerprint is missing required columns: " + ", ".join(missing)
        )
    fingerprints: list[str] = []
    for _, plate in plate_map.groupby("plate_id", sort=False):
        records: list[dict[str, JsonValue]] = []
        for row in plate.sort_values("well", kind="stable").itertuples(index=False):
            dose = getattr(row, "dose")
            records.append(
                {
                    "well": str(getattr(row, "well")),
                    "well_role": str(getattr(row, "well_role")),
                    "treatment": (
                        None
                        if pd.isna(getattr(row, "treatment"))
                        else str(getattr(row, "treatment"))
                    ),
                    "dose": None if pd.isna(dose) else float(dose),
                }
            )
        payload = {
            "plate_format": config.plate_format.value,
            "assignments": records,
        }
        fingerprints.append(
            hashlib.sha256(_json_text(payload).encode("utf-8")).hexdigest()
        )
    unique_fingerprints = sorted(set(fingerprints))
    if not unique_fingerprints:
        raise ValueError("layout fingerprint requires at least one plate")
    if len(unique_fingerprints) == 1:
        return unique_fingerprints[0]
    return hashlib.sha256(
        _json_text(unique_fingerprints).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_text(value: str, field_name: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized + "\n"


def _json_text(value: Any) -> str:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _canonical_csv(frame: pd.DataFrame, columns: list[str]) -> str:
    canonical = frame.loc[:, columns].sort_values(
        ["plate_id", "well"],
        kind="stable",
    )
    return canonical.to_csv(
        index=False,
        columns=columns,
        float_format="%.12g",
        lineterminator="\n",
        na_rep="",
    )


def _validate_exact_columns(
    frame: pd.DataFrame,
    expected: list[str],
    table_name: str,
) -> None:
    missing = sorted(set(expected) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected))
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError(f"{table_name} columns are invalid: " + "; ".join(details))


def validate_case_content(
    measurements: pd.DataFrame,
    plate_map: pd.DataFrame,
    metadata: CaseMetadata,
    config: GeneratorConfig,
    protocol: str,
    problem_statement: str,
) -> None:
    """Validate cross-file structure and semantics before or after persistence."""
    _validate_exact_columns(measurements, MEASUREMENT_COLUMNS, "measurements")
    _validate_exact_columns(plate_map, PLATE_MAP_COLUMNS, "plate map")
    _canonical_text(protocol, "protocol")
    _canonical_text(problem_statement, "problem statement")

    for name, frame in (("measurements", measurements), ("plate map", plate_map)):
        if frame[["plate_id", "well"]].isna().any().any():
            raise ValueError(f"{name} contains null plate/well keys")
        if frame.duplicated(["plate_id", "well"]).any():
            raise ValueError(f"{name} contains duplicate plate/well keys")

    measurement_keys = set(
        measurements[["plate_id", "well"]].itertuples(index=False, name=None)
    )
    plate_map_keys = set(
        plate_map[["plate_id", "well"]].itertuples(index=False, name=None)
    )
    if measurement_keys != plate_map_keys:
        raise ValueError("measurement and plate-map keys do not match")

    case_ids = set(measurements["case_id"].dropna())
    if case_ids != {config.case_id}:
        raise ValueError("measurement case IDs disagree with the configuration")

    plate_ids = set(plate_map["plate_id"])
    metadata_plate_ids = {plate.plate_id for plate in metadata.plates}
    if plate_ids != metadata_plate_ids:
        raise ValueError("metadata plate references do not match the plate map")
    if len(plate_ids) != config.plate_count:
        raise ValueError("serialized plate count disagrees with the configuration")
    if (
        metadata.case_id != config.case_id
        or metadata.assay_type != config.assay_type
        or metadata.signal_direction != config.signal_direction
    ):
        raise ValueError("metadata disagrees with the generator configuration")
    if metadata.layout_fingerprint != layout_fingerprint(plate_map, config):
        raise ValueError("metadata layout fingerprint does not match the plate map")

    expected_geometry = enumerate_wells(config.plate_format).set_index("well")
    expected_wells = set(expected_geometry.index)
    for plate_id, plate in plate_map.groupby("plate_id", sort=False):
        if set(plate["well"]) != expected_wells:
            raise ValueError(
                f"plate {plate_id!r} does not match the declared plate geometry"
            )
        for row in plate[["well", "row", "column"]].itertuples(index=False):
            coordinate = WellCoordinate.model_validate(
                {"row": row.row, "column": row.column}
            )
            if coordinate.well != row.well:
                raise ValueError(f"well {row.well!r} disagrees with row/column fields")

    allowed_roles = {role.value for role in WellRole}
    observed_roles = set(plate_map["well_role"].dropna())
    if not observed_roles <= allowed_roles or plate_map["well_role"].isna().any():
        raise ValueError("plate map contains unsupported or missing well roles")

    treatment_rows = plate_map["well_role"] == WellRole.TREATMENT.value
    control_rows = plate_map["well_role"].isin(
        [WellRole.NEGATIVE_CONTROL.value, WellRole.POSITIVE_CONTROL.value]
    )
    treatment_doses = pd.to_numeric(
        plate_map.loc[treatment_rows, "dose"],
        errors="coerce",
    )
    treatment_replicates = pd.to_numeric(
        plate_map.loc[treatment_rows, "replicate"],
        errors="coerce",
    )
    if (
        plate_map.loc[treatment_rows, "treatment"].isna().any()
        or treatment_doses.isna().any()
        or (treatment_doses <= 0).any()
        or plate_map.loc[treatment_rows, "dose_unit"].isna().any()
        or treatment_replicates.isna().any()
        or (treatment_replicates < 1).any()
    ):
        raise ValueError("treatment rows contain inconsistent dose or replicate data")
    if (
        plate_map.loc[control_rows, ["dose", "dose_unit"]]
        .notna()
        .any()
        .any()
    ):
        raise ValueError("control rows must not contain dose fields")
    control_treatments = plate_map.loc[control_rows, "treatment"].dropna()
    if control_treatments.astype(str).str.strip().eq("").any():
        raise ValueError("control treatment labels must not be blank")
    control_replicates = pd.to_numeric(
        plate_map.loc[control_rows, "replicate"].dropna(),
        errors="coerce",
    )
    if control_replicates.isna().any() or (control_replicates < 1).any():
        raise ValueError("control replicate values must be positive when present")

    for plate_id, plate in plate_map.groupby("plate_id", sort=False):
        roles = plate["well_role"].value_counts()
        if (
            roles.get(WellRole.NEGATIVE_CONTROL.value, 0)
            != config.negative_control_count
        ):
            raise ValueError(f"plate {plate_id!r} has the wrong negative-control count")
        if (
            roles.get(WellRole.POSITIVE_CONTROL.value, 0)
            != config.positive_control_count
        ):
            raise ValueError(f"plate {plate_id!r} has the wrong positive-control count")
        if roles.get(WellRole.EMPTY.value, 0) != config.empty_count:
            raise ValueError(f"plate {plate_id!r} has the wrong empty-well count")
        observed = plate.loc[treatment_rows.loc[plate.index]].groupby(
            ["treatment", "dose"],
            dropna=False,
        ).size()
        expected_conditions = {
            (treatment, dose): config.replicates_per_condition
            for treatment in config.treatments
            for dose in config.doses_micromolar
        }
        if observed.to_dict() != expected_conditions:
            raise ValueError(f"plate {plate_id!r} treatment-dose counts are inconsistent")

    numeric_signal = pd.to_numeric(measurements["raw_signal"], errors="coerce")
    roles_by_key = plate_map.set_index(["plate_id", "well"])["well_role"]
    measurement_index = pd.MultiIndex.from_frame(measurements[["plate_id", "well"]])
    measurement_roles = roles_by_key.reindex(measurement_index).to_numpy()
    required_finite = measurement_roles != WellRole.EMPTY.value
    if not np.isfinite(numeric_signal.to_numpy()[required_finite]).all():
        raise ValueError("non-empty wells require finite measurements")


def _validate_payload(payload: CasePayload) -> None:
    generation = payload.assay.generation_metadata
    if (
        generation.case_id != payload.config.case_id
        or generation.root_seed != payload.config.root_seed
        or generation.plate_count != payload.config.plate_count
    ):
        raise ValueError("assay generation metadata disagrees with the configuration")
    if payload.assay.injection.mechanism != payload.config.failure_mode:
        raise ValueError("injected mechanism disagrees with the configuration")
    if not payload.expected_discriminating_evidence:
        raise ValueError("at least one expected evidence statement is required")
    validate_case_content(
        payload.assay.measurements,
        payload.assay.plate_map,
        payload.metadata,
        payload.config,
        payload.protocol,
        payload.problem_statement,
    )


def _artifact_reference(directory: Path, filename: str) -> ArtifactReference:
    return ArtifactReference(path=filename, sha256=sha256_file(directory / filename))


def _write_staged_case(
    payload: CasePayload,
    directory: Path,
    plot_writer: PlotWriter | None,
) -> CaseManifest:
    measurements_path = directory / PUBLIC_FILENAMES["measurements"]
    plate_map_path = directory / PUBLIC_FILENAMES["plate_map"]
    metadata_path = directory / PUBLIC_FILENAMES["metadata"]
    protocol_path = directory / PUBLIC_FILENAMES["protocol"]
    problem_path = directory / PUBLIC_FILENAMES["problem_statement"]

    measurements_path.write_text(
        _canonical_csv(payload.assay.measurements, MEASUREMENT_COLUMNS),
        encoding="utf-8",
        newline="",
    )
    plate_map_path.write_text(
        _canonical_csv(payload.assay.plate_map, PLATE_MAP_COLUMNS),
        encoding="utf-8",
        newline="",
    )
    metadata_path.write_text(
        _json_text(payload.metadata.model_dump(mode="json")),
        encoding="utf-8",
        newline="",
    )
    protocol_path.write_text(
        _canonical_text(payload.protocol, "protocol"),
        encoding="utf-8",
        newline="",
    )
    problem_path.write_text(
        _canonical_text(payload.problem_statement, "problem statement"),
        encoding="utf-8",
        newline="",
    )

    plot_references: dict[str, ArtifactReference] = {}
    if plot_writer is not None:
        plot_directory = directory / "plots"
        plot_directory.mkdir()
        written_plots = dict(plot_writer(plot_directory))
        if not written_plots:
            raise ValueError("plot writer must return at least one plot")
        for name, path in written_plots.items():
            if not name.strip():
                raise ValueError("plot names must not be blank")
            resolved_path = path.resolve()
            if not resolved_path.is_relative_to(directory.resolve()):
                raise ValueError("plot writer returned a path outside the case directory")
            if not path.is_file():
                raise ValueError(f"plot writer did not create {name!r}")
            relative_path = path.relative_to(directory).as_posix()
            plot_references[name] = _artifact_reference(directory, relative_path)

    files = CaseFiles(
        measurements=_artifact_reference(
            directory,
            PUBLIC_FILENAMES["measurements"],
        ),
        plate_map=_artifact_reference(directory, PUBLIC_FILENAMES["plate_map"]),
        metadata=_artifact_reference(directory, PUBLIC_FILENAMES["metadata"]),
        protocol=_artifact_reference(directory, PUBLIC_FILENAMES["protocol"]),
        problem_statement=_artifact_reference(
            directory,
            PUBLIC_FILENAMES["problem_statement"],
        ),
        plots=plot_references,
    )
    injection_parameters = payload.assay.injection.parameters
    serialized_traversal = injection_parameters.get("traversal")
    if serialized_traversal is None:
        simulated_traversal = None
    elif isinstance(serialized_traversal, str):
        simulated_traversal = SimulatedTraversal(serialized_traversal)
    else:
        raise ValueError("serialized traversal must be a string when present")
    manifest = CaseManifest(
        case_id=payload.config.case_id,
        case_variant=payload.config.case_variant,
        root_seed=payload.config.root_seed,
        assay_type=payload.config.assay_type,
        signal_direction=payload.config.signal_direction,
        planted_failure_mode=payload.config.failure_mode,
        generation=payload.config,
        ground_truth=GroundTruth(
            mechanism=payload.assay.injection.mechanism,
            child_seeds={
                "baseline_noise": payload.assay.generation_metadata.baseline_noise_seed,
                "failure_injection": payload.assay.injection.child_seed,
            },
            injection_parameters=injection_parameters,
            simulated_traversal=simulated_traversal,
        ),
        expected_discriminating_evidence=list(
            payload.expected_discriminating_evidence
        ),
        plausible_confounders=list(payload.plausible_confounders),
        files=files,
    )
    (directory / MANIFEST_FILENAME).write_text(
        _json_text(manifest.model_dump(mode="json")),
        encoding="utf-8",
        newline="",
    )
    return manifest


def write_case(
    payload: CasePayload,
    output_root: Path,
    *,
    force: bool = False,
    plot_writer: PlotWriter | None = None,
) -> Path:
    """Validate and atomically write one case beneath ``output_root``."""
    _validate_payload(payload)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / payload.config.case_id
    if destination.exists() and not force:
        raise FileExistsError(f"case directory already exists: {destination}")
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError("case destination must be a regular directory")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{payload.config.case_id}.staging-", dir=output_root)
    )
    backup: Path | None = None
    try:
        _write_staged_case(payload, staging, plot_writer)
        load_case(staging)
        if destination.exists():
            backup = output_root / f".{payload.config.case_id}.backup-{uuid4().hex}"
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except Exception:
            if backup is not None:
                os.replace(backup, destination)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _resolve_artifact(case_directory: Path, artifact: ArtifactReference) -> Path:
    path = case_directory / artifact.path
    resolved_root = case_directory.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"manifest artifact escapes the case directory: {artifact.path}")
    if not path.is_file():
        raise ValueError(f"manifest artifact is missing: {artifact.path}")
    if sha256_file(path) != artifact.sha256:
        raise ValueError(f"artifact hash mismatch: {artifact.path}")
    return path


def load_case(case_directory: Path) -> LoadedCase:
    """Load a case only after validating its manifest, hashes, and content."""
    manifest_path = case_directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError("case manifest is missing")
    manifest = CaseManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    artifacts = [
        manifest.files.measurements,
        manifest.files.plate_map,
        manifest.files.metadata,
        manifest.files.protocol,
        manifest.files.problem_statement,
        *manifest.files.plots.values(),
    ]
    artifact_paths = [artifact.path for artifact in artifacts]
    if len(artifact_paths) != len(set(artifact_paths)):
        raise ValueError("manifest artifact paths must be unique")

    measurements_path = _resolve_artifact(case_directory, manifest.files.measurements)
    plate_map_path = _resolve_artifact(case_directory, manifest.files.plate_map)
    metadata_path = _resolve_artifact(case_directory, manifest.files.metadata)
    protocol_path = _resolve_artifact(case_directory, manifest.files.protocol)
    problem_path = _resolve_artifact(case_directory, manifest.files.problem_statement)
    for plot in manifest.files.plots.values():
        _resolve_artifact(case_directory, plot)

    measurements = pd.read_csv(measurements_path)
    plate_map = pd.read_csv(plate_map_path)
    metadata = CaseMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    protocol = protocol_path.read_text(encoding="utf-8")
    problem_statement = problem_path.read_text(encoding="utf-8")
    validate_case_content(
        measurements,
        plate_map,
        metadata,
        manifest.generation,
        protocol,
        problem_statement,
    )
    return LoadedCase(
        directory=case_directory,
        manifest=manifest,
        measurements=measurements,
        plate_map=plate_map,
        metadata=metadata,
        protocol=protocol,
        problem_statement=problem_statement,
    )
