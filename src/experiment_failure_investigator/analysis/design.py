"""Build canonical investigator inputs and derive observed design capabilities."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import pandas as pd

from experiment_failure_investigator.analysis.contracts import (
    ConditionCoverage,
    DesignCapabilities,
    DesignSummary,
    InvestigatorCase,
    InvestigatorMeasurement,
    InvestigatorMetadata,
    InvestigatorPlateMapWell,
    InvestigatorPlateMetadata,
    PlateDesignSummary,
    PublicArtifactHashes,
    TreatmentDoseSeries,
)
from experiment_failure_investigator.benchmark.layouts import (
    PLATE_MAP_COLUMNS,
    enumerate_wells,
)
from experiment_failure_investigator.benchmark.models import (
    CaseMetadata,
    PlateFormat,
    WellCoordinate,
    WellRole,
)
from experiment_failure_investigator.benchmark.signals import MEASUREMENT_COLUMNS


def _require_exact_columns(
    frame: pd.DataFrame,
    expected: Iterable[str],
    table_name: str,
) -> None:
    expected_set = set(expected)
    missing = sorted(expected_set - set(frame.columns))
    extra = sorted(set(frame.columns) - expected_set)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError(f"{table_name} columns are invalid: " + "; ".join(details))


def _optional_string(value: Any, field_name: str) -> str | None:
    if pd.isna(value):
        return None
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank when present")
    return normalized


def _required_string(value: Any, field_name: str) -> str:
    normalized = _optional_string(value, field_name)
    if normalized is None:
        raise ValueError(f"{field_name} must not be missing")
    return normalized


def _optional_positive_float(value: Any, field_name: str) -> float | None:
    if pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be numeric when present") from error
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{field_name} must be finite and positive when present")
    return number


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be an integer when present") from error
    if not np.isfinite(number) or not number.is_integer() or number < 1:
        raise ValueError(f"{field_name} must be a positive integer when present")
    return int(number)


def _infer_plate_format(wells: tuple[InvestigatorPlateMapWell, ...]) -> PlateFormat:
    if not wells:
        raise ValueError("each metadata plate must contain at least one mapped well")
    max_row = max(ord(well.row) - ord("A") + 1 for well in wells)
    max_column = max(well.column for well in wells)
    if (
        max_row <= PlateFormat.WELLS_96.rows
        and max_column <= PlateFormat.WELLS_96.columns
    ):
        return PlateFormat.WELLS_96
    if (
        max_row <= PlateFormat.WELLS_384.rows
        and max_column <= PlateFormat.WELLS_384.columns
    ):
        return PlateFormat.WELLS_384
    raise ValueError("well coordinates exceed supported 384-well geometry")


def _condition_key(well: InvestigatorPlateMapWell) -> tuple[str, str, str, str]:
    return (
        well.well_role.value,
        well.treatment or "",
        "" if well.dose is None else format(well.dose, ".12g"),
        well.dose_unit or "",
    )


def _opaque_case_id(hashes: PublicArtifactHashes) -> str:
    canonical = "\n".join(
        f"{name}={value}"
        for name, value in sorted(hashes.model_dump(mode="json").items())
    )
    return "case_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _parse_plate_map(plate_map: pd.DataFrame) -> tuple[InvestigatorPlateMapWell, ...]:
    _require_exact_columns(plate_map, PLATE_MAP_COLUMNS, "plate map")
    if plate_map.duplicated(["plate_id", "well"]).any():
        raise ValueError("plate map contains duplicate plate/well keys")

    records: list[InvestigatorPlateMapWell] = []
    for row in plate_map.itertuples(index=False):
        plate_id = _required_string(row.plate_id, "plate_id")
        row_label = _required_string(row.row, "row").upper()
        column = _optional_positive_int(row.column, "column")
        if column is None:
            raise ValueError("column must not be missing")
        coordinate = WellCoordinate(row=row_label, column=column)
        well = _required_string(row.well, "well").upper()
        if coordinate.well != well:
            raise ValueError(f"well {well!r} disagrees with row/column fields")
        role_text = _required_string(row.well_role, "well_role")
        try:
            role = WellRole(role_text)
        except ValueError as error:
            raise ValueError(f"unsupported well role: {role_text!r}") from error

        treatment = _optional_string(row.treatment, "treatment")
        dose = _optional_positive_float(row.dose, "dose")
        dose_unit = _optional_string(row.dose_unit, "dose_unit")
        replicate = _optional_positive_int(row.replicate, "replicate")
        if role is not WellRole.TREATMENT and (
            dose is not None or dose_unit is not None
        ):
            raise ValueError("non-treatment wells must not contain dose fields")

        records.append(
            InvestigatorPlateMapWell(
                plate_id=plate_id,
                well=well,
                row=row_label,
                column=column,
                sample_id=_required_string(row.sample_id, "sample_id"),
                well_role=role,
                treatment=treatment,
                dose=dose,
                dose_unit=dose_unit,
                replicate=replicate,
            )
        )
    return tuple(sorted(records, key=lambda item: (item.plate_id, item.well)))


def _parse_measurements(
    measurements: pd.DataFrame,
    source_case_id: str,
) -> tuple[
    tuple[InvestigatorMeasurement, ...],
    set[tuple[str, str]],
    Counter[str],
    Counter[str],
]:
    _require_exact_columns(measurements, MEASUREMENT_COLUMNS, "measurements")
    if measurements.duplicated(["plate_id", "well"]).any():
        raise ValueError("measurements contain duplicate plate/well keys")

    observed_case_ids = {
        str(value) for value in measurements["case_id"].dropna().unique()
    }
    if observed_case_ids != {source_case_id}:
        raise ValueError("measurement case IDs disagree with public metadata")

    records: list[InvestigatorMeasurement] = []
    keys: set[tuple[str, str]] = set()
    null_counts: Counter[str] = Counter()
    nonfinite_counts: Counter[str] = Counter()
    for row in measurements.itertuples(index=False):
        plate_id = _required_string(row.plate_id, "plate_id")
        well = _required_string(row.well, "well").upper()
        # Pandas stubs expose values from ``itertuples`` as the broad ``Scalar``
        # union, which includes complex numbers. Runtime validation below is the
        # intentional type boundary for CSV-derived values.
        raw_value = cast(Any, row.raw_signal)
        if pd.isna(raw_value):
            signal = None
            null_counts[plate_id] += 1
        else:
            try:
                parsed = float(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError("raw_signal must be numeric when present") from error
            if np.isfinite(parsed):
                signal = parsed
            else:
                signal = None
                nonfinite_counts[plate_id] += 1
        records.append(
            InvestigatorMeasurement(plate_id=plate_id, well=well, raw_signal=signal)
        )
        keys.add((plate_id, well))
    return (
        tuple(sorted(records, key=lambda item: (item.plate_id, item.well))),
        keys,
        null_counts,
        nonfinite_counts,
    )


def _derive_design(
    plate_map: tuple[InvestigatorPlateMapWell, ...],
    measurement_keys: set[tuple[str, str]],
    null_counts: Counter[str],
    nonfinite_counts: Counter[str],
    plate_ids: tuple[str, ...],
) -> DesignSummary:
    wells_by_plate = {
        plate_id: tuple(well for well in plate_map if well.plate_id == plate_id)
        for plate_id in plate_ids
    }
    plate_summaries: list[PlateDesignSummary] = []
    limitations: list[str] = []
    for plate_id in plate_ids:
        wells = wells_by_plate[plate_id]
        plate_format = _infer_plate_format(wells)
        expected = set(enumerate_wells(plate_format)["well"].astype(str))
        observed = {well.well for well in wells}
        missing = tuple(sorted(expected - observed))
        warning = None
        if missing:
            warning = (
                f"{plate_format.value} is inferred as the smallest supported geometry; "
                "missing boundary wells can make that inference ambiguous"
            )
        annotation_count = sum(
            1
            for well in wells
            if well.well_role is WellRole.TREATMENT
            and (
                well.treatment is None
                or well.dose is None
                or well.dose_unit is None
                or well.replicate is None
            )
        )
        role_counts = Counter(well.well_role for well in wells)
        missing_measurements = sum(
            (plate_id, well.well) not in measurement_keys for well in wells
        )
        plate_summaries.append(
            PlateDesignSummary(
                plate_id=plate_id,
                plate_format=plate_format,
                geometry_inference_warning=warning,
                observed_well_count=len(wells),
                expected_well_count=plate_format.capacity,
                missing_wells=missing,
                missing_measurement_count=missing_measurements,
                null_measurement_count=null_counts[plate_id],
                nonfinite_measurement_count=nonfinite_counts[plate_id],
                missing_design_annotation_count=annotation_count,
                role_counts=dict(
                    sorted(role_counts.items(), key=lambda item: item[0].value)
                ),
            )
        )

    condition_counters = {
        plate_id: Counter(_condition_key(well) for well in wells_by_plate[plate_id])
        for plate_id in plate_ids
    }
    all_condition_keys: set[tuple[str, str, str, str]] = set()
    for counter in condition_counters.values():
        all_condition_keys.update(counter)
    all_conditions = sorted(all_condition_keys)
    coverage = tuple(
        ConditionCoverage(
            well_role=WellRole(key[0]),
            treatment=key[1] or None,
            dose=float(key[2]) if key[2] else None,
            dose_unit=key[3] or None,
            counts_by_plate={
                plate_id: condition_counters[plate_id][key] for plate_id in plate_ids
            },
        )
        for key in all_conditions
    )
    complete_replicates: bool | None
    if len(plate_ids) == 1:
        complete_replicates = None
    else:
        first = condition_counters[plate_ids[0]]
        complete_replicates = all(
            condition_counters[plate_id] == first for plate_id in plate_ids[1:]
        )
        if not complete_replicates:
            limitations.append(
                "Cross-plate and pooled analysis is unsupported because plates are "
                "not complete replicates of the same experimental design."
            )

    treatment_series: list[TreatmentDoseSeries] = []
    series_values: dict[tuple[str, str | None], set[float]] = {}
    for well in plate_map:
        if (
            well.well_role is WellRole.TREATMENT
            and well.treatment is not None
            and well.dose is not None
        ):
            series_values.setdefault((well.treatment, well.dose_unit), set()).add(
                well.dose
            )
    for (treatment, unit), doses in sorted(
        series_values.items(), key=lambda item: (item[0][0], item[0][1] or "")
    ):
        treatment_series.append(
            TreatmentDoseSeries(
                treatment=treatment,
                dose_unit=unit,
                doses=tuple(sorted(doses)),
            )
        )

    replicate_sizes = Counter(
        count
        for condition in coverage
        for count in condition.counts_by_plate.values()
        if count > 0
    )
    available_roles = tuple(
        sorted({well.well_role for well in plate_map}, key=lambda role: role.value)
    )
    treatments = tuple(
        sorted(
            {
                well.treatment
                for well in plate_map
                if well.well_role is WellRole.TREATMENT
                and well.treatment is not None
            }
        )
    )
    has_replicates = any(size >= 2 for size in replicate_sizes)
    has_dose_series = any(len(series.doses) >= 2 for series in treatment_series)
    return DesignSummary(
        plate_count=len(plate_ids),
        plates=tuple(plate_summaries),
        available_well_roles=available_roles,
        treatments=treatments,
        treatment_dose_series=tuple(treatment_series),
        condition_coverage=coverage,
        replicate_count_distribution=dict(sorted(replicate_sizes.items())),
        total_missing_well_count=sum(
            len(plate.missing_wells) for plate in plate_summaries
        ),
        total_missing_measurement_count=sum(
            plate.missing_measurement_count for plate in plate_summaries
        ),
        total_null_measurement_count=sum(null_counts.values()),
        total_nonfinite_measurement_count=sum(nonfinite_counts.values()),
        capabilities=DesignCapabilities(
            has_negative_controls=WellRole.NEGATIVE_CONTROL in available_roles,
            has_positive_controls=WellRole.POSITIVE_CONTROL in available_roles,
            has_replicates=has_replicates,
            has_dose_series=has_dose_series,
            has_multiple_plates=len(plate_ids) > 1,
            plates_are_complete_design_replicates=complete_replicates,
            supports_cross_plate_analysis=(
                len(plate_ids) > 1 and complete_replicates is True
            ),
        ),
        limitations=tuple(limitations),
    )


def build_investigator_case(
    *,
    measurements: pd.DataFrame,
    plate_map: pd.DataFrame,
    metadata: CaseMetadata,
    protocol: str,
    problem_statement: str,
    public_artifact_hashes: PublicArtifactHashes,
) -> InvestigatorCase:
    """Build a canonical, ground-truth-free case from public assay inputs."""
    if not protocol.strip():
        raise ValueError("protocol must not be blank")
    if not problem_statement.strip():
        raise ValueError("problem statement must not be blank")

    parsed_map = _parse_plate_map(plate_map)
    parsed_measurements, measurement_keys, null_counts, nonfinite_counts = (
        _parse_measurements(measurements, metadata.case_id)
    )
    map_keys = {(well.plate_id, well.well) for well in parsed_map}
    unexpected_measurements = measurement_keys - map_keys
    if unexpected_measurements:
        raise ValueError("measurements contain wells absent from the plate map")

    metadata_plate_ids = tuple(sorted(plate.plate_id for plate in metadata.plates))
    mapped_plate_ids = tuple(sorted({well.plate_id for well in parsed_map}))
    if metadata_plate_ids != mapped_plate_ids:
        raise ValueError("metadata plate references do not match the plate map")

    investigator_metadata = InvestigatorMetadata(
        assay_type=metadata.assay_type,
        signal_direction=metadata.signal_direction,
        plates=tuple(
            InvestigatorPlateMetadata(
                plate_id=plate.plate_id,
                batch_id=plate.batch_id,
                operator_label=plate.operator_label,
                run_date=plate.run_date,
                instrument_label=plate.instrument_label,
            )
            for plate in sorted(metadata.plates, key=lambda item: item.plate_id)
        ),
    )
    design = _derive_design(
        parsed_map,
        measurement_keys,
        null_counts,
        nonfinite_counts,
        metadata_plate_ids,
    )
    return InvestigatorCase(
        case_id=_opaque_case_id(public_artifact_hashes),
        measurements=parsed_measurements,
        plate_map=parsed_map,
        metadata=investigator_metadata,
        protocol=protocol.strip(),
        problem_statement=problem_statement.strip(),
        public_artifact_hashes=public_artifact_hashes,
        design=design,
    )
