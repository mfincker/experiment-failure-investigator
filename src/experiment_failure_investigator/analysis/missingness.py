"""Deterministic missing-well, measurement, and annotation diagnostics."""

from __future__ import annotations

from collections import defaultdict

from pydantic import JsonValue

from experiment_failure_investigator.analysis.contracts import (
    InvestigatorCase,
    InvestigatorPlateMapWell,
    MeasurementStatus,
)
from experiment_failure_investigator.analysis.results import (
    EvidenceRecord,
    EvidenceScope,
    ResultWarning,
    ScientificProvenance,
    ScientificToolResult,
    ToolStatus,
    build_evidence_record,
)
from experiment_failure_investigator.benchmark.models import WellRole

TOOL_NAME = "inspect_missingness"
TOOL_VERSION = "1.0.0"


def _missing_design_annotation(well: InvestigatorPlateMapWell) -> bool:
    return well.well_role is WellRole.TREATMENT and (
        well.treatment is None
        or well.dose is None
        or well.dose_unit is None
        or well.replicate is None
    )


def _evidence(
    *,
    metric_name: str,
    value: int,
    sample_count: int,
    scope: EvidenceScope,
    description: str,
) -> EvidenceRecord:
    parameters: dict[str, JsonValue] = {}
    return build_evidence_record(
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        parameters=parameters,
        metric_name=metric_name,
        value=value,
        scope=scope,
        sample_count=sample_count,
        description=description,
        unit="wells",
    )


def inspect_missingness(case: InvestigatorCase) -> ScientificToolResult:
    """Count public-input incompleteness by plate and mapped well role."""
    measurements = {
        (measurement.plate_id, measurement.well): measurement
        for measurement in case.measurements
    }
    wells_by_plate: dict[str, list[InvestigatorPlateMapWell]] = defaultdict(list)
    for well in case.plate_map:
        wells_by_plate[well.plate_id].append(well)

    records: list[EvidenceRecord] = []
    concerning_ids: list[str] = []
    plate_ids = tuple(plate.plate_id for plate in case.design.plates)
    for plate_summary in case.design.plates:
        plate_id = plate_summary.plate_id
        plate_scope = EvidenceScope(case_id=case.case_id, plate_ids=(plate_id,))
        assay_wells = [
            well
            for well in wells_by_plate[plate_id]
            if well.well_role is not WellRole.EMPTY
        ]
        assay_measurements = [
            measurements.get((plate_id, well.well)) for well in assay_wells
        ]
        missing_measurement_count = sum(
            measurement is None for measurement in assay_measurements
        )
        null_measurement_count = sum(
            measurement is not None
            and measurement.measurement_status is MeasurementStatus.NULL
            for measurement in assay_measurements
        )
        nonfinite_measurement_count = sum(
            measurement is not None
            and measurement.measurement_status is MeasurementStatus.NONFINITE
            for measurement in assay_measurements
        )
        measurement_row_count = len(assay_wells) - missing_measurement_count
        treatment_well_count = sum(
            well.well_role is WellRole.TREATMENT
            for well in wells_by_plate[plate_id]
        )
        plate_counts = (
            (
                "missingness.expected_well_count",
                plate_summary.expected_well_count,
                plate_summary.expected_well_count,
                "Expected wells for the inferred plate geometry.",
                False,
            ),
            (
                "missingness.mapped_well_count",
                plate_summary.observed_well_count,
                plate_summary.expected_well_count,
                "Wells represented in the public plate map.",
                False,
            ),
            (
                "missingness.unmapped_well_count",
                len(plate_summary.missing_wells),
                plate_summary.expected_well_count,
                "Expected wells absent from the public plate map.",
                True,
            ),
            (
                "missingness.missing_measurement_count",
                missing_measurement_count,
                len(assay_wells),
                "Mapped non-empty wells without a measurement row.",
                True,
            ),
            (
                "missingness.null_measurement_count",
                null_measurement_count,
                measurement_row_count,
                "Non-empty measurement rows containing a null signal.",
                True,
            ),
            (
                "missingness.nonfinite_measurement_count",
                nonfinite_measurement_count,
                measurement_row_count,
                "Non-empty measurement rows containing a non-finite signal.",
                True,
            ),
            (
                "missingness.missing_design_annotation_count",
                plate_summary.missing_design_annotation_count,
                treatment_well_count,
                "Treatment wells missing a required design annotation.",
                True,
            ),
        )
        for metric_name, value, sample_count, description, concerning in plate_counts:
            record = _evidence(
                metric_name=metric_name,
                value=value,
                sample_count=sample_count,
                scope=plate_scope,
                description=description,
            )
            records.append(record)
            if concerning and value > 0:
                concerning_ids.append(record.evidence_id)

        plate_wells = wells_by_plate[plate_id]
        roles = sorted(
            {well.well_role for well in plate_wells},
            key=lambda role: role.value,
        )
        for role in roles:
            role_wells = [well for well in plate_wells if well.well_role is role]
            role_scope = EvidenceScope(
                case_id=case.case_id,
                plate_ids=(plate_id,),
                well_roles=(role,),
            )
            missing_rows = 0
            null_values = 0
            nonfinite_values = 0
            missing_annotations = 0
            for well in role_wells:
                measurement = measurements.get((plate_id, well.well))
                if measurement is None:
                    missing_rows += 1
                elif measurement.measurement_status is MeasurementStatus.NULL:
                    null_values += 1
                elif measurement.measurement_status is MeasurementStatus.NONFINITE:
                    nonfinite_values += 1
                if _missing_design_annotation(well):
                    missing_annotations += 1

            role_counts = (
                (
                    "missingness.role_mapped_well_count",
                    len(role_wells),
                    len(role_wells),
                    f"Mapped {role.value} wells.",
                    False,
                ),
                (
                    "missingness.role_missing_measurement_count",
                    missing_rows,
                    len(role_wells),
                    f"Mapped {role.value} wells without a measurement row.",
                    role is not WellRole.EMPTY,
                ),
                (
                    "missingness.role_null_measurement_count",
                    null_values,
                    len(role_wells) - missing_rows,
                    f"{role.value} measurement rows containing a null signal.",
                    role is not WellRole.EMPTY,
                ),
                (
                    "missingness.role_nonfinite_measurement_count",
                    nonfinite_values,
                    len(role_wells) - missing_rows,
                    f"{role.value} measurement rows containing a non-finite signal.",
                    role is not WellRole.EMPTY,
                ),
                (
                    "missingness.role_missing_design_annotation_count",
                    missing_annotations,
                    len(role_wells),
                    f"{role.value} wells missing a required design annotation.",
                    True,
                ),
            )
            for (
                metric_name,
                value,
                sample_count,
                description,
                concerning,
            ) in role_counts:
                record = _evidence(
                    metric_name=metric_name,
                    value=value,
                    sample_count=sample_count,
                    scope=role_scope,
                    description=description,
                )
                records.append(record)
                if concerning and value > 0:
                    concerning_ids.append(record.evidence_id)

    warnings: tuple[ResultWarning, ...] = ()
    if concerning_ids:
        warnings = (
            ResultWarning(
                code="incomplete_public_input",
                message=(
                    "One or more wells, measurements, values, or required treatment "
                    "annotations are incomplete."
                ),
                evidence_ids=tuple(concerning_ids),
            ),
        )
    parameters: dict[str, JsonValue] = {}
    return ScientificToolResult(
        status=ToolStatus.SUCCESS,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        parameters=parameters,
        scope=EvidenceScope(case_id=case.case_id, plate_ids=plate_ids),
        evidence=tuple(records),
        warnings=warnings,
        limitations=(
            "Missingness counts identify incomplete public inputs but do not determine "
            "why values or annotations are absent.",
        ),
        provenance=ScientificProvenance(
            case_id=case.case_id,
            public_artifact_hashes=case.public_artifact_hashes,
        ),
    )
