"""Geometry-aware spatial diagnostics using public condition residuals."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from math import isfinite, sqrt
from statistics import fmean, median, stdev
from typing import Any

from pydantic import Field, JsonValue

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
from experiment_failure_investigator.benchmark.models import StrictModel, WellRole

TOOL_NAME = "detect_spatial_effects"
TOOL_VERSION = "1.0.0"


class ResidualObservation(StrictModel):
    """One finite public measurement centered within its observed condition."""

    plate_id: str
    well: str
    row: str
    column: int
    well_role: WellRole
    treatment: str | None = None
    dose: float | None = None
    dose_unit: str | None = None
    raw_signal: float = Field(allow_inf_nan=False)
    condition_center: float = Field(allow_inf_nan=False)
    condition_residual: float = Field(allow_inf_nan=False)
    is_edge: bool


def _condition_key(well: InvestigatorPlateMapWell) -> tuple[str, str, str, str]:
    if well.well_role is WellRole.TREATMENT:
        if well.treatment is None or well.dose is None or well.dose_unit is None:
            raise ValueError("treatment condition lacks a complete public identity")
        return (
            well.well_role.value,
            well.treatment,
            format(well.dose, ".12g"),
            well.dose_unit,
        )
    return (well.well_role.value, "", "", "")


def condition_centered_residuals(
    case: InvestigatorCase,
    *,
    minimum_group_size: int = 2,
) -> tuple[ResidualObservation, ...]:
    """Return median-centered residuals for finite, identifiable conditions."""
    if minimum_group_size < 2:
        raise ValueError("minimum_group_size must be at least 2")
    measurement_lookup = {
        (measurement.plate_id, measurement.well): measurement
        for measurement in case.measurements
    }
    plate_formats = {
        plate.plate_id: plate.plate_format for plate in case.design.plates
    }
    grouped: dict[
        tuple[str, tuple[str, str, str, str]],
        list[tuple[InvestigatorPlateMapWell, float]],
    ] = defaultdict(list)
    for well in case.plate_map:
        if well.well_role is WellRole.EMPTY:
            continue
        measurement = measurement_lookup.get((well.plate_id, well.well))
        if (
            measurement is None
            or measurement.measurement_status is not MeasurementStatus.OBSERVED
            or measurement.raw_signal is None
        ):
            continue
        try:
            key = _condition_key(well)
        except ValueError:
            continue
        grouped[(well.plate_id, key)].append((well, measurement.raw_signal))

    observations: list[ResidualObservation] = []
    for (plate_id, _), group in sorted(grouped.items()):
        if len(group) < minimum_group_size:
            continue
        center = float(median(signal for _, signal in group))
        plate_format = plate_formats[plate_id]
        last_row = chr(ord("A") + plate_format.rows - 1)
        for well, signal in group:
            observations.append(
                ResidualObservation(
                    plate_id=plate_id,
                    well=well.well,
                    row=well.row,
                    column=well.column,
                    well_role=well.well_role,
                    treatment=well.treatment,
                    dose=well.dose,
                    dose_unit=well.dose_unit,
                    raw_signal=signal,
                    condition_center=center,
                    condition_residual=signal - center,
                    is_edge=(
                        well.row in {"A", last_row}
                        or well.column in {1, plate_format.columns}
                    ),
                )
            )
    return tuple(sorted(observations, key=lambda item: (item.plate_id, item.well)))


def _provenance(case: InvestigatorCase) -> ScientificProvenance:
    return ScientificProvenance(
        case_id=case.case_id,
        public_artifact_hashes=case.public_artifact_hashes,
    )


def _record(
    *,
    metric_name: str,
    value: Any,
    scope: EvidenceScope,
    parameters: Mapping[str, JsonValue],
    description: str,
    sample_count: int,
    unit: str | None = None,
) -> EvidenceRecord:
    return build_evidence_record(
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        parameters=parameters,
        metric_name=metric_name,
        value=value,
        unit=unit,
        scope=scope,
        sample_count=sample_count,
        description=description,
    )


def _linear_trend(
    coordinates: list[float],
    residuals: list[float],
    denominator_epsilon: float,
) -> tuple[float, float] | None:
    if len(coordinates) < 2 or len(set(coordinates)) < 2:
        return None
    coordinate_mean = fmean(coordinates)
    residual_mean = fmean(residuals)
    centered_coordinates = [value - coordinate_mean for value in coordinates]
    centered_residuals = [value - residual_mean for value in residuals]
    coordinate_sum_squares = sum(value * value for value in centered_coordinates)
    residual_sum_squares = sum(value * value for value in centered_residuals)
    if coordinate_sum_squares <= denominator_epsilon:
        return None
    cross_product = sum(
        coordinate * residual
        for coordinate, residual in zip(
            centered_coordinates,
            centered_residuals,
            strict=True,
        )
    )
    slope = cross_product / coordinate_sum_squares
    correlation_denominator = sqrt(
        coordinate_sum_squares * residual_sum_squares
    )
    correlation = (
        0.0
        if correlation_denominator <= denominator_epsilon
        else cross_product / correlation_denominator
    )
    if not isfinite(slope) or not isfinite(correlation):
        return None
    return slope, correlation


def _pooled_standardized_difference(
    first: list[float],
    second: list[float],
    denominator_epsilon: float,
) -> float | None:
    if len(first) < 2 or len(second) < 2:
        return None
    degrees_of_freedom = len(first) + len(second) - 2
    pooled_variance = (
        (len(first) - 1) * stdev(first) ** 2
        + (len(second) - 1) * stdev(second) ** 2
    ) / degrees_of_freedom
    if pooled_variance <= denominator_epsilon:
        return None
    value = (fmean(first) - fmean(second)) / sqrt(pooled_variance)
    return value if isfinite(value) else None


def _adjacent_components(
    observations: Iterable[ResidualObservation],
) -> list[tuple[str, ...]]:
    by_coordinate = {(item.row, item.column): item for item in observations}
    remaining = set(by_coordinate)
    components: list[tuple[str, ...]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        sign = by_coordinate[start].condition_residual >= 0
        stack = [start]
        wells: list[str] = []
        while stack:
            coordinate = stack.pop()
            wells.append(by_coordinate[coordinate].well)
            row, column = coordinate
            row_index = ord(row) - ord("A")
            neighbors = (
                (chr(ord("A") + row_index - 1), column)
                if row_index > 0
                else None,
                (chr(ord("A") + row_index + 1), column),
                (row, column - 1) if column > 1 else None,
                (row, column + 1),
            )
            for neighbor in neighbors:
                if (
                    neighbor is not None
                    and neighbor in remaining
                    and (by_coordinate[neighbor].condition_residual >= 0) is sign
                ):
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(wells)))
    return sorted(components, key=lambda component: (-len(component), component))


def _coverage_by_factor(
    wells: list[InvestigatorPlateMapWell],
    row_count: int,
    column_count: int,
) -> dict[str, JsonValue]:
    last_row = chr(ord("A") + row_count - 1)

    def coverage(groups: Mapping[str, list[InvestigatorPlateMapWell]]) -> JsonValue:
        values: dict[str, JsonValue] = {}
        for label, group in sorted(groups.items()):
            regions = {
                (
                    "edge"
                    if well.row in {"A", last_row}
                    or well.column in {1, column_count}
                    else "interior"
                )
                for well in group
            }
            values[label] = {
                "column_fraction": len({well.column for well in group}) / column_count,
                "region_fraction": len(regions) / 2,
                "row_fraction": len({well.row for well in group}) / row_count,
                "well_count": len(group),
            }
        return values

    roles: dict[str, list[InvestigatorPlateMapWell]] = defaultdict(list)
    treatments: dict[str, list[InvestigatorPlateMapWell]] = defaultdict(list)
    doses: dict[str, list[InvestigatorPlateMapWell]] = defaultdict(list)
    for well in wells:
        roles[well.well_role.value].append(well)
        if well.well_role is WellRole.TREATMENT and well.treatment is not None:
            treatments[well.treatment].append(well)
        if (
            well.well_role is WellRole.TREATMENT
            and well.dose is not None
            and well.dose_unit is not None
        ):
            doses[f"{well.dose:g} {well.dose_unit}"].append(well)
    return {
        "dose": coverage(doses),
        "treatment": coverage(treatments),
        "well_role": coverage(roles),
    }


def _condition_coverage(
    wells: list[InvestigatorPlateMapWell],
    row_count: int,
    column_count: int,
) -> tuple[dict[str, JsonValue], int, int, int]:
    last_row = chr(ord("A") + row_count - 1)
    groups: dict[tuple[str, str, str, str], list[InvestigatorPlateMapWell]] = (
        defaultdict(list)
    )
    for well in wells:
        if well.well_role is WellRole.EMPTY:
            continue
        try:
            groups[_condition_key(well)].append(well)
        except ValueError:
            continue
    coverage: dict[str, JsonValue] = {}
    single_row_count = 0
    single_column_count = 0
    single_region_count = 0
    for index, (key, group) in enumerate(sorted(groups.items()), start=1):
        rows = {well.row for well in group}
        columns = {well.column for well in group}
        regions = {
            (
                "edge"
                if well.row in {"A", last_row}
                or well.column in {1, column_count}
                else "interior"
            )
            for well in group
        }
        single_row_count += len(rows) == 1
        single_column_count += len(columns) == 1
        single_region_count += len(regions) == 1
        coverage[f"condition_{index:02d}"] = {
            "column_count": len(columns),
            "dose": float(key[2]) if key[2] else None,
            "dose_unit": key[3] or None,
            "region_count": len(regions),
            "row_count": len(rows),
            "treatment": key[1] or None,
            "well_count": len(group),
            "well_role": key[0],
        }
    return coverage, single_row_count, single_column_count, single_region_count


def _detect_plate(
    case: InvestigatorCase,
    plate_id: str,
    residuals: tuple[ResidualObservation, ...],
    *,
    robust_z_threshold: float,
    denominator_epsilon: float,
    minimum_group_size: int,
) -> ScientificToolResult:
    parameters: dict[str, JsonValue] = {
        "condition_center": "median",
        "denominator_epsilon": denominator_epsilon,
        "extreme_residual_method": "absolute_robust_z",
        "minimum_group_size": minimum_group_size,
        "robust_scale": "1.4826_times_mad_with_sample_sd_fallback",
        "robust_z_threshold": robust_z_threshold,
    }
    scope = EvidenceScope(case_id=case.case_id, plate_ids=(plate_id,))
    plate_summary = next(plate for plate in case.design.plates if plate.plate_id == plate_id)
    plate_wells = [well for well in case.plate_map if well.plate_id == plate_id]
    factor_coverage = _coverage_by_factor(
        plate_wells,
        plate_summary.plate_format.rows,
        plate_summary.plate_format.columns,
    )
    condition_coverage, single_rows, single_columns, single_regions = (
        _condition_coverage(
            plate_wells,
            plate_summary.plate_format.rows,
            plate_summary.plate_format.columns,
        )
    )
    records = [
        _record(
            metric_name="spatial.factor_coverage",
            value=factor_coverage,
            scope=scope,
            parameters=parameters,
            description=(
                "Fractions of rows, columns, and edge/interior regions covered by "
                "each well role, treatment, and dose level."
            ),
            sample_count=len(plate_wells),
        ),
        _record(
            metric_name="spatial.condition_coverage",
            value=condition_coverage,
            scope=scope,
            parameters=parameters,
            description="Spatial coverage for each observed experimental condition.",
            sample_count=len(condition_coverage),
        ),
        _record(
            metric_name="spatial.single_row_condition_count",
            value=single_rows,
            scope=scope,
            parameters=parameters,
            description="Conditions represented in only one plate row.",
            sample_count=len(condition_coverage),
            unit="conditions",
        ),
        _record(
            metric_name="spatial.single_column_condition_count",
            value=single_columns,
            scope=scope,
            parameters=parameters,
            description="Conditions represented in only one plate column.",
            sample_count=len(condition_coverage),
            unit="conditions",
        ),
        _record(
            metric_name="spatial.single_region_condition_count",
            value=single_regions,
            scope=scope,
            parameters=parameters,
            description="Conditions represented only at the edge or only in the interior.",
            sample_count=len(condition_coverage),
            unit="conditions",
        ),
    ]
    result_warnings: list[ResultWarning] = []
    if single_rows or single_columns or single_regions:
        coverage_ids = tuple(record.evidence_id for record in records[1:5])
        result_warnings.append(
            ResultWarning(
                code="condition_position_nonidentifiability",
                message=(
                    "At least one condition occurs in only one spatial stratum; "
                    "condition and position effects are not separately identifiable "
                    "for those comparisons."
                ),
                evidence_ids=coverage_ids,
            )
        )

    plate_residuals = [item for item in residuals if item.plate_id == plate_id]
    records.append(
        _record(
            metric_name="spatial.residual_count",
            value=len(plate_residuals),
            scope=scope,
            parameters=parameters,
            description="Finite observations with an estimable condition median.",
            sample_count=len(plate_wells),
            unit="wells",
        )
    )
    if len(plate_residuals) < 2:
        return ScientificToolResult(
            status=ToolStatus.INSUFFICIENT_DATA,
            status_reason=(
                f"Plate {plate_id} has fewer than two condition-centered residuals."
            ),
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=scope,
            evidence=tuple(records),
            warnings=tuple(result_warnings),
            provenance=_provenance(case),
        )

    values = [item.condition_residual for item in plate_residuals]
    edge = [item.condition_residual for item in plate_residuals if item.is_edge]
    interior = [item.condition_residual for item in plate_residuals if not item.is_edge]
    if edge and interior:
        edge_mean = fmean(edge)
        interior_mean = fmean(interior)
        records.extend(
            [
                _record(
                    metric_name="spatial.edge_residual_mean",
                    value=edge_mean,
                    scope=scope,
                    parameters=parameters,
                    description="Mean condition-centered residual among edge wells.",
                    sample_count=len(edge),
                    unit="raw_signal",
                ),
                _record(
                    metric_name="spatial.interior_residual_mean",
                    value=interior_mean,
                    scope=scope,
                    parameters=parameters,
                    description="Mean condition-centered residual among interior wells.",
                    sample_count=len(interior),
                    unit="raw_signal",
                ),
                _record(
                    metric_name="spatial.edge_minus_interior",
                    value=edge_mean - interior_mean,
                    scope=scope,
                    parameters=parameters,
                    description="Edge mean residual minus interior mean residual.",
                    sample_count=len(edge) + len(interior),
                    unit="raw_signal",
                ),
            ]
        )
        standardized = _pooled_standardized_difference(
            edge,
            interior,
            denominator_epsilon,
        )
        if standardized is not None:
            records.append(
                _record(
                    metric_name="spatial.edge_standardized_mean_difference",
                    value=standardized,
                    scope=scope,
                    parameters=parameters,
                    description=(
                        "Edge-minus-interior difference divided by pooled within-region "
                        "sample standard deviation."
                    ),
                    sample_count=len(edge) + len(interior),
                    unit="dimensionless",
                )
            )
    else:
        result_warnings.append(
            ResultWarning(
                code="edge_comparison_not_estimable",
                message="Both edge and interior residuals are required for comparison.",
            )
        )

    for axis, coordinates in (
        ("row", [float(ord(item.row) - ord("A")) for item in plate_residuals]),
        ("column", [float(item.column - 1) for item in plate_residuals]),
    ):
        trend = _linear_trend(coordinates, values, denominator_epsilon)
        if trend is None:
            result_warnings.append(
                ResultWarning(
                    code=f"{axis}_trend_not_estimable",
                    message=f"The {axis} residual trend cannot be estimated.",
                )
            )
            continue
        slope, correlation = trend
        records.extend(
            [
                _record(
                    metric_name=f"spatial.{axis}_slope",
                    value=slope,
                    scope=scope,
                    parameters=parameters,
                    description=f"Linear residual slope per {axis} position.",
                    sample_count=len(values),
                    unit=f"raw_signal_per_{axis}",
                ),
                _record(
                    metric_name=f"spatial.{axis}_correlation",
                    value=correlation,
                    scope=scope,
                    parameters=parameters,
                    description=f"Pearson correlation of residual with {axis} position.",
                    sample_count=len(values),
                    unit="dimensionless",
                ),
            ]
        )

    residual_center = median(values)
    mad = median(abs(value - residual_center) for value in values)
    robust_scale = 1.4826 * mad
    scale_method = "scaled_mad"
    if robust_scale <= denominator_epsilon:
        robust_scale = stdev(values)
        scale_method = "sample_standard_deviation_fallback"
    extreme: list[ResidualObservation] = []
    if robust_scale > denominator_epsilon and isfinite(robust_scale):
        extreme = [
            item
            for item in plate_residuals
            if abs(item.condition_residual - residual_center) / robust_scale
            >= robust_z_threshold
        ]
    else:
        result_warnings.append(
            ResultWarning(
                code="residual_scale_not_estimable",
                message="Residual dispersion is zero; extreme residuals are not scored.",
            )
        )
    components = _adjacent_components(extreme)
    largest_component = components[0] if components else ()
    localization_values = (
        (
            "spatial.residual_scale",
            robust_scale,
            f"Residual scale estimated by {scale_method}.",
            "raw_signal",
        ),
        (
            "spatial.extreme_well_count",
            len(extreme),
            "Wells exceeding the configured absolute robust-z threshold.",
            "wells",
        ),
        (
            "spatial.extreme_well_ids",
            sorted(item.well for item in extreme),
            "Canonical well IDs exceeding the robust-z threshold.",
            None,
        ),
        (
            "spatial.adjacent_extreme_component_count",
            len(components),
            "Orthogonally adjacent same-sign extreme-residual components.",
            "components",
        ),
        (
            "spatial.largest_adjacent_extreme_run_size",
            len(largest_component),
            "Wells in the largest adjacent same-sign extreme-residual component.",
            "wells",
        ),
        (
            "spatial.largest_adjacent_extreme_run_well_ids",
            list(largest_component),
            "Canonical well IDs in the largest adjacent extreme component.",
            None,
        ),
    )
    for metric_name, value, description, unit in localization_values:
        records.append(
            _record(
                metric_name=metric_name,
                value=value,
                scope=scope,
                parameters=parameters,
                description=description,
                sample_count=len(values),
                unit=unit,
            )
        )
    return ScientificToolResult(
        status=ToolStatus.SUCCESS,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        parameters=parameters,
        scope=scope,
        evidence=tuple(records),
        warnings=tuple(result_warnings),
        limitations=(
            "Spatial association does not identify a physical cause or reconstruct "
            "unavailable dispense order.",
            "Localized adjacent residuals do not establish tip identity, clogging, or "
            "reload timing.",
        ),
        provenance=_provenance(case),
    )


def detect_spatial_effects(
    case: InvestigatorCase,
    *,
    robust_z_threshold: float = 3.5,
    denominator_epsilon: float = 1e-12,
    minimum_group_size: int = 2,
) -> tuple[ScientificToolResult, ...]:
    """Measure residual spatial structure independently for every observed plate."""
    if not isfinite(robust_z_threshold) or robust_z_threshold <= 0:
        raise ValueError("robust_z_threshold must be finite and positive")
    if not isfinite(denominator_epsilon) or denominator_epsilon <= 0:
        raise ValueError("denominator_epsilon must be finite and positive")
    residuals = condition_centered_residuals(
        case,
        minimum_group_size=minimum_group_size,
    )
    return tuple(
        _detect_plate(
            case,
            plate.plate_id,
            residuals,
            robust_z_threshold=robust_z_threshold,
            denominator_epsilon=denominator_epsilon,
            minimum_group_size=minimum_group_size,
        )
        for plate in case.design.plates
    )
