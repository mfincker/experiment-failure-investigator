"""Deterministic within-condition replicate variability summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from math import isfinite
from statistics import fmean, median, stdev

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

TOOL_NAME = "calculate_replicate_variability"
TOOL_VERSION = "1.0.0"


def _provenance(case: InvestigatorCase) -> ScientificProvenance:
    return ScientificProvenance(
        case_id=case.case_id,
        public_artifact_hashes=case.public_artifact_hashes,
    )


def _record(
    *,
    metric_name: str,
    value: int | float,
    scope: EvidenceScope,
    parameters: Mapping[str, JsonValue],
    description: str,
    sample_count: int,
    unit: str,
) -> EvidenceRecord:
    if isinstance(value, float) and not isfinite(value):
        raise ArithmeticError(f"{metric_name} produced a non-finite value")
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


def _condition_key(well: InvestigatorPlateMapWell) -> tuple[str, str, float, str]:
    if well.treatment is None or well.dose is None or well.dose_unit is None:
        raise ValueError("treatment condition is missing an identifying annotation")
    return (well.plate_id, well.treatment, well.dose, well.dose_unit)


def _condition_result(
    *,
    case: InvestigatorCase,
    plate_id: str,
    treatment: str,
    dose: float,
    dose_unit: str,
    mapped_wells: tuple[str, ...],
    signals: list[float],
) -> ScientificToolResult:
    parameters: dict[str, JsonValue] = {
        "component": "condition",
        "dispersion": "sample_sd_and_unscaled_mad",
    }
    result_scope = EvidenceScope(case_id=case.case_id, plate_ids=(plate_id,))
    condition_scope = EvidenceScope(
        case_id=case.case_id,
        plate_ids=(plate_id,),
        treatments=(treatment,),
        doses=(dose,),
        dose_unit=dose_unit,
        well_ids=mapped_wells,
    )
    observed_count = len(signals)
    records = [
        _record(
            metric_name="replicates.mapped_well_count",
            value=len(mapped_wells),
            scope=condition_scope,
            parameters=parameters,
            description=(
                f"Mapped wells for {treatment} at {dose:g} {dose_unit} on {plate_id}."
            ),
            sample_count=len(mapped_wells),
            unit="wells",
        ),
        _record(
            metric_name="replicates.observed_signal_count",
            value=observed_count,
            scope=condition_scope,
            parameters=parameters,
            description=(
                f"Finite signals for {treatment} at {dose:g} {dose_unit} on "
                f"{plate_id}."
            ),
            sample_count=observed_count,
            unit="wells",
        ),
    ]
    warnings: tuple[ResultWarning, ...] = ()
    if observed_count < len(mapped_wells):
        warnings = (
            ResultWarning(
                code="missing_replicate_signals",
                message=(
                    f"{len(mapped_wells) - observed_count} mapped replicate wells "
                    "lack a finite signal."
                ),
                evidence_ids=(records[0].evidence_id, records[1].evidence_id),
            ),
        )
    if observed_count == 0:
        return ScientificToolResult(
            status=ToolStatus.INSUFFICIENT_DATA,
            status_reason=(
                "No finite replicate signals are available for this condition."
            ),
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=result_scope,
            evidence=tuple(records),
            warnings=warnings,
            provenance=_provenance(case),
        )

    try:
        mean_value = fmean(signals)
        median_value = median(signals)
        records.extend(
            [
                _record(
                    metric_name="replicates.mean",
                    value=mean_value,
                    scope=condition_scope,
                    parameters=parameters,
                    description="Mean finite signal for this replicate group.",
                    sample_count=observed_count,
                    unit="raw_signal",
                ),
                _record(
                    metric_name="replicates.median",
                    value=median_value,
                    scope=condition_scope,
                    parameters=parameters,
                    description="Median finite signal for this replicate group.",
                    sample_count=observed_count,
                    unit="raw_signal",
                ),
            ]
        )
        if observed_count >= 2:
            records.extend(
                [
                    _record(
                        metric_name="replicates.sample_standard_deviation",
                        value=stdev(signals),
                        scope=condition_scope,
                        parameters=parameters,
                        description="Sample standard deviation within this condition.",
                        sample_count=observed_count,
                        unit="raw_signal",
                    ),
                    _record(
                        metric_name="replicates.median_absolute_deviation",
                        value=median([abs(value - median_value) for value in signals]),
                        scope=condition_scope,
                        parameters=parameters,
                        description=(
                            "Unscaled median absolute deviation within this condition."
                        ),
                        sample_count=observed_count,
                        unit="raw_signal",
                    ),
                ]
            )
    except (ArithmeticError, OverflowError, ValueError) as error:
        return ScientificToolResult(
            status=ToolStatus.ERROR,
            status_reason=f"Replicate variability calculation failed: {error}",
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=result_scope,
            provenance=_provenance(case),
        )

    if observed_count == 1:
        return ScientificToolResult(
            status=ToolStatus.INSUFFICIENT_DATA,
            status_reason=(
                "One finite replicate supports a level estimate but not variability."
            ),
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=result_scope,
            evidence=tuple(records),
            warnings=warnings,
            limitations=(
                "At least two finite replicate signals are required for dispersion.",
            ),
            provenance=_provenance(case),
        )
    return ScientificToolResult(
        status=ToolStatus.SUCCESS,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        parameters=parameters,
        scope=result_scope,
        evidence=tuple(records),
        warnings=warnings,
        provenance=_provenance(case),
    )


def _distribution_result(
    case: InvestigatorCase,
    plate_id: str,
    condition_results: tuple[ScientificToolResult, ...],
) -> ScientificToolResult:
    parameters: dict[str, JsonValue] = {
        "component": "plate_distribution",
        "dispersion": "sample_sd_and_unscaled_mad",
    }
    scope = EvidenceScope(case_id=case.case_id, plate_ids=(plate_id,))
    successful = [
        result for result in condition_results if result.status is ToolStatus.SUCCESS
    ]
    count_record = _record(
        metric_name="replicates.conditions_with_variability_count",
        value=len(successful),
        scope=scope,
        parameters=parameters,
        description="Conditions with at least two finite replicate signals.",
        sample_count=len(condition_results),
        unit="conditions",
    )
    if not successful:
        return ScientificToolResult(
            status=ToolStatus.INSUFFICIENT_DATA,
            status_reason=(
                f"No treatment conditions on {plate_id} have enough finite "
                "replicates for a variability distribution."
            ),
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=scope,
            evidence=(count_record,),
            provenance=_provenance(case),
        )

    sample_sds = [
        float(evidence.value)
        for result in successful
        for evidence in result.evidence
        if evidence.metric_name == "replicates.sample_standard_deviation"
        and isinstance(evidence.value, int | float)
    ]
    mads = [
        float(evidence.value)
        for result in successful
        for evidence in result.evidence
        if evidence.metric_name == "replicates.median_absolute_deviation"
        and isinstance(evidence.value, int | float)
    ]
    records = [count_record]
    for metric_prefix, values, label in (
        ("sample_standard_deviation", sample_sds, "sample standard deviation"),
        ("median_absolute_deviation", mads, "unscaled median absolute deviation"),
    ):
        for suffix, value in (
            ("minimum", min(values)),
            ("median", median(values)),
            ("maximum", max(values)),
        ):
            records.append(
                _record(
                    metric_name=f"replicates.{metric_prefix}_{suffix}",
                    value=value,
                    scope=scope,
                    parameters=parameters,
                    description=f"{suffix.capitalize()} within-condition {label}.",
                    sample_count=len(values),
                    unit="raw_signal",
                )
            )
    return ScientificToolResult(
        status=ToolStatus.SUCCESS,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        parameters=parameters,
        scope=scope,
        evidence=tuple(records),
        provenance=_provenance(case),
    )


def calculate_replicate_variability(
    case: InvestigatorCase,
) -> tuple[ScientificToolResult, ...]:
    """Summarize treatment replicates per observed plate, treatment, and dose."""
    measurement_lookup = {
        (measurement.plate_id, measurement.well): measurement
        for measurement in case.measurements
    }
    grouped_wells: dict[tuple[str, str, float, str], list[str]] = defaultdict(list)
    for well in case.plate_map:
        if (
            well.well_role is WellRole.TREATMENT
            and well.treatment is not None
            and well.dose is not None
            and well.dose_unit is not None
        ):
            grouped_wells[_condition_key(well)].append(well.well)

    results_by_plate: dict[str, list[ScientificToolResult]] = defaultdict(list)
    for (plate_id, treatment, dose, dose_unit), wells in sorted(grouped_wells.items()):
        mapped_wells = tuple(sorted(wells))
        signals = [
            measurement.raw_signal
            for well in mapped_wells
            if (measurement := measurement_lookup.get((plate_id, well))) is not None
            and measurement.measurement_status is MeasurementStatus.OBSERVED
            and measurement.raw_signal is not None
        ]
        results_by_plate[plate_id].append(
            _condition_result(
                case=case,
                plate_id=plate_id,
                treatment=treatment,
                dose=dose,
                dose_unit=dose_unit,
                mapped_wells=mapped_wells,
                signals=signals,
            )
        )

    results: list[ScientificToolResult] = []
    for plate in case.design.plates:
        condition_results = tuple(results_by_plate[plate.plate_id])
        if not condition_results:
            parameters: dict[str, JsonValue] = {"component": "condition"}
            results.append(
                ScientificToolResult(
                    status=ToolStatus.NOT_APPLICABLE,
                    status_reason=(
                        f"Plate {plate.plate_id} has no annotated treatments."
                    ),
                    tool_name=TOOL_NAME,
                    tool_version=TOOL_VERSION,
                    parameters=parameters,
                    scope=EvidenceScope(
                        case_id=case.case_id,
                        plate_ids=(plate.plate_id,),
                    ),
                    provenance=_provenance(case),
                )
            )
            continue
        results.extend(condition_results)
        results.append(_distribution_result(case, plate.plate_id, condition_results))
    return tuple(results)
