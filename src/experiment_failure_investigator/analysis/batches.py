"""Matched, plate-specific comparisons for complete replicate designs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from itertools import combinations
from math import isfinite, sqrt
from statistics import fmean, stdev
from typing import Any

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

TOOL_NAME = "compare_batches"
TOOL_VERSION = "1.0.0"
ConditionKey = tuple[WellRole, str | None, float | None, str | None]


def _provenance(case: InvestigatorCase) -> ScientificProvenance:
    return ScientificProvenance(
        case_id=case.case_id,
        public_artifact_hashes=case.public_artifact_hashes,
    )


def _condition_key(well: InvestigatorPlateMapWell) -> ConditionKey | None:
    if well.well_role is WellRole.EMPTY:
        return None
    if well.well_role is WellRole.TREATMENT:
        if well.treatment is None or well.dose is None or well.dose_unit is None:
            return None
        return (well.well_role, well.treatment, well.dose, well.dose_unit)
    return (well.well_role, None, None, None)


def _condition_scope(
    case_id: str,
    plate_ids: tuple[str, str],
    key: ConditionKey,
) -> EvidenceScope:
    role, treatment, dose, dose_unit = key
    return EvidenceScope(
        case_id=case_id,
        plate_ids=plate_ids,
        well_roles=(role,),
        treatments=() if treatment is None else (treatment,),
        doses=() if dose is None else (dose,),
        dose_unit=dose_unit,
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


def _observed_conditions(
    case: InvestigatorCase,
) -> dict[str, dict[ConditionKey, list[float]]]:
    measurement_lookup = {
        (measurement.plate_id, measurement.well): measurement
        for measurement in case.measurements
    }
    grouped: dict[str, dict[ConditionKey, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for well in case.plate_map:
        key = _condition_key(well)
        if key is None:
            continue
        measurement = measurement_lookup.get((well.plate_id, well.well))
        if (
            measurement is not None
            and measurement.measurement_status is MeasurementStatus.OBSERVED
            and measurement.raw_signal is not None
        ):
            grouped[well.plate_id][key].append(measurement.raw_signal)
    return grouped


def _comparison_result(
    *,
    case: InvestigatorCase,
    first_plate_id: str,
    second_plate_id: str,
    first_batch_id: str,
    second_batch_id: str,
    grouped: dict[str, dict[ConditionKey, list[float]]],
    denominator_epsilon: float,
) -> ScientificToolResult:
    parameters: dict[str, JsonValue] = {
        "component": "matched_plate_comparison",
        "denominator_epsilon": denominator_epsilon,
        "first_batch_id": first_batch_id,
        "first_plate_id": first_plate_id,
        "second_batch_id": second_batch_id,
        "second_plate_id": second_plate_id,
    }
    plate_ids = (first_plate_id, second_plate_id)
    scope = EvidenceScope(case_id=case.case_id, plate_ids=plate_ids)
    first = grouped.get(first_plate_id, {})
    second = grouped.get(second_plate_id, {})
    matched_keys = sorted(
        set(first) & set(second),
        key=lambda key: (
            key[0].value,
            key[1] or "",
            -1.0 if key[2] is None else key[2],
            key[3] or "",
        ),
    )
    if not matched_keys:
        return ScientificToolResult(
            status=ToolStatus.INSUFFICIENT_DATA,
            status_reason=(
                f"{first_plate_id} and {second_plate_id} have no conditions with "
                "finite observations on both plates."
            ),
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=scope,
            provenance=_provenance(case),
        )

    records: list[EvidenceRecord] = []
    warnings: list[ResultWarning] = []
    first_means: dict[ConditionKey, float] = {}
    second_means: dict[ConditionKey, float] = {}
    for key in matched_keys:
        first_values = first[key]
        second_values = second[key]
        first_mean = fmean(first_values)
        second_mean = fmean(second_values)
        first_means[key] = first_mean
        second_means[key] = second_mean
        condition_scope = _condition_scope(case.case_id, plate_ids, key)
        records.extend(
            [
                _record(
                    metric_name="batches.first_plate_condition_mean",
                    value=first_mean,
                    scope=condition_scope,
                    parameters=parameters,
                    description=f"Condition mean on {first_plate_id}.",
                    sample_count=len(first_values),
                    unit="raw_signal",
                ),
                _record(
                    metric_name="batches.second_plate_condition_mean",
                    value=second_mean,
                    scope=condition_scope,
                    parameters=parameters,
                    description=f"Condition mean on {second_plate_id}.",
                    sample_count=len(second_values),
                    unit="raw_signal",
                ),
                _record(
                    metric_name="batches.condition_mean_difference",
                    value=second_mean - first_mean,
                    scope=condition_scope,
                    parameters=parameters,
                    description=(
                        f"Condition mean on {second_plate_id} minus its matched mean "
                        f"on {first_plate_id}."
                    ),
                    sample_count=len(first_values) + len(second_values),
                    unit="raw_signal",
                ),
            ]
        )
        if len(first_values) >= 2 and len(second_values) >= 2:
            standard_error = sqrt(
                stdev(first_values) ** 2 / len(first_values)
                + stdev(second_values) ** 2 / len(second_values)
            )
            records.append(
                _record(
                    metric_name="batches.condition_difference_standard_error",
                    value=standard_error,
                    scope=condition_scope,
                    parameters=parameters,
                    description="Standard error of the matched condition difference.",
                    sample_count=len(first_values) + len(second_values),
                    unit="raw_signal",
                )
            )

    unmatched = (set(first) | set(second)) - set(matched_keys)
    if unmatched:
        warnings.append(
            ResultWarning(
                code="unmatched_observed_conditions",
                message=(
                    f"{len(unmatched)} conditions lack finite observations on one of "
                    "the compared plates and were not compared."
                ),
            )
        )

    negative_key: ConditionKey = (WellRole.NEGATIVE_CONTROL, None, None, None)
    positive_key: ConditionKey = (WellRole.POSITIVE_CONTROL, None, None, None)
    if negative_key in first_means:
        negative_scope = _condition_scope(case.case_id, plate_ids, negative_key)
        records.append(
            _record(
                metric_name="batches.negative_control_anchor_difference",
                value=second_means[negative_key] - first_means[negative_key],
                scope=negative_scope,
                parameters=parameters,
                description=(
                    f"Negative-control mean on {second_plate_id} minus the mean on "
                    f"{first_plate_id}."
                ),
                sample_count=len(first[negative_key]) + len(second[negative_key]),
                unit="raw_signal",
            )
        )

    if negative_key in first_means and positive_key in first_means:
        first_window = abs(first_means[negative_key] - first_means[positive_key])
        second_window = abs(second_means[negative_key] - second_means[positive_key])
        records.extend(
            [
                _record(
                    metric_name="batches.first_plate_control_window",
                    value=first_window,
                    scope=scope,
                    parameters=parameters,
                    description=f"Absolute control-mean separation on {first_plate_id}.",
                    sample_count=len(first[negative_key]) + len(first[positive_key]),
                    unit="raw_signal",
                ),
                _record(
                    metric_name="batches.second_plate_control_window",
                    value=second_window,
                    scope=scope,
                    parameters=parameters,
                    description=f"Absolute control-mean separation on {second_plate_id}.",
                    sample_count=len(second[negative_key]) + len(second[positive_key]),
                    unit="raw_signal",
                ),
            ]
        )
        if first_window > denominator_epsilon:
            records.append(
                _record(
                    metric_name="batches.control_window_ratio",
                    value=second_window / first_window,
                    scope=scope,
                    parameters=parameters,
                    description=(
                        f"Control window on {second_plate_id} divided by the window "
                        f"on {first_plate_id}."
                    ),
                    sample_count=(
                        len(first[negative_key])
                        + len(first[positive_key])
                        + len(second[negative_key])
                        + len(second[positive_key])
                    ),
                    unit="dimensionless",
                )
            )
        else:
            warnings.append(
                ResultWarning(
                    code="control_window_ratio_not_estimable",
                    message=(
                        f"The control window on {first_plate_id} is zero or near zero."
                    ),
                )
            )
    else:
        warnings.append(
            ResultWarning(
                code="control_window_not_estimable",
                message="Matched positive and negative controls are required.",
            )
        )

    first_range = max(first_means.values()) - min(first_means.values())
    second_range = max(second_means.values()) - min(second_means.values())
    records.extend(
        [
            _record(
                metric_name="batches.first_plate_response_range",
                value=first_range,
                scope=scope,
                parameters=parameters,
                description=f"Range of matched condition means on {first_plate_id}.",
                sample_count=len(matched_keys),
                unit="raw_signal",
            ),
            _record(
                metric_name="batches.second_plate_response_range",
                value=second_range,
                scope=scope,
                parameters=parameters,
                description=f"Range of matched condition means on {second_plate_id}.",
                sample_count=len(matched_keys),
                unit="raw_signal",
            ),
        ]
    )
    if first_range > denominator_epsilon:
        records.append(
            _record(
                metric_name="batches.response_range_ratio",
                value=second_range / first_range,
                scope=scope,
                parameters=parameters,
                description=(
                    f"Matched-condition response range on {second_plate_id} divided "
                    f"by the range on {first_plate_id}."
                ),
                sample_count=len(matched_keys),
                unit="dimensionless",
            )
        )
    else:
        warnings.append(
            ResultWarning(
                code="response_range_ratio_not_estimable",
                message=(
                    f"The matched-condition range on {first_plate_id} is zero or "
                    "near zero."
                ),
            )
        )
    return ScientificToolResult(
        status=ToolStatus.SUCCESS,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        parameters=parameters,
        scope=scope,
        evidence=tuple(records),
        warnings=tuple(warnings),
        limitations=(
            "Matched plate differences do not identify whether batch, operator, "
            "reagent, instrument, or another plate-level factor caused the change.",
            "Comparisons assume public plate and batch metadata are trustworthy.",
        ),
        provenance=_provenance(case),
    )


def compare_batches(
    case: InvestigatorCase,
    *,
    denominator_epsilon: float = 1e-12,
) -> tuple[ScientificToolResult, ...]:
    """Compare complete replicate plates only through matched conditions."""
    if not isfinite(denominator_epsilon) or denominator_epsilon <= 0:
        raise ValueError("denominator_epsilon must be finite and positive")
    plate_metadata = {
        plate.plate_id: plate for plate in case.metadata.plates
    }
    plate_ids = tuple(sorted(plate_metadata))
    base_parameters: dict[str, JsonValue] = {"component": "design_eligibility"}
    if len(plate_ids) < 2:
        return (
            ScientificToolResult(
                status=ToolStatus.NOT_APPLICABLE,
                status_reason="Batch comparison requires at least two plates.",
                tool_name=TOOL_NAME,
                tool_version=TOOL_VERSION,
                parameters=base_parameters,
                scope=EvidenceScope(case_id=case.case_id, plate_ids=plate_ids),
                provenance=_provenance(case),
            ),
        )
    if not case.design.capabilities.supports_cross_plate_analysis:
        return (
            ScientificToolResult(
                status=ToolStatus.INSUFFICIENT_DATA,
                status_reason=(
                    "Cross-plate comparison is unsupported because plates are not "
                    "complete replicates of the same experimental design."
                ),
                tool_name=TOOL_NAME,
                tool_version=TOOL_VERSION,
                parameters=base_parameters,
                scope=EvidenceScope(case_id=case.case_id, plate_ids=plate_ids),
                limitations=case.design.limitations,
                provenance=_provenance(case),
            ),
        )

    grouped = _observed_conditions(case)
    results = [
        _comparison_result(
            case=case,
            first_plate_id=first,
            second_plate_id=second,
            first_batch_id=plate_metadata[first].batch_id,
            second_batch_id=plate_metadata[second].batch_id,
            grouped=grouped,
            denominator_epsilon=denominator_epsilon,
        )
        for first, second in combinations(plate_ids, 2)
        if plate_metadata[first].batch_id != plate_metadata[second].batch_id
    ]
    if results:
        return tuple(results)
    return (
        ScientificToolResult(
            status=ToolStatus.NOT_APPLICABLE,
            status_reason="The observed plates do not represent distinct batches.",
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=base_parameters,
            scope=EvidenceScope(case_id=case.case_id, plate_ids=plate_ids),
            provenance=_provenance(case),
        ),
    )
