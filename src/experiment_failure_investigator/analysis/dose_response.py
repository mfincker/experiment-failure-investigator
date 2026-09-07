"""Bounded decreasing four-parameter logistic dose-response fitting."""

from __future__ import annotations

import warnings as python_warnings
from collections import defaultdict
from collections.abc import Mapping
from math import isfinite, sqrt
from typing import Any

import numpy as np
from pydantic import JsonValue
from scipy.optimize import OptimizeWarning, curve_fit

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

TOOL_NAME = "fit_dose_response"
TOOL_VERSION = "1.0.0"


def _decreasing_four_parameter_logistic(
    log10_dose: np.ndarray,
    bottom: float,
    span: float,
    log10_ic50: float,
    hill_slope: float,
) -> np.ndarray:
    exponent = np.clip((log10_dose - log10_ic50) * hill_slope, -300.0, 300.0)
    return bottom + span / (1.0 + np.power(10.0, exponent))


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


def _condition_key(well: InvestigatorPlateMapWell) -> tuple[str, str, str]:
    if well.treatment is None or well.dose_unit is None:
        raise ValueError("dose-response group is missing an identifying annotation")
    return (well.plate_id, well.treatment, well.dose_unit)


def _base_records(
    *,
    scope: EvidenceScope,
    parameters: Mapping[str, JsonValue],
    doses: np.ndarray,
    signals: np.ndarray,
    mapped_count: int,
) -> list[EvidenceRecord]:
    unique_dose_count = len(np.unique(doses))
    response_range = float(np.max(signals) - np.min(signals)) if len(signals) else 0.0
    return [
        _record(
            metric_name="dose_response.mapped_well_count",
            value=mapped_count,
            scope=scope,
            parameters=parameters,
            description="Mapped wells in this treatment and dose-unit series.",
            sample_count=mapped_count,
            unit="wells",
        ),
        _record(
            metric_name="dose_response.observation_count",
            value=len(signals),
            scope=scope,
            parameters=parameters,
            description="Finite observations used to assess fit eligibility.",
            sample_count=len(signals),
            unit="wells",
        ),
        _record(
            metric_name="dose_response.unique_dose_count",
            value=unique_dose_count,
            scope=scope,
            parameters=parameters,
            description="Unique finite dose levels represented in the observations.",
            sample_count=len(signals),
            unit="dose_levels",
        ),
        _record(
            metric_name="dose_response.observed_response_range",
            value=response_range,
            scope=scope,
            parameters=parameters,
            description="Maximum minus minimum finite observed signal.",
            sample_count=len(signals),
            unit="raw_signal",
        ),
    ]


def _fit_one_series(
    *,
    case: InvestigatorCase,
    plate_id: str,
    treatment: str,
    dose_unit: str,
    mapped_count: int,
    observed: list[tuple[float, float]],
    min_unique_doses: int,
    response_epsilon: float,
    bound_tolerance_fraction: float,
    max_function_evaluations: int,
) -> ScientificToolResult:
    parameters: dict[str, JsonValue] = {
        "bound_strategy": "observed_range_and_log_dose",
        "bound_tolerance_fraction": bound_tolerance_fraction,
        "component": "treatment_fit",
        "max_function_evaluations": max_function_evaluations,
        "min_unique_doses": min_unique_doses,
        "response_epsilon": response_epsilon,
    }
    result_scope = EvidenceScope(case_id=case.case_id, plate_ids=(plate_id,))
    doses = np.asarray([dose for dose, _ in observed], dtype=float)
    signals = np.asarray([signal for _, signal in observed], dtype=float)
    unique_doses = tuple(float(value) for value in sorted(set(doses.tolist())))
    fit_scope = EvidenceScope(
        case_id=case.case_id,
        plate_ids=(plate_id,),
        treatments=(treatment,),
        doses=unique_doses,
        dose_unit=dose_unit,
    )
    records = _base_records(
        scope=fit_scope,
        parameters=parameters,
        doses=doses,
        signals=signals,
        mapped_count=mapped_count,
    )
    if len(unique_doses) < min_unique_doses:
        return ScientificToolResult(
            status=ToolStatus.INSUFFICIENT_DATA,
            status_reason=(
                f"{treatment} on {plate_id} has {len(unique_doses)} unique finite "
                f"doses; at least {min_unique_doses} are required for a 4PL fit."
            ),
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=result_scope,
            evidence=tuple(records),
            provenance=_provenance(case),
        )
    response_range = float(np.max(signals) - np.min(signals))
    if response_range <= response_epsilon:
        range_id = next(
            record.evidence_id
            for record in records
            if record.metric_name == "dose_response.observed_response_range"
        )
        return ScientificToolResult(
            status=ToolStatus.INSUFFICIENT_DATA,
            status_reason=(
                f"{treatment} on {plate_id} has no response variation from which to "
                "identify a four-parameter curve."
            ),
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=result_scope,
            evidence=tuple(records),
            warnings=(
                ResultWarning(
                    code="flat_observed_response",
                    message="The observed response range is zero or near zero.",
                    evidence_ids=(range_id,),
                ),
            ),
            provenance=_provenance(case),
        )

    log_doses = np.log10(doses)
    minimum_signal = float(np.min(signals))
    maximum_signal = float(np.max(signals))
    signal_scale = max(
        response_range,
        max(abs(minimum_signal), abs(maximum_signal)) * 0.1,
        response_epsilon * 10.0,
    )
    lower_bounds = np.asarray(
        [
            minimum_signal - 2.0 * signal_scale,
            0.0,
            float(np.min(log_doses)) - 3.0,
            0.05,
        ],
        dtype=float,
    )
    upper_bounds = np.asarray(
        [
            maximum_signal + 2.0 * signal_scale,
            4.0 * signal_scale,
            float(np.max(log_doses)) + 3.0,
            10.0,
        ],
        dtype=float,
    )
    initial = np.asarray(
        [
            minimum_signal,
            response_range,
            float(np.median(np.unique(log_doses))),
            1.0,
        ],
        dtype=float,
    )

    caught_optimizer_warning = False
    try:
        with python_warnings.catch_warnings(record=True) as caught:
            python_warnings.simplefilter("always", OptimizeWarning)
            fitted, covariance = curve_fit(
                _decreasing_four_parameter_logistic,
                log_doses,
                signals,
                p0=initial,
                bounds=(lower_bounds, upper_bounds),
                maxfev=max_function_evaluations,
            )
        caught_optimizer_warning = any(
            issubclass(warning.category, OptimizeWarning) for warning in caught
        )
    except (FloatingPointError, OverflowError, RuntimeError, ValueError) as error:
        return ScientificToolResult(
            status=ToolStatus.ERROR,
            status_reason=(
                f"Dose-response optimizer failed for {treatment} on {plate_id}: "
                f"{type(error).__name__}."
            ),
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=result_scope,
            provenance=_provenance(case),
        )

    if not np.isfinite(fitted).all():
        return ScientificToolResult(
            status=ToolStatus.ERROR,
            status_reason="Dose-response optimizer returned non-finite parameters.",
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=result_scope,
            provenance=_provenance(case),
        )
    bottom, span, log10_ic50, hill_slope = (float(value) for value in fitted)
    top = bottom + span
    ic50 = 10.0**log10_ic50
    predicted = _decreasing_four_parameter_logistic(
        log_doses,
        bottom,
        span,
        log10_ic50,
        hill_slope,
    )
    residuals = signals - predicted
    residual_sum_squares = float(np.sum(np.square(residuals)))
    total_sum_squares = float(np.sum(np.square(signals - np.mean(signals))))
    rmse = sqrt(residual_sum_squares / len(signals))
    mean_residual = float(np.mean(residuals))
    maximum_absolute_residual = float(np.max(np.abs(residuals)))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    numeric_metrics = (
        bottom,
        top,
        ic50,
        hill_slope,
        rmse,
        mean_residual,
        maximum_absolute_residual,
        r_squared,
    )
    if not all(isfinite(value) for value in numeric_metrics):
        return ScientificToolResult(
            status=ToolStatus.ERROR,
            status_reason="Dose-response fit diagnostics produced non-finite values.",
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=result_scope,
            provenance=_provenance(case),
        )

    fit_values = (
        ("dose_response.bottom", bottom, "Fitted lower asymptote.", "raw_signal"),
        ("dose_response.top", top, "Fitted upper asymptote.", "raw_signal"),
        ("dose_response.ic50", ic50, "Fitted half-maximal concentration.", dose_unit),
        ("dose_response.hill_slope", hill_slope, "Fitted positive Hill slope.", None),
        ("dose_response.rmse", rmse, "Root mean squared fit residual.", "raw_signal"),
        (
            "dose_response.mean_residual",
            mean_residual,
            "Mean observed-minus-fitted residual.",
            "raw_signal",
        ),
        (
            "dose_response.maximum_absolute_residual",
            maximum_absolute_residual,
            "Maximum absolute observed-minus-fitted residual.",
            "raw_signal",
        ),
        (
            "dose_response.r_squared",
            r_squared,
            "Coefficient of determination for observations used in the fit.",
            "dimensionless",
        ),
    )
    for metric_name, value, description, unit in fit_values:
        records.append(
            _record(
                metric_name=metric_name,
                value=value,
                scope=fit_scope,
                parameters=parameters,
                description=description,
                sample_count=len(signals),
                unit=unit,
            )
        )

    parameter_names = ("bottom", "span", "log10_ic50", "hill_slope")
    near_bounds = [
        name
        for name, value, lower, upper in zip(
            parameter_names,
            fitted,
            lower_bounds,
            upper_bounds,
            strict=True,
        )
        if min(value - lower, upper - value)
        <= bound_tolerance_fraction * (upper - lower)
    ]
    result_warnings: list[ResultWarning] = []
    if near_bounds:
        boundary_record = _record(
            metric_name="dose_response.parameters_near_bounds",
            value=near_bounds,
            scope=fit_scope,
            parameters=parameters,
            description="Fitted parameters within the configured bound tolerance.",
            sample_count=len(near_bounds),
        )
        records.append(boundary_record)
        result_warnings.append(
            ResultWarning(
                code="parameters_near_bounds",
                message=(
                    "One or more fitted parameters are near optimization bounds; "
                    "interpret parameter estimates cautiously."
                ),
                evidence_ids=(boundary_record.evidence_id,),
            )
        )

    if covariance.shape == (4, 4) and np.isfinite(covariance).all():
        variances = np.diag(covariance)
        top_variance = covariance[0, 0] + covariance[1, 1] + 2 * covariance[0, 1]
        if (variances >= 0).all() and top_variance >= 0:
            standard_errors: dict[str, JsonValue] = {
                "bottom": sqrt(float(variances[0])),
                "top": sqrt(float(top_variance)),
                "ic50": float(
                    np.log(10.0) * ic50 * sqrt(float(variances[2]))
                ),
                "hill_slope": sqrt(float(variances[3])),
            }
            records.append(
                _record(
                    metric_name="dose_response.parameter_standard_errors",
                    value=standard_errors,
                    scope=fit_scope,
                    parameters=parameters,
                    description=(
                        "Approximate parameter standard errors from fit covariance."
                    ),
                    sample_count=len(signals),
                )
            )
        else:
            caught_optimizer_warning = True
    else:
        caught_optimizer_warning = True
    if caught_optimizer_warning:
        result_warnings.append(
            ResultWarning(
                code="uncertainty_not_estimable",
                message=(
                    "The optimizer did not provide a finite covariance estimate; "
                    "parameter uncertainty is not reported."
                ),
            )
        )

    missing_count = mapped_count - len(signals)
    if missing_count:
        result_warnings.append(
            ResultWarning(
                code="missing_dose_response_signals",
                message=(
                    f"{missing_count} mapped wells in this series lack a finite signal."
                ),
                evidence_ids=(records[1].evidence_id,),
            )
        )
    return ScientificToolResult(
        status=ToolStatus.SUCCESS,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        parameters=parameters,
        scope=result_scope,
        evidence=tuple(records),
        warnings=tuple(result_warnings),
        limitations=(
            "The bounded 4PL fit is descriptive and does not establish a biological "
            "mechanism or validate the annotated concentrations.",
        ),
        provenance=_provenance(case),
    )


def fit_dose_response(
    case: InvestigatorCase,
    *,
    min_unique_doses: int = 5,
    response_epsilon: float = 1e-12,
    bound_tolerance_fraction: float = 0.01,
    max_function_evaluations: int = 20_000,
) -> tuple[ScientificToolResult, ...]:
    """Fit each observed treatment and dose-unit series independently per plate."""
    if min_unique_doses < 5:
        raise ValueError("min_unique_doses must be at least 5 for a four-parameter fit")
    if not isfinite(response_epsilon) or response_epsilon <= 0:
        raise ValueError("response_epsilon must be finite and positive")
    if (
        not isfinite(bound_tolerance_fraction)
        or not 0 < bound_tolerance_fraction < 0.5
    ):
        raise ValueError("bound_tolerance_fraction must be between 0 and 0.5")
    if max_function_evaluations < 1:
        raise ValueError("max_function_evaluations must be positive")

    measurement_lookup = {
        (measurement.plate_id, measurement.well): measurement
        for measurement in case.measurements
    }
    grouped_wells: dict[
        tuple[str, str, str], list[InvestigatorPlateMapWell]
    ] = defaultdict(list)
    for well in case.plate_map:
        if (
            well.well_role is WellRole.TREATMENT
            and well.treatment is not None
            and well.dose is not None
            and well.dose_unit is not None
        ):
            grouped_wells[_condition_key(well)].append(well)

    results: list[ScientificToolResult] = []
    for (plate_id, treatment, dose_unit), wells in sorted(grouped_wells.items()):
        observed: list[tuple[float, float]] = []
        for well in wells:
            measurement = measurement_lookup.get((plate_id, well.well))
            if (
                measurement is not None
                and measurement.measurement_status is MeasurementStatus.OBSERVED
                and measurement.raw_signal is not None
                and well.dose is not None
            ):
                observed.append((well.dose, measurement.raw_signal))
        observed.sort()
        results.append(
            _fit_one_series(
                case=case,
                plate_id=plate_id,
                treatment=treatment,
                dose_unit=dose_unit,
                mapped_count=len(wells),
                observed=observed,
                min_unique_doses=min_unique_doses,
                response_epsilon=response_epsilon,
                bound_tolerance_fraction=bound_tolerance_fraction,
                max_function_evaluations=max_function_evaluations,
            )
        )

    if results:
        return tuple(results)
    parameters: dict[str, JsonValue] = {"component": "treatment_fit"}
    return (
        ScientificToolResult(
            status=ToolStatus.NOT_APPLICABLE,
            status_reason=(
                "The experiment has no fully annotated treatment dose series."
            ),
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=EvidenceScope(case_id=case.case_id),
            provenance=_provenance(case),
        ),
    )
