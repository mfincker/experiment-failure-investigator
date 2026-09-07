"""Deterministic summaries of control levels, dispersion, and separation."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from statistics import fmean, median, stdev

from pydantic import JsonValue

from experiment_failure_investigator.analysis.contracts import (
    InvestigatorCase,
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

TOOL_NAME = "summarize_controls"
TOOL_VERSION = "1.0.0"
CONTROL_ROLES = (WellRole.NEGATIVE_CONTROL, WellRole.POSITIVE_CONTROL)


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


def _signals_by_role(
    case: InvestigatorCase,
    plate_id: str,
    role: WellRole,
) -> tuple[int, list[float]]:
    role_wells = {
        well.well
        for well in case.plate_map
        if well.plate_id == plate_id and well.well_role is role
    }
    signals = [
        measurement.raw_signal
        for measurement in case.measurements
        if measurement.plate_id == plate_id
        and measurement.well in role_wells
        and measurement.measurement_status is MeasurementStatus.OBSERVED
        and measurement.raw_signal is not None
    ]
    return len(role_wells), signals


def _error_result(
    *,
    case: InvestigatorCase,
    plate_id: str,
    parameters: Mapping[str, JsonValue],
    reason: str,
) -> ScientificToolResult:
    return ScientificToolResult(
        status=ToolStatus.ERROR,
        status_reason=reason,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        parameters=parameters,
        scope=EvidenceScope(case_id=case.case_id, plate_ids=(plate_id,)),
        provenance=_provenance(case),
    )


def _summarize_role(
    case: InvestigatorCase,
    plate_id: str,
    role: WellRole,
) -> ScientificToolResult:
    parameters: dict[str, JsonValue] = {
        "component": "role_summary",
        "dispersion": "sample_sd_and_unscaled_mad",
        "role": role.value,
    }
    result_scope = EvidenceScope(case_id=case.case_id, plate_ids=(plate_id,))
    evidence_scope = EvidenceScope(
        case_id=case.case_id,
        plate_ids=(plate_id,),
        well_roles=(role,),
    )
    well_count, signals = _signals_by_role(case, plate_id, role)
    if well_count == 0:
        return ScientificToolResult(
            status=ToolStatus.NOT_APPLICABLE,
            status_reason=f"Plate {plate_id} has no {role.value} wells.",
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=result_scope,
            provenance=_provenance(case),
        )

    observed_count = len(signals)
    records = [
        _record(
            metric_name="controls.mapped_well_count",
            value=well_count,
            scope=evidence_scope,
            parameters=parameters,
            description=f"Mapped {role.value} wells on {plate_id}.",
            sample_count=well_count,
            unit="wells",
        ),
        _record(
            metric_name="controls.observed_signal_count",
            value=observed_count,
            scope=evidence_scope,
            parameters=parameters,
            description=f"Finite {role.value} signals on {plate_id}.",
            sample_count=observed_count,
            unit="wells",
        ),
    ]
    warnings: list[ResultWarning] = []
    if observed_count < well_count:
        warnings.append(
            ResultWarning(
                code="missing_control_signals",
                message=(
                    f"{well_count - observed_count} of {well_count} mapped "
                    f"{role.value} wells lack a finite signal on {plate_id}."
                ),
                evidence_ids=(records[0].evidence_id, records[1].evidence_id),
            )
        )
    if observed_count == 0:
        return ScientificToolResult(
            status=ToolStatus.INSUFFICIENT_DATA,
            status_reason=f"Plate {plate_id} has no finite {role.value} signals.",
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=result_scope,
            evidence=tuple(records),
            warnings=tuple(warnings),
            provenance=_provenance(case),
        )

    try:
        mean_value = fmean(signals)
        median_value = median(signals)
        records.extend(
            [
                _record(
                    metric_name="controls.mean",
                    value=mean_value,
                    scope=evidence_scope,
                    parameters=parameters,
                    description=f"Mean finite {role.value} signal on {plate_id}.",
                    sample_count=observed_count,
                    unit="raw_signal",
                ),
                _record(
                    metric_name="controls.median",
                    value=median_value,
                    scope=evidence_scope,
                    parameters=parameters,
                    description=f"Median finite {role.value} signal on {plate_id}.",
                    sample_count=observed_count,
                    unit="raw_signal",
                ),
            ]
        )
        if observed_count >= 2:
            sample_sd = stdev(signals)
            mad = median([abs(value - median_value) for value in signals])
            records.extend(
                [
                    _record(
                        metric_name="controls.sample_standard_deviation",
                        value=sample_sd,
                        scope=evidence_scope,
                        parameters=parameters,
                        description=(
                            f"Sample standard deviation of finite {role.value} "
                            f"signals on {plate_id}."
                        ),
                        sample_count=observed_count,
                        unit="raw_signal",
                    ),
                    _record(
                        metric_name="controls.median_absolute_deviation",
                        value=mad,
                        scope=evidence_scope,
                        parameters=parameters,
                        description=(
                            f"Unscaled median absolute deviation of finite "
                            f"{role.value} signals on {plate_id}."
                        ),
                        sample_count=observed_count,
                        unit="raw_signal",
                    ),
                ]
            )
    except (ArithmeticError, OverflowError, ValueError) as error:
        return _error_result(
            case=case,
            plate_id=plate_id,
            parameters=parameters,
            reason=f"Control summary calculation failed: {error}",
        )

    status = ToolStatus.SUCCESS
    reason = None
    limitations: tuple[str, ...] = ()
    if observed_count == 1:
        status = ToolStatus.INSUFFICIENT_DATA
        reason = (
            f"Plate {plate_id} has only one finite {role.value} signal; "
            "variability cannot be estimated."
        )
        limitations = (
            "A singleton control supports a level estimate but not a variability "
            "estimate.",
        )
    return ScientificToolResult(
        status=status,
        status_reason=reason,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        parameters=parameters,
        scope=result_scope,
        evidence=tuple(records),
        warnings=tuple(warnings),
        limitations=limitations,
        provenance=_provenance(case),
    )


def _summarize_separation(
    case: InvestigatorCase,
    plate_id: str,
    denominator_epsilon: float,
) -> ScientificToolResult:
    parameters: dict[str, JsonValue] = {
        "component": "control_separation",
        "denominator_epsilon": denominator_epsilon,
        "z_prime_formula": "1 - 3 * (negative_sd + positive_sd) / absolute_separation",
    }
    result_scope = EvidenceScope(case_id=case.case_id, plate_ids=(plate_id,))
    comparison_scope = EvidenceScope(
        case_id=case.case_id,
        plate_ids=(plate_id,),
        well_roles=CONTROL_ROLES,
    )
    negative_wells, negative = _signals_by_role(
        case, plate_id, WellRole.NEGATIVE_CONTROL
    )
    positive_wells, positive = _signals_by_role(
        case, plate_id, WellRole.POSITIVE_CONTROL
    )
    absent = [
        role.value
        for role, count in (
            (WellRole.NEGATIVE_CONTROL, negative_wells),
            (WellRole.POSITIVE_CONTROL, positive_wells),
        )
        if count == 0
    ]
    if absent:
        return ScientificToolResult(
            status=ToolStatus.NOT_APPLICABLE,
            status_reason=(
                f"Control separation is not applicable on {plate_id}; missing role(s): "
                + ", ".join(absent)
                + "."
            ),
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=result_scope,
            provenance=_provenance(case),
        )

    records = [
        _record(
            metric_name="controls.negative_observed_count",
            value=len(negative),
            scope=comparison_scope,
            parameters=parameters,
            description=f"Finite negative-control signals on {plate_id}.",
            sample_count=len(negative),
            unit="wells",
        ),
        _record(
            metric_name="controls.positive_observed_count",
            value=len(positive),
            scope=comparison_scope,
            parameters=parameters,
            description=f"Finite positive-control signals on {plate_id}.",
            sample_count=len(positive),
            unit="wells",
        ),
    ]
    if not negative or not positive:
        return ScientificToolResult(
            status=ToolStatus.INSUFFICIENT_DATA,
            status_reason=(
                f"Control separation on {plate_id} requires at least one finite "
                "signal from each control role."
            ),
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=result_scope,
            evidence=tuple(records),
            provenance=_provenance(case),
        )

    try:
        negative_mean = fmean(negative)
        positive_mean = fmean(positive)
        signed_separation = negative_mean - positive_mean
        absolute_separation = abs(signed_separation)
        records.extend(
            [
                _record(
                    metric_name="controls.negative_mean",
                    value=negative_mean,
                    scope=comparison_scope,
                    parameters=parameters,
                    description=f"Mean finite negative-control signal on {plate_id}.",
                    sample_count=len(negative),
                    unit="raw_signal",
                ),
                _record(
                    metric_name="controls.positive_mean",
                    value=positive_mean,
                    scope=comparison_scope,
                    parameters=parameters,
                    description=f"Mean finite positive-control signal on {plate_id}.",
                    sample_count=len(positive),
                    unit="raw_signal",
                ),
                _record(
                    metric_name="controls.signed_separation",
                    value=signed_separation,
                    scope=comparison_scope,
                    parameters=parameters,
                    description=(
                        "Negative-control mean minus positive-control mean on "
                        f"{plate_id}."
                    ),
                    sample_count=len(negative) + len(positive),
                    unit="raw_signal",
                ),
                _record(
                    metric_name="controls.absolute_separation",
                    value=absolute_separation,
                    scope=comparison_scope,
                    parameters=parameters,
                    description=(
                        f"Absolute difference between control means on {plate_id}."
                    ),
                    sample_count=len(negative) + len(positive),
                    unit="raw_signal",
                ),
            ]
        )
    except (ArithmeticError, OverflowError, ValueError) as error:
        return _error_result(
            case=case,
            plate_id=plate_id,
            parameters=parameters,
            reason=f"Control separation calculation failed: {error}",
        )

    if len(negative) < 2 or len(positive) < 2:
        return ScientificToolResult(
            status=ToolStatus.INSUFFICIENT_DATA,
            status_reason=(
                f"Control separation is available on {plate_id}, but the assay-window "
                "statistic requires at least two finite values per control role."
            ),
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=result_scope,
            evidence=tuple(records),
            limitations=(
                "Control means can be compared, but within-role variability is not "
                "estimable for both roles.",
            ),
            provenance=_provenance(case),
        )
    if absolute_separation <= denominator_epsilon:
        separation_id = next(
            record.evidence_id
            for record in records
            if record.metric_name == "controls.absolute_separation"
        )
        return ScientificToolResult(
            status=ToolStatus.INSUFFICIENT_DATA,
            status_reason=(
                f"Control means on {plate_id} have no usable separation; the "
                "assay-window denominator is zero or near zero."
            ),
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=result_scope,
            evidence=tuple(records),
            warnings=(
                ResultWarning(
                    code="zero_control_window",
                    message="The assay-window statistic is undefined.",
                    evidence_ids=(separation_id,),
                ),
            ),
            provenance=_provenance(case),
        )

    try:
        negative_sd = stdev(negative)
        positive_sd = stdev(positive)
        z_prime = 1.0 - 3.0 * (negative_sd + positive_sd) / absolute_separation
        records.append(
            _record(
                metric_name="controls.z_prime",
                value=z_prime,
                scope=comparison_scope,
                parameters=parameters,
                description=f"Z-prime assay-window statistic on {plate_id}.",
                sample_count=len(negative) + len(positive),
                unit="dimensionless",
            )
        )
    except (ArithmeticError, OverflowError, ValueError) as error:
        return _error_result(
            case=case,
            plate_id=plate_id,
            parameters=parameters,
            reason=f"Assay-window calculation failed: {error}",
        )

    warnings: tuple[ResultWarning, ...] = ()
    missing_signal_count = (
        negative_wells + positive_wells - len(negative) - len(positive)
    )
    if missing_signal_count:
        warnings = (
            ResultWarning(
                code="missing_control_signals",
                message=(
                    f"{missing_signal_count} mapped control wells lack a finite signal "
                    f"on {plate_id}."
                ),
                evidence_ids=(records[0].evidence_id, records[1].evidence_id),
            ),
        )
    return ScientificToolResult(
        status=ToolStatus.SUCCESS,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        parameters=parameters,
        scope=result_scope,
        evidence=tuple(records),
        warnings=warnings,
        limitations=(
            "Z-prime is reported descriptively; acceptable values and thresholds are "
            "assay-specific.",
        ),
        provenance=_provenance(case),
    )


def summarize_controls(
    case: InvestigatorCase,
    *,
    denominator_epsilon: float = 1e-12,
) -> tuple[ScientificToolResult, ...]:
    """Return independent role and separation results for every observed plate."""
    if not isfinite(denominator_epsilon) or denominator_epsilon <= 0:
        raise ValueError("denominator_epsilon must be finite and positive")
    results: list[ScientificToolResult] = []
    for plate in case.design.plates:
        for role in CONTROL_ROLES:
            results.append(_summarize_role(case, plate.plate_id, role))
        results.append(_summarize_separation(case, plate.plate_id, denominator_epsilon))
    return tuple(results)
