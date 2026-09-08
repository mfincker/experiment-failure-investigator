"""Investigator-safe diagnostic plots backed by the shared Altair renderer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import JsonValue

from experiment_failure_investigator.analysis.contracts import (
    InvestigatorCase,
    MeasurementStatus,
)
from experiment_failure_investigator.analysis.dose_response import fit_dose_response
from experiment_failure_investigator.analysis.results import (
    ArtifactAttachment,
    EvidenceScope,
    ResultWarning,
    ScientificProvenance,
    ScientificToolResult,
    ToolStatus,
)
from experiment_failure_investigator.analysis.spatial import (
    condition_centered_residuals,
)
from experiment_failure_investigator.benchmark.models import WellRole
from experiment_failure_investigator.benchmark.plotting import (
    PublicPlotMetadata,
    build_control_qc_chart,
    build_dose_response_chart,
    build_plate_heatmap,
    build_residual_heatmap,
    export_chart_png,
    shared_residual_domain,
    shared_signal_domain,
)

TOOL_NAME = "generate_diagnostic_plot"
TOOL_VERSION = "1.0.0"
HeatmapView = Literal["raw_signal", "condition_residual"]
DiagnosticPlotKind = Literal[
    "raw_signal",
    "condition_residual",
    "control_distribution",
    "dose_response",
]


@dataclass(frozen=True)
class _PublicPlotTables:
    plate_map: pd.DataFrame
    measurements: pd.DataFrame


def _plot_tables(case: InvestigatorCase) -> _PublicPlotTables:
    observed = {
        (measurement.plate_id, measurement.well): measurement.raw_signal
        for measurement in case.measurements
        if measurement.measurement_status is MeasurementStatus.OBSERVED
        and measurement.raw_signal is not None
    }
    map_rows = [
        well.model_dump(mode="json")
        for well in case.plate_map
        if well.well_role is WellRole.EMPTY
        or (well.plate_id, well.well) in observed
    ]
    measurement_rows = [
        {
            "plate_id": well["plate_id"],
            "well": well["well"],
            "raw_signal": observed.get((well["plate_id"], well["well"])),
        }
        for well in map_rows
    ]
    return _PublicPlotTables(
        plate_map=pd.DataFrame(map_rows),
        measurements=pd.DataFrame(measurement_rows),
    )


def _add_residuals(
    case: InvestigatorCase,
    tables: _PublicPlotTables,
) -> _PublicPlotTables:
    residuals = {
        (observation.plate_id, observation.well): observation.condition_residual
        for observation in condition_centered_residuals(case)
    }
    if not residuals:
        return _PublicPlotTables(pd.DataFrame(), pd.DataFrame())
    plate_map = tables.plate_map[
        tables.plate_map.apply(
            lambda row: (str(row["plate_id"]), str(row["well"])) in residuals,
            axis=1,
        )
    ].copy()
    measurements = tables.measurements.merge(
        pd.DataFrame(
            [
                {
                    "plate_id": plate_id,
                    "well": well,
                    "condition_residual": residual,
                }
                for (plate_id, well), residual in sorted(residuals.items())
            ]
        ),
        on=["plate_id", "well"],
        how="inner",
        validate="one_to_one",
    )
    return _PublicPlotTables(plate_map=plate_map, measurements=measurements)


def _safe_output_path(output_directory: Path, filename: str) -> Path:
    if (
        not filename.strip()
        or filename != Path(filename).name
        or "\\" in filename
        or Path(filename).suffix.lower() != ".png"
    ):
        raise ValueError("filename must be a non-escaping .png basename")
    root = output_directory.resolve()
    output = (root / filename).resolve()
    if not output.is_relative_to(root):
        raise ValueError("heatmap output must remain inside the output directory")
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fitted_curve_frame(case: InvestigatorCase) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    required = {
        "dose_response.bottom",
        "dose_response.top",
        "dose_response.ic50",
        "dose_response.hill_slope",
    }
    for result in fit_dose_response(case):
        if result.status is not ToolStatus.SUCCESS:
            continue
        records = {record.metric_name: record for record in result.evidence}
        if not required <= set(records):
            continue
        scope = records["dose_response.ic50"].scope
        if not scope.plate_ids or not scope.treatments or not scope.doses:
            continue
        numeric = {
            name: records[name].value
            for name in required
            if isinstance(records[name].value, int | float)
            and not isinstance(records[name].value, bool)
        }
        if set(numeric) != required:
            continue
        bottom = float(numeric["dose_response.bottom"])
        top = float(numeric["dose_response.top"])
        ic50 = float(numeric["dose_response.ic50"])
        hill_slope = float(numeric["dose_response.hill_slope"])
        doses = np.geomspace(min(scope.doses), max(scope.doses), num=120)
        fitted = bottom + (top - bottom) / (
            1.0 + np.power(doses / ic50, hill_slope)
        )
        rows.extend(
            {
                "plate_id": scope.plate_ids[0],
                "treatment": scope.treatments[0],
                "dose": float(dose),
                "fitted_signal": float(signal),
            }
            for dose, signal in zip(doses, fitted, strict=True)
        )
    return pd.DataFrame(rows)


def _status_result(
    *,
    status: ToolStatus,
    reason: str,
    parameters: dict[str, JsonValue],
    scope: EvidenceScope,
    provenance: ScientificProvenance,
) -> ScientificToolResult:
    return ScientificToolResult(
        status=status,
        status_reason=reason,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        parameters=parameters,
        scope=scope,
        provenance=provenance,
    )


def generate_diagnostic_plot(
    case: InvestigatorCase,
    output_directory: Path,
    *,
    kind: DiagnosticPlotKind,
    filename: str,
) -> ScientificToolResult:
    """Export one public-data diagnostic plot as a hashed attachment."""
    if kind not in {
        "raw_signal",
        "condition_residual",
        "control_distribution",
        "dose_response",
    }:
        raise ValueError("unsupported diagnostic plot kind")
    output = _safe_output_path(output_directory, filename)
    parameters: dict[str, JsonValue] = {"filename": filename, "kind": kind}
    scope = EvidenceScope(
        case_id=case.case_id,
        plate_ids=tuple(plate.plate_id for plate in case.design.plates),
    )
    provenance = ScientificProvenance(
        case_id=case.case_id,
        public_artifact_hashes=case.public_artifact_hashes,
    )
    tables = _plot_tables(case)
    metadata = PublicPlotMetadata(case_id=case.case_id)
    warnings: tuple[ResultWarning, ...] = ()

    if kind == "condition_residual":
        tables = _add_residuals(case, tables)
        if tables.measurements.empty:
            return _status_result(
                status=ToolStatus.INSUFFICIENT_DATA,
                reason=(
                    "No conditions have at least two finite observations for "
                    "residual plotting."
                ),
                parameters=parameters,
                scope=scope,
                provenance=provenance,
            )
        domain = shared_residual_domain(tables)
        chart = build_residual_heatmap(tables, metadata, residual_domain=domain)
    else:
        mapped_roles = {well.well_role for well in case.plate_map}
        observed_roles = set(tables.plate_map.get("well_role", ()))
        if kind == "control_distribution":
            control_roles = {
                WellRole.NEGATIVE_CONTROL,
                WellRole.POSITIVE_CONTROL,
            }
            if not mapped_roles & control_roles:
                return _status_result(
                    status=ToolStatus.NOT_APPLICABLE,
                    reason="The experiment has no mapped control wells.",
                    parameters=parameters,
                    scope=scope,
                    provenance=provenance,
                )
            if not observed_roles & {role.value for role in control_roles}:
                return _status_result(
                    status=ToolStatus.INSUFFICIENT_DATA,
                    reason="No mapped control wells have finite observations.",
                    parameters=parameters,
                    scope=scope,
                    provenance=provenance,
                )
        if kind == "dose_response":
            mapped_treatments = [
                well
                for well in case.plate_map
                if well.well_role is WellRole.TREATMENT and well.dose is not None
            ]
            if not mapped_treatments:
                return _status_result(
                    status=ToolStatus.NOT_APPLICABLE,
                    reason="The experiment has no annotated treatment dose series.",
                    parameters=parameters,
                    scope=scope,
                    provenance=provenance,
                )
            if WellRole.TREATMENT.value not in observed_roles:
                return _status_result(
                    status=ToolStatus.INSUFFICIENT_DATA,
                    reason="No treatment wells have finite observations.",
                    parameters=parameters,
                    scope=scope,
                    provenance=provenance,
                )

        finite_non_empty_count = sum(
            role != WellRole.EMPTY.value and signal is not None
            for role, signal in zip(
                tables.plate_map.get("well_role", ()),
                tables.measurements.get("raw_signal", ()),
                strict=True,
            )
        )
        if finite_non_empty_count == 0:
            return _status_result(
                status=ToolStatus.INSUFFICIENT_DATA,
                reason=(
                    "No finite public measurements from non-empty wells are "
                    "available to plot."
                ),
                parameters=parameters,
                scope=scope,
                provenance=provenance,
            )
        domain = shared_signal_domain(tables)
        if kind == "raw_signal":
            chart = build_plate_heatmap(tables, metadata, signal_domain=domain)
        elif kind == "control_distribution":
            chart = build_control_qc_chart(tables, metadata, signal_domain=domain)
        else:
            fit_curves = _fitted_curve_frame(case)
            parameters["fitted_curve_count"] = (
                0
                if fit_curves.empty
                else int(
                    len(fit_curves[["plate_id", "treatment"]].drop_duplicates())
                )
            )
            if fit_curves.empty:
                warnings = (
                    ResultWarning(
                        code="fitted_curve_unavailable",
                        message=(
                            "No fitted curve was estimable; the plot retains "
                            "replicates and condition means."
                        ),
                    ),
                )
            chart = build_dose_response_chart(
                tables,
                metadata,
                signal_domain=domain,
                fit_curves=None if fit_curves.empty else fit_curves,
            )

    parameters["color_domain"] = list(domain)
    export_chart_png(chart, output)
    return ScientificToolResult(
        status=ToolStatus.SUCCESS,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        parameters=parameters,
        scope=scope,
        attachments=(
            ArtifactAttachment(
                name=kind,
                path=filename,
                sha256=_sha256(output),
                media_type="image/png",
                description=(
                    "Diagnostic plot rendered only from investigator-visible "
                    f"public data ({kind})."
                ),
            ),
        ),
        warnings=warnings,
        limitations=(
            "Plots are visual supporting evidence and are not themselves numeric "
            "diagnostics or causal conclusions.",
        ),
        provenance=provenance,
    )


def generate_plate_heatmap(
    case: InvestigatorCase,
    output_directory: Path,
    *,
    filename: str = "plate_heatmap.png",
    view: HeatmapView = "raw_signal",
) -> ScientificToolResult:
    """Backward-compatible typed wrapper for either heatmap view."""
    if view not in {"raw_signal", "condition_residual"}:
        raise ValueError("view must be raw_signal or condition_residual")
    return generate_diagnostic_plot(
        case,
        output_directory,
        kind=view,
        filename=filename,
    )
