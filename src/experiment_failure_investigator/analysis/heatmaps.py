"""Investigator-safe heatmap export backed by the benchmark Altair renderer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import JsonValue

from experiment_failure_investigator.analysis.contracts import (
    InvestigatorCase,
    MeasurementStatus,
)
from experiment_failure_investigator.analysis.results import (
    ArtifactAttachment,
    EvidenceScope,
    ScientificProvenance,
    ScientificToolResult,
    ToolStatus,
)
from experiment_failure_investigator.analysis.spatial import (
    condition_centered_residuals,
)
from experiment_failure_investigator.benchmark.plotting import (
    PublicPlotMetadata,
    build_plate_heatmap,
    build_residual_heatmap,
    export_chart_png,
    shared_residual_domain,
    shared_signal_domain,
)
from experiment_failure_investigator.benchmark.models import WellRole

TOOL_NAME = "generate_plate_heatmap"
TOOL_VERSION = "1.0.0"
HeatmapView = Literal["raw_signal", "condition_residual"]


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


def generate_plate_heatmap(
    case: InvestigatorCase,
    output_directory: Path,
    *,
    filename: str = "plate_heatmap.png",
    view: HeatmapView = "raw_signal",
) -> ScientificToolResult:
    """Export one public-data heatmap and return it as a hashed attachment."""
    if view not in {"raw_signal", "condition_residual"}:
        raise ValueError("view must be raw_signal or condition_residual")
    output = _safe_output_path(output_directory, filename)
    parameters: dict[str, JsonValue] = {
        "filename": filename,
        "view": view,
    }
    scope = EvidenceScope(
        case_id=case.case_id,
        plate_ids=tuple(plate.plate_id for plate in case.design.plates),
    )
    provenance = ScientificProvenance(
        case_id=case.case_id,
        public_artifact_hashes=case.public_artifact_hashes,
    )
    tables = _plot_tables(case)
    finite_non_empty_count = sum(
        role != WellRole.EMPTY.value and signal is not None
        for role, signal in zip(
            tables.plate_map.get("well_role", ()),
            tables.measurements.get("raw_signal", ()),
            strict=True,
        )
    )
    if finite_non_empty_count == 0:
        return ScientificToolResult(
            status=ToolStatus.INSUFFICIENT_DATA,
            status_reason=(
                "No finite public measurements from non-empty wells are available "
                "to plot."
            ),
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
            parameters=parameters,
            scope=scope,
            provenance=provenance,
        )

    metadata = PublicPlotMetadata(case_id=case.case_id)
    if view == "condition_residual":
        tables = _add_residuals(case, tables)
        if tables.measurements.empty:
            return ScientificToolResult(
                status=ToolStatus.INSUFFICIENT_DATA,
                status_reason=(
                    "No conditions have at least two finite observations for "
                    "residual plotting."
                ),
                tool_name=TOOL_NAME,
                tool_version=TOOL_VERSION,
                parameters=parameters,
                scope=scope,
                provenance=provenance,
            )
        domain = shared_residual_domain(tables)
        chart = build_residual_heatmap(
            tables,
            metadata,
            residual_domain=domain,
        )
    else:
        domain = shared_signal_domain(tables)
        chart = build_plate_heatmap(tables, metadata, signal_domain=domain)
    export_chart_png(chart, output)
    return ScientificToolResult(
        status=ToolStatus.SUCCESS,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        parameters={**parameters, "color_domain": list(domain)},
        scope=scope,
        attachments=(
            ArtifactAttachment(
                name=view,
                path=str(output),
                sha256=_sha256(output),
                media_type="image/png",
                description=(
                    "Plate heatmap rendered only from investigator-visible public "
                    f"data ({view})."
                ),
            ),
        ),
        limitations=(
            "The heatmap is visual supporting evidence and is not itself a numeric "
            "diagnostic or causal conclusion.",
        ),
        provenance=provenance,
    )
