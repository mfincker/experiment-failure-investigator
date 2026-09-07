"""Altair inspection charts for clean and failure-injected assay plates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import altair as alt
import numpy as np
import pandas as pd
import vl_convert as vlc

from experiment_failure_investigator.benchmark.models import WellRole

TREATMENT_COLORS = ["#2563EB", "#D97706", "#647A30", "#BE185D", "#CA8A04"]
PLATE_COLORS = ["#2563EB", "#D97706", "#647A30", "#BE185D", "#CA8A04"]
CONTROL_STYLES = {
    WellRole.NEGATIVE_CONTROL.value: ("#374151", [6, 3]),
    WellRole.POSITIVE_CONTROL.value: ("#9CA3AF", [2, 2]),
}


class AssayTables(Protocol):
    """Structural input required by plotting functions."""

    plate_map: pd.DataFrame
    measurements: pd.DataFrame


@dataclass(frozen=True)
class PlotMetadata:
    """Visible provenance included in every inspection chart title."""

    case_id: str
    failure_label: str
    variant: str
    root_seed: int


@dataclass(frozen=True)
class _AssayTableView:
    plate_map: pd.DataFrame
    measurements: pd.DataFrame


def _title(name: str, metadata: PlotMetadata, note: str) -> alt.TitleParams:
    scenario = metadata.failure_label.replace("_", " ")
    variant = metadata.variant.replace("_", " ")
    return alt.TitleParams(
        text=f"{name} — {metadata.case_id}",
        subtitle=[
            f"Scenario: {scenario} · Variant: {variant}",
            f"Root seed: {metadata.root_seed} · {note}",
        ],
        anchor="start",
        color="#111827",
        fontSize=16,
        subtitleColor="#4B5563",
        subtitleFontSize=11,
    )


def _plot_frame(assay: AssayTables) -> pd.DataFrame:
    required_map = {
        "plate_id",
        "well",
        "row",
        "column",
        "sample_id",
        "well_role",
        "treatment",
        "dose",
        "dose_unit",
        "replicate",
    }
    required_measurements = {"plate_id", "well", "raw_signal"}
    missing_map = sorted(required_map - set(assay.plate_map.columns))
    missing_measurements = sorted(
        required_measurements - set(assay.measurements.columns)
    )
    if missing_map or missing_measurements:
        missing = [*missing_map, *missing_measurements]
        raise ValueError("plot input is missing required columns: " + ", ".join(missing))
    if assay.plate_map.duplicated(["plate_id", "well"]).any():
        raise ValueError("plate map contains duplicate plate/well keys")
    if assay.measurements.duplicated(["plate_id", "well"]).any():
        raise ValueError("measurements contain duplicate plate/well keys")

    frame = assay.plate_map.merge(
        assay.measurements[["plate_id", "well", "raw_signal"]],
        on=["plate_id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(assay.plate_map) or len(frame) != len(assay.measurements):
        raise ValueError("plate-map and measurement keys do not match")
    if not np.isfinite(frame["raw_signal"]).all():
        raise ValueError("inspection charts require finite raw signals")
    return frame.sort_values(["plate_id", "row", "column"], kind="stable").reset_index(
        drop=True
    )


def shared_signal_domain(
    *assays: AssayTables,
    padding_fraction: float = 0.05,
) -> tuple[float, float]:
    """Return one padded signal domain shared by comparable assay charts."""
    if not assays:
        raise ValueError("at least one assay is required")
    if padding_fraction < 0:
        raise ValueError("padding fraction must not be negative")
    values = np.concatenate(
        [_plot_frame(assay)["raw_signal"].to_numpy(dtype=float) for assay in assays]
    )
    lower = float(values.min())
    upper = float(values.max())
    span = upper - lower
    padding = max(span * padding_fraction, 0.02 if span == 0 else 0.0)
    return lower - padding, upper + padding


def _residual_frame(assay: AssayTables) -> pd.DataFrame:
    frame = _plot_frame(assay)
    condition_columns = ["well_role", "treatment", "dose"]
    condition_mean = frame.groupby(
        condition_columns,
        dropna=False,
    )["raw_signal"].transform("mean")
    frame["condition_residual"] = frame["raw_signal"] - condition_mean
    return frame


def shared_residual_domain(
    *assays: AssayTables,
    padding_fraction: float = 0.05,
) -> tuple[float, float]:
    """Return a symmetric residual domain shared by comparable assays."""
    if not assays:
        raise ValueError("at least one assay is required")
    if padding_fraction < 0:
        raise ValueError("padding fraction must not be negative")
    values = np.concatenate(
        [
            _residual_frame(assay)["condition_residual"].to_numpy(dtype=float)
            for assay in assays
        ]
    )
    bound = float(np.abs(values).max())
    padded_bound = max(bound * (1 + padding_fraction), 0.02)
    return -padded_bound, padded_bound


def _validate_domain(signal_domain: tuple[float, float]) -> tuple[float, float]:
    lower, upper = signal_domain
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("signal domain must contain two increasing finite values")
    return float(lower), float(upper)


def _facet_if_needed(
    chart: alt.LayerChart,
    frame: pd.DataFrame,
    title: alt.TitleParams,
) -> alt.LayerChart | alt.FacetChart:
    if frame["plate_id"].nunique() == 1:
        return chart.properties(title=title)
    return (
        chart.facet(
            column=alt.Column(
                "plate_id:N",
                title=None,
                header=alt.Header(labelColor="#374151", labelFontSize=12),
            )
        )
        .resolve_scale(color="shared")
        .properties(title=title)
    )


def build_plate_heatmap(
    assay: AssayTables,
    metadata: PlotMetadata,
    *,
    signal_domain: tuple[float, float] | None = None,
) -> alt.LayerChart | alt.FacetChart:
    """Build a fixed-orientation plate heatmap with numeric well labels."""
    frame = _plot_frame(assay)
    domain = _validate_domain(signal_domain or shared_signal_domain(assay))
    row_order = sorted(frame["row"].unique())
    column_order = sorted(int(value) for value in frame["column"].unique())
    midpoint = (domain[0] + domain[1]) / 2

    base = alt.Chart(frame)
    rectangles = base.mark_rect(stroke="#F9FAFB", strokeWidth=1).encode(
        x=alt.X(
            "column:O",
            title="Column",
            sort=column_order,
            axis=alt.Axis(labelAngle=0, grid=False),
        ),
        y=alt.Y(
            "row:O",
            title="Row",
            sort=row_order,
            axis=alt.Axis(grid=False),
        ),
        color=alt.Color(
            "raw_signal:Q",
            title="Raw signal",
            scale=alt.Scale(domain=list(domain), scheme="viridis", clamp=True),
        ),
        tooltip=[
            alt.Tooltip("plate_id:N", title="Plate"),
            alt.Tooltip("well:N", title="Well"),
            alt.Tooltip("well_role:N", title="Role"),
            alt.Tooltip("treatment:N", title="Treatment"),
            alt.Tooltip("dose:Q", title="Dose", format=".4g"),
            alt.Tooltip("raw_signal:Q", title="Raw signal", format=".4f"),
        ],
    )
    labels = base.mark_text(fontSize=8).encode(
        x=alt.X("column:O", sort=column_order),
        y=alt.Y("row:O", sort=row_order),
        text=alt.Text("raw_signal:Q", format=".2f"),
        color=alt.condition(
            f"datum.raw_signal >= {midpoint}",
            alt.value("#111827"),
            alt.value("#FFFFFF"),
        ),
    )
    width = min(720, max(480, len(column_order) * 44))
    height = min(560, max(320, len(row_order) * 42))
    chart = alt.layer(rectangles, labels).properties(width=width, height=height)
    return _facet_if_needed(
        chart,
        frame,
        _title(
            "Plate heatmap",
            metadata,
            f"shared signal scale {domain[0]:.3f} to {domain[1]:.3f}",
        ),
    )


def build_residual_heatmap(
    assay: AssayTables,
    metadata: PlotMetadata,
    *,
    residual_domain: tuple[float, float] | None = None,
) -> alt.LayerChart | alt.FacetChart:
    """Build a heatmap after centering each observed experimental condition."""
    frame = _residual_frame(assay)
    domain = _validate_domain(residual_domain or shared_residual_domain(assay))
    if not np.isclose(abs(domain[0]), abs(domain[1])):
        raise ValueError("residual domain must be symmetric around zero")
    row_order = sorted(frame["row"].unique())
    column_order = sorted(int(value) for value in frame["column"].unique())
    contrast_threshold = domain[1] * 0.55

    base = alt.Chart(frame)
    rectangles = base.mark_rect(stroke="#F9FAFB", strokeWidth=1).encode(
        x=alt.X(
            "column:O",
            title="Column",
            sort=column_order,
            axis=alt.Axis(labelAngle=0, grid=False),
        ),
        y=alt.Y(
            "row:O",
            title="Row",
            sort=row_order,
            axis=alt.Axis(grid=False),
        ),
        color=alt.Color(
            "condition_residual:Q",
            title="Residual",
            scale=alt.Scale(
                domain=list(domain),
                domainMid=0,
                scheme="blueorange",
                clamp=True,
            ),
        ),
        tooltip=[
            alt.Tooltip("plate_id:N", title="Plate"),
            alt.Tooltip("well:N", title="Well"),
            alt.Tooltip("well_role:N", title="Role"),
            alt.Tooltip("treatment:N", title="Treatment"),
            alt.Tooltip("dose:Q", title="Dose", format=".4g"),
            alt.Tooltip("raw_signal:Q", title="Raw signal", format=".4f"),
            alt.Tooltip(
                "condition_residual:Q", title="Condition residual", format="+.4f"
            ),
        ],
    )
    labels = base.mark_text(fontSize=8).encode(
        x=alt.X("column:O", sort=column_order),
        y=alt.Y("row:O", sort=row_order),
        text=alt.Text("condition_residual:Q", format="+.2f"),
        color=alt.condition(
            f"abs(datum.condition_residual) <= {contrast_threshold}",
            alt.value("#111827"),
            alt.value("#FFFFFF"),
        ),
    )
    width = min(720, max(480, len(column_order) * 44))
    height = min(560, max(320, len(row_order) * 42))
    chart = alt.layer(rectangles, labels).properties(width=width, height=height)
    return _facet_if_needed(
        chart,
        frame,
        _title(
            "Condition-centered residuals",
            metadata,
            f"shared symmetric scale ±{domain[1]:.3f}",
        ),
    )


def _control_layers(
    frame: pd.DataFrame,
    dose_domain: tuple[float, float],
    signal_domain: tuple[float, float],
) -> list[alt.Chart]:
    controls = frame[
        frame["well_role"].isin(
            [WellRole.NEGATIVE_CONTROL.value, WellRole.POSITIVE_CONTROL.value]
        )
    ]
    summary = (
        controls.groupby(["plate_id", "well_role"], as_index=False)["raw_signal"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    summary["lower"] = summary["mean"] - summary["std"]
    summary["upper"] = summary["mean"] + summary["std"]
    summary["dose_min"] = dose_domain[0]
    summary["dose_max"] = dose_domain[1]

    layers: list[alt.Chart] = []
    for role, (color, dash) in CONTROL_STYLES.items():
        role_summary = summary[summary["well_role"] == role]
        band = alt.Chart(role_summary).mark_rect(
            color=color,
            opacity=0.08,
        ).encode(
            x=alt.X("dose_min:Q", scale=alt.Scale(type="log", domain=list(dose_domain))),
            x2="dose_max:Q",
            y=alt.Y("lower:Q", scale=alt.Scale(domain=list(signal_domain))),
            y2="upper:Q",
        )
        rule = alt.Chart(role_summary).mark_rule(
            color=color,
            strokeDash=dash,
            strokeWidth=1.5,
        ).encode(
            y=alt.Y("mean:Q", scale=alt.Scale(domain=list(signal_domain))),
            tooltip=[
                alt.Tooltip("well_role:N", title="Control"),
                alt.Tooltip("mean:Q", title="Mean", format=".4f"),
                alt.Tooltip("std:Q", title="SD", format=".4f"),
            ],
        )
        layers.extend([band, rule])
    return layers


def build_dose_response_chart(
    assay: AssayTables,
    metadata: PlotMetadata,
    *,
    signal_domain: tuple[float, float] | None = None,
) -> alt.LayerChart | alt.HConcatChart:
    """Build replicate and mean dose responses with control reference bands."""
    frame = _plot_frame(assay)
    domain = _validate_domain(signal_domain or shared_signal_domain(assay))
    plate_ids = sorted(frame["plate_id"].unique())
    if len(plate_ids) > 1:
        panels: list[alt.LayerChart] = []
        for plate_id in plate_ids:
            view = _AssayTableView(
                plate_map=assay.plate_map[
                    assay.plate_map["plate_id"] == plate_id
                ].copy(),
                measurements=assay.measurements[
                    assay.measurements["plate_id"] == plate_id
                ].copy(),
            )
            panel = build_dose_response_chart(
                view,
                metadata,
                signal_domain=domain,
            ).properties(title=alt.TitleParams(text=str(plate_id), anchor="middle"))
            panels.append(panel)
        return alt.hconcat(*panels, spacing=24).properties(
            title=_title(
                "Dose response",
                metadata,
                "points = replicates · lines = means · control bands = mean ± SD",
            )
        )

    treatment = frame[frame["well_role"] == WellRole.TREATMENT.value].copy()
    if treatment.empty:
        raise ValueError("dose-response chart requires treatment wells")
    if (treatment["dose"] <= 0).any() or treatment["dose"].isna().any():
        raise ValueError("dose-response chart requires positive treatment doses")
    treatment_order = sorted(treatment["treatment"].unique())
    if len(treatment_order) > len(TREATMENT_COLORS):
        raise ValueError("dose-response chart supports at most five treatments")
    dose_domain = (float(treatment["dose"].min()), float(treatment["dose"].max()))
    color = alt.Color(
        "treatment:N",
        title="Treatment",
        sort=treatment_order,
        scale=alt.Scale(
            domain=treatment_order,
            range=TREATMENT_COLORS[: len(treatment_order)],
        ),
        legend=alt.Legend(
            orient="top",
            labelExpr="replace(datum.label, '_', ' ')",
        ),
    )
    shape = alt.Shape(
        "treatment:N",
        title="Treatment",
        sort=treatment_order,
        scale=alt.Scale(domain=treatment_order),
        legend=alt.Legend(
            orient="top",
            labelExpr="replace(datum.label, '_', ' ')",
        ),
    )
    x = alt.X(
        "dose:Q",
        title="Dose (µM)",
        scale=alt.Scale(type="log", domain=list(dose_domain)),
        axis=alt.Axis(grid=False, format="~g"),
    )
    y = alt.Y(
        "raw_signal:Q",
        title="Raw signal",
        scale=alt.Scale(domain=list(domain)),
        axis=alt.Axis(grid=True, gridColor="#E5E7EB"),
    )

    points = alt.Chart(treatment).mark_point(
        filled=True,
        opacity=0.45,
        size=55,
    ).encode(
        x=x,
        y=y,
        color=color,
        shape=shape,
        tooltip=[
            alt.Tooltip("plate_id:N", title="Plate"),
            alt.Tooltip("well:N", title="Well"),
            alt.Tooltip("treatment:N", title="Treatment"),
            alt.Tooltip("dose:Q", title="Dose (µM)", format=".4g"),
            alt.Tooltip("replicate:Q", title="Replicate"),
            alt.Tooltip("raw_signal:Q", title="Raw signal", format=".4f"),
        ],
    )
    error_bars = alt.Chart(treatment).mark_errorbar(
        extent="stdev",
        ticks=True,
    ).encode(x=x, y=y, color=color)
    means = (
        alt.Chart(treatment)
        .transform_aggregate(
            mean_signal="mean(raw_signal)",
            groupby=["plate_id", "treatment", "dose"],
        )
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=65), strokeWidth=2)
        .encode(
            x=x,
            y=alt.Y(
                "mean_signal:Q",
                title="Raw signal",
                scale=alt.Scale(domain=list(domain)),
            ),
            color=color,
            tooltip=[
                alt.Tooltip("plate_id:N", title="Plate"),
                alt.Tooltip("treatment:N", title="Treatment"),
                alt.Tooltip("dose:Q", title="Dose (µM)", format=".4g"),
                alt.Tooltip("mean_signal:Q", title="Mean", format=".4f"),
            ],
        )
    )
    layers = [*_control_layers(frame, dose_domain, domain), error_bars, points, means]
    chart = alt.layer(*layers).properties(width=560, height=360)
    return _facet_if_needed(
        chart,
        frame,
        _title(
            "Dose response",
            metadata,
            "points = replicates · lines = means · control bands = mean ± SD",
        ),
    )


def build_control_qc_chart(
    assay: AssayTables,
    metadata: PlotMetadata,
    *,
    signal_domain: tuple[float, float] | None = None,
) -> alt.LayerChart:
    """Build a control-distribution QC chart grouped by plate."""
    frame = _plot_frame(assay)
    domain = _validate_domain(signal_domain or shared_signal_domain(assay))
    controls = frame[
        frame["well_role"].isin(
            [WellRole.NEGATIVE_CONTROL.value, WellRole.POSITIVE_CONTROL.value]
        )
    ].copy()
    if controls.empty:
        raise ValueError("control QC chart requires control wells")
    role_order = [
        WellRole.NEGATIVE_CONTROL.value,
        WellRole.POSITIVE_CONTROL.value,
    ]
    plate_order = sorted(controls["plate_id"].unique())
    if len(plate_order) > len(PLATE_COLORS):
        raise ValueError("control QC chart supports at most five plates")
    x = alt.X(
        "well_role:N",
        title=None,
        sort=role_order,
        axis=alt.Axis(
            labelAngle=0,
            labelExpr="datum.label == 'negative_control' ? 'Negative control' : 'Positive control'",
        ),
    )
    offset = alt.XOffset("plate_id:N", sort=plate_order)
    color = alt.Color(
        "plate_id:N",
        title="Plate",
        sort=plate_order,
        scale=alt.Scale(domain=plate_order, range=PLATE_COLORS[: len(plate_order)]),
        legend=None if len(plate_order) == 1 else alt.Legend(orient="top"),
    )
    y = alt.Y(
        "raw_signal:Q",
        title="Raw signal",
        scale=alt.Scale(domain=list(domain)),
        axis=alt.Axis(grid=True, gridColor="#E5E7EB"),
    )
    points = alt.Chart(controls).mark_circle(
        opacity=0.5,
        size=65,
        stroke="#FFFFFF",
        strokeWidth=0.5,
    ).encode(
        x=x,
        xOffset=offset,
        y=y,
        color=color,
        tooltip=[
            alt.Tooltip("plate_id:N", title="Plate"),
            alt.Tooltip("well:N", title="Well"),
            alt.Tooltip("well_role:N", title="Role"),
            alt.Tooltip("raw_signal:Q", title="Raw signal", format=".4f"),
        ],
    )
    error_bars = alt.Chart(controls).mark_errorbar(
        extent="stdev",
        ticks=True,
        color="#111827",
    ).encode(x=x, xOffset=offset, y=y)
    means = alt.Chart(controls).mark_tick(
        thickness=3,
        size=24,
        color="#111827",
    ).encode(
        x=x,
        xOffset=offset,
        y=alt.Y("mean(raw_signal):Q", scale=alt.Scale(domain=list(domain))),
    )
    return alt.layer(error_bars, points, means).properties(
        width=460,
        height=340,
        title=_title(
            "Control QC",
            metadata,
            "points = wells · ticks = means · intervals = ±1 SD",
        ),
    )


def export_chart_png(
    chart: alt.TopLevelMixin,
    output: Path,
    *,
    scale: float = 2.0,
) -> Path:
    """Export an Altair chart to PNG locally without a browser or Node.js."""
    if output.suffix.lower() != ".png":
        raise ValueError("chart output path must end in .png")
    output.parent.mkdir(parents=True, exist_ok=True)
    png = vlc.vegalite_to_png(chart.to_dict(), scale=scale)
    output.write_bytes(png)
    return output


def export_plot_set(
    assay: AssayTables,
    metadata: PlotMetadata,
    output_directory: Path,
    *,
    signal_domain: tuple[float, float] | None = None,
    residual_domain: tuple[float, float] | None = None,
) -> dict[str, Path]:
    """Export the standard human-inspection plots for one assay."""
    domain = signal_domain or shared_signal_domain(assay)
    centered_domain = residual_domain or shared_residual_domain(assay)
    charts: dict[str, alt.TopLevelMixin] = {
        "plate_heatmap": build_plate_heatmap(assay, metadata, signal_domain=domain),
        "residual_heatmap": build_residual_heatmap(
            assay,
            metadata,
            residual_domain=centered_domain,
        ),
        "dose_response": build_dose_response_chart(
            assay, metadata, signal_domain=domain
        ),
        "control_qc": build_control_qc_chart(assay, metadata, signal_domain=domain),
    }
    return {
        name: export_chart_png(chart, output_directory / f"{name}.png")
        for name, chart in charts.items()
    }


def build_comparison_grid(
    clean: AssayTables,
    injected: AssayTables,
    metadata: PlotMetadata,
) -> alt.VConcatChart:
    """Build a two-row clean/injected grid of all four inspection views."""
    signal_domain = shared_signal_domain(clean, injected)
    residual_domain = shared_residual_domain(clean, injected)

    def row(assay: AssayTables, label: str) -> alt.HConcatChart:
        charts: list[alt.TopLevelMixin] = [
            build_plate_heatmap(assay, metadata, signal_domain=signal_domain),
            build_residual_heatmap(
                assay,
                metadata,
                residual_domain=residual_domain,
            ),
            build_dose_response_chart(
                assay,
                metadata,
                signal_domain=signal_domain,
            ),
            build_control_qc_chart(
                assay,
                metadata,
                signal_domain=signal_domain,
            ),
        ]
        return alt.hconcat(*charts, spacing=24).properties(
            title=alt.TitleParams(
                text=label,
                anchor="start",
                fontSize=18,
                color="#111827",
            )
        )

    return (
        alt.vconcat(row(clean, "Clean"), row(injected, "Injected"), spacing=32)
        .resolve_scale(color="independent", shape="independent")
        .properties(
            title=_title(
                "Clean / injected comparison",
                metadata,
                "rows = assay state · columns = inspection view",
            )
        )
    )


def export_comparison_grid(
    clean: AssayTables,
    injected: AssayTables,
    metadata: PlotMetadata,
    output: Path,
    *,
    scale: float = 1.0,
) -> Path:
    """Export the eight-panel clean/injected comparison grid as one PNG."""
    return export_chart_png(
        build_comparison_grid(clean, injected, metadata),
        output,
        scale=scale,
    )
