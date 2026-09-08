"""Tests for Altair benchmark inspection charts and local PNG export."""

from __future__ import annotations

from pathlib import Path
from struct import unpack
from types import SimpleNamespace

import pandas as pd
import pytest

from experiment_failure_investigator.benchmark.injectors import (
    BatchShiftParameters,
    EdgeEffectParameters,
    inject_batch_shift,
    inject_edge_effect,
)
from experiment_failure_investigator.benchmark.layouts import build_balanced_layout
from experiment_failure_investigator.benchmark.models import (
    CaseVariant,
    FailureMode,
    GeneratorConfig,
)
from experiment_failure_investigator.benchmark.plotting import (
    PlotMetadata,
    build_comparison_grid,
    build_control_qc_chart,
    build_dose_response_chart,
    build_plate_heatmap,
    build_residual_heatmap,
    export_plot_set,
    shared_residual_domain,
    shared_signal_domain,
)
from experiment_failure_investigator.benchmark.signals import (
    CleanAssayResult,
    generate_clean_assay,
)


@pytest.fixture(scope="module")
def config() -> GeneratorConfig:
    return GeneratorConfig(
        case_id="edge_effect_obvious",
        case_variant=CaseVariant.OBVIOUS,
        failure_mode=FailureMode.EDGE_EFFECT,
        root_seed=42,
        noise_sd=0.03,
    )


@pytest.fixture(scope="module")
def clean(config: GeneratorConfig) -> CleanAssayResult:
    return generate_clean_assay(build_balanced_layout(config), config)


@pytest.fixture(scope="module")
def edge_failure(clean: CleanAssayResult):
    return inject_edge_effect(clean, EdgeEffectParameters(magnitude=0.25))


@pytest.fixture(scope="module")
def metadata(config: GeneratorConfig) -> PlotMetadata:
    return PlotMetadata(
        case_id=config.case_id,
        failure_label=config.failure_mode.value,
        variant=config.case_variant.value,
        root_seed=config.root_seed,
    )


def test_shared_signal_domain_covers_clean_and_failure(
    clean: CleanAssayResult,
    edge_failure,
) -> None:
    domain = shared_signal_domain(clean, edge_failure)

    assert domain[0] < min(
        clean.measurements["raw_signal"].min(),
        edge_failure.measurements["raw_signal"].min(),
    )
    assert domain[1] > max(
        clean.measurements["raw_signal"].max(),
        edge_failure.measurements["raw_signal"].max(),
    )


def test_plate_heatmap_has_fixed_orientation_and_shared_scale(
    clean: CleanAssayResult,
    edge_failure,
    metadata: PlotMetadata,
) -> None:
    domain = shared_signal_domain(clean, edge_failure)
    spec = build_plate_heatmap(
        edge_failure,
        metadata,
        signal_domain=domain,
    ).to_dict()
    encoding = spec["layer"][0]["encoding"]

    assert spec["title"]["text"] == "Plate heatmap — edge_effect_obvious"
    assert "Root seed: 42" in spec["title"]["subtitle"][1]
    assert encoding["x"]["field"] == "column"
    assert encoding["x"]["sort"] == list(range(1, 13))
    assert encoding["y"]["field"] == "row"
    assert encoding["y"]["sort"] == list("ABCDEFGH")
    assert encoding["color"]["scale"]["domain"] == list(domain)
    assert encoding["color"]["scale"]["scheme"] == "viridis"


def test_empty_wells_are_gray_and_do_not_set_signal_or_residual_domains(
    clean: CleanAssayResult,
    metadata: PlotMetadata,
) -> None:
    plate_map = clean.plate_map.copy()
    measurements = clean.measurements.copy()
    empty_index = plate_map.index[0]
    empty_well = plate_map.loc[empty_index, "well"]
    plate_map.loc[empty_index, "well_role"] = "empty"
    plate_map.loc[empty_index, ["treatment", "dose", "dose_unit", "replicate"]] = None
    measurements.loc[measurements["well"] == empty_well, "raw_signal"] = 999.0
    assay = SimpleNamespace(plate_map=plate_map, measurements=measurements)

    domain = shared_signal_domain(assay)
    spec = build_plate_heatmap(assay, metadata, signal_domain=domain).to_dict()
    residual_domain = shared_residual_domain(assay)

    assert domain[1] < 2
    assert residual_domain[1] < 2
    assert "#D1D5DB" in str(spec)
    assert "empty" in str(spec["layer"][1]["encoding"]["text"])


def test_residual_heatmap_uses_symmetric_diverging_scale(
    clean: CleanAssayResult,
    edge_failure,
    metadata: PlotMetadata,
) -> None:
    domain = shared_residual_domain(clean, edge_failure)
    spec = build_residual_heatmap(
        edge_failure,
        metadata,
        residual_domain=domain,
    ).to_dict()
    color = spec["layer"][0]["encoding"]["color"]

    assert domain[0] == pytest.approx(-domain[1])
    assert color["field"] == "condition_residual"
    assert color["scale"]["domainMid"] == 0
    assert color["scale"]["scheme"] == "blueorange"


def test_dose_response_has_log_doses_replicates_means_and_controls(
    clean: CleanAssayResult,
    metadata: PlotMetadata,
) -> None:
    spec = build_dose_response_chart(clean, metadata).to_dict()
    layers = spec["layer"]
    point_layer = next(layer for layer in layers if layer.get("mark", {}).get("type") == "point")
    mean_layer = next(layer for layer in layers if "transform" in layer)

    assert point_layer["encoding"]["x"]["scale"]["type"] == "log"
    assert point_layer["encoding"]["x"]["scale"]["domain"] == [0.003, 10.0]
    assert point_layer["encoding"]["shape"]["field"] == "treatment"
    assert point_layer["encoding"]["shape"]["legend"]["orient"] == "top"
    assert mean_layer["transform"][0]["aggregate"][0] == {
        "op": "mean",
        "field": "raw_signal",
        "as": "mean_signal",
    }
    assert sum(layer.get("mark", {}).get("type") == "rect" for layer in layers) == 2
    assert sum(layer.get("mark", {}).get("type") == "rule" for layer in layers) == 2

def test_control_qc_has_boxplots_wells_and_means(
    clean: CleanAssayResult,
    metadata: PlotMetadata,
) -> None:
    spec = build_control_qc_chart(clean, metadata).to_dict()
    mark_types = [layer["mark"]["type"] for layer in spec["layer"]]

    assert mark_types == ["boxplot", "circle", "tick"]
    assert spec["layer"][2]["encoding"]["y"]["aggregate"] == "mean"


def test_dose_response_can_overlay_fitted_curves(
    clean: CleanAssayResult,
    metadata: PlotMetadata,
) -> None:
    fit_curves = pd.DataFrame(
        {
            "plate_id": ["plate_01", "plate_01"],
            "treatment": ["test_treatment", "test_treatment"],
            "dose": [0.003, 10.0],
            "fitted_signal": [1.0, 0.2],
        }
    )

    spec = build_dose_response_chart(
        clean,
        metadata,
        fit_curves=fit_curves,
    ).to_dict()
    fitted_layer = spec["layer"][-1]

    assert fitted_layer["mark"]["type"] == "line"
    assert fitted_layer["mark"]["strokeDash"] == [7, 4]
    assert fitted_layer["encoding"]["y"]["field"] == "fitted_signal"


def test_comparison_grid_has_clean_and_injected_rows_with_four_views(
    clean: CleanAssayResult,
    edge_failure,
    metadata: PlotMetadata,
) -> None:
    spec = build_comparison_grid(clean, edge_failure, metadata).to_dict()

    assert len(spec["vconcat"]) == 2
    assert [row["title"]["text"] for row in spec["vconcat"]] == [
        "Clean",
        "Injected",
    ]
    assert all(len(row["hconcat"]) == 4 for row in spec["vconcat"])


def test_multi_plate_charts_facet_by_plate() -> None:
    config = GeneratorConfig(
        case_id="batch_shift_obvious",
        case_variant=CaseVariant.OBVIOUS,
        failure_mode=FailureMode.BATCH_SHIFT,
        root_seed=81,
        noise_sd=0.03,
        plate_count=2,
    )
    layout = pd.concat(
        [
            build_balanced_layout(config, plate_id="plate_01"),
            build_balanced_layout(config, plate_id="plate_02"),
        ],
        ignore_index=True,
    )
    clean = generate_clean_assay(layout, config)
    shifted = inject_batch_shift(
        clean,
        BatchShiftParameters(
            shifted_plate_id="plate_02",
            response_scale_factor=0.70,
        ),
    )
    metadata = PlotMetadata(
        case_id=config.case_id,
        failure_label=config.failure_mode.value,
        variant=config.case_variant.value,
        root_seed=config.root_seed,
    )

    spec = build_plate_heatmap(shifted, metadata).to_dict()
    dose_spec = build_dose_response_chart(shifted, metadata).to_dict()

    assert spec["facet"]["column"]["field"] == "plate_id"
    assert spec["resolve"]["scale"]["color"] == "shared"
    assert len(dose_spec["hconcat"]) == 2
    assert [panel["title"]["text"] for panel in dose_spec["hconcat"]] == [
        "plate_01",
        "plate_02",
    ]


def test_chart_data_does_not_expose_private_generation_truth(
    edge_failure,
    metadata: PlotMetadata,
) -> None:
    spec = build_residual_heatmap(edge_failure, metadata).to_dict()
    rows = [row for dataset in spec["datasets"].values() for row in dataset]

    assert rows
    for private_field in ("expected_signal", "baseline_noise", "injected_effect"):
        assert all(private_field not in row for row in rows)


def test_plot_set_exports_decodable_png_files(
    tmp_path: Path,
    clean: CleanAssayResult,
    edge_failure,
    metadata: PlotMetadata,
) -> None:
    signal_domain = shared_signal_domain(clean, edge_failure)
    residual_domain = shared_residual_domain(clean, edge_failure)

    paths = export_plot_set(
        edge_failure,
        metadata,
        tmp_path,
        signal_domain=signal_domain,
        residual_domain=residual_domain,
    )

    assert set(paths) == {
        "plate_heatmap",
        "residual_heatmap",
        "dose_response",
        "control_qc",
    }
    for path in paths.values():
        payload = path.read_bytes()
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = unpack(">II", payload[16:24])
        assert width > 100
        assert height > 100


def test_plotting_rejects_mismatched_keys_and_invalid_domains(
    clean: CleanAssayResult,
    metadata: PlotMetadata,
) -> None:
    invalid = CleanAssayResult(
        plate_map=clean.plate_map,
        measurements=clean.measurements.iloc[:-1],
        latent_signals=clean.latent_signals,
        metadata=clean.metadata,
    )

    with pytest.raises(ValueError, match="keys do not match"):
        build_plate_heatmap(invalid, metadata)
    with pytest.raises(ValueError, match="increasing finite"):
        build_plate_heatmap(clean, metadata, signal_domain=(1.0, 1.0))
