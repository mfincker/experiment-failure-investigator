"""Tests for investigator-safe heatmap artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from experiment_failure_investigator.analysis.contracts import PublicArtifactHashes
from experiment_failure_investigator.analysis.design import build_investigator_case
from experiment_failure_investigator.analysis.heatmaps import (
    DiagnosticPlotKind,
    generate_diagnostic_plot,
    generate_plate_heatmap,
)
from experiment_failure_investigator.analysis.results import ToolStatus, canonical_json
from experiment_failure_investigator.benchmark.adapter import load_investigator_case
from experiment_failure_investigator.benchmark.serialization import load_case

ZERO_HASH = "0" * 64
HASHES = PublicArtifactHashes(
    measurements=ZERO_HASH,
    plate_map=ZERO_HASH,
    metadata=ZERO_HASH,
    protocol=ZERO_HASH,
    problem_statement=ZERO_HASH,
)


def _case_with_roles(
    roles: set[str],
    *,
    null_measurements_for: set[str] | None = None,
    treatment_doses: set[float] | None = None,
):
    loaded = load_case(Path("cases/edge_effect_obvious"))
    selected = loaded.plate_map["well_role"].isin(roles)
    if treatment_doses is not None:
        selected &= (loaded.plate_map["well_role"] != "treatment") | (
            loaded.plate_map["dose"].isin(treatment_doses)
        )
    plate_map = loaded.plate_map[selected].copy()
    retained_keys = pd.MultiIndex.from_frame(plate_map[["plate_id", "well"]])
    measurements = (
        loaded.measurements.set_index(["plate_id", "well"])
        .loc[retained_keys]
        .reset_index()
    )
    if null_measurements_for:
        null_wells = set(
            plate_map.loc[
                plate_map["well_role"].isin(null_measurements_for), "well"
            ]
        )
        measurements.loc[measurements["well"].isin(null_wells), "raw_signal"] = None
    return build_investigator_case(
        measurements=measurements,
        plate_map=plate_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )


@pytest.mark.parametrize(
    ("kind", "filename"),
    [
        ("raw_signal", "raw.png"),
        ("condition_residual", "residual.png"),
        ("control_distribution", "controls.png"),
        ("dose_response", "dose_response.png"),
    ],
)
def test_diagnostic_plot_export_is_hashed_public_attachment(
    tmp_path: Path,
    kind: str,
    filename: str,
) -> None:
    case = load_investigator_case(Path("cases/edge_effect_obvious"))

    result = generate_diagnostic_plot(
        case,
        tmp_path,
        kind=cast(DiagnosticPlotKind, kind),
        filename=filename,
    )
    repeated = generate_diagnostic_plot(
        case,
        tmp_path / "repeat",
        kind=cast(DiagnosticPlotKind, kind),
        filename=filename,
    )

    assert result.status is ToolStatus.SUCCESS
    assert result.evidence == ()
    assert len(result.attachments) == 1
    attachment = result.attachments[0]
    assert (tmp_path / attachment.path).read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    assert len(attachment.sha256) == 64
    assert repeated.attachments[0].sha256 == attachment.sha256
    if kind == "dose_response":
        assert result.parameters["fitted_curve_count"] == 2
    serialized = canonical_json(result)
    for private_name in (
        "failure_label",
        "variant",
        "root_seed",
        "injection_parameters",
        "ground_truth",
        "expected_signal",
    ):
        assert private_name not in serialized


def test_heatmap_png_is_deterministic_for_same_input_and_environment(
    tmp_path: Path,
) -> None:
    case = load_investigator_case(Path("cases/edge_effect_obvious"))
    first = generate_plate_heatmap(case, tmp_path / "one")
    second = generate_plate_heatmap(case, tmp_path / "two")

    assert first.attachments[0].sha256 == second.attachments[0].sha256
    assert (tmp_path / "one" / first.attachments[0].path).read_bytes() == (
        tmp_path / "two" / second.attachments[0].path
    ).read_bytes()


@pytest.mark.parametrize(
    "filename",
    ["../escape.png", "/tmp/escape.png", "nested/escape.png", "escape.svg"],
)
def test_invalid_or_escaping_output_names_fail_closed(
    tmp_path: Path,
    filename: str,
) -> None:
    case = load_investigator_case(Path("cases/edge_effect_obvious"))

    with pytest.raises(ValueError, match="non-escaping .png basename"):
        generate_plate_heatmap(case, tmp_path, filename=filename)


@pytest.mark.parametrize(
    ("kind", "roles"),
    [
        ("control_distribution", {"treatment"}),
        ("dose_response", {"negative_control", "positive_control"}),
    ],
)
def test_absent_plot_concept_is_not_applicable(
    tmp_path: Path,
    kind: str,
    roles: set[str],
) -> None:
    case = _case_with_roles(roles)

    result = generate_diagnostic_plot(
        case,
        tmp_path,
        kind=cast(DiagnosticPlotKind, kind),
        filename=f"{kind}.png",
    )

    assert result.status is ToolStatus.NOT_APPLICABLE
    assert result.attachments == ()


@pytest.mark.parametrize(
    ("kind", "omitted_role"),
    [
        ("control_distribution", "negative_control"),
        ("dose_response", "treatment"),
    ],
)
def test_mapped_plot_concept_without_finite_data_is_insufficient(
    tmp_path: Path,
    kind: str,
    omitted_role: str,
) -> None:
    roles = {"treatment"} if omitted_role == "treatment" else {omitted_role}
    case = _case_with_roles(roles, null_measurements_for={omitted_role})

    result = generate_diagnostic_plot(
        case,
        tmp_path,
        kind=cast(DiagnosticPlotKind, kind),
        filename=f"{kind}.png",
    )

    assert result.status is ToolStatus.INSUFFICIENT_DATA
    assert result.attachments == ()


def test_dose_plot_remains_available_when_fitted_curve_is_not_estimable(
    tmp_path: Path,
) -> None:
    case = _case_with_roles(
        {"negative_control", "positive_control", "treatment"},
        treatment_doses={0.003},
    )

    result = generate_diagnostic_plot(
        case,
        tmp_path,
        kind="dose_response",
        filename="dose_response.png",
    )

    assert result.status is ToolStatus.SUCCESS
    assert result.parameters["fitted_curve_count"] == 0
    assert {warning.code for warning in result.warnings} == {
        "fitted_curve_unavailable"
    }
    assert (tmp_path / "dose_response.png").is_file()
