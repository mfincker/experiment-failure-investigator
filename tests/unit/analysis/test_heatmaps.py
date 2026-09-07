"""Tests for investigator-safe heatmap artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from experiment_failure_investigator.analysis.heatmaps import (
    HeatmapView,
    generate_plate_heatmap,
)
from experiment_failure_investigator.analysis.results import ToolStatus, canonical_json
from experiment_failure_investigator.benchmark.adapter import load_investigator_case


@pytest.mark.parametrize(
    ("view", "filename"),
    [
        ("raw_signal", "raw.png"),
        ("condition_residual", "residual.png"),
    ],
)
def test_heatmap_export_is_hashed_public_attachment(
    tmp_path: Path,
    view: str,
    filename: str,
) -> None:
    case = load_investigator_case(Path("cases/edge_effect_obvious"))

    result = generate_plate_heatmap(
        case,
        tmp_path,
        filename=filename,
        view=cast(HeatmapView, view),
    )

    assert result.status is ToolStatus.SUCCESS
    assert result.evidence == ()
    assert len(result.attachments) == 1
    attachment = result.attachments[0]
    assert Path(attachment.path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(attachment.sha256) == 64
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
    assert Path(first.attachments[0].path).read_bytes() == Path(
        second.attachments[0].path
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
