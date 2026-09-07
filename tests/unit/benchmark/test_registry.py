"""Tests for the version-controlled Week 1 case registry and batch workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

import experiment_failure_investigator.benchmark.registry as registry_module
from experiment_failure_investigator.benchmark.models import CaseVariant, FailureMode
from experiment_failure_investigator.benchmark.registry import (
    build_registered_case,
    generate_registered_cases,
    validate_registered_cases,
    week_1_case_registry,
)


def test_registry_contains_two_variants_for_each_failure_mode() -> None:
    registry = week_1_case_registry()

    assert len(registry) == 14
    assert len({definition.config.case_id for definition in registry}) == 14
    assert {
        (definition.config.failure_mode, definition.config.case_variant)
        for definition in registry
    } == {
        (mode, variant)
        for mode in FailureMode
        for variant in CaseVariant
    }


def test_registered_layout_fingerprint_split_is_explicit() -> None:
    generated = [
        build_registered_case(definition) for definition in week_1_case_registry()
    ]
    ordinary = {
        case.payload.metadata.layout_fingerprint
        for case in generated
        if case.definition.config.failure_mode is not FailureMode.LAYOUT_CONFOUNDING
    }
    confounded = {
        case.payload.metadata.layout_fingerprint
        for case in generated
        if case.definition.config.failure_mode is FailureMode.LAYOUT_CONFOUNDING
    }

    assert len(ordinary) == 1
    assert len(confounded) == 1
    assert ordinary.isdisjoint(confounded)


def _fake_plot_writer(_generated):
    def write(directory: Path) -> dict[str, Path]:
        paths = {}
        for name in (
            "plate_heatmap",
            "residual_heatmap",
            "dose_response",
            "control_qc",
            "comparison_grid",
        ):
            path = directory / f"{name}.png"
            path.write_bytes(f"synthetic {name}\n".encode())
            paths[name] = path
        return paths

    return write


def test_batch_generation_writes_index_review_page_and_valid_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_module, "_plot_writer", _fake_plot_writer)

    generated = generate_registered_cases(tmp_path)
    validated = validate_registered_cases(tmp_path)

    assert len(generated) == len(validated) == 14
    assert (tmp_path / "index.json").is_file()
    assert (tmp_path / "review.html").is_file()
    assert "comparison_grid.png" in (tmp_path / "review.html").read_text()
    with pytest.raises(FileExistsError, match="not empty"):
        generate_registered_cases(tmp_path)


def test_validation_rejects_missing_or_extra_case_directories(
    tmp_path: Path,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "unexpected_case").mkdir()

    with pytest.raises(ValueError, match="case directory set mismatch"):
        validate_registered_cases(tmp_path)
