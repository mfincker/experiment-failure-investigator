"""Tests for geometry-aware balanced plate layouts."""

from __future__ import annotations

import pandas as pd
import pytest

from experiment_failure_investigator.benchmark.layouts import (
    PLATE_MAP_COLUMNS,
    LayoutAudit,
    audit_layout,
    build_balanced_layout,
    derive_child_seed,
    enumerate_wells,
    render_layout_grid,
)
from experiment_failure_investigator.benchmark.models import (
    CaseVariant,
    FailureMode,
    GeneratorConfig,
    PlateFormat,
)


EXPECTED_BASELINE_GRID = """\
| Row | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | T1D7 | NC | PC | T1D6 | T2D3 | T2D1 | T1D3 | T2D2 | T2D7 | T2D5 | T2D8 | T1D4 |
| B | T1D2 | PC | T1D5 | T1D7 | T2D2 | T1D1 | NC | T1D8 | T2D6 | T1D3 | T2D7 | T2D4 |
| C | T2D3 | T1D7 | T1D2 | NC | PC | T2D4 | T2D1 | T1D4 | T2D7 | T1D5 | T1D1 | T2D6 |
| D | T1D8 | T1D1 | NC | T1D2 | T1D5 | T2D5 | PC | T2D3 | T2D8 | T2D6 | T1D6 | T2D1 |
| E | T2D8 | T2D7 | T2D5 | T1D4 | T2D6 | T1D3 | T1D5 | PC | T2D2 | T2D3 | NC | T1D7 |
| F | T1D6 | T1D3 | T2D3 | T2D8 | T1D7 | T2D4 | PC | T1D8 | T2D5 | NC | T1D1 | T1D5 |
| G | T1D1 | T1D6 | T1D2 | PC | T2D8 | NC | T2D1 | T2D5 | T1D8 | T1D4 | T2D4 | T2D2 |
| H | T1D8 | T1D2 | T2D7 | T2D2 | T1D6 | T2D1 | T1D3 | T2D6 | PC | T2D4 | T1D4 | NC |"""


@pytest.fixture(scope="module")
def baseline_config() -> GeneratorConfig:
    return GeneratorConfig(
        case_id="edge_effect_obvious",
        case_variant=CaseVariant.OBVIOUS,
        failure_mode=FailureMode.EDGE_EFFECT,
        root_seed=42,
        noise_sd=0.03,
    )


@pytest.fixture(scope="module")
def baseline_layout(baseline_config: GeneratorConfig) -> pd.DataFrame:
    return build_balanced_layout(baseline_config)


@pytest.fixture(scope="module")
def baseline_audit(
    baseline_layout: pd.DataFrame, baseline_config: GeneratorConfig
) -> LayoutAudit:
    return audit_layout(baseline_layout, baseline_config)


@pytest.mark.parametrize(
    ("plate_format", "first", "last", "count"),
    [
        (PlateFormat.WELLS_96, "A01", "H12", 96),
        (PlateFormat.WELLS_384, "A01", "P24", 384),
    ],
)
def test_well_enumeration_uses_declared_geometry(
    plate_format: PlateFormat, first: str, last: str, count: int
) -> None:
    wells = enumerate_wells(plate_format)

    assert len(wells) == count
    assert wells.iloc[0]["well"] == first
    assert wells.iloc[-1]["well"] == last
    assert wells["well"].is_unique


def test_edge_status_is_derived_from_geometry() -> None:
    wells = enumerate_wells(PlateFormat.WELLS_96).set_index("well")

    assert bool(wells.loc["A06", "is_edge"])
    assert bool(wells.loc["D01", "is_edge"])
    assert not bool(wells.loc["D06", "is_edge"])


def test_layout_contains_expected_rows_and_columns(
    baseline_layout: pd.DataFrame,
) -> None:
    assert list(baseline_layout.columns) == PLATE_MAP_COLUMNS
    assert baseline_layout["well"].tolist() == enumerate_wells(
        PlateFormat.WELLS_96
    )["well"].tolist()
    assert baseline_layout["well"].is_unique


def test_layout_has_expected_condition_counts(
    baseline_audit: LayoutAudit,
) -> None:
    assert baseline_audit.role_counts == {
        "negative_control": 8,
        "positive_control": 8,
        "treatment": 80,
    }
    assert baseline_audit.treatment_counts == {
        "reference_treatment": 40,
        "test_treatment": 40,
    }
    assert set(baseline_audit.condition_replicate_counts.values()) == {5}
    assert len(baseline_audit.condition_replicate_counts) == 16


def test_layout_distributes_controls_and_replicates(
    baseline_audit: LayoutAudit,
) -> None:
    assert baseline_audit.condition_spatial_coverage["negative_control"]["rows"] == 8
    assert baseline_audit.condition_spatial_coverage["positive_control"]["rows"] == 8

    treatment_conditions = {
        key: value
        for key, value in baseline_audit.condition_spatial_coverage.items()
        if "@" in key
    }
    for coverage in treatment_conditions.values():
        assert coverage["rows"] == 5
        assert coverage["columns"] >= 3
        assert coverage["edge"] >= 1
        assert coverage["interior"] >= 1


def test_independent_audit_accepts_baseline_layout(
    baseline_audit: LayoutAudit,
) -> None:
    assert baseline_audit.is_valid
    assert baseline_audit.duplicate_wells == ()
    assert baseline_audit.missing_wells == ()
    assert baseline_audit.potential_confounding == ()


def test_layout_is_deterministic_for_seed(
    baseline_config: GeneratorConfig,
) -> None:
    first = build_balanced_layout(baseline_config, candidate_count=100)
    second = build_balanced_layout(baseline_config, candidate_count=100)

    pd.testing.assert_frame_equal(first, second)
    assert derive_child_seed(42, "layout") == derive_child_seed(42, "layout")


def test_case_seed_does_not_change_shared_baseline_layout(
    baseline_config: GeneratorConfig,
) -> None:
    other_config = baseline_config.model_copy(
        update={"case_id": "edge_effect_noisy", "root_seed": 43}
    )

    first = build_balanced_layout(baseline_config, candidate_count=100)
    second = build_balanced_layout(other_config, candidate_count=100)

    pd.testing.assert_frame_equal(first, second)


def test_layout_changes_with_layout_seed(baseline_config: GeneratorConfig) -> None:
    other_config = baseline_config.model_copy(update={"layout_seed": 43})

    first = build_balanced_layout(baseline_config, candidate_count=100)
    second = build_balanced_layout(other_config, candidate_count=100)

    assert not first["sample_id"].equals(second["sample_id"])
    assert audit_layout(second, other_config).is_valid


def test_audit_detects_duplicate_and_missing_wells(
    baseline_layout: pd.DataFrame, baseline_config: GeneratorConfig
) -> None:
    invalid = baseline_layout.copy()
    invalid.loc[0, "well"] = invalid.loc[1, "well"]

    audit = audit_layout(invalid, baseline_config)

    assert not audit.is_valid
    assert audit.duplicate_wells
    assert audit.missing_wells == ("A01",)


def test_audit_detects_condition_confined_to_one_row(
    baseline_layout: pd.DataFrame, baseline_config: GeneratorConfig
) -> None:
    invalid = baseline_layout.copy()
    condition_mask = (
        (invalid["treatment"] == "reference_treatment")
        & (invalid["dose"] == 0.003)
    )
    condition_indices = invalid.index[condition_mask].tolist()
    replacement_indices = invalid.index[
        (invalid["row"] == "A") & ~condition_mask
    ].tolist()[: len(condition_indices)]
    assignment_columns = [
        "sample_id",
        "well_role",
        "treatment",
        "dose",
        "dose_unit",
        "replicate",
    ]
    condition_assignments = invalid.loc[
        condition_indices, assignment_columns
    ].to_numpy(copy=True)
    replacement_assignments = invalid.loc[
        replacement_indices, assignment_columns
    ].to_numpy(copy=True)
    invalid.loc[condition_indices, assignment_columns] = replacement_assignments
    invalid.loc[replacement_indices, assignment_columns] = condition_assignments

    audit = audit_layout(invalid, baseline_config)

    assert not audit.is_valid
    assert (
        "reference_treatment@0.003 is confined to one row"
        in audit.potential_confounding
    )


def test_layout_grid_is_readable(baseline_layout: pd.DataFrame) -> None:
    rendered = render_layout_grid(baseline_layout)

    assert rendered == EXPECTED_BASELINE_GRID
