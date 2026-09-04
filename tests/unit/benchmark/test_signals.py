"""Tests for deterministic clean assay signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiment_failure_investigator.benchmark.layouts import build_balanced_layout
from experiment_failure_investigator.benchmark.models import (
    CaseVariant,
    FailureMode,
    GeneratorConfig,
    TreatmentCurve,
    WellRole,
)
from experiment_failure_investigator.benchmark.randomness import derive_child_seed
from experiment_failure_investigator.benchmark.signals import (
    BASELINE_NOISE_NAMESPACE,
    LATENT_SIGNAL_COLUMNS,
    MEASUREMENT_COLUMNS,
    four_parameter_logistic,
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
def layout(config: GeneratorConfig) -> pd.DataFrame:
    return build_balanced_layout(config)


def test_four_parameter_logistic_has_expected_midpoint() -> None:
    curve = TreatmentCurve(top=1.0, bottom=0.2, ic50=0.5, hill_slope=1.3)

    assert four_parameter_logistic(curve.ic50, curve) == pytest.approx(0.6)


def test_four_parameter_logistic_rejects_invalid_doses() -> None:
    curve = TreatmentCurve(ic50=1.0)

    for dose in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            four_parameter_logistic(dose, curve)


def test_clean_generation_has_public_and_private_columns(
    config: GeneratorConfig,
    layout: pd.DataFrame,
) -> None:
    result = generate_clean_assay(layout, config)

    assert list(result.measurements.columns) == MEASUREMENT_COLUMNS
    assert list(result.latent_signals.columns) == LATENT_SIGNAL_COLUMNS
    assert len(result.measurements) == config.plate_format.capacity
    assert not result.measurements.isna().any().any()
    assert np.isfinite(result.measurements["raw_signal"]).all()
    assert np.isfinite(result.latent_signals["expected_signal"]).all()
    assert result.metadata.baseline_noise_seed == derive_child_seed(
        config.root_seed, BASELINE_NOISE_NAMESPACE
    )
    measurement_keys = set(
        result.measurements[["plate_id", "well"]].itertuples(index=False)
    )
    plate_map_keys = set(
        result.plate_map[["plate_id", "well"]].itertuples(index=False)
    )
    assert measurement_keys == plate_map_keys


def test_baseline_noisy_control_means_remain_ordered(
    config: GeneratorConfig,
    layout: pd.DataFrame,
) -> None:
    result = generate_clean_assay(layout, config)
    annotated = result.plate_map.merge(
        result.measurements,
        on=["plate_id", "well"],
        validate="one_to_one",
    )
    control_means = annotated.groupby("well_role")["raw_signal"].mean()

    assert (
        control_means[WellRole.NEGATIVE_CONTROL.value]
        > control_means[WellRole.POSITIVE_CONTROL.value]
    )


def test_clean_generation_is_deterministic_and_row_order_invariant(
    config: GeneratorConfig,
    layout: pd.DataFrame,
) -> None:
    first = generate_clean_assay(layout, config)
    shuffled = layout.sample(frac=1.0, random_state=17).reset_index(drop=True)
    second = generate_clean_assay(shuffled, config)

    pd.testing.assert_frame_equal(first.plate_map, second.plate_map)
    pd.testing.assert_frame_equal(first.measurements, second.measurements)
    pd.testing.assert_frame_equal(first.latent_signals, second.latent_signals)
    assert first.metadata == second.metadata


def test_root_seed_changes_noise_not_layout_or_latent_signal(
    config: GeneratorConfig,
    layout: pd.DataFrame,
) -> None:
    other_config = config.model_copy(update={"root_seed": 43})

    first = generate_clean_assay(layout, config)
    second = generate_clean_assay(layout, other_config)

    pd.testing.assert_frame_equal(first.plate_map, second.plate_map)
    pd.testing.assert_series_equal(
        first.latent_signals["expected_signal"],
        second.latent_signals["expected_signal"],
    )
    assert not first.measurements["raw_signal"].equals(
        second.measurements["raw_signal"]
    )
    assert first.metadata.baseline_noise_seed != second.metadata.baseline_noise_seed


def test_noiseless_controls_and_treatment_curves_follow_configuration(
    config: GeneratorConfig,
    layout: pd.DataFrame,
) -> None:
    noiseless_config = config.model_copy(update={"noise_sd": 0.0})
    result = generate_clean_assay(layout, noiseless_config)
    annotated = result.plate_map.merge(
        result.latent_signals,
        on=["plate_id", "well"],
        validate="one_to_one",
    )

    negative = annotated.loc[
        annotated["well_role"] == WellRole.NEGATIVE_CONTROL.value,
        "expected_signal",
    ]
    positive = annotated.loc[
        annotated["well_role"] == WellRole.POSITIVE_CONTROL.value,
        "expected_signal",
    ]
    assert (negative == noiseless_config.negative_control_mean).all()
    assert (positive == noiseless_config.positive_control_mean).all()
    assert negative.mean() > positive.mean()

    treatment_rows = annotated[
        annotated["well_role"] == WellRole.TREATMENT.value
    ]
    for treatment, group in treatment_rows.groupby("treatment"):
        curve = noiseless_config.curves[treatment]
        dose_response = (
            group.groupby("dose")["expected_signal"].first().sort_index()
        )
        assert (np.diff(dose_response.to_numpy()) < 0).all()
        assert (dose_response > curve.bottom).all()
        assert (dose_response < curve.top).all()

    assert (result.latent_signals["baseline_noise"] == 0.0).all()
    assert result.measurements["raw_signal"].equals(
        result.latent_signals["expected_signal"]
    )


def test_clean_generation_does_not_mutate_layout(
    config: GeneratorConfig,
    layout: pd.DataFrame,
) -> None:
    original = layout.copy(deep=True)

    generate_clean_assay(layout, config)

    pd.testing.assert_frame_equal(layout, original)


def test_empty_wells_require_an_explicit_readout_policy() -> None:
    empty_config = GeneratorConfig(
        case_id="empty_fixture",
        case_variant=CaseVariant.OBVIOUS,
        failure_mode=FailureMode.EDGE_EFFECT,
        root_seed=9,
        noise_sd=0.0,
        negative_control_count=12,
        positive_control_count=12,
        empty_count=8,
        replicates_per_condition=4,
    )
    empty_layout = build_balanced_layout(empty_config)

    with pytest.raises(ValueError, match="no modeled biological signal"):
        generate_clean_assay(empty_layout, empty_config)


def test_clean_generation_rejects_incomplete_or_unknown_layouts(
    config: GeneratorConfig,
    layout: pd.DataFrame,
) -> None:
    incomplete = layout.iloc[:-1].copy()
    with pytest.raises(ValueError, match="must contain"):
        generate_clean_assay(incomplete, config)

    unknown_role = layout.copy()
    unknown_role.loc[0, "well_role"] = "mystery"
    with pytest.raises(ValueError, match="unsupported well role"):
        generate_clean_assay(unknown_role, config)
