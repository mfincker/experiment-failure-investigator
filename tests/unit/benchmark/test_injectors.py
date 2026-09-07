"""Tests for pure synthetic failure injectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from experiment_failure_investigator.benchmark.injectors import (
    BatchShiftParameters,
    EdgeEffectParameters,
    LayoutConfoundingParameters,
    PipettingDriftParameters,
    TransientTipClogParameters,
    TrueNonResponseParameters,
    WeakControlsParameters,
    inject_batch_shift,
    inject_edge_effect,
    inject_layout_confounding,
    inject_pipetting_drift,
    inject_transient_tip_clog,
    inject_true_non_response,
    inject_weak_controls,
)
from experiment_failure_investigator.benchmark.layouts import (
    audit_layout,
    build_balanced_layout,
)
from experiment_failure_investigator.benchmark.models import (
    CaseVariant,
    FailureMode,
    GeneratorConfig,
    SimulatedTraversal,
    WellRole,
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
def multi_plate_clean() -> CleanAssayResult:
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
    return generate_clean_assay(layout, config)


def assert_clean_unchanged(
    clean: CleanAssayResult,
    original_plate_map: pd.DataFrame,
    original_measurements: pd.DataFrame,
    original_latent: pd.DataFrame,
) -> None:
    pd.testing.assert_frame_equal(clean.plate_map, original_plate_map)
    pd.testing.assert_frame_equal(clean.measurements, original_measurements)
    pd.testing.assert_frame_equal(clean.latent_signals, original_latent)


def test_edge_effect_changes_only_perimeter_wells(
    clean: CleanAssayResult,
) -> None:
    result = inject_edge_effect(
        clean,
        EdgeEffectParameters(magnitude=0.25, direction="increase"),
    )
    effect = result.latent_signals["injected_effect"]
    row_edge = result.plate_map["row"].isin(["A", "H"])
    column_edge = result.plate_map["column"].isin([1, 12])
    edge = row_edge | column_edge

    assert (effect[edge] == 0.25).all()
    assert (effect[~edge] == 0.0).all()
    assert result.injection.affected_well_count == 36
    assert result.injection.mechanism is FailureMode.EDGE_EFFECT


@pytest.mark.parametrize(
    ("sides", "expected_count"),
    [
        (("top",), 12),
        (("left", "right"), 16),
        (("top", "bottom", "left"), 30),
        (("top", "bottom", "left", "right"), 36),
    ],
)
def test_edge_effect_supports_one_to_four_selected_sides(
    clean: CleanAssayResult,
    sides: tuple[str, ...],
    expected_count: int,
) -> None:
    result = inject_edge_effect(
        clean,
        EdgeEffectParameters(magnitude=0.25, sides=sides),
    )

    assert result.injection.affected_well_count == expected_count
    assert result.injection.parameters["sides"] == list(sides)


def test_edge_effect_requires_at_least_one_unique_side() -> None:
    with pytest.raises(ValueError, match="at least one edge side"):
        EdgeEffectParameters(magnitude=0.25, sides=())
    with pytest.raises(ValueError, match="edge sides must be unique"):
        EdgeEffectParameters(magnitude=0.25, sides=("top", "top"))


def test_pipetting_drift_follows_private_traversal(
    clean: CleanAssayResult,
) -> None:
    result = inject_pipetting_drift(
        clean,
        PipettingDriftParameters(
            magnitude=0.30,
            direction="increase",
            traversal=SimulatedTraversal.ROW_MAJOR,
        ),
    )
    effect = result.latent_signals["injected_effect"].to_numpy()
    row_major_rank = np.arange(len(effect))

    assert effect.min() == pytest.approx(-0.15)
    assert effect.max() == pytest.approx(0.15)
    assert effect.mean() == pytest.approx(0.0, abs=1e-12)
    assert np.corrcoef(row_major_rank, effect)[0, 1] == pytest.approx(1.0)
    assert result.injection.parameters["traversal"] == "row_major"


def test_transient_tip_clog_affects_channels_for_a_limited_group_span(
    clean: CleanAssayResult,
) -> None:
    result = inject_transient_tip_clog(
        clean,
        TransientTipClogParameters(
            magnitude=0.25,
            traversal=SimulatedTraversal.COLUMN_MAJOR,
            dispense_group_size=8,
            affected_group_start=2,
            affected_group_count=3,
            affected_channels=(1, 4),
        ),
    )
    affected = result.latent_signals["injected_effect"].ne(0)

    assert set(result.plate_map.loc[affected, "well"]) == {
        "B03",
        "E03",
        "B04",
        "E04",
        "B05",
        "E05",
    }
    assert (result.latent_signals.loc[affected, "injected_effect"] == -0.25).all()
    assert result.injection.affected_well_count == 6
    assert result.injection.mechanism is FailureMode.TRANSIENT_TIP_CLOG


def test_transient_tip_clog_rejects_invalid_group_geometry(
    clean: CleanAssayResult,
) -> None:
    with pytest.raises(ValueError, match="channel must be within"):
        inject_transient_tip_clog(
            clean,
            TransientTipClogParameters(
                magnitude=0.25,
                dispense_group_size=8,
                affected_group_start=0,
                affected_group_count=1,
                affected_channels=(8,),
            ),
        )
    with pytest.raises(ValueError, match="groups exceed"):
        inject_transient_tip_clog(
            clean,
            TransientTipClogParameters(
                magnitude=0.25,
                affected_group_start=11,
                affected_group_count=2,
            ),
        )


def test_layout_confounding_moves_assignments_with_their_signals(
    clean: CleanAssayResult,
    config: GeneratorConfig,
) -> None:
    result = inject_layout_confounding(
        clean,
        LayoutConfoundingParameters(order_by="treatment_then_dose"),
    )

    audit = audit_layout(result.plate_map, config)
    assert not audit.is_valid
    assert audit.potential_confounding
    assert not result.plate_map["sample_id"].equals(clean.plate_map["sample_id"])

    clean_by_sample = clean.plate_map[["sample_id", "plate_id", "well"]].merge(
        clean.latent_signals,
        on=["plate_id", "well"],
        validate="one_to_one",
    )
    injected_by_sample = result.plate_map[
        ["sample_id", "plate_id", "well"]
    ].merge(
        result.latent_signals,
        on=["plate_id", "well"],
        validate="one_to_one",
    )
    paired = clean_by_sample.merge(
        injected_by_sample,
        on="sample_id",
        suffixes=("_clean", "_injected"),
        validate="one_to_one",
    )
    assert np.array_equal(
        paired["expected_signal_clean"], paired["expected_signal_injected"]
    )
    assert np.array_equal(
        clean.latent_signals["baseline_noise"],
        result.latent_signals["baseline_noise"],
    )
    assert np.allclose(
        result.measurements["raw_signal"],
        result.latent_signals["expected_signal"]
        + result.latent_signals["baseline_noise"],
    )
    assert (result.latent_signals["injected_effect"] == 0.0).all()


def test_weak_controls_reduce_only_positive_control_separation(
    clean: CleanAssayResult,
) -> None:
    fraction = 0.8
    result = inject_weak_controls(
        clean,
        WeakControlsParameters(collapse_fraction=fraction),
    )
    roles = result.plate_map["well_role"]
    effect = result.latent_signals["injected_effect"]
    positive = roles == WellRole.POSITIVE_CONTROL.value
    effective_expected = (
        result.latent_signals["expected_signal"] + effect
    )
    negative_mean = effective_expected[
        roles == WellRole.NEGATIVE_CONTROL.value
    ].mean()
    original_positive_mean = clean.latent_signals.loc[
        positive, "expected_signal"
    ].mean()
    new_positive_mean = effective_expected[positive].mean()

    assert (effect[positive] > 0).all()
    assert (effect[~positive] == 0).all()
    assert new_positive_mean == pytest.approx(
        original_positive_mean
        + fraction * (negative_mean - original_positive_mean)
    )
    assert result.injection.affected_well_count == 8


def test_batch_shift_changes_only_selected_plate(
    multi_plate_clean: CleanAssayResult,
) -> None:
    result = inject_batch_shift(
        multi_plate_clean,
        BatchShiftParameters(
            shifted_plate_id="plate_02",
            response_scale_factor=0.7,
        ),
    )
    effect = result.latent_signals["injected_effect"]
    shifted = result.plate_map["plate_id"] == "plate_02"
    negative = shifted & (
        result.plate_map["well_role"] == WellRole.NEGATIVE_CONTROL.value
    )
    original = multi_plate_clean.measurements["raw_signal"]
    transformed = result.measurements["raw_signal"]
    anchor = original[negative].mean()

    assert (effect[~shifted] == 0.0).all()
    assert result.injection.affected_well_count == 96
    assert transformed[negative].mean() == pytest.approx(anchor)
    assert np.allclose(
        transformed[shifted] - anchor,
        0.7 * (original[shifted] - anchor),
    )
    assert result.injection.parameters["response_scale_factor"] == 0.7


def test_true_non_response_flattens_only_target_treatment(
    clean: CleanAssayResult,
) -> None:
    result = inject_true_non_response(
        clean,
        TrueNonResponseParameters(
            target_treatment="test_treatment",
            flat_expected_signal=1.0,
        ),
    )
    target = result.plate_map["treatment"] == "test_treatment"
    effective_expected = (
        result.latent_signals["expected_signal"]
        + result.latent_signals["injected_effect"]
    )

    assert np.allclose(effective_expected[target], 1.0)
    assert (result.latent_signals.loc[~target, "injected_effect"] == 0.0).all()
    assert result.injection.affected_well_count == 40


@pytest.mark.parametrize(
    ("injector", "parameters"),
    [
        (inject_edge_effect, EdgeEffectParameters(magnitude=0.25)),
        (
            inject_pipetting_drift,
            PipettingDriftParameters(
                magnitude=0.30,
                traversal=SimulatedTraversal.SERPENTINE_ROWS,
            ),
        ),
        (
            inject_transient_tip_clog,
            TransientTipClogParameters(
                magnitude=0.25,
                affected_group_start=2,
                affected_group_count=3,
            ),
        ),
        (inject_layout_confounding, LayoutConfoundingParameters()),
        (inject_weak_controls, WeakControlsParameters(collapse_fraction=0.8)),
        (
            inject_true_non_response,
            TrueNonResponseParameters(
                target_treatment="test_treatment",
                flat_expected_signal=1.0,
            ),
        ),
    ],
)
def test_single_plate_injectors_are_deterministic_and_pure(
    clean: CleanAssayResult,
    injector: object,
    parameters: object,
) -> None:
    original_plate_map = clean.plate_map.copy(deep=True)
    original_measurements = clean.measurements.copy(deep=True)
    original_latent = clean.latent_signals.copy(deep=True)

    first = injector(clean, parameters)  # type: ignore[operator]
    second = injector(clean, parameters)  # type: ignore[operator]

    pd.testing.assert_frame_equal(first.plate_map, second.plate_map)
    pd.testing.assert_frame_equal(first.measurements, second.measurements)
    pd.testing.assert_frame_equal(first.latent_signals, second.latent_signals)
    assert first.injection == second.injection
    assert_clean_unchanged(
        clean,
        original_plate_map,
        original_measurements,
        original_latent,
    )


def test_batch_shift_is_deterministic_and_pure(
    multi_plate_clean: CleanAssayResult,
) -> None:
    original_plate_map = multi_plate_clean.plate_map.copy(deep=True)
    original_measurements = multi_plate_clean.measurements.copy(deep=True)
    original_latent = multi_plate_clean.latent_signals.copy(deep=True)
    parameters = BatchShiftParameters(
        shifted_plate_id="plate_02",
        response_scale_factor=0.7,
    )

    first = inject_batch_shift(multi_plate_clean, parameters)
    second = inject_batch_shift(multi_plate_clean, parameters)

    pd.testing.assert_frame_equal(first.plate_map, second.plate_map)
    pd.testing.assert_frame_equal(first.measurements, second.measurements)
    pd.testing.assert_frame_equal(first.latent_signals, second.latent_signals)
    assert first.injection == second.injection
    assert_clean_unchanged(
        multi_plate_clean,
        original_plate_map,
        original_measurements,
        original_latent,
    )


def test_injector_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValidationError):
        EdgeEffectParameters(magnitude=0.0)
    with pytest.raises(ValidationError):
        WeakControlsParameters(collapse_fraction=1.1)
    with pytest.raises(ValidationError, match="must differ from 1"):
        BatchShiftParameters(
            shifted_plate_id="plate_02",
            response_scale_factor=1.0,
        )
    with pytest.raises(ValidationError):
        TrueNonResponseParameters(
            target_treatment=" ",
            flat_expected_signal=1.0,
        )


def test_batch_shift_and_non_response_require_valid_targets(
    clean: CleanAssayResult,
) -> None:
    with pytest.raises(ValueError, match="at least two plates"):
        inject_batch_shift(
            clean,
            BatchShiftParameters(
                shifted_plate_id="plate_01",
                response_scale_factor=0.8,
            ),
        )
    with pytest.raises(ValueError, match="not present"):
        inject_true_non_response(
            clean,
            TrueNonResponseParameters(
                target_treatment="unknown_treatment",
                flat_expected_signal=1.0,
            ),
        )
