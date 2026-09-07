"""Pure, typed failure injectors for clean synthetic assay data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, JsonValue, TypeAdapter, field_validator

from experiment_failure_investigator.benchmark.models import (
    FailureMode,
    SimulatedTraversal,
    StrictModel,
    WellRole,
)
from experiment_failure_investigator.benchmark.randomness import derive_child_seed
from experiment_failure_investigator.benchmark.signals import (
    CleanAssayResult,
    CleanGenerationMetadata,
    LATENT_SIGNAL_COLUMNS,
    MEASUREMENT_COLUMNS,
)

INJECTION_NAMESPACE = "failure_injection"
INJECTED_LATENT_SIGNAL_COLUMNS = [*LATENT_SIGNAL_COLUMNS, "injected_effect"]
JSON_PARAMETERS_ADAPTER = TypeAdapter(dict[str, JsonValue])


class DirectedMagnitude(StrictModel):
    """Positive magnitude plus an explicit direction."""

    magnitude: float = Field(gt=0, allow_inf_nan=False)
    direction: Literal["increase", "decrease"] = "increase"

    @property
    def signed_magnitude(self) -> float:
        """Return the signed effect represented by the two fields."""
        return self.magnitude if self.direction == "increase" else -self.magnitude


class EdgeEffectParameters(DirectedMagnitude):
    """Parameters for shifting wells on one or more selected plate sides."""

    sides: tuple[Literal["top", "bottom", "left", "right"], ...] = (
        "top",
        "bottom",
        "left",
        "right",
    )

    @field_validator("sides")
    @classmethod
    def sides_must_be_nonempty_and_unique(
        cls,
        value: tuple[Literal["top", "bottom", "left", "right"], ...],
    ) -> tuple[Literal["top", "bottom", "left", "right"], ...]:
        if not value:
            raise ValueError("at least one edge side is required")
        if len(value) != len(set(value)):
            raise ValueError("edge sides must be unique")
        return value


class PipettingDriftParameters(DirectedMagnitude):
    """Parameters for a traversal-aligned peak-to-peak gradient."""

    traversal: SimulatedTraversal = SimulatedTraversal.ROW_MAJOR


class TransientTipClogParameters(DirectedMagnitude):
    """Parameters for a tip-channel fault that clears after several groups."""

    direction: Literal["increase", "decrease"] = "decrease"
    traversal: SimulatedTraversal = SimulatedTraversal.COLUMN_MAJOR
    dispense_group_size: int = Field(default=8, ge=1)
    affected_group_start: int = Field(ge=0)
    affected_group_count: int = Field(ge=1)
    affected_channels: tuple[int, ...] = (0,)

    @field_validator("affected_channels")
    @classmethod
    def channels_must_be_nonempty_unique_and_nonnegative(
        cls,
        value: tuple[int, ...],
    ) -> tuple[int, ...]:
        if not value:
            raise ValueError("at least one affected channel is required")
        if len(value) != len(set(value)):
            raise ValueError("affected channels must be unique")
        if any(channel < 0 for channel in value):
            raise ValueError("affected channels must be nonnegative")
        return value


class LayoutConfoundingParameters(StrictModel):
    """Parameters controlling semantic condition ordering across the plate."""

    order_by: Literal["treatment_then_dose", "dose_then_treatment"] = (
        "treatment_then_dose"
    )


class WeakControlsParameters(StrictModel):
    """Fraction of positive-control separation removed."""

    collapse_fraction: float = Field(gt=0, le=1, allow_inf_nan=False)


class BatchShiftParameters(StrictModel):
    """Parameters for scaling one plate's response around its control anchor."""

    shifted_plate_id: str = Field(min_length=1)
    response_scale_factor: float = Field(gt=0, allow_inf_nan=False)

    @field_validator("response_scale_factor")
    @classmethod
    def scale_factor_must_change_response(cls, value: float) -> float:
        if value == 1:
            raise ValueError("response scale factor must differ from 1")
        return value


class TrueNonResponseParameters(StrictModel):
    """Parameters for replacing one treatment curve with a flat expectation."""

    target_treatment: str = Field(min_length=1)
    flat_expected_signal: float = Field(allow_inf_nan=False)

    @field_validator("target_treatment")
    @classmethod
    def treatment_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("target treatment must not be blank")
        return value


@dataclass(frozen=True)
class InjectionRecord:
    """Private structured truth for one injected mechanism."""

    mechanism: FailureMode
    child_seed: int
    parameters: dict[str, JsonValue]
    affected_well_count: int


@dataclass(frozen=True)
class InjectedAssayResult:
    """An injected assay plus its private structured injection record."""

    plate_map: pd.DataFrame
    measurements: pd.DataFrame
    latent_signals: pd.DataFrame
    generation_metadata: CleanGenerationMetadata
    injection: InjectionRecord


def _validate_clean_result(clean: CleanAssayResult) -> None:
    measurement_keys = clean.measurements[["plate_id", "well"]]
    latent_keys = clean.latent_signals[["plate_id", "well"]]
    plate_map_keys = clean.plate_map[["plate_id", "well"]]
    if not measurement_keys.equals(latent_keys) or not measurement_keys.equals(
        plate_map_keys
    ):
        raise ValueError("clean assay tables must have aligned canonical keys")
    if "injected_effect" in clean.latent_signals:
        raise ValueError("clean assay already contains an injected effect")


def _injection_seed(clean: CleanAssayResult) -> int:
    return derive_child_seed(clean.metadata.root_seed, INJECTION_NAMESPACE)


def _json_parameters(parameters: StrictModel) -> dict[str, JsonValue]:
    """Serialize and validate injector parameters as JSON-compatible values."""
    return JSON_PARAMETERS_ADAPTER.validate_python(
        parameters.model_dump(mode="json")
    )


def _apply_numeric_effect(
    clean: CleanAssayResult,
    effect: np.ndarray,
    *,
    mechanism: FailureMode,
    parameters: StrictModel,
) -> InjectedAssayResult:
    _validate_clean_result(clean)
    values = np.asarray(effect, dtype=float)
    if values.shape != (len(clean.measurements),):
        raise ValueError("injected effect must contain one value per measurement")
    if not np.isfinite(values).all():
        raise ValueError("injected effects must be finite")

    measurements = clean.measurements.copy(deep=True)
    measurements["raw_signal"] = measurements["raw_signal"].to_numpy() + values
    latent = clean.latent_signals.copy(deep=True)
    latent["injected_effect"] = values
    return InjectedAssayResult(
        plate_map=clean.plate_map.copy(deep=True),
        measurements=measurements.loc[:, MEASUREMENT_COLUMNS],
        latent_signals=latent.loc[:, INJECTED_LATENT_SIGNAL_COLUMNS],
        generation_metadata=clean.metadata,
        injection=InjectionRecord(
            mechanism=mechanism,
            child_seed=_injection_seed(clean),
            parameters=_json_parameters(parameters),
            affected_well_count=int(np.count_nonzero(values)),
        ),
    )


def inject_edge_effect(
    clean: CleanAssayResult,
    parameters: EdgeEffectParameters,
) -> InjectedAssayResult:
    """Shift wells on the configured plate sides by a signed magnitude."""
    plate_map = clean.plate_map
    grouped = plate_map.groupby("plate_id", sort=False)
    edge = pd.Series(False, index=plate_map.index)
    for _, plate in grouped:
        selected = pd.Series(False, index=plate.index)
        if "top" in parameters.sides:
            selected |= plate["row"] == plate["row"].min()
        if "bottom" in parameters.sides:
            selected |= plate["row"] == plate["row"].max()
        if "left" in parameters.sides:
            selected |= plate["column"] == plate["column"].min()
        if "right" in parameters.sides:
            selected |= plate["column"] == plate["column"].max()
        edge.loc[plate.index] = selected
    effect = np.where(edge.to_numpy(), parameters.signed_magnitude, 0.0)
    return _apply_numeric_effect(
        clean,
        effect,
        mechanism=FailureMode.EDGE_EFFECT,
        parameters=parameters,
    )


def _traversal_rank(
    row_index: np.ndarray,
    column_index: np.ndarray,
    row_count: int,
    column_count: int,
    traversal: SimulatedTraversal,
) -> np.ndarray:
    if traversal is SimulatedTraversal.ROW_MAJOR:
        return row_index * column_count + column_index
    if traversal is SimulatedTraversal.COLUMN_MAJOR:
        return column_index * row_count + row_index
    if traversal is SimulatedTraversal.SERPENTINE_ROWS:
        traversed_column = np.where(
            row_index % 2 == 0,
            column_index,
            column_count - 1 - column_index,
        )
        return row_index * column_count + traversed_column
    traversed_row = np.where(
        column_index % 2 == 0,
        row_index,
        row_count - 1 - row_index,
    )
    return column_index * row_count + traversed_row


def inject_pipetting_drift(
    clean: CleanAssayResult,
    parameters: PipettingDriftParameters,
) -> InjectedAssayResult:
    """Apply a zero-centered linear gradient along a private traversal."""
    effect = np.zeros(len(clean.plate_map), dtype=float)
    for _, plate in clean.plate_map.groupby("plate_id", sort=False):
        row_labels = sorted(plate["row"].unique())
        row_lookup = {row: index for index, row in enumerate(row_labels)}
        row_index = plate["row"].map(row_lookup).to_numpy(dtype=int)
        column_values = sorted(plate["column"].unique())
        column_lookup = {
            column: index for index, column in enumerate(column_values)
        }
        column_index = plate["column"].map(column_lookup).to_numpy(dtype=int)
        rank = _traversal_rank(
            row_index,
            column_index,
            len(row_labels),
            len(column_values),
            parameters.traversal,
        )
        centered = rank / (len(plate) - 1) - 0.5
        effect[plate.index] = parameters.signed_magnitude * centered
    return _apply_numeric_effect(
        clean,
        effect,
        mechanism=FailureMode.PIPETTING_DRIFT,
        parameters=parameters,
    )


def inject_transient_tip_clog(
    clean: CleanAssayResult,
    parameters: TransientTipClogParameters,
) -> InjectedAssayResult:
    """Affect selected tip channels temporarily across dispense groups."""
    effect = np.zeros(len(clean.plate_map), dtype=float)
    for _, plate in clean.plate_map.groupby("plate_id", sort=False):
        row_labels = sorted(plate["row"].unique())
        row_lookup = {row: index for index, row in enumerate(row_labels)}
        row_index = plate["row"].map(row_lookup).to_numpy(dtype=int)
        column_values = sorted(plate["column"].unique())
        column_lookup = {
            column: index for index, column in enumerate(column_values)
        }
        column_index = plate["column"].map(column_lookup).to_numpy(dtype=int)
        rank = _traversal_rank(
            row_index,
            column_index,
            len(row_labels),
            len(column_values),
            parameters.traversal,
        )
        if parameters.dispense_group_size > len(plate):
            raise ValueError("dispense group size exceeds the plate well count")
        if max(parameters.affected_channels) >= parameters.dispense_group_size:
            raise ValueError("affected channel must be within the dispense group")
        group_count = int(np.ceil(len(plate) / parameters.dispense_group_size))
        group_stop = (
            parameters.affected_group_start + parameters.affected_group_count
        )
        if group_stop > group_count:
            raise ValueError("affected dispense groups exceed the traversal")

        group_index = rank // parameters.dispense_group_size
        channel_index = rank % parameters.dispense_group_size
        affected = (
            (group_index >= parameters.affected_group_start)
            & (group_index < group_stop)
            & np.isin(channel_index, parameters.affected_channels)
        )
        effect[plate.index] = np.where(
            affected,
            parameters.signed_magnitude,
            0.0,
        )
    return _apply_numeric_effect(
        clean,
        effect,
        mechanism=FailureMode.TRANSIENT_TIP_CLOG,
        parameters=parameters,
    )


def _semantic_sort_columns(order_by: str) -> list[str]:
    if order_by == "treatment_then_dose":
        return ["_role_order", "treatment", "dose", "replicate"]
    return ["_role_order", "dose", "treatment", "replicate"]


def inject_layout_confounding(
    clean: CleanAssayResult,
    parameters: LayoutConfoundingParameters,
) -> InjectedAssayResult:
    """Reassign conditions and their signals into a confounded spatial order."""
    _validate_clean_result(clean)
    rng = np.random.default_rng(_injection_seed(clean))
    plate_maps: list[pd.DataFrame] = []
    measurements: list[pd.DataFrame] = []
    latent_signals: list[pd.DataFrame] = []
    assignment_columns = [
        "sample_id",
        "well_role",
        "treatment",
        "dose",
        "dose_unit",
        "replicate",
    ]
    role_order = {
        WellRole.NEGATIVE_CONTROL.value: 0,
        WellRole.POSITIVE_CONTROL.value: 1,
        WellRole.TREATMENT.value: 2,
        WellRole.EMPTY.value: 3,
    }

    for plate_id, destination_map in clean.plate_map.groupby(
        "plate_id", sort=False
    ):
        destination_map = destination_map.copy(deep=True).reset_index(drop=True)
        source = clean.plate_map[clean.plate_map["plate_id"] == plate_id].copy()
        source["_role_order"] = source["well_role"].map(role_order)
        source["_tie_breaker"] = rng.random(len(source))
        sort_columns = [*_semantic_sort_columns(parameters.order_by), "_tie_breaker"]
        source = source.sort_values(sort_columns, kind="stable").reset_index()

        destination_map.loc[:, assignment_columns] = source[
            assignment_columns
        ].to_numpy()
        plate_maps.append(destination_map)

        source_latent = clean.latent_signals.loc[source["index"]].reset_index(
            drop=True
        )
        destination_latent = clean.latent_signals[
            clean.latent_signals["plate_id"] == plate_id
        ].reset_index(drop=True)
        moved_latent = destination_map[["plate_id", "well"]].copy()
        moved_latent["expected_signal"] = source_latent[
            "expected_signal"
        ].to_numpy()
        moved_latent["baseline_noise"] = destination_latent[
            "baseline_noise"
        ].to_numpy()
        moved_latent["injected_effect"] = 0.0
        latent_signals.append(moved_latent)

        moved_measurements = destination_map[["plate_id", "well"]].copy()
        moved_measurements.insert(0, "case_id", clean.metadata.case_id)
        moved_measurements["raw_signal"] = (
            moved_latent["expected_signal"] + moved_latent["baseline_noise"]
        )
        measurements.append(moved_measurements.loc[:, MEASUREMENT_COLUMNS])

    plate_map = pd.concat(plate_maps, ignore_index=True)
    measurement_frame = pd.concat(measurements, ignore_index=True)
    latent_frame = pd.concat(latent_signals, ignore_index=True)
    changed = (
        plate_map["sample_id"].to_numpy()
        != clean.plate_map["sample_id"].to_numpy()
    )
    return InjectedAssayResult(
        plate_map=plate_map,
        measurements=measurement_frame.loc[:, MEASUREMENT_COLUMNS],
        latent_signals=latent_frame.loc[:, INJECTED_LATENT_SIGNAL_COLUMNS],
        generation_metadata=clean.metadata,
        injection=InjectionRecord(
            mechanism=FailureMode.LAYOUT_CONFOUNDING,
            child_seed=_injection_seed(clean),
            parameters=_json_parameters(parameters),
            affected_well_count=int(np.count_nonzero(changed)),
        ),
    )


def inject_weak_controls(
    clean: CleanAssayResult,
    parameters: WeakControlsParameters,
) -> InjectedAssayResult:
    """Collapse positive controls toward the negative-control expectation."""
    roles = clean.plate_map["well_role"]
    latent = clean.latent_signals["expected_signal"]
    negative_mean = float(
        latent[roles == WellRole.NEGATIVE_CONTROL.value].mean()
    )
    positive_mask = roles == WellRole.POSITIVE_CONTROL.value
    effect = np.zeros(len(clean.plate_map), dtype=float)
    effect[positive_mask] = parameters.collapse_fraction * (
        negative_mean - latent[positive_mask].to_numpy()
    )
    return _apply_numeric_effect(
        clean,
        effect,
        mechanism=FailureMode.WEAK_CONTROLS,
        parameters=parameters,
    )


def inject_batch_shift(
    clean: CleanAssayResult,
    parameters: BatchShiftParameters,
) -> InjectedAssayResult:
    """Scale one plate's response around its observed negative-control mean."""
    plate_ids = set(clean.plate_map["plate_id"])
    if len(plate_ids) < 2:
        raise ValueError("batch-shift injection requires at least two plates")
    if parameters.shifted_plate_id not in plate_ids:
        raise ValueError("shifted plate is not present in the clean assay")
    mask = clean.plate_map["plate_id"] == parameters.shifted_plate_id
    negative_controls = mask & (
        clean.plate_map["well_role"] == WellRole.NEGATIVE_CONTROL.value
    )
    if not negative_controls.any():
        raise ValueError("shifted plate requires negative controls")
    anchor = float(clean.measurements.loc[negative_controls, "raw_signal"].mean())
    original = clean.measurements["raw_signal"].to_numpy(dtype=float)
    transformed = anchor + parameters.response_scale_factor * (original - anchor)
    effect = np.where(mask, transformed - original, 0.0)
    return _apply_numeric_effect(
        clean,
        effect,
        mechanism=FailureMode.BATCH_SHIFT,
        parameters=parameters,
    )


def inject_true_non_response(
    clean: CleanAssayResult,
    parameters: TrueNonResponseParameters,
) -> InjectedAssayResult:
    """Flatten one treatment while preserving its existing baseline noise."""
    treatment_mask = (
        (clean.plate_map["well_role"] == WellRole.TREATMENT.value)
        & (clean.plate_map["treatment"] == parameters.target_treatment)
    )
    if not treatment_mask.any():
        raise ValueError("target treatment is not present in the clean assay")
    effect = np.zeros(len(clean.plate_map), dtype=float)
    effect[treatment_mask] = (
        parameters.flat_expected_signal
        - clean.latent_signals.loc[treatment_mask, "expected_signal"].to_numpy()
    )
    return _apply_numeric_effect(
        clean,
        effect,
        mechanism=FailureMode.TRUE_NON_RESPONSE,
        parameters=parameters,
    )
