"""Pure clean-signal generation for synthetic plate assays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiment_failure_investigator.benchmark.layouts import (
    PLATE_MAP_COLUMNS,
    enumerate_wells,
)
from experiment_failure_investigator.benchmark.models import (
    GeneratorConfig,
    TreatmentCurve,
    WellRole,
)
from experiment_failure_investigator.benchmark.randomness import derive_child_seed

MEASUREMENT_COLUMNS = ["case_id", "plate_id", "well", "raw_signal"]
LATENT_SIGNAL_COLUMNS = [
    "plate_id",
    "well",
    "expected_signal",
    "baseline_noise",
]
BASELINE_NOISE_NAMESPACE = "baseline_noise"


@dataclass(frozen=True)
class CleanGenerationMetadata:
    """Private reproducibility metadata for one clean generation step."""

    case_id: str
    baseline_noise_seed: int
    noise_sd: float
    plate_count: int
    well_count: int


@dataclass(frozen=True)
class CleanAssayResult:
    """Clean public measurements plus private latent generation values."""

    plate_map: pd.DataFrame
    measurements: pd.DataFrame
    latent_signals: pd.DataFrame
    metadata: CleanGenerationMetadata


def four_parameter_logistic(
    dose: float | np.ndarray,
    curve: TreatmentCurve,
) -> float | np.ndarray:
    """Evaluate the benchmark's decreasing four-parameter logistic curve."""
    values = np.asarray(dose, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("treatment doses must be finite and positive")
    response = curve.bottom + (curve.top - curve.bottom) / (
        1.0 + np.power(values / curve.ic50, curve.hill_slope)
    )
    if values.ndim == 0:
        return float(response)
    return response


def _validate_plate_map(layout: pd.DataFrame, config: GeneratorConfig) -> None:
    missing_columns = sorted(set(PLATE_MAP_COLUMNS) - set(layout.columns))
    if missing_columns:
        raise ValueError(
            "plate map is missing required columns: " + ", ".join(missing_columns)
        )
    if layout[["plate_id", "well"]].isna().any().any():
        raise ValueError("plate and well identifiers must not be null")
    if layout.duplicated(["plate_id", "well"]).any():
        raise ValueError("plate map contains duplicate plate/well keys")
    if layout["plate_id"].nunique() != config.plate_count:
        raise ValueError("plate-map plate count disagrees with generator configuration")
    expected_rows = config.plate_count * config.plate_format.capacity
    if len(layout) != expected_rows:
        raise ValueError(
            f"plate map must contain {expected_rows} rows, observed {len(layout)}"
        )
    per_plate_counts = layout.groupby("plate_id", sort=False)["well"].size()
    if not (per_plate_counts == config.plate_format.capacity).all():
        raise ValueError("every plate must contain the declared plate capacity")
    expected_wells = set(enumerate_wells(config.plate_format)["well"])
    for plate_id, plate in layout.groupby("plate_id", sort=False):
        if set(plate["well"]) != expected_wells:
            raise ValueError(
                f"plate {plate_id!r} does not match the declared plate geometry"
            )


def _expected_signal(row: pd.Series, config: GeneratorConfig) -> float:
    role = row["well_role"]
    if role == WellRole.NEGATIVE_CONTROL.value:
        return config.negative_control_mean
    if role == WellRole.POSITIVE_CONTROL.value:
        return config.positive_control_mean
    if role == WellRole.EMPTY.value:
        raise ValueError(
            "empty wells have no modeled biological signal; "
            "an explicit instrument readout policy is required"
        )
    if role != WellRole.TREATMENT.value:
        raise ValueError(f"unsupported well role: {role!r}")

    treatment = row["treatment"]
    if treatment not in config.curves:
        raise ValueError(f"no configured curve for treatment: {treatment!r}")
    try:
        dose = float(row["dose"])
    except (TypeError, ValueError) as error:
        raise ValueError("treatment wells require a numeric dose") from error
    return float(four_parameter_logistic(dose, config.curves[treatment]))


def generate_clean_assay(
    layout: pd.DataFrame,
    config: GeneratorConfig,
) -> CleanAssayResult:
    """Generate reproducible clean signals without file I/O or mutation."""
    _validate_plate_map(layout, config)
    canonical_layout = (
        layout.loc[:, PLATE_MAP_COLUMNS]
        .copy(deep=True)
        .sort_values(["plate_id", "well"], kind="stable")
        .reset_index(drop=True)
    )
    expected = canonical_layout.apply(
        _expected_signal,
        axis=1,
        config=config,
    ).to_numpy(dtype=float)
    if not np.isfinite(expected).all():
        raise ValueError("expected signals must all be finite")

    noise_seed = derive_child_seed(config.root_seed, BASELINE_NOISE_NAMESPACE)
    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(loc=0.0, scale=config.noise_sd, size=len(canonical_layout))
    observed = expected + noise
    if not np.isfinite(observed).all():
        raise ValueError("observed signals must all be finite")

    measurements = canonical_layout[["plate_id", "well"]].copy()
    measurements.insert(0, "case_id", config.case_id)
    measurements["raw_signal"] = observed
    measurements = measurements.loc[:, MEASUREMENT_COLUMNS]

    latent_signals = canonical_layout[["plate_id", "well"]].copy()
    latent_signals["expected_signal"] = expected
    latent_signals["baseline_noise"] = noise
    latent_signals = latent_signals.loc[:, LATENT_SIGNAL_COLUMNS]

    return CleanAssayResult(
        plate_map=canonical_layout,
        measurements=measurements,
        latent_signals=latent_signals,
        metadata=CleanGenerationMetadata(
            case_id=config.case_id,
            baseline_noise_seed=noise_seed,
            noise_sd=config.noise_sd,
            plate_count=config.plate_count,
            well_count=len(canonical_layout),
        ),
    )
