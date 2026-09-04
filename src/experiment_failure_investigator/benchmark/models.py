"""Typed contracts for synthetic assay benchmark cases."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = "1.0.0"
BENCHMARK_VERSION = "1.0.0"
MAX_SEED = 2**32 - 1

CaseId = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$", min_length=3),
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    """Base class for stable contracts that reject unknown fields."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AssayType(StrEnum):
    """Assay designs supported by the initial benchmark."""

    CELL_VIABILITY_ENDPOINT = "cell_viability_endpoint"


class PlateFormat(StrEnum):
    """Standard plate geometries supported by benchmark configurations."""

    WELLS_96 = "96_well"
    WELLS_384 = "384_well"

    @property
    def rows(self) -> int:
        return 8 if self is PlateFormat.WELLS_96 else 16

    @property
    def columns(self) -> int:
        return 12 if self is PlateFormat.WELLS_96 else 24

    @property
    def capacity(self) -> int:
        return self.rows * self.columns


class SignalDirection(StrEnum):
    """How to interpret movement in the assay signal."""

    LOWER_IS_STRONGER = "lower_is_stronger"


class WellRole(StrEnum):
    """Allowed roles for wells in an investigator-visible plate map."""

    NEGATIVE_CONTROL = "negative_control"
    POSITIVE_CONTROL = "positive_control"
    TREATMENT = "treatment"
    EMPTY = "empty"


class FailureMode(StrEnum):
    """Single planted mechanisms in the initial benchmark."""

    EDGE_EFFECT = "edge_effect"
    PIPETTING_DRIFT = "pipetting_drift"
    LAYOUT_CONFOUNDING = "layout_confounding"
    WEAK_CONTROLS = "weak_controls"
    BATCH_SHIFT = "batch_shift"
    TRUE_NON_RESPONSE = "true_non_response"


class CaseVariant(StrEnum):
    """Initial benchmark difficulty variants."""

    OBVIOUS = "obvious"
    NOISY = "noisy"


class SimulatedTraversal(StrEnum):
    """Private synthetic traversals used to inject pipetting drift."""

    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"
    SERPENTINE_ROWS = "serpentine_rows"
    SERPENTINE_COLUMNS = "serpentine_columns"


class WellCoordinate(StrictModel):
    """A syntactically valid well location, independent of plate geometry."""

    row: str
    column: int

    @field_validator("row")
    @classmethod
    def validate_row(cls, value: str) -> str:
        normalized = value.upper()
        if len(normalized) != 1 or not "A" <= normalized <= "Z":
            raise ValueError("row must be one letter from A through Z")
        return normalized

    @field_validator("column")
    @classmethod
    def validate_column(cls, value: int) -> int:
        if not 1 <= value <= 99:
            raise ValueError("column must be between 1 and 99")
        return value

    @property
    def well(self) -> str:
        """Return the canonical zero-padded well identifier."""
        return f"{self.row}{self.column:02d}"


class TreatmentCurve(StrictModel):
    """Parameters for a decreasing four-parameter logistic curve."""

    top: float = Field(default=1.0, allow_inf_nan=False)
    bottom: float = Field(default=0.15, allow_inf_nan=False)
    ic50: float = Field(gt=0, allow_inf_nan=False)
    hill_slope: float = Field(default=1.2, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_asymptotes(self) -> TreatmentCurve:
        if self.top <= self.bottom:
            raise ValueError("top must be greater than bottom")
        return self


def default_curves() -> dict[str, TreatmentCurve]:
    """Return independent defaults for the two clean responsive treatments."""
    return {
        "reference_treatment": TreatmentCurve(ic50=0.3),
        "test_treatment": TreatmentCurve(ic50=1.0),
    }


class GeneratorConfig(StrictModel):
    """Versioned configuration for a synthetic benchmark case.

    This is a generator contract, not the future investigator input contract.
    Investigator code must derive design properties from loaded case data.
    """

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    benchmark_version: Literal["1.0.0"] = BENCHMARK_VERSION
    case_id: CaseId
    case_variant: CaseVariant
    failure_mode: FailureMode
    root_seed: int = Field(ge=0, le=MAX_SEED)
    layout_seed: int = Field(default=20_260_904, ge=0, le=MAX_SEED)
    assay_type: AssayType = AssayType.CELL_VIABILITY_ENDPOINT
    signal_direction: SignalDirection = SignalDirection.LOWER_IS_STRONGER
    plate_format: PlateFormat = PlateFormat.WELLS_96
    plate_count: int = Field(default=1, ge=1)
    negative_control_count: int = Field(default=8, ge=1)
    positive_control_count: int = Field(default=8, ge=1)
    empty_count: int = Field(default=0, ge=0)
    treatments: list[str] = Field(
        default_factory=lambda: ["reference_treatment", "test_treatment"],
        min_length=1,
    )
    doses_micromolar: list[float] = Field(
        default_factory=lambda: [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        min_length=1,
    )
    replicates_per_condition: int = Field(default=5, ge=1)
    negative_control_mean: float = Field(default=1.0, allow_inf_nan=False)
    positive_control_mean: float = Field(default=0.15, allow_inf_nan=False)
    noise_sd: float = Field(ge=0, allow_inf_nan=False)
    curves: dict[str, TreatmentCurve] = Field(default_factory=default_curves)

    @field_validator("treatments")
    @classmethod
    def validate_treatments(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("treatments must be unique")
        if any(not treatment.strip() for treatment in value):
            raise ValueError("treatment names must not be blank")
        return value

    @field_validator("doses_micromolar")
    @classmethod
    def validate_doses(cls, value: list[float]) -> list[float]:
        if any(not isfinite(dose) or dose <= 0 for dose in value):
            raise ValueError("doses must be finite and positive")
        if len(set(value)) != len(value):
            raise ValueError("doses must be unique")
        if value != sorted(value):
            raise ValueError("doses must be in ascending order")
        return value

    @model_validator(mode="after")
    def validate_design(self) -> GeneratorConfig:
        treatment_wells = (
            len(self.treatments)
            * len(self.doses_micromolar)
            * self.replicates_per_condition
        )
        total_wells = (
            self.negative_control_count
            + self.positive_control_count
            + self.empty_count
            + treatment_wells
        )
        if total_wells != self.plate_format.capacity:
            raise ValueError(
                "plate design must fill exactly "
                f"{self.plate_format.capacity} wells for {self.plate_format.value}; "
                f"configured design fills {total_wells}"
            )
        if set(self.curves) != set(self.treatments):
            raise ValueError("curve names must match treatment names exactly")
        if self.negative_control_mean <= self.positive_control_mean:
            raise ValueError(
                "negative-control mean must exceed positive-control mean when "
                "lower signal represents stronger inhibition"
            )
        if self.failure_mode is FailureMode.BATCH_SHIFT and self.plate_count < 2:
            raise ValueError("batch-shift cases require at least two plates")
        return self


class ArtifactReference(StrictModel):
    """Tamper-evident reference to a serialized case artifact."""

    path: str
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("artifact paths must use forward slashes")
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts or value == ".":
            raise ValueError("artifact path must remain within the case directory")
        return value


class CaseFiles(StrictModel):
    """Required serialized artifacts, excluding the manifest itself."""

    measurements: ArtifactReference
    plate_map: ArtifactReference
    metadata: ArtifactReference
    protocol: ArtifactReference
    problem_statement: ArtifactReference


class GroundTruth(StrictModel):
    """Private answer-key fields that must never be passed to an investigator."""

    mechanism: FailureMode
    child_seeds: dict[str, int]
    injection_parameters: dict[str, JsonValue]
    simulated_traversal: SimulatedTraversal | None = None

    @field_validator("child_seeds")
    @classmethod
    def validate_child_seeds(cls, value: dict[str, int]) -> dict[str, int]:
        if not value:
            raise ValueError("at least one named child seed is required")
        for name, seed in value.items():
            if not name.strip():
                raise ValueError("child-seed names must not be blank")
            if not 0 <= seed <= MAX_SEED:
                raise ValueError(f"child seed {name!r} is outside uint32 range")
        return value

    @model_validator(mode="after")
    def validate_traversal(self) -> GroundTruth:
        if (
            self.simulated_traversal is not None
            and self.mechanism is not FailureMode.PIPETTING_DRIFT
        ):
            raise ValueError(
                "simulated traversal is only valid for pipetting-drift ground truth"
            )
        return self


class CaseManifest(StrictModel):
    """Evaluation-only manifest for a finalized synthetic case."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    benchmark_version: Literal["1.0.0"] = BENCHMARK_VERSION
    case_id: CaseId
    case_variant: CaseVariant
    root_seed: int = Field(ge=0, le=MAX_SEED)
    assay_type: AssayType
    signal_direction: SignalDirection
    planted_failure_mode: FailureMode
    generation: GeneratorConfig
    ground_truth: GroundTruth
    expected_discriminating_evidence: list[str] = Field(min_length=1)
    plausible_confounders: list[str] = Field(default_factory=list)
    files: CaseFiles

    @field_validator("expected_discriminating_evidence", "plausible_confounders")
    @classmethod
    def reject_blank_statements(cls, value: list[str]) -> list[str]:
        if any(not statement.strip() for statement in value):
            raise ValueError("evidence and confounder statements must not be blank")
        return value

    @model_validator(mode="after")
    def validate_internal_consistency(self) -> CaseManifest:
        matching_fields = {
            "case_id": (self.case_id, self.generation.case_id),
            "case_variant": (self.case_variant, self.generation.case_variant),
            "root_seed": (self.root_seed, self.generation.root_seed),
            "assay_type": (self.assay_type, self.generation.assay_type),
            "signal_direction": (
                self.signal_direction,
                self.generation.signal_direction,
            ),
            "planted_failure_mode": (
                self.planted_failure_mode,
                self.generation.failure_mode,
            ),
            "ground_truth.mechanism": (
                self.planted_failure_mode,
                self.ground_truth.mechanism,
            ),
        }
        mismatches = [
            name for name, (manifest_value, nested_value) in matching_fields.items()
            if manifest_value != nested_value
        ]
        if mismatches:
            raise ValueError(
                "manifest fields disagree with nested configuration: "
                + ", ".join(mismatches)
            )
        return self
