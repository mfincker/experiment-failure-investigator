"""Strict public-input and design-summary contracts for investigators."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from experiment_failure_investigator.benchmark.models import (
    AssayType,
    PlateFormat,
    Sha256,
    SignalDirection,
    StrictModel,
    WellRole,
)


class PublicArtifactHashes(StrictModel):
    """Verified hashes for the five investigator-visible case artifacts."""

    measurements: Sha256
    plate_map: Sha256
    metadata: Sha256
    protocol: Sha256
    problem_statement: Sha256


class InvestigatorPlateMetadata(StrictModel):
    """Public plate provenance without benchmark-specific case metadata."""

    plate_id: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    operator_label: str | None = None
    run_date: date | None = None
    instrument_label: str | None = None


class InvestigatorMetadata(StrictModel):
    """Metadata safe to expose to deterministic tools and future agents."""

    assay_type: AssayType
    signal_direction: SignalDirection
    plates: tuple[InvestigatorPlateMetadata, ...] = Field(min_length=1)


class InvestigatorMeasurement(StrictModel):
    """One public assay measurement with benchmark labels removed."""

    plate_id: str = Field(min_length=1)
    well: str = Field(pattern=r"^[A-Z][0-9]{2}$")
    raw_signal: float | None = Field(default=None, allow_inf_nan=False)


class InvestigatorPlateMapWell(StrictModel):
    """One investigator-visible well assignment."""

    plate_id: str = Field(min_length=1)
    well: str = Field(pattern=r"^[A-Z][0-9]{2}$")
    row: str = Field(pattern=r"^[A-Z]$")
    column: int = Field(ge=1, le=99)
    sample_id: str = Field(min_length=1)
    well_role: WellRole
    treatment: str | None = None
    dose: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    dose_unit: str | None = None
    replicate: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_coordinate(self) -> InvestigatorPlateMapWell:
        if self.well != f"{self.row}{self.column:02d}":
            raise ValueError("well disagrees with row and column")
        return self


class PlateDesignSummary(StrictModel):
    """Geometry and completeness derived for one observed plate."""

    plate_id: str
    plate_format: PlateFormat
    geometry_source: Literal["inferred"] = "inferred"
    geometry_inference_warning: str | None = None
    observed_well_count: int = Field(ge=0)
    expected_well_count: int = Field(ge=1)
    missing_wells: tuple[str, ...] = ()
    missing_measurement_count: int = Field(ge=0)
    null_measurement_count: int = Field(ge=0)
    nonfinite_measurement_count: int = Field(ge=0)
    missing_design_annotation_count: int = Field(ge=0)
    role_counts: dict[WellRole, int]


class TreatmentDoseSeries(StrictModel):
    """Observed doses for one treatment and unit combination."""

    treatment: str
    dose_unit: str | None = None
    doses: tuple[float, ...]


class ConditionCoverage(StrictModel):
    """Observed replicate counts for one condition on every plate."""

    well_role: WellRole
    treatment: str | None = None
    dose: float | None = None
    dose_unit: str | None = None
    counts_by_plate: dict[str, int]


class DesignCapabilities(StrictModel):
    """Facts about analyses supported by the observed design."""

    has_negative_controls: bool
    has_positive_controls: bool
    has_replicates: bool
    has_dose_series: bool
    has_multiple_plates: bool
    plates_are_complete_design_replicates: bool | None
    supports_cross_plate_analysis: bool


class DesignSummary(StrictModel):
    """Canonical summary derived only from investigator-visible inputs."""

    plate_count: int = Field(ge=1)
    plates: tuple[PlateDesignSummary, ...]
    available_well_roles: tuple[WellRole, ...]
    treatments: tuple[str, ...]
    treatment_dose_series: tuple[TreatmentDoseSeries, ...]
    condition_coverage: tuple[ConditionCoverage, ...]
    replicate_count_distribution: dict[int, int]
    total_missing_well_count: int = Field(ge=0)
    total_missing_measurement_count: int = Field(ge=0)
    total_null_measurement_count: int = Field(ge=0)
    total_nonfinite_measurement_count: int = Field(ge=0)
    capabilities: DesignCapabilities
    limitations: tuple[str, ...] = ()


class InvestigatorCase(StrictModel):
    """Ground-truth-free input accepted by analysis tools and agents."""

    case_id: str = Field(pattern=r"^case_[0-9a-f]{16}$")
    measurements: tuple[InvestigatorMeasurement, ...]
    plate_map: tuple[InvestigatorPlateMapWell, ...]
    metadata: InvestigatorMetadata
    protocol: str = Field(min_length=1)
    problem_statement: str = Field(min_length=1)
    public_artifact_hashes: PublicArtifactHashes
    design: DesignSummary
