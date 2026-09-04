"""Contracts and generators for the synthetic assay benchmark."""

from experiment_failure_investigator.benchmark.layouts import (
    LayoutAudit,
    audit_layout,
    build_balanced_layout,
    enumerate_wells,
)
from experiment_failure_investigator.benchmark.models import (
    ArtifactReference,
    AssayType,
    CaseFiles,
    CaseManifest,
    CaseVariant,
    FailureMode,
    GeneratorConfig,
    GroundTruth,
    PlateFormat,
    SignalDirection,
    SimulatedTraversal,
    TreatmentCurve,
    WellCoordinate,
    WellRole,
)
from experiment_failure_investigator.benchmark.randomness import derive_child_seed
from experiment_failure_investigator.benchmark.signals import (
    CleanAssayResult,
    CleanGenerationMetadata,
    four_parameter_logistic,
    generate_clean_assay,
)

__all__ = [
    "ArtifactReference",
    "AssayType",
    "CaseFiles",
    "CaseManifest",
    "CaseVariant",
    "CleanAssayResult",
    "CleanGenerationMetadata",
    "FailureMode",
    "GeneratorConfig",
    "GroundTruth",
    "LayoutAudit",
    "PlateFormat",
    "SignalDirection",
    "SimulatedTraversal",
    "TreatmentCurve",
    "WellCoordinate",
    "WellRole",
    "audit_layout",
    "build_balanced_layout",
    "derive_child_seed",
    "enumerate_wells",
    "four_parameter_logistic",
    "generate_clean_assay",
]
