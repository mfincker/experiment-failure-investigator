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

__all__ = [
    "ArtifactReference",
    "AssayType",
    "CaseFiles",
    "CaseManifest",
    "CaseVariant",
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
    "enumerate_wells",
]
