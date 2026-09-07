"""Deterministic, investigator-facing assay analysis contracts and tools."""

from experiment_failure_investigator.analysis.contracts import (
    ConditionCoverage,
    DesignCapabilities,
    DesignSummary,
    InvestigatorCase,
    InvestigatorMeasurement,
    InvestigatorMetadata,
    InvestigatorPlateMapWell,
    PlateDesignSummary,
    PublicArtifactHashes,
    TreatmentDoseSeries,
)
from experiment_failure_investigator.analysis.design import build_investigator_case

__all__ = [
    "ConditionCoverage",
    "DesignCapabilities",
    "DesignSummary",
    "InvestigatorCase",
    "InvestigatorMeasurement",
    "InvestigatorMetadata",
    "InvestigatorPlateMapWell",
    "PlateDesignSummary",
    "PublicArtifactHashes",
    "TreatmentDoseSeries",
    "build_investigator_case",
]
