"""Deterministic, investigator-facing assay analysis contracts and tools."""

from experiment_failure_investigator.analysis.contracts import (
    ConditionCoverage,
    DesignCapabilities,
    DesignSummary,
    InvestigatorCase,
    InvestigatorMeasurement,
    InvestigatorMetadata,
    InvestigatorPlateMapWell,
    MeasurementStatus,
    PlateDesignSummary,
    PublicArtifactHashes,
    TreatmentDoseSeries,
)
from experiment_failure_investigator.analysis.batches import compare_batches
from experiment_failure_investigator.analysis.controls import summarize_controls
from experiment_failure_investigator.analysis.design import build_investigator_case
from experiment_failure_investigator.analysis.dose_response import fit_dose_response
from experiment_failure_investigator.analysis.heatmaps import generate_plate_heatmap
from experiment_failure_investigator.analysis.missingness import inspect_missingness
from experiment_failure_investigator.analysis.replicates import (
    calculate_replicate_variability,
)
from experiment_failure_investigator.analysis.results import (
    ArtifactAttachment,
    EvidenceId,
    EvidenceRecord,
    EvidenceScope,
    ResultWarning,
    RuntimeTelemetry,
    ScientificProvenance,
    ScientificToolResult,
    ToolExecution,
    ToolStatus,
    build_evidence_id,
    build_evidence_record,
    canonical_json,
)
from experiment_failure_investigator.analysis.spatial import (
    ResidualObservation,
    condition_centered_residuals,
    detect_spatial_effects,
)

__all__ = [
    "ArtifactAttachment",
    "ConditionCoverage",
    "DesignCapabilities",
    "DesignSummary",
    "EvidenceId",
    "EvidenceRecord",
    "EvidenceScope",
    "InvestigatorCase",
    "InvestigatorMeasurement",
    "InvestigatorMetadata",
    "InvestigatorPlateMapWell",
    "MeasurementStatus",
    "PlateDesignSummary",
    "PublicArtifactHashes",
    "ResultWarning",
    "ResidualObservation",
    "RuntimeTelemetry",
    "ScientificProvenance",
    "ScientificToolResult",
    "TreatmentDoseSeries",
    "ToolExecution",
    "ToolStatus",
    "build_evidence_id",
    "build_evidence_record",
    "build_investigator_case",
    "calculate_replicate_variability",
    "canonical_json",
    "compare_batches",
    "condition_centered_residuals",
    "detect_spatial_effects",
    "fit_dose_response",
    "generate_plate_heatmap",
    "inspect_missingness",
    "summarize_controls",
]
