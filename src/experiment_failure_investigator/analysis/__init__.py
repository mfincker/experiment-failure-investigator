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
from experiment_failure_investigator.analysis.design import build_investigator_case
from experiment_failure_investigator.analysis.controls import summarize_controls
from experiment_failure_investigator.analysis.missingness import inspect_missingness
from experiment_failure_investigator.analysis.results import (
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

__all__ = [
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
    "RuntimeTelemetry",
    "ScientificProvenance",
    "ScientificToolResult",
    "TreatmentDoseSeries",
    "ToolExecution",
    "ToolStatus",
    "build_evidence_id",
    "build_evidence_record",
    "build_investigator_case",
    "canonical_json",
    "inspect_missingness",
    "summarize_controls",
]
