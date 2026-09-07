"""Stable result, evidence, provenance, and runtime telemetry contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from experiment_failure_investigator.analysis.contracts import PublicArtifactHashes
from experiment_failure_investigator.benchmark.models import Sha256, StrictModel, WellRole

EvidenceId = Annotated[str, StringConstraints(pattern=r"^ev_[0-9a-f]{20}$")]


class ToolStatus(StrEnum):
    """Scientific outcome of one deterministic tool invocation."""

    SUCCESS = "success"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_DATA = "insufficient_data"
    ERROR = "error"


def _canonical_json_value(value: Any, path: str = "value") -> JsonValue:
    """Reject non-JSON and non-finite values recursively."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, list | tuple):
        return [
            _canonical_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{path} object keys must be non-blank strings")
            normalized[key] = _canonical_json_value(item, f"{path}.{key}")
        return normalized
    raise ValueError(f"{path} contains a value that is not JSON-compatible")


def canonical_json(value: BaseModel | JsonValue) -> str:
    """Serialize a scientific contract as canonical, newline-terminated JSON."""
    payload: Any
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json")
    else:
        payload = value
    normalized = _canonical_json_value(payload)
    return (
        json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


class EvidenceScope(StrictModel):
    """Canonical public-data scope for one scientific measurement."""

    case_id: str = Field(pattern=r"^case_[0-9a-f]{16}$")
    plate_ids: tuple[str, ...] = ()
    well_roles: tuple[WellRole, ...] = ()
    treatments: tuple[str, ...] = ()
    doses: tuple[float, ...] = ()
    dose_unit: str | None = None
    well_ids: tuple[str, ...] = ()
    dimensions: Mapping[str, JsonValue] = Field(default_factory=dict)

    @field_validator("plate_ids", "treatments", "well_ids")
    @classmethod
    def sort_unique_strings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("scope strings must not be blank")
        if len(values) != len(set(values)):
            raise ValueError("scope values must be unique")
        return tuple(sorted(values))

    @field_validator("well_roles")
    @classmethod
    def sort_unique_roles(cls, values: tuple[WellRole, ...]) -> tuple[WellRole, ...]:
        if len(values) != len(set(values)):
            raise ValueError("scope well roles must be unique")
        return tuple(sorted(values, key=lambda role: role.value))

    @field_validator("doses")
    @classmethod
    def sort_unique_finite_doses(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if any(not isfinite(value) for value in values):
            raise ValueError("scope doses must be finite")
        if len(values) != len(set(values)):
            raise ValueError("scope doses must be unique")
        return tuple(sorted(values))

    @field_validator("dose_unit")
    @classmethod
    def validate_optional_unit(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("dose unit must not be blank")
        return value

    @field_validator("dimensions", mode="before")
    @classmethod
    def validate_dimensions(cls, value: Any) -> JsonValue:
        return _canonical_json_value(value, "dimensions")


class EvidenceRecord(StrictModel):
    """One traceable numeric or structured scientific observation."""

    evidence_id: EvidenceId
    metric_name: str = Field(pattern=r"^[a-z][a-z0-9_.]*$")
    value: JsonValue
    unit: str | None = None
    scope: EvidenceScope
    sample_count: int | None = Field(default=None, ge=0)
    description: str = Field(min_length=1)

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, value: Any) -> JsonValue:
        return _canonical_json_value(value)

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("unit must not be blank")
        return value

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must not be blank")
        return value


class ResultWarning(StrictModel):
    """Structured caution associated with a scientific result."""

    code: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    message: str = Field(min_length=1)
    evidence_ids: tuple[EvidenceId, ...] = ()

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("warning message must not be blank")
        return value

    @field_validator("evidence_ids")
    @classmethod
    def canonicalize_evidence_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("warning evidence IDs must be unique")
        return tuple(sorted(values))


class ScientificProvenance(StrictModel):
    """Stable public-input provenance for a deterministic result."""

    case_id: str = Field(pattern=r"^case_[0-9a-f]{16}$")
    public_artifact_hashes: PublicArtifactHashes


class ArtifactAttachment(StrictModel):
    """A non-numeric file artifact produced by a scientific tool."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    path: str = Field(min_length=1)
    sha256: Sha256
    media_type: str = Field(pattern=r"^[a-z0-9.+-]+/[a-z0-9.+-]+$")
    description: str = Field(min_length=1)

    @field_validator("path", "description")
    @classmethod
    def reject_blank_attachment_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("attachment text must not be blank")
        return value


def build_evidence_id(
    *,
    tool_name: str,
    tool_version: str,
    parameters: Mapping[str, JsonValue],
    metric_name: str,
    scope: EvidenceScope,
) -> str:
    """Create a stable evidence identity independent of execution order."""
    payload: JsonValue = {
        "metric_name": metric_name,
        "parameters": dict(parameters),
        "scope": scope.model_dump(mode="json"),
        "tool_name": tool_name,
        "tool_version": tool_version,
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return "ev_" + digest[:20]


def build_evidence_record(
    *,
    tool_name: str,
    tool_version: str,
    parameters: Mapping[str, JsonValue],
    metric_name: str,
    value: JsonValue,
    scope: EvidenceScope,
    description: str,
    unit: str | None = None,
    sample_count: int | None = None,
) -> EvidenceRecord:
    """Build an evidence record with its canonical stable identifier."""
    return EvidenceRecord(
        evidence_id=build_evidence_id(
            tool_name=tool_name,
            tool_version=tool_version,
            parameters=parameters,
            metric_name=metric_name,
            scope=scope,
        ),
        metric_name=metric_name,
        value=value,
        unit=unit,
        scope=scope,
        sample_count=sample_count,
        description=description,
    )


class ScientificToolResult(StrictModel):
    """Deterministic scientific output, excluding execution telemetry."""

    status: ToolStatus
    status_reason: str | None = None
    tool_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    tool_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    parameters: Mapping[str, JsonValue] = Field(default_factory=dict)
    scope: EvidenceScope
    evidence: tuple[EvidenceRecord, ...] = ()
    attachments: tuple[ArtifactAttachment, ...] = ()
    warnings: tuple[ResultWarning, ...] = ()
    limitations: tuple[str, ...] = ()
    provenance: ScientificProvenance

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, value: Any) -> JsonValue:
        return _canonical_json_value(value, "parameters")

    @field_validator("status_reason")
    @classmethod
    def validate_optional_reason(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("status reason must not be blank")
        return value

    @field_validator("evidence")
    @classmethod
    def canonicalize_evidence(
        cls,
        values: tuple[EvidenceRecord, ...],
    ) -> tuple[EvidenceRecord, ...]:
        ids = [evidence.evidence_id for evidence in values]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence IDs must be unique within a tool result")
        return tuple(sorted(values, key=lambda evidence: evidence.evidence_id))

    @field_validator("attachments")
    @classmethod
    def canonicalize_attachments(
        cls,
        values: tuple[ArtifactAttachment, ...],
    ) -> tuple[ArtifactAttachment, ...]:
        names = [attachment.name for attachment in values]
        paths = [attachment.path for attachment in values]
        if len(names) != len(set(names)) or len(paths) != len(set(paths)):
            raise ValueError("attachment names and paths must be unique")
        return tuple(sorted(values, key=lambda attachment: attachment.name))

    @field_validator("warnings")
    @classmethod
    def canonicalize_warnings(
        cls,
        values: tuple[ResultWarning, ...],
    ) -> tuple[ResultWarning, ...]:
        return tuple(
            sorted(values, key=lambda warning: (warning.code, warning.message))
        )

    @field_validator("limitations")
    @classmethod
    def canonicalize_limitations(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not limitation.strip() for limitation in values):
            raise ValueError("limitations must not be blank")
        if len(values) != len(set(values)):
            raise ValueError("limitations must be unique")
        return tuple(sorted(values))

    @model_validator(mode="after")
    def validate_semantics_and_identity(self) -> ScientificToolResult:
        if self.status is ToolStatus.SUCCESS and self.status_reason is not None:
            raise ValueError("successful results must not include a status reason")
        if self.status is not ToolStatus.SUCCESS and self.status_reason is None:
            raise ValueError("non-success results require a status reason")
        if (
            self.status in {ToolStatus.NOT_APPLICABLE, ToolStatus.ERROR}
            and (self.evidence or self.attachments)
        ):
            raise ValueError(
                f"{self.status.value} results must not contain evidence or attachments"
            )
        if self.scope.case_id != self.provenance.case_id:
            raise ValueError("result scope and provenance case IDs must match")

        evidence_ids = {evidence.evidence_id for evidence in self.evidence}
        for evidence in self.evidence:
            if evidence.scope.case_id != self.scope.case_id:
                raise ValueError("evidence and result scope case IDs must match")
            expected_id = build_evidence_id(
                tool_name=self.tool_name,
                tool_version=self.tool_version,
                parameters=self.parameters,
                metric_name=evidence.metric_name,
                scope=evidence.scope,
            )
            if evidence.evidence_id != expected_id:
                raise ValueError(
                    f"evidence ID does not match its canonical identity: "
                    f"{evidence.evidence_id}"
                )
        for warning in self.warnings:
            unknown_ids = set(warning.evidence_ids) - evidence_ids
            if unknown_ids:
                raise ValueError("warning references evidence outside this result")
        return self


class RuntimeTelemetry(StrictModel):
    """Non-scientific execution metadata kept outside deterministic results."""

    execution_id: str = Field(min_length=1)
    started_at: datetime
    duration_ms: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("started_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("started_at must include a timezone")
        return value


class ToolExecution(StrictModel):
    """A scientific result paired with explicitly separate runtime telemetry."""

    result: ScientificToolResult
    telemetry: RuntimeTelemetry
