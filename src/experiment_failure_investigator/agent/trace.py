"""Machine-readable investigation trace and runtime telemetry contracts."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue, StringConstraints, field_validator, model_validator

from experiment_failure_investigator.agent.config import AgentRuntimeConfig, RuntimeMode
from experiment_failure_investigator.agent.contracts import (
    AgentStrictModel,
    InvestigatorOutput,
    RunId,
)
from experiment_failure_investigator.analysis.contracts import PublicArtifactHashes
from experiment_failure_investigator.analysis.results import canonical_json

TRACE_VERSION = "1.0.0"
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def hash_prompt(prompt: str) -> str:
    """Hash the exact UTF-8 prompt artifact after rejecting blank content."""
    if not prompt.strip():
        raise ValueError("prompt must not be blank")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def hash_json_payload(payload: Any) -> str:
    """Hash a canonical JSON payload used in a trace event."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def new_run_id() -> str:
    """Create an opaque execution identifier unrelated to benchmark truth."""
    return "run_" + uuid.uuid4().hex[:20]


class RunStatus(StrEnum):
    """Terminal status of one bounded investigator execution."""

    SUCCESS = "success"
    FAILED = "failed"


class RunFailureCode(StrEnum):
    """Failure families exposed by the future single-run controller."""

    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MODEL_MISSING = "model_missing"
    VALIDATION_FAILED = "validation_failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"


class ModelEventType(StrEnum):
    """Model-side transitions captured without provider-specific classes."""

    REQUEST = "request"
    RESPONSE = "response"
    VALIDATION_RETRY = "validation_retry"


class ModelEvent(AgentStrictModel):
    """One sanitized JSON model event with an integrity digest."""

    sequence: int = Field(ge=1)
    event_type: ModelEventType
    payload: JsonValue
    payload_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_payload_hash(self) -> ModelEvent:
        if self.payload_sha256 != hash_json_payload(self.payload):
            raise ValueError("model-event payload hash does not match its payload")
        return self


class ToolEvent(AgentStrictModel):
    """One bounded agent tool call and validated JSON return."""

    sequence: int = Field(ge=1)
    tool_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    arguments: JsonValue
    arguments_sha256: Sha256Digest
    result: JsonValue
    result_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_payload_hashes(self) -> ToolEvent:
        if self.arguments_sha256 != hash_json_payload(self.arguments):
            raise ValueError("tool-event argument hash does not match its payload")
        if self.result_sha256 != hash_json_payload(self.result):
            raise ValueError("tool-event result hash does not match its payload")
        return self


class VersionRecord(AgentStrictModel):
    """Named semantic version included in runtime provenance."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")


class AgentUsage(AgentStrictModel):
    """Provider-reported usage, allowing absent token counts."""

    requests: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    monetary_provider_cost_usd: float = Field(default=0.0, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_reported_token_total(self) -> AgentUsage:
        if (
            self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens is not None
            and self.total_tokens != self.input_tokens + self.output_tokens
        ):
            raise ValueError("reported total tokens must equal input plus output")
        if self.monetary_provider_cost_usd != 0:
            raise ValueError("Week 3 local Ollama runs must record provider cost as zero")
        return self


class RuntimeSnapshot(AgentStrictModel):
    """Model, prompt, and budget settings captured for one run."""

    runtime_mode: RuntimeMode
    provider: Literal["ollama"]
    model_tag: str = Field(min_length=1)
    ollama_base_url: str = Field(min_length=1)
    prompt_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    prompt_sha256: Sha256Digest
    temperature: float = Field(ge=0, le=2, allow_inf_nan=False)
    max_output_tokens: int = Field(ge=1)
    request_limit: int = Field(ge=1)
    tool_calls_limit: int = Field(ge=0)
    input_tokens_limit: int = Field(ge=1)
    output_tokens_limit: int = Field(ge=1)
    total_tokens_limit: int = Field(ge=1)
    output_validation_retries: int = Field(ge=0)
    request_timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    run_timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    max_evidence_records: int = Field(ge=1)

    @classmethod
    def from_config(
        cls,
        config: AgentRuntimeConfig,
        *,
        prompt: str,
    ) -> RuntimeSnapshot:
        values = config.model_dump(mode="python")
        values["prompt_sha256"] = hash_prompt(prompt)
        return cls.model_validate(values)


class RunFailure(AgentStrictModel):
    """Sanitized typed failure retained instead of unvalidated prose."""

    code: RunFailureCode
    detail: str = Field(min_length=1)
    validation_attempts: int = Field(default=0, ge=0)


class InvestigationTrace(AgentStrictModel):
    """Complete validated trace for one bounded Week 3 execution."""

    trace_version: str = Field(
        default=TRACE_VERSION,
        pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$",
    )
    run_id: RunId
    case_id: str = Field(pattern=r"^case_[0-9a-f]{16}$")
    public_artifact_hashes: PublicArtifactHashes
    application_commit: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    pydantic_ai_version: str = Field(min_length=1)
    runtime: RuntimeSnapshot
    baseline_report_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    heuristic_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    tool_versions: tuple[VersionRecord, ...]
    model_events: tuple[ModelEvent, ...] = ()
    tool_events: tuple[ToolEvent, ...] = ()
    status: RunStatus
    output: InvestigatorOutput | None = None
    failure: RunFailure | None = None
    usage: AgentUsage
    started_at: datetime
    duration_seconds: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("started_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("trace start time must include a timezone")
        return value

    @field_validator("tool_versions")
    @classmethod
    def validate_tool_versions(
        cls, values: tuple[VersionRecord, ...]
    ) -> tuple[VersionRecord, ...]:
        names = [value.name for value in values]
        if len(names) != len(set(names)):
            raise ValueError("tool-version names must be unique")
        return tuple(sorted(values, key=lambda value: value.name))

    @model_validator(mode="after")
    def validate_terminal_state(self) -> InvestigationTrace:
        if self.status is RunStatus.SUCCESS:
            if self.output is None or self.failure is not None:
                raise ValueError("successful traces require output and forbid failure")
            if self.output.case_id != self.case_id:
                raise ValueError("trace and investigator output case IDs must match")
        elif self.output is not None or self.failure is None:
            raise ValueError("failed traces require failure and forbid output")

        sequences = [event.sequence for event in self.model_events] + [
            event.sequence for event in self.tool_events
        ]
        if sequences and sorted(sequences) != list(range(1, len(sequences) + 1)):
            raise ValueError("trace event sequences must be unique and contiguous")
        if self.usage.requests > self.runtime.request_limit:
            raise ValueError("reported requests exceed the configured limit")
        if self.usage.tool_calls > self.runtime.tool_calls_limit:
            raise ValueError("reported tool calls exceed the configured limit")
        token_limits = (
            (self.usage.input_tokens, self.runtime.input_tokens_limit, "input"),
            (self.usage.output_tokens, self.runtime.output_tokens_limit, "output"),
            (self.usage.total_tokens, self.runtime.total_tokens_limit, "total"),
        )
        for observed, limit, name in token_limits:
            if observed is not None and observed > limit:
                raise ValueError(f"reported {name} tokens exceed the configured limit")
        return self
