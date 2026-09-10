"""Compact machine-readable contract for one Investigator execution."""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import Field, JsonValue, StringConstraints, field_validator, model_validator

from experiment_failure_investigator.agent.config import AgentRuntimeConfig
from experiment_failure_investigator.agent.contracts import (
    AgentStrictModel,
    InvestigatorOutput,
    RunId,
)
from experiment_failure_investigator.analysis.contracts import PublicArtifactHashes

TRACE_VERSION = "2.0.0"
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def hash_prompt(prompt: str) -> str:
    """Hash the exact UTF-8 prompt artifact after rejecting blank content."""
    if not prompt.strip():
        raise ValueError("prompt must not be blank")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def new_run_id() -> str:
    """Create an opaque execution identifier unrelated to benchmark truth."""
    return "run_" + uuid.uuid4().hex[:20]


class RunStatus(StrEnum):
    """Terminal status of one bounded investigator execution."""

    SUCCESS = "success"
    FAILED = "failed"


class RunFailureCode(StrEnum):
    """Stable failure families exposed by the single-run controller."""

    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MODEL_MISSING = "model_missing"
    VALIDATION_FAILED = "validation_failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"


class TraceEventKind(StrEnum):
    """Kinds of ordered framework activity retained in the trace."""

    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    VALIDATION_RETRY = "validation_retry"
    TOOL = "tool"


class TraceEvent(AgentStrictModel):
    """One ordered model message or extracted tool exchange."""

    sequence: int = Field(ge=1)
    kind: TraceEventKind
    payload: JsonValue
    tool_name: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]*$",
    )

    @model_validator(mode="after")
    def validate_tool_name(self) -> TraceEvent:
        if self.kind is TraceEventKind.TOOL and self.tool_name is None:
            raise ValueError("tool events require a tool name")
        if self.kind is not TraceEventKind.TOOL and self.tool_name is not None:
            raise ValueError("model events must not include a tool name")
        return self


class InvestigationTrace(AgentStrictModel):
    """Validated persisted record of one bounded Investigator run."""

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
    config: AgentRuntimeConfig
    prompt_sha256: Sha256Digest
    baseline_report_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    heuristic_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    tool_versions: dict[str, str]
    events: tuple[TraceEvent, ...] = ()
    status: RunStatus
    output: InvestigatorOutput | None = None
    failure_code: RunFailureCode | None = None
    failure_detail: str | None = Field(default=None, min_length=1, max_length=1000)
    validation_attempts: int = Field(default=0, ge=0)
    request_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    monetary_provider_cost_usd: float = Field(
        default=0.0,
        ge=0,
        allow_inf_nan=False,
    )
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
    def validate_tool_versions(cls, values: dict[str, str]) -> dict[str, str]:
        for name, version in values.items():
            if re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
                raise ValueError(f"invalid tool-version name: {name!r}")
            if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is None:
                raise ValueError(f"invalid semantic version for tool {name!r}")
        return dict(sorted(values.items()))

    @model_validator(mode="after")
    def validate_run(self) -> InvestigationTrace:
        if self.status is RunStatus.SUCCESS:
            if self.output is None:
                raise ValueError("successful traces require output")
            if self.failure_code is not None or self.failure_detail is not None:
                raise ValueError("successful traces forbid failure fields")
            if self.output.case_id != self.case_id:
                raise ValueError("trace and investigator output case IDs must match")
        else:
            if self.output is not None:
                raise ValueError("failed traces forbid output")
            if self.failure_code is None or self.failure_detail is None:
                raise ValueError("failed traces require failure code and detail")

        if tuple(event.sequence for event in self.events) != tuple(
            range(1, len(self.events) + 1)
        ):
            raise ValueError("trace event sequences must be ordered and contiguous")
        if self.request_count > self.config.request_limit:
            raise ValueError("reported requests exceed the configured limit")
        if self.tool_call_count > self.config.tool_calls_limit:
            raise ValueError("reported tool calls exceed the configured limit")
        if (
            self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens is not None
            and self.total_tokens != self.input_tokens + self.output_tokens
        ):
            raise ValueError("reported total tokens must equal input plus output")
        if self.monetary_provider_cost_usd != 0:
            raise ValueError("Week 3 local Ollama runs must record provider cost as zero")

        budget_failure = (
            self.status is RunStatus.FAILED
            and self.failure_code is RunFailureCode.BUDGET_EXHAUSTED
        )
        token_limits = (
            (self.input_tokens, self.config.input_tokens_limit, "input"),
            (self.output_tokens, self.config.output_tokens_limit, "output"),
            (self.total_tokens, self.config.total_tokens_limit, "total"),
        )
        for observed, limit, name in token_limits:
            if observed is not None and observed > limit and not budget_failure:
                raise ValueError(f"reported {name} tokens exceed the configured limit")
        return self
