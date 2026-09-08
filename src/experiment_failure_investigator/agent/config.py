"""Validated Week 3 runtime configuration and local Ollama preflight."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from enum import StrEnum
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

DEFAULT_MODEL_TAG = "qwen2.5:7b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
MAX_PREFLIGHT_RESPONSE_BYTES = 1_000_000


class RuntimeMode(StrEnum):
    """Supported Week 3 execution profiles."""

    MOCK = "mock"
    REPLAY = "replay"
    LOCAL = "local"
    SMOKE = "smoke"


class AgentRuntimeConfig(BaseModel):
    """Strict configuration shared by the controller, model, and trace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_mode: RuntimeMode = RuntimeMode.LOCAL
    provider: Literal["ollama"] = "ollama"
    model_tag: str = Field(default=DEFAULT_MODEL_TAG, min_length=1, max_length=200)
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    prompt_version: str = Field(
        default="1.0.0",
        pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, allow_inf_nan=False)
    max_output_tokens: int = Field(default=2048, ge=128, le=32768)
    request_limit: int = Field(default=4, ge=1, le=10)
    tool_calls_limit: int = Field(default=3, ge=0, le=10)
    input_tokens_limit: int = Field(default=30000, ge=1)
    output_tokens_limit: int = Field(default=4096, ge=1)
    total_tokens_limit: int = Field(default=34096, ge=1)
    output_validation_retries: int = Field(default=1, ge=0, le=3)
    request_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        le=600,
        allow_inf_nan=False,
    )
    run_timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        le=1800,
        allow_inf_nan=False,
    )
    max_evidence_records: int = Field(default=25, ge=1, le=50)

    @field_validator("model_tag")
    @classmethod
    def validate_model_tag(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or any(character.isspace() for character in normalized):
            raise ValueError("model tag must be non-blank and contain no whitespace")
        return normalized

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Ollama base URL must be an absolute HTTP(S) URL")
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Week 3 Ollama base URL must use a loopback host")
        if any(character.isspace() for character in normalized):
            raise ValueError("Ollama base URL must not contain whitespace")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "Ollama base URL must not contain credentials, query, or fragment"
            )
        if not parsed.path.endswith("/v1"):
            raise ValueError("Ollama base URL path must end with /v1")
        return normalized

    @model_validator(mode="after")
    def validate_budgets(self) -> AgentRuntimeConfig:
        if self.max_output_tokens > self.output_tokens_limit:
            raise ValueError(
                "maximum output tokens cannot exceed the run output-token limit"
            )
        if self.total_tokens_limit < (
            self.input_tokens_limit + self.output_tokens_limit
        ):
            raise ValueError(
                "total-token limit must cover configured input and output limits"
            )
        if self.run_timeout_seconds < self.request_timeout_seconds:
            raise ValueError("run timeout cannot be shorter than request timeout")
        return self


ENVIRONMENT_FIELDS: dict[str, str] = {
    "EFI_RUNTIME_MODE": "runtime_mode",
    "EFI_MODEL_TAG": "model_tag",
    "EFI_OLLAMA_BASE_URL": "ollama_base_url",
    "EFI_PROMPT_VERSION": "prompt_version",
    "EFI_TEMPERATURE": "temperature",
    "EFI_MAX_OUTPUT_TOKENS": "max_output_tokens",
    "EFI_REQUEST_LIMIT": "request_limit",
    "EFI_TOOL_CALLS_LIMIT": "tool_calls_limit",
    "EFI_INPUT_TOKENS_LIMIT": "input_tokens_limit",
    "EFI_OUTPUT_TOKENS_LIMIT": "output_tokens_limit",
    "EFI_TOTAL_TOKENS_LIMIT": "total_tokens_limit",
    "EFI_OUTPUT_VALIDATION_RETRIES": "output_validation_retries",
    "EFI_REQUEST_TIMEOUT_SECONDS": "request_timeout_seconds",
    "EFI_RUN_TIMEOUT_SECONDS": "run_timeout_seconds",
    "EFI_MAX_EVIDENCE_RECORDS": "max_evidence_records",
}

def config_for_mode(mode: RuntimeMode) -> AgentRuntimeConfig:
    """Build the conservative defaults for one named runtime profile."""
    if mode in {RuntimeMode.MOCK, RuntimeMode.REPLAY}:
        return AgentRuntimeConfig(
            runtime_mode=mode,
            request_timeout_seconds=10.0,
            run_timeout_seconds=30.0,
        )
    if mode is RuntimeMode.SMOKE:
        return AgentRuntimeConfig(
            runtime_mode=mode,
            request_limit=3,
            tool_calls_limit=2,
            max_output_tokens=1536,
        )
    return AgentRuntimeConfig(runtime_mode=mode)


def load_runtime_config(
    environment: Mapping[str, str] | None = None,
) -> AgentRuntimeConfig:
    """Load only allowlisted ``EFI_*`` values from an environment mapping."""
    source = os.environ if environment is None else environment
    mode = RuntimeMode(source.get("EFI_RUNTIME_MODE", RuntimeMode.LOCAL.value))
    values = config_for_mode(mode).model_dump(mode="python")
    values.update(
        {
            field_name: source[environment_name]
            for environment_name, field_name in ENVIRONMENT_FIELDS.items()
            if environment_name in source
        }
    )
    return AgentRuntimeConfig.model_validate(values)


def build_ollama_model(config: AgentRuntimeConfig) -> OllamaModel:
    """Construct the configured model without probing, starting, or pulling it."""
    provider = OllamaProvider(base_url=config.ollama_base_url)
    return OllamaModel(config.model_tag, provider=provider)


class OllamaPreflightStatus(StrEnum):
    """Typed outcome of a bounded local Ollama availability check."""

    NOT_REQUIRED = "not_required"
    READY = "ready"
    UNREACHABLE = "unreachable"
    MODEL_MISSING = "model_missing"
    INVALID_RESPONSE = "invalid_response"


class OllamaPreflightResult(BaseModel):
    """Actionable preflight result that never starts or mutates Ollama."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: OllamaPreflightStatus
    model_tag: str
    base_url: str
    tags_endpoint: str | None = None
    available_models: tuple[str, ...] = ()
    detail: str = Field(min_length=1)


def _tags_endpoint(base_url: str) -> str:
    parsed = urlsplit(base_url)
    api_root = parsed.path.removesuffix("/v1").rstrip("/")
    return urlunsplit(
        (parsed.scheme, parsed.netloc, f"{api_root}/api/tags", "", "")
    )


def _result(
    config: AgentRuntimeConfig,
    status: OllamaPreflightStatus,
    detail: str,
    *,
    tags_endpoint: str | None = None,
    available_models: tuple[str, ...] = (),
) -> OllamaPreflightResult:
    return OllamaPreflightResult(
        status=status,
        model_tag=config.model_tag,
        base_url=config.ollama_base_url,
        tags_endpoint=tags_endpoint,
        available_models=tuple(sorted(set(available_models))),
        detail=detail,
    )


def _available_model_names(decoded: object) -> tuple[str, ...]:
    if not isinstance(decoded, dict) or not isinstance(decoded.get("models"), list):
        raise TypeError
    names: list[str] = []
    for item in decoded["models"]:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model"):
            name = item.get(key)
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return tuple(names)


def preflight_ollama(
    config: AgentRuntimeConfig,
    *,
    timeout_seconds: float = 2.0,
) -> OllamaPreflightResult:
    """Check the tags endpoint without starting Ollama or downloading a model."""
    if config.runtime_mode in {RuntimeMode.MOCK, RuntimeMode.REPLAY}:
        return _result(
            config,
            OllamaPreflightStatus.NOT_REQUIRED,
            "Ollama preflight is disabled for offline runtime modes.",
        )
    if timeout_seconds <= 0 or timeout_seconds > 30:
        raise ValueError("preflight timeout must be greater than zero and at most 30")

    endpoint = _tags_endpoint(config.ollama_base_url)
    request = Request(endpoint, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read(MAX_PREFLIGHT_RESPONSE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        return _result(
            config,
            OllamaPreflightStatus.UNREACHABLE,
            "Ollama did not answer its tags endpoint. Start Ollama or correct "
            f"the configured base URL, then retry ({type(error).__name__}).",
            tags_endpoint=endpoint,
        )
    if len(payload) > MAX_PREFLIGHT_RESPONSE_BYTES:
        return _result(
            config,
            OllamaPreflightStatus.INVALID_RESPONSE,
            "Ollama tags response exceeded the preflight size limit.",
            tags_endpoint=endpoint,
        )
    try:
        available = _available_model_names(json.loads(payload))
    except (json.JSONDecodeError, TypeError):
        return _result(
            config,
            OllamaPreflightStatus.INVALID_RESPONSE,
            "Ollama tags endpoint did not return the expected model list.",
            tags_endpoint=endpoint,
        )
    if config.model_tag not in available:
        return _result(
            config,
            OllamaPreflightStatus.MODEL_MISSING,
            "Ollama is reachable, but the configured model tag is not installed. "
            "Install that tag manually or select an installed EFI_MODEL_TAG.",
            tags_endpoint=endpoint,
            available_models=available,
        )
    return _result(
        config,
        OllamaPreflightStatus.READY,
        "Ollama is reachable and the configured model tag is installed.",
        tags_endpoint=endpoint,
        available_models=available,
    )
