"""Tests for Week 3 runtime configuration and Ollama preflight."""

from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request

import pytest
from pydantic import ValidationError

import experiment_failure_investigator.agent.config as config_module
from experiment_failure_investigator.agent.config import (
    AgentRuntimeConfig,
    OllamaPreflightStatus,
    RuntimeMode,
    build_ollama_model,
    config_for_mode,
    load_runtime_config,
    preflight_ollama,
)


class _Response:
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        return self.payload[:size]


def test_defaults_are_local_bounded_and_repeatable() -> None:
    first = load_runtime_config({})
    second = load_runtime_config({})

    assert first == second
    assert first.runtime_mode is RuntimeMode.LOCAL
    assert first.model_tag == "qwen2.5:7b"
    assert first.ollama_base_url == "http://localhost:11434/v1"
    assert first.request_limit == 4
    assert first.tool_calls_limit == 3


def test_named_profiles_have_conservative_mode_specific_limits() -> None:
    mock = config_for_mode(RuntimeMode.MOCK)
    smoke = config_for_mode(RuntimeMode.SMOKE)

    assert mock.run_timeout_seconds == 30
    assert smoke.request_limit == 3
    assert smoke.tool_calls_limit == 2
    assert smoke.max_output_tokens == 1536


def test_allowlisted_environment_overrides_are_parsed() -> None:
    config = load_runtime_config(
        {
            "EFI_RUNTIME_MODE": "replay",
            "EFI_MODEL_TAG": "qwen3:8b",
            "EFI_TEMPERATURE": "0.25",
            "EFI_REQUEST_LIMIT": "2",
            "UNRELATED_SECRET": "ignored",
        }
    )

    assert config.runtime_mode is RuntimeMode.REPLAY
    assert config.model_tag == "qwen3:8b"
    assert config.temperature == 0.25
    assert config.request_limit == 2
    assert "UNRELATED_SECRET" not in config.model_dump()


@pytest.mark.parametrize(
    "values",
    [
        {"runtime_mode": "remote"},
        {"provider": "openai"},
        {"ollama_base_url": "http://localhost:11434"},
        {"ollama_base_url": "file:///tmp/ollama/v1"},
        {"ollama_base_url": "https://example.com/v1"},
        {"ollama_base_url": "http://user:secret@localhost:11434/v1"},
        {"request_limit": 0},
        {"max_output_tokens": 5000, "output_tokens_limit": 4096},
        {"total_tokens_limit": 10},
        {"request_timeout_seconds": 200, "run_timeout_seconds": 100},
    ],
)
def test_invalid_configuration_fails_closed(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AgentRuntimeConfig.model_validate(values)


def test_unknown_environment_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="not-a-mode"):
        load_runtime_config({"EFI_RUNTIME_MODE": "not-a-mode"})


def test_model_construction_uses_explicit_tag_without_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_module,
        "preflight_ollama",
        lambda *_args, **_kwargs: pytest.fail("model construction probed Ollama"),
    )
    config = AgentRuntimeConfig(model_tag="qwen3:8b")

    model = build_ollama_model(config)

    assert model.model_name == "qwen3:8b"


@pytest.mark.parametrize("mode", [RuntimeMode.MOCK, RuntimeMode.REPLAY])
def test_offline_modes_skip_preflight_network(
    mode: RuntimeMode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_module,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("offline preflight opened a URL"),
    )

    result = preflight_ollama(AgentRuntimeConfig(runtime_mode=mode))

    assert result.status is OllamaPreflightStatus.NOT_REQUIRED
    assert result.tags_endpoint is None


def test_preflight_reports_ready_model_and_uses_tags_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls: list[str] = []

    def open_response(request: Request, *, timeout: float) -> _Response:
        requested_urls.append(request.full_url)
        assert timeout == 1.5
        return _Response({"models": [{"name": "qwen2.5:7b"}]})

    monkeypatch.setattr(config_module, "urlopen", open_response)

    result = preflight_ollama(AgentRuntimeConfig(), timeout_seconds=1.5)

    assert result.status is OllamaPreflightStatus.READY
    assert result.available_models == ("qwen2.5:7b",)
    assert requested_urls == ["http://localhost:11434/api/tags"]


def test_preflight_reports_missing_model_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_module,
        "urlopen",
        lambda *_args, **_kwargs: _Response({"models": [{"model": "qwen3:8b"}]}),
    )

    result = preflight_ollama(AgentRuntimeConfig())

    assert result.status is OllamaPreflightStatus.MODEL_MISSING
    assert result.available_models == ("qwen3:8b",)
    assert "not installed" in result.detail


@pytest.mark.parametrize("response", [{"unexpected": []}, {"models": "bad"}])
def test_preflight_rejects_invalid_payloads(
    response: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_module,
        "urlopen",
        lambda *_args, **_kwargs: _Response(response),
    )

    result = preflight_ollama(AgentRuntimeConfig())

    assert result.status is OllamaPreflightStatus.INVALID_RESPONSE


def test_preflight_reports_unreachable_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise URLError("connection refused")

    monkeypatch.setattr(config_module, "urlopen", fail)

    result = preflight_ollama(AgentRuntimeConfig())

    assert result.status is OllamaPreflightStatus.UNREACHABLE
    assert "URLError" in result.detail
