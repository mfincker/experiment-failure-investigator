"""Offline end-to-end tests for mock and fail-closed replay execution."""

from __future__ import annotations

from pathlib import Path

import pydantic_ai.models
import pytest
from pydantic_ai.models.test import TestModel

from experiment_failure_investigator.agent.config import (
    AgentRuntimeConfig,
    RuntimeMode,
    build_ollama_model,
)
from experiment_failure_investigator.agent.controller import run_investigation
from experiment_failure_investigator.agent.replay import (
    ReplayExpectedRequest,
    ReplayModel,
    ReplayRequestKind,
    ReplayToolReference,
    build_replay_model,
    load_replay_fixture,
)
from experiment_failure_investigator.agent.trace import (
    RunFailureCode,
    RunStatus,
    TraceEventKind,
)

CASE_DIRECTORY = Path("cases/weak_controls_obvious")
REPLAY_PATH = Path("tests/fixtures/agent/replay_week3_v1.json")


def _config(**updates: object) -> AgentRuntimeConfig:
    values = AgentRuntimeConfig(
        runtime_mode=RuntimeMode.MOCK,
        request_timeout_seconds=5,
        run_timeout_seconds=15,
    ).model_dump(mode="python")
    values.update(updates)
    return AgentRuntimeConfig.model_validate(values)


@pytest.mark.anyio
async def test_test_model_exercises_complete_controller_without_tools(
    tmp_path: Path,
) -> None:
    fixture = load_replay_fixture(REPLAY_PATH, "direct_output")
    expected = fixture.turns[0].respond.output
    assert expected is not None
    model = TestModel(call_tools=[], custom_output_args=expected.model_dump(mode="json"))

    run = await run_investigation(
        CASE_DIRECTORY,
        tmp_path / "mock",
        model=model,
        config=_config(),
    )

    assert run.trace.status is RunStatus.SUCCESS
    assert run.trace.output == expected
    assert all(event.kind is not TraceEventKind.TOOL for event in run.trace.events)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("scenario", "expected_tool_calls", "expected_retries"),
    [
        ("direct_output", 0, 0),
        ("catalog_then_output", 1, 0),
        ("bounded_evidence_then_output", 3, 0),
        ("retry_corrected_output", 0, 1),
    ],
)
async def test_replay_scenarios_exercise_expected_agent_path(
    tmp_path: Path,
    scenario: str,
    expected_tool_calls: int,
    expected_retries: int,
) -> None:
    fixture = load_replay_fixture(REPLAY_PATH, scenario)
    model = build_replay_model(fixture)

    run = await run_investigation(
        CASE_DIRECTORY,
        tmp_path / scenario,
        model=model,
        config=_config(runtime_mode=RuntimeMode.REPLAY),
    )

    model.assert_complete()
    assert run.trace.status is RunStatus.SUCCESS
    assert run.trace.output == fixture.turns[-1].respond.output
    assert run.trace.tool_call_count == expected_tool_calls
    assert sum(
        event.kind is TraceEventKind.TOOL for event in run.trace.events
    ) == expected_tool_calls
    assert run.trace.failure_code is None
    retries = sum(
        event.kind is TraceEventKind.VALIDATION_RETRY for event in run.trace.events
    )
    assert retries == expected_retries


@pytest.mark.anyio
async def test_replaying_same_fixture_preserves_scientific_output(
    tmp_path: Path,
) -> None:
    fixture = load_replay_fixture(REPLAY_PATH, "catalog_then_output")
    first_model = ReplayModel(fixture)
    second_model = ReplayModel(fixture)

    first = await run_investigation(
        CASE_DIRECTORY,
        tmp_path / "first",
        model=first_model,
        config=_config(runtime_mode=RuntimeMode.REPLAY),
    )
    second = await run_investigation(
        CASE_DIRECTORY,
        tmp_path / "second",
        model=second_model,
        config=_config(runtime_mode=RuntimeMode.REPLAY),
    )

    first_model.assert_complete()
    second_model.assert_complete()
    assert first.trace.output == second.trace.output
    assert first.investigation_json is not None
    assert second.investigation_json is not None
    assert (
        first.investigation_json.read_bytes()
        == second.investigation_json.read_bytes()
    )


@pytest.mark.anyio
async def test_replay_can_exercise_post_response_token_budget(
    tmp_path: Path,
) -> None:
    fixture = load_replay_fixture(REPLAY_PATH, "token_budget_exhausted")
    model = ReplayModel(fixture)

    run = await run_investigation(
        CASE_DIRECTORY,
        tmp_path / "budget",
        model=model,
        config=_config(runtime_mode=RuntimeMode.REPLAY),
    )

    model.assert_complete()
    assert run.trace.status is RunStatus.FAILED
    assert run.trace.failure_code is RunFailureCode.BUDGET_EXHAUSTED
    assert run.trace.input_tokens == 40000
    assert run.investigation_json is None


@pytest.mark.anyio
async def test_replay_transition_mismatch_fails_without_fallback(
    tmp_path: Path,
) -> None:
    fixture = load_replay_fixture(REPLAY_PATH, "direct_output")
    wrong_expectation = ReplayExpectedRequest(
        kind=ReplayRequestKind.TOOL_RESULTS,
        tool_results=(
            ReplayToolReference(
                tool_name="list_diagnostic_results",
                tool_call_id="unobserved",
            ),
        ),
    )
    first_turn = fixture.turns[0].model_copy(
        update={"expect": wrong_expectation}
    )
    mismatched = fixture.model_copy(update={"turns": (first_turn,)})
    model = ReplayModel(mismatched)

    run = await run_investigation(
        CASE_DIRECTORY,
        tmp_path / "mismatch",
        model=model,
        config=_config(runtime_mode=RuntimeMode.REPLAY),
    )

    assert model.consumed_turns == 0
    assert run.trace.status is RunStatus.FAILED
    assert run.trace.failure_code is RunFailureCode.INTERNAL_ERROR
    assert run.trace.failure_detail is not None
    assert "ReplayMismatchError" in run.trace.failure_detail
    assert run.investigation_json is None


@pytest.mark.anyio
async def test_real_provider_request_is_disabled_in_default_suite(
    tmp_path: Path,
) -> None:
    assert pydantic_ai.models.ALLOW_MODEL_REQUESTS is False
    config = _config(runtime_mode=RuntimeMode.MOCK)

    run = await run_investigation(
        CASE_DIRECTORY,
        tmp_path / "forbidden-network",
        model=build_ollama_model(config),
        config=config,
    )

    assert run.trace.status is RunStatus.FAILED
    assert run.trace.failure_code is RunFailureCode.INTERNAL_ERROR
    assert run.trace.failure_detail is not None
    assert "ALLOW_MODEL_REQUESTS is False" in run.trace.failure_detail
    assert run.trace.request_count == 0
    assert run.investigation_json is None


def test_replay_fixture_contains_no_private_benchmark_truth() -> None:
    payload = REPLAY_PATH.read_text(encoding="utf-8").lower()
    for forbidden in (
        "failure_mode",
        "ground_truth",
        "manifest",
        "weak_controls_obvious",
    ):
        assert forbidden not in payload


def test_unknown_replay_scenario_does_not_select_a_default() -> None:
    with pytest.raises(ValueError, match="unknown replay scenario"):
        load_replay_fixture(REPLAY_PATH, "absent_scenario")
