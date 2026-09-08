"""Tests for bounded single-run Investigator orchestration."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic_ai import ModelResponse
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings

import experiment_failure_investigator.agent.controller as controller_module
from experiment_failure_investigator.agent.config import (
    AgentRuntimeConfig,
    OllamaPreflightResult,
    OllamaPreflightStatus,
    RuntimeMode,
)
from experiment_failure_investigator.agent.contracts import (
    ConfidenceCategory,
    FailureHypothesis,
    InvestigatorOutput,
)
from experiment_failure_investigator.agent.controller import run_investigation
from experiment_failure_investigator.agent.trace import RunFailureCode, RunStatus
from experiment_failure_investigator.analysis.results import canonical_json
from experiment_failure_investigator.benchmark.adapter import load_investigator_case
from experiment_failure_investigator.reporting.baseline import (
    BaselineReport,
    build_baseline_report,
)

CASE_DIRECTORY = Path("cases/weak_controls_obvious")


@pytest.fixture(scope="module")
def expected_baseline(
    tmp_path_factory: pytest.TempPathFactory,
) -> BaselineReport:
    case = load_investigator_case(CASE_DIRECTORY)
    return build_baseline_report(
        case,
        tmp_path_factory.mktemp("controller-expected-plots"),
    )


@pytest.fixture(scope="module")
def expected_output(expected_baseline: BaselineReport) -> InvestigatorOutput:
    baseline = expected_baseline
    case_id = baseline.case_id
    first, second = baseline.evidence_index[:2]
    return InvestigatorOutput(
        case_id=case_id,
        hypotheses=(
            FailureHypothesis(
                rank=1,
                name="Control preparation variation",
                proposed_mechanism=(
                    "Control preparation may have reduced assay separation."
                ),
                confidence=ConfidenceCategory.MODERATE,
                supporting_evidence_ids=(first.evidence_id,),
                missing_evidence=("Independent control preparation records",),
                alternative_explanations=("Plate handling variation",),
                falsification_check=(
                    "Repeat controls using an independent preparation."
                ),
            ),
            FailureHypothesis(
                rank=2,
                name="Plate handling variation",
                proposed_mechanism="Plate handling may have changed signal quality.",
                confidence=ConfidenceCategory.LOW,
                supporting_evidence_ids=(second.evidence_id,),
                missing_evidence=("Independent plate handling records",),
                alternative_explanations=("Control preparation variation",),
                falsification_check="Repeat the assay with matched handling.",
            ),
        ),
        overall_assessment="The control behavior warrants review.",
        remaining_uncertainty=("The physical cause remains unavailable.",),
        recommended_next_check="Repeat controls from an independent preparation.",
        recommendation_evidence_ids=(first.evidence_id,),
        limitations=("The evidence does not identify a physical cause.",),
    )


def _config(**updates: object) -> AgentRuntimeConfig:
    values = AgentRuntimeConfig(
        runtime_mode=RuntimeMode.MOCK,
        request_timeout_seconds=1,
        run_timeout_seconds=2,
    ).model_dump(mode="python")
    values.update(updates)
    return AgentRuntimeConfig.model_validate(values)


def _has_tool_return(messages: list[ModelMessage]) -> bool:
    return any(
        isinstance(part, ToolReturnPart)
        for message in messages
        for part in message.parts
    )


def _one_tool_then_output_model(
    output: InvestigatorOutput,
    observed_settings: list[ModelSettings | None] | None = None,
) -> FunctionModel:
    def model_function(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        if observed_settings is not None:
            observed_settings.append(info.model_settings)
        if not _has_tool_return(messages):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "list_diagnostic_results",
                        {},
                        tool_call_id="catalog",
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    output.model_dump(mode="json"),
                    tool_call_id="output",
                )
            ]
        )

    return FunctionModel(model_function)


@pytest.mark.anyio
async def test_success_writes_validated_output_baseline_and_trace(
    tmp_path: Path,
    expected_output: InvestigatorOutput,
    expected_baseline: BaselineReport,
) -> None:
    destination = tmp_path / "run"
    observed_settings: list[ModelSettings | None] = []
    config = _config(
        temperature=0.25,
        max_output_tokens=512,
        request_timeout_seconds=0.75,
    )

    run = await run_investigation(
        CASE_DIRECTORY,
        destination,
        model=_one_tool_then_output_model(expected_output, observed_settings),
        config=config,
    )

    assert run.trace.status is RunStatus.SUCCESS
    assert run.trace.output == expected_output
    assert run.trace.failure is None
    assert run.trace.usage.requests == 2
    assert run.trace.usage.tool_calls == 1
    assert len(run.trace.tool_events) == 1
    assert run.trace.tool_events[0].tool_name == "list_diagnostic_results"
    assert run.artifacts.investigation_json is not None
    assert run.artifacts.investigation_json.read_text() == canonical_json(
        expected_output
    )
    assert run.artifacts.trace_json.read_text() == canonical_json(run.trace)
    baseline = BaselineReport.model_validate_json(
        run.artifacts.baseline_json.read_text()
    )
    assert canonical_json(baseline) == canonical_json(expected_baseline)
    assert {
        record.evidence_id for record in baseline.evidence_index
    } >= set(expected_output.recommendation_evidence_ids)
    assert observed_settings
    assert all(settings is not None for settings in observed_settings)
    nonempty_settings = [settings for settings in observed_settings if settings]
    assert all(settings["temperature"] == 0.25 for settings in nonempty_settings)
    assert all(settings["max_tokens"] == 512 for settings in nonempty_settings)
    assert all(settings["timeout"] == 0.75 for settings in nonempty_settings)
    trace_payload = run.artifacts.trace_json.read_text()
    assert "weak_controls_obvious" not in trace_payload
    assert '"failure_mode"' not in trace_payload


@pytest.mark.anyio
async def test_existing_output_directory_is_never_overwritten(
    tmp_path: Path,
    expected_output: InvestigatorOutput,
) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        await run_investigation(
            CASE_DIRECTORY,
            destination,
            model=_one_tool_then_output_model(expected_output),
            config=_config(),
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.anyio
async def test_request_budget_failure_writes_trace_but_no_investigation(
    tmp_path: Path,
    expected_output: InvestigatorOutput,
) -> None:
    run = await run_investigation(
        CASE_DIRECTORY,
        tmp_path / "request-limit",
        model=_one_tool_then_output_model(expected_output),
        config=_config(request_limit=1),
    )

    assert run.trace.status is RunStatus.FAILED
    assert run.trace.failure is not None
    assert run.trace.failure.code is RunFailureCode.BUDGET_EXHAUSTED
    assert run.trace.output is None
    assert run.artifacts.investigation_json is None
    assert not (run.artifacts.output_directory / "investigation.json").exists()
    assert run.artifacts.trace_json.exists()


@pytest.mark.anyio
async def test_tool_call_budget_is_enforced_before_excess_tools_run(
    tmp_path: Path,
) -> None:
    def model_function(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "list_diagnostic_results",
                    {},
                    tool_call_id="first",
                ),
                ToolCallPart(
                    "inspect_diagnostic_result",
                    {"tool_name": "summarize_controls", "limit": 1},
                    tool_call_id="second",
                ),
            ]
        )

    run = await run_investigation(
        CASE_DIRECTORY,
        tmp_path / "tool-limit",
        model=FunctionModel(model_function),
        config=_config(tool_calls_limit=1),
    )

    assert run.trace.status is RunStatus.FAILED
    assert run.trace.failure is not None
    assert run.trace.failure.code is RunFailureCode.BUDGET_EXHAUSTED
    assert run.trace.usage.tool_calls <= 1
    assert run.artifacts.investigation_json is None


@pytest.mark.anyio
async def test_token_budget_exhaustion_is_a_typed_failure(
    tmp_path: Path,
    expected_output: InvestigatorOutput,
) -> None:
    run = await run_investigation(
        CASE_DIRECTORY,
        tmp_path / "token-limit",
        model=_one_tool_then_output_model(expected_output),
        config=_config(input_tokens_limit=1),
    )

    assert run.trace.status is RunStatus.FAILED
    assert run.trace.failure is not None
    assert run.trace.failure.code is RunFailureCode.BUDGET_EXHAUSTED
    assert run.artifacts.investigation_json is None


@pytest.mark.anyio
async def test_exhausted_output_retry_is_a_typed_validation_failure(
    tmp_path: Path,
    expected_output: InvestigatorOutput,
) -> None:
    invalid = expected_output.model_copy(
        update={"recommendation_evidence_ids": ("ev_" + "f" * 20,)}
    )

    def model_function(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    invalid.model_dump(mode="json"),
                    tool_call_id="invalid-output",
                )
            ]
        )

    run = await run_investigation(
        CASE_DIRECTORY,
        tmp_path / "validation-limit",
        model=FunctionModel(model_function),
        config=_config(output_validation_retries=0),
    )

    assert run.trace.status is RunStatus.FAILED
    assert run.trace.failure is not None
    assert run.trace.failure.code is RunFailureCode.VALIDATION_FAILED
    assert run.trace.output is None
    assert run.artifacts.investigation_json is None


@pytest.mark.anyio
async def test_whole_run_timeout_is_a_typed_failure(tmp_path: Path) -> None:
    async def slow_model(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        await asyncio.sleep(0.1)
        return ModelResponse(parts=[])

    run = await run_investigation(
        CASE_DIRECTORY,
        tmp_path / "timeout",
        model=FunctionModel(slow_model),
        config=_config(request_timeout_seconds=0.01, run_timeout_seconds=0.02),
    )

    assert run.trace.status is RunStatus.FAILED
    assert run.trace.failure is not None
    assert run.trace.failure.code is RunFailureCode.TIMEOUT
    assert run.artifacts.investigation_json is None


@pytest.mark.anyio
async def test_provider_error_is_a_distinct_typed_failure(tmp_path: Path) -> None:
    def unavailable_model(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        raise ModelAPIError("offline-test", "provider unavailable")

    run = await run_investigation(
        CASE_DIRECTORY,
        tmp_path / "provider-error",
        model=FunctionModel(unavailable_model),
        config=_config(),
    )

    assert run.trace.status is RunStatus.FAILED
    assert run.trace.failure is not None
    assert run.trace.failure.code is RunFailureCode.PROVIDER_UNAVAILABLE
    assert run.artifacts.investigation_json is None


@pytest.mark.anyio
async def test_missing_local_model_fails_preflight_without_model_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_requested = False

    def unexpected_model(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        nonlocal model_requested
        model_requested = True
        return ModelResponse(parts=[])

    monkeypatch.setattr(
        controller_module,
        "preflight_ollama",
        lambda config, **_kwargs: OllamaPreflightResult(
            status=OllamaPreflightStatus.MODEL_MISSING,
            model_tag=config.model_tag,
            base_url=config.ollama_base_url,
            detail="Configured model tag is not installed.",
        ),
    )
    config = _config(runtime_mode=RuntimeMode.LOCAL)

    run = await run_investigation(
        CASE_DIRECTORY,
        tmp_path / "missing-model",
        model=FunctionModel(unexpected_model),
        config=config,
    )

    assert run.trace.status is RunStatus.FAILED
    assert run.trace.failure is not None
    assert run.trace.failure.code is RunFailureCode.MODEL_MISSING
    assert run.trace.usage.requests == 0
    assert model_requested is False
    assert run.artifacts.investigation_json is None
