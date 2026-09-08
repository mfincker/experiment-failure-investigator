"""Tests for the Pydantic AI Investigator and its evidence boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import ModelResponse, capture_run_messages
from pydantic_ai.messages import (
    ModelMessage,
    RetryPromptPart,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from experiment_failure_investigator.agent.contracts import (
    ConfidenceCategory,
    FailureHypothesis,
    InvestigatorOutput,
)
from experiment_failure_investigator.agent.evidence import (
    DiagnosticResultList,
    EvidencePage,
    ResolvedEvidence,
    build_agent_briefing,
)
from experiment_failure_investigator.agent.investigator import (
    EvidenceToolError,
    InvestigatorDependencies,
    build_investigator_agent,
)
from experiment_failure_investigator.agent.prompts import (
    assemble_investigator_prompt,
    load_system_prompt,
)
from experiment_failure_investigator.analysis.contracts import InvestigatorCase
from experiment_failure_investigator.benchmark.adapter import load_investigator_case
from experiment_failure_investigator.reporting.baseline import (
    BaselineReport,
    build_baseline_report,
)

TOOL_NAMES = {
    "inspect_diagnostic_result",
    "list_diagnostic_results",
    "resolve_evidence",
}


@pytest.fixture(scope="module")
def case_and_baseline(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[InvestigatorCase, BaselineReport]:
    case = load_investigator_case(Path("cases/weak_controls_obvious"))
    baseline = build_baseline_report(
        case,
        tmp_path_factory.mktemp("agent-investigator-plots"),
    )
    return case, baseline


def _valid_output(baseline: BaselineReport) -> InvestigatorOutput:
    first, second = baseline.evidence_index[:2]
    return InvestigatorOutput(
        case_id=baseline.case_id,
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


def _run_inputs(
    case: InvestigatorCase,
    baseline: BaselineReport,
) -> tuple[InvestigatorDependencies, str, str]:
    prompt = assemble_investigator_prompt(build_agent_briefing(case, baseline))
    dependencies = InvestigatorDependencies(case=case, baseline=baseline)
    return dependencies, prompt.system_prompt, prompt.user_prompt


def _tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def test_test_model_runs_with_typed_dependencies_and_only_registered_tools(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
) -> None:
    case, baseline = case_and_baseline
    dependencies, system_prompt, user_prompt = _run_inputs(case, baseline)
    expected = _valid_output(baseline)
    model = TestModel(
        call_tools=["list_diagnostic_results"],
        custom_output_args=expected.model_dump(mode="json"),
    )
    agent = build_investigator_agent(model, system_prompt=system_prompt)

    with capture_run_messages() as messages:
        result = agent.run_sync(user_prompt, deps=dependencies)

    assert result.output == expected
    assert model.last_model_request_parameters is not None
    assert {
        tool.name
        for tool in model.last_model_request_parameters.declared_function_tools
    } == TOOL_NAMES
    returns = _tool_returns(messages)
    catalog_return = next(
        item for item in returns if item.tool_name == "list_diagnostic_results"
    )
    assert isinstance(catalog_return.content, DiagnosticResultList)
    assert any(
        isinstance(part, SystemPromptPart) and part.content == system_prompt
        for message in messages
        for part in message.parts
    )
    assert any(
        isinstance(part, UserPromptPart) and part.content == user_prompt
        for message in messages
        for part in message.parts
    )


def test_function_model_can_select_all_three_bounded_tools(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
) -> None:
    case, baseline = case_and_baseline
    dependencies, system_prompt, user_prompt = _run_inputs(case, baseline)
    expected = _valid_output(baseline)
    selected_tools: list[set[str]] = []

    def model_function(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        if not _tool_returns(messages):
            selected_tools.append({tool.name for tool in info.function_tools})
            return ModelResponse(
                parts=[
                    ToolCallPart("list_diagnostic_results", {}, tool_call_id="catalog"),
                    ToolCallPart(
                        "inspect_diagnostic_result",
                        {
                            "tool_name": "summarize_controls",
                            "metric_prefix": "controls",
                            "plate_id": "plate_01",
                            "offset": 0,
                            "limit": 2,
                        },
                        tool_call_id="inspect",
                    ),
                    ToolCallPart(
                        "resolve_evidence",
                        {"evidence_ids": [baseline.evidence_index[0].evidence_id]},
                        tool_call_id="resolve",
                    ),
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    expected.model_dump(mode="json"),
                    tool_call_id="output",
                )
            ]
        )

    agent = build_investigator_agent(
        FunctionModel(model_function),
        system_prompt=system_prompt,
    )

    with capture_run_messages() as messages:
        result = agent.run_sync(user_prompt, deps=dependencies)

    assert result.output == expected
    assert selected_tools == [TOOL_NAMES]
    returns = {item.tool_name: item.content for item in _tool_returns(messages)}
    assert isinstance(returns["list_diagnostic_results"], DiagnosticResultList)
    assert isinstance(returns["inspect_diagnostic_result"], EvidencePage)
    assert isinstance(returns["resolve_evidence"], ResolvedEvidence)


def test_semantically_invalid_tool_request_returns_typed_error(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
) -> None:
    case, baseline = case_and_baseline
    dependencies, system_prompt, user_prompt = _run_inputs(case, baseline)
    expected = _valid_output(baseline)

    def model_function(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        if not _tool_returns(messages):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "inspect_diagnostic_result",
                        {"tool_name": "read_arbitrary_file", "limit": 2},
                        tool_call_id="invalid",
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    expected.model_dump(mode="json"),
                    tool_call_id="output",
                )
            ]
        )

    agent = build_investigator_agent(
        FunctionModel(model_function),
        system_prompt=system_prompt,
    )

    with capture_run_messages() as messages:
        agent.run_sync(user_prompt, deps=dependencies)

    invalid_return = next(
        item
        for item in _tool_returns(messages)
        if item.tool_name == "inspect_diagnostic_result"
    )
    assert isinstance(invalid_return.content, EvidenceToolError)
    assert invalid_return.content.code == "invalid_request"
    assert "allowlisted" in invalid_return.content.detail


def test_output_validator_requests_correction_for_unknown_evidence(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
) -> None:
    case, baseline = case_and_baseline
    dependencies, system_prompt, user_prompt = _run_inputs(case, baseline)
    expected = _valid_output(baseline)
    invalid = expected.model_copy(
        update={"recommendation_evidence_ids": ("ev_" + "f" * 20,)}
    )
    attempts = 0

    def model_function(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        output = invalid if attempts == 1 else expected
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    output.model_dump(mode="json"),
                    tool_call_id=f"output-{attempts}",
                )
            ]
        )

    agent = build_investigator_agent(
        FunctionModel(model_function),
        system_prompt=system_prompt,
        output_validation_retries=1,
    )

    with capture_run_messages() as messages:
        result = agent.run_sync(user_prompt, deps=dependencies)

    assert result.output == expected
    assert attempts == 2
    assert any(
        isinstance(part, RetryPromptPart)
        for message in messages
        for part in message.parts
    )


def test_dependencies_and_agent_limits_fail_before_model_execution(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
) -> None:
    case, baseline = case_and_baseline

    with pytest.raises(ValueError, match="between one"):
        InvestigatorDependencies(
            case=case,
            baseline=baseline,
            max_evidence_records=0,
        )
    with pytest.raises(ValueError, match="blank"):
        build_investigator_agent(TestModel(), system_prompt=" ")
    with pytest.raises(ValueError, match="between zero"):
        build_investigator_agent(
            TestModel(),
            system_prompt=load_system_prompt().content,
            output_validation_retries=4,
        )


def test_registered_tool_schemas_expose_no_arbitrary_resource_access(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
) -> None:
    case, baseline = case_and_baseline
    dependencies, system_prompt, user_prompt = _run_inputs(case, baseline)
    expected = _valid_output(baseline)
    model = TestModel(
        call_tools=[],
        custom_output_args=expected.model_dump(mode="json"),
    )
    agent = build_investigator_agent(model, system_prompt=system_prompt)

    agent.run_sync(user_prompt, deps=dependencies)

    assert model.last_model_request_parameters is not None
    schemas = " ".join(
        str(tool.parameters_json_schema)
        for tool in model.last_model_request_parameters.declared_function_tools
    ).lower()
    for forbidden in ("filesystem", "manifest", "python", "shell", "url", "path"):
        assert forbidden not in schemas
