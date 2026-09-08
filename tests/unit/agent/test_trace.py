"""Tests for stable prompt identity and machine-readable run traces."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from experiment_failure_investigator.agent.config import AgentRuntimeConfig, RuntimeMode
from experiment_failure_investigator.agent.contracts import (
    ConfidenceCategory,
    FailureHypothesis,
    InvestigatorOutput,
)
from experiment_failure_investigator.agent.trace import (
    AgentUsage,
    InvestigationTrace,
    ModelEvent,
    ModelEventType,
    RunFailure,
    RunFailureCode,
    RunStatus,
    RuntimeSnapshot,
    ToolEvent,
    VersionRecord,
    hash_json_payload,
    hash_prompt,
    new_run_id,
)
from experiment_failure_investigator.analysis.contracts import PublicArtifactHashes
from experiment_failure_investigator.analysis.results import canonical_json

CASE_ID = "case_0123456789abcdef"
ZERO_HASH = "0" * 64
HASHES = PublicArtifactHashes(
    measurements=ZERO_HASH,
    plate_map=ZERO_HASH,
    metadata=ZERO_HASH,
    protocol=ZERO_HASH,
    problem_statement=ZERO_HASH,
)
EVIDENCE_ID = "ev_" + "a" * 20


def _output() -> InvestigatorOutput:
    hypotheses = tuple(
        FailureHypothesis(
            rank=rank,
            name=name,
            proposed_mechanism=mechanism,
            confidence=confidence,
            supporting_evidence_ids=(EVIDENCE_ID,),
            missing_evidence=("Independent preparation records",),
            alternative_explanations=("Plate handling variation",),
            falsification_check="Repeat with an independent preparation.",
        )
        for rank, name, mechanism, confidence in (
            (
                1,
                "Control preparation variation",
                "Control preparation may have reduced assay separation.",
                ConfidenceCategory.MODERATE,
            ),
            (
                2,
                "Plate handling variation",
                "Plate handling may have changed the observed control behavior.",
                ConfidenceCategory.LOW,
            ),
        )
    )
    return InvestigatorOutput(
        case_id=CASE_ID,
        hypotheses=hypotheses,
        overall_assessment="The control behavior warrants review.",
        remaining_uncertainty=("The physical origin remains unavailable.",),
        recommended_next_check="Repeat controls from an independent preparation.",
        recommendation_evidence_ids=(EVIDENCE_ID,),
        limitations=("The evidence is associative rather than causal.",),
    )


def _events() -> tuple[tuple[ModelEvent, ...], tuple[ToolEvent, ...]]:
    request = {"role": "user", "content": "Investigate the public evidence."}
    arguments = {"tool_name": "summarize_controls", "limit": 5}
    result = {"status": "success", "evidence_ids": [EVIDENCE_ID]}
    response = {"role": "assistant", "status": "validated"}
    return (
        (
            ModelEvent(
                sequence=1,
                event_type=ModelEventType.REQUEST,
                payload=request,
                payload_sha256=hash_json_payload(request),
            ),
            ModelEvent(
                sequence=3,
                event_type=ModelEventType.RESPONSE,
                payload=response,
                payload_sha256=hash_json_payload(response),
            ),
        ),
        (
            ToolEvent(
                sequence=2,
                tool_name="inspect_diagnostic_result",
                arguments=arguments,
                arguments_sha256=hash_json_payload(arguments),
                result=result,
                result_sha256=hash_json_payload(result),
            ),
        ),
    )


def _trace() -> InvestigationTrace:
    model_events, tool_events = _events()
    return InvestigationTrace(
        run_id="run_" + "b" * 20,
        case_id=CASE_ID,
        public_artifact_hashes=HASHES,
        application_commit="c8cb8a6",
        python_version="3.12.0",
        pydantic_ai_version="2.38.0",
        runtime=RuntimeSnapshot.from_config(
            AgentRuntimeConfig(runtime_mode=RuntimeMode.MOCK),
            prompt="Investigate only the supplied public evidence.",
        ),
        baseline_report_version="1.0.0",
        heuristic_version="1.0.0",
        tool_versions=(
            VersionRecord(name="summarize_controls", version="1.0.0"),
        ),
        model_events=model_events,
        tool_events=tool_events,
        status=RunStatus.SUCCESS,
        output=_output(),
        usage=AgentUsage(
            requests=2,
            tool_calls=1,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        ),
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        duration_seconds=1.25,
    )


def test_prompt_hash_uses_exact_content_and_rejects_blank() -> None:
    assert hash_prompt("prompt") == hash_prompt("prompt")
    assert hash_prompt("prompt") != hash_prompt("prompt ")
    with pytest.raises(ValueError, match="must not be blank"):
        hash_prompt("  ")


def test_run_ids_are_opaque_and_unique() -> None:
    first = new_run_id()
    second = new_run_id()

    assert first.startswith("run_")
    assert len(first) == 24
    assert first != second


def test_successful_trace_round_trips_without_private_truth() -> None:
    trace = _trace()
    serialized = canonical_json(trace)

    assert InvestigationTrace.model_validate_json(serialized) == trace
    for private_name in (
        "planted_failure_mode",
        "injection_parameters",
        "expected_discriminating_evidence",
        "root_seed",
        "child_seeds",
    ):
        assert private_name not in serialized


def test_event_payload_hashes_are_verified() -> None:
    request = {"role": "user"}
    with pytest.raises(ValidationError, match="payload hash"):
        ModelEvent(
            sequence=1,
            event_type=ModelEventType.REQUEST,
            payload=request,
            payload_sha256="0" * 64,
        )

    with pytest.raises(ValidationError, match="argument hash"):
        ToolEvent(
            sequence=1,
            tool_name="resolve_evidence",
            arguments={},
            arguments_sha256="0" * 64,
            result={},
            result_sha256=hash_json_payload({}),
        )


def test_trace_requires_contiguous_global_event_sequence() -> None:
    values = _trace().model_dump(mode="python")
    values["tool_events"][0]["sequence"] = 4

    with pytest.raises(ValidationError, match="unique and contiguous"):
        InvestigationTrace.model_validate(values)


def test_trace_terminal_states_are_mutually_exclusive() -> None:
    success = _trace().model_dump(mode="python")
    success["failure"] = RunFailure(
        code=RunFailureCode.INTERNAL_ERROR,
        detail="Sanitized internal failure.",
    )
    with pytest.raises(ValidationError, match="forbid failure"):
        InvestigationTrace.model_validate(success)

    failed = _trace().model_dump(mode="python")
    failed["status"] = RunStatus.FAILED
    failed["output"] = None
    failed["failure"] = None
    with pytest.raises(ValidationError, match="require failure"):
        InvestigationTrace.model_validate(failed)


def test_trace_rejects_wrong_output_case_and_naive_time() -> None:
    wrong_case = _trace().model_dump(mode="python")
    wrong_case["output"]["case_id"] = "case_fedcba9876543210"
    with pytest.raises(ValidationError, match="case IDs must match"):
        InvestigationTrace.model_validate(wrong_case)

    naive = _trace().model_dump(mode="python")
    naive["started_at"] = datetime(2026, 1, 1)
    with pytest.raises(ValidationError, match="timezone"):
        InvestigationTrace.model_validate(naive)


def test_trace_rejects_usage_beyond_configured_budgets() -> None:
    values = _trace().model_dump(mode="python")
    values["usage"]["requests"] = 5

    with pytest.raises(ValidationError, match="requests exceed"):
        InvestigationTrace.model_validate(values)


def test_usage_rejects_inconsistent_tokens_or_nonzero_local_cost() -> None:
    with pytest.raises(ValidationError, match="input plus output"):
        AgentUsage(
            requests=1,
            tool_calls=0,
            input_tokens=10,
            output_tokens=5,
            total_tokens=20,
        )
    with pytest.raises(ValidationError, match="cost as zero"):
        AgentUsage(requests=1, tool_calls=0, monetary_provider_cost_usd=0.01)
