"""Tests for the compact persisted Investigator trace."""

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
    InvestigationTrace,
    RunFailureCode,
    RunStatus,
    TraceEvent,
    TraceEventKind,
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
    return InvestigatorOutput(
        case_id=CASE_ID,
        hypotheses=tuple(
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
        ),
        overall_assessment="The control behavior warrants review.",
        remaining_uncertainty=("The physical origin remains unavailable.",),
        recommended_next_check="Repeat controls from an independent preparation.",
        recommendation_evidence_ids=(EVIDENCE_ID,),
        limitations=("The evidence is associative rather than causal.",),
    )


def _trace() -> InvestigationTrace:
    return InvestigationTrace(
        run_id="run_" + "b" * 20,
        case_id=CASE_ID,
        public_artifact_hashes=HASHES,
        application_commit="c8cb8a6",
        python_version="3.12.0",
        pydantic_ai_version="2.38.0",
        config=AgentRuntimeConfig(runtime_mode=RuntimeMode.MOCK),
        prompt_sha256=hash_prompt("Investigate only the supplied public evidence."),
        baseline_report_version="1.0.0",
        heuristic_version="1.0.0",
        tool_versions={"summarize_controls": "1.0.0"},
        events=(
            TraceEvent(
                sequence=1,
                kind=TraceEventKind.MODEL_REQUEST,
                payload={"role": "user", "content": "Investigate public evidence."},
            ),
            TraceEvent(
                sequence=2,
                kind=TraceEventKind.TOOL,
                tool_name="inspect_diagnostic_result",
                payload={
                    "arguments": {"tool_name": "summarize_controls", "limit": 5},
                    "result": {"evidence_ids": [EVIDENCE_ID]},
                },
            ),
            TraceEvent(
                sequence=3,
                kind=TraceEventKind.MODEL_RESPONSE,
                payload={"role": "assistant", "status": "validated"},
            ),
        ),
        status=RunStatus.SUCCESS,
        output=_output(),
        request_count=2,
        tool_call_count=1,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
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

    assert trace.trace_version == "2.0.0"
    assert InvestigationTrace.model_validate_json(serialized) == trace
    for private_name in (
        "planted_failure_mode",
        "injection_parameters",
        "expected_discriminating_evidence",
        "root_seed",
        "child_seeds",
    ):
        assert private_name not in serialized


def test_event_shape_and_order_are_validated() -> None:
    with pytest.raises(ValidationError, match="require a tool name"):
        TraceEvent(sequence=1, kind=TraceEventKind.TOOL, payload={})
    with pytest.raises(ValidationError, match="must not include a tool name"):
        TraceEvent(
            sequence=1,
            kind=TraceEventKind.MODEL_REQUEST,
            tool_name="unexpected",
            payload={},
        )

    values = _trace().model_dump(mode="python")
    values["events"][1]["sequence"] = 4
    with pytest.raises(ValidationError, match="ordered and contiguous"):
        InvestigationTrace.model_validate(values)


def test_trace_terminal_states_are_mutually_exclusive() -> None:
    success = _trace().model_dump(mode="python")
    success["failure_code"] = RunFailureCode.INTERNAL_ERROR
    success["failure_detail"] = "Sanitized internal failure."
    with pytest.raises(ValidationError, match="forbid failure"):
        InvestigationTrace.model_validate(success)

    failed = _trace().model_dump(mode="python")
    failed["status"] = RunStatus.FAILED
    failed["output"] = None
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


def test_trace_rejects_invalid_usage_and_tool_versions() -> None:
    values = _trace().model_dump(mode="python")
    values["request_count"] = 5
    with pytest.raises(ValidationError, match="requests exceed"):
        InvestigationTrace.model_validate(values)

    inconsistent = _trace().model_dump(mode="python")
    inconsistent["total_tokens"] = 200
    with pytest.raises(ValidationError, match="input plus output"):
        InvestigationTrace.model_validate(inconsistent)

    paid = _trace().model_dump(mode="python")
    paid["monetary_provider_cost_usd"] = 0.01
    with pytest.raises(ValidationError, match="cost as zero"):
        InvestigationTrace.model_validate(paid)

    version = _trace().model_dump(mode="python")
    version["tool_versions"] = {"bad-name": "1.0.0"}
    with pytest.raises(ValidationError, match="tool-version name"):
        InvestigationTrace.model_validate(version)


def test_budget_failure_trace_may_record_measured_token_overage() -> None:
    values = _trace().model_dump(mode="python")
    values.update(
        {
            "status": RunStatus.FAILED,
            "output": None,
            "failure_code": RunFailureCode.BUDGET_EXHAUSTED,
            "failure_detail": "Provider-reported token use crossed the limit.",
            "input_tokens": 40000,
            "total_tokens": 40050,
        }
    )

    trace = InvestigationTrace.model_validate(values)

    assert trace.input_tokens == 40000
