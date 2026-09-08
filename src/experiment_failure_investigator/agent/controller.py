"""Bounded single-run orchestration for the Week 3 Investigator."""

from __future__ import annotations

import asyncio
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import JsonValue, TypeAdapter
from pydantic_ai import capture_run_messages
from pydantic_ai.exceptions import (
    ModelAPIError,
    ToolRetryError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_core import to_jsonable_python

from experiment_failure_investigator.agent.config import (
    AgentRuntimeConfig,
    OllamaPreflightStatus,
    preflight_ollama,
)
from experiment_failure_investigator.agent.contracts import (
    AgentStrictModel,
    InvestigatorOutput,
    validate_output_against_baseline,
)
from experiment_failure_investigator.agent.evidence import build_agent_briefing
from experiment_failure_investigator.agent.investigator import (
    INVESTIGATOR_TOOL_NAMES,
    InvestigatorDependencies,
    build_investigator_agent,
)
from experiment_failure_investigator.agent.prompts import (
    AssembledPrompt,
    assemble_investigator_prompt,
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
    new_run_id,
)
from experiment_failure_investigator.analysis.contracts import PublicArtifactHashes
from experiment_failure_investigator.analysis.results import canonical_json
from experiment_failure_investigator.benchmark.adapter import load_investigator_case
from experiment_failure_investigator.reporting.baseline import (
    BaselineReport,
    build_baseline_report,
    write_baseline_report,
)

_JSON_VALUE_ADAPTER = TypeAdapter(JsonValue)


class InvestigationArtifacts(AgentStrictModel):
    """Paths written during one controller execution."""

    output_directory: Path
    baseline_json: Path
    baseline_markdown: Path
    investigation_json: Path | None = None
    trace_json: Path


class InvestigationRun(AgentStrictModel):
    """Typed terminal result returned by the single-run controller."""

    trace: InvestigationTrace
    artifacts: InvestigationArtifacts


def _application_commit() -> str:
    """Return the source revision when available without making it mandatory."""
    repository = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = completed.stdout.strip()
    return commit if commit else "unknown"


def _tool_versions(baseline: BaselineReport) -> tuple[VersionRecord, ...]:
    """Extract one consistent version for every deterministic diagnostic."""
    versions: dict[str, set[str]] = {}
    for result in baseline.tool_results:
        versions.setdefault(result.tool_name, set()).add(result.tool_version)
    inconsistent = sorted(name for name, values in versions.items() if len(values) != 1)
    if inconsistent:
        raise ValueError(
            f"baseline contains inconsistent tool versions: {inconsistent}"
        )
    return tuple(
        VersionRecord(name=name, version=next(iter(versions[name])))
        for name in sorted(versions)
    )


def _json_value(value: Any) -> JsonValue:
    """Convert framework or Pydantic values into validated JSON data."""
    return _JSON_VALUE_ADAPTER.validate_python(to_jsonable_python(value))


def _trace_events(
    messages: list[ModelMessage],
) -> tuple[tuple[ModelEvent, ...], tuple[ToolEvent, ...]]:
    """Convert captured framework messages into sanitized ordered trace events."""
    dumped = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    model_events: list[ModelEvent] = []
    tool_events: list[ToolEvent] = []
    tool_calls: dict[str, ToolCallPart] = {}
    sequence = 0
    for message, payload in zip(messages, dumped, strict=True):
        sequence += 1
        event_type = ModelEventType.RESPONSE
        if isinstance(message, ModelRequest):
            event_type = (
                ModelEventType.VALIDATION_RETRY
                if any(isinstance(part, RetryPromptPart) for part in message.parts)
                else ModelEventType.REQUEST
            )
        model_payload = _json_value(payload)
        model_events.append(
            ModelEvent(
                sequence=sequence,
                event_type=event_type,
                payload=model_payload,
                payload_sha256=hash_json_payload(model_payload),
            )
        )
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if (
                    isinstance(part, ToolCallPart)
                    and part.tool_name in INVESTIGATOR_TOOL_NAMES
                    and part.tool_call_id is not None
                ):
                    tool_calls[part.tool_call_id] = part
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if (
                    not isinstance(part, ToolReturnPart)
                    or part.tool_name not in INVESTIGATOR_TOOL_NAMES
                    or part.tool_call_id is None
                ):
                    continue
                call = tool_calls.get(part.tool_call_id)
                if call is None:
                    raise ValueError("captured tool return has no matching tool call")
                arguments = _json_value(call.args_as_dict(raise_if_invalid=True))
                result = _json_value(part.content)
                sequence += 1
                tool_events.append(
                    ToolEvent(
                        sequence=sequence,
                        tool_name=part.tool_name,
                        arguments=arguments,
                        arguments_sha256=hash_json_payload(arguments),
                        result=result,
                        result_sha256=hash_json_payload(result),
                    )
                )
    return tuple(model_events), tuple(tool_events)


def _agent_usage(usage: RunUsage) -> AgentUsage:
    """Project framework usage counters into the stable trace contract."""
    return AgentUsage(
        requests=usage.requests,
        tool_calls=usage.tool_calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        monetary_provider_cost_usd=0.0,
    )


def _failure(
    code: RunFailureCode,
    error: BaseException,
    messages: list[ModelMessage],
) -> RunFailure:
    """Create a concise typed failure without retaining a traceback."""
    retries = sum(
        isinstance(part, RetryPromptPart)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
    )
    detail = str(error).strip() or type(error).__name__
    return RunFailure(
        code=code,
        detail=f"{type(error).__name__}: {detail}"[:1000],
        validation_attempts=retries,
    )


def _classify_failure(error: BaseException) -> RunFailureCode:
    """Map framework and timeout exceptions to stable terminal categories."""
    if isinstance(error, TimeoutError):
        return RunFailureCode.TIMEOUT
    if isinstance(error, UsageLimitExceeded):
        return RunFailureCode.BUDGET_EXHAUSTED
    if isinstance(error, ModelAPIError):
        return RunFailureCode.PROVIDER_UNAVAILABLE
    if isinstance(error, (ToolRetryError, UnexpectedModelBehavior)):
        return RunFailureCode.VALIDATION_FAILED
    return RunFailureCode.INTERNAL_ERROR


def _runtime_snapshot(
    config: AgentRuntimeConfig,
    prompt: AssembledPrompt,
) -> RuntimeSnapshot:
    """Build traceable runtime settings using the reviewed system prompt hash."""
    return RuntimeSnapshot.from_config(config, prompt=prompt.system_prompt)


def _build_trace(
    *,
    case_id: str,
    public_artifact_hashes: PublicArtifactHashes,
    baseline: BaselineReport,
    config: AgentRuntimeConfig,
    prompt: AssembledPrompt,
    messages: list[ModelMessage],
    usage: RunUsage,
    started_at: datetime,
    duration_seconds: float,
    output: InvestigatorOutput | None,
    failure: RunFailure | None,
) -> InvestigationTrace:
    """Assemble the terminal trace after success or a handled run failure."""
    model_events, tool_events = _trace_events(messages)
    status = RunStatus.SUCCESS if output is not None else RunStatus.FAILED
    return InvestigationTrace(
        run_id=new_run_id(),
        case_id=case_id,
        public_artifact_hashes=public_artifact_hashes,
        application_commit=_application_commit(),
        python_version=platform.python_version(),
        pydantic_ai_version=package_version("pydantic-ai"),
        runtime=_runtime_snapshot(config, prompt),
        baseline_report_version=baseline.report_version,
        heuristic_version=baseline.heuristic_configuration.version,
        tool_versions=_tool_versions(baseline),
        model_events=model_events,
        tool_events=tool_events,
        status=status,
        output=output,
        failure=failure,
        usage=_agent_usage(usage),
        started_at=started_at,
        duration_seconds=duration_seconds,
    )


async def run_investigation(
    case_directory: Path,
    output_directory: Path,
    *,
    model: Model,
    config: AgentRuntimeConfig,
) -> InvestigationRun:
    """Execute one bounded investigation and write only validated artifacts."""
    started_at = datetime.now(timezone.utc)
    started_clock = perf_counter()
    output_directory.mkdir(parents=True, exist_ok=False)
    case = load_investigator_case(case_directory)
    baseline_directory = output_directory / "baseline"
    baseline = build_baseline_report(case, baseline_directory / "plots")
    baseline_json, baseline_markdown = write_baseline_report(
        baseline,
        baseline_directory,
    )
    briefing = build_agent_briefing(case, baseline)
    prompt = assemble_investigator_prompt(
        briefing,
        version=config.prompt_version,
    )
    dependencies = InvestigatorDependencies(
        case=case,
        baseline=baseline,
        max_evidence_records=config.max_evidence_records,
    )
    agent = build_investigator_agent(
        model,
        system_prompt=prompt.system_prompt,
        output_validation_retries=config.output_validation_retries,
    )
    limits = UsageLimits(
        request_limit=config.request_limit,
        tool_calls_limit=config.tool_calls_limit,
        input_tokens_limit=config.input_tokens_limit,
        output_tokens_limit=config.output_tokens_limit,
        total_tokens_limit=config.total_tokens_limit,
    )
    settings = ModelSettings(
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
        timeout=config.request_timeout_seconds,
    )
    usage = RunUsage()
    output: InvestigatorOutput | None = None
    failure: RunFailure | None = None
    with capture_run_messages() as messages:
        try:
            preflight = preflight_ollama(
                config,
                timeout_seconds=min(config.request_timeout_seconds, 2.0),
            )
            if preflight.status not in {
                OllamaPreflightStatus.NOT_REQUIRED,
                OllamaPreflightStatus.READY,
            }:
                code = (
                    RunFailureCode.MODEL_MISSING
                    if preflight.status is OllamaPreflightStatus.MODEL_MISSING
                    else RunFailureCode.PROVIDER_UNAVAILABLE
                )
                failure = RunFailure(code=code, detail=preflight.detail)
            remaining_seconds = config.run_timeout_seconds - (
                perf_counter() - started_clock
            )
            if failure is None and remaining_seconds <= 0:
                raise TimeoutError("run timeout elapsed during deterministic setup")
            if failure is None:
                async with asyncio.timeout(remaining_seconds):
                    result = await agent.run(
                        prompt.user_prompt,
                        deps=dependencies,
                        model_settings=settings,
                        usage_limits=limits,
                        usage=usage,
                    )
                output = validate_output_against_baseline(result.output, baseline)
        except Exception as error:
            failure = _failure(_classify_failure(error), error, messages)
    duration_seconds = perf_counter() - started_clock
    trace = _build_trace(
        case_id=case.case_id,
        public_artifact_hashes=case.public_artifact_hashes,
        baseline=baseline,
        config=config,
        prompt=prompt,
        messages=messages,
        usage=usage,
        started_at=started_at,
        duration_seconds=duration_seconds,
        output=output,
        failure=failure,
    )
    investigation_path: Path | None = None
    if output is not None:
        investigation_path = output_directory / "investigation.json"
        investigation_path.write_text(canonical_json(output), encoding="utf-8")
    trace_path = output_directory / "trace.json"
    trace_path.write_text(canonical_json(trace), encoding="utf-8")
    return InvestigationRun(
        trace=trace,
        artifacts=InvestigationArtifacts.model_validate(
            {
                "output_directory": output_directory,
                "baseline_json": baseline_json,
                "baseline_markdown": baseline_markdown,
                "investigation_json": investigation_path,
                "trace_json": trace_path,
            }
        ),
    )
