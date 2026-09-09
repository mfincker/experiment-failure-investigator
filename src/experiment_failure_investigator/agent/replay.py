"""Fail-closed offline replay of sanitized Investigator model decisions."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field, JsonValue, field_validator, model_validator
from pydantic_ai import ModelResponse
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from experiment_failure_investigator.agent.contracts import (
    AgentStrictModel,
    InvestigatorOutput,
)
from experiment_failure_investigator.agent.investigator import INVESTIGATOR_TOOL_NAMES

REPLAY_FIXTURE_VERSION = "1.0.0"


class ReplayMismatchError(RuntimeError):
    """Raised when a live agent transition differs from the replay fixture."""


class ReplayRequestKind(StrEnum):
    """Request transitions that a replay fixture may expect."""

    INITIAL = "initial"
    TOOL_RESULTS = "tool_results"
    VALIDATION_RETRY = "validation_retry"


class ReplayToolReference(AgentStrictModel):
    """Identity of one tool result expected in the next model request."""

    tool_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    tool_call_id: str = Field(min_length=1)

    @field_validator("tool_name")
    @classmethod
    def require_investigator_tool(cls, value: str) -> str:
        if value not in INVESTIGATOR_TOOL_NAMES:
            raise ValueError("replay fixtures may reference only Investigator tools")
        return value


class ReplayExpectedRequest(AgentStrictModel):
    """Semantic request state required before a recorded response is returned."""

    kind: ReplayRequestKind
    tool_results: tuple[ReplayToolReference, ...] = ()
    retry_contains: str | None = None

    @model_validator(mode="after")
    def validate_kind_fields(self) -> ReplayExpectedRequest:
        if self.kind is ReplayRequestKind.INITIAL:
            if self.tool_results or self.retry_contains is not None:
                raise ValueError("an initial request cannot expect results or a retry")
        elif self.kind is ReplayRequestKind.TOOL_RESULTS:
            if not self.tool_results or self.retry_contains is not None:
                raise ValueError("a tool-results request requires only tool results")
        elif not self.retry_contains or self.tool_results:
            raise ValueError("a validation retry requires only retry_contains")
        return self


class ReplayToolCall(AgentStrictModel):
    """One recorded application-tool decision returned by the replay model."""

    tool_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    tool_call_id: str = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("tool_name")
    @classmethod
    def require_investigator_tool(cls, value: str) -> str:
        if value not in INVESTIGATOR_TOOL_NAMES:
            raise ValueError("replay fixtures may call only Investigator tools")
        return value


class ReplayUsage(AgentStrictModel):
    """Optional recorded usage for testing a post-response token boundary."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ReplayResponse(AgentStrictModel):
    """Exactly one recorded tool-call batch or structured output decision."""

    tool_calls: tuple[ReplayToolCall, ...] = ()
    output: InvestigatorOutput | None = None
    usage: ReplayUsage | None = None

    @model_validator(mode="after")
    def require_one_response_kind(self) -> ReplayResponse:
        if bool(self.tool_calls) == (self.output is not None):
            raise ValueError("a replay response requires tool_calls or output, not both")
        return self


class ReplayTurn(AgentStrictModel):
    """One expected request transition and its recorded model decision."""

    expect: ReplayExpectedRequest
    respond: ReplayResponse


class ReplayFixture(AgentStrictModel):
    """Versioned sanitized exchange for one opaque public case identifier."""

    scenario: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    case_id: str = Field(pattern=r"^case_[0-9a-f]{16}$")
    turns: tuple[ReplayTurn, ...] = Field(min_length=1)
    fixture_version: str = Field(
        default=REPLAY_FIXTURE_VERSION,
        pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$",
    )

    @field_validator("fixture_version")
    @classmethod
    def require_supported_version(cls, value: str) -> str:
        if value != REPLAY_FIXTURE_VERSION:
            raise ValueError(f"unsupported replay fixture version: {value}")
        return value


class ReplayFixtureCollection(AgentStrictModel):
    """Named replay scenarios stored together in one reviewed JSON artifact."""

    scenarios: tuple[ReplayFixture, ...] = Field(min_length=1)
    fixture_version: str = Field(
        default=REPLAY_FIXTURE_VERSION,
        pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$",
    )

    @field_validator("fixture_version")
    @classmethod
    def require_supported_version(cls, value: str) -> str:
        if value != REPLAY_FIXTURE_VERSION:
            raise ValueError(f"unsupported replay fixture version: {value}")
        return value

    @field_validator("scenarios")
    @classmethod
    def require_unique_scenarios(
        cls,
        values: tuple[ReplayFixture, ...],
    ) -> tuple[ReplayFixture, ...]:
        names = [value.scenario for value in values]
        if len(names) != len(set(names)):
            raise ValueError("replay scenario names must be unique")
        return values

    def select(self, scenario: str) -> ReplayFixture:
        """Return one exact scenario without substituting a default."""
        for fixture in self.scenarios:
            if fixture.scenario == scenario:
                return fixture
        raise ValueError(f"unknown replay scenario: {scenario}")


def load_replay_fixture(path: Path, scenario: str) -> ReplayFixture:
    """Load and validate one named scenario from a reviewed JSON fixture."""
    collection = ReplayFixtureCollection.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    return collection.select(scenario)


def _request_transition(
    messages: list[ModelMessage],
) -> tuple[ReplayRequestKind, tuple[ReplayToolReference, ...], str]:
    """Project the latest framework request into a stable semantic transition."""
    if not messages or not isinstance(messages[-1], ModelRequest):
        raise ReplayMismatchError("replay expected the latest message to be a request")
    request = messages[-1]
    retries = [part for part in request.parts if isinstance(part, RetryPromptPart)]
    results = [part for part in request.parts if isinstance(part, ToolReturnPart)]
    if retries and results:
        raise ReplayMismatchError("replay does not accept mixed retry and tool results")
    if retries:
        retry_text = "\n".join(str(part.content) for part in retries)
        return ReplayRequestKind.VALIDATION_RETRY, (), retry_text
    if results:
        references = tuple(
            ReplayToolReference(
                tool_name=part.tool_name,
                tool_call_id=part.tool_call_id,
            )
            for part in results
        )
        return ReplayRequestKind.TOOL_RESULTS, references, ""
    return ReplayRequestKind.INITIAL, (), ""


class ReplayModel(FunctionModel):
    """Stateful local model that replays decisions only after exact transitions."""

    fixture: ReplayFixture
    _turn_index: int

    def __init__(self, fixture: ReplayFixture) -> None:
        self.fixture = fixture
        self._turn_index = 0
        super().__init__(self._respond, model_name=f"replay:{fixture.scenario}")

    @property
    def consumed_turns(self) -> int:
        """Return the number of fixture turns successfully consumed."""
        return self._turn_index

    def assert_complete(self) -> None:
        """Fail if execution stopped before consuming the entire fixture."""
        if self._turn_index != len(self.fixture.turns):
            raise ReplayMismatchError(
                "replay stopped after "
                f"{self._turn_index} of {len(self.fixture.turns)} turns"
            )

    def _respond(
        self,
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        if self._turn_index >= len(self.fixture.turns):
            raise ReplayMismatchError("agent requested more turns than recorded")
        turn_number = self._turn_index + 1
        turn = self.fixture.turns[self._turn_index]
        kind, tool_results, retry_text = _request_transition(messages)
        if kind is not turn.expect.kind:
            raise ReplayMismatchError(
                f"turn {turn_number} expected {turn.expect.kind.value}, got {kind.value}"
            )
        if tool_results != turn.expect.tool_results:
            raise ReplayMismatchError(
                f"turn {turn_number} tool-result identities do not match"
            )
        if (
            turn.expect.retry_contains is not None
            and turn.expect.retry_contains not in retry_text
        ):
            raise ReplayMismatchError(
                f"turn {turn_number} validation-retry reason does not match"
            )
        if kind is ReplayRequestKind.INITIAL:
            user_text = "\n".join(
                str(part.content)
                for part in messages[-1].parts
                if isinstance(part, UserPromptPart)
            )
            if f'"case_id":"{self.fixture.case_id}"' not in user_text:
                raise ReplayMismatchError(
                    f"turn {turn_number} public case identifier does not match"
                )

        response = turn.respond
        parts: list[ToolCallPart] = []
        if response.output is not None:
            if len(info.output_tools) != 1:
                raise ReplayMismatchError("replay requires exactly one output tool")
            parts.append(
                ToolCallPart(
                    info.output_tools[0].name,
                    response.output.model_dump(mode="json"),
                    tool_call_id=f"replay-output-{turn_number}",
                )
            )
        else:
            parts.extend(
                ToolCallPart(
                    call.tool_name,
                    call.arguments,
                    tool_call_id=call.tool_call_id,
                )
                for call in response.tool_calls
            )
        usage = RequestUsage()
        if response.usage is not None:
            usage = RequestUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        self._turn_index += 1
        return ModelResponse(parts=parts, usage=usage)


def build_replay_model(fixture: ReplayFixture) -> ReplayModel:
    """Construct an isolated replay session for one validated fixture."""
    return ReplayModel(fixture)
