"""Pydantic AI Investigator with typed dependencies and bounded evidence tools."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from experiment_failure_investigator.agent.contracts import (
    AgentStrictModel,
    InvestigatorOutput,
    validate_output_against_baseline,
)
from experiment_failure_investigator.agent.evidence import (
    HARD_MAX_EVIDENCE_RECORDS,
    DiagnosticResultList,
    EvidencePage,
    ResolvedEvidence,
    inspect_diagnostic_result as inspect_frozen_result,
    list_diagnostic_results as list_frozen_results,
    resolve_evidence as resolve_frozen_evidence,
)
from experiment_failure_investigator.analysis.contracts import InvestigatorCase
from experiment_failure_investigator.reporting.baseline import BaselineReport

DEFAULT_MAX_RESOLVED_EVIDENCE_IDS = 10


class EvidenceToolErrorCode(StrEnum):
    """Stable model-visible evidence-access failure categories."""

    INVALID_REQUEST = "invalid_request"


class EvidenceToolError(AgentStrictModel):
    """Sanitized failure returned for a semantically invalid tool request."""

    tool_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    code: EvidenceToolErrorCode = EvidenceToolErrorCode.INVALID_REQUEST
    detail: str = Field(min_length=1)


CatalogToolResponse = DiagnosticResultList | EvidenceToolError
InspectToolResponse = EvidencePage | EvidenceToolError
ResolveToolResponse = ResolvedEvidence | EvidenceToolError


@dataclass(frozen=True)
class InvestigatorDependencies:
    """Public case state and frozen evidence available during one agent run."""

    case: InvestigatorCase
    baseline: BaselineReport
    max_evidence_records: int = 25

    def __post_init__(self) -> None:
        """Reject mismatched state and evidence limits before model execution."""
        if self.case.case_id != self.baseline.case_id:
            raise ValueError("investigator case ID does not match the baseline")
        if self.case.design.capabilities != self.baseline.capabilities:
            raise ValueError("investigator capabilities do not match the baseline")
        if not 1 <= self.max_evidence_records <= HARD_MAX_EVIDENCE_RECORDS:
            raise ValueError(
                "maximum evidence records must be between one and "
                f"{HARD_MAX_EVIDENCE_RECORDS}"
            )


def _tool_error(tool_name: str, error: ValueError) -> EvidenceToolError:
    """Convert a safe evidence-boundary exception into a typed tool response."""
    return EvidenceToolError(tool_name=tool_name, detail=str(error))


def build_investigator_agent(
    model: Model,
    *,
    system_prompt: str,
    output_validation_retries: int = 1,
    tool_validation_retries: int = 1,
) -> Agent[InvestigatorDependencies, InvestigatorOutput]:
    """Construct one Investigator exposing only bounded evidence retrieval."""
    if not system_prompt.strip():
        raise ValueError("system prompt must not be blank")
    if not 0 <= output_validation_retries <= 3:
        raise ValueError("output validation retries must be between zero and three")
    if not 0 <= tool_validation_retries <= 3:
        raise ValueError("tool validation retries must be between zero and three")

    agent = Agent(
        model,
        output_type=InvestigatorOutput,
        system_prompt=system_prompt,
        deps_type=InvestigatorDependencies,
        name="experiment_failure_investigator",
        retries={
            "output": output_validation_retries,
            "tools": tool_validation_retries,
        },
    )

    @agent.tool(
        name="list_diagnostic_results",
        description=(
            "List the allowlisted deterministic diagnostic families, statuses, "
            "scopes, warnings, limitations, evidence counts, and metric prefixes."
        ),
    )
    def list_diagnostic_results(
        context: RunContext[InvestigatorDependencies],
    ) -> CatalogToolResponse:
        """List the frozen diagnostic catalog without evidence values."""
        try:
            return list_frozen_results(context.deps.baseline)
        except ValueError as error:
            return _tool_error("list_diagnostic_results", error)

    @agent.tool(
        name="inspect_diagnostic_result",
        description=(
            "Inspect one filtered, paginated set of evidence records from an "
            "allowlisted deterministic diagnostic already present in the baseline."
        ),
    )
    def inspect_diagnostic_result(
        context: RunContext[InvestigatorDependencies],
        tool_name: str,
        metric_prefix: str | None = None,
        plate_id: str | None = None,
        treatment: str | None = None,
        offset: int = 0,
        limit: int = 10,
    ) -> InspectToolResponse:
        """Return a bounded evidence page selected by public scope filters."""
        try:
            return inspect_frozen_result(
                context.deps.baseline,
                tool_name,
                metric_prefix=metric_prefix,
                plate_id=plate_id,
                treatment=treatment,
                offset=offset,
                limit=limit,
                maximum_page_size=context.deps.max_evidence_records,
            )
        except ValueError as error:
            return _tool_error("inspect_diagnostic_result", error)

    @agent.tool(
        name="resolve_evidence",
        description=(
            "Resolve a small exact list of existing evidence IDs before using "
            "them in a hypothesis or recommendation."
        ),
    )
    def resolve_evidence(
        context: RunContext[InvestigatorDependencies],
        evidence_ids: tuple[str, ...],
    ) -> ResolveToolResponse:
        """Resolve exact evidence records within a fixed request bound."""
        try:
            return resolve_frozen_evidence(
                context.deps.baseline,
                evidence_ids,
                maximum_ids=min(
                    DEFAULT_MAX_RESOLVED_EVIDENCE_IDS,
                    context.deps.max_evidence_records,
                ),
            )
        except ValueError as error:
            return _tool_error("resolve_evidence", error)

    @agent.output_validator
    def validate_investigator_output(
        context: RunContext[InvestigatorDependencies],
        output: InvestigatorOutput,
    ) -> InvestigatorOutput:
        """Ask the model to correct case or evidence references that do not exist."""
        try:
            return validate_output_against_baseline(output, context.deps.baseline)
        except ValueError as error:
            raise ModelRetry(str(error)) from error

    return agent
