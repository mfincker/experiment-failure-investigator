"""Strict contracts for model-generated scientific investigation output."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from experiment_failure_investigator.analysis.results import EvidenceId
from experiment_failure_investigator.reporting.baseline import BaselineReport

INVESTIGATOR_OUTPUT_VERSION = "1.0.0"
RunId = Annotated[str, StringConstraints(pattern=r"^run_[0-9a-f]{20}$")]


class AgentStrictModel(BaseModel):
    """Base for stable agent contracts that reject unknown fields."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ConfidenceCategory(StrEnum):
    """Qualitative confidence without pretending the model is calibrated."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INDETERMINATE = "indeterminate"


def _canonical_text_items(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(_reject_uncited_numbers(value) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError("text entries must be unique")
    return normalized


def _reject_uncited_numbers(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("narrative text must not be blank")
    if re.search(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?", normalized):
        raise ValueError(
            "narrative text must not embed numbers; cite evidence records"
        )
    return normalized


class FailureHypothesis(AgentStrictModel):
    """One provisional mechanism grounded in deterministic evidence."""

    rank: int = Field(ge=1, le=4)
    name: str = Field(min_length=1)
    proposed_mechanism: str = Field(min_length=1)
    confidence: ConfidenceCategory
    supporting_evidence_ids: tuple[EvidenceId, ...] = ()
    contradicting_evidence_ids: tuple[EvidenceId, ...] = ()
    missing_evidence: tuple[str, ...] = Field(min_length=1)
    alternative_explanations: tuple[str, ...] = Field(min_length=1)
    falsification_check: str = Field(min_length=1)

    @field_validator("name", "proposed_mechanism", "falsification_check")
    @classmethod
    def validate_narrative(cls, value: str) -> str:
        return _reject_uncited_numbers(value)

    @field_validator("missing_evidence", "alternative_explanations")
    @classmethod
    def validate_text_items(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_text_items(values)

    @field_validator("supporting_evidence_ids", "contradicting_evidence_ids")
    @classmethod
    def canonicalize_evidence_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("evidence IDs must be unique within a citation role")
        return tuple(sorted(values))

    @model_validator(mode="after")
    def validate_evidence_roles(self) -> FailureHypothesis:
        overlap = set(self.supporting_evidence_ids) & set(
            self.contradicting_evidence_ids
        )
        if overlap:
            raise ValueError("supporting and contradicting evidence must not overlap")
        if (
            self.confidence is not ConfidenceCategory.INDETERMINATE
            and not self.supporting_evidence_ids
        ):
            raise ValueError("a determinate confidence requires supporting evidence")
        return self


class InvestigatorOutput(AgentStrictModel):
    """Validated structured result returned by the Week 3 investigator."""

    schema_version: str = Field(
        default=INVESTIGATOR_OUTPUT_VERSION,
        pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$",
    )
    case_id: str = Field(pattern=r"^case_[0-9a-f]{16}$")
    hypotheses: tuple[FailureHypothesis, ...] = Field(min_length=2, max_length=4)
    overall_assessment: str = Field(min_length=1)
    remaining_uncertainty: tuple[str, ...] = Field(min_length=1)
    recommended_next_check: str = Field(min_length=1)
    recommendation_evidence_ids: tuple[EvidenceId, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)

    @field_validator("overall_assessment", "recommended_next_check")
    @classmethod
    def validate_narrative(cls, value: str) -> str:
        return _reject_uncited_numbers(value)

    @field_validator("remaining_uncertainty", "limitations")
    @classmethod
    def validate_text_items(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_text_items(values)

    @field_validator("recommendation_evidence_ids")
    @classmethod
    def canonicalize_recommendation_evidence(
        cls, values: tuple[str, ...]
    ) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("recommendation evidence IDs must be unique")
        return tuple(sorted(values))

    @field_validator("hypotheses")
    @classmethod
    def canonicalize_hypotheses(
        cls, values: tuple[FailureHypothesis, ...]
    ) -> tuple[FailureHypothesis, ...]:
        ordered = tuple(sorted(values, key=lambda hypothesis: hypothesis.rank))
        ranks = tuple(hypothesis.rank for hypothesis in ordered)
        if ranks != tuple(range(1, len(ordered) + 1)):
            raise ValueError("hypothesis ranks must be unique and contiguous from one")
        return ordered

    @model_validator(mode="after")
    def validate_leading_hypothesis(self) -> InvestigatorOutput:
        leader = self.hypotheses[0]
        if (
            leader.confidence is not ConfidenceCategory.INDETERMINATE
            and not leader.supporting_evidence_ids
        ):
            raise ValueError(
                "the leading hypothesis requires support or indeterminate confidence"
            )
        return self


def validate_output_against_baseline(
    output: InvestigatorOutput,
    baseline: BaselineReport,
) -> InvestigatorOutput:
    """Verify every model citation against one frozen public-evidence graph."""
    if output.case_id != baseline.case_id:
        raise ValueError("investigator output case ID does not match the baseline")
    available = {record.evidence_id for record in baseline.evidence_index}
    cited = set(output.recommendation_evidence_ids)
    for hypothesis in output.hypotheses:
        cited.update(hypothesis.supporting_evidence_ids)
        cited.update(hypothesis.contradicting_evidence_ids)
    unknown = sorted(cited - available)
    if unknown:
        raise ValueError(f"investigator output cites unknown evidence IDs: {unknown}")
    return output
