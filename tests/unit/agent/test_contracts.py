"""Tests for typed investigator hypotheses and evidence-graph validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from experiment_failure_investigator.agent.contracts import (
    ConfidenceCategory,
    FailureHypothesis,
    InvestigatorOutput,
    validate_output_against_baseline,
)
from experiment_failure_investigator.analysis.contracts import (
    DesignCapabilities,
    PublicArtifactHashes,
)
from experiment_failure_investigator.analysis.results import (
    EvidenceScope,
    ScientificProvenance,
    ScientificToolResult,
    ToolStatus,
    build_evidence_record,
)
from experiment_failure_investigator.reporting.baseline import (
    BaselineReport,
    HeuristicConfiguration,
)

CASE_ID = "case_0123456789abcdef"
ZERO_HASH = "0" * 64
HASHES = PublicArtifactHashes(
    measurements=ZERO_HASH,
    plate_map=ZERO_HASH,
    metadata=ZERO_HASH,
    protocol=ZERO_HASH,
    problem_statement=ZERO_HASH,
)
SCOPE = EvidenceScope(case_id=CASE_ID, plate_ids=("plate_alpha",))
PROVENANCE = ScientificProvenance(case_id=CASE_ID, public_artifact_hashes=HASHES)


def _baseline() -> BaselineReport:
    records = tuple(
        build_evidence_record(
            tool_name="summarize_controls",
            tool_version="1.0.0",
            parameters={},
            metric_name=metric,
            value=value,
            scope=SCOPE,
            description=description,
        )
        for metric, value, description in (
            ("controls.separation", 0.25, "Observed control separation."),
            ("controls.negative_mean", 1.0, "Observed negative-control mean."),
        )
    )
    result = ScientificToolResult(
        status=ToolStatus.SUCCESS,
        tool_name="summarize_controls",
        tool_version="1.0.0",
        scope=SCOPE,
        evidence=records,
        provenance=PROVENANCE,
    )
    return BaselineReport(
        heuristic_configuration=HeuristicConfiguration(),
        case_id=CASE_ID,
        capabilities=DesignCapabilities(
            has_negative_controls=True,
            has_positive_controls=True,
            has_replicates=True,
            has_dose_series=False,
            has_multiple_plates=False,
            plates_are_complete_design_replicates=None,
            supports_cross_plate_analysis=False,
        ),
        tool_results=(result,),
        findings=(),
        evidence_index=records,
        limitations=("Threshold interpretation remains assay-specific.",),
        suggested_followups=(),
    )


def _hypothesis(
    rank: int,
    evidence_id: str,
    *,
    confidence: ConfidenceCategory = ConfidenceCategory.MODERATE,
) -> FailureHypothesis:
    return FailureHypothesis(
        rank=rank,
        name="Control preparation variation",
        proposed_mechanism="Control preparation may have reduced assay separation.",
        confidence=confidence,
        supporting_evidence_ids=(evidence_id,),
        missing_evidence=("Independent control preparation records",),
        alternative_explanations=("Plate handling variation",),
        falsification_check="Repeat the controls using an independent preparation.",
    )


def _output() -> InvestigatorOutput:
    evidence = _baseline().evidence_index
    return InvestigatorOutput(
        case_id=CASE_ID,
        hypotheses=(
            _hypothesis(2, evidence[1].evidence_id, confidence=ConfidenceCategory.LOW),
            _hypothesis(1, evidence[0].evidence_id),
        ),
        overall_assessment="The control behavior warrants review before interpretation.",
        remaining_uncertainty=("The physical origin of the pattern is unavailable.",),
        recommended_next_check="Repeat controls from an independent preparation.",
        recommendation_evidence_ids=(evidence[0].evidence_id,),
        limitations=("The evidence does not identify a physical cause.",),
    )


def test_valid_output_is_ranked_and_matches_baseline() -> None:
    output = _output()

    assert [hypothesis.rank for hypothesis in output.hypotheses] == [1, 2]
    assert validate_output_against_baseline(output, _baseline()) is output


def test_output_rejects_wrong_case_or_unknown_evidence() -> None:
    output = _output()
    wrong_case = output.model_copy(update={"case_id": "case_fedcba9876543210"})
    unknown = output.model_copy(
        update={"recommendation_evidence_ids": ("ev_" + "f" * 20,)}
    )

    with pytest.raises(ValueError, match="case ID"):
        validate_output_against_baseline(wrong_case, _baseline())
    with pytest.raises(ValueError, match="unknown evidence"):
        validate_output_against_baseline(unknown, _baseline())


def test_output_rejects_noncontiguous_or_duplicate_ranks() -> None:
    baseline = _baseline()
    evidence_id = baseline.evidence_index[0].evidence_id
    base = _output().model_dump(mode="python")
    base["hypotheses"] = [
        _hypothesis(1, evidence_id).model_dump(mode="python"),
        _hypothesis(3, evidence_id).model_dump(mode="python"),
    ]

    with pytest.raises(ValidationError, match="contiguous"):
        InvestigatorOutput.model_validate(base)


def test_hypothesis_rejects_overlapping_evidence_roles() -> None:
    evidence_id = _baseline().evidence_index[0].evidence_id
    values = _hypothesis(1, evidence_id).model_dump(mode="python")
    values["contradicting_evidence_ids"] = (evidence_id,)

    with pytest.raises(ValidationError, match="must not overlap"):
        FailureHypothesis.model_validate(values)


def test_determinate_hypothesis_requires_support() -> None:
    values = _hypothesis(1, _baseline().evidence_index[0].evidence_id).model_dump(
        mode="python"
    )
    values["supporting_evidence_ids"] = ()

    with pytest.raises(ValidationError, match="requires supporting evidence"):
        FailureHypothesis.model_validate(values)


def test_indeterminate_hypothesis_may_have_no_support() -> None:
    values = _hypothesis(1, _baseline().evidence_index[0].evidence_id).model_dump(
        mode="python"
    )
    values["confidence"] = ConfidenceCategory.INDETERMINATE
    values["supporting_evidence_ids"] = ()

    assert FailureHypothesis.model_validate(values).supporting_evidence_ids == ()


def test_all_narrative_fields_reject_uncited_numbers() -> None:
    values = _output().model_dump(mode="python")
    values["remaining_uncertainty"] = ("Plate 2 timing is unavailable.",)

    with pytest.raises(ValidationError, match="must not embed numbers"):
        InvestigatorOutput.model_validate(values)


def test_contracts_reject_unknown_fields() -> None:
    values = _output().model_dump(mode="python")
    values["planted_failure_mode"] = "hidden"

    with pytest.raises(ValidationError, match="Extra inputs"):
        InvestigatorOutput.model_validate(values)
