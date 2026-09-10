"""Tests for compact briefings and bounded deterministic evidence access."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

import pytest

from experiment_failure_investigator.agent.evidence import (
    ALLOWED_DIAGNOSTIC_TOOLS,
    JsonObject,
    build_agent_briefing,
    inspect_diagnostic_result,
    list_diagnostic_results,
    resolve_evidence,
)
from experiment_failure_investigator.analysis.contracts import InvestigatorCase
from experiment_failure_investigator.analysis.results import canonical_json
from experiment_failure_investigator.benchmark.adapter import load_investigator_case
from experiment_failure_investigator.reporting.baseline import (
    BaselineReport,
    build_baseline_report,
)


@pytest.fixture(scope="module")
def case_and_baseline(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[InvestigatorCase, BaselineReport]:
    case = load_investigator_case(Path("cases/weak_controls_obvious"))
    baseline = build_baseline_report(
        case,
        tmp_path_factory.mktemp("agent-evidence-plots"),
    )
    return case, baseline


def test_briefing_contains_compact_public_context_without_raw_tables(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
) -> None:
    case, baseline = case_and_baseline

    briefing = build_agent_briefing(case, baseline)
    payload = canonical_json(briefing)

    assert briefing["case_id"] == case.case_id
    assert briefing["capabilities"] == case.design.capabilities.model_dump(mode="json")
    assert briefing["problem_statement"] == case.problem_statement
    diagnostic_results = cast(list[JsonObject], briefing["diagnostic_results"])
    assert {cast(str, entry["tool_name"]) for entry in diagnostic_results} == set(
        ALLOWED_DIAGNOSTIC_TOOLS
    )
    assert len(payload) < len(canonical_json(baseline)) * 0.1
    for private_or_raw_field in (
        '"measurements"',
        '"plate_map"',
        '"manifest"',
        '"failure_mode"',
        '"expected_evidence"',
        '"injection_parameters"',
        '"root_seed"',
        '"simulated_traversal"',
        '"latent_signal"',
        '"path"',
    ):
        assert private_or_raw_field not in payload


def test_catalog_is_canonical_and_describes_statuses_scopes_and_metrics(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
) -> None:
    _, baseline = case_and_baseline

    catalog = list_diagnostic_results(baseline)
    assert hashlib.sha256(canonical_json(catalog).encode("utf-8")).hexdigest() == (
        "e07d4e4e42161fc7425d7403c3b8e3b90df31169c62bba6c52051408d36c5d4f"
    )
    results = cast(list[JsonObject], catalog["results"])
    entries = {cast(str, entry["tool_name"]): entry for entry in results}

    assert tuple(cast(str, entry["tool_name"]) for entry in results) == tuple(
        sorted(ALLOWED_DIAGNOSTIC_TOOLS)
    )
    assert cast(list[str], entries["compare_batches"]["statuses"]) == [
        "not_applicable"
    ]
    assert (
        cast(int, entries["calculate_replicate_variability"]["evidence_count"]) > 25
    )
    metric_prefixes = cast(
        list[str],
        entries["calculate_replicate_variability"]["metric_prefixes"],
    )
    assert "replicates" in metric_prefixes
    assert cast(
        list[str], entries["generate_diagnostic_plot"]["attachment_names"]
    ) == [
        "condition_residual",
        "control_distribution",
        "dose_response",
        "raw_signal",
    ]


def test_briefing_is_invariant_to_baseline_collection_order(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
) -> None:
    case, baseline = case_and_baseline
    reordered = baseline.model_copy(
        update={
            "tool_results": tuple(reversed(baseline.tool_results)),
            "findings": tuple(reversed(baseline.findings)),
            "evidence_index": tuple(reversed(baseline.evidence_index)),
        }
    )

    assert build_agent_briefing(case, reordered) == build_agent_briefing(
        case, baseline
    )


def test_paginated_evidence_has_no_gaps_or_duplicates(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
) -> None:
    _, baseline = case_and_baseline
    first = inspect_diagnostic_result(
        baseline,
        "calculate_replicate_variability",
        offset=0,
        limit=7,
    )
    second = inspect_diagnostic_result(
        baseline,
        "calculate_replicate_variability",
        offset=7,
        limit=7,
    )
    combined_ids = tuple(
        record.evidence_id for record in (*first.evidence, *second.evidence)
    )
    expected = inspect_diagnostic_result(
        baseline,
        "calculate_replicate_variability",
        offset=0,
        limit=14,
    )

    assert combined_ids == tuple(record.evidence_id for record in expected.evidence)
    assert len(combined_ids) == len(set(combined_ids))
    assert first.has_more is True
    assert first.total_tool_evidence_count == first.matched_count
    assert first.returned_count == 7


def test_evidence_filters_are_deterministic_and_records_exist_in_baseline(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
) -> None:
    _, baseline = case_and_baseline

    page = inspect_diagnostic_result(
        baseline,
        "calculate_replicate_variability",
        metric_prefix="replicates.mean",
        plate_id="plate_01",
        treatment="test_treatment",
        limit=10,
    )
    baseline_by_id = {
        record.evidence_id: record for record in baseline.evidence_index
    }

    assert page.evidence
    assert page.matched_count >= page.returned_count
    assert all(record.metric_name == "replicates.mean" for record in page.evidence)
    assert all("plate_01" in record.scope.plate_ids for record in page.evidence)
    assert all(
        "test_treatment" in record.scope.treatments for record in page.evidence
    )
    assert all(
        baseline_by_id[record.evidence_id] == record for record in page.evidence
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tool_name": "read_arbitrary_file"}, "allowlisted"),
        (
            {"tool_name": "summarize_controls", "plate_id": "unknown_plate"},
            "plate scope",
        ),
        (
            {"tool_name": "summarize_controls", "treatment": "unknown_drug"},
            "treatment scope",
        ),
        (
            {"tool_name": "summarize_controls", "metric_prefix": "private.truth"},
            "metric prefix",
        ),
        ({"tool_name": "summarize_controls", "offset": -1}, "offset"),
        ({"tool_name": "summarize_controls", "limit": 26}, "limit"),
    ],
)
def test_invalid_tool_filters_and_limits_fail_closed(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
    kwargs: dict[str, object],
    message: str,
) -> None:
    _, baseline = case_and_baseline

    with pytest.raises(ValueError, match=message):
        inspect_diagnostic_result(baseline, **kwargs)  # type: ignore[arg-type]


def test_exact_evidence_resolution_is_bounded_canonical_and_public(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
) -> None:
    _, baseline = case_and_baseline
    requested = tuple(
        record.evidence_id for record in reversed(baseline.evidence_index[:3])
    )

    resolved = resolve_evidence(baseline, requested)

    assert hashlib.sha256(canonical_json(resolved).encode("utf-8")).hexdigest() == (
        "4d1a63dda408769a0e3e8d77fde5e8cb570fee9902b93e637df0e8a95ef43a77"
    )
    records = cast(list[JsonObject], resolved["evidence"])
    assert tuple(
        cast(str, record["evidence_id"]) for record in records
    ) == tuple(sorted(requested))
    baseline_ids = {record.evidence_id for record in baseline.evidence_index}
    assert all(cast(str, record["evidence_id"]) in baseline_ids for record in records)
    assert '"path"' not in canonical_json(resolved)


def test_exact_evidence_resolution_rejects_invalid_requests(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
) -> None:
    _, baseline = case_and_baseline
    known = baseline.evidence_index[0].evidence_id

    with pytest.raises(ValueError, match="at least one"):
        resolve_evidence(baseline, ())
    with pytest.raises(ValueError, match="unique"):
        resolve_evidence(baseline, (known, known))
    with pytest.raises(ValueError, match="invalid"):
        resolve_evidence(baseline, ("not-an-evidence-id",))
    with pytest.raises(ValueError, match="unknown"):
        resolve_evidence(baseline, ("ev_" + "f" * 20,))
    with pytest.raises(ValueError, match="between one"):
        resolve_evidence(baseline, (known,), maximum_ids=0)
    with pytest.raises(ValueError, match="at most"):
        resolve_evidence(
            baseline,
            (known, baseline.evidence_index[1].evidence_id),
            maximum_ids=1,
        )


def test_briefing_rejects_a_mismatched_case(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
) -> None:
    case, baseline = case_and_baseline
    mismatched = case.model_copy(update={"case_id": "case_0123456789abcdef"})

    with pytest.raises(ValueError, match="case ID"):
        build_agent_briefing(mismatched, baseline)
