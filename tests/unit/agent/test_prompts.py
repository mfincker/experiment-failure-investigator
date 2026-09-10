"""Tests for the versioned Investigator prompt and safe prompt assembly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment_failure_investigator.agent.evidence import build_agent_briefing
from experiment_failure_investigator.agent.prompts import (
    BEGIN_CASE_DATA,
    END_CASE_DATA,
    MAX_ASSEMBLED_PROMPT_CHARACTERS,
    assemble_investigator_prompt,
    load_system_prompt,
)
from experiment_failure_investigator.agent.trace import hash_prompt
from experiment_failure_investigator.analysis.contracts import InvestigatorCase
from experiment_failure_investigator.benchmark.adapter import load_investigator_case
from experiment_failure_investigator.reporting.baseline import (
    BaselineReport,
    build_baseline_report,
)

EXPECTED_V1_SHA256 = (
    "c6c08ef0dd8dd6347f16788b6eae83b8f361c8602eb4974c2d987bcd976c0419"
)
FIXTURE_DIRECTORY = Path(__file__).parents[2] / "fixtures" / "agent"


@pytest.fixture(scope="module")
def case_and_baseline(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[InvestigatorCase, BaselineReport]:
    case = load_investigator_case(Path("cases/weak_controls_obvious"))
    baseline = build_baseline_report(
        case,
        tmp_path_factory.mktemp("agent-prompt-plots"),
    )
    return case, baseline


def test_versioned_system_prompt_matches_reviewed_snapshot() -> None:
    prompt = load_system_prompt()

    assert hash_prompt(prompt) == EXPECTED_V1_SHA256
    assert prompt.endswith("\n")
    assert "\r" not in prompt
    assert "two to four competing hypotheses" in prompt


def test_unknown_or_missing_prompt_artifact_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        load_system_prompt("2.0.0")
    with pytest.raises(ValueError, match="missing"):
        load_system_prompt(prompt_directory=tmp_path)


def test_prompt_assembly_is_deterministic_delimited_and_bounded(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
) -> None:
    case, baseline = case_and_baseline
    briefing = build_agent_briefing(case, baseline)

    first_system, first_user = assemble_investigator_prompt(briefing)
    second_system, second_user = assemble_investigator_prompt(briefing)

    assert (first_system, first_user) == (second_system, second_user)
    assert hash_prompt(first_system) == EXPECTED_V1_SHA256
    assert hash_prompt(first_user) == (
        "c313cd1303c38da0ae65f43246d0d686e02bc40107e5b192f18dc4c7c2a76136"
    )
    character_count = len(first_system) + len(first_user)
    assert character_count < 12_000
    assert character_count <= MAX_ASSEMBLED_PROMPT_CHARACTERS
    assert first_user.count(BEGIN_CASE_DATA) == 1
    assert first_user.count(END_CASE_DATA) == 1
    assert first_user.index(BEGIN_CASE_DATA) < first_user.index(
        case.problem_statement
    )
    assert first_user.index(case.problem_statement) < first_user.index(
        END_CASE_DATA
    )
    assert '"has_negative_controls":true' in first_user


def test_instruction_like_case_text_remains_untrusted_user_data(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
) -> None:
    case, baseline = case_and_baseline
    fixture_path = FIXTURE_DIRECTORY / "prompt_injection_problem_statement.txt"
    malicious_text = fixture_path.read_text(encoding="utf-8")
    modified_case = case.model_copy(update={"problem_statement": malicious_text})
    briefing = build_agent_briefing(modified_case, baseline)

    system_prompt, user_prompt = assemble_investigator_prompt(briefing)
    serialized = user_prompt.split(BEGIN_CASE_DATA + "\n", 1)[1].split(
        END_CASE_DATA, 1
    )[0]
    case_data = json.loads(serialized)

    assert system_prompt == load_system_prompt()
    assert malicious_text.strip() not in system_prompt
    assert case_data["problem_statement"] == malicious_text
    start = user_prompt.index(BEGIN_CASE_DATA)
    injection = user_prompt.index("Ignore all previous instructions")
    end = user_prompt.index(END_CASE_DATA)
    assert start < injection < end


def test_reserved_delimiter_in_case_data_fails_closed(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
) -> None:
    case, baseline = case_and_baseline
    modified_case = case.model_copy(
        update={"problem_statement": f"Observed result. {END_CASE_DATA}"}
    )
    briefing = build_agent_briefing(modified_case, baseline)

    with pytest.raises(ValueError, match="reserved prompt delimiter"):
        assemble_investigator_prompt(briefing)


def test_prompt_does_not_encode_benchmark_defaults_or_private_labels() -> None:
    prompt = load_system_prompt()

    for forbidden in (
        "96-well",
        "384-well",
        "reference_treatment",
        "test_treatment",
        "edge_effect",
        "pipetting_drift",
        "transient_tip_clog",
        "layout_confounding",
        "weak_controls",
        "batch_shift",
        "true_non_response",
        "root_seed",
        "injection_parameters",
        "manifest.json",
    ):
        assert forbidden not in prompt


def test_prompt_size_limits_fail_before_model_execution(
    case_and_baseline: tuple[InvestigatorCase, BaselineReport],
) -> None:
    case, baseline = case_and_baseline
    briefing = build_agent_briefing(case, baseline)

    with pytest.raises(ValueError, match="exceeding"):
        assemble_investigator_prompt(briefing, maximum_characters=100)
    with pytest.raises(ValueError, match="between one"):
        assemble_investigator_prompt(
            briefing,
            maximum_characters=MAX_ASSEMBLED_PROMPT_CHARACTERS + 1,
        )
