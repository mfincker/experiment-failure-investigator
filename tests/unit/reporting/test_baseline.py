"""Tests for deterministic evidence-linked baseline reports."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from experiment_failure_investigator.analysis.contracts import PublicArtifactHashes
from experiment_failure_investigator.analysis.design import build_investigator_case
from experiment_failure_investigator.analysis.results import canonical_json
from experiment_failure_investigator.benchmark.adapter import load_investigator_case
from experiment_failure_investigator.benchmark.layouts import build_balanced_layout
from experiment_failure_investigator.benchmark.models import (
    CaseMetadata,
    CaseVariant,
    FailureMode,
    GeneratorConfig,
    PlateMetadata,
)
from experiment_failure_investigator.benchmark.signals import generate_clean_assay
from experiment_failure_investigator.reporting.baseline import (
    BaselineFinding,
    BaselineReport,
    FindingCategory,
    build_baseline_report,
    render_baseline_markdown,
    write_baseline_report,
)

ZERO_HASH = "0" * 64
GOLDEN_DIRECTORY = Path(__file__).parents[2] / "golden"
HASHES = PublicArtifactHashes(
    measurements=ZERO_HASH,
    plate_map=ZERO_HASH,
    metadata=ZERO_HASH,
    protocol=ZERO_HASH,
    problem_statement=ZERO_HASH,
)


def _report(case_name: str, output: Path) -> BaselineReport:
    return build_baseline_report(
        load_investigator_case(Path("cases") / case_name),
        output,
    )


def _healthy_report(output: Path) -> BaselineReport:
    config = GeneratorConfig(
        case_id="healthy_fixture",
        case_variant=CaseVariant.OBVIOUS,
        failure_mode=FailureMode.EDGE_EFFECT,
        root_seed=0,
        noise_sd=0.03,
    )
    clean = generate_clean_assay(build_balanced_layout(config), config)
    metadata = CaseMetadata(
        case_id=config.case_id,
        assay_type=config.assay_type,
        signal_direction=config.signal_direction,
        layout_fingerprint=ZERO_HASH,
        plates=[PlateMetadata(plate_id="plate_01", batch_id="batch_01")],
    )
    case = build_investigator_case(
        measurements=clean.measurements,
        plate_map=clean.plate_map,
        metadata=metadata,
        protocol="Synthetic endpoint assay protocol.",
        problem_statement="Perform deterministic quality control.",
        public_artifact_hashes=HASHES,
    )
    return build_baseline_report(case, output)


def test_report_is_public_evidence_linked_and_writes_both_formats(
    tmp_path: Path,
) -> None:
    report = _report("edge_effect_obvious", tmp_path / "plots")
    json_path, markdown_path = write_baseline_report(report, tmp_path / "report")

    evidence_ids = {record.evidence_id for record in report.evidence_index}
    assert evidence_ids
    assert all(
        set(finding.evidence_ids) <= evidence_ids for finding in report.findings
    )
    assert json_path.read_text() == canonical_json(report)
    markdown = markdown_path.read_text()
    assert markdown == render_baseline_markdown(report)
    cited_ids = {
        evidence_id
        for finding in report.findings
        for evidence_id in finding.evidence_ids
    } | {
        evidence_id
        for result in report.tool_results
        for warning in result.warnings
        for evidence_id in warning.evidence_ids
    }
    uncited = next(
        record
        for record in report.evidence_index
        if record.evidence_id not in cited_ids
    )
    assert uncited.evidence_id not in markdown
    assert "complete evidence index" in markdown
    assert all(
        not Path(attachment.path).is_absolute()
        for result in report.tool_results
        for attachment in result.attachments
    )
    assert {
        attachment.name
        for result in report.tool_results
        for attachment in result.attachments
    } == {
        "raw_signal",
        "condition_residual",
        "control_distribution",
        "dose_response",
    }
    serialized = canonical_json(report)
    for private_name in (
        "planted_failure_mode",
        "failure_label",
        "injection_parameters",
        "child_seeds",
        "root_seed",
        "expected_signal",
        "injected_effect",
    ):
        assert private_name not in serialized


def test_report_is_identical_across_output_directories(tmp_path: Path) -> None:
    first = _report("true_non_response_obvious", tmp_path / "first")
    second = _report("true_non_response_obvious", tmp_path / "second")

    assert first == second
    assert canonical_json(first) == canonical_json(second)


def test_selected_markdown_report_matches_golden_file(tmp_path: Path) -> None:
    report = _report("weak_controls_obvious", tmp_path / "plots")

    assert render_baseline_markdown(report) == (
        GOLDEN_DIRECTORY / "weak_controls_obvious.report.md"
    ).read_text(encoding="utf-8")


def test_fixed_clean_fixture_has_no_baseline_findings(tmp_path: Path) -> None:
    report = _healthy_report(tmp_path / "healthy")

    assert report.findings == ()
    assert "No configured synthetic-benchmark heuristic" in render_baseline_markdown(
        report
    )


@pytest.mark.parametrize(
    ("case_name", "category"),
    [
        ("weak_controls_obvious", FindingCategory.CONTROL_SEPARATION),
        ("edge_effect_noisy", FindingCategory.EDGE_ASSOCIATION),
        ("pipetting_drift_noisy", FindingCategory.POSITION_TREND),
        ("transient_tip_clog_noisy", FindingCategory.LOCALIZED_PATTERN),
        ("layout_confounding_noisy", FindingCategory.LAYOUT_IDENTIFIABILITY),
        ("batch_shift_noisy", FindingCategory.PLATE_RESPONSE),
        ("true_non_response_noisy", FindingCategory.TREATMENT_RESPONSE),
    ],
)
def test_neutral_heuristics_cover_each_benchmark_evidence_family(
    tmp_path: Path,
    case_name: str,
    category: FindingCategory,
) -> None:
    report = _report(case_name, tmp_path / case_name)

    assert category in {finding.category for finding in report.findings}


def test_finding_narrative_cannot_embed_an_uncited_number() -> None:
    with pytest.raises(ValidationError, match="must not embed numbers"):
        BaselineFinding(
            finding_id="fd_" + "0" * 20,
            category=FindingCategory.EDGE_ASSOCIATION,
            title="Edge difference",
            summary="The standardized difference is 2.5.",
            evidence_ids=("ev_" + "0" * 20,),
        )


def test_report_rejects_findings_outside_its_evidence_graph(tmp_path: Path) -> None:
    report = _report("edge_effect_obvious", tmp_path / "plots")
    invalid = report.model_dump(mode="python")
    invalid["findings"][0]["evidence_ids"] = ("ev_" + "f" * 20,)

    with pytest.raises(ValidationError, match="outside the report"):
        BaselineReport.model_validate(invalid)
