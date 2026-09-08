"""Integration coverage for deterministic single-case and batch QC commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment_failure_investigator.cli import main
from experiment_failure_investigator.reporting import BaselineReport, BatchQcSummary

CASE_ROOT = Path(__file__).parents[2] / "cases"


def test_qc_writes_valid_report_and_refuses_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "single"
    arguments = [
        "qc",
        str(CASE_ROOT / "weak_controls_obvious"),
        "--output",
        str(output),
    ]

    main(arguments)

    report = BaselineReport.model_validate_json(
        (output / "report.json").read_text(encoding="utf-8")
    )
    assert report.case_id.startswith("case_")
    assert {path.name for path in output.iterdir()} == {
        "condition_residual.png",
        "control_distribution.png",
        "dose_response.png",
        "raw_signal.png",
        "report.json",
        "report.md",
    }
    assert "Wrote deterministic QC report" in capsys.readouterr().out

    original = (output / "report.json").read_bytes()
    with pytest.raises(SystemExit) as exc_info:
        main(arguments)
    assert exc_info.value.code == 2
    assert (output / "report.json").read_bytes() == original

    main([*arguments, "--force"])
    assert (output / "report.json").read_bytes() == original


def test_qc_default_output_uses_opaque_case_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    main(["qc", str(CASE_ROOT / "edge_effect_noisy")])

    output_directories = list((tmp_path / "reports").iterdir())
    assert len(output_directories) == 1
    assert output_directories[0].name.startswith("case_")
    assert "edge_effect" not in output_directories[0].name


def test_qc_batch_writes_all_reports_and_label_free_contract(
    tmp_path: Path,
) -> None:
    output = tmp_path / "batch"

    main(["qc-batch", str(CASE_ROOT), "--output", str(output)])

    summary_text = (output / "batch_summary.json").read_text(encoding="utf-8")
    summary = BatchQcSummary.model_validate_json(summary_text)
    assert len(summary.cases) == 14
    assert len({case.case_id for case in summary.cases}) == 14
    assert all((output / case.report_json).is_file() for case in summary.cases)
    assert all((output / case.report_markdown).is_file() for case in summary.cases)
    assert all(Path(case.report_json).parts[0] == case.case_id for case in summary.cases)
    assert all(case.applicable_tools for case in summary.cases)
    assert any(
        status.value == "not_applicable"
        for case in summary.cases
        for tool in case.tool_statuses
        for status in tool.statuses
    )

    serialized_keys: set[str] = set()

    def collect_keys(value: object) -> None:
        if isinstance(value, dict):
            serialized_keys.update(str(key) for key in value)
            for child in value.values():
                collect_keys(child)
        elif isinstance(value, list):
            for child in value:
                collect_keys(child)

    collect_keys(json.loads(summary_text))
    assert serialized_keys.isdisjoint(
        {
            "planted_failure_mode",
            "expected_discriminating_evidence",
            "injection_parameters",
            "child_seeds",
            "root_seed",
        }
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["qc-batch", str(CASE_ROOT), "--output", str(output)])
    assert exc_info.value.code == 2
