"""Offline integration coverage for the user-facing investigation command."""

from __future__ import annotations

from pathlib import Path

import pytest

import experiment_failure_investigator.cli as cli
from experiment_failure_investigator.agent.replay import (
    ReplayModel,
    load_replay_fixture,
)
from experiment_failure_investigator.agent.trace import InvestigationTrace, RunStatus

CASE_DIRECTORY = Path(__file__).parents[2] / "cases" / "weak_controls_obvious"
REPLAY_PATH = (
    Path(__file__).parents[1] / "fixtures" / "agent" / "replay_week3_v1.json"
)


def test_run_command_writes_validated_result_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "investigation"
    fixture = load_replay_fixture(REPLAY_PATH, "catalog_then_output")
    model = ReplayModel(fixture)
    monkeypatch.setenv("EFI_RUNTIME_MODE", "replay")
    monkeypatch.setattr(cli, "build_ollama_model", lambda _config: model)
    arguments = [
        "run",
        str(CASE_DIRECTORY),
        "--output",
        str(output),
        "--model",
        "qwen:test",
        "--ollama-base-url",
        "http://127.0.0.1:11434/v1",
    ]

    cli.main(arguments)

    model.assert_complete()
    trace = InvestigationTrace.model_validate_json(
        (output / "trace.json").read_text(encoding="utf-8")
    )
    assert trace.status is RunStatus.SUCCESS
    assert trace.runtime.model_tag == "qwen:test"
    assert trace.runtime.ollama_base_url == "http://127.0.0.1:11434/v1"
    assert (output / "investigation.json").is_file()
    assert (output / "baseline" / "report.json").is_file()
    assert "Wrote validated investigation" in capsys.readouterr().out

    original_trace = (output / "trace.json").read_bytes()
    with pytest.raises(SystemExit) as exc_info:
        cli.main(arguments)

    assert exc_info.value.code == 2
    assert (output / "trace.json").read_bytes() == original_trace
    assert "run output already exists" in capsys.readouterr().err
