"""Baseline tests for package and CLI wiring."""

from pathlib import Path

import pytest

import experiment_failure_investigator
import experiment_failure_investigator.cli as cli
from experiment_failure_investigator.cli import build_parser, main


def test_package_imports() -> None:
    assert experiment_failure_investigator.__doc__


def test_parser_uses_console_script_name() -> None:
    assert build_parser().prog == "experiment-failure-investigator"


def test_help_is_available(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    assert "Generate, validate, and investigate" in capsys.readouterr().out


def test_generate_cases_command_dispatches_requested_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "cases"
    calls: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        cli,
        "generate_registered_cases",
        lambda path, force=False: calls.append((path, force)) or [object()] * 14,
    )

    main(["generate-cases", "--output", str(output), "--force"])

    assert calls == [(output, True)]
    assert "Generated and validated 14 cases" in capsys.readouterr().out


def test_validate_cases_command_dispatches_requested_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "cases"
    monkeypatch.setattr(
        cli,
        "validate_registered_cases",
        lambda requested: [object()] * 14 if requested == path else [],
    )

    main(["validate-cases", str(path)])

    assert "Validated 14 cases" in capsys.readouterr().out


def test_run_parser_exposes_only_local_runtime_inputs() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "cases/example",
            "--output",
            "runs/example",
            "--model",
            "qwen:test",
            "--ollama-base-url",
            "http://127.0.0.1:11434/v1",
        ]
    )

    assert args.command == "run"
    assert args.path == Path("cases/example")
    assert args.output == Path("runs/example")
    assert args.model_tag == "qwen:test"
    assert args.ollama_base_url == "http://127.0.0.1:11434/v1"
