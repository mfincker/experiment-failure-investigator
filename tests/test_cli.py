"""Baseline tests for package and CLI wiring."""

import pytest

import experiment_failure_investigator
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
