"""Dependency-boundary tests for deterministic investigator code."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[3] / "src" / "experiment_failure_investigator"
DETERMINISTIC_ROOTS = (PACKAGE_ROOT / "analysis", PACKAGE_ROOT / "reporting")
FORBIDDEN_MODULE_PREFIXES = (
    "anthropic",
    "langgraph",
    "ollama",
    "openai",
    "pydantic_ai",
    "experiment_failure_investigator.benchmark.injectors",
)
PRIVATE_BENCHMARK_MODELS = {
    "ArtifactReference",
    "CaseFiles",
    "CaseManifest",
    "CaseVariant",
    "FailureMode",
    "GeneratorConfig",
    "GroundTruth",
    "SimulatedTraversal",
    "TreatmentCurve",
}


def _python_files() -> tuple[Path, ...]:
    return tuple(
        sorted(path for root in DETERMINISTIC_ROOTS for path in root.glob("*.py"))
    )


def test_deterministic_layer_has_no_agent_or_provider_imports() -> None:
    violations: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = (node.module,)
            for module in modules:
                if module.startswith(FORBIDDEN_MODULE_PREFIXES):
                    violations.append(f"{path.name}: {module}")
    assert not violations, violations


def test_deterministic_layer_does_not_import_private_benchmark_models() -> None:
    violations: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "experiment_failure_investigator.benchmark.models":
                continue
            imported = {alias.name for alias in node.names}
            private = imported & PRIVATE_BENCHMARK_MODELS
            if private:
                violations.append(f"{path.name}: {sorted(private)}")
    assert not violations, violations
