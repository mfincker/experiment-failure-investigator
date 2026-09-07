"""One-way adapter from labeled benchmark cases to investigator-safe inputs."""

from __future__ import annotations

from pathlib import Path

from experiment_failure_investigator.analysis.contracts import (
    InvestigatorCase,
    PublicArtifactHashes,
)
from experiment_failure_investigator.analysis.design import build_investigator_case
from experiment_failure_investigator.benchmark.serialization import load_case


def load_investigator_case(case_directory: Path) -> InvestigatorCase:
    """Verify a benchmark case, then discard all evaluation-only information."""
    loaded = load_case(case_directory)
    files = loaded.manifest.files
    hashes = PublicArtifactHashes(
        measurements=files.measurements.sha256,
        plate_map=files.plate_map.sha256,
        metadata=files.metadata.sha256,
        protocol=files.protocol.sha256,
        problem_statement=files.problem_statement.sha256,
    )
    return build_investigator_case(
        measurements=loaded.measurements,
        plate_map=loaded.plate_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=hashes,
    )
