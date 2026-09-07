"""Cross-tool behavior for literal empty wells with arbitrary readouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment_failure_investigator.analysis.batches import compare_batches
from experiment_failure_investigator.analysis.controls import summarize_controls
from experiment_failure_investigator.analysis.design import build_investigator_case
from experiment_failure_investigator.analysis.dose_response import fit_dose_response
from experiment_failure_investigator.analysis.heatmaps import generate_plate_heatmap
from experiment_failure_investigator.analysis.missingness import inspect_missingness
from experiment_failure_investigator.analysis.replicates import (
    calculate_replicate_variability,
)
from experiment_failure_investigator.analysis.results import ToolStatus
from experiment_failure_investigator.analysis.spatial import (
    condition_centered_residuals,
    detect_spatial_effects,
)
from experiment_failure_investigator.analysis.contracts import PublicArtifactHashes
from experiment_failure_investigator.benchmark.models import WellRole
from experiment_failure_investigator.benchmark.serialization import load_case

ZERO_HASH = "0" * 64
HASHES = PublicArtifactHashes(
    measurements=ZERO_HASH,
    plate_map=ZERO_HASH,
    metadata=ZERO_HASH,
    protocol=ZERO_HASH,
    problem_statement=ZERO_HASH,
)


def _case_with_empty_readout_variants(
    observed_high: float = 999.0,
    observed_low: float = -999.0,
):
    loaded = load_case(Path("cases/true_non_response_obvious"))
    plate_map = loaded.plate_map.copy()
    measurements = loaded.measurements.copy()
    empty_wells = list(plate_map["well"].iloc[:5])
    empty_mask = plate_map["well"].isin(empty_wells)
    plate_map.loc[empty_mask, "sample_id"] = [
        f"empty_{index}" for index in range(1, 6)
    ]
    plate_map.loc[empty_mask, "well_role"] = WellRole.EMPTY.value
    plate_map.loc[empty_mask, "treatment"] = None
    plate_map.loc[empty_mask, "dose"] = np.nan
    plate_map.loc[empty_mask, "dose_unit"] = None
    plate_map.loc[empty_mask, "replicate"] = np.nan
    measurements.loc[
        measurements["well"] == empty_wells[0], "raw_signal"
    ] = observed_high
    measurements.loc[
        measurements["well"] == empty_wells[1], "raw_signal"
    ] = observed_low
    measurements.loc[measurements["well"] == empty_wells[2], "raw_signal"] = np.nan
    measurements.loc[measurements["well"] == empty_wells[3], "raw_signal"] = np.inf
    measurements = measurements[measurements["well"] != empty_wells[4]].copy()
    case = build_investigator_case(
        measurements=measurements,
        plate_map=plate_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )
    return case, set(empty_wells)


def test_design_and_missingness_separate_empty_well_readouts() -> None:
    case, _ = _case_with_empty_readout_variants()
    plate = case.design.plates[0]

    assert plate.empty_well_count == 5
    assert plate.empty_missing_measurement_count == 1
    assert plate.empty_null_measurement_count == 1
    assert plate.empty_nonfinite_measurement_count == 1
    assert plate.missing_measurement_count == 0
    assert plate.null_measurement_count == 0
    assert plate.nonfinite_measurement_count == 0

    result = inspect_missingness(case)
    warning_ids = {
        evidence_id
        for warning in result.warnings
        for evidence_id in warning.evidence_ids
    }
    empty_records = [
        record
        for record in result.evidence
        if record.scope.well_roles == (WellRole.EMPTY,)
    ]
    assert empty_records
    assert warning_ids.isdisjoint(
        record.evidence_id for record in empty_records
    )


def test_numeric_diagnostics_do_not_treat_empty_readouts_as_biology() -> None:
    case, empty_wells = _case_with_empty_readout_variants()
    changed_readouts, _ = _case_with_empty_readout_variants(1_000_000.0, -0.0001)

    assert all(
        observation.well not in empty_wells
        for observation in condition_centered_residuals(case)
    )
    for tool in (
        summarize_controls,
        calculate_replicate_variability,
        fit_dose_response,
        detect_spatial_effects,
        compare_batches,
    ):
        original_results = tool(case)
        changed_results = tool(changed_readouts)
        assert all(
            result.status is not ToolStatus.ERROR for result in original_results
        )
        assert changed_results == original_results


def test_heatmap_domain_ignores_arbitrary_empty_readouts(tmp_path: Path) -> None:
    case, _ = _case_with_empty_readout_variants()

    raw = generate_plate_heatmap(case, tmp_path, filename="raw.png")
    residual = generate_plate_heatmap(
        case,
        tmp_path,
        filename="residual.png",
        view="condition_residual",
    )

    assert raw.status is ToolStatus.SUCCESS
    assert residual.status is ToolStatus.SUCCESS
    domain = raw.parameters["color_domain"]
    assert isinstance(domain, list)
    assert all(isinstance(value, int | float) for value in domain)
    numeric_domain = [
        float(value) for value in domain if isinstance(value, int | float)
    ]
    assert min(numeric_domain) > -2
    assert max(numeric_domain) < 2
