"""Tests for bounded decreasing four-parameter logistic fitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiment_failure_investigator.analysis.contracts import (
    InvestigatorCase,
    PublicArtifactHashes,
)
from experiment_failure_investigator.analysis.design import build_investigator_case
from experiment_failure_investigator.analysis.dose_response import fit_dose_response
from experiment_failure_investigator.analysis.results import (
    EvidenceRecord,
    ScientificToolResult,
    ToolStatus,
)
from experiment_failure_investigator.benchmark.adapter import load_investigator_case
from experiment_failure_investigator.benchmark.layouts import build_balanced_layout
from experiment_failure_investigator.benchmark.models import (
    CaseVariant,
    FailureMode,
    GeneratorConfig,
)
from experiment_failure_investigator.benchmark.serialization import (
    build_case_metadata,
    load_case,
)
from experiment_failure_investigator.benchmark.signals import generate_clean_assay

VALID_HASH = "0" * 64
HASHES = PublicArtifactHashes(
    measurements=VALID_HASH,
    plate_map=VALID_HASH,
    metadata=VALID_HASH,
    protocol=VALID_HASH,
    problem_statement=VALID_HASH,
)


def _generated_case(*, one_treatment: bool = False) -> InvestigatorCase:
    overrides: dict[str, Any] = {}
    if one_treatment:
        overrides = {
            "treatments": ["test_treatment"],
            "replicates_per_condition": 10,
            "curves": {"test_treatment": {"ic50": 1.0}},
        }
    config = GeneratorConfig(
        case_id="dose_fit_fixture",
        case_variant=CaseVariant.OBVIOUS,
        failure_mode=FailureMode.EDGE_EFFECT,
        root_seed=123,
        noise_sd=0.0,
        **overrides,
    )
    plate_map = build_balanced_layout(config)
    assay = generate_clean_assay(plate_map, config)
    return build_investigator_case(
        measurements=assay.measurements,
        plate_map=assay.plate_map,
        metadata=build_case_metadata(config, assay.plate_map),
        protocol="Synthetic dose-response protocol.",
        problem_statement="Assess the observed dose response.",
        public_artifact_hashes=HASHES,
    )


def _result_for(
    results: tuple[ScientificToolResult, ...],
    treatment: str,
) -> ScientificToolResult:
    matches = [
        result
        for result in results
        if any(
            evidence.scope.treatments == (treatment,) for evidence in result.evidence
        )
    ]
    assert len(matches) == 1
    return matches[0]


def _metric(result: ScientificToolResult, name: str) -> EvidenceRecord:
    matches = [record for record in result.evidence if record.metric_name == name]
    assert len(matches) == 1
    return matches[0]


def test_noiseless_fit_recovers_known_four_parameter_curves() -> None:
    results = fit_dose_response(_generated_case())

    for treatment, expected_ic50 in (
        ("reference_treatment", 0.3),
        ("test_treatment", 1.0),
    ):
        result = _result_for(results, treatment)
        assert result.status is ToolStatus.SUCCESS
        assert _metric(result, "dose_response.bottom").value == pytest.approx(
            0.15, abs=1e-7
        )
        assert _metric(result, "dose_response.top").value == pytest.approx(
            1.0, abs=1e-7
        )
        assert _metric(result, "dose_response.ic50").value == pytest.approx(
            expected_ic50, rel=1e-7
        )
        assert _metric(result, "dose_response.hill_slope").value == pytest.approx(
            1.2, rel=1e-7
        )
        assert _metric(result, "dose_response.rmse").value == pytest.approx(
            0.0, abs=1e-8
        )


def test_fit_does_not_require_a_reference_treatment() -> None:
    results = fit_dose_response(_generated_case(one_treatment=True))

    assert len(results) == 1
    assert results[0].status is ToolStatus.SUCCESS
    assert _metric(results[0], "dose_response.ic50").value == pytest.approx(
        1.0, rel=1e-7
    )


def test_exact_flat_response_is_not_forced_into_a_curve() -> None:
    loaded = load_case(Path("cases/true_non_response_obvious"))
    measurements = loaded.measurements.copy(deep=True)
    test_wells = set(
        loaded.plate_map.loc[
            loaded.plate_map["treatment"] == "test_treatment", "well"
        ]
    )
    measurements.loc[measurements["well"].isin(test_wells), "raw_signal"] = 0.5
    case = build_investigator_case(
        measurements=measurements,
        plate_map=loaded.plate_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )

    result = _result_for(fit_dose_response(case), "test_treatment")

    assert result.status is ToolStatus.INSUFFICIENT_DATA
    assert _metric(result, "dose_response.observed_response_range").value == 0.0
    assert result.warnings[0].code == "flat_observed_response"
    assert not any(
        evidence.metric_name == "dose_response.ic50" for evidence in result.evidence
    )


def test_too_few_unique_doses_is_insufficient_data() -> None:
    loaded = load_case(Path("cases/true_non_response_obvious"))
    plate_map = loaded.plate_map[
        (loaded.plate_map["well_role"] != "treatment")
        | (
            (loaded.plate_map["treatment"] == "test_treatment")
            & (loaded.plate_map["dose"].isin([0.003, 0.01, 0.03, 0.1]))
        )
    ].copy()
    retained = set(plate_map["well"])
    measurements = loaded.measurements[
        loaded.measurements["well"].isin(retained)
    ].copy()
    case = build_investigator_case(
        measurements=measurements,
        plate_map=plate_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )

    result = fit_dose_response(case)[0]

    assert result.status is ToolStatus.INSUFFICIENT_DATA
    assert _metric(result, "dose_response.unique_dose_count").value == 4


def test_missing_observations_are_excluded_with_a_warning() -> None:
    loaded = load_case(Path("cases/edge_effect_noisy"))
    measurements = loaded.measurements.copy(deep=True)
    target_well = loaded.plate_map.loc[
        loaded.plate_map["treatment"] == "test_treatment", "well"
    ].iloc[0]
    measurements.loc[measurements["well"] == target_well, "raw_signal"] = np.nan
    case = build_investigator_case(
        measurements=measurements,
        plate_map=loaded.plate_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )

    result = _result_for(fit_dose_response(case), "test_treatment")

    assert result.status is ToolStatus.SUCCESS
    assert _metric(result, "dose_response.mapped_well_count").value == 40
    assert _metric(result, "dose_response.observation_count").value == 39
    assert "missing_dose_response_signals" in {
        warning.code for warning in result.warnings
    }


def test_optimizer_failure_is_returned_as_structured_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_optimizer(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(
        "experiment_failure_investigator.analysis.dose_response.curve_fit",
        fail_optimizer,
    )

    results = fit_dose_response(_generated_case(one_treatment=True))

    assert results[0].status is ToolStatus.ERROR
    assert "RuntimeError" in (results[0].status_reason or "")
    assert results[0].evidence == ()


@pytest.mark.parametrize("variant", ["obvious", "noisy"])
def test_true_nonresponse_returns_identifiability_warning(variant: str) -> None:
    case = load_investigator_case(Path(f"cases/true_non_response_{variant}"))
    result = _result_for(fit_dose_response(case), "test_treatment")

    assert result.status is ToolStatus.SUCCESS
    assert "parameters_near_bounds" in {warning.code for warning in result.warnings}


def test_multi_plate_fits_remain_plate_specific() -> None:
    results = fit_dose_response(
        load_investigator_case(Path("cases/batch_shift_obvious"))
    )

    assert len(results) == 4
    assert {result.scope.plate_ids for result in results} == {
        ("plate_01",),
        ("plate_02",),
    }


def test_fit_is_invariant_to_source_row_order() -> None:
    loaded = load_case(Path("cases/edge_effect_noisy"))
    original = build_investigator_case(
        measurements=loaded.measurements,
        plate_map=loaded.plate_map,
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )
    reordered = build_investigator_case(
        measurements=loaded.measurements.sample(frac=1, random_state=7),
        plate_map=loaded.plate_map.sample(frac=1, random_state=11),
        metadata=loaded.metadata,
        protocol=loaded.protocol,
        problem_statement=loaded.problem_statement,
        public_artifact_hashes=HASHES,
    )

    assert fit_dose_response(original) == fit_dose_response(reordered)


def test_invalid_fit_parameters_are_rejected() -> None:
    case = _generated_case(one_treatment=True)

    with pytest.raises(ValueError):
        fit_dose_response(case, min_unique_doses=4)
    with pytest.raises(ValueError):
        fit_dose_response(case, response_epsilon=0.0)
    with pytest.raises(ValueError):
        fit_dose_response(case, response_epsilon=float("nan"))
    with pytest.raises(ValueError):
        fit_dose_response(case, bound_tolerance_fraction=0.0)
    with pytest.raises(ValueError):
        fit_dose_response(case, bound_tolerance_fraction=0.5)
    with pytest.raises(ValueError):
        fit_dose_response(case, max_function_evaluations=0)
