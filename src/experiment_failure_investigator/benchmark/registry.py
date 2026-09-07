"""Version-controlled Week 1 benchmark case registry and batch workflow."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TypeAlias

import pandas as pd

from experiment_failure_investigator.benchmark.injectors import (
    BatchShiftParameters,
    EdgeEffectParameters,
    InjectedAssayResult,
    LayoutConfoundingParameters,
    PipettingDriftParameters,
    TransientTipClogParameters,
    TrueNonResponseParameters,
    WeakControlsParameters,
    inject_batch_shift,
    inject_edge_effect,
    inject_layout_confounding,
    inject_pipetting_drift,
    inject_transient_tip_clog,
    inject_true_non_response,
    inject_weak_controls,
)
from experiment_failure_investigator.benchmark.layouts import build_balanced_layout
from experiment_failure_investigator.benchmark.models import (
    CaseMetadata,
    CaseVariant,
    FailureMode,
    GeneratorConfig,
    PlateMetadata,
    SimulatedTraversal,
)
from experiment_failure_investigator.benchmark.plotting import (
    PlotMetadata,
    export_comparison_grid,
    export_plot_set,
    shared_residual_domain,
    shared_signal_domain,
)
from experiment_failure_investigator.benchmark.serialization import (
    MANIFEST_FILENAME,
    CasePayload,
    LoadedCase,
    PlotWriter,
    build_case_metadata,
    load_case,
    write_case,
)
from experiment_failure_investigator.benchmark.signals import (
    CleanAssayResult,
    generate_clean_assay,
)

InjectionParameters: TypeAlias = (
    EdgeEffectParameters
    | PipettingDriftParameters
    | TransientTipClogParameters
    | LayoutConfoundingParameters
    | WeakControlsParameters
    | BatchShiftParameters
    | TrueNonResponseParameters
)


@dataclass(frozen=True)
class CaseDefinition:
    """One version-controlled benchmark configuration and interpretation target."""

    config: GeneratorConfig
    injection_parameters: InjectionParameters
    problem_statement: str
    expected_discriminating_evidence: tuple[str, ...]
    plausible_confounders: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedCase:
    """In-memory clean baseline, injected assay, and serializable payload."""

    definition: CaseDefinition
    clean: CleanAssayResult
    injected: InjectedAssayResult
    payload: CasePayload


PROTOCOL = """# Synthetic endpoint cell-viability assay

Cells were exposed to the reported treatment and concentration, followed by an
endpoint viability readout. Signals are normalized relative to negative controls,
and lower values indicate stronger inhibition. Plate-map assignments and available
plate metadata should be considered when interpreting unexpected results.
"""

EVIDENCE: dict[FailureMode, tuple[tuple[str, ...], tuple[str, ...], str]] = {
    FailureMode.EDGE_EFFECT: (
        ("A condition-adjusted difference is concentrated on plate boundaries.",),
        ("Condition placement associated with boundary position.",),
        "The assay response appears inconsistent across plate positions.",
    ),
    FailureMode.PIPETTING_DRIFT: (
        ("Condition-adjusted signal follows a row-wise or column-wise gradient.",),
        ("Incubation timing, temperature gradient, or layout confounding.",),
        "The assay response shows a gradual spatial trend.",
    ),
    FailureMode.TRANSIENT_TIP_CLOG: (
        ("A localized sequence of wells is consistent with grouped dispensing.",),
        ("Localized contamination, readout artifact, or annotation error.",),
        "A small localized set of wells differs unexpectedly from its replicates.",
    ),
    FailureMode.LAYOUT_CONFOUNDING: (
        ("Experimental condition is inseparable from plate position.",),
        ("A genuine treatment effect or an unobserved spatial artifact.",),
        "Treatment responses differ, but their positions may complicate interpretation.",
    ),
    FailureMode.WEAK_CONTROLS: (
        ("Positive and negative controls have substantially reduced separation.",),
        ("Incorrect control labels or globally elevated variability.",),
        "The positive controls appear weaker than expected.",
    ),
    FailureMode.BATCH_SHIFT: (
        ("Response dynamic range differs between plates with controls still anchored.",),
        ("Plate-specific protocol, instrument, or sample-composition differences.",),
        "Comparable plates show different response ranges.",
    ),
    FailureMode.TRUE_NON_RESPONSE: (
        ("The test treatment is flat while controls and the reference remain responsive.",),
        ("Test-treatment preparation or dose-annotation error.",),
        "One treatment shows little response over the tested concentration range.",
    ),
}


def _config(
    mode: FailureMode,
    variant: CaseVariant,
    seed: int,
) -> GeneratorConfig:
    return GeneratorConfig(
        case_id=f"{mode.value}_{variant.value}",
        case_variant=variant,
        failure_mode=mode,
        root_seed=seed,
        noise_sd=0.03 if variant is CaseVariant.OBVIOUS else 0.07,
        plate_count=2 if mode is FailureMode.BATCH_SHIFT else 1,
    )


def _definition(
    mode: FailureMode,
    variant: CaseVariant,
    seed: int,
    parameters: InjectionParameters,
) -> CaseDefinition:
    evidence, confounders, problem = EVIDENCE[mode]
    return CaseDefinition(
        config=_config(mode, variant, seed),
        injection_parameters=parameters,
        problem_statement=problem,
        expected_discriminating_evidence=evidence,
        plausible_confounders=confounders,
    )


def week_1_case_registry() -> tuple[CaseDefinition, ...]:
    """Return the ordered fourteen-case Week 1 benchmark registry."""
    obvious = CaseVariant.OBVIOUS
    noisy = CaseVariant.NOISY
    return (
        _definition(
            FailureMode.EDGE_EFFECT,
            obvious,
            1001,
            EdgeEffectParameters(magnitude=0.25),
        ),
        _definition(
            FailureMode.EDGE_EFFECT,
            noisy,
            1002,
            EdgeEffectParameters(magnitude=0.12, sides=("top", "right")),
        ),
        _definition(
            FailureMode.PIPETTING_DRIFT,
            obvious,
            1003,
            PipettingDriftParameters(
                magnitude=0.30,
                traversal=SimulatedTraversal.ROW_MAJOR,
            ),
        ),
        _definition(
            FailureMode.PIPETTING_DRIFT,
            noisy,
            1004,
            PipettingDriftParameters(
                magnitude=0.15,
                traversal=SimulatedTraversal.ROW_MAJOR,
            ),
        ),
        _definition(
            FailureMode.TRANSIENT_TIP_CLOG,
            obvious,
            1005,
            TransientTipClogParameters(
                magnitude=0.25,
                dispense_group_size=8,
                affected_group_start=3,
                affected_group_count=4,
                affected_channels=(2,),
            ),
        ),
        _definition(
            FailureMode.TRANSIENT_TIP_CLOG,
            noisy,
            1006,
            TransientTipClogParameters(
                magnitude=0.12,
                dispense_group_size=8,
                affected_group_start=3,
                affected_group_count=4,
                affected_channels=(2,),
            ),
        ),
        _definition(
            FailureMode.LAYOUT_CONFOUNDING,
            obvious,
            1007,
            LayoutConfoundingParameters(),
        ),
        _definition(
            FailureMode.LAYOUT_CONFOUNDING,
            noisy,
            1008,
            LayoutConfoundingParameters(),
        ),
        _definition(
            FailureMode.WEAK_CONTROLS,
            obvious,
            1009,
            WeakControlsParameters(collapse_fraction=0.80),
        ),
        _definition(
            FailureMode.WEAK_CONTROLS,
            noisy,
            1010,
            WeakControlsParameters(collapse_fraction=0.55),
        ),
        _definition(
            FailureMode.BATCH_SHIFT,
            obvious,
            1011,
            BatchShiftParameters(
                shifted_plate_id="plate_02",
                response_scale_factor=0.70,
            ),
        ),
        _definition(
            FailureMode.BATCH_SHIFT,
            noisy,
            1012,
            BatchShiftParameters(
                shifted_plate_id="plate_02",
                response_scale_factor=0.85,
            ),
        ),
        _definition(
            FailureMode.TRUE_NON_RESPONSE,
            obvious,
            1013,
            TrueNonResponseParameters(
                target_treatment="test_treatment",
                flat_expected_signal=1.0,
            ),
        ),
        _definition(
            FailureMode.TRUE_NON_RESPONSE,
            noisy,
            1014,
            TrueNonResponseParameters(
                target_treatment="test_treatment",
                flat_expected_signal=1.0,
            ),
        ),
    )


def _inject(
    clean: CleanAssayResult,
    parameters: InjectionParameters,
) -> InjectedAssayResult:
    if isinstance(parameters, EdgeEffectParameters):
        return inject_edge_effect(clean, parameters)
    if isinstance(parameters, PipettingDriftParameters):
        return inject_pipetting_drift(clean, parameters)
    if isinstance(parameters, TransientTipClogParameters):
        return inject_transient_tip_clog(clean, parameters)
    if isinstance(parameters, LayoutConfoundingParameters):
        return inject_layout_confounding(clean, parameters)
    if isinstance(parameters, WeakControlsParameters):
        return inject_weak_controls(clean, parameters)
    if isinstance(parameters, BatchShiftParameters):
        return inject_batch_shift(clean, parameters)
    if isinstance(parameters, TrueNonResponseParameters):
        return inject_true_non_response(clean, parameters)
    raise TypeError(f"unsupported injection parameters: {type(parameters).__name__}")


def build_registered_case(definition: CaseDefinition) -> GeneratedCase:
    """Generate one registry case in memory without writing files."""
    config = definition.config
    layouts = [
        build_balanced_layout(config, plate_id=f"plate_{index:02d}")
        for index in range(1, config.plate_count + 1)
    ]
    layout = pd.concat(layouts, ignore_index=True)
    clean = generate_clean_assay(layout, config)
    injected = _inject(clean, definition.injection_parameters)
    minimal_metadata = build_case_metadata(config, injected.plate_map)
    plates = [
        PlateMetadata(
            plate_id=plate_id,
            batch_id=(
                f"batch_{index:02d}"
                if config.failure_mode is FailureMode.BATCH_SHIFT
                else "batch_01"
            ),
            operator_label="synthetic_operator_01",
            run_date=date(2026, 9, index),
            instrument_label="synthetic_reader_01",
        )
        for index, plate_id in enumerate(
            sorted(str(value) for value in injected.plate_map["plate_id"].unique()),
            start=1,
        )
    ]
    metadata = CaseMetadata(
        case_id=minimal_metadata.case_id,
        assay_type=minimal_metadata.assay_type,
        signal_direction=minimal_metadata.signal_direction,
        layout_fingerprint=minimal_metadata.layout_fingerprint,
        plates=plates,
    )
    payload = CasePayload(
        config=config,
        assay=injected,
        metadata=metadata,
        protocol=PROTOCOL,
        problem_statement=definition.problem_statement,
        expected_discriminating_evidence=definition.expected_discriminating_evidence,
        plausible_confounders=definition.plausible_confounders,
    )
    return GeneratedCase(
        definition=definition,
        clean=clean,
        injected=injected,
        payload=payload,
    )


def _plot_writer(generated: GeneratedCase) -> PlotWriter:
    config = generated.definition.config
    metadata = PlotMetadata(
        case_id=config.case_id,
        failure_label=config.failure_mode.value,
        variant=config.case_variant.value,
        root_seed=config.root_seed,
    )
    signal_domain = shared_signal_domain(generated.clean, generated.injected)
    residual_domain = shared_residual_domain(generated.clean, generated.injected)

    def write(plot_directory: Path) -> dict[str, Path]:
        paths = export_plot_set(
            generated.injected,
            metadata,
            plot_directory,
            signal_domain=signal_domain,
            residual_domain=residual_domain,
        )
        paths["comparison_grid"] = export_comparison_grid(
            generated.clean,
            generated.injected,
            metadata,
            plot_directory / "comparison_grid.png",
        )
        return paths

    return write


def _index_records(loaded_cases: list[LoadedCase], root: Path) -> list[dict[str, object]]:
    return [
        {
            "case_id": case.manifest.case_id,
            "failure_mode": case.manifest.planted_failure_mode.value,
            "layout_fingerprint": case.metadata.layout_fingerprint,
            "manifest": (case.directory / MANIFEST_FILENAME).relative_to(root).as_posix(),
            "path": case.directory.relative_to(root).as_posix(),
            "root_seed": case.manifest.root_seed,
            "validated": True,
            "variant": case.manifest.case_variant.value,
        }
        for case in loaded_cases
    ]


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _write_index(loaded_cases: list[LoadedCase], root: Path) -> None:
    content = json.dumps(
        {
            "benchmark_version": "1.0.0",
            "case_count": len(loaded_cases),
            "cases": _index_records(loaded_cases, root),
            "schema_version": "1.0.0",
        },
        indent=2,
        sort_keys=True,
    ) + "\n"
    _atomic_write_text(root / "index.json", content)


def _write_review_page(loaded_cases: list[LoadedCase], root: Path) -> None:
    cards = "\n".join(
        (
            f'<article><h2>{case.manifest.case_id}</h2>'
            f'<a href="{case.manifest.case_id}/plots/comparison_grid.png">'
            f'<img src="{case.manifest.case_id}/plots/comparison_grid.png" '
            f'alt="{case.manifest.case_id} comparison grid"></a></article>'
        )
        for case in loaded_cases
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Week 1 benchmark review</title>
<style>body{{font-family:system-ui;margin:2rem;background:#f8fafc;color:#111827}}
main{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1.5rem}}
article{{background:white;padding:1rem;border:1px solid #d1d5db;border-radius:.5rem}}
h1,h2{{margin-top:0}}img{{display:block;width:100%;height:auto}}
@media(max-width:900px){{main{{grid-template-columns:1fr}}}}</style></head>
<body><h1>Week 1 clean/injected benchmark comparison</h1><main>
{cards}
</main></body></html>
"""
    _atomic_write_text(root / "review.html", html)


def _validate_layout_fingerprint_split(loaded_cases: list[LoadedCase]) -> None:
    ordinary = {
        case.metadata.layout_fingerprint
        for case in loaded_cases
        if case.manifest.planted_failure_mode is not FailureMode.LAYOUT_CONFOUNDING
    }
    confounded = {
        case.metadata.layout_fingerprint
        for case in loaded_cases
        if case.manifest.planted_failure_mode is FailureMode.LAYOUT_CONFOUNDING
    }
    if len(ordinary) != 1:
        raise ValueError("ordinary cases must share one baseline layout fingerprint")
    if len(confounded) != 1 or not ordinary.isdisjoint(confounded):
        raise ValueError("layout-confounding cases must share one distinct fingerprint")


def validate_registered_cases(root: Path) -> list[LoadedCase]:
    """Validate all and only the registered Week 1 case directories."""
    registry = week_1_case_registry()
    expected = {definition.config.case_id: definition for definition in registry}
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    } if root.is_dir() else set()
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ValueError(f"case directory set mismatch; missing={missing}, extra={extra}")
    loaded_cases = [load_case(root / case_id) for case_id in expected]
    for case in loaded_cases:
        definition = expected[case.manifest.case_id]
        if (
            case.manifest.generation != definition.config
            or case.manifest.planted_failure_mode is not definition.config.failure_mode
        ):
            raise ValueError(f"case {case.manifest.case_id!r} disagrees with the registry")
        required_plots = {
            "plate_heatmap",
            "residual_heatmap",
            "dose_response",
            "control_qc",
            "comparison_grid",
        }
        if set(case.manifest.files.plots) != required_plots:
            raise ValueError(f"case {case.manifest.case_id!r} has an invalid plot set")
    _validate_layout_fingerprint_split(loaded_cases)
    return loaded_cases


def generate_registered_cases(root: Path, *, force: bool = False) -> list[LoadedCase]:
    """Generate, plot, serialize, and validate all Week 1 registry cases."""
    if not force and root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for definition in week_1_case_registry():
        generated = build_registered_case(definition)
        write_case(
            generated.payload,
            root,
            force=force,
            plot_writer=_plot_writer(generated),
        )
    loaded_cases = validate_registered_cases(root)
    _write_index(loaded_cases, root)
    _write_review_page(loaded_cases, root)
    return loaded_cases
