"""Geometry-aware plate layout generation and independent balance audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from experiment_failure_investigator.benchmark.models import (
    GeneratorConfig,
    PlateFormat,
    WellRole,
)
from experiment_failure_investigator.benchmark.randomness import derive_child_seed

PLATE_MAP_COLUMNS = [
    "plate_id",
    "well",
    "row",
    "column",
    "sample_id",
    "well_role",
    "treatment",
    "dose",
    "dose_unit",
    "replicate",
]


@dataclass(frozen=True)
class LayoutAudit:
    """Serializable evidence about layout coverage and spatial balance."""

    plate_format: str
    well_count: int
    role_counts: dict[str, int]
    treatment_counts: dict[str, int]
    condition_replicate_counts: dict[str, int]
    edge_interior_counts: dict[str, dict[str, int]]
    row_role_counts: dict[str, dict[str, int]]
    column_role_counts: dict[str, dict[str, int]]
    condition_spatial_coverage: dict[str, dict[str, int]]
    duplicate_wells: tuple[str, ...]
    missing_wells: tuple[str, ...]
    potential_confounding: tuple[str, ...]
    issues: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        """Return whether structural and baseline-balance checks passed."""
        return not self.issues


def plate_rows(plate_format: PlateFormat) -> tuple[str, ...]:
    """Return row labels for a declared standard plate geometry."""
    return tuple(chr(ord("A") + index) for index in range(plate_format.rows))


def plate_columns(plate_format: PlateFormat) -> tuple[int, ...]:
    """Return numeric columns for a declared standard plate geometry."""
    return tuple(range(1, plate_format.columns + 1))


def enumerate_wells(plate_format: PlateFormat) -> pd.DataFrame:
    """Enumerate canonical wells and edge status in row-major order."""
    rows = plate_rows(plate_format)
    columns = plate_columns(plate_format)
    records = []
    for row_index, row in enumerate(rows):
        for column in columns:
            records.append(
                {
                    "well": f"{row}{column:02d}",
                    "row": row,
                    "column": column,
                    "row_index": row_index,
                    "column_index": column - 1,
                    "is_edge": (
                        row_index in {0, len(rows) - 1}
                        or column in {1, len(columns)}
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _condition_key(record: dict[str, Any]) -> str:
    role = record["well_role"]
    if role == WellRole.TREATMENT.value:
        return f"{record['treatment']}@{record['dose']:g}"
    if role == WellRole.EMPTY.value:
        return f"empty:{record['sample_id']}"
    return role


def _condition_records(config: GeneratorConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for replicate in range(1, config.negative_control_count + 1):
        records.append(
            {
                "sample_id": f"negative_control_{replicate:02d}",
                "well_role": WellRole.NEGATIVE_CONTROL.value,
                "treatment": "vehicle",
                "dose": None,
                "dose_unit": None,
                "replicate": replicate,
            }
        )
    for replicate in range(1, config.positive_control_count + 1):
        records.append(
            {
                "sample_id": f"positive_control_{replicate:02d}",
                "well_role": WellRole.POSITIVE_CONTROL.value,
                "treatment": "positive_control",
                "dose": None,
                "dose_unit": None,
                "replicate": replicate,
            }
        )
    for treatment in config.treatments:
        for dose in config.doses_micromolar:
            for replicate in range(1, config.replicates_per_condition + 1):
                records.append(
                    {
                        "sample_id": (
                            f"{treatment}_{dose:g}uM_rep{replicate:02d}"
                        ),
                        "well_role": WellRole.TREATMENT.value,
                        "treatment": treatment,
                        "dose": dose,
                        "dose_unit": "uM",
                        "replicate": replicate,
                    }
                )
    for empty_index in range(1, config.empty_count + 1):
        records.append(
            {
                "sample_id": f"empty_{empty_index:02d}",
                "well_role": WellRole.EMPTY.value,
                "treatment": None,
                "dose": None,
                "dose_unit": None,
                "replicate": None,
            }
        )

    for record in records:
        record["condition_key"] = _condition_key(record)
    return records


def _neighbor_pairs(wells: pd.DataFrame) -> np.ndarray:
    lookup = {
        (int(row.row_index), int(row.column_index)): index
        for index, row in wells.iterrows()
    }
    pairs = []
    for (row_index, column_index), index in lookup.items():
        for neighbor in (
            (row_index + 1, column_index),
            (row_index, column_index + 1),
        ):
            if neighbor in lookup:
                pairs.append((index, lookup[neighbor]))
    return np.asarray(pairs, dtype=int)


def _balance_score(
    permutation: np.ndarray,
    *,
    group_ids: np.ndarray,
    group_sizes: np.ndarray,
    group_roles: np.ndarray,
    treatment_ids: np.ndarray,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    edge_mask: np.ndarray,
    neighbor_pairs: np.ndarray,
    row_count: int,
    column_count: int,
) -> float:
    assigned_groups = group_ids[permutation]
    assigned_roles = group_roles[permutation]
    assigned_treatments = treatment_ids[permutation]
    score = 0.0

    for group_id, size in enumerate(group_sizes):
        if size <= 1:
            continue
        positions = assigned_groups == group_id
        row_bins = np.bincount(row_indices[positions], minlength=row_count)
        column_bins = np.bincount(
            column_indices[positions], minlength=column_count
        )
        score += 20.0 * float(np.square(row_bins).sum())
        score += 10.0 * float(np.square(column_bins).sum())
        expected_edge = size * float(edge_mask.mean())
        score += 30.0 * float((edge_mask[positions].sum() - expected_edge) ** 2)

    for role_id in np.unique(assigned_roles):
        positions = assigned_roles == role_id
        if positions.sum() <= 1:
            continue
        row_bins = np.bincount(row_indices[positions], minlength=row_count)
        score += 2.0 * float(np.square(row_bins - row_bins.mean()).sum())

    for treatment_id in np.unique(assigned_treatments):
        if treatment_id < 0:
            continue
        positions = assigned_treatments == treatment_id
        row_bins = np.bincount(row_indices[positions], minlength=row_count)
        column_bins = np.bincount(
            column_indices[positions], minlength=column_count
        )
        score += 3.0 * float(np.square(row_bins - row_bins.mean()).sum())
        score += float(np.square(column_bins - column_bins.mean()).sum())

    if neighbor_pairs.size:
        same_condition = (
            assigned_groups[neighbor_pairs[:, 0]]
            == assigned_groups[neighbor_pairs[:, 1]]
        )
        score += 30.0 * float(same_condition.sum())
    return score


def _allocate_records_to_rows(
    group_ids: np.ndarray,
    *,
    row_count: int,
    column_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Allocate records across rows before optimizing their columns."""
    assigned_rows = np.full(len(group_ids), -1, dtype=int)
    remaining_capacity = np.full(row_count, column_count, dtype=int)
    group_sizes = np.bincount(group_ids)

    for group_id in np.argsort(-group_sizes):
        record_indices = np.flatnonzero(group_ids == group_id)
        rng.shuffle(record_indices)
        used_by_group = np.zeros(row_count, dtype=int)
        for record_index in record_indices:
            candidates = np.flatnonzero(remaining_capacity > 0)
            tie_breakers = rng.random(len(candidates))
            selected = min(
                zip(candidates, tie_breakers, strict=True),
                key=lambda item: (
                    used_by_group[item[0]],
                    -remaining_capacity[item[0]],
                    item[1],
                ),
            )[0]
            assigned_rows[record_index] = selected
            used_by_group[selected] += 1
            remaining_capacity[selected] -= 1

    if (assigned_rows < 0).any() or remaining_capacity.any():
        raise RuntimeError("could not allocate all condition records across rows")
    return assigned_rows


def build_balanced_layout(
    config: GeneratorConfig,
    *,
    plate_id: str = "plate_01",
    candidate_count: int | None = None,
) -> pd.DataFrame:
    """Create a seeded layout selected for spatial balance.

    The default search is bounded and deterministic. Its output is audited by
    :func:`audit_layout`; callers should not infer balance from generation alone.
    """
    wells = enumerate_wells(config.plate_format)
    records = _condition_records(config)
    if len(records) != len(wells):
        raise ValueError("condition records do not match declared plate capacity")

    group_names = sorted({record["condition_key"] for record in records})
    group_lookup = {name: index for index, name in enumerate(group_names)}
    role_names = sorted({record["well_role"] for record in records})
    role_lookup = {name: index for index, name in enumerate(role_names)}
    treatment_lookup = {
        treatment: index for index, treatment in enumerate(config.treatments)
    }
    group_ids = np.asarray(
        [group_lookup[record["condition_key"]] for record in records], dtype=int
    )
    group_sizes = np.bincount(group_ids, minlength=len(group_names))
    group_roles = np.asarray(
        [role_lookup[record["well_role"]] for record in records], dtype=int
    )
    treatment_ids = np.asarray(
        [treatment_lookup.get(record["treatment"], -1) for record in records],
        dtype=int,
    )
    row_indices = wells["row_index"].to_numpy(dtype=int)
    column_indices = wells["column_index"].to_numpy(dtype=int)
    edge_mask = wells["is_edge"].to_numpy(dtype=bool)
    neighbor_pairs = _neighbor_pairs(wells)

    rng = np.random.default_rng(derive_child_seed(config.layout_seed, "layout"))
    assigned_rows = _allocate_records_to_rows(
        group_ids,
        row_count=config.plate_format.rows,
        column_count=config.plate_format.columns,
        rng=rng,
    )
    attempts = candidate_count or max(750, 150_000 // len(wells))
    best_permutation: np.ndarray | None = None
    best_score = float("inf")
    for _ in range(attempts):
        permutation = np.empty(len(records), dtype=int)
        for row_index in range(config.plate_format.rows):
            well_positions = np.flatnonzero(row_indices == row_index)
            row_records = np.flatnonzero(assigned_rows == row_index)
            permutation[well_positions] = rng.permutation(row_records)
        score = _balance_score(
            permutation,
            group_ids=group_ids,
            group_sizes=group_sizes,
            group_roles=group_roles,
            treatment_ids=treatment_ids,
            row_indices=row_indices,
            column_indices=column_indices,
            edge_mask=edge_mask,
            neighbor_pairs=neighbor_pairs,
            row_count=config.plate_format.rows,
            column_count=config.plate_format.columns,
        )
        if score < best_score:
            best_score = score
            best_permutation = permutation.copy()

    if best_permutation is None:  # pragma: no cover - attempts is always positive
        raise RuntimeError("layout search did not evaluate any candidates")

    assigned = pd.DataFrame.from_records(
        [records[index] for index in best_permutation]
    ).drop(columns="condition_key")
    layout = pd.concat(
        [
            pd.Series([plate_id] * len(wells), name="plate_id"),
            wells[["well", "row", "column"]].reset_index(drop=True),
            assigned.reset_index(drop=True),
        ],
        axis=1,
    )
    return layout[PLATE_MAP_COLUMNS]


def _integer_crosstab(
    frame: pd.DataFrame, row: str, column: str
) -> dict[str, dict[str, int]]:
    table = pd.crosstab(frame[row], frame[column])
    return {
        str(index): {str(key): int(value) for key, value in values.items()}
        for index, values in table.to_dict(orient="index").items()
    }


def audit_layout(layout: pd.DataFrame, config: GeneratorConfig) -> LayoutAudit:
    """Audit structural coverage and obvious spatial confounding."""
    expected_wells = enumerate_wells(config.plate_format)
    expected_set = set(expected_wells["well"])
    observed_set = set(layout["well"])
    duplicate_wells = tuple(
        sorted(layout.loc[layout["well"].duplicated(keep=False), "well"].unique())
    )
    missing_wells = tuple(sorted(expected_set - observed_set))

    coordinates = expected_wells.set_index("well")[["is_edge"]]
    audited = layout.join(coordinates, on="well", how="left")
    audited["region"] = np.where(audited["is_edge"], "edge", "interior")
    audited["condition_key"] = np.where(
        audited["well_role"] == WellRole.TREATMENT.value,
        audited["treatment"].astype(str)
        + "@"
        + audited["dose"].map(lambda value: f"{value:g}"),
        audited["well_role"],
    )

    role_counts = {
        str(key): int(value)
        for key, value in audited["well_role"].value_counts().sort_index().items()
    }
    treatment_rows = audited[
        audited["well_role"] == WellRole.TREATMENT.value
    ]
    treatment_counts = {
        str(key): int(value)
        for key, value in treatment_rows["treatment"]
        .value_counts()
        .sort_index()
        .items()
    }
    replicate_counts = {
        f"{treatment}@{dose:g}": int(len(group))
        for (treatment, dose), group in treatment_rows.groupby(
            ["treatment", "dose"], sort=True
        )
    }

    coverage: dict[str, dict[str, int]] = {}
    confounding: list[str] = []
    for condition, group in audited.groupby("condition_key", sort=True):
        if str(condition).startswith("empty"):
            continue
        counts = {
            "wells": int(len(group)),
            "rows": int(group["row"].nunique()),
            "columns": int(group["column"].nunique()),
            "edge": int(group["is_edge"].sum()),
            "interior": int((~group["is_edge"].astype(bool)).sum()),
        }
        coverage[str(condition)] = counts
        if counts["wells"] > 1 and counts["rows"] == 1:
            confounding.append(f"{condition} is confined to one row")
        if counts["wells"] > 1 and counts["columns"] == 1:
            confounding.append(f"{condition} is confined to one column")
        if counts["wells"] > 1 and min(counts["edge"], counts["interior"]) == 0:
            confounding.append(f"{condition} is confined to one plate region")

    issues: list[str] = []
    if len(layout) != config.plate_format.capacity:
        issues.append(
            f"expected {config.plate_format.capacity} rows, observed {len(layout)}"
        )
    if duplicate_wells:
        issues.append("duplicate wells are present")
    if missing_wells:
        issues.append("expected wells are missing")
    if observed_set - expected_set:
        issues.append("wells outside the declared geometry are present")
    if layout["well"].isna().any():
        issues.append("null well identifiers are present")
    if confounding:
        issues.append("one or more conditions have obvious spatial confounding")

    expected_roles = {
        WellRole.NEGATIVE_CONTROL.value: config.negative_control_count,
        WellRole.POSITIVE_CONTROL.value: config.positive_control_count,
        WellRole.TREATMENT.value: (
            len(config.treatments)
            * len(config.doses_micromolar)
            * config.replicates_per_condition
        ),
    }
    if config.empty_count:
        expected_roles[WellRole.EMPTY.value] = config.empty_count
    if role_counts != expected_roles:
        issues.append("well-role counts disagree with generator configuration")

    return LayoutAudit(
        plate_format=config.plate_format.value,
        well_count=int(len(layout)),
        role_counts=role_counts,
        treatment_counts=treatment_counts,
        condition_replicate_counts=replicate_counts,
        edge_interior_counts=_integer_crosstab(
            audited, "well_role", "region"
        ),
        row_role_counts=_integer_crosstab(audited, "row", "well_role"),
        column_role_counts=_integer_crosstab(
            audited, "column", "well_role"
        ),
        condition_spatial_coverage=coverage,
        duplicate_wells=duplicate_wells,
        missing_wells=missing_wells,
        potential_confounding=tuple(confounding),
        issues=tuple(issues),
    )


def render_layout_grid(layout: pd.DataFrame) -> str:
    """Render a compact Markdown grid for human review."""
    labels = layout.copy()
    treatments = sorted(
        treatment
        for treatment in labels["treatment"].dropna().unique()
        if treatment not in {"vehicle", "positive_control"}
    )
    treatment_abbreviations = {
        treatment: f"T{index + 1}"
        for index, treatment in enumerate(treatments)
    }

    def label(row: pd.Series) -> str:
        if row["well_role"] == WellRole.NEGATIVE_CONTROL.value:
            return "NC"
        if row["well_role"] == WellRole.POSITIVE_CONTROL.value:
            return "PC"
        if row["well_role"] == WellRole.EMPTY.value:
            return "EM"
        dose_index = sorted(labels["dose"].dropna().unique()).index(row["dose"]) + 1
        return f"{treatment_abbreviations[row['treatment']]}D{dose_index}"

    labels["label"] = labels.apply(label, axis=1)
    grid = labels.pivot(index="row", columns="column", values="label")
    header = "| Row | " + " | ".join(str(column) for column in grid.columns) + " |"
    separator = "|---|" + "---|" * len(grid.columns)
    body = [
        f"| {row} | " + " | ".join(str(value) for value in values) + " |"
        for row, values in grid.iterrows()
    ]
    return "\n".join([header, separator, *body])
