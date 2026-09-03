# Synthetic Assay Benchmark Specification

## Purpose

This benchmark provides reproducible cases for evaluating whether an experiment-failure investigator can distinguish assay-quality problems from true biological non-response. It defines the synthetic data-generating process, the information visible to the investigator, and the private labels used only by the evaluation harness.

The benchmark is not a source of universal laboratory acceptance thresholds. Means, noise levels, curve parameters, and injected-effect magnitudes are versioned simulation parameters chosen to create controlled test cases.

## Version and scope

- Benchmark version: `1.0.0`
- Manifest schema version: `1.0.0`
- Assay type: endpoint cell-viability dose response
- Initial benchmark profile: 96-well plate
- Response direction: lower signal represents stronger inhibition
- Initial scope: one planted mechanism per case
- Initial case variants: `obvious` and `noisy`
- Initial target: twelve cases, one variant per failure-mode and difficulty combination

Mixed-cause, clean-negative, and deliberately ambiguous cases are deferred until the initial cases pass deterministic and visual review.

## Unit of observation

Each measurement represents the endpoint signal from one well on one plate. A case may contain one plate or multiple plates when the planted mechanism requires a comparison, such as a batch shift.

The unique measurement key is:

```text
(case_id, plate_id, well)
```

The initial 96-well profile contains rows `A` through `H` and columns `1` through `12`. A 384-well plate contains rows `A` through `P` and columns `1` through `24`. Canonical well identifiers use a zero-padded column, such as `A01`, `H12`, and `P24`.

## MVP profile versus application contract

The `8/8/40/40` design below is a versioned benchmark profile, not an assumption the investigator may make about incoming experiments. It provides a controlled starting point for testing the initial failure mechanisms.

Realistic investigator inputs may differ in all of the following ways:

- 96- or 384-well plate geometry
- Different numbers or types of controls
- One, two, or many treatments
- No reference treatment
- Different dose series between treatments
- Unequal or missing replicates
- Blank or intentionally unused wells
- Multiple plates without a batch failure
- Missing optional metadata

The eventual case loader will normalize each supplied plate while preserving its actual design. It will expose a typed design summary describing available controls, factors, doses, replicate counts, plate geometry, and missing capabilities. Agents and deterministic tools must consume that summary or the normalized rows; they must never hard-code the MVP profile.

When evidence cannot be computed for a design—for example, a reference-treatment comparison when no reference exists—the tool must return a structured `not_applicable` or `insufficient_data` result. The agent must treat that as an evidence limitation rather than an input failure.

## Baseline MVP plate design

Each clean plate contains 96 wells:

| Well role | Design | Wells |
|---|---:|---:|
| Negative control | Vehicle only | 8 |
| Positive control | Known strong inhibitory control | 8 |
| Reference treatment | 8 doses × 5 replicates | 40 |
| Test treatment | 8 doses × 5 replicates | 40 |
| Total |  | 96 |

The reference and test treatments are both responsive in the initial clean profile. Retaining a responsive reference treatment helps distinguish a test-treatment-specific non-response from global assay failure in these cases, but reference treatment is not required by the general generator contract or future investigator input contract.

The default dose series is expressed in micromolar:

```text
0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0
```

### Balance requirements

The clean layout must distribute controls and treatment-dose replicates across rows, columns, edge wells, and interior wells. In particular:

- Controls must not be confined to one row, column, corner, or plate region.
- Replicates of a treatment-dose condition must not all be adjacent.
- Neither treatment identity nor dose may map perfectly to row, column, or edge status.
- File row order carries no experimental meaning.
- Dispensing order is not required and must not be inferred from file order.

Step 3 will define and audit the exact deterministic well assignment. The layout-confounding injector may replace the balanced layout with a deliberately confounded layout.

## Clean signal model

Signals are normalized to the negative-control response rather than emitted in instrument-specific units.

Default control means:

- Negative control: `1.00`
- Positive control: `0.15`

For a treatment well at dose `d > 0`, the latent mean follows a decreasing four-parameter logistic curve:

```text
expected_signal(d) = bottom + (top - bottom) / (1 + (d / IC50) ** hill_slope)
```

Default clean curve parameters:

| Treatment | Top | Bottom | IC50 (µM) | Hill slope |
|---|---:|---:|---:|---:|
| Reference treatment | 1.00 | 0.15 | 0.30 | 1.20 |
| Test treatment | 1.00 | 0.15 | 1.00 | 1.20 |

Observed signal is the latent mean plus seeded well-level noise:

```text
observed_signal = expected_signal + Normal(0, noise_sd)
```

The initial default noise standard deviations are:

- Obvious cases: `0.03`
- Noisy cases: `0.07`

These values are benchmark parameters. They are not assay acceptance criteria. The generator records latent means separately during construction but does not include them in investigator-visible measurement files.

## Randomness and reproducibility

Every case has one unsigned 32-bit root seed. The generator derives stable named child seeds for:

- Layout assignment
- Baseline well noise
- Failure injection
- Optional missingness added in later benchmark versions

Adding a new random operation must not silently change existing streams. Named child seeds are recorded in private ground truth. With the same benchmark version, configuration, and root seed, canonical serialized case files must be byte-reproducible.

## Investigator-visible inputs

The application under evaluation may receive only the inputs a scientist could reasonably supply:

1. `measurements.csv`
2. `plate_map.csv`
3. `metadata.json`
4. `protocol.md`
5. `problem_statement.txt`

### `measurements.csv`

Required columns:

- `case_id`
- `plate_id`
- `well`
- `raw_signal`

### `plate_map.csv`

Required columns:

- `plate_id`
- `well`
- `row`
- `column`
- `sample_id`
- `well_role`
- `treatment`
- `dose`
- `dose_unit`
- `replicate`

Allowed benchmark well roles are `negative_control`, `positive_control`, `treatment`, and `blank`. Control and blank rows may have null dose and replicate fields when those concepts do not apply.

Dispensing order is deliberately absent from the required schema. If trustworthy liquid-handler or acquisition logs become available in a future case, they may be included as optional metadata with explicit provenance. Their absence must not invalidate an investigation.

### `metadata.json`

Metadata identifies the assay, plate, batch, operator label, run date, instrument label, and any other available case-level factors. Values are synthetic. Missing optional metadata remains missing rather than being inferred.

### Protocol and problem statement

The protocol contains a generic assay description and expected response direction. The problem statement is a short human observation, such as “the positive controls were weaker than expected.” Neither file contains the planted label.

## Evaluation-only manifest

`manifest.json` is consumed by generation, validation, and evaluation code. It is not an investigator input because it contains the answer key.

The manifest records:

- Case and schema versions
- Case identifier, variant, and root seed
- Assay type and response direction
- Planted failure mode
- Public generation parameters needed for reproduction
- Private injection parameters and named child seeds
- Expected discriminating evidence
- Plausible confounders
- Relative artifact paths and SHA-256 hashes

The manifest itself is not included in its artifact hash map, avoiding a self-referential hash.

## Planted mechanisms

### 1. Edge effect

Mechanism:

- Shift signals in boundary wells by a configured signed magnitude.
- Do not change well roles, treatments, or doses.

Expected observable evidence:

- A residual difference between edge and interior wells after accounting for experimental condition.
- A perimeter-shaped pattern in the plate heatmap.

Important alternatives:

- Treatment or control placement concentrated at the edge.
- Row- or column-specific drift.

### 2. Pipetting drift

Mechanism:

- Apply a monotonic gradient using a simulated row-wise or column-wise traversal.
- Store the simulated traversal only in private generator truth.

Expected observable evidence:

- A row-wise or column-wise residual gradient after accounting for well role, treatment, and dose.

Interpretation limit:

- Without observed run-order metadata, the pattern is only consistent with pipetting or timing drift.
- The investigator cannot confirm the physical dispensing sequence and must retain other spatial explanations.

Useful follow-up:

- Request instrument or liquid-handler logs if they exist.
- Repeat with a randomized or reversed layout that separates condition from the suspected gradient.

### 3. Plate-layout confounding

Mechanism:

- Replace the balanced layout with one where treatment or dose is systematically associated with position.

Expected observable evidence:

- Treatment or dose is inseparable from row, column, or edge status.
- A spatial association exists, but the available plate cannot identify whether position or treatment caused it.

Correct behavior:

- Report non-identifiability rather than claiming a spatial artifact as proven.

### 4. Failed or weak controls

Mechanism:

- Move the positive-control mean toward the negative-control mean while leaving labels unchanged.

Expected observable evidence:

- Reduced separation between positive and negative controls.
- A degraded deterministic control-quality statistic once that tool is implemented.

Important alternatives:

- Excessive variability affecting all wells.
- Incorrect control annotations.

### 5. Batch shift

Mechanism:

- Generate at least two otherwise comparable plates or batches.
- Apply a configured shift to one batch and expose the batch label in metadata.

Expected observable evidence:

- A between-batch difference among like-for-like controls or treatment-dose conditions.
- The shift persists after accounting for condition composition.

Important alternatives:

- Different layouts or condition mixes between batches.
- A batch-specific spatial artifact.

### 6. True biological non-response

Mechanism:

- Flatten the test-treatment curve while leaving positive controls and the reference-treatment curve responsive.

Expected observable evidence:

- Healthy control separation.
- A responsive reference treatment.
- Little or no dose-dependent response for the test treatment.

Important alternatives:

- Test-treatment preparation error.
- Treatment-specific plate-map or dose annotation error.

## Difficulty variants

An `obvious` case uses lower baseline noise and a larger planted effect. A `noisy` case uses higher baseline noise and a smaller planted effect while retaining the same single causal mechanism.

A noisy case must not acquire an unlabeled second failure. Difficulty parameters will be calibrated during generator implementation using deterministic summaries and visual review rather than assumed to be adequate from configuration alone.

## Initial case identifiers

```text
edge_effect_obvious
edge_effect_noisy
pipetting_drift_obvious
pipetting_drift_noisy
layout_confounding_obvious
layout_confounding_noisy
weak_controls_obvious
weak_controls_noisy
batch_shift_obvious
batch_shift_noisy
true_non_response_obvious
true_non_response_noisy
```

## Validation requirements

Before serialization, generation must reject:

- Invalid or duplicate well identifiers
- Missing or extra plate-map rows relative to measurements
- Unsupported well roles, failure modes, or variants
- Negative noise values
- Non-positive, duplicated, or unordered doses
- A generator configuration whose assigned controls, treatments, and blanks do not match its declared plate capacity
- Absolute artifact paths or paths escaping the case directory
- Missing or malformed SHA-256 hashes in a finalized manifest

Cross-row and cross-file validation will be implemented alongside serialization in Step 6. Step 2 defines the typed configuration and manifest boundaries that those validators will consume.

## Anti-brittleness requirements

These requirements apply throughout implementation:

- Benchmark defaults remain configuration, never prompt instructions or tool constants.
- Plate boundaries come from declared geometry, not assumptions about `H12`.
- Control summaries group the roles actually present and report absent controls explicitly.
- Dose-response fitting operates per eligible treatment and reports why a curve cannot be fit.
- Replicate calculations use observed group sizes and retain unequal or singleton groups.
- Missing wells remain missing; they are not silently converted to blank wells or zero signal.
- Reference-dependent evidence is optional and cannot be required for a terminal decision.
- Spatial tools support both 8×12 and 16×24 geometries.
- Agent prompts receive a case-specific capability summary rather than a description of the MVP layout.
- Evaluation must eventually include format-shift cases so successful performance on the balanced profile is not mistaken for generality.

## Review gate

Before implementing the numerical generator, review:

- Whether the two-treatment design supports the intended scientific comparisons.
- Whether the default control allocation and dose series are plausible for the synthetic MVP.
- Whether hidden ground truth is cleanly separated from investigator-visible inputs.
- Whether each failure mode has observable evidence and appropriately stated identification limits.
- Whether any parameter is accidentally presented as a real-world acceptance threshold.
