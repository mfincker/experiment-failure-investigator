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
- Initial target: fourteen cases, one variant per failure-mode and difficulty combination

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
- Empty or intentionally unused wells
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

Step 3 defines and audits the exact deterministic well assignment below. The layout-confounding injector may replace the balanced layout with a deliberately confounded layout.

### Frozen baseline layout

The initial balanced layout uses layout seed `20260904`. `NC` and `PC` denote negative and positive controls. `T1` is the reference treatment, `T2` is the test treatment, and `D1` through `D8` follow the ascending dose series defined above.

| Row | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | T1D7 | NC | PC | T1D6 | T2D3 | T2D1 | T1D3 | T2D2 | T2D7 | T2D5 | T2D8 | T1D4 |
| B | T1D2 | PC | T1D5 | T1D7 | T2D2 | T1D1 | NC | T1D8 | T2D6 | T1D3 | T2D7 | T2D4 |
| C | T2D3 | T1D7 | T1D2 | NC | PC | T2D4 | T2D1 | T1D4 | T2D7 | T1D5 | T1D1 | T2D6 |
| D | T1D8 | T1D1 | NC | T1D2 | T1D5 | T2D5 | PC | T2D3 | T2D8 | T2D6 | T1D6 | T2D1 |
| E | T2D8 | T2D7 | T2D5 | T1D4 | T2D6 | T1D3 | T1D5 | PC | T2D2 | T2D3 | NC | T1D7 |
| F | T1D6 | T1D3 | T2D3 | T2D8 | T1D7 | T2D4 | PC | T1D8 | T2D5 | NC | T1D1 | T1D5 |
| G | T1D1 | T1D6 | T1D2 | PC | T2D8 | NC | T2D1 | T2D5 | T1D8 | T1D4 | T2D4 | T2D2 |
| H | T1D8 | T1D2 | T2D7 | T2D2 | T1D6 | T2D1 | T1D3 | T2D6 | PC | T2D4 | T1D4 | NC |

The independent audit confirms:

- 96 unique wells with complete `A01`–`H12` coverage
- 8 negative controls, 8 positive controls, and 80 treatment wells
- Both control types represented in every row
- Every treatment-dose replicate set represented in five distinct rows, at least three columns, and both edge and interior regions
- No condition confined to one row, one column, or one plate region

### Current fixed-layout limitation

The initial fourteen-case benchmark slice is intentionally a fixed-layout development fixture. All cases other than the two `layout_confounding` cases reuse the frozen baseline map above; their case root seeds vary signal noise and failure injection, not well assignment. The `layout_confounding` cases are deliberate exceptions because their planted mechanism requires a different, confounded map.

Results on this slice measure performance within the baseline layout. They do **not** demonstrate that an investigator generalizes to unseen well assignments, alternative control allocations, different replicate counts, missing reference treatments, or 384-well plates. Repeated well positions may become an unintended shortcut for prompts, tools, or models even when the layout is not described explicitly.

Any evaluation report based on this slice must:

- Label the results as fixed-layout or in-layout results.
- Report the number of distinct layout fingerprints represented.
- Avoid claims of layout robustness or generalization.
- Keep layout-confounding performance separate from unseen-layout generalization; detecting a deliberately confounded map is a different capability.

Before making a layout-generalization claim, expand the benchmark with multiple balanced layout seeds and evaluate on held-out layouts that were not used for prompt, tool, or threshold development. Report fixed-layout development results separately from held-out-layout results, stratify by plate format and layout family, and verify automatically that development and held-out layout fingerprints do not overlap.

## Clean signal model

Signals are normalized to the negative-control response rather than emitted in instrument-specific units.

Default control means:

- Negative control: `1.00`
- Positive control: `0.15`

An `empty` well means a physically empty, intentionally unused position. It is not an assay blank containing medium, reagent, or another background-control material. Empty wells have no biological expected signal, and the initial benchmark profile contains none. Until a future instrument-readout policy defines whether empty positions produce a finite value, a missing value, or no measurement row, clean signal generation rejects layouts containing empty wells rather than inventing a value.

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

The generator does not clip observed values to the latent curve bounds. Clipping would change the noise distribution and can hide behavior near an assay boundary; finite out-of-range values remain available for downstream QC.

The initial default noise standard deviations are:

- Obvious cases: `0.03`
- Noisy cases: `0.07`

These values are benchmark parameters. They are not assay acceptance criteria. The generator records latent means separately during construction but does not include them in investigator-visible measurement files.

## Randomness and reproducibility

Every case has one unsigned 32-bit root seed. The generator derives stable named child seeds for:

- Baseline well noise
- Failure injection
- Optional missingness added in later benchmark versions

The balanced MVP profile has a separate, explicit layout seed shared across ordinary cases. This keeps the plate map fixed while case root seeds vary the signal noise and failure injection. A layout-confounding case uses an explicitly different layout configuration rather than acquiring a new map accidentally from its noise seed.

Noise is assigned after canonical sorting by plate and well, so reordering input rows does not change which random draw belongs to a well.

Adding a new random operation must not silently change existing streams. Named child seeds are recorded in private ground truth. With the same benchmark version, configuration, root seed, and layout seed, canonical serialized case files must be byte-reproducible.

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

Allowed benchmark well roles are `negative_control`, `positive_control`, `treatment`, and `empty`. Control and empty rows may have null dose and replicate fields when those concepts do not apply. An assay blank is not currently represented as a separate well role.

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

Numerical injectors preserve the clean expected signal and baseline-noise columns in private generation data and add a per-well `injected_effect`. Their public measurement satisfies:

```text
raw_signal = expected_signal + baseline_noise + injected_effect
```

Layout confounding is different: it rearranges experimental assignments and their expected signals while leaving each physical well's baseline-noise draw in place. Every injector returns a structured private record containing the mechanism, typed parameters, stable child seed, and affected-well count. None of these private fields is investigator-visible.

The initial magnitudes below are provisional benchmark calibration values. They are chosen to make deterministic mechanism checks possible and will be reviewed with Altair plots in Step 7. They are not laboratory acceptance thresholds.

### 1. Edge effect

Mechanism:

- Shift signals on an explicitly selected set of one to four plate sides: top,
  bottom, left, and/or right. Corners are affected once when either adjoining
  selected side is affected.
- Do not change well roles, treatments, or doses.
- Initial increase: `0.25` for obvious cases and `0.12` for noisy cases.

Expected observable evidence:

- A residual difference between edge and interior wells after accounting for experimental condition.
- A one-, two-, three-, or four-sided boundary pattern in the plate heatmap.

Important alternatives:

- Treatment or control placement concentrated at the edge.
- Row- or column-specific drift.

### 2. Pipetting drift

Mechanism:

- Apply a monotonic gradient using a simulated row-wise or column-wise traversal.
- Store the simulated traversal only in private generator truth.
- Initial peak-to-peak change: `0.30` for obvious cases and `0.15` for noisy cases.

Expected observable evidence:

- A row-wise or column-wise residual gradient after accounting for well role, treatment, and dose.

Interpretation limit:

- Without observed run-order metadata, the pattern is only consistent with pipetting or timing drift.
- The investigator cannot confirm the physical dispensing sequence and must retain other spatial explanations.

Useful follow-up:

- Request instrument or liquid-handler logs if they exist.
- Repeat with a randomized or reversed layout that separates condition from the suspected gradient.

### 3. Transient tip clog

Mechanism:

- Simulate grouped dispensing with a private traversal and configurable group size.
- Reduce signal from one or more selected tip channels for a contiguous span of
  dispense groups, then return subsequent groups to baseline.
- Initial obvious cases use an eight-channel column-wise traversal, one affected
  channel, and several consecutive affected groups.

Expected observable evidence:

- A localized sequence of affected wells consistent with one channel across
  several dispensing groups.
- Recovery in later groups rather than a gradient spanning the entire plate.

Interpretation limit:

- Without liquid-handler logs, the investigator should describe the pattern as
  consistent with a transient dispensing fault rather than prove a clogged tip.
- Actual head geometry, grouping, reload behavior, and channel-to-well mapping
  vary by protocol and remain private simulation assumptions in this MVP.

Useful follow-up:

- Inspect liquid-handler pressure/error logs and tip-change or reload events.
- Repeat the affected conditions at randomized positions or with fresh tips.

### 4. Plate-layout confounding

Mechanism:

- Replace the balanced layout with one where treatment or dose is systematically associated with position.
- Initially order assignments by treatment and dose; both variants use the same confounded design while their baseline noise differs.

Expected observable evidence:

- Treatment or dose is inseparable from row, column, or edge status.
- A spatial association exists, but the available plate cannot identify whether position or treatment caused it.

Correct behavior:

- Report non-identifiability rather than claiming a spatial artifact as proven.

### 5. Failed or weak controls

Mechanism:

- Move the positive-control mean toward the negative-control mean while leaving labels unchanged.
- Initially remove `80%` of control separation in obvious cases and `55%` in noisy cases.

Expected observable evidence:

- Reduced separation between positive and negative controls.
- A degraded deterministic control-quality statistic once that tool is implemented.

Important alternatives:

- Excessive variability affecting all wells.
- Incorrect control annotations.

### 6. Batch shift

Mechanism:

- Generate at least two otherwise comparable plates or batches.
- Preserve each plate's observed negative-control mean as its response anchor.
- Scale one plate's distance from that anchor and expose the plate or batch label
  in metadata: `shifted = anchor + scale_factor × (original - anchor)`.
- Initial response-scale factors: `0.70` for obvious cases and `0.85` for noisy
  cases. These compress the assay window without moving the negative-control mean.

Expected observable evidence:

- A between-batch difference among like-for-like positive controls or
  treatment-dose conditions while negative controls remain anchored.
- The response-scale change persists after accounting for condition composition.

Important alternatives:

- Different layouts or condition mixes between batches.
- A batch-specific spatial artifact.

Interpretation limit:

- A purely multiplicative global change applied before negative-control
  normalization would cancel and cannot be recovered from normalized values.
- This benchmark case therefore represents a batch-specific change in assay
  dynamic range, not a uniform additive displacement of normalized values.

### 7. True biological non-response

Mechanism:

- Flatten the test-treatment curve while leaving positive controls and the reference-treatment curve responsive.
- Initially flatten the test treatment at normalized expected signal `1.00`; both variants use the same flat expectation while their baseline noise differs.

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
transient_tip_clog_obvious
transient_tip_clog_noisy
layout_confounding_obvious
layout_confounding_noisy
weak_controls_obvious
weak_controls_noisy
batch_shift_obvious
batch_shift_noisy
true_non_response_obvious
true_non_response_noisy
```

## Human-inspection charts

Step 7 produces four Altair views directly from in-memory plate maps and measurements:

- A raw-signal plate heatmap with fixed row and column orientation.
- A condition-centered residual heatmap that subtracts the observed mean for each well-role or treatment-dose group.
- A log-dose response chart with replicate points, condition means, standard-deviation intervals, and control reference bands.
- A control-QC chart with individual control wells, means, and standard-deviation intervals by plate.
- An eight-panel comparison grid with clean and injected rows and the four inspection views as columns.

Clean and injected versions of a case use shared raw-signal and residual domains so apparent differences cannot be created by automatic rescaling. Chart titles record the case ID, scenario, variant, and root seed. Plot data is restricted to investigator-observable fields plus derived summaries; latent expectations, baseline-noise draws, injected effects, and private injection parameters are not included.

Condition centering is a diagnostic transformation, not a root-cause label. It can reveal spatial residual structure after accounting for the reported condition, but it does not prove a physical mechanism. It may also remove patterns that are inseparable from condition assignment, so layout-confounding cases must be reviewed with the raw heatmap and plate-map audit rather than the residual heatmap alone. For multi-plate cases, condition means are calculated across plates so a batch-specific response-scale change remains visible.

Temporary review plots are written under ignored `artifacts/`. Step 8 will combine
the serializer and plot exporter so finalized plots are written under
`cases/<case_id>/plots/`.

## Validation requirements

Before serialization, generation must reject:

- Invalid or duplicate well identifiers
- Missing or extra plate-map rows relative to measurements
- Unsupported well roles, failure modes, or variants
- Negative noise values
- Non-positive, duplicated, or unordered doses
- A generator configuration whose assigned controls, treatments, and empty wells do not match its declared plate capacity
- Absolute artifact paths or paths escaping the case directory
- Missing or malformed SHA-256 hashes in a finalized manifest

## Serialization and integrity

Case tables are sorted by `plate_id` and canonical `well` before writing. CSV
files use fixed column order, `\n` line endings, empty fields for null values,
and a stable significant-digit format for floating-point values. JSON files use
sorted keys, two-space indentation, UTF-8, and a final newline. Protocol and
problem-statement line endings are normalized and blank content is rejected.

The writer validates all in-memory content before creating a staging directory,
serializes the five investigator-visible inputs and requested plots, computes
their SHA-256 hashes,
builds the evaluation-only manifest, and loads the staged case through the same
hash and content validator used for existing cases. Only then is the staging
directory atomically renamed to its final case ID. Existing case directories are
rejected unless the caller explicitly opts into replacement with `force=True`.

The manifest is deliberately excluded from its own file map. Artifact references
must be unique relative paths within the case directory; missing files, hash
mismatches, and symlinks that resolve outside the case directory are rejected.
Public measurement and plate-map tables must contain exactly their documented
columns, preventing private latent signals and injected-effect truth from leaking
into investigator-visible inputs.

Each case metadata file records a layout fingerprint derived from plate format and
the canonically sorted `well`, `well_role`, `treatment`, and `dose` assignments.
It excludes measurements, case and plate identifiers, seeds, sample IDs, and
interchangeable replicate labels. Repeated identical layouts on a multi-plate case
therefore have the same fingerprint as a single instance of that layout.

Step 6 implements these cross-row and cross-file checks at both write and load
time. Step 2 defines the typed configuration and manifest boundaries they consume.

## Anti-brittleness requirements

These requirements apply throughout implementation:

- Benchmark defaults remain configuration, never prompt instructions or tool constants.
- Plate boundaries come from declared geometry, not assumptions about `H12`.
- Control summaries group the roles actually present and report absent controls explicitly.
- Dose-response fitting operates per eligible treatment and reports why a curve cannot be fit.
- Replicate calculations use observed group sizes and retain unequal or singleton groups.
- Missing wells remain missing; they are not silently converted to empty wells or zero signal.
- Reference-dependent evidence is optional and cannot be required for a terminal decision.
- Spatial tools support both 8×12 and 16×24 geometries.
- Agent prompts receive a case-specific capability summary rather than a description of the MVP layout.
- Evaluation must include multiple balanced layouts and format-shift cases before successful performance on the frozen baseline profile is described as generality.
- Evaluation artifacts must identify their layout split, unique layout count, and plate formats; fixed-layout results must be labeled explicitly.
- Held-out layout fingerprints must not overlap layouts used to develop prompts, tools, thresholds, or deterministic diagnostics.

## Week 1 review gate

Before freezing the generated benchmark, review:

- Whether the two-treatment design supports the intended scientific comparisons.
- Whether the default control allocation and dose series are plausible for the synthetic MVP.
- The fourteen clean/injected comparison grids and the limitations recorded in
  [`week_1_review.md`](week_1_review.md).
- Whether hidden ground truth is cleanly separated from investigator-visible inputs.
- Whether each failure mode has observable evidence and appropriately stated identification limits.
- Whether any parameter is accidentally presented as a real-world acceptance threshold.
