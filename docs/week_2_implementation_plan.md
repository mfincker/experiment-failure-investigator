# Week 2 Implementation Plan: Deterministic QC and Evidence Contracts

## Week 2 outcome

At the end of Week 2, the CLI will load any valid Week 1 benchmark case and
produce a deterministic QC report in JSON and Markdown. Every scientific value
will come from an independently tested tool result with a stable evidence ID and
input provenance. The report path will never expose the evaluation manifest,
planted failure label, latent signals, or injector parameters.

This week uses no language-model calls. Ollama may remain stopped, no remote API
credential is required, and prompt development is deliberately deferred until the
scientific evidence layer is stable.

## Scope decisions

### Reuse from Week 1

- Reuse benchmark serialization, hash validation, public metadata, plate maps,
  measurements, and plotting functions.
- Reuse `CaseManifest` only inside benchmark validation and evaluation code.
- Do not pass a loaded manifest or `GroundTruth` object into an analysis tool or
  report builder.
- Wrap the existing Altair heatmap implementation rather than creating a second
  plotting system.

### General application contract versus benchmark validation

The Week 1 benchmark validator correctly requires complete, internally consistent
generated cases. Week 2 adds a separate investigator-facing design summary that
is derived from observed public inputs. Analysis tools must consume that summary
and must not assume:

- exactly 96 wells;
- eight negative or positive controls;
- two treatments or a reference treatment;
- a shared dose series;
- five replicates per condition;
- complete observations; or
- multiple plates.

The initial CLI operates on valid Week 1 cases, but unit tests will exercise
reduced controls, unequal replicates, absent reference treatments, missing values,
and 384-well geometry. Unsupported analyses return a typed status rather than
crashing or silently inventing defaults.

### Multi-plate assumption

Week 2 supports a narrow multi-plate design: each plate is a complete replicate of
the same experimental design, with the same well roles, treatments, doses, and
conditions available for like-for-like comparison. The current two-plate benchmark
cases follow this design.

Week 2 does not support experiments in which biological condition replicates are
distributed across plates, so that an individual plate contains only part of the
design. The design summary must expose condition coverage by plate. If coverage is
not consistent with complete replicate plates, cross-plate and pooled diagnostics
must return a structured `insufficient_data` result with an unsupported-design
limitation; they must not treat plates as interchangeable or pool the observations.
Supporting split-replicate and incomplete-block multi-plate designs is a backlog
item.

### Statistics and interpretation policy

- Tools return measurements, estimates, effect sizes, uncertainty, and explicit
  limitations. They do not claim a physical root cause.
- Control-separation statistics may include standard assay-window metrics, but no
  literature threshold is presented as universally valid.
- Any rule-based flags used by the non-agentic baseline live in a separate,
  versioned synthetic-benchmark heuristic configuration.
- Effect sizes and data sufficiency take priority over isolated p-values.
- Multiple comparisons, small groups, failed fits, and near-zero denominators must
  produce warnings or structured insufficient-data results.

## Proposed repository additions

```text
src/experiment_failure_investigator/
├── analysis/
│   ├── __init__.py
│   ├── contracts.py
│   ├── design.py
│   ├── controls.py
│   ├── replicates.py
│   ├── spatial.py
│   ├── dose_response.py
│   ├── batches.py
│   ├── missingness.py
│   └── heatmaps.py
├── reporting/
│   ├── __init__.py
│   └── baseline.py
└── cli.py
tests/
├── unit/analysis/
├── unit/reporting/
└── integration/test_baseline_qc.py
```

Keep modules focused on one scientific responsibility. Shared contracts belong in
`analysis/contracts.py`; tools must not import the benchmark registry or injector
modules.

## Core contracts

### `InvestigatorCase`

Contains only:

- measurements;
- plate map;
- public metadata;
- protocol;
- problem statement;
- verified public artifact hashes; and
- derived `DesignSummary`.

The benchmark adapter may read `manifest.json` to verify hashes, but it must return
this restricted object. A regression test will search its serialized form and the
final report for planted labels, injection parameters, child seeds, and private
latent columns.

### `DesignSummary`

Derive and record:

- plate IDs and declared or inferred geometry;
- observed and expected well counts;
- available well roles and control counts per plate;
- treatments and dose series per treatment;
- replicate-count distributions per condition;
- missing wells and missing/non-finite measurements;
- public batch, operator, run-date, and instrument fields;
- condition coverage by plate and whether plates appear to be complete design
  replicates;
- whether conditions are spatially distributed or structurally confounded; and
- capabilities such as `has_negative_controls`, `has_positive_controls`,
  `has_replicates`, `has_dose_series`, and `has_multiple_plates`.

Capabilities describe what can be computed; they are not quality judgments.

### Tool result and evidence contracts

Every tool returns a typed result with:

- `status`: `success`, `not_applicable`, `insufficient_data`, or `error`;
- tool name and semantic tool version;
- deterministic parameters and analyzed scope;
- zero or more typed evidence records;
- warnings and limitations;
- public input artifact hashes; and
- optional stable derived tables or plot paths.

Each evidence record includes a stable ID, metric name, value or structured value,
units when applicable, comparison scope, sample counts, and a plain-language
description that does not overstate causality.

Evidence IDs will be derived from the tool name, version, case ID, plate scope,
condition scope, and metric—not from execution order or runtime. Runtime and wall
clock timestamps belong in a separate telemetry envelope and are excluded from
scientific-result equality, evidence IDs, and golden-file comparisons.

## Expected diagnostic coverage

| Benchmark mechanism | Primary deterministic evidence | Necessary caution |
|---|---|---|
| Edge effect | Condition-centered edge/interior effect size and spatial summary | Position dependence does not prove a physical edge mechanism |
| Progressive pipetting drift | Row/column residual trend and monotonic association | Dispense order is unavailable; timing and temperature remain alternatives |
| Transient tip clog | Localized residual outliers or adjacent residual run | A local pattern cannot prove tip identity or reload timing |
| Layout confounding | Condition-by-position coverage and confounding diagnostics | The supplied plate cannot separate condition from position |
| Weak controls | Control counts, means, variability, separation, and assay-window metric | Thresholds are assay-specific; label errors remain possible |
| Batch response change | Matched condition comparisons and per-plate response range | Requires comparable conditions and trustworthy plate metadata |
| True non-response | Flat test-treatment fit alongside healthy controls/reference response | Preparation and dose-annotation errors can mimic biology |

The Week 2 report presents these findings; it does not consume the table above as
a hidden case-to-label lookup.

## Execution plan

Each step ends with a reviewable checkpoint. Codex should implement and verify one
checkpoint at a time, then summarize scientific assumptions before proceeding
when those assumptions materially affect interpretation.

### 1. Add the investigator-safe loader and design summary

Actions:

- Add `InvestigatorCase`, `DesignSummary`, and capability models.
- Implement a benchmark adapter that verifies the five public artifacts through
  existing hashes, then discards the manifest before returning analysis input.
- Derive design properties from data and public metadata rather than
  `GeneratorConfig` defaults.
- Preserve incomplete observations in the investigator contract even though the
  frozen Week 1 benchmark itself is complete.
- Validate column meaning, key uniqueness, coordinates, role values, and metadata
  references without requiring the frozen `8/8/40/40` allocation.

Verification:

- All fourteen benchmark cases load into the restricted contract.
- Serializing the contract exposes no ground-truth fields or planted label.
- Synthetic unit fixtures cover 96- and 384-well geometry, unequal replicates,
  missing reference treatment, and missing values.
- A split-across-plates fixture is identified as unsupported for cross-plate and
  pooled analysis rather than being interpreted as complete replicate plates.
- Reordering CSV rows does not change the derived design summary.

Checkpoint: inspect the derived summary for one ordinary case, the two-plate batch
case, and one deliberately irregular unit fixture.

### 2. Define stable evidence, status, and provenance models

Actions:

- Add enums and Pydantic models for tool status, evidence, warnings, scientific
  result provenance, and telemetry.
- Implement stable evidence-ID construction using canonical JSON plus a short
  SHA-256 digest.
- Define canonical serialization for JSON tool results.
- Make `not_applicable` distinct from `insufficient_data`: the former means the
  design lacks the concept, while the latter means the concept exists but data are
  inadequate.
- Reserve `error` for invalid parameters or unexpected computation failure.

Verification:

- Equivalent scopes and parameters yield identical evidence IDs regardless of
  dictionary or input row order.
- Different plate, treatment, metric, or tool versions yield different IDs.
- Runtime changes do not alter evidence or golden scientific output.
- Unknown fields and non-finite numeric evidence are rejected.

Checkpoint: approve example JSON for one successful result and each non-success
status before implementing statistics.

### 3. Implement missingness and control summaries

Actions:

- Implement `inspect_missingness` for missing wells, missing measurements, and
  missing design annotations, separated by plate and role.
- Implement `summarize_controls` with per-plate counts, mean, median, standard
  deviation, robust dispersion, control separation, and an assay-window statistic
  when both control types are available.
- Report absent control roles as `not_applicable` and inadequate control counts as
  `insufficient_data` for variability-dependent statistics.
- Guard calculations with explicit denominator and finite-value checks.

Verification:

- Compare all calculations against hand-computed small fixtures.
- Cover absent positive controls, singleton controls, zero separation, outliers,
  missing control values, and two-plate inputs.
- Confirm the weak-control cases reduce separation without reading their labels.

Checkpoint: review control results for clean, weak-control, and batch cases.

### 4. Implement replicate variability and dose-response fitting

Actions:

- Implement `calculate_replicate_variability` per observed plate, treatment, and
  dose with counts, mean, standard deviation, robust dispersion, and explicit
  handling of singleton groups.
- Summarize the distribution of within-condition variability without assuming
  equal replicate counts.
- Implement bounded decreasing four-parameter logistic fits per eligible
  treatment using SciPy.
- Return fitted parameters, convergence status, observations/doses used,
  residual-error summaries, goodness of fit, and parameter-bound warnings.
- Do not require a reference treatment; compare treatments only when the observed
  design supports it.

Verification:

- Recover known curve parameters within tolerances on noiseless synthetic data.
- Cover noisy fits, flat response, too few unique doses, unequal replicates,
  missing doses, and solver failure.
- Row reordering does not change results.
- True-non-response cases return a flat/poorly identified test-treatment result
  while the reference and controls remain analyzable.

Checkpoint: review fitted curves and structured failures for responsive, flat,
and insufficient-data fixtures.

### 5. Implement spatial and layout diagnostics

Actions:

- Build condition-centered residuals using only public plate-map groups.
- Implement edge/interior comparisons from declared or inferred geometry.
- Implement row-wise and column-wise residual trend summaries without treating
  file order as dispensing order.
- Detect localized extreme residuals and simple adjacent runs without asserting a
  physical tip channel.
- Quantify whether well role, treatment, or dose is insufficiently distributed
  across row, column, and edge/interior regions.
- Return explicit non-identifiability warnings when condition and position cannot
  be separated.

Verification:

- Detect planted obvious spatial patterns and retain weaker evidence for noisy
  variants.
- Layout-confounding fixtures produce a design-confounding warning rather than a
  confident spatial-cause claim.
- Transient-clog fixtures produce localized evidence without access to traversal
  or affected-channel truth.
- Geometry tests cover 96- and 384-well coordinates.
- Shuffling input rows does not change any result.

Checkpoint: compare the numerical evidence with the approved residual heatmaps for
edge, drift, transient-clog, and confounded-layout cases.

### 6. Implement matched batch comparisons

Actions:

- Implement `compare_batches` using public plate and batch metadata.
- Require plates to represent complete replicates of the same design for Week 2
  cross-plate comparisons.
- Match like-for-like well roles and treatment-dose conditions before comparing
  plates; never compare unmatched aggregate condition mixtures.
- Report per-condition differences, uncertainty, negative-control anchors,
  positive-control range, and response-range ratios.
- Return `not_applicable` for a single plate/batch and `insufficient_data` when
  plates lack comparable conditions.

Verification:

- The two batch cases show the planted dynamic-range difference with negative
  controls remaining anchored.
- Identical plates show no systematic difference.
- Different condition mixes do not produce a misleading aggregate comparison.
- Biological condition replicates split across plates produce a structured
  unsupported-design result and are not pooled.
- Plate order and CSV row order do not change results.

Checkpoint: approve one matched-comparison result and its limitations.

### 7. Wrap existing heatmaps as an analysis tool

Actions:

- Implement `generate_plate_heatmap` as a typed wrapper around the existing
  Altair plotting functions.
- Accept only public case data and explicit output paths.
- Return plot artifact paths and hashes as evidence attachments, not as numeric
  evidence.
- Preserve fixed orientation and explicit color domains.

Verification:

- PNG export remains deterministic for the same environment and inputs.
- Tool results contain no latent or planted-failure fields.
- Invalid or escaping output paths fail closed.

Checkpoint: compare the wrapper output with an approved Week 1 case plot.

### 8. Build the deterministic baseline report

Actions:

- Run the applicable tools in a fixed, documented order; this is ordinary Python
  orchestration, not an agent.
- Keep all successful and non-success tool results so evidence limitations remain
  visible.
- Add versioned synthetic-benchmark heuristics that translate metrics into neutral
  QC findings without claiming a definitive physical cause.
- Produce a typed report containing input summary, capabilities, findings,
  evidence index, warnings, limitations, and suggested deterministic follow-ups.
- Render the same report as canonical JSON and concise Markdown.
- Ensure every reported numeric claim cites one or more evidence IDs.

Verification:

- A report can be rebuilt byte-for-byte after excluding the separate telemetry
  envelope.
- A consistency test rejects narrative numbers without evidence references.
- The report contains no manifest label, expected evidence, injector parameter,
  child seed, or latent value.
- Golden tests cover a healthy baseline input and representative technical and
  biological failures.

Checkpoint: review one JSON/Markdown pair and trace every number back to a tool
result.

### 9. Add CLI commands and run the fourteen-case baseline

Actions:

- Add:

  ```bash
  uv run experiment-failure-investigator qc cases/<case_id>
  uv run experiment-failure-investigator qc cases/<case_id> --output reports/<case_id>
  uv run experiment-failure-investigator qc-batch cases --output reports
  ```

- Refuse to overwrite reports without `--force`.
- Keep generated development reports under ignored `reports/` until scientific
  review is complete; commit only selected golden examples.
- Produce a compact batch summary with case path, applicable tools, statuses,
  warning count, evidence count, and report paths. Do not include planted labels
  in investigator reports.
- Compare results with the evaluation-only manifest in a separate development
  audit that never feeds labels back into report generation.

Verification:

```bash
uv run experiment-failure-investigator qc cases/weak_controls_obvious
uv run experiment-failure-investigator qc-batch cases --output reports
uv run pytest
git diff --check
```

- All fourteen reports validate and contain only evidence-linked numbers.
- The batch runs with Ollama stopped and no network access.
- At least one case exercises every successful tool path, and non-applicable tools
  are represented explicitly.

Checkpoint: review the fourteen-case summary, record false positives and missed or
ambiguous findings, and approve the deterministic baseline before Week 3 prompts
are written.

### 10. Freeze the Week 2 contracts

Actions:

- Add JSON Schemas for investigator input, tool results, and baseline report.
- Record tool and heuristic versions in `docs/design_log.md`.
- Update the README with exact QC commands and one evidence-tracing example.
- Document known statistical and benchmark limitations.
- Confirm the analysis package has no imports from agent frameworks, model
  providers, benchmark injectors, or private ground-truth models.

Verification:

- Schemas match their Pydantic sources.
- A clean checkout reproduces the selected golden report.
- Static searches find no model call in deterministic analysis or reporting code.
- Full tests pass without Ollama, credentials, or network access.

## Test matrix

| Layer | What is tested | Model required |
|---|---|---|
| Contract | Strict schemas, statuses, evidence IDs, canonical JSON | No |
| Unit | Each statistic, fit, edge case, and failure status | No |
| Property/invariant | Row-order invariance, stable IDs, geometry independence | No |
| Integration | Public loader through JSON/Markdown report | No |
| Benchmark regression | Fourteen cases and expected evidence families | No |
| Leakage | No evaluation-only truth in inputs, tools, or reports | No |
| Visual | Heatmap wrapper matches approved orientation and scale behavior | No |

## Week 2 exit checklist

- [ ] Investigator-facing case and design-summary contracts are strict and
  ground-truth-free.
- [ ] Tool statuses distinguish success, not applicable, insufficient data, and
  errors.
- [ ] Every scientific result has stable evidence IDs and public-input provenance.
- [ ] Missingness, controls, replicates, spatial effects, dose response, batches,
  and heatmap tools are independently tested.
- [ ] Tools handle observed design variation or return a structured limitation.
- [ ] Multi-plate analysis is limited to complete replicate plates, and
  split-across-plate designs fail closed with an explicit limitation.
- [ ] The deterministic report contains no uncited numeric claims.
- [ ] No investigator input or report contains planted labels or private truth.
- [ ] All fourteen benchmark cases produce validated JSON and Markdown reports.
- [ ] The CLI and tests run with Ollama stopped and without network access.
- [ ] A human has reviewed the deterministic baseline before Week 3 prompting.

## Explicitly deferred

- Hypothesis generation and ranking by a language model: Week 3.
- Prompt templates and Codex prompt rehearsal: Week 3, after evidence contracts
  are frozen.
- Live Ollama/Qwen calls and tool selection: Week 3.
- Skeptic-agent behavior and iterative orchestration: Week 4.
- Universal laboratory QC thresholds: outside the synthetic benchmark unless
  supported by assay-specific evidence.
- Claims of layout or plate-format generalization: only after fingerprint-disjoint
  held-out layouts and 384-well evaluation cases exist.
