# Week 1 Implementation Plan: Scientific Benchmark

## Outcome

At the end of Week 1, the repository will generate twelve reproducible, labeled 96-well synthetic assay cases: one obvious and one noisy case for each of the six MVP failure modes. Each case will contain validated tabular inputs, a machine-readable manifest, and inspection plots. No agent or model call is needed this week.

This week intentionally builds the environment in which later agents will be evaluated. It does not implement Pydantic AI orchestration yet. That separation makes it possible to tell whether a later failure comes from the scientific benchmark, a deterministic tool, the workflow, or the model.

## Working assumptions to encode

These defaults are implementation starting points, not universal scientific thresholds. They must be named constants, documented, and easy to revise.

They define the initial benchmark profile only. Investigator-facing schemas and deterministic tools must derive plate dimensions, controls, treatments, concentrations, and replicate structure from each case rather than assuming this profile.

- Assay: generic endpoint cell-viability dose-response assay.
- Plate: 96 wells, rows `A`–`H`, columns `1`–`12`.
- Readout: continuous viability signal, normalized so vehicle controls center near `1.0` and effective positive controls near `0.1`–`0.2`.
- Response direction: lower signal means stronger inhibition.
- Treatment curve: four-parameter logistic response generated from known parameters.
- Replicates: at least three wells per experimental condition.
- Controls: vehicle/negative and positive controls are represented explicitly in the plate map.
- Randomness: every case has a root seed and named child seeds so clean generation and failure injection can be reproduced independently.
- MVP cases: single planted cause only. The noisy variant increases background noise but does not add a second labeled failure.
- Labels describe generator truth, not values that a future agent may quote as observed evidence.

Before implementation, record any scientifically motivated change to these assumptions in the benchmark specification.

## Target repository slice

```text
experiment-failure-investigator/
├── docs/
│   └── benchmark_spec.md
├── cases/
│   └── <case_id>/
│       ├── measurements.csv
│       ├── plate_map.csv
│       ├── metadata.json
│       ├── protocol.md
│       ├── problem_statement.txt
│       ├── manifest.json
│       └── plots/
│           ├── plate_heatmap.png
│           └── dose_response.png
├── src/experiment_failure_investigator/
│   ├── benchmark/
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── models.py
│   │   ├── layouts.py
│   │   ├── generator.py
│   │   ├── injectors.py
│   │   ├── validation.py
│   │   └── plotting.py
│   └── cli.py
└── tests/
    ├── unit/benchmark/
    └── integration/test_generate_benchmark_cases.py
```

Generated case files are committed directly under `cases/<case_id>/` for inspection and later golden tests. A period-specific subfolder is unnecessary because case identity and schema version already provide organization and provenance. Temporary plots and caches outside `cases/` are ignored.

## Data contracts

Define these contracts in `benchmark/models.py` before writing the generator.

### Measurement row

Required columns:

- `case_id`: stable case identifier.
- `plate_id`: stable plate identifier.
- `well`: canonical well such as `A01`.
- `raw_signal`: generated observed value.

There must be exactly one measurement row per plate-map well. Do not place generator truth or injected effect size in this CSV.

### Plate-map row

Required columns:

- `plate_id`, `well`, `row`, `column`.
- `sample_id`.
- `well_role`: `negative_control`, `positive_control`, or `treatment`.
- `treatment`.
- `dose` and `dose_unit`; controls may use null dose fields.
- `replicate`.

The plate map must make spatial position explicit, but it must not require `dispense_order`. In realistic use, scientists may provide a plate layout without liquid-handler logs or a trustworthy chronological dispensing sequence. File row order must never be treated as dispensing order.

If actual run-order metadata is available in a future case, it may be accepted as optional metadata with explicit provenance. Its absence must not invalidate a case.

### Case manifest

Required fields:

- `schema_version`, `case_id`, `case_variant`, and `root_seed`.
- `assay_type` and expected signal direction.
- `planted_failure_mode`.
- Public case parameters needed for reproducibility.
- Private generator-truth parameters under a clearly named `ground_truth` object. For synthetic pipetting-drift cases, this may include the simulated traversal used to inject the effect; it is not part of the scientist-facing input.
- Expected discriminating evidence and plausible confounders.
- Relative paths and SHA-256 hashes for all case inputs.

Use a Pydantic model as the source of truth and emit its JSON Schema to `docs/case_manifest.schema.json`.

## Implementation sequence for Codex

Each step ends with a reviewable checkpoint. Codex should implement one checkpoint at a time, run the named verification, summarize what changed, and wait for human review when a scientific assumption changes.

### 1. Establish the Python project baseline

Actions:

- Keep Python `3.12` and the existing `src/` package layout.
- Update project metadata and add only Week 1 dependencies: Pydantic, NumPy, pandas, SciPy, Altair, `vl-convert-python`, pytest, and any small test helper that is demonstrably needed. Altair owns chart specifications; `vl-convert-python` provides deterministic local PNG export without requiring a browser or Node.js.
- Add pytest configuration and CLI wiring without introducing an agent dependency into benchmark modules.
- Replace the placeholder package entry point with an argument-parsing CLI that can later grow subcommands.

Verification:

```bash
uv sync
uv run python -c "import experiment_failure_investigator"
uv run pytest --collect-only
uv run experiment-failure-investigator --help
```

Checkpoint: the environment installs, the package imports, tests are discoverable, and the CLI displays help.

### 2. Write the benchmark specification and contracts

Actions:

- Create `docs/benchmark_spec.md` with the clean signal model, fixed plate layout, noise model, six injector definitions, case naming scheme, and expected diagnostic evidence.
- Define enums and Pydantic models for well roles, failure modes, case variants, manifest data, and generator configuration.
- Document which fields are observable inputs and which are hidden benchmark labels.
- Document the boundary between fixed MVP fixture parameters and the future flexible investigator input contract.
- Ensure shared identifiers and geometry types can represent 96- and 384-well plates, different control allocations, blank wells, and cases without a reference treatment.
- Generate the manifest JSON Schema from the Pydantic model.

Verification:

- Model tests accept a minimal valid manifest.
- Model tests reject an unknown failure mode, invalid well, negative noise scale, and a manifest missing hashes.
- Model tests demonstrate that the generator contract can represent changed control counts, a missing reference treatment, and 384-well capacity without changing agent logic.
- Regenerating the JSON Schema produces no diff.

Checkpoint: the scientific assumptions and file contracts can be reviewed without reading generator code.

### 3. Implement well identities and the fixed plate layout

Actions:

- Implement canonical conversion among row, column, and well ID.
- Define one balanced layout with explicit negative controls, positive controls, doses, treatments, and replicates.
- Avoid confounding the clean layout with row, column, or edge status.
- Add a layout audit that reports counts and cross-tabs for well role, dose, row, column, and edge status.

Verification:

- Exactly 96 unique wells cover `A01` through `H12`.
- Every measurement key will have exactly one plate-map key.
- Required control counts and treatment replicate counts match the specification.
- The clean layout audit detects no perfect treatment-position confounding.

Checkpoint: save a readable layout table in the specification and review it before signal generation.

### 4. Implement the clean assay generator

Actions:

- Generate latent expected signal from control means and a four-parameter logistic treatment curve.
- Add seeded well-level noise separately from the latent expectation.
- Return data frames plus structured generation metadata; perform no file I/O inside the core numerical function.
- Keep raw signal generation independent from plotting and case serialization.

Verification:

- The same configuration and seed produce byte-equivalent tabular values after canonical sorting and serialization.
- Different seeds change noisy values but not layout or expected curve parameters.
- Clean positive and negative controls remain ordered in the expected direction.
- The noiseless dose response is monotonic and bounded by configured asymptotes.
- No NaN or infinite value appears unless missingness is deliberately injected.

Checkpoint: generate one temporary clean case and inspect its control summary, heatmap, and dose-response plot.

### 5. Implement six pure failure injectors

Each injector accepts a clean generated plate plus typed parameters and returns a new plate plus an injection record. It must not mutate its input.

1. `inject_edge_effect`
   - Shift boundary wells using a configurable magnitude and direction.
   - Expected evidence: edge-versus-interior difference and a perimeter pattern in the heatmap.

2. `inject_pipetting_drift`
   - Apply a monotonic signal gradient using a simulated dispensing traversal stored only as generator ground truth.
   - Scientist-facing evidence: a row-wise or column-wise gradient that remains after accounting for well role, treatment, and dose.
   - Interpretation limit: without observed run-order metadata, the investigator may call the pattern *consistent with* pipetting drift but cannot confirm dispensing order as the cause. Spatial gradients, timing effects, and layout confounding remain alternatives.
   - Useful follow-up: request instrument or liquid-handler logs if they exist, or repeat with a randomized/reversed layout that separates treatment position from the suspected gradient.

3. `inject_layout_confounding`
   - Create or select a deliberately confounded layout rather than merely changing values.
   - Expected evidence: treatment or dose becomes inseparable from position, so the appropriate conclusion may be non-identifiability rather than a spatial root cause.

4. `inject_weak_controls`
   - Move positive-control response toward negative controls without altering treatment labels.
   - Expected evidence: reduced control separation and poor assay-quality statistic.

5. `inject_batch_shift`
   - Generate at least two plates or batches and shift one batch using an explicit metadata field.
   - Expected evidence: between-batch difference after comparing like well roles or conditions.

6. `inject_true_non_response`
   - Flatten the treatment dose-response while leaving assay controls healthy.
   - Expected evidence: acceptable QC with little or no treatment response across dose.

Verification for every injector:

- Same input and parameters yield the same output.
- Input data are unchanged.
- Only intended values, layout fields, or metadata change.
- The obvious parameterization crosses a documented generator-level detectability check.
- The noisy parameterization remains present but is harder than the obvious case.
- The injection record contains the mechanism and magnitude but no prose conclusion generated by a model.

Checkpoint: review a before/after summary and plot for each injector before generating the twelve-case set.

### 6. Add serialization, validation, and hashes

Actions:

- Serialize CSV and JSON fields in a stable order and format.
- Write a case directory atomically after all validation passes.
- Validate well coverage, key uniqueness, allowed roles, dose consistency, finite measurements where required, metadata references, and manifest paths.
- Compute hashes after serialization and store them in the manifest without making the manifest hash self-referential.
- Refuse to overwrite an existing case directory unless an explicit `--force` option is supplied. The Week 1 run should not use `--force` until a diff has been reviewed.

Verification:

- Loading a freshly written case reproduces its typed manifest.
- Tampering with one CSV value causes hash validation to fail.
- Removing or duplicating a well causes structural validation to fail.
- A failed generation leaves no partial case directory.

Checkpoint: one case round-trips through generate, save, load, and validate.

### 7. Add plots for human inspection

Actions:

- Create an Altair plate heatmap with fixed row/column orientation and a consistent color scale within each clean/failure comparison.
- Create an Altair dose-response chart showing replicate points, condition summaries, and control reference bands.
- Put case ID, failure label, variant, and seed in plot metadata or title.
- Build charts in pure functions that return `alt.Chart` objects, then export PNG files locally with `vl-convert-python`.
- Use explicit dimensions, domains, sort orders, colors, and titles instead of relying on renderer defaults.

Verification:

- Chart tests inspect `chart.to_dict()` for the expected encodings, fields, sort order, scale domains, and titles.
- Export tests assert PNG files exist, are non-empty, and can be decoded.
- A manual review checks label readability, well orientation, dose ordering, and whether the planted pattern is visible without reading the manifest truth fields.

Checkpoint: approve one clean plot set and one failure plot set before batch generation.

### 8. Generate and inspect the twelve cases

Case IDs:

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

Actions:

- Implement general-purpose `generate-cases` and `validate-cases` CLI commands; Week 1 uses the initial twelve-case configuration registry.
- Generate all cases from a version-controlled registry of typed configurations.
- Produce a compact index containing case ID, seed, failure mode, variant, paths, and validation status.
- Inspect all plots in a grid or contact sheet, but retain the individual source plots.
- Record review notes, including cases that are too obvious, too subtle, or accidentally ambiguous.

Verification:

```bash
uv run experiment-failure-investigator generate-cases --output cases
uv run experiment-failure-investigator validate-cases cases
uv run pytest
git diff --stat
git diff -- docs/benchmark_spec.md
```

Checkpoint: all twelve cases validate and the human reviewer agrees that each planted mechanism is visible in the raw inputs and baseline plots at the intended difficulty.

### 9. Freeze the Week 1 benchmark slice

Actions:

- Record generator and manifest schema versions.
- Add a README section with exact reproduction and validation commands.
- Add a short design log entry for the layout, curve, and injector choices.
- Confirm no local paths, secrets, model outputs, or proprietary details are present.
- Commit source code, tests, specifications, case files, and inspection plots only after reviewing the generated diff.

Verification:

- A clean checkout can regenerate the same case hashes using the documented command.
- All tests pass with Ollama stopped; Week 1 must have no model dependency.
- `rg` finds no Claude-specific workflow instructions and no required remote API key.

## Test matrix

| Layer | What is tested | Model required |
|---|---|---|
| Unit | IDs, schemas, curve math, noise, injectors, validation, hashing | No |
| Integration | Generate/save/load/validate one case and the twelve-case batch | No |
| Property/invariant | Reproducibility, 96-well coverage, non-mutation, finite values | No |
| Visual review | Layout, heatmaps, dose-response shape, planted pattern | No |
| Prompt rehearsal | Not in Week 1 | No |
| Ollama/Pydantic AI | Not in Week 1 | No |

## Week 1 exit checklist

- [ ] Benchmark specification documents the clean model and all six mechanisms.
- [ ] Observable inputs and hidden ground truth are clearly separated.
- [ ] Twelve named cases are generated from committed configurations.
- [ ] Every case contains measurements, plate map, metadata, protocol, problem statement, manifest, and plots.
- [ ] Seeds and serialized file hashes make the cases reproducible and tamper-evident.
- [ ] Unit and integration tests pass without Ollama or network access.
- [ ] Obvious cases clearly express one mechanism.
- [ ] Noisy cases remain solvable without becoming mixed-cause cases.
- [ ] A human has reviewed the raw tables and plots for all twelve cases.
- [ ] The README contains exact generate, validate, and test commands.

## Decisions deliberately deferred

- Exact Pydantic AI agent prompts and output types: Week 3.
- Ollama tool-calling and structured-output reliability tests: Week 3.
- Skeptic-agent protocol and loop limits: Week 4.
- LangGraph state and checkpoint format: Week 5.
- Production-grade assay acceptance thresholds: outside the synthetic MVP unless supported by a cited public specification.
- Mixed-cause cases, ambiguous cases, and clean negative cases: add after the twelve single-cause cases pass review.

## Learning review questions

At the end of the week, answer these in `docs/design_log.md`:

1. Which facts belong in typed state, and which belong only in generated files?
2. Which invariants are enforced by Pydantic, and which require cross-row validation?
3. Why must the agent never see private generator truth during an investigation?
4. Which planted failures are identifiable from this layout, and which can only be flagged as confounded?
5. What later tool result would discriminate each failure from its nearest alternative?
6. Which parts of Week 1 will remain unchanged when orchestration moves from Pydantic AI to LangGraph?
