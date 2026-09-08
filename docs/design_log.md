# Design Log

## Week 1: synthetic benchmark foundation

Frozen contract versions: benchmark `1.0.0`, manifest schema `1.0.0`.

### Scope frozen for the initial slice

The initial benchmark contains fourteen deterministic cases: obvious and noisy
variants of edge effect, progressive pipetting drift, transient tip clog, layout
confounding, weak controls, batch response-scale change, and true biological
non-response. The fixture uses one 96-well balanced layout except for the two
deliberately confounded-layout cases.

The clean response uses two decreasing four-parameter logistic curves, negative
controls centered near `1.0`, positive controls centered near `0.15`, and seeded
Gaussian well noise. These are versioned simulation choices, not universal assay
acceptance thresholds.

### Why transient tip clog is separate from progressive drift

Progressive pipetting drift creates a traversal-wide gradient. A transient clog
affects selected channels across only a limited run of simulated dispense groups
and then recovers. They have different spatial signatures and follow-up actions,
so treating them as separate failure modes makes the evaluation target clearer.
The traversal and dispensing assumptions remain private truth because scientists
cannot normally supply trustworthy dispense order without instrument logs.

### Why the batch case scales around the control anchor

The benchmark signal is already normalized to negative controls. Adding a
constant to every normalized value would incorrectly move that reference away
from `1.0`. The batch injector instead scales response around the observed
negative-control mean:

```text
shifted = anchor + scale_factor * (original - anchor)
```

This represents a batch-specific change in assay dynamic range. A purely
multiplicative pre-normalization shift would cancel during normalization and is
not identifiable from the normalized values alone.

### Typed state versus generated files

Generator configuration, public metadata, manifest structure, failure parameters,
seeds, and artifact references belong in typed state. CSV measurements, plate
maps, protocol text, problem statements, PNG plots, and canonical JSON are durable
generated representations. The serializer validates both sides of this boundary
and rejects unexpected public columns.

### Pydantic validation versus cross-row validation

Pydantic enforces field-level types, enumerations, ranges, safe relative paths,
schema versions, and manifest consistency. Data-frame validation enforces
plate/well uniqueness and coverage, geometry, matching keys across files, role and
dose semantics, replicate counts, finite measurements, metadata references, and
layout fingerprints. These checks require relationships across rows or files and
therefore do not fit a single-row model.

### Ground-truth isolation

Private expected signals, baseline-noise draws, injected effects, traversals,
seeds, and planted labels are needed for generation and scoring but would leak the
answer to an investigator. Investigator-visible inputs are limited to the five
public files. The manifest is explicitly evaluation-only, and serialization
rejects latent columns in public tables.

### Identifiability limits

Weak controls, treatment non-response, and between-plate response-scale changes
have direct comparisons in the current design. Spatial patterns can be detected,
but without dispensing logs they are only consistent with an edge, timing, or
liquid-handling mechanism. Layout confounding is deliberately non-identifiable:
condition and position cannot be separated from the supplied plate, so the correct
outcome is to report that limitation and recommend a redesigned repeat.

### Most informative later checks

- Edge effect: condition-adjusted edge/interior comparison and a relocated repeat.
- Progressive drift: row/column trend tests plus acquisition or liquid-handler logs.
- Transient tip clog: localized residual scan plus pressure, error, and reload logs.
- Layout confounding: design audit and a balanced randomized repeat.
- Weak controls: control separation, variability, and assay-window statistics.
- Batch response change: like-for-like per-plate comparisons and batch metadata.
- True non-response: treatment curve fit alongside healthy control/reference checks.

### Orchestration-independent components

Generation, serialization, validation, plotting, deterministic QC tools, evidence
models, and final report contracts should remain unchanged when orchestration
moves from Pydantic AI to LangGraph. Only workflow state transitions, model/tool
routing, checkpointing, and loop control should differ.

## Week 2: deterministic investigator contracts

Frozen versions:

- investigator, scientific-tool-result, and baseline-report JSON Schemas: first
  committed Week 2 revisions;
- missingness tool: `1.0.0`;
- control-summary tool: `1.0.0`;
- replicate-variability tool: `1.0.0`;
- dose-response tool: `1.0.0`;
- spatial-effects tool: `1.0.0`;
- batch-comparison tool: `1.0.0`;
- diagnostic-plot tool: `1.0.0`;
- deterministic baseline report: `1.0.0`;
- synthetic-benchmark heuristic configuration: `1.0.0`; and
- batch QC summary: `1.0.0`.

The committed schemas in `docs/schemas` are the compatibility boundary for the
first agent workflow. Changing a field's meaning, requiredness, or serialized
shape requires an explicit version decision rather than silently regenerating
the schema.

### Evidence before hypotheses

Week 2 deliberately contains no model-selected tools or generated hypotheses.
The loader verifies the public artifact hashes, builds a ground-truth-free
investigator input, and discards the evaluation manifest. Deterministic tools
then return stable evidence IDs derived from public scope, parameters, tool name,
and tool version. Findings may cite those records but may not embed uncited
numbers. Runtime timestamps remain outside scientific equality.

### Applicability and failure-closed behavior

`not_applicable` means the observed design lacks a concept required by a tool;
`insufficient_data` means the concept exists but usable observations are
inadequate. Unsupported split-across-plate biological replicates are not pooled
or treated as complete plate replicates. Empty positions remain part of design
and missingness accounting but are excluded from signal statistics and fits.

### Frozen benchmark limitations

The current benchmark is suitable for developing deterministic contracts, not
for claiming general laboratory performance. Most cases reuse one 96-well layout,
all cases are synthetic and single-cause, assay thresholds are fixture-specific,
and the only supported multi-plate design uses complete replicate plates.
Spatial evidence is associative rather than causal, and normalized measurements
cannot recover pre-normalization shifts that normalization removed. Novel
layouts, 384-well plates, split-replicate designs, assay blanks, mapped-empty
signal anomalies, and mixed causes remain explicitly deferred.

## Week 3: single-investigator agent contracts

Initial contract versions:

- investigator output: `1.0.0`; and
- investigation trace: `1.0.0`.

Hypotheses use categorical confidence rather than uncalibrated numeric
probabilities. Model-generated narrative cannot embed numbers; quantitative
claims must instead cite existing Week 2 evidence IDs. Pydantic validates the
shape of the model output, while a separate application check verifies its opaque
case ID and every citation against the frozen baseline evidence graph.

Trace events retain sanitized JSON payloads plus canonical SHA-256 digests.
Execution-specific timestamps and durations remain runtime telemetry and do not
alter scientific evidence identities. Successful and failed terminal states are
mutually exclusive, and reported request, tool-call, and token usage cannot
exceed the captured runtime configuration.

### Compact context and evidence access

The model receives a compact briefing derived only from `InvestigatorCase` and
`BaselineReport`, not either complete object. The briefing retains public assay
context, plate and treatment summaries, design capabilities, neutral findings,
diagnostic statuses, warnings, limitations, evidence references, and metric
prefixes. It omits raw measurements, the complete plate map, tool parameters,
artifact paths, and the complete evidence index.

The deterministic diagnostics still run once before model inference. Agent tools
only query their frozen results: an explicit tool-name allowlist prevents access
from expanding when unrelated code is added, scope and metric filters must match
the selected result catalog, pages are ordered by stable evidence ID, and both
page size and exact evidence-ID resolution are hard-capped. For the representative
`weak_controls_obvious` development case, canonical briefing JSON is about four
percent of the complete deterministic baseline JSON; this is a payload comparison,
not a token-count or model-performance claim.

### Prompt versioning and untrusted case text

Investigator prompt `1.0.0` is stored as a reviewed UTF-8 text artifact with LF
line endings and a required final newline. Its exact SHA-256 digest is frozen by
an offline snapshot test. Runtime configuration selects a supported version and
fails if its artifact is absent; it never silently substitutes another prompt.

System instructions and case data remain separate messages. The public briefing
is canonical JSON inside reserved untrusted-data delimiters, and assembly fails if
case text contains either delimiter. This is a testable application boundary, not
a claim that delimiters alone prevent model prompt injection. The system prompt
also tells the model to interpret protocol and problem-statement content only as
data, while output validation and tool registration enforce the material limits.

The hard assembly ceiling is 24,000 characters, used as a deterministic preflight
guard rather than a token estimate. The representative `weak_controls_obvious`
prompt is 10,290 characters: 2,140 trusted-system characters and 8,150 user-message
characters. Provider-reported token limits remain separate runtime controls.

### Framework-managed exchange versus application-owned boundaries

Pydantic AI owns tool-schema publication, tool-call dispatch, typed output
parsing, validation retry messages, and the model request/response exchange. The
application owns the per-run `InvestigatorCase` and frozen `BaselineReport`, the
evidence page limit, semantic evidence validation, and conversion of rejected
queries into sanitized typed tool errors. Case data is injected through typed run
dependencies rather than globals or serialized into tool definitions.

The agent is constructed by a factory so tests can supply `TestModel` or
`FunctionModel` and production can supply the configured Ollama model without
changing the scientific boundary. Its only application tools are diagnostic
catalog listing, bounded result inspection, and exact evidence resolution.
Structured output is a separate framework output tool, not another scientific
capability. Cross-object output failures raise a bounded `ModelRetry`; they never
turn an invalid model response into accepted prose.

### Bounded single-run controller

The controller is the supported composition boundary: it loads a public case,
builds and writes the unchanged deterministic baseline, creates the briefing and
versioned prompt, constructs typed dependencies, invokes the agent, revalidates
its output against the baseline, and writes a terminal trace. It refuses an
existing output directory. A failed run writes its baseline and failure trace but
never writes `investigation.json`.

Pydantic AI enforces cumulative request, application-tool-call, and reported token
limits. The configured request timeout is passed to the model, while an
application timeout bounds the remaining model phase after deterministic setup.
Local and smoke modes also run the non-mutating Ollama preflight; a missing model
tag and an unavailable provider produce distinct failures before any model call.

Request and tool-call budgets can be checked before another action. Token counts
may only become available after a provider response, so a truthful
`budget_exhausted` trace may record token usage beyond its configured limit.
Successful and non-budget-failure traces still reject such an overage. This
distinction avoids hiding the very measurement that explains why execution
stopped.
