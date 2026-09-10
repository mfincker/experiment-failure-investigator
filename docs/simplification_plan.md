# Week 3 model and workflow simplification plan

## Decision

Simplify the current investigator implementation, but do not turn this branch
into a repository-wide cleanup.

The project plan makes several apparently elaborate pieces part of the actual
learning and portfolio goals rather than hypothetical infrastructure:

- observe a real Pydantic AI model/tool exchange;
- run locally through Ollama without Codex supervising the application;
- keep model requests, tool calls, retries, tokens, and time bounded;
- replay model behavior without repeating inference;
- retain a machine-readable trace that can later support evaluation; and
- preserve shared scientific and output contracts for a possible LangGraph
  comparison.

Those goals do not justify every current class or configuration field, but they
do mean that replay, traceability, schemas, and framework-neutral contracts
cannot be deleted merely because their later consumer has not been built yet.

This plan therefore focuses on one visible vertical slice:

```text
experiment-failure-investigator run <case>
    -> load and validate public assay data
    -> run deterministic diagnostics
    -> let one Pydantic AI investigator inspect bounded evidence
    -> validate the structured hypotheses
    -> write an investigation and a reviewable trace
```

## Primary problem: the Pydantic model graph

The main readability problem is not the number of modules. It is that internal
data passes through many nested Pydantic projections before reaching the model or
the output directory:

```text
InvestigatorCase + BaselineReport
    -> PlateBriefing / TreatmentBriefing / DiagnosticResultCatalogEntry
    -> AgentBriefing
    -> PromptArtifact + AssembledPrompt
    -> InvestigatorDependencies
    -> Pydantic AI
    -> ModelEvent / ToolEvent / AgentUsage / RuntimeSnapshot / RunFailure
    -> InvestigationTrace
    -> InvestigationArtifacts + InvestigationRun
```

Most arrows do not cross a trust boundary. The source case and baseline are
already validated, the prompt objects are assembled internally, and the
controller result is consumed internally. Revalidating each intermediate shape
makes the relationships harder to see without adding equivalent safety.

The target relationship should be visible directly in the controller:

```text
validated case + validated baseline
    -> build briefing JSON
    -> assemble system and user strings
    -> run Agent with case/baseline dependencies
    -> validate InvestigatorOutput
    -> write output and one compact trace
```

### Pydantic boundary rule

Use Pydantic when data enters from, or is written across, a meaningful boundary:

- runtime configuration supplied by a user or environment;
- deterministic scientific inputs and results, which are frozen Week 1–2 work;
- the model-generated `InvestigatorOutput` and its `FailureHypothesis` entries;
- bounded evidence returned to the model, if a schema materially clarifies the
  tool contract; and
- the compact machine-readable trace written for later review and evaluation.

Do not use Pydantic merely to pass already validated data between two local
functions. Prefer a direct value, a plain JSON-compatible dictionary, or a small
dataclass when a named internal return value genuinely improves readability.

## Rules for this refactor

- Work from the CLI and controller inward.
- Change only code needed to make the current single-investigator workflow easy
  to run and understand.
- Make the case-to-prompt-to-agent-to-result relationship readable from the
  controller without opening a chain of model definitions.
- Keep Week 1 and Week 2 scientific behavior and serialized contracts frozen.
- Preserve an abstraction when a current test, a Week 3 deliverable, or the
  planned framework comparison gives it a concrete purpose.
- Do not remove something based only on source-line count or the absence of a
  consumer that is explicitly scheduled for a later project milestone.
- Prefer simplifying an implementation in place over moving files or inventing
  a replacement layer.
- Use the first real Qwen trace as evidence before reducing the tool surface,
  replay behavior, or trace fields.

## What remains justified

The following boundaries have a current, stated purpose and remain in scope:

- `InvestigatorOutput` and `FailureHypothesis` provide the framework-neutral
  model-output boundary that a later backend and evaluation harness can share.
- `AgentRuntimeConfig` validates user-controlled local model selection and
  execution limits outside scientific code.
- Evidence-building functions keep private benchmark truth out of model context
  and bound payloads for a local model; their current intermediate Pydantic DTOs
  are not automatically justified.
- Prompt-loading functions keep the reviewed prompt versioned, separate trusted
  instructions from untrusted case text, and supply the recorded prompt hash;
  `PromptArtifact` and `AssembledPrompt` do not need to be Pydantic models to do
  that.
- `agent/investigator.py` demonstrates Pydantic AI dependencies, tools, and
  output validation.
- `agent/controller.py` demonstrates what the application—not the model or
  framework—owns: sequencing, limits, terminal status, and safe artifact writes.
- A replay path supports the explicit no-inference regression strategy. Its
  implementation may become test-only, but the capability should remain.
- A compact, framework-neutral `InvestigationTrace` is required for review and
  later evaluation. It does not require a separate Pydantic model for every
  internal event or metadata grouping. Pydantic AI messages may be attached, but
  should not be the only durable result if that would couple evaluation to one
  framework.
- `InvestigatorOutput` and trace JSON Schemas support the stated shared-contract
  and portfolio goals.
- Evidence IDs, public/private separation, deterministic diagnostics, empty-well
  handling, capabilities, insufficient-data results, and Altair reports remain
  unchanged.

## Models that can be simplified without a live run

These models describe internal plumbing rather than external boundaries and can
be simplified based on the current code alone:

- Replace `PlateBriefing`, `TreatmentBriefing`, `AgentBriefing`, and catalog
  entry/list models with one `build_briefing_json()` projection. Its output is
  immediately serialized into the prompt and is never accepted from a caller.
- Remove `EvidenceFilters`; the tool function arguments already describe the
  filters.
- Return resolved records directly rather than wrapping them in
  `ResolvedEvidence`.
- Replace `PromptArtifact` and `AssembledPrompt` with prompt-loading and assembly
  functions returning strings plus the one hash that the trace needs.
- Replace `InvestigationArtifacts` and `InvestigationRun` with one small
  non-Pydantic `RunResult` dataclass, or return the trace and paths directly if
  that is clearer at the CLI call site.
- Flatten `RuntimeSnapshot` into the trace rather than copying nearly every
  `AgentRuntimeConfig` field into another validated model.
- Consolidate `ModelEvent` and `ToolEvent` only after identifying the minimum
  event fields required to understand ordering and tool use. Per-payload hashes
  are internal integrity checks, not separate scientific boundaries.
- Flatten `AgentUsage`, `VersionRecord`, and `RunFailure` into the compact trace
  when doing so keeps successful and failed terminal states unambiguous.

The intended result is not “no models.” It is a short, explainable model graph:

```text
AgentRuntimeConfig
InvestigatorOutput -> FailureHypothesis
InvestigationTrace -> InvestigatorOutput (on success)
```

An evidence response model may remain if the final tool design benefits from an
explicit response schema. Replay-fixture models remain test concerns and should
not appear in the production workflow explanation.

## Complexity that requires a live run to validate

These are candidates, not pre-approved deletions:

- `mock`, `replay`, `local`, and `smoke` may be test/run profiles rather than
  four production modes exposed through one configuration model.
- The custom Ollama preflight state model may be more machinery than is needed
  to produce an actionable connection or missing-model error.
- `list_diagnostic_results`, `inspect_diagnostic_result`, and
  `resolve_evidence` may overlap, but only actual Qwen tool behavior can show
  whether the catalog and exact-resolution steps help a small local model.
- Per-event payload hashes and repeated environment/version snapshots may add
  little beyond one run-level provenance record.
- Some trace event classes may be replaceable with a smaller stable event shape
  plus the framework's captured messages.
- Some environment variables may never need user configuration and can become
  internal conservative constants.

The decision test for each candidate is: what current failure does it prevent,
what observable requirement depends on it, and can the same requirement be met
with less code without weakening the scientific or framework boundary?

## Execution plan

Implementation status:

- Step 0 is complete. The 306-test offline baseline passed, and Step 8 plus this
  plan were committed separately without staging user-owned files.
- Step 1 is implemented for review. The normal CLI now reaches the existing
  controller with local Ollama configuration, uses an opaque default run path,
  refuses overwrite, and is covered end to end by an offline replay model.
- Step 2A is implemented for review. Briefing, catalog, resolution, and prompt
  assembly now use direct JSON and string values; nine internal Pydantic models
  were removed while preserving the reviewed prompt and tool-payload hashes.
- Step 2B is implemented for review. Eight specialized controller and trace
  models were replaced by one `RunResult` dataclass, one ordered `TraceEvent`
  model, and flattened trace fields. `AgentRuntimeConfig` is embedded directly,
  redundant payload hashes were removed, and the trace schema is now `2.0.0`.
- Steps 3–5 remain pending.

### Step 0 — Preserve and verify the starting point

Inspect:

- the current branch status and diff;
- the existing Step 8 rehearsal notes;
- user-owned `memory.md`, `.python-version`, and `test.py`; and
- the full offline test suite.

Actions:

- keep the work on `simplification`;
- commit the Step 8 deliverable separately from refactor changes if it is ready;
- do not edit or stage user-owned files; and
- record the current full-suite result as the behavioral baseline.

Pass condition: the branch has an understood starting state and the full offline
suite passes before refactoring.

### Step 1 — Expose the existing investigator through one CLI command

Likely changes:

- `src/experiment_failure_investigator/cli.py`;
- `src/experiment_failure_investigator/agent/controller.py` only if a thin public
  entry point is missing;
- focused CLI/controller tests; and
- a short README usage section.

Add the smallest useful command:

```bash
uv run experiment-failure-investigator run CASE_DIRECTORY \
  --output OUTPUT_DIRECTORY
```

The default application path is local Ollama. Model tag and base URL remain
configurable. Tests inject a `FunctionModel` or existing replay model without
turning mock behavior into a user-facing production workflow.

The command should print the output location on success and an actionable typed
error on failure. It should not add `run-smoke`, evaluation batching, framework
selection, or new configuration files.

Checks:

- CLI help and argument tests;
- one offline end-to-end command test;
- overwrite-safety test;
- provider requests remain disabled in default tests; and
- existing `generate`, `validate`, and `qc` commands still pass.

Pass condition: the complete investigator is runnable and inspectable from a
normal terminal without knowing the internal module graph.

### Step 2 — Flatten internal Pydantic projections

Likely changes:

- `agent/evidence.py` and its focused tests;
- `agent/prompts.py` and its focused tests;
- `agent/controller.py`;
- `agent/trace.py` and its focused tests; and
- imports in `agent/investigator.py`.

Actions:

- make the validated `InvestigatorCase` and `BaselineReport` the visible source
  objects throughout prompt construction;
- build the briefing as one JSON-compatible projection instead of a nested
  family of Pydantic models;
- make prompt loading and assembly ordinary functions;
- retain only the meaningful agent-output models;
- reduce controller return wrappers to one simple result;
- redesign the trace as one compact written contract with the minimum nested
  structure needed to distinguish output, failure, usage, and ordered tool
  activity; and
- remove tests that only prove an internal wrapper can validate data that its
  already validated source produced.

Checks:

- compare the briefing and assembled prompt byte-for-byte before and after the
  change;
- compare successful `investigation.json` semantically before and after;
- verify failed and successful traces remain unambiguous;
- run focused prompt, evidence, investigator, controller, and trace tests; and
- run the full offline suite.

Pass condition: a reader can follow the complete object flow from
`run_investigation()` without navigating through nested internal model families,
while the model-visible prompt and scientific result remain unchanged.

### Step 3 — Run and review one bounded local Qwen investigation

This is an observation step, not a refactor.

Actions:

- confirm an installed Qwen tag through the existing preflight;
- run the smallest representative public case with current limits;
- inspect the prompt, model requests, tool calls, tool returns, validated output,
  trace, runtime, and failure messages; and
- record which tools, trace fields, DTOs, and settings were actually useful in
  understanding the run.

Do not automatically pull a model or contact a remote provider.

Pass condition: the application either produces a schema-valid investigation or
a typed failure that clearly identifies the next compatibility change. In both
cases, there is concrete evidence to guide simplification.

### Step 4 — Make one evidence-backed behavioral simplification pass

Inspect only the Week 3 agent execution path:

- `agent/config.py`;
- `agent/evidence.py`;
- `agent/prompts.py`;
- `agent/investigator.py`;
- `agent/controller.py`;
- `agent/replay.py`;
- `agent/trace.py`; and
- their focused tests.

For each suspected-complexity item, choose one of three outcomes and document it
in the commit or review notes:

1. keep it because the live run or an explicit requirement demonstrated its
   purpose;
2. simplify it in place while preserving behavior; or
3. remove it and its implementation-only tests because no current requirement
   depends on it.

Likely safe reductions, subject to the live trace:

- make test-only model profiles test fixtures rather than normal CLI choices;
- replace repeated run metadata with one run-level provenance block;
- collapse trace event variants that are not needed to reconstruct or evaluate
  the exchange;
- make fixed safety limits constants when users have no real reason to tune
  them; and
- remove response wrappers that only duplicate an existing validated object.

Do not predetermine that the investigator must have exactly one tool. Retain the
smallest tool set that the Qwen run shows is understandable and that keeps
evidence payloads bounded. A 384-well-capable input contract makes truncation and
bounded retrieval a likely requirement, not a hypothetical edge case.

Checks:

- the same offline end-to-end fixture passes before and after the change;
- one saved/replayed Qwen exchange still validates without live inference;
- invalid evidence IDs, case IDs, budgets, and tool requests still fail closed;
- the trace still explains model/tool sequencing and supports later evaluation;
- no benchmark-private data reaches model context or artifacts; and
- the full offline suite passes.

Pass condition: the vertical slice has fewer concepts or less configuration for
a reader to understand, with no loss of a demonstrated behavior.

### Step 5 — Complete Week 3 through the simplified path

After reviewing Step 4:

- run the bounded three-case local smoke set sequentially;
- keep reviewed sanitized responses for offline regression where useful;
- document differences between Codex rehearsal and Qwen behavior; and
- update the README with one reproducible local example and one small execution
  diagram.

This step completes the remaining Week 3 outcome. It does not begin the skeptic,
human-review workflow, or LangGraph port.

Pass condition: the Week 3 exit criteria are met through the normal CLI and a
new reader can reproduce one investigation from the README.

## Deferred decisions

The following may deserve simplification later, but are intentionally outside
this branch unless the Week 3 vertical slice directly forces a change:

- redesigning `BaselineReport`, `ScientificToolResult`, provenance, or its
  evidence index;
- deleting Week 1/2 schemas or schema snapshot tests;
- flattening package exports across `analysis`, `benchmark`, and `reporting`;
- reorganizing the source tree;
- deleting or archiving weekly plans and the project plan;
- reducing committed generated plots;
- changing the skeptic/human-review design;
- deciding whether LangGraph remains worth implementing; and
- generalizing diagnostics to 384-well or split-replicate designs.

These should be reconsidered when their consuming workflow is visible. The
project and weekly plans remain active learning records until portfolio packaging
in Week 6; they should not be removed during a Week 3 code refactor.

## Non-goals

- Do not redesign scientific calculations, thresholds, benchmark cases, or
  Week 2 contracts.
- Do not add dependencies, folders, configuration files, frameworks, providers,
  databases, or interfaces.
- Do not implement Week 4 or Week 5 while simplifying Week 3.
- Do not preserve complexity solely because tests encode it; preserve behavior
  that serves a current requirement.
- Do not remove observability, replay, or shared contracts solely because their
  evaluation and second-framework consumers arrive later in the stated roadmap.
- Do not use fewer files or fewer tests as the success metric. Optimize for the
  number of concepts a reader must understand to follow one real run.

## Review checkpoint

Approve Steps 0–2 first. They create the top-down vertical slice and flatten
internal models whose inputs are already trusted. Review the real Qwen trace from
Step 3 before approving the behavioral removals in Step 4. Reassess the broader
roadmap only after Week 3 works end to end.
