# Agentic Experiment Failure Investigator

An API-driven scientific reasoning system for investigating failed plate-based biological assays. The application combines deterministic quality-control tools with typed agent workflows to compare competing root-cause hypotheses, surface uncertainty, and recommend the most informative next check or experiment.

The project is intentionally split between scientific computation and language-model reasoning:

- Python tools calculate statistics and produce traceable evidence.
- Pydantic models define state and component boundaries.
- Pydantic AI will orchestrate the first agent workflow.
- LangGraph will later reimplement the same workflow for comparison.
- Ollama with a local Qwen model will provide inference without per-token API charges.
- A human remains responsible for accepting, rejecting, or revising recommendations.

## Project status

The Week 1 synthetic benchmark foundation is complete and visually approved.

Completed:

- Python 3.12 project and dependency setup
- Installable command-line entry point
- Baseline CLI tests
- Versioned synthetic benchmark specification
- Typed generator and evaluation-manifest contracts
- Reproducible case-manifest JSON Schema
- Geometry-aware balanced plate-layout generation and independent audits
- Deterministic clean assay generation with private latent-signal records
- Seven typed, deterministic, non-mutating failure injectors
- Altair heatmaps, dose-response views, and control-QC plots with local PNG export
- Deterministic case serialization with validation, SHA-256 integrity checks, and atomic writes
- A version-controlled fourteen-case registry with layout fingerprints and hashed plots
- Offline `generate-cases` and `validate-cases` CLI workflows

Next:

- Execute the detailed [Week 2 implementation plan](docs/week_2_implementation_plan.md)
  to add deterministic QC tools and evidence-linked baseline reports before
  introducing model-driven decisions.

The CLI can generate and validate the Week 1 benchmark without Ollama, network access, or a model API.

## Scientific scope

The MVP uses synthetic 96-well cell-viability or dose-response assays with seven planted scenarios:

1. Edge effect
2. Pipetting drift or a similar row/column gradient
3. Transient tip clog affecting a limited run of dispense groups
4. Plate-layout confounding
5. Failed or weak controls
6. Batch response-scale change
7. True biological non-response with otherwise acceptable QC

Initial cases will contain one planted cause. Mixed-cause and deliberately ambiguous cases will be added after the single-cause benchmark is reliable.

All data and failure scenarios are synthetic. Recommendations are designed for human review; the system has not been validated for decisions involving real laboratory experiments.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- Ollama and a compatible local Qwen model later in the project; neither is needed for the current tests

## Setup

Clone the repository, enter its directory, and synchronize the environment:

```bash
uv sync
```

If `uv` is installed in its default user location but is not on your shell path, invoke it directly:

```bash
~/.local/bin/uv sync
```

Alternatively, add `~/.local/bin` to your shell path or follow the uv installation instructions.

## Run the CLI

Show the available options:

```bash
uv run experiment-failure-investigator --help
```

Show the installed project version:

```bash
uv run experiment-failure-investigator --version
```

Generate the complete benchmark from a clean checkout:

```bash
uv run experiment-failure-investigator generate-cases --output cases
```

The generator refuses to write into a non-empty output directory. Use `--force`
only after reviewing the existing generated diff. Validate an existing benchmark:

```bash
uv run experiment-failure-investigator validate-cases cases
```

The generated [case index](cases/index.json) records case IDs, modes, variants,
seeds, paths, validation state, and layout fingerprints. Open the local
[review grid](cases/review.html) to inspect every clean/injected comparison.

## Run the tests

Run the complete test suite:

```bash
uv run pytest
```

Useful variants:

```bash
# Verbose output
uv run pytest -v

# CLI tests only
uv run pytest tests/test_cli.py

# One test
uv run pytest tests/test_cli.py::test_help_is_available
```

The deterministic test suite must remain runnable without Ollama, network access, or a remote model API key.

## Planned workflow

```text
validate inputs
    -> run deterministic baseline QC
    -> propose competing hypotheses
    -> select an allowlisted analysis tool
    -> challenge the leading explanation
    -> continue, abstain, or request human review
    -> produce an evidence-linked report
```

The application—not the language model—will enforce tool allowlists, iteration limits, timeouts, validation retries, and terminal states.

## Development roadmap

1. Define and generate the scientific benchmark.
2. Implement typed deterministic analysis tools and a non-agentic baseline.
3. Add an Ollama-backed Pydantic AI investigator.
4. Add a bounded skeptic loop and human-review checkpoint.
5. Port orchestration to LangGraph without changing scientific tools or output contracts.
6. Evaluate both workflows on labeled synthetic cases and document successes and failures.

Deferred benchmark extensions and their implementation triggers are tracked in [the project backlog](docs/backlog.md).

## Design principles

- Never ask the model to perform scientific arithmetic that a deterministic tool can perform.
- Validate every model output that crosses a component boundary.
- Keep benchmark ground truth separate from investigator-visible inputs.
- Treat the balanced 96-well design as a benchmark fixture, not an application assumption.
- Derive plate geometry, controls, doses, and replicate structure from each experiment.
- Return explicit insufficient-data results when a design cannot support an analysis.
- Cite evidence identifiers for every conclusion.
- Treat missing metadata as uncertainty rather than inventing values.
- Bound model requests, tool calls, retries, and workflow iterations.
- Make abstention a valid outcome.
- Keep model-provider configuration separate from scientific logic.
