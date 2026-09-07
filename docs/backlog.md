# Project Backlog

This backlog records work that is intentionally outside the current implementation checkpoint. Items move into an implementation plan only when their trigger is reached; their presence here is not a commitment to build them during Week 1.

## Layout generalization

### Add layout fingerprints during case serialization

**When:** Step 6, when canonical case serialization and hashes are implemented.

Create a stable SHA-256 fingerprint from the plate format and the semantic assignment of well role, treatment, and dose to canonically sorted plate/well coordinates. Exclude measurements, random seeds, case identifiers, and interchangeable replicate labels. Store the fingerprint in generated benchmark metadata so equivalent layouts can be recognized regardless of CSV row order.

**Done when:** regenerated equivalent layouts have the same fingerprint, meaningfully different well assignments have different fingerprints, and tests cover both behaviors.

### Generate multiple balanced layouts

**When:** the benchmark expands beyond the initial fixed-layout development slice.

Generate candidates from multiple layout seeds, audit every candidate, and reject or retry candidates that do not meet the declared balance requirements. A random seed is not expected to guarantee balance by itself.

**Done when:** the generator can produce a requested number of distinct, audited layouts reproducibly without assuming that every seed succeeds.

### Create a held-out layout evaluation split

**When:** before making any claim that the investigator generalizes across layouts.

Use layout fingerprints to keep evaluation layouts separate from layouts used to develop prompts, tools, thresholds, and deterministic diagnostics. Report fixed-layout, held-out balanced-layout, and deliberately confounded-layout results separately.

**Done when:** split validation rejects fingerprint overlap and the evaluation report states the number of distinct layouts in each split.

### Add realistic design variants

**When:** after the initial fourteen cases and their failure injectors are reliable.

Add benchmark cases with varied control allocations, dose series, treatment counts, replicate counts, missing wells, empty wells, and no reference treatment. Keep analysis code driven by observed case data rather than MVP defaults.

**Done when:** these cases run through the same investigator contracts without layout-specific prompt or tool changes.

### Model assay blanks as a distinct well role

**When:** a benchmark case needs wells containing medium, reagent, or another protocol-defined background material.

Add an `assay_blank` role that is distinct from a physically empty well. Its expected readout must be configured per assay and instrument context rather than assigned a universal value. Define how blank subtraction and investigator-visible metadata are represented before generating these cases.

**Done when:** assay blanks have explicit contents and readout semantics, and neither generators nor investigators confuse them with empty positions.

### Add unexpected content in empty wells

**When:** after the initial seven single-cause injectors and fourteen-case slice are reliable.

Add a failure family in which the reported plate map marks positions as empty but their readouts are inconsistent with empty positions. Treat accidental dispensing, contamination or carryover, plate-map annotation error, optical crosstalk, and instrument behavior as competing explanations unless the case provides discriminating evidence. Avoid attributing the discrepancy to user error without such evidence.

**Done when:** the benchmark can test recognition of the map/measurement inconsistency separately from confident identification of its physical cause, with appropriate abstention or follow-up recommendations.

### Add plate-format and multi-plate shifts

**When:** during expanded benchmark and evaluation work.

Add 384-well cases and scientifically applicable multi-plate cases. Evaluate format shifts separately from changes in well assignment so the source of any performance loss remains visible.

**Done when:** spatial analysis derives geometry from case metadata and evaluation results are stratified by plate format.

## Explicitly deferred infrastructure

A dedicated layout registry is not planned now. Version-controlled case configuration and generated fingerprints should be sufficient for the expected benchmark size. Reconsider a registry only if managing layouts through ordinary case metadata becomes error-prone.
