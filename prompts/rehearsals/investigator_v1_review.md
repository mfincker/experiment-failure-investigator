# Investigator prompt v1 Codex rehearsal

## Purpose and boundary

This is a development rehearsal of prompt `1.0.0`, not an evaluation of Qwen or
an estimate of benchmark performance. Codex received the exact trusted system
prompt and exact untrusted public briefing assembled by the application. During
case review it used only the diagnostic catalog already present in that briefing
and the bounded public evidence functions available to the Investigator.

Inputs and notes use opaque public case and evidence IDs. No case directory name,
manifest field, generator setting, planted failure label, or other evaluation-only
truth was used to form a hypothesis.

System prompt SHA-256 for every rehearsal:
`c6c08ef0dd8dd6347f16788b6eae83b8f361c8602eb4974c2d987bcd976c0419`.

## Rehearsal results

### `case_923eb39b90afc333`

- User prompt SHA-256:
  `c313cd1303c38da0ae65f43246d0d686e02bc40107e5b192f18dc4c7c2a76136`
- Evidence calls: complete control summary, test-treatment dose-response result,
  and reference-treatment dose-response result.
- Leading hypothesis: positive-control preparation or activity loss, at moderate
  confidence, supported by the control means and limited control-window evidence.
- Competing hypothesis: a plate-wide handling or readout disturbance, at low
  confidence; coherent treatment fits weigh against this broader explanation.
- Missing evidence: independent control preparation and storage records,
  instrument checks, and a replicate plate.
- Falsification-oriented recommendation: compare the current positive control
  with an independently prepared stock on a repeat plate.
- Contract result: passed typed output validation and baseline evidence-ID
  validation without correction.

Review: the prompt supported competing, cautious explanations and a useful next
check. It did not force a physical cause from limited control separation. The
current output contract attaches citations to hypotheses and the recommendation,
but not directly to `overall_assessment`, `remaining_uncertainty`, or
`limitations`; prompt compliance in those fields therefore remains review-based.

### `case_add1a2a4c90fb1ee`

- User prompt SHA-256:
  `cdf0936a20fea89af4aaf5f06bf0b3240033824c154baa9b62293fcc7680d168`
- Evidence calls: complete spatial result, complete control summary, and the
  aggregate median within-condition variability record.
- Leading hypothesis: a position-associated execution effect, at moderate
  confidence, based on a plate-axis residual association after conditions were
  centered.
- Competing hypothesis: condition-level random variability, at low confidence,
  supported by within-condition variability and an isolated extreme residual.
- Missing evidence: acquisition timing, liquid-handling records linked to
  position, and an independently randomized plate.
- Falsification-oriented recommendation: repeat the assay with conditions
  randomized across the implicated plate axis and retain matched handling data.
- Contract result: passed typed output validation and baseline evidence-ID
  validation without correction.

Review: the response preserved the important distinction between a positional
association and a specific pipetting event. Broad condition coverage made the
spatial diagnostic interpretable, but neither the prompt nor the evidence can
recover unavailable timing or dispense order.

### `case_6da112399f906fb5`

- User prompt SHA-256:
  `21b376166eba3668ebb605554db93e746ab2d520e771287fe878e4c05e4e2ba4`
- Evidence calls: complete spatial result and one complete dose-response result
  for each treatment.
- Leading hypothesis: layout-driven non-identifiability, at high confidence,
  because many conditions occupy only one row or spatial region.
- Competing hypothesis: a genuine treatment potency difference, at low
  confidence, supported by descriptive fits but contradicted by the layout
  evidence for causal interpretation.
- Missing evidence: an independent plate with each condition distributed across
  spatial strata.
- Falsification-oriented recommendation: repeat with conditions balanced across
  rows and plate regions, then test whether the treatment difference persists.
- Contract result: passed typed output validation and baseline evidence-ID
  validation without correction.

Review: the prompt successfully prioritized an experimental-design limitation
over an attractive causal interpretation of well-fitting curves. Here the
leading item is a limitation on inference rather than a claim about the physical
failure mechanism, which is appropriate for the available evidence.

### `case_db7e56f14d2ad84b`

- User prompt SHA-256:
  `745ac3035e6969ff9e860dbb46292e8f0a8158d458b4eb0fd1e90f0779761171`
- Evidence calls: one complete dose-response result for each treatment and the
  complete control summary.
- Leading hypothesis: limited test-treatment activity, at moderate confidence,
  supported by its restricted observed response and poor descriptive fit while
  the reference treatment spans a broader response.
- Competing hypothesis: a concentration or test-treatment handling problem, at
  indeterminate confidence because the observed pattern does not distinguish it
  from limited biological activity.
- Missing evidence: independent stock identity, preparation records, and
  concentration verification.
- Falsification-oriented recommendation: repeat the test treatment from an
  independently verified stock alongside the same reference and controls.
- Contract result: passed typed output validation and baseline evidence-ID
  validation without correction after semantic review of evidence roles.

Review: the prompt kept biological non-response provisional and retained a
technical alternative. During review, good global control behavior was removed
from the second hypothesis's `contradicting_evidence_ids`: it does not weigh
against a treatment-specific preparation error. Structural validation cannot
detect that semantic misuse.

## Cross-case observations

- All four assembled inputs were below the deterministic character ceiling and
  contained no private benchmark field or source case name.
- Every candidate produced two competing hypotheses, explicit missing evidence,
  a falsification check, and a discriminating next action.
- The prompt consistently discouraged causal claims from spatial association and
  encouraged the layout warning to outrank descriptive curve fits.
- Three tool calls were sufficient for these cases, but often left no call for
  `resolve_evidence`. Records returned by an inspection already contain the exact
  evidence IDs, so resolving those same records would be redundant.
- Evidence-ID existence and citation-role overlap are enforced, but whether a
  record truly supports or contradicts a narrative remains a semantic review
  problem.
- `overall_assessment`, `remaining_uncertainty`, and `limitations` have no direct
  citation fields. The prompt asks for evidence-grounded claims there, but the
  contract cannot enforce a claim-to-evidence mapping.

## Candidate changes requiring approval

No prompt change is recommended before the local Qwen capability spike. The
current prompt was adequate for these Codex rehearsals, and changing it now would
risk tuning to one model's behavior without evidence that Qwen has the same
failure pattern.

Two possible contract improvements are recorded for later consideration rather
than accepted here:

- attach evidence IDs directly to the overall assessment; and
- provide a more structured claim-to-evidence representation that can make
  semantic citation review easier, though not fully automatic.

No new regression test is added because no new prompt expectation or contract
change has yet been accepted. Existing tests already enforce citation existence,
disjoint support and contradiction roles, competing-hypothesis count, missing
evidence, alternatives, and falsification checks.

## Questions this rehearsal cannot answer about Qwen

- Will Qwen select the most informative evidence pages within the tool-call
  budget?
- Will it distinguish association, non-identifiability, and physical cause as
  consistently as Codex did?
- Will it use `supporting_evidence_ids` and `contradicting_evidence_ids`
  semantically rather than merely satisfy their shape?
- Can it reliably produce the strict structured output, including after one
  validation retry?
- What context length, output budget, latency, and memory footprint will the
  configured local model require?
- Does the installed Qwen tag work through the current Pydantic AI and Ollama
  compatibility path without a provider-specific adjustment?
