# Deterministic assay QC report

Case: `case_923eb39b90afc333`

## Findings

### Limited control separation

Control distributions show limited separation under the synthetic benchmark heuristic.

Evidence: `ev_214e6670e3728b9ba554`

## Tool results

| Tool | Status |
|---|---|
| `calculate_replicate_variability` | `success` |
| `compare_batches` | `not_applicable` |
| `detect_spatial_effects` | `success` |
| `fit_dose_response` | `success` |
| `generate_diagnostic_plot` | `success` |
| `inspect_missingness` | `success` |
| `summarize_controls` | `success` |

## Evidence index

- `ev_214e6670e3728b9ba554` — `controls.z_prime`: -0.1306871677252921 dimensionless

The canonical JSON report contains the complete evidence index and tool results.

## Plot attachments

- `raw_signal`: `raw_signal.png` (SHA-256 `c3f50f7d7da86f3ff08a4e6e1c90fd2dcb2809f66a25a0d8382fd3f7eabfadde`)
- `condition_residual`: `condition_residual.png` (SHA-256 `da7c2c1f6d355a95e59d22d0186cbff917b2bedd4640ea874dfbb36d66f4cfb3`)
- `control_distribution`: `control_distribution.png` (SHA-256 `f17a9f85d848ec6d82954d21be1f4c1ab8066f815d0abec5f6a96df1d6a3ceec`)
- `dose_response`: `dose_response.png` (SHA-256 `1b7cb620bcd91245f2032fe4043b5101d06d1746c646f6f66f302e0842833208`)

## Suggested follow-ups

- Review control identities, preparation, and assay-specific acceptance criteria.

## Limitations

- Heuristic thresholds are calibrated only for the synthetic benchmark and are not universal laboratory QC criteria.
- Localized adjacent residuals do not establish tip identity, clogging, or reload timing.
- Missingness counts identify incomplete public inputs but do not determine why values or annotations are absent.
- Plots are visual supporting evidence and are not themselves numeric diagnostics or causal conclusions.
- Spatial association does not identify a physical cause or reconstruct unavailable dispense order.
- The bounded 4PL fit is descriptive and does not establish a biological mechanism or validate the annotated concentrations.
- Z-prime is reported descriptively; acceptable values and thresholds are assay-specific.
