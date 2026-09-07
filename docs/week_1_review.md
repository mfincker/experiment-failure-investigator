# Week 1 Benchmark Review

## Review status

Automated validation, visual calibration, and human approval were completed on
2026-09-07. Open
[`cases/review.html`](../cases/review.html) to review every clean/injected
comparison; each case retains its individual source plots.

## Reproducibility and structure

- Fourteen of fourteen registered cases load and validate.
- A clean second generation produced byte-identical output for all 156 files,
  including PNG plots.
- The twelve ordinary cases share layout fingerprint
  `8063a5e5d0fe1c3eef78698f569998728b67949763aa44d89ee90572e28b1c88`.
- The two layout-confounding cases share the distinct fingerprint
  `76f2d1a147daa21c3ae163400c21b01ef7de654389924bdb0f723238e9c96193`.
- This is a fixed-layout development slice and cannot support claims of
  generalization to novel plate layouts.

## Initial visual calibration

| Case | Initial assessment | Primary view | Important limitation |
|---|---|---|---|
| `edge_effect_obvious` | Clear four-sided boundary residual | Residual heatmap | Spatial pattern does not prove physical cause |
| `edge_effect_noisy` | Visible but less uniform top/right boundary effect | Residual heatmap | Fixed layout may become a positional shortcut |
| `pipetting_drift_obvious` | Clear traversal-wide gradient | Residual heatmap | Timing and temperature gradients remain alternatives |
| `pipetting_drift_noisy` | Detectable gradual trend amid well noise | Residual heatmap | Dispense order is private and cannot be confirmed |
| `transient_tip_clog_obvious` | Clear localized four-group sequence | Residual heatmap | Actual liquid-handler geometry varies |
| `transient_tip_clog_noisy` | Local sequence remains visible but is plausibly ambiguous | Residual heatmap | Could resemble a localized readout or annotation problem |
| `layout_confounding_obvious` | Strong condition-position blocks | Raw heatmap and plate map | Root cause is intentionally non-identifiable |
| `layout_confounding_noisy` | Confounded assignment remains structurally clear | Raw heatmap and plate map | Residual centering can conceal the design problem |
| `weak_controls_obvious` | Control separation collapses clearly | Control QC | Incorrect labels remain an alternative |
| `weak_controls_noisy` | Reduced separation remains visible | Control QC | Formal acceptance thresholds are not yet implemented |
| `batch_shift_obvious` | Plate 2 dynamic range is clearly compressed | Dose response and control QC | Represents response scaling, not an additive normalized shift |
| `batch_shift_noisy` | Between-plate compression remains detectable | Dose response and control QC | Requires like-for-like plate comparisons |
| `true_non_response_obvious` | Test treatment is flat with healthy comparators | Dose response | Preparation or dose annotation could mimic biology |
| `true_non_response_noisy` | Flat test response remains clear despite noise | Dose response | Benchmark truth cannot resolve real-world preparation errors |

## Human sign-off checklist

- [x] Labels and numeric annotations are readable at the intended review size.
- [x] Plate orientation is intuitive: row A is at the top and column 1 at the left.
- [x] Obvious variants are unambiguous enough for baseline evaluation.
- [x] Noisy variants remain detectable without becoming trivial.
- [x] The transient-tip-clog pattern is plausible for the intended dispensing abstraction.
- [x] The batch response-scale interpretation matches the normalized assay model.
- [x] The fixed-layout limitation is prominent enough for future result reporting.
