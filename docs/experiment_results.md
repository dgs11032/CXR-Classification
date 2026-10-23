# Experiment Results

## Three-Class Validation Results

| Experiment | Method | Macro-F1 |
|---|---|---:|
| E0 | Baseline | 0.8078 |
| E2 | Hierarchical | 0.8063 |
| E3 | Hierarchical | 0.8100 |
| E6 | Focal Loss (reference) | 0.8252 |
| E10 | Hierarchical | 0.8309 |
| E13 | RAD-DINO Linear Probe | 0.7846 |
| E14 | Patch Specialist | 0.8246 |
| E15 | RAD-DINO Fine-tuning | 0.8428 |
| E16 | Global-Local Fusion | 0.8337 |
| E17 | Pyramid Pooling | 0.8208 |
| E18 | Residual Fusion | 0.8304 |

These results summarize recorded validation metrics.
They should not be interpreted as independent test-set performance.
Evaluation protocols and model-selection conditions must be considered when comparing experiments.

## Selected Model

E15 achieved a recorded three-class validation Macro-F1 of 0.8428.
The three-class prediction combines a Bacterial/Viral specialist with a Normal-preserving classification pathway.

## E19 Cross-Validation

- Bacterial/Viral OOF Macro-F1: 0.7548
- This is a binary classification result and is not directly comparable with the three-class scores above.

## Provenance

Metrics were extracted from local experiment JSON artifacts:

- E0: `e0/best_validation_metrics.json` → `macro_f1`
- E2: `e2/best_validation_metrics.json` → `macro_f1`
- E3: `e3/best_validation_metrics.json` → `macro_f1`
- E6: `e10/hierarchical_metrics.json` → `e6_baseline.metrics.macro_f1`
- E10: `e10/hierarchical_metrics.json` → `e10_hierarchical.metrics.macro_f1`
- E13: `e13/validation_metrics.json` → `macro_f1`
- E14: `e14/validation_metrics.json` → `e14_normal_preserving_full_validation.macro_f1`
- E15: `e15/validation_metrics.json` → `e15_normal_preserving_full_validation.macro_f1`
- E16: `e16/validation_metrics.json` → `e16_full_validation.macro_f1`
- E17: `e17/validation_metrics.json` → `e17_normal_preserving_full_validation.macro_f1`
- E18: `e18/validation_metrics.json` → `e18_normal_preserving_full_validation.macro_f1`
- E19: `e19_cnn_bv_cv/oof_metrics.json` → `oof.macro_f1`
