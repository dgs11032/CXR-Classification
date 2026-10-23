# Key Findings

## 1. Loss Function Optimization

Focal Loss improved classification performance
in the early CNN experiments.

E6 reference Macro-F1: 0.8252.

However, modifying the loss function alone
did not fully resolve pneumonia subtype confusion.

## 2. Hierarchical Classification

E10 improved three-class validation Macro-F1
to 0.8309.

This supported investigating pneumonia detection
and subtype classification as related tasks.

## 3. RAD-DINO Representation

E13 Linear Probe achieved a Macro-F1 of 0.7846.

The later E15 fine-tuning and Normal-preserving
combination achieved a three-class Macro-F1 of 0.8428.

These findings suggest that adaptation strategy and
prediction integration matter, rather than simply
selecting a pretrained encoder.

## 4. Fusion Architecture

E16 achieved 0.8337.
E17 achieved 0.8208.
E18 achieved 0.8304.

More complex feature fusion did not consistently
outperform E15 under the recorded evaluations.

## 5. Cross-Validation

E19 Bacterial/Viral OOF Macro-F1: 0.7548.

This is a binary cross-validation result and
must not be directly compared with three-class
validation metrics.

## Limitations

- Reported values are validation results, not
  independent clinical test-set performance.
- Some experiments use different classification
  structures and evaluation settings.
- Clinical applicability has not been established.
