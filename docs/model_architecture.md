# Model Architecture

## Research Progression

The project investigates multiple approaches for
three-class chest X-ray pneumonia classification.

### Phase 1: CNN Learning Strategies

- ConvNeXt-based classification
- Cross-Entropy and Focal Loss
- Supervised Contrastive Learning
- Auxiliary classification heads

### Phase 2: Hierarchical Classification

- Normal vs Pneumonia classification
- Bacterial vs Viral subtype classification
- Confidence-based routing analysis

### Phase 3: RAD-DINO

- E13: Linear Probe
- E14: Patch Specialist
- E15: Last-Block Fine-tuning

### Phase 4: Representation and Fusion

- E16: Global-Local Fusion
- E17: Pyramid Pooling
- E18: CNN-RAD-DINO Residual Fusion

## Selected Approach: E15

E15 uses a RAD-DINO-based Bacterial/Viral specialist.

Its predictions are integrated into a
Normal-preserving three-class classification pipeline.

Recorded three-class validation Macro-F1: 0.8428.

This is a conceptual architecture summary.
Implementation details are available in the source code.

## Demo

The current Gradio demo uses the E6 model adapter.
It should not be interpreted as an E15 deployment.
