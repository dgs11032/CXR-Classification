"""E3-only fresh wrapper for E0's unchanged ConvNeXt-Tiny 3-class classifier."""

from __future__ import annotations

from typing import Any, Mapping

from .models import ConvNeXtTinyE0, FEATURE_DIMENSION, NUM_CLASSES


class ConvNeXtTinyE3(ConvNeXtTinyE0):
    """A freshly constructed E0-identical model for the weighted-CE E3 ablation.

    This subclass adds no layers and does not load or copy any E0/E1/E2 checkpoint
    parameters. Its construction remains ConvNeXt-Tiny -> 768 features -> Linear(768, 3).
    """


def build_e3_model(config: Mapping[str, Any]) -> ConvNeXtTinyE3:
    """Build a fresh E3 model only after enforcing the unchanged E0 architecture."""
    if config.get("experiment") != "e3":
        raise ValueError("E3 model requires experiment=e3.")
    model = config["model"]
    expected = {
        "architecture": "flat",
        "backbone": "convnext_tiny",
        "weights": "IMAGENET1K_V1",
        "feature_dimension": FEATURE_DIMENSION,
        "num_classes": NUM_CLASSES,
        "classification_head": "Linear(768, 3)",
        "trainable_scope": "full",
    }
    mismatches = {
        key: {"expected": value, "actual": model.get(key)}
        for key, value in expected.items()
        if model.get(key) != value
    }
    if mismatches:
        raise ValueError(f"E3 model contract drift: {mismatches}")
    return ConvNeXtTinyE3(
        pretrained=bool(model["pretrained"]),
        allow_weight_download=bool(model["allow_weight_download"]),
        trainable_scope="full",
    )
