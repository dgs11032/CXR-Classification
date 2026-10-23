"""E6-only fresh model wrapper: unchanged E0 ConvNeXt-Tiny 3-class classifier."""

from __future__ import annotations
from typing import Any, Mapping
from .models import ConvNeXtTinyE0, FEATURE_DIMENSION, NUM_CLASSES


class ConvNeXtTinyE6(ConvNeXtTinyE0):
    """Fresh ConvNeXt-Tiny -> 768 -> Linear(768, 3), with no extra heads or checkpoint reuse."""


def build_e6_model(config: Mapping[str, Any]) -> ConvNeXtTinyE6:
    if config.get("experiment") != "e6":
        raise ValueError("E6 model requires experiment=e6.")
    model = config["model"]
    expected = {
        "architecture": "flat", "backbone": "convnext_tiny", "weights": "IMAGENET1K_V1",
        "feature_dimension": FEATURE_DIMENSION, "num_classes": NUM_CLASSES,
        "classification_head": "Linear(768, 3)", "trainable_scope": "full",
    }
    mismatches = {key: {"expected": value, "actual": model.get(key)} for key, value in expected.items() if model.get(key) != value}
    if mismatches:
        raise ValueError(f"E6 model contract drift: {mismatches}")
    return ConvNeXtTinyE6(pretrained=bool(model["pretrained"]), allow_weight_download=bool(model["allow_weight_download"]), trainable_scope="full")
