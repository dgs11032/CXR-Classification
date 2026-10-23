"""E5-only fresh model wrapper: unchanged E4/E0 ConvNeXt-Tiny 3-class classifier."""

from __future__ import annotations

from typing import Any, Mapping

from .e4_models import ConvNeXtTinyE4
from .models import FEATURE_DIMENSION, NUM_CLASSES


class ConvNeXtTinyE5(ConvNeXtTinyE4):
    """Fresh E4-identical ConvNeXt-Tiny -> 768 -> Linear(768, 3) model.

    No checkpoint is loaded and no classifier parameters are copied from E0–E4.
    """


def build_e5_model(config: Mapping[str, Any]) -> ConvNeXtTinyE5:
    if config.get("experiment") != "e5":
        raise ValueError("E5 model requires experiment=e5.")
    model = config["model"]
    expected = {
        "architecture": "flat", "backbone": "convnext_tiny", "weights": "IMAGENET1K_V1",
        "feature_dimension": FEATURE_DIMENSION, "num_classes": NUM_CLASSES,
        "classification_head": "Linear(768, 3)", "trainable_scope": "full",
    }
    mismatches = {key: {"expected": value, "actual": model.get(key)} for key, value in expected.items() if model.get(key) != value}
    if mismatches:
        raise ValueError(f"E5 model contract drift: {mismatches}")
    return ConvNeXtTinyE5(
        pretrained=bool(model["pretrained"]),
        allow_weight_download=bool(model["allow_weight_download"]),
        trainable_scope="full",
    )
