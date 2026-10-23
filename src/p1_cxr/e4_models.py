"""E4-only fresh model wrapper: E3/E0-identical ConvNeXt-Tiny 3-class classifier."""

from __future__ import annotations

from typing import Any, Mapping

from .e3_models import ConvNeXtTinyE3
from .models import FEATURE_DIMENSION, NUM_CLASSES


class ConvNeXtTinyE4(ConvNeXtTinyE3):
    """Fresh E0/E3-identical encoder and Linear(768, 3) classifier for E4.

    The class adds no head, loads no checkpoint, and never copies E0/E1/E2/E3
    classifier parameters. It invokes the standard ImageNet initialization policy anew.
    """


def build_e4_model(config: Mapping[str, Any]) -> ConvNeXtTinyE4:
    if config.get("experiment") != "e4":
        raise ValueError("E4 model requires experiment=e4.")
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
    mismatches = {key: {"expected": value, "actual": model.get(key)} for key, value in expected.items() if model.get(key) != value}
    if mismatches:
        raise ValueError(f"E4 model contract drift: {mismatches}")
    return ConvNeXtTinyE4(
        pretrained=bool(model["pretrained"]),
        allow_weight_download=bool(model["allow_weight_download"]),
        trainable_scope="full",
    )
