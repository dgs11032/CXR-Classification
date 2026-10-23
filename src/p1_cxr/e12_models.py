"""E12-only E6-transfer binary specialist model wrapper."""
from __future__ import annotations
from typing import Any, Mapping
from .e9_models import ConvNeXtTinyE9
from .models import FEATURE_DIMENSION


class ConvNeXtTinyE12(ConvNeXtTinyE9):
    """E9-identical ConvNeXt-Tiny and fresh Linear(768,2) target for E6 transfer."""


def build_e12_model(config: Mapping[str, Any]) -> ConvNeXtTinyE12:
    if config.get("experiment") != "e12": raise ValueError("E12 model requires experiment=e12.")
    model = config["model"]
    expected = {"architecture": "flat_binary_specialist", "backbone": "convnext_tiny", "weights": "IMAGENET1K_V1", "pretrained": False, "allow_weight_download": False, "feature_dimension": FEATURE_DIMENSION, "num_classes": 2, "classification_head": "Linear(768, 2)", "trainable_scope": "full"}
    mismatches = {key: {"expected": value, "actual": model.get(key)} for key, value in expected.items() if model.get(key) != value}
    if mismatches: raise ValueError(f"E12 model contract drift: {mismatches}")
    return ConvNeXtTinyE12(pretrained=False, allow_weight_download=False, trainable_scope="full")
