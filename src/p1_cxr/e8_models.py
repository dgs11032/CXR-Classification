"""E8-only fresh ImageNet ConvNeXt-Tiny Bacterial/Viral binary specialist."""

from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor, nn

from .models import ConvNeXtTinyE0, FEATURE_DIMENSION

NUM_BINARY_CLASSES = 2


class ConvNeXtTinyE8(ConvNeXtTinyE0):
    """Fresh ImageNet encoder with an E8-owned Linear(768, 2) specialist head."""

    def __init__(self, *, pretrained: bool, allow_weight_download: bool, trainable_scope: str) -> None:
        super().__init__(
            pretrained=pretrained,
            allow_weight_download=allow_weight_download,
            trainable_scope=trainable_scope,  # type: ignore[arg-type]
        )
        self.classifier = nn.Linear(FEATURE_DIMENSION, NUM_BINARY_CLASSES)

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.forward_features(images))


def build_e8_model(config: Mapping[str, Any]) -> ConvNeXtTinyE8:
    if config.get("experiment") != "e8":
        raise ValueError("E8 model requires experiment=e8.")
    model = config["model"]
    expected = {
        "architecture": "flat_binary_specialist",
        "backbone": "convnext_tiny",
        "weights": "IMAGENET1K_V1",
        "feature_dimension": FEATURE_DIMENSION,
        "num_classes": NUM_BINARY_CLASSES,
        "classification_head": "Linear(768, 2)",
        "trainable_scope": "full",
    }
    mismatches = {
        key: {"expected": value, "actual": model.get(key)}
        for key, value in expected.items()
        if model.get(key) != value
    }
    if mismatches:
        raise ValueError(f"E8 model contract drift: {mismatches}")
    return ConvNeXtTinyE8(
        pretrained=bool(model["pretrained"]),
        allow_weight_download=bool(model["allow_weight_download"]),
        trainable_scope=str(model["trainable_scope"]),
    )
