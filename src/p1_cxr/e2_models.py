"""E2-only ConvNeXt-Tiny model with a 3-class main head and 2-class subtype head."""

from __future__ import annotations

from typing import Literal, Mapping, Any

import torch
from torch import Tensor, nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from .models import FEATURE_DIMENSION, NUM_CLASSES, pretrained_weight_cache_path

SUBTYPE_CLASSES = 2
TrainableScope = Literal["head_only", "full"]


class ConvNeXtTinyE2(nn.Module):
    """Fresh E2 model: E0-equivalent encoder/main head plus Linear(768, 2).

    This class constructs a new torchvision ConvNeXt-Tiny instance. It never loads
    E0/E1 checkpoints or reuses either experiment's classifier parameters.
    """

    def __init__(
        self,
        *,
        pretrained: bool = True,
        allow_weight_download: bool = False,
        trainable_scope: TrainableScope = "full",
    ) -> None:
        super().__init__()
        weights = None
        if pretrained:
            cache_path = pretrained_weight_cache_path()
            if not cache_path.is_file() and not allow_weight_download:
                raise FileNotFoundError(
                    "ConvNeXt-Tiny IMAGENET1K_V1 weights are not cached and download is disabled: "
                    f"{cache_path}"
                )
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1

        base = convnext_tiny(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool
        self.feature_norm = base.classifier[0]
        self.feature_dim = int(base.classifier[-1].in_features)
        if self.feature_dim != FEATURE_DIMENSION:
            raise RuntimeError(
                f"Unexpected ConvNeXt-Tiny feature dimension: {self.feature_dim}"
            )
        self.classifier = nn.Linear(self.feature_dim, NUM_CLASSES)
        self.subtype_classifier = nn.Linear(self.feature_dim, SUBTYPE_CLASSES)
        self.pretrained = pretrained
        self.weights_name = (
            ConvNeXt_Tiny_Weights.IMAGENET1K_V1.name if pretrained else "NONE"
        )
        self.set_trainable_scope(trainable_scope)

    def forward_features(self, images: Tensor) -> Tensor:
        """Return the shared classifier-preceding encoder features [N, 768]."""
        feature_map = self.features(images)
        pooled = self.avgpool(feature_map)
        normalized = self.feature_norm(pooled)
        return torch.flatten(normalized, 1)

    def forward_with_subtype(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return main logits, subtype logits, and their shared encoder features."""
        features = self.forward_features(images)
        return self.classifier(features), self.subtype_classifier(features), features

    def forward(self, images: Tensor) -> Tensor:
        """Return only E0-compatible main 3-class logits for validation/inference."""
        return self.classifier(self.forward_features(images))

    def set_trainable_scope(self, scope: TrainableScope) -> None:
        if scope not in ("head_only", "full"):
            raise ValueError(f"Unsupported trainable_scope: {scope}")
        backbone_trainable = scope == "full"
        for module in (self.features, self.avgpool, self.feature_norm):
            for parameter in module.parameters():
                parameter.requires_grad = backbone_trainable
        for module in (self.classifier, self.subtype_classifier):
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.trainable_scope: TrainableScope = scope


def build_e2_model(config: Mapping[str, Any]) -> ConvNeXtTinyE2:
    """Build a fresh E2 model after checking the strict auxiliary-head contract."""
    if config.get("experiment") != "e2":
        raise ValueError("E2 model requires experiment=e2.")
    model = config["model"]
    if (
        model.get("architecture") != "flat_with_auxiliary_subtype_head"
        or int(model.get("feature_dimension", -1)) != FEATURE_DIMENSION
        or int(model.get("num_classes", -1)) != NUM_CLASSES
        or model.get("classification_head") != "Linear(768, 3)"
        or model.get("subtype_head") != "Linear(768, 2)"
    ):
        raise ValueError("E2 requires exactly Linear(768, 3) and Linear(768, 2) heads.")
    return ConvNeXtTinyE2(
        pretrained=bool(model["pretrained"]),
        allow_weight_download=bool(model["allow_weight_download"]),
        trainable_scope=str(model["trainable_scope"]),  # type: ignore[arg-type]
    )
