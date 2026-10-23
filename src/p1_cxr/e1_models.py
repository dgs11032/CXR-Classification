"""E1 전용 ConvNeXt-Tiny classifier와 pneumonia subtype SupCon projection head."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from .models import FEATURE_DIMENSION, NUM_CLASSES, pretrained_weight_cache_path

TrainableScope = Literal["head_only", "full"]


class ConvNeXtTinyE1(nn.Module):
    """E0와 동일한 768-dim encoder/3-class CE head에 E1 projector를 추가한다.

    E0 모델의 classifier 모듈을 재등록하지 않고 torchvision backbone을 새로 조립한다.
    따라서 E1 optimizer에는 E1 classifier와 projector만 포함되며 E0 parameter는 없다.
    """

    def __init__(
        self,
        *,
        pretrained: bool = True,
        allow_weight_download: bool = False,
        projection_hidden_dimension: int = 256,
        projection_dimension: int = 128,
        trainable_scope: TrainableScope = "full",
    ) -> None:
        super().__init__()
        if projection_hidden_dimension <= 0 or projection_dimension <= 0:
            raise ValueError("Projection head dimensions must be positive.")

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
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, projection_hidden_dimension),
            nn.ReLU(),
            nn.Linear(projection_hidden_dimension, projection_dimension),
        )
        self.projection_hidden_dimension = projection_hidden_dimension
        self.projection_dimension = projection_dimension
        self.pretrained = pretrained
        self.weights_name = (
            ConvNeXt_Tiny_Weights.IMAGENET1K_V1.name if pretrained else "NONE"
        )
        self.set_trainable_scope(trainable_scope)

    def forward_features(self, images: Tensor) -> Tensor:
        """Return E0-compatible classifier-preceding [B, 768] encoder features."""
        feature_map = self.features(images)
        pooled = self.avgpool(feature_map)
        normalized = self.feature_norm(pooled)
        return torch.flatten(normalized, 1)

    def project(self, features: Tensor) -> Tensor:
        """Project encoder features to normalized [B, 128] SupCon embeddings."""
        return torch.nn.functional.normalize(self.projector(features), p=2, dim=1)

    def forward_with_projection(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return logits, encoder features, and L2-normalized projection in one pass."""
        features = self.forward_features(images)
        return self.classifier(features), features, self.project(features)

    def forward(self, images: Tensor) -> Tensor:
        """Return 3-class logits for E0-equivalent CE and validation metrics."""
        return self.classifier(self.forward_features(images))

    def set_trainable_scope(self, scope: TrainableScope) -> None:
        if scope not in ("head_only", "full"):
            raise ValueError(f"Unsupported trainable_scope: {scope}")
        backbone_trainable = scope == "full"
        for module in (self.features, self.avgpool, self.feature_norm):
            for parameter in module.parameters():
                parameter.requires_grad = backbone_trainable
        for module in (self.classifier, self.projector):
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.trainable_scope: TrainableScope = scope
