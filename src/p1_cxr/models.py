"""STEP 7의 ConvNeXt-Tiny E0 flat classification 모델.

E1 projection head, E2/E3 hierarchical head 및 SupConLoss는 의도적으로 포함하지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping, Any

import torch
from torch import Tensor, nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from .constants import CLASS_NAMES

FEATURE_DIMENSION = 768
NUM_CLASSES = 3
TrainableScope = Literal["head_only", "full"]


def pretrained_weight_cache_path() -> Path:
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    filename = weights.url.rsplit("/", 1)[-1]
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


class ConvNeXtTinyE0(nn.Module):
    """ConvNeXt-Tiny encoder와 Linear(768, 3) head로 구성된 E0 모델."""

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
                    "ConvNeXt-Tiny IMAGENET1K_V1 가중치가 캐시에 없습니다. "
                    "자동 다운로드는 비활성화되어 있습니다. "
                    f"예상 경로: {cache_path}"
                )
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1

        base = convnext_tiny(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool
        self.feature_norm = base.classifier[0]
        feature_dimension = int(base.classifier[-1].in_features)
        if feature_dimension != FEATURE_DIMENSION:
            raise RuntimeError(
                f"ConvNeXt-Tiny feature dimension 변경 감지: {feature_dimension}"
            )
        self.feature_dim = feature_dimension
        self.classifier = nn.Linear(self.feature_dim, NUM_CLASSES)
        self.pretrained = pretrained
        self.weights_name = (
            ConvNeXt_Tiny_Weights.IMAGENET1K_V1.name if pretrained else "NONE"
        )
        self.set_trainable_scope(trainable_scope)

    def forward_features(self, images: Tensor) -> Tensor:
        """분류 head 이전의 768차원 Encoder Feature를 반환한다."""
        feature_map = self.features(images)
        pooled = self.avgpool(feature_map)
        normalized = self.feature_norm(pooled)
        return torch.flatten(normalized, 1)

    def forward(self, images: Tensor) -> Tensor:
        """CrossEntropyLoss에 직접 전달할 3-class logits를 반환한다."""
        return self.classifier(self.forward_features(images))

    def set_trainable_scope(self, scope: TrainableScope) -> None:
        """head-only 또는 전체 fine-tuning을 위한 requires_grad를 설정한다."""
        if scope not in ("head_only", "full"):
            raise ValueError(f"지원하지 않는 trainable_scope: {scope}")
        backbone_trainable = scope == "full"
        for module in (self.features, self.avgpool, self.feature_norm):
            for parameter in module.parameters():
                parameter.requires_grad = backbone_trainable
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True
        self.trainable_scope: TrainableScope = scope

    def parameter_summary(self) -> dict[str, int | str | bool]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        head = sum(parameter.numel() for parameter in self.classifier.parameters())
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "backbone_parameters": total - head,
            "classification_head_parameters": head,
            "feature_dimension": self.feature_dim,
            "num_classes": NUM_CLASSES,
            "trainable_scope": self.trainable_scope,
            "pretrained": self.pretrained,
            "weights": self.weights_name,
        }


def build_e0_model(config: Mapping[str, Any]) -> ConvNeXtTinyE0:
    """E0 이외 실험 설정을 거부하고 flat model만 생성한다."""
    if str(config.get("experiment", "")).lower() != "e0":
        raise ValueError("STEP 7에서는 experiment=e0만 구현할 수 있습니다.")
    model_config = config["model"]
    if str(model_config.get("architecture", "")).lower() != "flat":
        raise ValueError("STEP 7 E0 architecture는 flat이어야 합니다.")
    if int(model_config.get("num_classes", -1)) != len(CLASS_NAMES):
        raise ValueError("E0 num_classes는 3이어야 합니다.")
    if str(model_config.get("weights")) != "IMAGENET1K_V1":
        raise ValueError("E0 pretrained weights는 IMAGENET1K_V1이어야 합니다.")
    return ConvNeXtTinyE0(
        pretrained=bool(model_config["pretrained"]),
        allow_weight_download=bool(model_config["allow_weight_download"]),
        trainable_scope=str(model_config["trainable_scope"]),  # type: ignore[arg-type]
    )


# 기존 import를 사용하는 코드에 명확한 E0 이름을 제공하되 구현은 하나만 유지한다.
FlatClassifier = ConvNeXtTinyE0
