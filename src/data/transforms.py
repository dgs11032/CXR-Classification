"""ConvNeXt-Tiny용 공통 CXR 전처리와 분석용 역정규화 함수.

확정된 STEP 6 설정만 구현한다. Crop, flip, scale jitter는 포함하지 않는다.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torchvision import transforms as T
from torchvision.models import ConvNeXt_Tiny_Weights
from torchvision.transforms import InterpolationMode

_INTERPOLATION = {
    "nearest": InterpolationMode.NEAREST,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
}


def convnext_weights_metadata() -> dict[str, Any]:
    """torchvision의 ConvNeXt-Tiny 기본 weights 전처리 메타데이터를 반환한다."""
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    preset = weights.transforms()
    return {
        "weights": weights.name,
        "mean": list(preset.mean),
        "std": list(preset.std),
        "interpolation": preset.interpolation.value,
        "antialias": bool(preset.antialias),
    }


def _validate_transform_config(config: Mapping[str, Any]) -> None:
    image_size = config.get("image_size")
    if image_size != [224, 224]:
        raise ValueError(f"STEP 6 image_size는 [224, 224]여야 합니다: {image_size}")
    if config.get("resize") != "direct":
        raise ValueError("STEP 6에서는 Direct Resize만 허용합니다.")
    if config.get("channel_policy") != "grayscale_then_rgb3":
        raise ValueError("채널 정책은 grayscale_then_rgb3여야 합니다.")

    metadata = convnext_weights_metadata()
    normalization = config.get("normalization", {})
    configured_mean = [float(value) for value in normalization.get("mean", [])]
    configured_std = [float(value) for value in normalization.get("std", [])]
    if configured_mean != metadata["mean"] or configured_std != metadata["std"]:
        raise ValueError(
            "정규화 Mean/Std가 ConvNeXt-Tiny ImageNet weights 메타데이터와 다릅니다."
        )
    configured_interpolation = str(config.get("interpolation", "")).lower()
    if configured_interpolation != metadata["interpolation"]:
        raise ValueError(
            "보간 방식이 ConvNeXt-Tiny weights 메타데이터와 다릅니다: "
            f"{configured_interpolation} != {metadata['interpolation']}"
        )
    if bool(config.get("antialias")) != metadata["antialias"]:
        raise ValueError("antialias 설정이 ConvNeXt-Tiny weights 메타데이터와 다릅니다.")


def build_transform(config: Mapping[str, Any], *, training: bool) -> T.Compose:
    """Grayscale→RGB3→Direct Resize→선택적 약한 증강→Tensor→Normalize를 구성한다."""
    _validate_transform_config(config)
    image_size = tuple(int(value) for value in config["image_size"])
    interpolation_name = str(config["interpolation"]).lower()
    if interpolation_name not in _INTERPOLATION:
        raise ValueError(f"지원하지 않는 보간 방식: {interpolation_name}")
    interpolation = _INTERPOLATION[interpolation_name]
    augmentation = config["augmentation"]

    operations: list[Any] = [
        # 원본 L/RGB 저장 모드 차이를 제거하고 동일 값의 RGB 3채널을 만든다.
        T.Grayscale(num_output_channels=3),
        T.Resize(
            image_size,
            interpolation=interpolation,
            antialias=bool(config["antialias"]),
        ),
    ]
    if training:
        rotation = augmentation["rotation"]
        if bool(rotation["enabled"]):
            operations.append(
                T.RandomApply(
                    [
                        T.RandomRotation(
                            degrees=float(rotation["degrees"]),
                            interpolation=interpolation,
                            expand=False,
                            fill=int(rotation["fill"]),
                        )
                    ],
                    p=float(rotation["probability"]),
                )
            )
        jitter = augmentation["brightness_contrast"]
        if bool(jitter["enabled"]):
            operations.append(
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=float(jitter["brightness"]),
                            contrast=float(jitter["contrast"]),
                        )
                    ],
                    p=float(jitter["probability"]),
                )
            )
        forbidden = {
            "horizontal_flip": augmentation.get("horizontal_flip"),
            "random_crop": augmentation.get("random_crop"),
            "scale_jitter": augmentation.get("scale_jitter"),
        }
        enabled_forbidden = [name for name, enabled in forbidden.items() if bool(enabled)]
        if enabled_forbidden:
            raise ValueError(f"STEP 6에서 금지된 증강이 활성화되었습니다: {enabled_forbidden}")

    normalization = config["normalization"]
    operations.extend(
        [
            T.ToTensor(),
            T.Normalize(
                mean=[float(value) for value in normalization["mean"]],
                std=[float(value) for value in normalization["std"]],
            ),
        ]
    )
    return T.Compose(operations)


def denormalize(
    tensor: Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tensor:
    """시각화를 위해 CHW/BCHW Tensor를 역정규화하고 [0, 1]로 제한한다."""
    if tensor.ndim not in (3, 4):
        raise ValueError(f"역정규화 입력은 CHW 또는 BCHW여야 합니다: {tuple(tensor.shape)}")
    channel_axis_shape = (1, -1, 1, 1) if tensor.ndim == 4 else (-1, 1, 1)
    mean_tensor = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(
        channel_axis_shape
    )
    std_tensor = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(
        channel_axis_shape
    )
    return (tensor * std_tensor + mean_tensor).clamp(0.0, 1.0)
