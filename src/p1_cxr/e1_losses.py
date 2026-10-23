"""E1 전용 pneumonia subtype-only supervised contrastive loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SupConResult:
    """SupCon scalar와 mask 감사 정보를 함께 반환한다."""

    loss: Tensor
    eligible_view_count: int
    excluded_normal_view_count: int
    valid_anchor_count: int


class PneumoniaSubtypeSupConLoss(nn.Module):
    """Bacterial(1) / Viral(2) view만 이용하는 numerically safe SupCon loss.

    Dataset labels are never remapped or mutated. Label 0 views are removed before
    all pairwise similarity operations. If no subtype views or positive pairs exist,
    a graph-connected zero scalar is returned instead of NaN.
    """

    def __init__(self, *, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        self.temperature = float(temperature)

    def forward(self, projections: Tensor, labels: Tensor) -> SupConResult:
        if projections.ndim != 2:
            raise ValueError(f"projections must be [N, D], got {tuple(projections.shape)}")
        if labels.ndim != 1 or len(labels) != len(projections):
            raise ValueError(
                "labels must be [N] and match the number of projections: "
                f"labels={tuple(labels.shape)}, projections={tuple(projections.shape)}"
            )
        if not torch.isfinite(projections).all():
            raise FloatingPointError("SupCon projections contain NaN or Inf.")

        subtype_mask = (labels == 1) | (labels == 2)
        subtype_projections = projections[subtype_mask]
        subtype_labels = labels[subtype_mask]
        eligible_view_count = int(subtype_mask.sum().detach().cpu())
        excluded_normal_view_count = int((labels == 0).sum().detach().cpu())

        # Keeps dtype/device/gradient connectivity without creating NaNs.
        zero = projections.sum() * 0.0
        if eligible_view_count < 2:
            return SupConResult(zero, eligible_view_count, excluded_normal_view_count, 0)

        normalized = torch.nn.functional.normalize(subtype_projections, p=2, dim=1)
        similarities = normalized @ normalized.T / self.temperature
        eye = torch.eye(eligible_view_count, dtype=torch.bool, device=projections.device)
        similarities = similarities - similarities.max(dim=1, keepdim=True).values.detach()
        logits = similarities.masked_fill(eye, float("-inf"))
        log_denominator = torch.logsumexp(logits, dim=1)

        positive_mask = subtype_labels[:, None].eq(subtype_labels[None, :]) & ~eye
        positive_count = positive_mask.sum(dim=1)
        valid_anchors = positive_count > 0
        valid_anchor_count = int(valid_anchors.sum().detach().cpu())
        if valid_anchor_count == 0:
            return SupConResult(zero, eligible_view_count, excluded_normal_view_count, 0)

        log_probabilities = logits - log_denominator[:, None]
        positive_log_probabilities = log_probabilities.masked_fill(~positive_mask, 0.0)
        mean_log_probability = positive_log_probabilities.sum(dim=1) / positive_count.clamp_min(1)
        loss = -mean_log_probability[valid_anchors].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("SupCon loss became NaN or Inf.")
        return SupConResult(
            loss, eligible_view_count, excluded_normal_view_count, valid_anchor_count
        )
