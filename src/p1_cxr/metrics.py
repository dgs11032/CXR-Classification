"""E0~E3가 공통으로 재사용할 3클래스 평가 지표."""

from __future__ import annotations

from typing import Sequence, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .constants import CLASS_NAMES


def classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """확률 기반 3-class 지표를 계산하며 정의 불가능한 AUROC는 None으로 둔다."""
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 1:
        raise ValueError(f"targets는 [N]이어야 합니다: {targets.shape}")
    if probabilities.shape != (len(targets), len(CLASS_NAMES)):
        raise ValueError(
            f"probabilities는 [N, {len(CLASS_NAMES)}]이어야 합니다: "
            f"{probabilities.shape}"
        )
    if len(targets) == 0:
        raise ValueError("평가 대상이 비어 있습니다.")
    if not np.isfinite(probabilities).all():
        raise ValueError("확률에 NaN 또는 Inf가 있습니다.")
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("확률은 모두 [0, 1] 범위여야 합니다.")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("클래스 확률 합이 1이 아닙니다.")
    invalid_targets = sorted(set(targets.tolist()) - set(range(len(CLASS_NAMES))))
    if invalid_targets:
        raise ValueError(f"지원하지 않는 target: {invalid_targets}")

    predictions = probabilities.argmax(axis=1)
    labels = np.arange(len(CLASS_NAMES))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )

    per_class: dict[str, dict[str, float | int | None]] = {}
    class_aurocs: list[float] = []
    undefined_auroc_classes: list[str] = []
    for index, class_name in enumerate(CLASS_NAMES):
        binary_targets = (targets == index).astype(np.int64)
        if np.unique(binary_targets).size == 2:
            auroc: float | None = float(
                roc_auc_score(binary_targets, probabilities[:, index])
            )
            class_aurocs.append(auroc)
        else:
            auroc = None
            undefined_auroc_classes.append(class_name)
        per_class[class_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "auroc_ovr": auroc,
        }

    # Macro AUROC는 세 클래스 모두 양성·음성을 포함할 때만 보고한다.
    macro_auroc = (
        float(np.mean(class_aurocs))
        if len(class_aurocs) == len(CLASS_NAMES)
        else None
    )
    return {
        "n_samples": int(len(targets)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=labels, average="macro")),
        "macro_auroc_ovr": macro_auroc,
        "undefined_auroc_classes": undefined_auroc_classes,
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
        "per_class": per_class,
    }


def prediction_frame(
    image_paths: Sequence[str],
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Validation 순서와 일치하는 이미지별 정답·예측·확률 표를 만든다."""
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(image_paths) != len(targets) or probabilities.shape != (
        len(targets),
        len(CLASS_NAMES),
    ):
        raise ValueError("image_paths, targets, probabilities 길이 또는 shape가 다릅니다.")
    predictions = probabilities.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "image_path": list(image_paths),
            "ground_truth": targets,
            "ground_truth_name": [CLASS_NAMES[index] for index in targets],
            "predicted_class": predictions,
            "predicted_class_name": [CLASS_NAMES[index] for index in predictions],
        }
    )
    for index, class_name in enumerate(CLASS_NAMES):
        frame[f"prob_{class_name.lower()}"] = probabilities[:, index]
    return frame
