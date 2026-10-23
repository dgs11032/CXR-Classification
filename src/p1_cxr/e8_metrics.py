"""E8-only Bacterial/Viral binary metrics with explicit class order."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

from .e8_data import BINARY_CLASS_ORDER


def binary_classification_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 1 or probabilities.shape != (len(targets), 2):
        raise ValueError(f"E8 expects targets [N] and probabilities [N,2], got {targets.shape}/{probabilities.shape}")
    if len(targets) == 0 or set(targets.tolist()) - {0, 1}:
        raise ValueError("E8 binary targets must be nonempty values in {0, 1}.")
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("E8 probabilities must be finite and within [0,1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("E8 probability rows must sum to one.")
    predictions = probabilities.argmax(axis=1)
    labels = np.asarray([0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    auroc = float(roc_auc_score(targets, probabilities[:, 1])) if np.unique(targets).size == 2 else None
    return {
        "n_samples": int(len(targets)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=labels, average="macro")),
        "auroc": auroc,
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(BINARY_CLASS_ORDER)
        },
    }


def binary_prediction_frame(image_paths: Sequence[str], targets: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(image_paths) != len(targets) or probabilities.shape != (len(targets), 2):
        raise ValueError("E8 image_paths, targets, and probabilities must align.")
    predictions = probabilities.argmax(axis=1)
    return pd.DataFrame({
        "image_path": list(image_paths),
        "ground_truth": targets,
        "ground_truth_name": [BINARY_CLASS_ORDER[index] for index in targets],
        "predicted_class": predictions,
        "predicted_class_name": [BINARY_CLASS_ORDER[index] for index in predictions],
        "prob_bacterial_pneumonia": probabilities[:, 0],
        "prob_viral_pneumonia": probabilities[:, 1],
    })
