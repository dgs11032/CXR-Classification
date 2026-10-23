#!/usr/bin/env python3
"""Read-only E1 best-checkpoint evaluation on the fixed validation manifest only.

This script never trains, regenerates a split, accesses a test manifest, or writes a
checkpoint/artifact. It reuses the existing deterministic single-view validation
Dataset/DataLoader and the shared three-class metrics utility.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataloader import build_dataset, build_loader, load_data_config  # noqa: E402
from p1_cxr.checkpoints import load_checkpoint  # noqa: E402
from p1_cxr.constants import CLASS_NAMES  # noqa: E402
from p1_cxr.e1_training import build_e1_model, load_e1_training_config  # noqa: E402
from p1_cxr.metrics import classification_metrics  # noqa: E402
from p1_cxr.training import resolve_device  # noqa: E402

DISPLAY_NAMES = ("Normal", "Bacterial Pneumonia", "Viral Pneumonia")


def _project_path(path: Path) -> Path:
    resolved = path.expanduser()
    resolved = resolved.resolve() if resolved.is_absolute() else (PROJECT_ROOT / resolved).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path must be inside the project root: {path}")
    return resolved


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "e1_training.yaml",
        help="E1 training configuration used to reconstruct the model architecture.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "experiments" / "e1" / "best.pt",
        help="Read-only E1 best checkpoint path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cuda, mps, or cpu. Use --device cuda on AWS GPU.",
    )
    return parser.parse_args()


@torch.inference_mode()
def run_validation_inference(
    model: torch.nn.Module,
    validation_loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run deterministic single-view validation inference without gradients."""
    model.eval()
    logits_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    for images, labels in validation_loader:
        if images.ndim != 4:
            raise ValueError(
                "Validation loader must provide deterministic single-view [B,C,H,W] tensors, "
                f"got {tuple(images.shape)}"
            )
        logits = model(images.to(device))
        if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES):
            raise ValueError(f"Expected validation logits [B, 3], got {tuple(logits.shape)}")
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Validation logits contain NaN or Inf.")
        logits_parts.append(logits.float().cpu())
        target_parts.append(labels.cpu())
    if not logits_parts:
        raise ValueError("Validation loader yielded no batches.")
    return torch.cat(logits_parts).numpy(), torch.cat(target_parts).numpy()


def print_checkpoint_summary(checkpoint_path: Path, payload: dict[str, Any]) -> None:
    best_f1 = payload.get("best_validation_macro_f1")
    print("\n=== Checkpoint ===")
    print(f"checkpoint path: {checkpoint_path}")
    print(f"saved epoch: {payload.get('epoch', 'N/A')}")
    print(f"saved best validation Macro F1: {_format_metric(best_f1)}")


def print_metrics(metrics: dict[str, Any], targets: np.ndarray, probabilities: np.ndarray) -> None:
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    actual_counts = np.bincount(targets, minlength=len(CLASS_NAMES))
    predicted_counts = np.bincount(predictions, minlength=len(CLASS_NAMES))

    print("\n=== Full validation performance ===")
    print(f"Accuracy: {_format_metric(metrics['accuracy'])}")
    print(f"Macro F1: {_format_metric(metrics['macro_f1'])}")
    print(f"Macro AUROC (OvR): {_format_metric(metrics['macro_auroc_ovr'])}")

    print("\n=== Class-wise performance and counts ===")
    for index, class_name in enumerate(CLASS_NAMES):
        class_metrics = metrics["per_class"][class_name]
        print(
            f"{index} = {DISPLAY_NAMES[index]} | "
            f"actual={actual_counts[index]} | predicted={predicted_counts[index]} | "
            f"precision={_format_metric(class_metrics['precision'])} | "
            f"recall={_format_metric(class_metrics['recall'])} | "
            f"F1={_format_metric(class_metrics['f1'])}"
        )

    print("\n=== Confusion matrix (rows=true, columns=predicted) ===")
    header = " " * 25 + "  ".join(f"pred {name}" for name in DISPLAY_NAMES)
    print(header)
    for index, name in enumerate(DISPLAY_NAMES):
        values = "  ".join(f"{value:>18d}" for value in confusion[index])
        print(f"true {name:<20}  {values}")

    bacterial_to_viral = int(confusion[1, 2])
    viral_to_bacterial = int(confusion[2, 1])
    print("\n=== Pneumonia subtype confusion ===")
    print(f"Bacterial -> Viral: {bacterial_to_viral}")
    print(f"Viral -> Bacterial: {viral_to_bacterial}")
    print(f"Bacterial <-> Viral total misclassifications: {bacterial_to_viral + viral_to_bacterial}")


def main() -> None:
    args = parse_args()
    config = load_e1_training_config(_project_path(args.config), project_root=PROJECT_ROOT)
    checkpoint_path = _project_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"E1 checkpoint not found: {checkpoint_path}")

    # Reuses configs/data_config.yaml, its fixed val_manifest.csv, and its validation
    # transform/view_mode/label mapping. No test Dataset is built or accessed.
    data_config = load_data_config(config["resolved"]["data_config_path"], project_root=PROJECT_ROOT)
    if data_config["views"]["validation"] != "single":
        raise ValueError("E1 evaluation requires the configured deterministic single-view validation loader.")
    validation_dataset = build_dataset(data_config, "validation", view_mode="single")
    validation_loader = build_loader(validation_dataset, data_config, training=False)

    device = resolve_device(args.device)
    model = build_e1_model(config).to(device)
    payload = load_checkpoint(checkpoint_path, model=model, map_location=device)
    saved_config = payload.get("config", {})
    if saved_config.get("experiment") != "e1":
        raise ValueError(f"Checkpoint is not an E1 checkpoint: {saved_config.get('experiment')}")

    logits, targets = run_validation_inference(model, validation_loader, device=device)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    metrics = classification_metrics(targets, probabilities)

    print(f"device: {device}")
    print(f"validation manifest: {data_config['resolved']['validation_manifest']}")
    print(f"validation samples evaluated: {len(targets)}")
    print_checkpoint_summary(checkpoint_path, payload)
    print_metrics(metrics, targets, probabilities)


if __name__ == "__main__":
    main()
