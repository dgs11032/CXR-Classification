#!/usr/bin/env python3
"""Read-only deterministic validation evaluation of an E8 binary specialist checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataloader import build_loader, load_data_config  # noqa: E402
from p1_cxr.checkpoints import load_checkpoint  # noqa: E402
from p1_cxr.e8_data import BINARY_CLASS_ORDER, build_e8_dataset  # noqa: E402
from p1_cxr.e8_metrics import binary_classification_metrics  # noqa: E402
from p1_cxr.e8_models import build_e8_model  # noqa: E402
from p1_cxr.e8_training import load_e8_training_config  # noqa: E402
from p1_cxr.training import resolve_device  # noqa: E402


def inside_project(path: Path) -> Path:
    resolved = path.expanduser()
    resolved = resolved.resolve() if resolved.is_absolute() else (PROJECT_ROOT / resolved).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path must remain inside project root: {path}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e8_training.yaml")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "artifacts" / "experiments" / "e8" / "best.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_e8_training_config(inside_project(args.config), project_root=PROJECT_ROOT)
    checkpoint = inside_project(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"E8 checkpoint not found: {checkpoint}")
    data_config = load_data_config(config["resolved"]["data_config_path"], project_root=PROJECT_ROOT)
    if data_config["views"]["validation"] != "single" or "test" in data_config.get("manifests", {}):
        raise ValueError("E8 evaluation allows deterministic binary validation only; test is forbidden.")
    dataset = build_e8_dataset(data_config, "validation")
    loader = build_loader(dataset, data_config, training=False)
    if len(dataset) != 550:
        raise ValueError("E8 evaluation requires the fixed 550-image Bacterial/Viral validation subset.")

    device = resolve_device(args.device)
    model = build_e8_model(config).to(device)
    payload = load_checkpoint(checkpoint, model=model, map_location=device)
    if payload.get("run_kind") != "e8_binary_full_epoch" or payload.get("config", {}).get("experiment") != "e8":
        raise ValueError("Checkpoint is not an E8 binary-specialist checkpoint.")
    if payload.get("control_state", {}).get("binary_class_order") != list(BINARY_CLASS_ORDER):
        raise ValueError("E8 checkpoint binary class order does not match Bacterial/Viral.")

    logits, targets = [], []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            output = model(images.to(device))
            if output.ndim != 2 or output.shape[1] != 2 or not bool(torch.isfinite(output).all()):
                raise FloatingPointError("Invalid E8 validation logits.")
            logits.append(output.float().cpu())
            targets.append(labels.cpu())
    probabilities = torch.softmax(torch.cat(logits), dim=1).numpy()
    targets_array = torch.cat(targets).numpy()
    metrics = binary_classification_metrics(targets_array, probabilities)
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)

    print(f"device: {device}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Macro F1: {metrics['macro_f1']:.6f}")
    print(f"AUROC (Viral positive): {metrics['auroc']}")
    print(f"Bacterial -> Viral: {int(confusion[0, 1])}")
    print(f"Viral -> Bacterial: {int(confusion[1, 0])}")
    print("confusion matrix [rows=true, columns=pred; Bacterial, Viral]:")
    print(confusion)
    print("per-class precision / recall / F1:")
    for class_name in BINARY_CLASS_ORDER:
        values = metrics["per_class"][class_name]
        print(
            f"{class_name}: precision={values['precision']:.6f}, recall={values['recall']:.6f}, "
            f"f1={values['f1']:.6f}, support={values['support']}"
        )
    print("prediction source: softmax(binary logits) -> argmax; Focal Loss is not used in prediction")


if __name__ == "__main__":
    main()
