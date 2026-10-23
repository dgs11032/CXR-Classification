#!/usr/bin/env python3
"""Read-only full validation evaluation of an E3 checkpoint using unweighted predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataloader import build_dataset, build_loader, load_data_config  # noqa: E402
from p1_cxr.checkpoints import load_checkpoint  # noqa: E402
from p1_cxr.e3_models import build_e3_model  # noqa: E402
from p1_cxr.e3_training import load_e3_training_config  # noqa: E402
from p1_cxr.metrics import classification_metrics  # noqa: E402
from p1_cxr.training import resolve_device  # noqa: E402


def project_path(path: Path) -> Path:
    resolved = path.expanduser()
    resolved = resolved.resolve() if resolved.is_absolute() else (PROJECT_ROOT / resolved).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path must remain inside the project root: {path}")
    return resolved


@torch.inference_mode()
def infer_logits(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    for images, labels in loader:
        if images.ndim != 4:
            raise ValueError(f"Expected validation [B,C,H,W], got {tuple(images.shape)}")
        logits = model(images.to(device))
        if logits.ndim != 2 or logits.shape[1] != 3 or not bool(torch.isfinite(logits).all()):
            raise FloatingPointError(f"Invalid E3 validation logits: {tuple(logits.shape)}")
        logits_parts.append(logits.float().cpu())
        target_parts.append(labels.cpu())
    if not logits_parts:
        raise ValueError("Validation loader yielded no batches.")
    return torch.cat(logits_parts).numpy(), torch.cat(target_parts).numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e3_training.yaml")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "artifacts" / "experiments" / "e3" / "best.pt")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_e3_training_config(project_path(args.config), project_root=PROJECT_ROOT)
    checkpoint = project_path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"E3 checkpoint not found: {checkpoint}")
    data_config = load_data_config(config["resolved"]["data_config_path"], project_root=PROJECT_ROOT)
    if data_config["views"]["validation"] != "single" or "test" in data_config.get("manifests", {}):
        raise ValueError("E3 evaluation allows only the configured deterministic validation split.")
    dataset = build_dataset(data_config, "validation", view_mode="single")
    loader = build_loader(dataset, data_config, training=False)
    device = resolve_device(args.device)
    model = build_e3_model(config).to(device)
    payload = load_checkpoint(checkpoint, model=model, map_location=device)
    if payload.get("run_kind") != "e3_full_epoch" or payload.get("config", {}).get("experiment") != "e3":
        raise ValueError("Checkpoint is not an E3 full-epoch checkpoint.")
    logits, targets = infer_logits(model, loader, device)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    metrics = classification_metrics(targets, probabilities)
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    print(f"device: {device}")
    print(f"validation samples evaluated: {len(targets)}")
    print(f"checkpoint: {checkpoint}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Macro F1: {metrics['macro_f1']:.6f}")
    macro_auroc = metrics["macro_auroc_ovr"]
    print(f"Macro AUROC (OvR): {'N/A' if macro_auroc is None else f'{macro_auroc:.6f}'}")
    print(f"Bacterial -> Viral: {int(confusion[1, 2])}")
    print(f"Viral -> Bacterial: {int(confusion[2, 1])}")
    print(f"Bacterial <-> Viral total misclassifications: {int(confusion[1, 2] + confusion[2, 1])}")
    print("prediction source: softmax(main logits) -> argmax; class weights are not used")


if __name__ == "__main__":
    main()
