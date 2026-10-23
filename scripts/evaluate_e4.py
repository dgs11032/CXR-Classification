#!/usr/bin/env python3
"""Read-only E4 validation evaluation; prediction is softmax(main logits) then argmax."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data.dataloader import build_dataset, build_loader, load_data_config  # noqa: E402
from p1_cxr.checkpoints import load_checkpoint  # noqa: E402
from p1_cxr.e4_models import build_e4_model  # noqa: E402
from p1_cxr.e4_training import load_e4_training_config  # noqa: E402
from p1_cxr.metrics import classification_metrics  # noqa: E402
from p1_cxr.training import resolve_device  # noqa: E402

def project_path(path: Path) -> Path:
    result = path.expanduser(); result = result.resolve() if result.is_absolute() else (PROJECT_ROOT / result).resolve()
    if not result.is_relative_to(PROJECT_ROOT): raise ValueError(f"Path must remain inside project root: {path}")
    return result

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e4_training.yaml")
parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "artifacts" / "experiments" / "e4" / "best.pt")
parser.add_argument("--device", default="auto")
args = parser.parse_args()
config = load_e4_training_config(project_path(args.config), project_root=PROJECT_ROOT)
checkpoint = project_path(args.checkpoint)
if not checkpoint.is_file(): raise FileNotFoundError(f"E4 checkpoint not found: {checkpoint}")
data_config = load_data_config(config["resolved"]["data_config_path"], project_root=PROJECT_ROOT)
if data_config["views"]["validation"] != "single" or "test" in data_config.get("manifests", {}): raise ValueError("E4 evaluation allows only deterministic validation.")
loader = build_loader(build_dataset(data_config, "validation", view_mode="single"), data_config, training=False)
device = resolve_device(args.device); model = build_e4_model(config).to(device)
payload = load_checkpoint(checkpoint, model=model, map_location=device)
if payload.get("run_kind") != "e4_full_epoch" or payload.get("config", {}).get("experiment") != "e4": raise ValueError("Checkpoint is not E4.")
model.eval(); logits, targets = [], []
with torch.inference_mode():
    for images, labels in loader:
        output = model(images.to(device))
        if output.ndim != 2 or output.shape[1] != 3 or not bool(torch.isfinite(output).all()): raise FloatingPointError("Invalid E4 validation logits.")
        logits.append(output.float().cpu()); targets.append(labels.cpu())
logits_tensor, targets_array = torch.cat(logits), torch.cat(targets).numpy()
metrics = classification_metrics(targets_array, torch.softmax(logits_tensor, dim=1).numpy())
confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
print(f"device: {device}")
print(f"Accuracy: {metrics['accuracy']:.6f}")
print(f"Macro F1: {metrics['macro_f1']:.6f}")
print(f"Macro AUROC (OvR): {metrics['macro_auroc_ovr']}")
print(f"Bacterial -> Viral: {int(confusion[1, 2])}")
print(f"Viral -> Bacterial: {int(confusion[2, 1])}")
print(f"Bacterial <-> Viral total misclassifications: {int(confusion[1, 2] + confusion[2, 1])}")
print("prediction source: softmax(main logits) -> argmax; class weights are not used")
