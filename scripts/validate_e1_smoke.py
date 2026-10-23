#!/usr/bin/env python3
"""No-artifact E1 one-batch smoke test with E0/manifest immutability verification."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataloader import build_dataloaders, load_data_config  # noqa: E402
from p1_cxr.e1_losses import PneumoniaSubtypeSupConLoss  # noqa: E402
from p1_cxr.e1_training import (  # noqa: E402
    build_e1_model,
    compute_e1_losses,
    create_e1_optimizer,
    evaluate_e1,
    load_e1_training_config,
)
from p1_cxr.training import flatten_two_view_batch, resolve_device, seed_everything  # noqa: E402

PROTECTED_FILES = (
    "configs/e0_training.yaml",
    "configs/e0.json",
    "src/p1_cxr/models.py",
    "src/p1_cxr/training.py",
    "scripts/train.py",
    "src/data/dataset.py",
    "src/data/dataloader.py",
    "src/data/transforms.py",
    "artifacts/splits/train_manifest.csv",
    "artifacts/splits/val_manifest.csv",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_snapshot() -> dict[str, str]:
    snapshot = {relative: sha256(PROJECT_ROOT / relative) for relative in PROTECTED_FILES}
    for relative_root in ("artifacts/e0", "artifacts/experiments/e0"):
        root = PROJECT_ROOT / relative_root
        if root.is_dir():
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                snapshot[str(path.relative_to(PROJECT_ROOT))] = sha256(path)
    return snapshot


def balanced_validation_loader(validation_dataset, samples_per_class: int) -> DataLoader:
    frame = validation_dataset.frame
    indices: list[int] = []
    for label in (0, 1, 2):
        selected = frame.index[frame["label"] == label].tolist()[:samples_per_class]
        if len(selected) != samples_per_class:
            raise RuntimeError(f"Validation manifest lacks {samples_per_class} samples for label {label}.")
        indices.extend(selected)
    return DataLoader(Subset(validation_dataset, indices), batch_size=len(indices), shuffle=False)


def gradient_present(module: nn.Module) -> bool:
    return any(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in module.parameters() if parameter.requires_grad)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e1_training.yaml")
    parser.add_argument("--device", default=None, help="Defaults to e1_training.yaml smoke_test.device (auto).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before = protected_snapshot()
    config = load_e1_training_config(args.config, project_root=PROJECT_ROOT)
    data_config = load_data_config(config["resolved"]["data_config_path"], project_root=PROJECT_ROOT)
    if "test" in data_config.get("manifests", {}):
        raise RuntimeError("E1 smoke test must not use a test manifest.")
    seed_everything(int(config["training"]["seed"]))
    device = resolve_device(args.device or str(config["smoke_test"]["device"]))
    train_loader, validation_loader = build_dataloaders(data_config)
    model = build_e1_model(config).to(device)
    optimizer = create_e1_optimizer(model, config)
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=float(config["loss"]["classification"]["label_smoothing"]))
    supcon_criterion = PneumoniaSubtypeSupConLoss(temperature=float(config["loss"]["supervised_contrastive"]["temperature"]))
    lambda_supcon = float(config["loss"]["supervised_contrastive"]["lambda_supcon"])

    model.train()
    views, labels = next(iter(train_loader))
    flattened, expanded_labels = flatten_two_view_batch(views, labels)
    flattened, expanded_labels = flattened.to(device), expanded_labels.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits, features, projections, ce_loss, total_loss, supcon = compute_e1_losses(
        model, flattened, expanded_labels, ce_criterion=ce_criterion,
        supcon_criterion=supcon_criterion, lambda_supcon=lambda_supcon,
    )
    no_pneumonia = supcon_criterion(
        projections[:1], torch.zeros(1, dtype=expanded_labels.dtype, device=device)
    )
    zero_target_fallback_ok = (
        no_pneumonia.eligible_view_count == 0
        and no_pneumonia.loss.item() == 0.0
        and bool(torch.isfinite(no_pneumonia.loss))
    )
    if not zero_target_fallback_ok:
        raise RuntimeError("SupCon zero-target fallback is not finite zero.")
    total_loss.backward()
    gradients = {
        "encoder": gradient_present(model.features) and gradient_present(model.feature_norm),
        "classifier": gradient_present(model.classifier),
        "projector": gradient_present(model.projector),
    }
    if not all(gradients.values()):
        raise RuntimeError(f"Gradient did not reach all E1 branches: {gradients}")
    classifier_before = model.classifier.weight.detach().clone()
    optimizer.step()
    optimizer_changed_classifier = not torch.equal(classifier_before, model.classifier.weight.detach())
    if not optimizer_changed_classifier:
        raise RuntimeError("Optimizer step did not change the E1 classifier.")

    validation_subset = balanced_validation_loader(
        validation_loader.dataset, int(config["smoke_test"]["validation_samples_per_class"])
    )
    validation = evaluate_e1(model, validation_subset, ce_criterion=ce_criterion, device=device)
    metrics_ok = (
        validation.metrics["macro_f1"] is not None
        and validation.metrics["macro_auroc_ovr"] is not None
    )
    if not metrics_ok:
        raise RuntimeError("Balanced validation did not produce Macro F1/AUROC.")
    after = protected_snapshot()
    e0_protected = before == after
    if not e0_protected:
        changed = sorted(set(before) | set(after))
        changed = [path for path in changed if before.get(path) != after.get(path)]
        raise RuntimeError(f"Protected E0/manifest files changed: {changed}")

    projection_norm = torch.linalg.vector_norm(projections.detach(), dim=1)
    print(f"device: {device}")
    print(f"train input shape: {tuple(views.shape)}")
    print(f"flattened input shape: {tuple(flattened.shape)}")
    print(f"encoder feature shape: {tuple(features.shape)}")
    print(f"classifier logits shape: {tuple(logits.shape)}")
    print(f"projection shape: {tuple(projections.shape)}")
    print(f"projection L2 norm: min={projection_norm.min().item():.6f}, max={projection_norm.max().item():.6f}")
    print(f"CE loss: {ce_loss.item():.6f}")
    print(f"SupCon target source samples (Bacterial/Viral): {int(((labels == 1) | (labels == 2)).sum())}")
    print(f"SupCon target views (Bacterial/Viral): {supcon.eligible_view_count}")
    print(f"SupCon excluded Normal source samples: {int((labels == 0).sum())}")
    print(f"SupCon excluded Normal views: {supcon.excluded_normal_view_count}")
    print(f"SupCon loss: {supcon.loss.item():.6f}")
    print(f"SupCon no-pneumonia fallback (finite zero): passed={zero_target_fallback_ok}")
    print(f"total loss: {total_loss.item():.6f}")
    print(f"gradient propagation: {gradients}")
    print(f"optimizer step 1x: passed={optimizer_changed_classifier}")
    print(f"validation logits shape: {tuple(validation.logits.shape)}")
    print(f"Macro F1/AUROC calculation: passed={metrics_ok}")
    print(f"validation macro_f1: {validation.metrics['macro_f1']:.6f}")
    print(f"validation macro_auroc_ovr: {validation.metrics['macro_auroc_ovr']:.6f}")
    print(f"E0 protection verification: passed={e0_protected}; protected_files={len(after)}")


if __name__ == "__main__":
    main()
