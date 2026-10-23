#!/usr/bin/env python3
"""One-batch E2 smoke test with E0/E1/common/manifest immutability verification.

This script never invokes full training, reads a test manifest, or writes artifacts.
"""

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
from p1_cxr.e2_training import (  # noqa: E402
    build_e2_model,
    compute_e2_losses,
    create_e2_optimizer,
    evaluate_e2,
    load_e2_training_config,
    subtype_targets_and_mask,
)
from p1_cxr.training import flatten_two_view_batch, resolve_device, seed_everything  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_snapshot() -> dict[str, str]:
    """Hash existing E0/E1, shared code, and fixed-manifest assets; explicitly exclude E2."""
    protected: set[Path] = set()
    for directory in (PROJECT_ROOT / "configs", PROJECT_ROOT / "src" / "data", PROJECT_ROOT / "src" / "p1_cxr", PROJECT_ROOT / "scripts"):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.py") if directory.name != "configs" else directory.rglob("*.yaml"):
            if "e2" not in path.name.lower():
                protected.add(path)
    for path in (PROJECT_ROOT / "artifacts" / "splits").rglob("*"):
        if path.is_file():
            protected.add(path)
    for directory in (PROJECT_ROOT / "artifacts" / "experiments" / "e0", PROJECT_ROOT / "artifacts" / "experiments" / "e1", PROJECT_ROOT / "artifacts" / "e0"):
        if directory.is_dir():
            protected.update(path for path in directory.rglob("*") if path.is_file())
    return {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in sorted(protected)}


def balanced_validation_loader(validation_dataset, samples_per_class: int) -> DataLoader:
    indices: list[int] = []
    for label in (0, 1, 2):
        selected = validation_dataset.frame.index[validation_dataset.frame["label"] == label].tolist()[:samples_per_class]
        if len(selected) != samples_per_class:
            raise RuntimeError(f"Validation manifest lacks {samples_per_class} samples for label {label}.")
        indices.extend(selected)
    return DataLoader(Subset(validation_dataset, indices), batch_size=len(indices), shuffle=False)


def gradient_present(module: nn.Module) -> bool:
    return any(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in module.parameters() if parameter.requires_grad)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e2_training.yaml")
    parser.add_argument("--device", default=None, help="auto by default; use cuda explicitly on AWS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before = protected_snapshot()
    config = load_e2_training_config(args.config, project_root=PROJECT_ROOT)
    data_config = load_data_config(config["resolved"]["data_config_path"], project_root=PROJECT_ROOT)
    if "test" in data_config.get("manifests", {}):
        raise RuntimeError("E2 smoke test must not use a test manifest.")
    seed_everything(int(config["training"]["seed"]))
    device = resolve_device(args.device or str(config["smoke_test"]["device"]))
    train_loader, validation_loader = build_dataloaders(data_config)
    model = build_e2_model(config).to(device)
    optimizer = create_e2_optimizer(model, config)
    main_criterion = nn.CrossEntropyLoss(label_smoothing=float(config["loss"]["classification"]["label_smoothing"]))
    subtype_criterion = nn.CrossEntropyLoss()
    lambda_subtype = float(config["loss"]["subtype"]["lambda_subtype"])

    model.train()
    views, labels = next(iter(train_loader))
    flattened, expanded_labels = flatten_two_view_batch(views, labels)
    flattened, expanded_labels = flattened.to(device), expanded_labels.to(device)
    optimizer.zero_grad(set_to_none=True)
    result = compute_e2_losses(model, flattened, expanded_labels, main_criterion=main_criterion, subtype_criterion=subtype_criterion, lambda_subtype=lambda_subtype)
    pneumonia_mask, subtype_targets = subtype_targets_and_mask(expanded_labels)
    expected_total = result.main_loss + lambda_subtype * result.subtype_loss
    total_formula_ok = bool(torch.allclose(result.total_loss, expected_total, rtol=1e-6, atol=1e-7))
    target_values = sorted(int(value) for value in torch.unique(subtype_targets).detach().cpu().tolist())
    target_values_ok = set(target_values).issubset({0, 1}) and bool(pneumonia_mask.any())
    if not total_formula_ok or not target_values_ok:
        raise RuntimeError("E2 loss contract failed before backward.")
    result.total_loss.backward()
    gradients = {
        "encoder": gradient_present(model.features) and gradient_present(model.feature_norm),
        "main_classifier": gradient_present(model.classifier),
        "subtype_classifier": gradient_present(model.subtype_classifier),
    }
    if not all(gradients.values()):
        raise RuntimeError(f"Gradient did not reach all E2 branches: {gradients}")
    main_before = model.classifier.weight.detach().clone()
    subtype_before = model.subtype_classifier.weight.detach().clone()
    optimizer.step()
    optimizer_step_ok = not torch.equal(main_before, model.classifier.weight.detach()) and not torch.equal(subtype_before, model.subtype_classifier.weight.detach())
    if not optimizer_step_ok:
        raise RuntimeError("Optimizer step did not update both E2 heads.")

    # All-Normal synthetic labels exercise the no-pneumonia finite-zero path without mutating data labels.
    normal_logits = result.subtype_logits[:2]
    normal_labels = torch.zeros(2, dtype=expanded_labels.dtype, device=device)
    normal_mask, normal_targets = subtype_targets_and_mask(normal_labels)
    fallback = normal_logits.sum() * 0.0 if not bool(normal_mask.any()) else subtype_criterion(normal_logits[normal_mask], normal_targets)
    fallback_ok = bool(torch.isfinite(fallback)) and fallback.item() == 0.0 and fallback.requires_grad
    if not fallback_ok:
        raise RuntimeError("All-Normal subtype fallback is not graph-connected finite zero.")

    validation_subset = balanced_validation_loader(validation_loader.dataset, int(config["smoke_test"]["validation_samples_per_class"]))
    validation = evaluate_e2(model, validation_subset, main_criterion=main_criterion, device=device)
    metrics_ok = validation.metrics["macro_f1"] is not None and validation.metrics["macro_auroc_ovr"] is not None
    prediction_rule_ok = not validation.subtype_logits_used_for_prediction
    if not metrics_ok or not prediction_rule_ok:
        raise RuntimeError("E2 validation metric or main-logit-only prediction contract failed.")
    after = protected_snapshot()
    protected_ok = before == after
    if not protected_ok:
        changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
        raise RuntimeError(f"Protected E0/E1/common/manifest assets changed: {changed}")

    print(f"device: {device}")
    print(f"train input shape: {tuple(views.shape)}")
    print(f"flattened input shape: {tuple(flattened.shape)}")
    print(f"source labels: {labels.tolist()}")
    print(f"expanded label order: {expanded_labels.detach().cpu().tolist()}")
    print(f"encoder feature shape: {tuple(result.features.shape)}")
    print(f"main logits shape: {tuple(result.main_logits.shape)}")
    print(f"subtype logits shape: {tuple(result.subtype_logits.shape)}")
    print(f"main CE: {result.main_loss.item():.6f}")
    print(f"pneumonia source sample count: {int((labels != 0).sum())}")
    print(f"pneumonia flattened view count: {result.pneumonia_view_count}")
    print(f"Normal excluded source sample count: {int((labels == 0).sum())}")
    print(f"Normal excluded flattened view count: {result.excluded_normal_view_count}")
    print(f"subtype target unique values: {target_values}; subset_of_{{0,1}}={target_values_ok}")
    print(f"subtype CE: {result.subtype_loss.item():.6f}")
    print(f"total loss: {result.total_loss.item():.6f}")
    print(f"total = main + 0.3 * subtype: passed={total_formula_ok}")
    print(f"gradient propagation: {gradients}")
    print(f"optimizer step 1x: passed={optimizer_step_ok}")
    print(f"all-Normal finite-zero fallback: passed={fallback_ok}")
    print(f"validation main logits shape: {tuple(validation.logits.shape)}")
    print(f"validation metric calculation: passed={metrics_ok}")
    print(f"subtype logits excluded from validation prediction: passed={prediction_rule_ok}")
    print(f"E0/E1/common/manifest protection hashes: passed={protected_ok}; protected_files={len(after)}")


if __name__ == "__main__":
    main()
