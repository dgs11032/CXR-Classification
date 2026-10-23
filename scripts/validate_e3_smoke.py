#!/usr/bin/env python3
"""One-batch E3 smoke test; never trains fully, writes artifacts, or reads test data."""

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
from p1_cxr.e3_models import build_e3_model  # noqa: E402
from p1_cxr.e3_training import (  # noqa: E402
    CLASS_WEIGHTS,
    create_e3_criterion,
    create_e3_optimizer,
    evaluate_e3,
    load_e3_training_config,
)
from p1_cxr.training import flatten_two_view_batch, resolve_device, seed_everything  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_snapshot() -> dict[str, str]:
    """Hash all E0/E1/E2/common source/config assets plus fixed train/val manifests."""
    protected: set[Path] = set()
    for directory, pattern in ((PROJECT_ROOT / "configs", "*.yaml"), (PROJECT_ROOT / "src" / "data", "*.py"), (PROJECT_ROOT / "src" / "p1_cxr", "*.py"), (PROJECT_ROOT / "scripts", "*.py")):
        if directory.is_dir():
            protected.update(path for path in directory.rglob(pattern) if "e3" not in path.name.lower())
    for name in ("train_manifest.csv", "val_manifest.csv"):
        path = PROJECT_ROOT / "artifacts" / "splits" / name
        if path.is_file():
            protected.add(path)
    for name in ("e0", "e1", "e2"):
        directory = PROJECT_ROOT / "artifacts" / "experiments" / name
        if directory.is_dir():
            protected.update(path for path in directory.rglob("*") if path.is_file())
    return {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in sorted(protected)}


def balanced_validation_loader(dataset, samples_per_class: int) -> DataLoader:
    indices: list[int] = []
    for label in (0, 1, 2):
        selected = dataset.frame.index[dataset.frame["label"] == label].tolist()[:samples_per_class]
        if len(selected) != samples_per_class:
            raise RuntimeError(f"Validation manifest lacks {samples_per_class} samples for label {label}.")
        indices.extend(selected)
    return DataLoader(Subset(dataset, indices), batch_size=len(indices), shuffle=False)


def gradient_present(module: nn.Module) -> bool:
    return any(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in module.parameters() if parameter.requires_grad)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e3_training.yaml")
    parser.add_argument("--device", default=None, help="auto by default; pass cuda explicitly on AWS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before = protected_snapshot()
    config = load_e3_training_config(args.config, project_root=PROJECT_ROOT)
    data_config = load_data_config(config["resolved"]["data_config_path"], project_root=PROJECT_ROOT)
    if "test" in data_config.get("manifests", {}):
        raise RuntimeError("E3 smoke test must use train/validation manifests only.")
    seed_everything(int(config["training"]["seed"]))
    device = resolve_device(args.device or str(config["smoke_test"]["device"]))
    train_loader, validation_loader = build_dataloaders(data_config)
    model = build_e3_model(config).to(device)
    criterion = create_e3_criterion(config, device=device)
    unweighted_criterion = nn.CrossEntropyLoss()
    optimizer = create_e3_optimizer(model, config)

    model.train()
    views, labels = next(iter(train_loader))
    flattened, expanded_labels = flatten_two_view_batch(views, labels)
    flattened, expanded_labels = flattened.to(device), expanded_labels.to(device)
    if not bool((expanded_labels == 1).any()):
        raise RuntimeError("Smoke batch unexpectedly lacks Bacterial labels needed for weighted-CE comparison.")
    optimizer.zero_grad(set_to_none=True)
    logits = model(flattened)
    weighted_loss = criterion(logits, expanded_labels)
    unweighted_loss = unweighted_criterion(logits, expanded_labels)
    weights_ok = criterion.weight is not None and criterion.weight.dtype == torch.float32 and torch.equal(criterion.weight.detach().cpu(), torch.tensor(CLASS_WEIGHTS))
    losses_finite = bool(torch.isfinite(weighted_loss)) and bool(torch.isfinite(unweighted_loss))
    loss_differs = not bool(torch.allclose(weighted_loss, unweighted_loss, rtol=1e-7, atol=1e-8))
    if not weights_ok or not losses_finite or not loss_differs:
        raise RuntimeError(f"E3 weighted CE contract failed: weights_ok={weights_ok}, finite={losses_finite}, differs={loss_differs}")
    weighted_loss.backward()
    gradients = {"encoder": gradient_present(model.features) and gradient_present(model.feature_norm), "classifier": gradient_present(model.classifier)}
    if not all(gradients.values()):
        raise RuntimeError(f"Gradient did not reach unchanged E3 encoder/classifier: {gradients}")
    classifier_before = model.classifier.weight.detach().clone()
    optimizer.step()
    optimizer_step_ok = not torch.equal(classifier_before, model.classifier.weight.detach())
    if not optimizer_step_ok:
        raise RuntimeError("E3 optimizer step did not update the classifier.")

    # Synthetic per-sample CE isolates the intended label-1 scaling exactly.
    sanity_logits = torch.tensor([[0.2, 1.1, -0.4], [0.2, 1.1, -0.4], [0.2, 1.1, -0.4]], dtype=torch.float32, device=device)
    sanity_labels = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
    plain_per_sample = nn.CrossEntropyLoss(reduction="none")(sanity_logits, sanity_labels)
    weighted_per_sample = nn.CrossEntropyLoss(weight=criterion.weight, reduction="none")(sanity_logits, sanity_labels)
    bacterial_ratio_ok = bool(torch.allclose(weighted_per_sample[1], plain_per_sample[1] * 0.9, rtol=1e-6, atol=1e-7))
    normal_viral_one_ok = bool(torch.allclose(weighted_per_sample[[0, 2]], plain_per_sample[[0, 2]], rtol=1e-6, atol=1e-7))
    if not bacterial_ratio_ok or not normal_viral_one_ok:
        raise RuntimeError("Per-class weighted CE sanity check failed.")

    validation_subset = balanced_validation_loader(validation_loader.dataset, int(config["smoke_test"]["validation_samples_per_class"]))
    validation = evaluate_e3(model, validation_subset, device=device)
    metrics_ok = validation.metrics["macro_f1"] is not None and validation.metrics["macro_auroc_ovr"] is not None
    prediction_weights_excluded = not validation.class_weights_used_for_prediction
    if not metrics_ok or not prediction_weights_excluded:
        raise RuntimeError("E3 validation metric or unweighted prediction contract failed.")
    after = protected_snapshot()
    protected_ok = before == after
    if not protected_ok:
        changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
        raise RuntimeError(f"Protected E0/E1/E2/common/manifest assets changed: {changed}")

    print(f"device: {device}")
    print(f"train input shape: {tuple(views.shape)}")
    print(f"flattened input shape: {tuple(flattened.shape)}")
    print(f"source labels: {labels.tolist()}")
    print(f"expanded label order: {expanded_labels.detach().cpu().tolist()}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"class weight tensor: {criterion.weight.detach().cpu().tolist()}; exact={weights_ok}")
    print(f"weighted CE loss: {weighted_loss.item():.6f}; finite={losses_finite}")
    print(f"unweighted CE loss (reference): {unweighted_loss.item():.6f}")
    print(f"weighted and unweighted CE differ: passed={loss_differs}")
    print(f"Bacterial individual CE weighted at 0.9: passed={bacterial_ratio_ok}")
    print(f"Normal/Viral individual CE weighted at 1.0: passed={normal_viral_one_ok}")
    print(f"gradient propagation: {gradients}")
    print(f"optimizer step 1x: passed={optimizer_step_ok}")
    print(f"validation logits shape: {tuple(validation.logits.shape)}")
    print(f"validation metric calculation: passed={metrics_ok}")
    print(f"class weights excluded from prediction path: passed={prediction_weights_excluded}")
    print(f"E0/E1/E2/common/train-val manifest protection hashes: passed={protected_ok}; protected_files={len(after)}")


if __name__ == "__main__":
    main()
