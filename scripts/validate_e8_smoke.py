#!/usr/bin/env python3
"""One-batch E8 binary-specialist smoke test: no full training, artifact writes, or test access."""

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

from data.dataloader import load_data_config  # noqa: E402
from p1_cxr.e8_data import BINARY_CLASS_ORDER, EXPECTED_COUNTS, build_e8_dataloaders  # noqa: E402
from p1_cxr.e8_metrics import binary_classification_metrics  # noqa: E402
from p1_cxr.e8_models import build_e8_model  # noqa: E402
from p1_cxr.e8_training import (  # noqa: E402
    GAMMA,
    BinaryMulticlassFocalLoss,
    create_e8_criterion,
    create_e8_optimizer,
    evaluate_e8,
    load_e8_training_config,
)
from p1_cxr.training import flatten_two_view_batch, resolve_device, seed_everything  # noqa: E402

E8_ARTIFACT = PROJECT_ROOT / "artifacts" / "experiments" / "e8"


def snapshot() -> dict[str, str]:
    protected: set[Path] = set()
    for directory, pattern in (
        (PROJECT_ROOT / "configs", "*.yaml"),
        (PROJECT_ROOT / "src" / "data", "*.py"),
        (PROJECT_ROOT / "src" / "p1_cxr", "*.py"),
        (PROJECT_ROOT / "scripts", "*.py"),
    ):
        if directory.is_dir():
            protected.update(path for path in directory.rglob(pattern) if "e8" not in path.name.lower())
    for name in ("train_manifest.csv", "val_manifest.csv"):
        path = PROJECT_ROOT / "artifacts" / "splits" / name
        if path.is_file():
            protected.add(path)
    for name in ("e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7"):
        directory = PROJECT_ROOT / "artifacts" / "experiments" / name
        if directory.is_dir():
            protected.update(path for path in directory.rglob("*") if path.is_file())
    return {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(protected)
    }


def balanced_loader(dataset: object, per_class: int) -> DataLoader:
    frame = dataset.frame
    indices: list[int] = []
    for label in (0, 1):
        selected = frame.index[frame["label"] == label].tolist()[:per_class]
        if len(selected) != per_class:
            raise RuntimeError(f"E8 validation lacks {per_class} samples for binary label {label}.")
        indices.extend(selected)
    return DataLoader(Subset(dataset, indices), batch_size=len(indices), shuffle=False)


def has_gradient(module: nn.Module) -> bool:
    return any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e8_training.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    artifact_absent_before = not E8_ARTIFACT.exists()
    if not artifact_absent_before:
        raise RuntimeError("E8 artifact directory already exists; smoke test must not write or reuse it.")
    before = snapshot()
    config = load_e8_training_config(args.config, project_root=PROJECT_ROOT)
    data_config = load_data_config(config["resolved"]["data_config_path"], project_root=PROJECT_ROOT)
    if "test" in data_config.get("manifests", {}):
        raise RuntimeError("E8 smoke test must not use a test manifest.")
    seed_everything(int(config["training"]["seed"]))
    device = resolve_device(args.device or str(config["smoke_test"]["device"]))
    train_loader, validation_loader = build_e8_dataloaders(data_config)

    train_counts = train_loader.dataset.class_counts()
    validation_counts = validation_loader.dataset.class_counts()
    subsets_ok = (
        len(train_loader.dataset) == 3300
        and len(validation_loader.dataset) == 550
        and train_counts == EXPECTED_COUNTS["train"]
        and validation_counts == EXPECTED_COUNTS["validation"]
        and set(train_loader.dataset.frame["class_name"].unique()) == set(BINARY_CLASS_ORDER)
        and set(train_loader.dataset.frame["canonical_label"].unique()) == {1, 2}
        and set(train_loader.dataset.frame["label"].unique()) == {0, 1}
    )
    if not subsets_ok:
        raise RuntimeError("E8 fixed-manifest Bacterial/Viral filtering or remapping contract failed.")

    model = build_e8_model(config).to(device)
    criterion = create_e8_criterion(config)
    optimizer = create_e8_optimizer(model, config)
    views, labels = next(iter(train_loader))
    flattened, expanded = flatten_two_view_batch(views, labels)
    flattened, expanded = flattened.to(device), expanded.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(flattened)
    components = criterion.components(logits, expanded)
    focal_loss = components.per_sample_focal.mean()
    ce_loss = nn.CrossEntropyLoss()(logits, expanded)
    pt_bounded = bool((components.pt >= 0.0).all() and (components.pt <= 1.0).all())
    finite = all(
        bool(torch.isfinite(value).all())
        for value in (components.per_sample_ce, components.pt, components.modulation, components.per_sample_focal, focal_loss)
    )
    differs = not bool(torch.allclose(focal_loss, ce_loss, rtol=1e-7, atol=1e-8))
    settings_ok = (
        criterion.gamma == 2.0
        and criterion.alpha is None
        and GAMMA == 2.0
        and config["loss"]["class_weights"] is None
        and config["loss"]["label_smoothing"] == 0.0
    )
    if not settings_ok or not pt_bounded or not finite or not differs:
        raise RuntimeError(
            f"E8 focal contract failed: settings={settings_ok}, pt_bounded={pt_bounded}, finite={finite}, differs={differs}"
        )

    focal_loss.backward()
    gradients = {
        "encoder": has_gradient(model.features) and has_gradient(model.feature_norm),
        "classifier": has_gradient(model.classifier),
    }
    if not all(gradients.values()):
        raise RuntimeError(f"Missing E8 gradients: {gradients}")
    head_before = model.classifier.weight.detach().clone()
    optimizer.step()
    step_ok = not torch.equal(head_before, model.classifier.weight.detach())
    if not step_ok:
        raise RuntimeError("E8 optimizer step did not update the binary classifier.")

    sanity_logits = torch.tensor([[6.0, 0.0], [0.2, 0.0]], dtype=torch.float32, device=device)
    sanity_targets = torch.tensor([0, 0], dtype=torch.long, device=device)
    sanity = criterion.components(sanity_logits, sanity_targets)
    expected_modulation = (1.0 - torch.exp(-sanity.per_sample_ce)).pow(2.0)
    modulation_correct = bool(torch.allclose(sanity.modulation, expected_modulation, rtol=1e-6, atol=1e-7))
    hard_contribution_larger = (
        bool(sanity.per_sample_focal[1] > sanity.per_sample_focal[0])
        and bool(sanity.modulation[1] > sanity.modulation[0])
    )
    if not modulation_correct or not hard_contribution_larger:
        raise RuntimeError("E8 hard/easy gamma=2 focal sanity check failed.")

    validation = evaluate_e8(
        model,
        balanced_loader(validation_loader.dataset, int(config["smoke_test"]["validation_samples_per_class"])),
        device=device,
    )
    metrics_ok = validation.metrics["macro_f1"] is not None and validation.metrics["auroc"] is not None
    prediction_path_ok = not validation.focal_used_for_prediction and validation.logits.shape == (4, 2)
    if not metrics_ok or not prediction_path_ok:
        raise RuntimeError("E8 binary validation metric/prediction contract failed.")

    after = snapshot()
    protected_ok = before == after
    artifact_absent_after = not E8_ARTIFACT.exists()
    if not protected_ok:
        changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
        raise RuntimeError(f"Protected E0-E7/common/manifest files changed: {changed}")
    if not artifact_absent_after:
        raise RuntimeError("Smoke test unexpectedly created artifacts/experiments/e8.")

    print(f"device: {device}")
    print(f"E8 source subset counts: train={train_counts}; validation={validation_counts}; passed={subsets_ok}")
    print(f"train input shape: {tuple(views.shape)}")
    print(f"flattened input shape: {tuple(flattened.shape)}")
    print(f"source binary labels: {labels.tolist()}")
    print(f"expanded binary label order: {expanded.detach().cpu().tolist()}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"binary class order: {list(BINARY_CLASS_ORDER)}")
    print(f"gamma: {criterion.gamma}; alpha: {criterion.alpha}; class weights: None")
    print(f"per-sample CE shape: {tuple(components.per_sample_ce.shape)}")
    print(f"pt min/max: {components.pt.min().item():.6f}/{components.pt.max().item():.6f}; bounded={pt_bounded}")
    print(f"focal modulation (1-pt)^2 finite: {bool(torch.isfinite(components.modulation).all())}")
    print(f"focal loss: {focal_loss.item():.6f}; finite={finite}; reference CE={ce_loss.item():.6f}; differs={differs}")
    print(f"gamma=2 modulation calculation: passed={modulation_correct}")
    print(f"hard sample focal contribution > easy sample: passed={hard_contribution_larger}")
    print(f"gradient propagation: {gradients}; optimizer step 1x: passed={step_ok}")
    print(f"validation logits shape: {tuple(validation.logits.shape)}; metrics passed={metrics_ok}")
    print(f"Focal Loss excluded from prediction path: passed={prediction_path_ok}")
    print(f"E0-E7/common/train-val manifest protection hashes: passed={protected_ok}; protected_files={len(after)}")
    print(f"artifacts/experiments/e8 absent before/after smoke: passed={artifact_absent_before and artifact_absent_after}")


if __name__ == "__main__":
    main()
