#!/usr/bin/env python3
"""E9 smoke validation; full mode requires real E6 best.pt, static mode is workspace-safe."""
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
from p1_cxr.e9_models import build_e9_model, transfer_e6_encoder  # noqa: E402
from p1_cxr.e9_training import create_e9_criterion, e8_e9_control_comparison, load_e9_training_config  # noqa: E402
from p1_cxr.e8_training import evaluate_e8  # noqa: E402
from p1_cxr.training import flatten_two_view_batch, resolve_device, seed_everything, create_optimizer  # noqa: E402

E9_ARTIFACT = PROJECT_ROOT / "artifacts" / "experiments" / "e9"


def snapshot() -> dict[str, str]:
    protected: set[Path] = set()
    for directory, pattern in ((PROJECT_ROOT / "configs", "*.yaml"), (PROJECT_ROOT / "src" / "data", "*.py"), (PROJECT_ROOT / "src" / "p1_cxr", "*.py"), (PROJECT_ROOT / "scripts", "*.py")):
        if directory.is_dir():
            protected.update(path for path in directory.rglob(pattern) if "e9" not in path.name.lower())
    for name in ("train_manifest.csv", "val_manifest.csv"):
        path = PROJECT_ROOT / "artifacts" / "splits" / name
        if path.is_file(): protected.add(path)
    for name in ("e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"):
        directory = PROJECT_ROOT / "artifacts" / "experiments" / name
        if directory.is_dir(): protected.update(path for path in directory.rglob("*") if path.is_file())
    return {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(protected)}


def balanced_loader(dataset: object, per_class: int) -> DataLoader:
    frame = dataset.frame
    indices: list[int] = []
    for label in (0, 1):
        selected = frame.index[frame["label"] == label].tolist()[:per_class]
        if len(selected) != per_class: raise RuntimeError(f"E9 validation lacks {per_class} samples for label {label}.")
        indices.extend(selected)
    return DataLoader(Subset(dataset, indices), batch_size=len(indices), shuffle=False)


def has_gradient(module: nn.Module) -> bool:
    return any(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in module.parameters() if parameter.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e9_training.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--static-only", action="store_true", help="Validate config/model contracts without requiring absent source checkpoint.")
    args = parser.parse_args()
    if E9_ARTIFACT.exists(): raise RuntimeError("E9 artifact directory already exists; smoke test must not write or reuse it.")
    before = snapshot()
    config = load_e9_training_config(args.config, project_root=PROJECT_ROOT)
    source = Path(config["resolved"]["source_checkpoint_path"])
    model = build_e9_model(config)
    static_ok = tuple(model.classifier.weight.shape) == (2, 768) and tuple(model.classifier.bias.shape) == (2,)
    if not static_ok: raise RuntimeError("E9 fresh binary classifier contract failed.")
    if args.static_only:
        after = snapshot()
        if before != after or E9_ARTIFACT.exists(): raise RuntimeError("Static E9 smoke modified protected files or created an artifact.")
        print(f"static config/model validation: passed; binary classifier shapes={(2, 768)}/{(2,)}")
        print(f"E8/E9 parity: {config['resolved']['e8_parity']}")
        print(f"E6 source checkpoint present: {source.is_file()} (full transfer smoke intentionally skipped)")
        print(f"E0-E8/common/train-val protection hashes: passed; protected_files={len(after)}")
        return
    if not source.is_file(): raise FileNotFoundError(f"E9 full smoke requires E6 best checkpoint: {source}")
    data_config = load_data_config(config["resolved"]["data_config_path"], project_root=PROJECT_ROOT)
    if "test" in data_config.get("manifests", {}): raise RuntimeError("E9 smoke must not use test manifests.")
    seed_everything(int(config["training"]["seed"]))
    device = resolve_device(args.device or str(config["smoke_test"]["device"]))
    train_loader, validation_loader = build_e8_dataloaders(data_config)
    train_counts, validation_counts = train_loader.dataset.class_counts(), validation_loader.dataset.class_counts()
    subset_ok = len(train_loader.dataset) == 3300 and len(validation_loader.dataset) == 550 and train_counts == EXPECTED_COUNTS["train"] and validation_counts == EXPECTED_COUNTS["validation"] and set(train_loader.dataset.frame["label"].unique()) == {0, 1}
    if not subset_ok: raise RuntimeError("E9 fixed manifest/subset/remapping contract failed.")
    report = transfer_e6_encoder(model, checkpoint_path=source, project_root=PROJECT_ROOT)
    if not (report["all_encoder_tensors_equal"] and report["fresh_binary_classifier_preserved"] and report["transferred_encoder_tensor_count"] == 180 and report["source_classifier_shapes"]["classifier.weight"] == [3, 768] and report["target_classifier_shapes"]["classifier.weight"] == [2, 768]):
        raise RuntimeError(f"E9 E6 transfer report failed: {report}")
    model = model.to(device)
    criterion = create_e9_criterion(config)
    optimizer = create_optimizer(model, config)
    views, labels = next(iter(train_loader))
    flattened, expanded = flatten_two_view_batch(views, labels)
    flattened, expanded = flattened.to(device), expanded.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(flattened)
    components = criterion.components(logits, expanded)
    focal_loss = components.per_sample_focal.mean()
    ce_loss = nn.CrossEntropyLoss()(logits, expanded)
    focal_ok = criterion.gamma == 2.0 and criterion.alpha is None and bool(torch.isfinite(focal_loss)) and bool((components.pt >= 0).all() and (components.pt <= 1).all()) and not bool(torch.allclose(focal_loss, ce_loss))
    if logits.shape[1] != 2 or not focal_ok: raise RuntimeError("E9 binary focal/logit contract failed.")
    focal_loss.backward()
    gradients = {"encoder": has_gradient(model.features) and has_gradient(model.feature_norm), "classifier": has_gradient(model.classifier)}
    if not all(gradients.values()): raise RuntimeError(f"E9 gradients failed: {gradients}")
    head_before = model.classifier.weight.detach().clone(); optimizer.step(); step_ok = not torch.equal(head_before, model.classifier.weight.detach())
    if not step_ok: raise RuntimeError("E9 optimizer step did not update binary classifier.")
    validation = evaluate_e8(model, balanced_loader(validation_loader.dataset, int(config["smoke_test"]["validation_samples_per_class"])), device=device)
    metrics_ok = validation.metrics["macro_f1"] is not None and validation.metrics["auroc"] is not None and not validation.focal_used_for_prediction
    if not metrics_ok: raise RuntimeError("E9 binary validation/prediction contract failed.")
    after = snapshot()
    if before != after or E9_ARTIFACT.exists(): raise RuntimeError("E9 smoke modified protected files or created an artifact.")
    print(f"device: {device}; E8/E9 parity: {config['resolved']['e8_parity']}")
    print(f"subset counts: train={train_counts}; validation={validation_counts}; passed={subset_ok}")
    print(f"train/flattened/logits shapes: {tuple(views.shape)} -> {tuple(flattened.shape)} -> {tuple(logits.shape)}")
    print(f"binary order={list(BINARY_CLASS_ORDER)}; expanded labels={expanded.cpu().tolist()}")
    print(f"E6 transfer: {report}")
    print(f"focal gamma=2/finite/bounded/different-CE: passed={focal_ok}; gradients={gradients}; optimizer_step={step_ok}")
    print(f"validation logits={tuple(validation.logits.shape)}; binary metrics/prediction path passed={metrics_ok}")
    print(f"E0-E8/common/train-val protection hashes: passed; protected_files={len(after)}")
    print("artifacts/experiments/e9 absent before/after smoke: passed=True")


if __name__ == "__main__": main()
