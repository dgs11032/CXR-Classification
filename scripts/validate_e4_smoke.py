#!/usr/bin/env python3
"""One-batch E4 smoke test: no full training, artifacts, or test-data access."""
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
from p1_cxr.e4_models import build_e4_model  # noqa: E402
from p1_cxr.e4_training import CLASS_WEIGHTS, create_e4_criterion, create_e4_optimizer, load_e4_training_config  # noqa: E402
from p1_cxr.e3_training import evaluate_e3  # noqa: E402
from p1_cxr.training import flatten_two_view_batch, resolve_device, seed_everything  # noqa: E402

def snapshot() -> dict[str, str]:
    protected: set[Path] = set()
    for directory, pattern in ((PROJECT_ROOT / "configs", "*.yaml"), (PROJECT_ROOT / "src" / "data", "*.py"), (PROJECT_ROOT / "src" / "p1_cxr", "*.py"), (PROJECT_ROOT / "scripts", "*.py")):
        if directory.is_dir(): protected.update(path for path in directory.rglob(pattern) if "e4" not in path.name.lower())
    for name in ("train_manifest.csv", "val_manifest.csv"):
        path = PROJECT_ROOT / "artifacts" / "splits" / name
        if path.is_file(): protected.add(path)
    for name in ("e0", "e1", "e2", "e3"):
        directory = PROJECT_ROOT / "artifacts" / "experiments" / name
        if directory.is_dir(): protected.update(path for path in directory.rglob("*") if path.is_file())
    return {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(protected)}

def balanced_loader(dataset, per_class: int) -> DataLoader:
    indices: list[int] = []
    for label in (0, 1, 2):
        chosen = dataset.frame.index[dataset.frame["label"] == label].tolist()[:per_class]
        if len(chosen) != per_class: raise RuntimeError(f"Validation lacks {per_class} samples for label {label}.")
        indices.extend(chosen)
    return DataLoader(Subset(dataset, indices), batch_size=len(indices), shuffle=False)

def has_gradient(module: nn.Module) -> bool:
    return any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in module.parameters() if p.requires_grad)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e4_training.yaml")
parser.add_argument("--device", default=None)
args = parser.parse_args()
before = snapshot(); config = load_e4_training_config(args.config, project_root=PROJECT_ROOT)
data_config = load_data_config(config["resolved"]["data_config_path"], project_root=PROJECT_ROOT)
if "test" in data_config.get("manifests", {}): raise RuntimeError("E4 smoke test must not use test manifests.")
seed_everything(int(config["training"]["seed"])); device = resolve_device(args.device or str(config["smoke_test"]["device"]))
train_loader, validation_loader = build_dataloaders(data_config)
model = build_e4_model(config).to(device); criterion = create_e4_criterion(config, device=device); optimizer = create_e4_optimizer(model, config)
views, labels = next(iter(train_loader)); flattened, expanded = flatten_two_view_batch(views, labels); flattened, expanded = flattened.to(device), expanded.to(device)
if not bool((expanded == 1).any()): raise RuntimeError("Smoke batch lacks Bacterial labels.")
optimizer.zero_grad(set_to_none=True); logits = model(flattened); weighted = criterion(logits, expanded); unweighted = nn.CrossEntropyLoss()(logits, expanded)
weights_ok = criterion.weight is not None and torch.equal(criterion.weight.detach().cpu(), torch.tensor(CLASS_WEIGHTS))
finite, differs = bool(torch.isfinite(weighted)) and bool(torch.isfinite(unweighted)), not bool(torch.allclose(weighted, unweighted, rtol=1e-7, atol=1e-8))
if not weights_ok or not finite or not differs: raise RuntimeError(f"E4 weighted CE contract failed: weights={weights_ok}, finite={finite}, differs={differs}")
weighted.backward(); gradients = {"encoder": has_gradient(model.features) and has_gradient(model.feature_norm), "classifier": has_gradient(model.classifier)}
if not all(gradients.values()): raise RuntimeError(f"Missing E4 gradients: {gradients}")
head_before = model.classifier.weight.detach().clone(); optimizer.step(); step_ok = not torch.equal(head_before, model.classifier.weight.detach())
if not step_ok: raise RuntimeError("E4 optimizer step did not update classifier.")
sanity_logits = torch.tensor([[0.2, 1.1, -0.4]] * 3, dtype=torch.float32, device=device); sanity_labels = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
plain = nn.CrossEntropyLoss(reduction="none")(sanity_logits, sanity_labels); scaled = nn.CrossEntropyLoss(weight=criterion.weight, reduction="none")(sanity_logits, sanity_labels)
bacterial_085 = bool(torch.allclose(scaled[1], plain[1] * 0.85, rtol=1e-6, atol=1e-7)); normal_viral_one = bool(torch.allclose(scaled[[0,2]], plain[[0,2]], rtol=1e-6, atol=1e-7))
if not bacterial_085 or not normal_viral_one: raise RuntimeError("E4 per-class CE sanity failed.")
validation = evaluate_e3(model, balanced_loader(validation_loader.dataset, int(config["smoke_test"]["validation_samples_per_class"])), device=device)
metrics_ok = validation.metrics["macro_f1"] is not None and validation.metrics["macro_auroc_ovr"] is not None; prediction_unweighted = not validation.class_weights_used_for_prediction
if not metrics_ok or not prediction_unweighted: raise RuntimeError("E4 validation metric/prediction contract failed.")
after = snapshot(); protected_ok = before == after
if not protected_ok: raise RuntimeError(f"Protected E0/E1/E2/E3/common/manifest files changed: {sorted(path for path in set(before)|set(after) if before.get(path)!=after.get(path))}")
print(f"device: {device}")
print(f"train input shape: {tuple(views.shape)}")
print(f"flattened input shape: {tuple(flattened.shape)}")
print(f"source labels: {labels.tolist()}")
print(f"expanded label order: {expanded.detach().cpu().tolist()}")
print(f"logits shape: {tuple(logits.shape)}")
print(f"class weight tensor: {criterion.weight.detach().cpu().tolist()}; exact={weights_ok}")
print(f"weighted CE loss: {weighted.item():.6f}; finite={finite}")
print(f"unweighted CE loss (reference): {unweighted.item():.6f}")
print(f"weighted and unweighted CE differ: passed={differs}")
print(f"Bacterial individual CE weighted at 0.85: passed={bacterial_085}")
print(f"Normal/Viral individual CE weighted at 1.0: passed={normal_viral_one}")
print(f"gradient propagation: {gradients}")
print(f"optimizer step 1x: passed={step_ok}")
print(f"validation logits shape: {tuple(validation.logits.shape)}")
print(f"validation metric calculation: passed={metrics_ok}")
print(f"class weights excluded from prediction path: passed={prediction_unweighted}")
print(f"E0/E1/E2/E3/common/train-val manifest protection hashes: passed={protected_ok}; protected_files={len(after)}")
