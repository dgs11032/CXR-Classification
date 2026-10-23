"""E5-only fresh training: E4 contract with Bacterial CrossEntropyLoss weight 0.80."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import nn

from data.dataloader import build_dataloaders, load_data_config

from .checkpoints import save_checkpoint
from .e3_training import evaluate_e3, train_one_epoch_e3
from .e5_models import ConvNeXtTinyE5, build_e5_model
from .training import PROJECT_ROOT, create_optimizer, create_scheduler, file_sha256, resolve_device, seed_everything

CLASS_WEIGHTS = (1.0, 0.80, 1.0)
_E4_PARITY = {
    ("model", "architecture"): "flat", ("model", "backbone"): "convnext_tiny", ("model", "weights"): "IMAGENET1K_V1",
    ("model", "pretrained"): True, ("model", "allow_weight_download"): False, ("model", "feature_dimension"): 768,
    ("model", "num_classes"): 3, ("model", "classification_head"): "Linear(768, 3)", ("model", "trainable_scope"): "full",
    ("optimizer", "name"): "AdamW", ("optimizer", "strategy"): "differential_lr", ("optimizer", "backbone_learning_rate"): 1e-5,
    ("optimizer", "head_learning_rate"): 1e-4, ("optimizer", "weight_decay"): 1e-4,
    ("scheduler", "name"): "CosineAnnealingLR", ("scheduler", "t_max"): 15, ("scheduler", "eta_min"): 1e-6,
    ("training", "seed"): 42, ("training", "epochs"): 15, ("training", "early_stopping_patience"): 5,
    ("training", "selection_metric"): "macro_f1", ("training", "device"): "cuda", ("training", "precision"): "fp32",
    ("training", "amp"): False, ("training", "gradient_clip_norm"): 1.0, ("training", "detect_nonfinite"): True,
    ("training", "checkpoint", "best_filename"): "best.pt", ("training", "checkpoint", "last_filename"): "last.pt",
    ("training", "checkpoint", "save_rng_state"): True,
}


def _value(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        value = value[key]
    return value


def validate_e5_fairness_contract(config: Mapping[str, Any]) -> None:
    """Fail closed unless E5 differs from E4 only by Bacterial weight 0.80."""
    if config.get("experiment") != "e5":
        raise ValueError("E5 config must declare experiment=e5.")
    mismatches: list[dict[str, Any]] = []
    for path, expected in _E4_PARITY.items():
        try: actual = _value(config, path)
        except (KeyError, TypeError): actual = "__MISSING__"
        if actual != expected: mismatches.append({"path": ".".join(path), "expected": expected, "actual": actual})
    loss = config.get("loss", {})
    expected_loss = {"name": "CrossEntropyLoss", "label_smoothing": 0.0, "class_weighting": "modest_bacterial_0_80_e5", "class_weights": list(CLASS_WEIGHTS), "label_order": ["Normal", "Bacterial", "Viral"]}
    for key, expected in expected_loss.items():
        if loss.get(key) != expected: mismatches.append({"path": f"loss.{key}", "expected": expected, "actual": loss.get(key)})
    for key, expected in {"data_config": "configs/data_config.yaml", "output_dir": "artifacts/experiments/e5"}.items():
        if config.get(key) != expected: mismatches.append({"path": key, "expected": expected, "actual": config.get(key)})
    forbidden = {"resume_from", "initial_checkpoint", "checkpoint_path", "load_checkpoint"}
    if present := sorted(forbidden & set(config)): mismatches.append({"path": "top_level_checkpoint_loading", "expected": "absent", "actual": present})
    if mismatches: raise ValueError(f"E5 fairness contract drift: {mismatches}")


def load_e5_training_config(config_path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve(); path = Path(config_path).expanduser()
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file(): raise FileNotFoundError(f"E5 config unavailable inside project: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict): raise ValueError("E5 config must be a YAML mapping.")
    validate_e5_fairness_contract(config)
    output, data = (root / config["output_dir"]).resolve(), (root / config["data_config"]).resolve()
    if not output.is_relative_to(root) or not data.is_relative_to(root): raise ValueError("E5 paths must remain inside project root.")
    config["resolved"] = {"project_root": str(root), "config_path": str(path), "data_config_path": str(data), "output_dir": str(output)}
    return config


def create_e5_criterion(config: Mapping[str, Any], *, device: torch.device) -> nn.CrossEntropyLoss:
    validate_e5_fairness_contract(config)
    weight = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    expected = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    if not torch.allclose(weight, expected, rtol=0.0, atol=0.0): raise RuntimeError(f"Unexpected E5 class weights: {weight.tolist()}")
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=0.0, reduction="mean")


def create_e5_optimizer(model: ConvNeXtTinyE5, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    return create_optimizer(model, config)


def create_e5_scheduler(optimizer: torch.optim.Optimizer, config: Mapping[str, Any]) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return create_scheduler(optimizer, config)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fit_e5(training_config: Mapping[str, Any]) -> dict[str, Any]:
    """Fresh E5 only; no E0–E4 checkpoint load/resume API exists."""
    validate_e5_fairness_contract(training_config); seed = int(training_config["training"]["seed"]); seed_everything(seed)
    root = Path(training_config["resolved"]["project_root"])
    data_config = load_data_config(training_config["resolved"]["data_config_path"], project_root=root)
    if data_config["views"]["train"] != "two" or data_config["views"]["validation"] != "single" or int(data_config["loader"]["batch_size"]) != 16 or "test" in data_config.get("manifests", {}): raise ValueError("E5 requires E4's train/validation-only two-view/single-view batch-size-16 contract.")
    train_loader, validation_loader = build_dataloaders(data_config)
    if len(validation_loader.dataset) != 742: raise ValueError("E5 must validate on E4's complete 742-image split.")
    device = resolve_device(str(training_config["training"]["device"]))
    if device.type != "cuda": raise RuntimeError("Full E5 training requires CUDA, matching E4.")
    model = build_e5_model(training_config).to(device); criterion = create_e5_criterion(training_config, device=device)
    optimizer = create_e5_optimizer(model, training_config); scheduler = create_e5_scheduler(optimizer, training_config)
    output_dir = Path(training_config["resolved"]["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True); _write_json(output_dir / "training_config.json", training_config)
    fingerprints = {"train": file_sha256(data_config["resolved"]["train_manifest"]), "validation": file_sha256(data_config["resolved"]["validation_manifest"])}
    history: list[dict[str, Any]] = []; best_metric, best_epoch, stale = float("-inf"), 0, 0
    for epoch in range(1, int(training_config["training"]["epochs"]) + 1):
        train = train_one_epoch_e3(model, train_loader, criterion=criterion, optimizer=optimizer, device=device, gradient_clip_norm=float(training_config["training"]["gradient_clip_norm"]))
        validation = evaluate_e3(model, validation_loader, device=device); metric = float(validation.metrics["macro_f1"]); improved = metric > best_metric
        if improved: best_metric, best_epoch, stale = metric, epoch, 0
        else: stale += 1
        row = {"epoch": epoch, "train_weighted_ce": train.loss, "validation_unweighted_ce": validation.loss, "validation_accuracy": validation.metrics["accuracy"], "validation_macro_f1": metric, "validation_macro_auroc": validation.metrics["macro_auroc_ovr"], "is_best": improved, "stale_epochs": stale}
        history.append(row); _write_json(output_dir / "history.json", history); scheduler.step()
        payload = {"model": model, "epoch": epoch, "metrics": row, "config": training_config, "optimizer": optimizer, "scheduler": scheduler, "best_validation_macro_f1": best_metric, "random_seed": seed, "control_state": {"best_epoch": best_epoch, "stale_epochs": stale, "manifest_fingerprints": fingerprints, "initialization": "fresh_imagenet1k_v1_no_e0_e1_e2_e3_e4_checkpoint", "class_weights": list(CLASS_WEIGHTS)}, "history": history, "train_loader_generator": train_loader.generator, "save_rng_state": bool(training_config["training"]["checkpoint"]["save_rng_state"]), "run_kind": "e5_full_epoch"}
        save_checkpoint(output_dir / training_config["training"]["checkpoint"]["last_filename"], **payload)
        if improved: save_checkpoint(output_dir / training_config["training"]["checkpoint"]["best_filename"], **payload)
        if stale >= int(training_config["training"]["early_stopping_patience"]): break
    return {"best_validation_macro_f1": best_metric, "best_epoch": best_epoch, "epochs_completed": len(history), "manifest_fingerprints": fingerprints, "output_dir": str(output_dir)}
