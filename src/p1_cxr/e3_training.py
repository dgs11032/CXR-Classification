"""E3-only fresh training engine: E0 architecture/data flow plus modest weighted CE."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data.dataloader import build_dataloaders, load_data_config

from .checkpoints import save_checkpoint
from .e3_models import ConvNeXtTinyE3, build_e3_model
from .metrics import classification_metrics, prediction_frame
from .training import (
    PROJECT_ROOT,
    create_optimizer,
    create_scheduler,
    file_sha256,
    flatten_two_view_batch,
    resolve_device,
    seed_everything,
)

CLASS_WEIGHTS = (1.0, 0.9, 1.0)

_E0_PARITY = {
    ("model", "architecture"): "flat",
    ("model", "backbone"): "convnext_tiny",
    ("model", "weights"): "IMAGENET1K_V1",
    ("model", "pretrained"): True,
    ("model", "allow_weight_download"): False,
    ("model", "feature_dimension"): 768,
    ("model", "num_classes"): 3,
    ("model", "classification_head"): "Linear(768, 3)",
    ("model", "trainable_scope"): "full",
    ("optimizer", "name"): "AdamW",
    ("optimizer", "strategy"): "differential_lr",
    ("optimizer", "backbone_learning_rate"): 1e-5,
    ("optimizer", "head_learning_rate"): 1e-4,
    ("optimizer", "weight_decay"): 1e-4,
    ("scheduler", "name"): "CosineAnnealingLR",
    ("scheduler", "t_max"): 15,
    ("scheduler", "eta_min"): 1e-6,
    ("training", "seed"): 42,
    ("training", "epochs"): 15,
    ("training", "early_stopping_patience"): 5,
    ("training", "selection_metric"): "macro_f1",
    ("training", "device"): "cuda",
    ("training", "precision"): "fp32",
    ("training", "amp"): False,
    ("training", "gradient_clip_norm"): 1.0,
    ("training", "detect_nonfinite"): True,
    ("training", "checkpoint", "best_filename"): "best.pt",
    ("training", "checkpoint", "last_filename"): "last.pt",
    ("training", "checkpoint", "save_rng_state"): True,
}


@dataclass
class E3TrainEpochResult:
    loss: float
    processed_source_images: int
    processed_views: int
    batches: int
    gradients_finite: bool


@dataclass
class E3EvaluationResult:
    loss: float
    metrics: dict[str, Any]
    logits: np.ndarray
    targets: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    processed_images: int
    batches: int
    class_weights_used_for_prediction: bool = False


def _value(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        value = value[key]
    return value


def validate_e3_fairness_contract(config: Mapping[str, Any]) -> None:
    """Fail closed unless E3 differs from E0 only by [1.0, 0.9, 1.0] CE weights."""
    if config.get("experiment") != "e3":
        raise ValueError("E3 config must declare experiment=e3.")
    mismatches: list[dict[str, Any]] = []
    for path, expected in _E0_PARITY.items():
        try:
            actual = _value(config, path)
        except (KeyError, TypeError):
            actual = "__MISSING__"
        if actual != expected:
            mismatches.append({"path": ".".join(path), "expected": expected, "actual": actual})
    loss = config.get("loss", {})
    expected_loss = {
        "name": "CrossEntropyLoss",
        "label_smoothing": 0.0,
        "class_weighting": "modest_bacterial_0_90_e3",
        "class_weights": list(CLASS_WEIGHTS),
        "label_order": ["Normal", "Bacterial", "Viral"],
    }
    for key, expected in expected_loss.items():
        if loss.get(key) != expected:
            mismatches.append({"path": f"loss.{key}", "expected": expected, "actual": loss.get(key)})
    if config.get("data_config") != "configs/data_config.yaml":
        mismatches.append({"path": "data_config", "expected": "configs/data_config.yaml", "actual": config.get("data_config")})
    if config.get("output_dir") != "artifacts/experiments/e3":
        mismatches.append({"path": "output_dir", "expected": "artifacts/experiments/e3", "actual": config.get("output_dir")})
    forbidden = {"resume_from", "initial_checkpoint", "checkpoint_path", "load_checkpoint"}
    present = sorted(key for key in forbidden if key in config)
    if present:
        mismatches.append({"path": "top_level_checkpoint_loading", "expected": "absent", "actual": present})
    if mismatches:
        raise ValueError(f"E3 fairness contract drift: {mismatches}")


def load_e3_training_config(config_path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    path = Path(config_path).expanduser()
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"E3 config is not available inside the project: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("E3 config must be a YAML mapping.")
    validate_e3_fairness_contract(config)
    output_dir = (root / config["output_dir"]).resolve()
    data_config_path = (root / config["data_config"]).resolve()
    if not output_dir.is_relative_to(root) or not data_config_path.is_relative_to(root):
        raise ValueError("E3 paths must remain inside the project root.")
    config["resolved"] = {
        "project_root": str(root),
        "config_path": str(path),
        "data_config_path": str(data_config_path),
        "output_dir": str(output_dir),
    }
    return config


def create_e3_criterion(config: Mapping[str, Any], *, device: torch.device) -> nn.CrossEntropyLoss:
    """Create E3's sole intentional change: modest class-weighted 3-class CE."""
    validate_e3_fairness_contract(config)
    weight = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    expected = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    if not torch.allclose(weight, expected, rtol=0.0, atol=0.0):
        raise RuntimeError(f"Unexpected E3 class weights: {weight.tolist()}")
    return nn.CrossEntropyLoss(
        weight=weight,
        label_smoothing=float(config["loss"]["label_smoothing"]),
        reduction="mean",
    )


def create_e3_optimizer(model: ConvNeXtTinyE3, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    """Reuse E0's verified differential-LR optimizer for the unchanged module layout."""
    return create_optimizer(model, config)


def create_e3_scheduler(optimizer: torch.optim.Optimizer, config: Mapping[str, Any]) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return create_scheduler(optimizer, config)


def _finite_gradients(model: nn.Module) -> bool:
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    return bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


def train_one_epoch_e3(model: ConvNeXtTinyE3, loader: DataLoader, *, criterion: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer, device: torch.device, gradient_clip_norm: float | None, max_batches: int | None = None) -> E3TrainEpochResult:
    model.train()
    loss_sum = 0.0
    source_images = 0
    views_seen = 0
    batches = 0
    for batch_index, (views, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        flattened, expanded_labels = flatten_two_view_batch(views, labels)
        flattened, expanded_labels = flattened.to(device), expanded_labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(flattened)
        loss = criterion(logits, expanded_labels)
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite E3 train tensor at batch {batch_index}.")
        loss.backward()
        if not _finite_gradients(model):
            raise FloatingPointError(f"Non-finite or absent E3 gradient at train batch {batch_index}.")
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        count = len(expanded_labels)
        loss_sum += float(loss.detach().cpu()) * count
        source_images += len(labels)
        views_seen += count
        batches += 1
    if batches == 0:
        raise ValueError("No E3 train batch was processed.")
    return E3TrainEpochResult(loss_sum / views_seen, source_images, views_seen, batches, True)


@torch.inference_mode()
def evaluate_e3(model: ConvNeXtTinyE3, loader: DataLoader, *, device: torch.device, max_batches: int | None = None) -> E3EvaluationResult:
    """Score deterministic validation views from unweighted softmax/argmax predictions."""
    model.eval()
    logits_parts: list[Tensor] = []
    label_parts: list[Tensor] = []
    loss_sum = 0.0
    samples = 0
    batches = 0
    unweighted_criterion = nn.CrossEntropyLoss()
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if images.ndim != 4:
            raise ValueError(f"E3 validation expects [B,C,H,W], got {tuple(images.shape)}")
        logits = model(images.to(device))
        loss = unweighted_criterion(logits, labels.to(device))
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite E3 validation tensor at batch {batch_index}.")
        logits_parts.append(logits.float().cpu())
        label_parts.append(labels.cpu())
        loss_sum += float(loss.cpu()) * len(labels)
        samples += len(labels)
        batches += 1
    if batches == 0:
        raise ValueError("No E3 validation batch was processed.")
    logits_tensor = torch.cat(logits_parts)
    targets = torch.cat(label_parts).numpy()
    probabilities = torch.softmax(logits_tensor, dim=1).numpy()
    return E3EvaluationResult(
        loss=loss_sum / samples,
        metrics=classification_metrics(targets, probabilities),
        logits=logits_tensor.numpy(),
        targets=targets,
        probabilities=probabilities,
        predictions=probabilities.argmax(axis=1),
        processed_images=samples,
        batches=batches,
        class_weights_used_for_prediction=False,
    )


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_validation_artifacts(output_dir: Path, prefix: str, epoch: int, train: E3TrainEpochResult, result: E3EvaluationResult, image_paths: list[str]) -> None:
    _write_json(output_dir / f"{prefix}_validation_metrics.json", {
        "epoch": epoch, "train_weighted_ce": train.loss, "validation_unweighted_ce": result.loss,
        "prediction_source": "softmax_main_3_class_logits_argmax_no_class_weights", **result.metrics,
    })
    prediction_frame(image_paths, result.targets, result.probabilities).to_csv(output_dir / f"{prefix}_validation_predictions.csv", index=False)
    pd.DataFrame(result.metrics["confusion_matrix"], index=["true_normal", "true_bacterial", "true_viral"], columns=["pred_normal", "pred_bacterial", "pred_viral"]).reset_index(names="ground_truth").to_csv(output_dir / f"{prefix}_validation_confusion_matrix.csv", index=False)


def fit_e3(training_config: Mapping[str, Any]) -> dict[str, Any]:
    """Run a fresh E3 experiment; checkpoint resume/loading is intentionally unavailable."""
    validate_e3_fairness_contract(training_config)
    seed = int(training_config["training"]["seed"])
    seed_everything(seed)
    root = Path(training_config["resolved"]["project_root"])
    data_config = load_data_config(training_config["resolved"]["data_config_path"], project_root=root)
    if data_config["views"]["train"] != "two" or data_config["views"]["validation"] != "single" or int(data_config["loader"]["batch_size"]) != 16 or "test" in data_config.get("manifests", {}):
        raise ValueError("E3 requires E0's train/validation-only two-view/single-view batch-size-16 data contract.")
    train_loader, validation_loader = build_dataloaders(data_config)
    if len(validation_loader.dataset) != 742:
        raise ValueError("E3 must validate on E0's full 742-image validation split.")
    device = resolve_device(str(training_config["training"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("Full E3 training requires CUDA, matching E0.")
    model = build_e3_model(training_config).to(device)
    criterion = create_e3_criterion(training_config, device=device)
    optimizer = create_e3_optimizer(model, training_config)
    scheduler = create_e3_scheduler(optimizer, training_config)
    output_dir = Path(training_config["resolved"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "training_config.json", training_config)
    fingerprints = {"train": file_sha256(data_config["resolved"]["train_manifest"]), "validation": file_sha256(data_config["resolved"]["validation_manifest"])}
    history: list[dict[str, Any]] = []
    best_metric, best_epoch, stale_epochs = float("-inf"), 0, 0
    validation_paths = validation_loader.dataset.frame["image_path"].tolist()
    for epoch in range(1, int(training_config["training"]["epochs"]) + 1):
        train = train_one_epoch_e3(model, train_loader, criterion=criterion, optimizer=optimizer, device=device, gradient_clip_norm=float(training_config["training"]["gradient_clip_norm"]))
        validation = evaluate_e3(model, validation_loader, device=device)
        metric = float(validation.metrics["macro_f1"])
        improved = metric > best_metric
        if improved:
            best_metric, best_epoch, stale_epochs = metric, epoch, 0
        else:
            stale_epochs += 1
        row = {"epoch": epoch, "train_weighted_ce": train.loss, "validation_unweighted_ce": validation.loss, "validation_accuracy": validation.metrics["accuracy"], "validation_macro_f1": metric, "validation_macro_auroc": validation.metrics["macro_auroc_ovr"], "is_best": improved, "stale_epochs": stale_epochs}
        history.append(row)
        _write_json(output_dir / "history.json", history)
        _write_validation_artifacts(output_dir, "last", epoch, train, validation, validation_paths)
        if improved:
            _write_validation_artifacts(output_dir, "best", epoch, train, validation, validation_paths)
        scheduler.step()
        checkpoint_kwargs = {"model": model, "epoch": epoch, "metrics": row, "config": training_config, "optimizer": optimizer, "scheduler": scheduler, "best_validation_macro_f1": best_metric, "random_seed": seed, "control_state": {"best_epoch": best_epoch, "stale_epochs": stale_epochs, "manifest_fingerprints": fingerprints, "initialization": "fresh_imagenet1k_v1_no_e0_e1_e2_checkpoint", "class_weights": list(CLASS_WEIGHTS)}, "history": history, "train_loader_generator": train_loader.generator, "save_rng_state": bool(training_config["training"]["checkpoint"]["save_rng_state"]), "run_kind": "e3_full_epoch"}
        save_checkpoint(output_dir / training_config["training"]["checkpoint"]["last_filename"], **checkpoint_kwargs)
        if improved:
            save_checkpoint(output_dir / training_config["training"]["checkpoint"]["best_filename"], **checkpoint_kwargs)
        if stale_epochs >= int(training_config["training"]["early_stopping_patience"]):
            break
    return {"best_validation_macro_f1": best_metric, "best_epoch": best_epoch, "epochs_completed": len(history), "manifest_fingerprints": fingerprints, "output_dir": str(output_dir)}
