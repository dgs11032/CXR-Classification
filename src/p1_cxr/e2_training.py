"""E2-only CE + auxiliary pneumonia-subtype CE training and validation utilities."""

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
from .e2_models import ConvNeXtTinyE2, build_e2_model
from .metrics import classification_metrics, prediction_frame
from .training import (
    PROJECT_ROOT,
    file_sha256,
    flatten_two_view_batch,
    resolve_device,
    seed_everything,
)

_E0_PARITY = {
    ("model", "backbone"): "convnext_tiny",
    ("model", "weights"): "IMAGENET1K_V1",
    ("model", "pretrained"): True,
    ("model", "allow_weight_download"): False,
    ("model", "feature_dimension"): 768,
    ("model", "num_classes"): 3,
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
class E2LossResult:
    main_logits: Tensor
    subtype_logits: Tensor
    features: Tensor
    main_loss: Tensor
    subtype_loss: Tensor
    total_loss: Tensor
    pneumonia_view_count: int
    excluded_normal_view_count: int


@dataclass
class E2TrainEpochResult:
    main_loss: float
    subtype_loss: float
    total_loss: float
    pneumonia_source_samples: int
    excluded_normal_source_samples: int
    pneumonia_views: int
    excluded_normal_views: int
    batches: int
    gradients_finite: bool


@dataclass
class E2EvaluationResult:
    loss: float
    metrics: dict[str, Any]
    logits: np.ndarray
    targets: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    processed_images: int
    batches: int
    subtype_logits_used_for_prediction: bool = False


def _value(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        current = current[key]
    return current


def validate_e2_fairness_contract(config: Mapping[str, Any]) -> None:
    """Fail closed unless every non-E2 hyperparameter equals finalized E0."""
    if config.get("experiment") != "e2":
        raise ValueError("E2 config must declare experiment=e2.")
    mismatches = []
    for path, expected in _E0_PARITY.items():
        try:
            actual = _value(config, path)
        except (KeyError, TypeError):
            actual = "__MISSING__"
        if actual != expected:
            mismatches.append({"path": ".".join(path), "expected": expected, "actual": actual})
    model = config.get("model", {})
    subtype = config.get("loss", {}).get("subtype", {})
    classification = config.get("loss", {}).get("classification", {})
    expected_e2 = {
        "model.architecture": (model.get("architecture"), "flat_with_auxiliary_subtype_head"),
        "model.classification_head": (model.get("classification_head"), "Linear(768, 3)"),
        "model.subtype_head": (model.get("subtype_head"), "Linear(768, 2)"),
        "classification.name": (classification.get("name"), "CrossEntropyLoss"),
        "classification.label_smoothing": (classification.get("label_smoothing"), 0.0),
        "classification.class_weighting": (classification.get("class_weighting"), "none_unweighted_e0_baseline"),
        "subtype.name": (subtype.get("name"), "CrossEntropyLoss"),
        "subtype.included_labels": (subtype.get("included_labels"), [1, 2]),
        "subtype.excluded_label": (subtype.get("excluded_label"), 0),
        "subtype.lambda_subtype": (subtype.get("lambda_subtype"), 0.3),
        "subtype.finite_zero_when_no_pneumonia": (subtype.get("finite_zero_when_no_pneumonia"), True),
        "output_dir": (config.get("output_dir"), "artifacts/experiments/e2"),
        "data_config": (config.get("data_config"), "configs/data_config.yaml"),
    }
    for name, (actual, expected) in expected_e2.items():
        if actual != expected:
            mismatches.append({"path": name, "expected": expected, "actual": actual})
    remapping = subtype.get("temporary_remapping", {})
    if remapping != {"bacterial_label_1": 0, "viral_label_2": 1}:
        mismatches.append({"path": "subtype.temporary_remapping", "expected": {"bacterial_label_1": 0, "viral_label_2": 1}, "actual": remapping})
    forbidden = {"resume_from", "initial_checkpoint", "checkpoint_path", "load_checkpoint"}
    found_forbidden = sorted(key for key in config if key in forbidden)
    if found_forbidden:
        mismatches.append({"path": "top_level_checkpoint_loading", "expected": "absent", "actual": found_forbidden})
    if mismatches:
        raise ValueError(f"E2 fairness contract drift detected: {mismatches}")


def load_e2_training_config(
    config_path: str | Path, *, project_root: str | Path = PROJECT_ROOT
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    path = Path(config_path).expanduser()
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"E2 config is not available inside the project: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("E2 config must be a YAML mapping.")
    validate_e2_fairness_contract(config)
    output_dir = (root / config["output_dir"]).resolve()
    data_config_path = (root / config["data_config"]).resolve()
    if not output_dir.is_relative_to(root) or not data_config_path.is_relative_to(root):
        raise ValueError("E2 paths must remain inside the project root.")
    config["resolved"] = {
        "project_root": str(root),
        "config_path": str(path),
        "data_config_path": str(data_config_path),
        "output_dir": str(output_dir),
    }
    return config


def create_e2_optimizer(model: ConvNeXtTinyE2, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    optimizer_config = config["optimizer"]
    backbone_modules = (model.features, model.avgpool, model.feature_norm)
    head_modules = (model.classifier, model.subtype_classifier)
    backbone = [parameter for module in backbone_modules for parameter in module.parameters() if parameter.requires_grad]
    head = [parameter for module in head_modules for parameter in module.parameters() if parameter.requires_grad]
    all_trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    grouped = {id(parameter) for parameter in backbone + head}
    if not backbone or not head or {id(parameter) for parameter in backbone} & {id(parameter) for parameter in head} or grouped != all_trainable:
        raise RuntimeError("E2 optimizer parameter groups are incomplete or overlapping.")
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": float(optimizer_config["backbone_learning_rate"]), "name": "backbone"},
            {"params": head, "lr": float(optimizer_config["head_learning_rate"]), "name": "head"},
        ],
        weight_decay=float(optimizer_config["weight_decay"]),
    )


def create_e2_scheduler(optimizer: torch.optim.Optimizer, config: Mapping[str, Any]) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    scheduler_config = config["scheduler"]
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(scheduler_config["t_max"]), eta_min=float(scheduler_config["eta_min"])
    )


def subtype_targets_and_mask(labels: Tensor) -> tuple[Tensor, Tensor]:
    """Return pneumonia mask and temporary 1->0 / 2->1 targets without mutating labels."""
    if labels.ndim != 1:
        raise ValueError(f"Expected labels [N], got {tuple(labels.shape)}")
    invalid = torch.unique(labels[(labels < 0) | (labels > 2)])
    if len(invalid):
        raise ValueError(f"Invalid main labels: {invalid.detach().cpu().tolist()}")
    pneumonia_mask = labels != 0
    return pneumonia_mask, labels[pneumonia_mask] - 1


def compute_e2_losses(
    model: ConvNeXtTinyE2,
    flattened_images: Tensor,
    expanded_labels: Tensor,
    *,
    main_criterion: nn.CrossEntropyLoss,
    subtype_criterion: nn.CrossEntropyLoss,
    lambda_subtype: float,
) -> E2LossResult:
    """Compute E0-equivalent main CE and E2's masked subtype CE on shared features."""
    main_logits, subtype_logits, features = model.forward_with_subtype(flattened_images)
    main_loss = main_criterion(main_logits, expanded_labels)
    pneumonia_mask, subtype_targets = subtype_targets_and_mask(expanded_labels)
    if bool(pneumonia_mask.any()):
        subtype_loss = subtype_criterion(subtype_logits[pneumonia_mask], subtype_targets)
    else:
        # Finite zero tied to subtype logits preserves the backward graph for all-Normal batches.
        subtype_loss = subtype_logits.sum() * 0.0
    total_loss = main_loss + float(lambda_subtype) * subtype_loss
    for name, tensor in (
        ("main_logits", main_logits), ("subtype_logits", subtype_logits), ("features", features),
        ("main_loss", main_loss), ("subtype_loss", subtype_loss), ("total_loss", total_loss),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"Non-finite E2 {name}.")
    return E2LossResult(
        main_logits=main_logits, subtype_logits=subtype_logits, features=features,
        main_loss=main_loss, subtype_loss=subtype_loss, total_loss=total_loss,
        pneumonia_view_count=int(pneumonia_mask.sum().item()),
        excluded_normal_view_count=int((~pneumonia_mask).sum().item()),
    )


def _finite_gradients(model: nn.Module) -> bool:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
    return bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


def train_one_epoch_e2(
    model: ConvNeXtTinyE2, loader: DataLoader, *, optimizer: torch.optim.Optimizer,
    main_criterion: nn.CrossEntropyLoss, subtype_criterion: nn.CrossEntropyLoss,
    lambda_subtype: float, device: torch.device, gradient_clip_norm: float | None,
    max_batches: int | None = None,
) -> E2TrainEpochResult:
    """Train one E2 epoch using exactly E0's flattened two-view main-CE inputs."""
    model.train()
    totals = {"main": 0.0, "subtype": 0.0, "total": 0.0, "pneumonia_sources": 0, "normal_sources": 0, "pneumonia_views": 0, "normal_views": 0}
    processed_views = 0
    batches = 0
    for batch_index, (views, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        flattened, expanded_labels = flatten_two_view_batch(views, labels)
        flattened, expanded_labels = flattened.to(device), expanded_labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        result = compute_e2_losses(
            model, flattened, expanded_labels, main_criterion=main_criterion,
            subtype_criterion=subtype_criterion, lambda_subtype=lambda_subtype,
        )
        result.total_loss.backward()
        if not _finite_gradients(model):
            raise FloatingPointError(f"Non-finite or absent E2 gradient at train batch {batch_index}.")
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        view_count = len(expanded_labels)
        totals["main"] += float(result.main_loss.detach().cpu()) * view_count
        totals["subtype"] += float(result.subtype_loss.detach().cpu()) * view_count
        totals["total"] += float(result.total_loss.detach().cpu()) * view_count
        totals["pneumonia_sources"] += int((labels != 0).sum().item())
        totals["normal_sources"] += int((labels == 0).sum().item())
        totals["pneumonia_views"] += result.pneumonia_view_count
        totals["normal_views"] += result.excluded_normal_view_count
        processed_views += view_count
        batches += 1
    if batches == 0:
        raise ValueError("No train batch was processed for E2.")
    return E2TrainEpochResult(
        main_loss=totals["main"] / processed_views, subtype_loss=totals["subtype"] / processed_views,
        total_loss=totals["total"] / processed_views,
        pneumonia_source_samples=totals["pneumonia_sources"], excluded_normal_source_samples=totals["normal_sources"],
        pneumonia_views=totals["pneumonia_views"], excluded_normal_views=totals["normal_views"],
        batches=batches, gradients_finite=True,
    )


@torch.inference_mode()
def evaluate_e2(
    model: ConvNeXtTinyE2, loader: DataLoader, *, main_criterion: nn.CrossEntropyLoss,
    device: torch.device, max_batches: int | None = None,
) -> E2EvaluationResult:
    """Evaluate deterministic single views from main logits only; subtype head is unused."""
    model.eval()
    logits_parts: list[Tensor] = []
    label_parts: list[Tensor] = []
    loss_sum = 0.0
    samples = 0
    batches = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if images.ndim != 4:
            raise ValueError(f"Validation expects [B,C,H,W], got {tuple(images.shape)}")
        main_logits = model(images.to(device))
        loss = main_criterion(main_logits, labels.to(device))
        if not bool(torch.isfinite(main_logits).all()) or not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite E2 validation tensor at batch {batch_index}.")
        logits_parts.append(main_logits.float().cpu())
        label_parts.append(labels.cpu())
        loss_sum += float(loss.cpu()) * len(labels)
        samples += len(labels)
        batches += 1
    if batches == 0:
        raise ValueError("No validation batch was processed for E2.")
    logits_tensor = torch.cat(logits_parts)
    targets_tensor = torch.cat(label_parts)
    probabilities = torch.softmax(logits_tensor, dim=1).numpy()
    targets = targets_tensor.numpy()
    return E2EvaluationResult(
        loss=loss_sum / samples, metrics=classification_metrics(targets, probabilities),
        logits=logits_tensor.numpy(), targets=targets, probabilities=probabilities,
        predictions=probabilities.argmax(axis=1), processed_images=samples, batches=batches,
        subtype_logits_used_for_prediction=False,
    )


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_validation_artifacts(output_dir: Path, prefix: str, epoch: int, train: E2TrainEpochResult, result: E2EvaluationResult, image_paths: list[str]) -> None:
    _write_json(output_dir / f"{prefix}_validation_metrics.json", {
        "epoch": epoch, "train_main_ce": train.main_loss, "train_subtype_ce": train.subtype_loss,
        "train_total_loss": train.total_loss, "validation_loss": result.loss,
        "prediction_source": "main_3_class_logits_only", **result.metrics,
    })
    prediction_frame(image_paths, result.targets, result.probabilities).to_csv(output_dir / f"{prefix}_validation_predictions.csv", index=False)
    pd.DataFrame(result.metrics["confusion_matrix"], index=["true_normal", "true_bacterial", "true_viral"], columns=["pred_normal", "pred_bacterial", "pred_viral"]).reset_index(names="ground_truth").to_csv(output_dir / f"{prefix}_validation_confusion_matrix.csv", index=False)


def fit_e2(training_config: Mapping[str, Any]) -> dict[str, Any]:
    """Full fresh E2 training. There is intentionally no checkpoint-resume argument."""
    validate_e2_fairness_contract(training_config)
    seed = int(training_config["training"]["seed"])
    seed_everything(seed)
    root = Path(training_config["resolved"]["project_root"])
    data_config = load_data_config(training_config["resolved"]["data_config_path"], project_root=root)
    if data_config["views"]["train"] != "two" or data_config["views"]["validation"] != "single" or int(data_config["loader"]["batch_size"]) != 16 or "test" in data_config.get("manifests", {}):
        raise ValueError("E2 requires E0's two-view/single-view, batch-size-16, train/validation-only data contract.")
    train_loader, validation_loader = build_dataloaders(data_config)
    if len(validation_loader.dataset) != 742:
        raise ValueError("E2 must validate on E0's complete 742-image validation split.")
    device = resolve_device(str(training_config["training"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("Full E2 training requires CUDA, matching E0.")
    model = build_e2_model(training_config).to(device)
    main_criterion = nn.CrossEntropyLoss(label_smoothing=float(training_config["loss"]["classification"]["label_smoothing"]))
    subtype_criterion = nn.CrossEntropyLoss()
    optimizer = create_e2_optimizer(model, training_config)
    scheduler = create_e2_scheduler(optimizer, training_config)
    lambda_subtype = float(training_config["loss"]["subtype"]["lambda_subtype"])
    output_dir = Path(training_config["resolved"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "training_config.json", training_config)
    manifest_fingerprints = {"train": file_sha256(data_config["resolved"]["train_manifest"]), "validation": file_sha256(data_config["resolved"]["validation_manifest"])}
    history: list[dict[str, Any]] = []
    best_metric = float("-inf")
    stale_epochs = 0
    best_epoch = 0
    checkpoint_config = training_config["training"]["checkpoint"]
    validation_paths = validation_loader.dataset.frame["image_path"].tolist()
    for epoch in range(1, int(training_config["training"]["epochs"]) + 1):
        train = train_one_epoch_e2(model, train_loader, optimizer=optimizer, main_criterion=main_criterion, subtype_criterion=subtype_criterion, lambda_subtype=lambda_subtype, device=device, gradient_clip_norm=float(training_config["training"]["gradient_clip_norm"]))
        validation = evaluate_e2(model, validation_loader, main_criterion=main_criterion, device=device)
        metric = float(validation.metrics["macro_f1"])
        improved = metric > best_metric
        if improved:
            best_metric, best_epoch, stale_epochs = metric, epoch, 0
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch, "train_main_ce": train.main_loss, "train_subtype_ce": train.subtype_loss,
            "train_total_loss": train.total_loss, "train_pneumonia_source_samples": train.pneumonia_source_samples,
            "train_excluded_normal_source_samples": train.excluded_normal_source_samples,
            "train_pneumonia_views": train.pneumonia_views, "train_excluded_normal_views": train.excluded_normal_views,
            "validation_loss": validation.loss, "validation_accuracy": validation.metrics["accuracy"],
            "validation_macro_f1": metric, "validation_macro_auroc": validation.metrics["macro_auroc_ovr"],
            "selection_metric_source": "main_3_class_logits_only", "is_best": improved, "stale_epochs": stale_epochs,
        }
        history.append(row)
        _write_json(output_dir / "history.json", history)
        _write_validation_artifacts(output_dir, "last", epoch, train, validation, validation_paths)
        if improved:
            _write_validation_artifacts(output_dir, "best", epoch, train, validation, validation_paths)
        scheduler.step()
        checkpoint_kwargs = {
            "model": model, "epoch": epoch, "metrics": row, "config": training_config,
            "optimizer": optimizer, "scheduler": scheduler, "best_validation_macro_f1": best_metric,
            "random_seed": seed, "control_state": {"best_epoch": best_epoch, "stale_epochs": stale_epochs, "manifest_fingerprints": manifest_fingerprints, "initialization": "fresh_imagenet1k_v1_no_e0_e1_checkpoint"},
            "history": history, "train_loader_generator": train_loader.generator,
            "save_rng_state": bool(checkpoint_config["save_rng_state"]), "run_kind": "e2_full_epoch",
        }
        save_checkpoint(output_dir / checkpoint_config["last_filename"], **checkpoint_kwargs)
        if improved:
            save_checkpoint(output_dir / checkpoint_config["best_filename"], **checkpoint_kwargs)
        if stale_epochs >= int(training_config["training"]["early_stopping_patience"]):
            break
    return {"best_validation_macro_f1": best_metric, "best_epoch": best_epoch, "epochs_completed": len(history), "manifest_fingerprints": manifest_fingerprints, "output_dir": str(output_dir)}
