"""E1 CE + pneumonia subtype-only SupCon training and validation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data.dataloader import build_dataloaders, load_data_config

from .e1_losses import PneumoniaSubtypeSupConLoss, SupConResult
from .checkpoints import save_checkpoint
from .e1_models import ConvNeXtTinyE1
from .metrics import classification_metrics
from .training import (
    PROJECT_ROOT,
    file_sha256,
    flatten_two_view_batch,
    resolve_device,
    seed_everything,
)


@dataclass
class E1TrainBatchResult:
    ce_loss: float
    supcon_loss: float
    total_loss: float
    pneumonia_source_samples: int
    excluded_normal_source_samples: int
    pneumonia_views: int
    excluded_normal_views: int
    gradients_finite: bool


@dataclass
class E1EvaluationResult:
    loss: float
    metrics: dict[str, Any]
    logits: np.ndarray
    targets: np.ndarray
    probabilities: np.ndarray


def load_e1_training_config(
    config_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Load and validate only the independent E1 configuration contract."""
    root = Path(project_root).expanduser().resolve()
    path = Path(config_path).expanduser()
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"E1 config is not available inside the project: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("experiment") != "e1":
        raise ValueError("E1 config must be a YAML mapping with experiment=e1.")
    model = config["model"]
    loss = config["loss"]
    if int(model["feature_dimension"]) != 768 or int(model["num_classes"]) != 3:
        raise ValueError("E1 requires a 768-dim encoder and three classifier classes.")
    if (int(model["projection_hidden_dimension"]), int(model["projection_dimension"])) != (256, 128):
        raise ValueError("E1 projector must be 768 -> 256 -> 128.")
    supcon = loss["supervised_contrastive"]
    if float(supcon["temperature"]) != 0.07 or float(supcon["lambda_supcon"]) != 0.1:
        raise ValueError("E1 requires temperature=0.07 and lambda_supcon=0.1.")
    if list(supcon["included_labels"]) != [1, 2] or int(supcon["excluded_label"]) != 0:
        raise ValueError("E1 SupCon must include only labels 1 and 2 and exclude label 0.")
    output_dir = (root / config["output_dir"]).resolve()
    data_config_path = (root / config["data_config"]).resolve()
    if not output_dir.is_relative_to(root) or not data_config_path.is_relative_to(root):
        raise ValueError("E1 paths must remain inside the project root.")
    config["resolved"] = {
        "project_root": str(root),
        "config_path": str(path),
        "data_config_path": str(data_config_path),
        "output_dir": str(output_dir),
    }
    return config


def build_e1_model(config: Mapping[str, Any]) -> ConvNeXtTinyE1:
    model_config = config["model"]
    return ConvNeXtTinyE1(
        pretrained=bool(model_config["pretrained"]),
        allow_weight_download=bool(model_config["allow_weight_download"]),
        projection_hidden_dimension=int(model_config["projection_hidden_dimension"]),
        projection_dimension=int(model_config["projection_dimension"]),
        trainable_scope=str(model_config["trainable_scope"]),  # type: ignore[arg-type]
    )


def create_e1_optimizer(
    model: ConvNeXtTinyE1, config: Mapping[str, Any]
) -> torch.optim.Optimizer:
    optimizer_config = config["optimizer"]
    backbone_modules = (model.features, model.avgpool, model.feature_norm)
    head_modules = (model.classifier, model.projector)
    backbone = [p for module in backbone_modules for p in module.parameters() if p.requires_grad]
    head = [p for module in head_modules for p in module.parameters() if p.requires_grad]
    if not backbone or not head or {id(p) for p in backbone} & {id(p) for p in head}:
        raise RuntimeError("E1 optimizer parameter groups are invalid.")
    grouped = {id(p) for p in backbone + head}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if grouped != trainable:
        raise RuntimeError("E1 optimizer omits or duplicates trainable parameters.")
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": float(optimizer_config["backbone_learning_rate"]), "name": "backbone"},
            {"params": head, "lr": float(optimizer_config["head_learning_rate"]), "name": "head"},
        ],
        weight_decay=float(optimizer_config["weight_decay"]),
    )


def create_e1_scheduler(
    optimizer: torch.optim.Optimizer, config: Mapping[str, Any]
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    scheduler_config = config["scheduler"]
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(scheduler_config["t_max"]),
        eta_min=float(scheduler_config["eta_min"]),
    )


def _finite_gradients(model: nn.Module) -> bool:
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    return bool(gradients) and all(bool(torch.isfinite(grad).all()) for grad in gradients)


def compute_e1_losses(
    model: ConvNeXtTinyE1,
    flattened_images: Tensor,
    expanded_labels: Tensor,
    *,
    ce_criterion: nn.CrossEntropyLoss,
    supcon_criterion: PneumoniaSubtypeSupConLoss,
    lambda_supcon: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, SupConResult]:
    """Compute unchanged three-class CE plus masked subtype-only SupCon."""
    logits, features, projections = model.forward_with_projection(flattened_images)
    ce_loss = ce_criterion(logits, expanded_labels)
    supcon_result = supcon_criterion(projections, expanded_labels)
    total_loss = ce_loss + float(lambda_supcon) * supcon_result.loss
    for name, tensor in (("logits", logits), ("features", features), ("ce", ce_loss), ("supcon", supcon_result.loss), ("total", total_loss)):
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"Non-finite E1 {name} tensor.")
    return logits, features, projections, ce_loss, total_loss, supcon_result


def train_one_epoch_e1(
    model: ConvNeXtTinyE1,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    ce_criterion: nn.CrossEntropyLoss,
    supcon_criterion: PneumoniaSubtypeSupConLoss,
    lambda_supcon: float,
    device: torch.device,
    gradient_clip_norm: float | None,
    max_batches: int | None = None,
) -> E1TrainBatchResult:
    """Run E1 training while retaining E0's two-view CE label expansion exactly."""
    model.train()
    totals = {"ce": 0.0, "supcon": 0.0, "total": 0.0, "pneumonia_sources": 0, "normal_sources": 0, "pneumonia_views": 0, "normal_views": 0}
    processed = 0
    for batch_index, (views, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        flattened, expanded_labels = flatten_two_view_batch(views, labels)
        flattened, expanded_labels = flattened.to(device), expanded_labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, _, _, ce_loss, total_loss, supcon = compute_e1_losses(
            model, flattened, expanded_labels, ce_criterion=ce_criterion,
            supcon_criterion=supcon_criterion, lambda_supcon=lambda_supcon,
        )
        total_loss.backward()
        if not _finite_gradients(model):
            raise FloatingPointError(f"Non-finite or absent E1 gradient at batch {batch_index}.")
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        source_pneumonia = int(((labels == 1) | (labels == 2)).sum())
        source_normal = int((labels == 0).sum())
        count = len(expanded_labels)
        totals["ce"] += float(ce_loss.detach().cpu()) * count
        totals["supcon"] += float(supcon.loss.detach().cpu()) * count
        totals["total"] += float(total_loss.detach().cpu()) * count
        totals["pneumonia_sources"] += source_pneumonia
        totals["normal_sources"] += source_normal
        totals["pneumonia_views"] += supcon.eligible_view_count
        totals["normal_views"] += supcon.excluded_normal_view_count
        processed += count
    if processed == 0:
        raise ValueError("No train batch was processed for E1.")
    return E1TrainBatchResult(
        ce_loss=totals["ce"] / processed, supcon_loss=totals["supcon"] / processed,
        total_loss=totals["total"] / processed,
        pneumonia_source_samples=totals["pneumonia_sources"],
        excluded_normal_source_samples=totals["normal_sources"],
        pneumonia_views=totals["pneumonia_views"], excluded_normal_views=totals["normal_views"],
        gradients_finite=True,
    )


@torch.inference_mode()
def evaluate_e1(
    model: ConvNeXtTinyE1,
    loader: DataLoader,
    *,
    ce_criterion: nn.CrossEntropyLoss,
    device: torch.device,
    max_batches: int | None = None,
) -> E1EvaluationResult:
    """Evaluate only E0-equivalent 3-class CE logits on deterministic single views."""
    model.eval()
    logits_parts: list[Tensor] = []
    label_parts: list[Tensor] = []
    loss_sum = 0.0
    samples = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if images.ndim != 4:
            raise ValueError(f"Validation expects [B,C,H,W], got {tuple(images.shape)}")
        logits = model(images.to(device))
        loss = ce_criterion(logits, labels.to(device))
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite E1 validation tensor at batch {batch_index}.")
        logits_parts.append(logits.float().cpu())
        label_parts.append(labels.cpu())
        loss_sum += float(loss.cpu()) * len(labels)
        samples += len(labels)
    if samples == 0:
        raise ValueError("No validation batch was processed for E1.")
    logits_tensor, labels_tensor = torch.cat(logits_parts), torch.cat(label_parts)
    probabilities = torch.softmax(logits_tensor, dim=1).numpy()
    targets = labels_tensor.numpy()
    return E1EvaluationResult(
        loss=loss_sum / samples, metrics=classification_metrics(targets, probabilities),
        logits=logits_tensor.numpy(), targets=targets, probabilities=probabilities,
    )


def fit_e1(training_config: Mapping[str, Any]) -> dict[str, Any]:
    """Full E1 training entrypoint; call only through the explicit CLI confirmation."""
    seed = int(training_config["training"]["seed"])
    seed_everything(seed)
    root = Path(training_config["resolved"]["project_root"])
    data_config = load_data_config(
        training_config["resolved"]["data_config_path"], project_root=root
    )
    train_loader, validation_loader = build_dataloaders(data_config)
    model = build_e1_model(training_config)
    device = resolve_device(str(training_config["training"]["device"]))
    model.to(device)
    ce_criterion = nn.CrossEntropyLoss(
        label_smoothing=float(training_config["loss"]["classification"]["label_smoothing"])
    )
    supcon_criterion = PneumoniaSubtypeSupConLoss(
        temperature=float(training_config["loss"]["supervised_contrastive"]["temperature"])
    )
    optimizer = create_e1_optimizer(model, training_config)
    scheduler = create_e1_scheduler(optimizer, training_config)
    output_dir = Path(training_config["resolved"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_fingerprints = {
        "train": file_sha256(data_config["resolved"]["train_manifest"]),
        "validation": file_sha256(data_config["resolved"]["validation_manifest"]),
    }
    history: list[dict[str, Any]] = []
    best_metric = float("-inf")
    stale_epochs = 0
    patience = int(training_config["training"]["early_stopping_patience"])
    checkpoint_config = training_config["training"]["checkpoint"]
    lambda_supcon = float(training_config["loss"]["supervised_contrastive"]["lambda_supcon"])

    for epoch in range(1, int(training_config["training"]["epochs"]) + 1):
        train = train_one_epoch_e1(
            model,
            train_loader,
            optimizer=optimizer,
            ce_criterion=ce_criterion,
            supcon_criterion=supcon_criterion,
            lambda_supcon=lambda_supcon,
            device=device,
            gradient_clip_norm=float(training_config["training"]["gradient_clip_norm"]),
        )
        validation = evaluate_e1(
            model, validation_loader, ce_criterion=ce_criterion, device=device
        )
        scheduler.step()
        metric = float(validation.metrics["macro_f1"])
        improved = metric > best_metric
        if improved:
            best_metric = metric
            stale_epochs = 0
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_ce_loss": train.ce_loss,
            "train_supcon_loss": train.supcon_loss,
            "train_total_loss": train.total_loss,
            "validation_loss": validation.loss,
            "validation_accuracy": validation.metrics["accuracy"],
            "validation_macro_f1": metric,
            "validation_macro_auroc": validation.metrics["macro_auroc_ovr"],
            "is_best": improved,
            "stale_epochs": stale_epochs,
        }
        history.append(row)
        (output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        checkpoint_kwargs = {
            "model": model,
            "epoch": epoch,
            "metrics": row,
            "config": training_config,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "best_validation_macro_f1": best_metric,
            "random_seed": seed,
            "control_state": {
                "manifest_fingerprints": manifest_fingerprints,
                "stale_epochs": stale_epochs,
            },
            "history": history,
            "train_loader_generator": train_loader.generator,
            "save_rng_state": bool(checkpoint_config["save_rng_state"]),
            "run_kind": "e1_full_epoch",
        }
        save_checkpoint(output_dir / checkpoint_config["last_filename"], **checkpoint_kwargs)
        if improved:
            save_checkpoint(output_dir / checkpoint_config["best_filename"], **checkpoint_kwargs)
        if stale_epochs >= patience:
            break

    return {
        "best_validation_macro_f1": best_metric,
        "epochs_completed": len(history),
        "manifest_fingerprints": manifest_fingerprints,
        "output_dir": str(output_dir),
    }
