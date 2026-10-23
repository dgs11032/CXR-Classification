"""E7-only E0-parity training with unweighted multiclass Focal Loss (gamma=1.0)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from data.dataloader import build_dataloaders, load_data_config

from .checkpoints import save_checkpoint
from .e3_training import evaluate_e3, train_one_epoch_e3
from .e7_models import ConvNeXtTinyE7, build_e7_model
from .training import (
    PROJECT_ROOT,
    create_optimizer,
    create_scheduler,
    file_sha256,
    resolve_device,
    seed_everything,
)

GAMMA = 1.0

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
class FocalLossComponents:
    per_sample_ce: Tensor
    pt: Tensor
    modulation: Tensor
    per_sample_focal: Tensor


class MulticlassFocalLoss(nn.Module):
    """Mean((1 - exp(-CE_i))**gamma * CE_i) with no alpha/class weighting."""

    def __init__(self, *, gamma: float = GAMMA, alpha: None = None, reduction: str = "mean") -> None:
        super().__init__()
        if gamma != GAMMA or alpha is not None or reduction != "mean":
            raise ValueError("E7 requires gamma=1.0, alpha=None, and reduction='mean'.")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def components(self, logits: Tensor, targets: Tensor) -> FocalLossComponents:
        if logits.ndim != 2 or logits.shape[1] != 3 or targets.ndim != 1 or len(logits) != len(targets):
            raise ValueError(
                f"Expected logits [N,3] and targets [N], got {tuple(logits.shape)} / {tuple(targets.shape)}"
            )
        per_sample_ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=0.0)
        pt = torch.exp(-per_sample_ce)
        modulation = (1.0 - pt).pow(self.gamma)
        return FocalLossComponents(per_sample_ce, pt, modulation, modulation * per_sample_ce)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.components(logits, targets).per_sample_focal.mean()


def _value(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        value = value[key]
    return value


def validate_e7_fairness_contract(config: Mapping[str, Any]) -> None:
    """Fail closed unless E7 differs from E6 only by focal-loss gamma."""
    if config.get("experiment") != "e7":
        raise ValueError("E7 config must declare experiment=e7.")
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
        "name": "MulticlassFocalLoss",
        "gamma": GAMMA,
        "alpha": None,
        "class_weights": None,
        "label_smoothing": 0.0,
        "reduction": "mean",
    }
    for key, expected in expected_loss.items():
        if loss.get(key) != expected:
            mismatches.append({"path": f"loss.{key}", "expected": expected, "actual": loss.get(key)})
    for key, expected in {
        "data_config": "configs/data_config.yaml",
        "output_dir": "artifacts/experiments/e7",
    }.items():
        if config.get(key) != expected:
            mismatches.append({"path": key, "expected": expected, "actual": config.get(key)})
    forbidden = {"resume_from", "initial_checkpoint", "checkpoint_path", "load_checkpoint"}
    if present := sorted(forbidden & set(config)):
        mismatches.append({"path": "top_level_checkpoint_loading", "expected": "absent", "actual": present})
    if mismatches:
        raise ValueError(f"E7 fairness contract drift: {mismatches}")


def load_e7_training_config(
    config_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    path = Path(config_path).expanduser()
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"E7 config unavailable inside project: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("E7 config must be a YAML mapping.")
    validate_e7_fairness_contract(config)
    output = (root / config["output_dir"]).resolve()
    data = (root / config["data_config"]).resolve()
    if not output.is_relative_to(root) or not data.is_relative_to(root):
        raise ValueError("E7 paths must remain inside project root.")
    config["resolved"] = {
        "project_root": str(root),
        "config_path": str(path),
        "data_config_path": str(data),
        "output_dir": str(output),
    }
    return config


def create_e7_criterion(config: Mapping[str, Any]) -> MulticlassFocalLoss:
    validate_e7_fairness_contract(config)
    return MulticlassFocalLoss(
        gamma=float(config["loss"]["gamma"]),
        alpha=config["loss"]["alpha"],
        reduction=str(config["loss"]["reduction"]),
    )


def create_e7_optimizer(model: ConvNeXtTinyE7, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    return create_optimizer(model, config)


def create_e7_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return create_scheduler(optimizer, config)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def fit_e7(training_config: Mapping[str, Any]) -> dict[str, Any]:
    """Fresh E7 run only; no E0–E6 checkpoint load/resume API exists."""
    validate_e7_fairness_contract(training_config)
    seed = int(training_config["training"]["seed"])
    seed_everything(seed)
    root = Path(training_config["resolved"]["project_root"])
    data_config = load_data_config(training_config["resolved"]["data_config_path"], project_root=root)
    if (
        data_config["views"]["train"] != "two"
        or data_config["views"]["validation"] != "single"
        or int(data_config["loader"]["batch_size"]) != 16
        or "test" in data_config.get("manifests", {})
    ):
        raise ValueError("E7 requires E0's train/validation-only two-view/single-view batch-size-16 contract.")
    train_loader, validation_loader = build_dataloaders(data_config)
    if len(validation_loader.dataset) != 742:
        raise ValueError("E7 must validate on E0's full 742-image split.")
    device = resolve_device(str(training_config["training"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("Full E7 training requires CUDA, matching E0.")
    model = build_e7_model(training_config).to(device)
    criterion = create_e7_criterion(training_config)
    optimizer = create_e7_optimizer(model, training_config)
    scheduler = create_e7_scheduler(optimizer, training_config)
    output_dir = Path(training_config["resolved"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "training_config.json", training_config)
    fingerprints = {
        "train": file_sha256(data_config["resolved"]["train_manifest"]),
        "validation": file_sha256(data_config["resolved"]["validation_manifest"]),
    }
    history: list[dict[str, Any]] = []
    best, best_epoch, stale = float("-inf"), 0, 0
    for epoch in range(1, int(training_config["training"]["epochs"]) + 1):
        train = train_one_epoch_e3(
            model,
            train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=float(training_config["training"]["gradient_clip_norm"]),
        )
        validation = evaluate_e3(model, validation_loader, device=device)
        metric = float(validation.metrics["macro_f1"])
        improved = metric > best
        if improved:
            best, best_epoch, stale = metric, epoch, 0
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "train_focal_loss": train.loss,
            "validation_unweighted_ce": validation.loss,
            "validation_accuracy": validation.metrics["accuracy"],
            "validation_macro_f1": metric,
            "validation_macro_auroc": validation.metrics["macro_auroc_ovr"],
            "is_best": improved,
            "stale_epochs": stale,
        }
        history.append(row)
        _write_json(output_dir / "history.json", history)
        scheduler.step()
        payload = {
            "model": model,
            "epoch": epoch,
            "metrics": row,
            "config": training_config,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "best_validation_macro_f1": best,
            "random_seed": seed,
            "control_state": {
                "best_epoch": best_epoch,
                "stale_epochs": stale,
                "manifest_fingerprints": fingerprints,
                "initialization": "fresh_imagenet1k_v1_no_e0_e1_e2_e3_e4_e5_e6_checkpoint",
                "loss": "unweighted_multiclass_focal_gamma_1_alpha_none",
            },
            "history": history,
            "train_loader_generator": train_loader.generator,
            "save_rng_state": bool(training_config["training"]["checkpoint"]["save_rng_state"]),
            "run_kind": "e7_full_epoch",
        }
        save_checkpoint(output_dir / training_config["training"]["checkpoint"]["last_filename"], **payload)
        if improved:
            save_checkpoint(output_dir / training_config["training"]["checkpoint"]["best_filename"], **payload)
        if stale >= int(training_config["training"]["early_stopping_patience"]):
            break
    return {
        "best_validation_macro_f1": best,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "manifest_fingerprints": fingerprints,
        "output_dir": str(output_dir),
    }
