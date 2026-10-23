"""E9-only E6 CXR-encoder transfer training for the E8 binary specialist protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from data.dataloader import load_data_config

from .checkpoints import save_checkpoint
from .e8_data import BINARY_CLASS_ORDER, EXPECTED_COUNTS, build_e8_dataloaders
from .e8_training import BinaryMulticlassFocalLoss, evaluate_e8, train_one_epoch_e8
from .e9_models import build_e9_model, transfer_e6_encoder
from .training import PROJECT_ROOT, create_optimizer, create_scheduler, file_sha256, resolve_device, seed_everything

GAMMA = 2.0

_COMMON_E8_CONTROLS = {
    ("data_config",): "configs/data_config.yaml",
    ("binary_specialist", "class_order"): list(BINARY_CLASS_ORDER),
    ("binary_specialist", "canonical_label_mapping"): {"BACTERIAL_PNEUMONIA": 1, "VIRAL_PNEUMONIA": 2},
    ("binary_specialist", "binary_label_mapping"): {"BACTERIAL_PNEUMONIA": 0, "VIRAL_PNEUMONIA": 1},
    ("binary_specialist", "expected_source_counts"): EXPECTED_COUNTS,
    ("model", "architecture"): "flat_binary_specialist",
    ("model", "backbone"): "convnext_tiny",
    ("model", "weights"): "IMAGENET1K_V1",
    ("model", "allow_weight_download"): False,
    ("model", "feature_dimension"): 768,
    ("model", "num_classes"): 2,
    ("model", "classification_head"): "Linear(768, 2)",
    ("model", "trainable_scope"): "full",
    ("loss", "name"): "BinaryMulticlassFocalLoss",
    ("loss", "gamma"): GAMMA,
    ("loss", "alpha"): None,
    ("loss", "class_weights"): None,
    ("loss", "label_smoothing"): 0.0,
    ("loss", "reduction"): "mean",
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


def _value(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        value = value[key]
    return value


def validate_e9_fairness_contract(config: Mapping[str, Any]) -> None:
    """E9 locks every E8 training/data control; source initialization is the sole planned difference."""
    if config.get("experiment") != "e9":
        raise ValueError("E9 config must declare experiment=e9.")
    mismatches: list[dict[str, Any]] = []
    for path, expected in _COMMON_E8_CONTROLS.items():
        try:
            actual = _value(config, path)
        except (KeyError, TypeError):
            actual = "__MISSING__"
        if actual != expected:
            mismatches.append({"path": ".".join(path), "expected": expected, "actual": actual})
    expected_init = {
        "method": "e6_best_encoder_transfer", "source_checkpoint": "artifacts/experiments/e6/best.pt",
        "source_checkpoint_version": 2, "source_run_kind": "e6_full_epoch", "source_experiment": "e6",
        "source_classifier": "Linear(768, 3)", "transfer_prefixes": ["features.", "feature_norm."],
        "excluded_source_keys": ["classifier.weight", "classifier.bias"],
        "transferred_encoder_tensor_count": 180, "target_classifier": "Linear(768, 2)",
        "strict_state_assembly": True,
    }
    initialization = config.get("initialization", {})
    for key, expected in expected_init.items():
        if initialization.get(key) != expected:
            mismatches.append({"path": f"initialization.{key}", "expected": expected, "actual": initialization.get(key)})
    if config.get("output_dir") != "artifacts/experiments/e9":
        mismatches.append({"path": "output_dir", "expected": "artifacts/experiments/e9", "actual": config.get("output_dir")})
    # E9's only model-control divergence from E8 is not bootstrapping from ImageNet before overwriting all encoder tensors.
    if config.get("model", {}).get("pretrained") is not False:
        mismatches.append({"path": "model.pretrained", "expected": False, "actual": config.get("model", {}).get("pretrained")})
    forbidden = {"resume_from", "checkpoint_path", "load_checkpoint", "initial_checkpoint"}
    if present := sorted(forbidden & set(config)):
        mismatches.append({"path": "top_level_checkpoint_loading", "expected": "absent", "actual": present})
    if mismatches:
        raise ValueError(f"E9 fairness contract drift: {mismatches}")


def _read_yaml(path: Path) -> dict[str, Any]:
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return content


def e8_e9_control_comparison(e8_config: Mapping[str, Any], e9_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a fail-closed report proving all non-initialization controls match E8."""
    paths = list(_COMMON_E8_CONTROLS)
    mismatches: list[dict[str, Any]] = []
    for path in paths:
        try:
            e8_value, e9_value = _value(e8_config, path), _value(e9_config, path)
        except (KeyError, TypeError):
            mismatches.append({"path": ".".join(path), "error": "missing"})
            continue
        if e8_value != e9_value:
            mismatches.append({"path": ".".join(path), "e8": e8_value, "e9": e9_value})
    allowed_difference = {
        "e8_model_pretrained": e8_config.get("model", {}).get("pretrained"),
        "e9_model_pretrained": e9_config.get("model", {}).get("pretrained"),
        "e9_initialization": e9_config.get("initialization", {}).get("method"),
    }
    if allowed_difference != {"e8_model_pretrained": True, "e9_model_pretrained": False, "e9_initialization": "e6_best_encoder_transfer"}:
        mismatches.append({"path": "initialization_difference", "actual": allowed_difference})
    if mismatches:
        raise ValueError(f"E8/E9 non-initialization control mismatch: {mismatches}")
    return {"controls_match": True, "compared_control_count": len(paths), "only_difference": "E8 ImageNet encoder initialization; E9 E6 best encoder transfer"}


def load_e9_training_config(config_path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    path = Path(config_path).expanduser()
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"E9 config unavailable inside project: {path}")
    config = _read_yaml(path)
    validate_e9_fairness_contract(config)
    data_path = (root / config["data_config"]).resolve()
    output_path = (root / config["output_dir"]).resolve()
    e8_reference_path = (root / config["e8_reference_config"]).resolve()
    source_path = (root / config["initialization"]["source_checkpoint"]).resolve()
    if not all(item.is_relative_to(root) for item in (data_path, output_path, e8_reference_path, source_path)):
        raise ValueError("E9 paths must remain inside project root.")
    comparison = e8_e9_control_comparison(_read_yaml(e8_reference_path), config)
    config["resolved"] = {
        "project_root": str(root), "config_path": str(path), "data_config_path": str(data_path),
        "output_dir": str(output_path), "e8_reference_config_path": str(e8_reference_path),
        "source_checkpoint_path": str(source_path), "e8_parity": comparison,
    }
    return config


def create_e9_criterion(config: Mapping[str, Any]) -> BinaryMulticlassFocalLoss:
    validate_e9_fairness_contract(config)
    return BinaryMulticlassFocalLoss(gamma=float(config["loss"]["gamma"]), alpha=config["loss"]["alpha"], reduction=str(config["loss"]["reduction"]))


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fit_e9(training_config: Mapping[str, Any]) -> dict[str, Any]:
    """Fresh binary head + verified E6 encoder initialization; no resume path exists."""
    validate_e9_fairness_contract(training_config)
    seed = int(training_config["training"]["seed"])
    seed_everything(seed)
    root = Path(training_config["resolved"]["project_root"])
    data_config = load_data_config(training_config["resolved"]["data_config_path"], project_root=root)
    if data_config["views"]["train"] != "two" or data_config["views"]["validation"] != "single" or int(data_config["loader"]["batch_size"]) != 16 or "test" in data_config.get("manifests", {}):
        raise ValueError("E9 requires the E8 train/validation-only two-view/single-view batch-size-16 contract.")
    train_loader, validation_loader = build_e8_dataloaders(data_config)
    if len(train_loader.dataset) != 3300 or len(validation_loader.dataset) != 550:
        raise ValueError("E9 requires E8's fixed 3300/550 Bacterial/Viral subsets.")
    device = resolve_device(str(training_config["training"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("Full E9 training requires CUDA, matching E8.")
    model = build_e9_model(training_config)
    transfer_report = transfer_e6_encoder(model, checkpoint_path=training_config["resolved"]["source_checkpoint_path"], project_root=root)
    model = model.to(device)
    criterion = create_e9_criterion(training_config)
    optimizer = create_optimizer(model, training_config)
    scheduler = create_scheduler(optimizer, training_config)
    output_dir = Path(training_config["resolved"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "training_config.json", training_config)
    fingerprints = {"train": file_sha256(data_config["resolved"]["train_manifest"]), "validation": file_sha256(data_config["resolved"]["validation_manifest"])}
    history: list[dict[str, Any]] = []
    best, best_epoch, stale = float("-inf"), 0, 0
    for epoch in range(1, int(training_config["training"]["epochs"]) + 1):
        train = train_one_epoch_e8(model, train_loader, criterion=criterion, optimizer=optimizer, device=device, gradient_clip_norm=float(training_config["training"]["gradient_clip_norm"]))
        validation = evaluate_e8(model, validation_loader, device=device)
        metric = float(validation.metrics["macro_f1"])
        improved = metric > best
        if improved:
            best, best_epoch, stale = metric, epoch, 0
        else:
            stale += 1
        row = {"epoch": epoch, "train_focal_loss": train.loss, "validation_unweighted_ce": validation.loss, "validation_accuracy": validation.metrics["accuracy"], "validation_macro_f1": metric, "validation_auroc": validation.metrics["auroc"], "is_best": improved, "stale_epochs": stale}
        history.append(row)
        _write_json(output_dir / "history.json", history)
        scheduler.step()
        payload = {"model": model, "epoch": epoch, "metrics": row, "config": training_config, "optimizer": optimizer, "scheduler": scheduler, "best_validation_macro_f1": best, "random_seed": seed, "control_state": {"best_epoch": best_epoch, "stale_epochs": stale, "manifest_fingerprints": fingerprints, "initialization": "e6_best_encoder_transfer_fresh_binary_linear_768_2", "e6_encoder_transfer": transfer_report, "binary_class_order": list(BINARY_CLASS_ORDER), "canonical_to_binary_labels": {"1": 0, "2": 1}, "filtered_source_counts": EXPECTED_COUNTS, "loss": "unweighted_binary_focal_gamma_2_alpha_none"}, "history": history, "train_loader_generator": train_loader.generator, "save_rng_state": bool(training_config["training"]["checkpoint"]["save_rng_state"]), "run_kind": "e9_binary_full_epoch"}
        save_checkpoint(output_dir / training_config["training"]["checkpoint"]["last_filename"], **payload)
        if improved:
            save_checkpoint(output_dir / training_config["training"]["checkpoint"]["best_filename"], **payload)
        if stale >= int(training_config["training"]["early_stopping_patience"]):
            break
    return {"best_validation_macro_f1": best, "best_epoch": best_epoch, "epochs_completed": len(history), "manifest_fingerprints": fingerprints, "e6_encoder_transfer": transfer_report, "output_dir": str(output_dir)}
