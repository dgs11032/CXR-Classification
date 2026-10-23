"""E9-only fail-closed E6 encoder transfer into a fresh binary specialist head."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .e8_models import ConvNeXtTinyE8, NUM_BINARY_CLASSES
from .models import FEATURE_DIMENSION

EXPECTED_TOP_LEVEL_KEYS = {
    "checkpoint_version", "run_kind", "config_fingerprint", "epoch", "next_epoch",
    "state_contract", "model_state", "optimizer_state", "scheduler_state", "scaler_state",
    "metrics", "best_validation_macro_f1", "random_seed", "control_state", "history",
    "train_loader_generator_state", "rng_state", "config",
}
SOURCE_CLASSIFIER_KEYS = ("classifier.weight", "classifier.bias")
ENCODER_PREFIXES = ("features.", "feature_norm.")
EXPECTED_ENCODER_TENSOR_COUNT = 180


class ConvNeXtTinyE9(ConvNeXtTinyE8):
    """ConvNeXt-Tiny with a fresh Linear(768, 2) head before E6 encoder transfer."""


def build_e9_model(config: Mapping[str, Any]) -> ConvNeXtTinyE9:
    if config.get("experiment") != "e9":
        raise ValueError("E9 model requires experiment=e9.")
    model = config["model"]
    expected = {
        "architecture": "flat_binary_specialist", "backbone": "convnext_tiny",
        "weights": "IMAGENET1K_V1", "pretrained": False, "allow_weight_download": False,
        "feature_dimension": FEATURE_DIMENSION, "num_classes": NUM_BINARY_CLASSES,
        "classification_head": "Linear(768, 2)", "trainable_scope": "full",
    }
    mismatches = {key: {"expected": value, "actual": model.get(key)} for key, value in expected.items() if model.get(key) != value}
    if mismatches:
        raise ValueError(f"E9 model contract drift: {mismatches}")
    return ConvNeXtTinyE9(pretrained=False, allow_weight_download=False, trainable_scope="full")


def _inside_project(path: str | Path, project_root: str | Path) -> Path:
    root = Path(project_root).expanduser().resolve()
    candidate = Path(path).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"E6 source checkpoint must remain inside project root: {path}")
    return resolved


def _require_e6_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required E6 best checkpoint is unavailable: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != EXPECTED_TOP_LEVEL_KEYS:
        actual = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"E6 checkpoint top-level contract mismatch: {actual}")
    if payload["checkpoint_version"] != 2 or payload["run_kind"] != "e6_full_epoch":
        raise ValueError("E6 checkpoint must be checkpoint_version=2 and run_kind=e6_full_epoch.")
    source_config = payload["config"]
    if not isinstance(source_config, dict) or source_config.get("experiment") != "e6":
        raise ValueError("E6 checkpoint config.experiment must be e6.")
    source_model = source_config.get("model")
    expected_model = {
        "architecture": "flat", "backbone": "convnext_tiny", "weights": "IMAGENET1K_V1",
        "feature_dimension": FEATURE_DIMENSION, "num_classes": 3,
        "classification_head": "Linear(768, 3)", "trainable_scope": "full",
    }
    if not isinstance(source_model, dict) or any(source_model.get(key) != value for key, value in expected_model.items()):
        raise ValueError("E6 checkpoint model contract is not the supported ConvNeXt-Tiny Linear(768,3) layout.")
    state = payload["model_state"]
    if not isinstance(state, dict):
        raise ValueError("E6 model_state must be a mapping.")
    return payload


def transfer_e6_encoder(
    model: ConvNeXtTinyE9,
    *,
    checkpoint_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Strictly copy only E6 features/feature_norm tensors; preserve the new binary head."""
    source_path = _inside_project(checkpoint_path, project_root)
    payload = _require_e6_payload(source_path)
    source = payload["model_state"]
    target_before = model.state_dict()
    target_encoder_keys = {key for key in target_before if key.startswith(ENCODER_PREFIXES)}
    target_non_encoder_keys = set(target_before) - target_encoder_keys
    if target_non_encoder_keys != set(SOURCE_CLASSIFIER_KEYS) or len(target_encoder_keys) != EXPECTED_ENCODER_TENSOR_COUNT:
        raise RuntimeError("E9 target state is not exactly 180 encoder tensors plus a binary classifier weight/bias.")
    expected_source_keys = target_encoder_keys | set(SOURCE_CLASSIFIER_KEYS)
    if set(source) != expected_source_keys or len(source) != 182:
        raise ValueError("E6 model_state key set is not the exact expected 180 encoder + 2 classifier layout.")
    if tuple(source["classifier.weight"].shape) != (3, FEATURE_DIMENSION) or tuple(source["classifier.bias"].shape) != (3,):
        raise ValueError("E6 source classifier must be Linear(768, 3).")
    if tuple(target_before["classifier.weight"].shape) != (2, FEATURE_DIMENSION) or tuple(target_before["classifier.bias"].shape) != (2,):
        raise ValueError("E9 target classifier must be fresh Linear(768, 2).")
    shape_mismatches = {
        key: {"source": tuple(source[key].shape), "target": tuple(target_before[key].shape)}
        for key in sorted(target_encoder_keys)
        if not isinstance(source[key], torch.Tensor) or tuple(source[key].shape) != tuple(target_before[key].shape)
    }
    if shape_mismatches:
        raise ValueError(f"E6/E9 encoder tensor shape mismatch: {shape_mismatches}")
    fresh_head = {key: target_before[key].detach().clone() for key in SOURCE_CLASSIFIER_KEYS}
    assembled = dict(target_before)
    assembled.update({key: source[key].detach().clone().to(dtype=target_before[key].dtype) for key in target_encoder_keys})
    incompatible = model.load_state_dict(assembled, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict E9 assembled state load failed: {incompatible}")
    loaded_state = model.state_dict()
    unequal = [key for key in sorted(target_encoder_keys) if not torch.equal(loaded_state[key].cpu(), source[key].cpu())]
    head_preserved = all(torch.equal(loaded_state[key].cpu(), fresh_head[key].cpu()) for key in SOURCE_CLASSIFIER_KEYS)
    if unequal or not head_preserved:
        raise RuntimeError(f"E6 encoder transfer verification failed: unequal={unequal[:5]}, fresh_binary_head_preserved={head_preserved}")
    return {
        "source_checkpoint": str(source_path),
        "source_epoch": int(payload["epoch"]),
        "source_best_validation_macro_f1": float(payload["best_validation_macro_f1"]),
        "transferred_encoder_tensor_count": len(target_encoder_keys),
        "encoder_tensor_ratio": len(target_encoder_keys) / len(target_before),
        "source_classifier_shapes": {key: list(source[key].shape) for key in SOURCE_CLASSIFIER_KEYS},
        "target_classifier_shapes": {key: list(loaded_state[key].shape) for key in SOURCE_CLASSIFIER_KEYS},
        "missing_keys": [],
        "unexpected_keys": [],
        "all_encoder_tensors_equal": True,
        "fresh_binary_classifier_preserved": True,
    }
