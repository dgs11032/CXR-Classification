"""E17: frozen E15 RAD-DINO with PSPNet-inspired multi-scale pyramid pooling.

This is a classification experiment, not a PSPNet segmentation implementation.  It
restores the E15 final transformer block and E15 PatchAttentionHead strictly, then
uses only frozen RAD-DINO final-layer CLS and adaptive-pooled spatial patch tokens.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .e14_patch_specialist import (
    BINARY_CLASSES,
    CANONICAL_CLASSES,
    PatchAttentionHead,
    RawManifestDataset,
    _binary_metrics,
    _binary_train_frame,
    _collate,
    _load_e10_trace,
    _manifest,
)
from .e15_finetune import _load_encoder_and_last_block
from .metrics import classification_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_VERSION = 1


@dataclass(frozen=True)
class PyramidFeatureSet:
    split: str
    spatial_tokens: np.ndarray
    cls_features: np.ndarray
    labels: np.ndarray
    index: pd.DataFrame
    from_cache: bool


def _inside_project(value: str | Path, root: Path) -> Path:
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"E17 path leaves project root: {value}")
    return resolved


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"E17 YAML must be a mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_hash(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, *, spatial_tokens: np.ndarray, cls_features: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, spatial_tokens=spatial_tokens, cls_features=cls_features, labels=labels)
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def validate_e17_contract(config: Mapping[str, Any]) -> None:
    required = {
        "experiment": "e17",
        "status": "frozen_e15_rad_dino_pspnet_inspired_multiscale_pyramid_pooling_bv_classifier",
        "data_config": "configs/data_config.yaml",
        "output_dir": "artifacts/experiments/e17",
    }
    errors = [{"key": key, "expected": value, "actual": config.get(key)} for key, value in required.items() if config.get(key) != value]
    expected_model = {
        "hf_model_id": "microsoft/rad-dino", "revision": "main", "trust_remote_code": False,
        "required_transformers_version": "4.57.6", "expected_hidden_size": 768,
        "expected_num_hidden_layers": 12, "patch_size": 14, "expected_image_size": 518,
        "e15_restore": "base_rad_dino_plus_strict_final_last_block_and_patch_attention_head",
        "encoder_trainable": False, "e15_head_trainable": False,
    }
    model = config.get("model", {})
    errors.extend({"key": f"model.{key}", "expected": value, "actual": model.get(key)} for key, value in expected_model.items() if model.get(key) != value)
    e15_head = config.get("e15_head", {})
    expected_e15_head = {"architecture": "cls_plus_4x4_patch_attention_pooling_binary_head", "attention_hidden_dimension": 128, "classifier_hidden_dimension": 256, "dropout": 0.1, "num_classes": 2}
    errors.extend({"key": f"e15_head.{key}", "expected": value, "actual": e15_head.get(key)} for key, value in expected_e15_head.items() if e15_head.get(key) != value)
    pooling = config.get("pyramid_pooling", {})
    expected_pooling = {"method": "adaptive_average_pool2d_on_rad_dino_patch_grid", "patch_grid": [37, 37], "scales": [[1, 1], [2, 2], [3, 3], [6, 6]], "tokens_per_scale": [1, 4, 9, 36], "total_spatial_tokens": 50, "token_order": "scale_order_then_row_major_within_scale", "cache_dtype": "float16"}
    errors.extend({"key": f"pyramid_pooling.{key}", "expected": value, "actual": pooling.get(key)} for key, value in expected_pooling.items() if pooling.get(key) != value)
    head = config.get("multiscale_head", {})
    expected_head = {"architecture": "per_scale_spatial_attention_pooling_plus_e15_cls_mlp", "attention_hidden_dimension": 128, "descriptor_count": 5, "feature_dimension": 3840, "classifier_hidden_dimension": 128, "dropout": 0.3, "num_classes": 2}
    errors.extend({"key": f"multiscale_head.{key}", "expected": value, "actual": head.get(key)} for key, value in expected_head.items() if head.get(key) != value)
    runtime, training, split = config.get("runtime", {}), config.get("training", {}), config.get("split_contract", {})
    expected_runtime = {"device": "cuda", "feature_batch_size": 1, "head_batch_size": 64, "num_workers": 0, "pin_memory": True, "amp": True, "cache_features": True}
    errors.extend({"key": f"runtime.{key}", "expected": value, "actual": runtime.get(key)} for key, value in expected_runtime.items() if runtime.get(key) != value)
    expected_training = {"seed": 42, "fixed_epochs": 5, "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001, "checkpoint_policy": "final_epoch_only", "internal_holdout": "disabled", "early_stopping": False, "validation_weight_selection": False, "validation_threshold_selection": False, "hyperparameter_search": False}
    errors.extend({"key": f"training.{key}", "expected": value, "actual": training.get(key)} for key, value in expected_training.items() if training.get(key) != value)
    if split.get("binary_class_order") != list(BINARY_CLASSES) or split.get("canonical_to_binary") != {1: 0, 2: 1} or bool(split.get("no_test_access")) is not True:
        errors.append({"key": "split_contract", "actual": split})
    if config.get("e10_routing_input", {}).get("normal_rule") != "e6_pred_zero_kept_else_e17_binary_plus_one":
        errors.append({"key": "e10_routing_input.normal_rule", "actual": config.get("e10_routing_input", {}).get("normal_rule")})
    if float(config.get("e15_initialization", {}).get("global_baseline_probability_atol", -1)) != 0.0005:
        errors.append({"key": "e15_initialization.global_baseline_probability_atol", "actual": config.get("e15_initialization", {}).get("global_baseline_probability_atol")})
    if errors:
        raise ValueError(f"E17 configuration contract drift: {errors}")


def load_e17_config(path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config_path = _inside_project(path, root)
    if not config_path.is_file():
        raise FileNotFoundError(f"E17 config unavailable: {config_path}")
    config = _read_yaml(config_path)
    validate_e17_contract(config)
    shared = _read_yaml(_inside_project(config["data_config"], root))
    if set(shared.get("manifests", {})) != {"train", "validation"}:
        raise ValueError("E17 requires shared train/validation-only data config; test access is forbidden.")
    split, e15, e10 = config["split_contract"], config["e15_initialization"], config["e10_routing_input"]
    if _inside_project(shared["manifests"]["train"], root) != _inside_project(split["train_manifest"], root) or _inside_project(shared["manifests"]["validation"], root) != _inside_project(split["validation_manifest"], root):
        raise ValueError("E17 must exactly reuse the shared train/validation manifests.")
    config["resolved"] = {
        "project_root": str(root), "config_path": str(config_path),
        "train_manifest": str(_inside_project(split["train_manifest"], root)),
        "validation_manifest": str(_inside_project(split["validation_manifest"], root)),
        "e15_final": str(_inside_project(e15["final_checkpoint"], root)),
        "e15_predictions": str(_inside_project(e15["predictions"], root)),
        "e15_metrics": str(_inside_project(e15["metrics"], root)),
        "e10_predictions": str(_inside_project(e10["predictions"], root)),
        "output_dir": str(_inside_project(config["output_dir"], root)),
    }
    return config


def _frames(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    root, split = Path(config["resolved"]["project_root"]), config["split_contract"]
    full_train = _manifest(Path(config["resolved"]["train_manifest"]), root, "train", {"total": 4448, "NORMAL": 1148, "BACTERIAL_PNEUMONIA": 2153, "VIRAL_PNEUMONIA": 1147})
    train = _binary_train_frame(full_train, config)
    validation = _manifest(Path(config["resolved"]["validation_manifest"]), root, "validation", split["expected_counts"]["validation"])
    if set(train["group_key"]) & set(validation["group_key"]):
        raise RuntimeError("E17 train/validation group overlap detected.")
    return train, validation


def _device(request: str) -> torch.device:
    if request == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("E17 config requires CUDA but CUDA is unavailable.")
        return torch.device("cuda")
    if request == "cpu":
        return torch.device("cpu")
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unsupported E17 device: {request}")


def _e15_head_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {"model": {"expected_hidden_size": 768, "spatial_pool_grid": [4, 4]}, "head": dict(config["e15_head"])}


def _restore_e15(config: Mapping[str, Any], device: torch.device) -> tuple[Any, nn.Module, PatchAttentionHead, dict[str, Any]]:
    paths = {key: Path(config["resolved"][key]) for key in ("e15_final", "e15_predictions", "e15_metrics")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("E17 requires completed E15 final checkpoint/predictions/metrics: " + ", ".join(missing))
    processor, encoder, last_block, model_info = _load_encoder_and_last_block(config, device)
    payload = torch.load(paths["e15_final"], map_location="cpu", weights_only=True)
    required = {"experiment": "e15", "run_kind": "e15_last_block_finetune_fixed_final_epoch", "checkpoint_policy": "final_epoch_only"}
    if any(payload.get(key) != value for key, value in required.items()) or not isinstance(payload.get("last_block_state_dict"), Mapping) or not isinstance(payload.get("head_state_dict"), Mapping):
        raise ValueError("E17 E15 final checkpoint identity/state-dict contract mismatch.")
    saved_config = payload.get("config", {})
    if saved_config.get("head") != config["e15_head"] or saved_config.get("model", {}).get("hf_model_id") != config["model"]["hf_model_id"]:
        raise ValueError("E17 E15 checkpoint head/model configuration differs from the locked E17 restore contract.")
    last_block.load_state_dict(payload["last_block_state_dict"], strict=True)
    head = PatchAttentionHead(_e15_head_config(config)).to(device)
    head.load_state_dict(payload["head_state_dict"], strict=True)
    for module in (encoder, head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.eval()
    if encoder.training or head.training or any(parameter.requires_grad for parameter in encoder.parameters()) or any(parameter.requires_grad for parameter in head.parameters()):
        raise RuntimeError("E17 requires a fully frozen, eval-mode E15 encoder and head.")
    checkpoint_identity = {
        "path": str(paths["e15_final"]), "sha256": _sha256(paths["e15_final"]),
        "experiment": payload["experiment"], "run_kind": payload["run_kind"], "epoch": int(payload.get("epoch", -1)),
        "encoder_model_class": payload.get("encoder_model_class"),
    }
    model_info.update({
        "e15_last_block_restored_strict": True,
        "e15_head_restored_strict": True,
        "e15_checkpoint_identity": checkpoint_identity,
        "parameter_counts": {"encoder": _count_parameters(encoder), "e15_head": _count_parameters(head)},
    })
    return processor, encoder, head, model_info


def _pyramid_tokens(hidden: Tensor, config: Mapping[str, Any]) -> tuple[Tensor, dict[str, Any]]:
    hidden_size = int(config["model"]["expected_hidden_size"])
    if hidden.ndim != 3 or hidden.shape[-1] != hidden_size:
        raise RuntimeError(f"E17 expected RAD-DINO hidden [B,1370,768], got {tuple(hidden.shape)}")
    cls, patches = hidden[:, 0, :], hidden[:, 1:, :]
    side = math.isqrt(int(patches.shape[1]))
    expected_side = int(config["model"]["expected_image_size"]) // int(config["model"]["patch_size"])
    if side * side != int(patches.shape[1]) or side != expected_side or (side, side) != tuple(config["pyramid_pooling"]["patch_grid"]):
        raise RuntimeError(f"E17 patch-grid contract failed: patches={patches.shape[1]}, expected={expected_side}x{expected_side}")
    patch_map = patches.reshape(hidden.shape[0], side, side, hidden_size).permute(0, 3, 1, 2)
    parts: list[Tensor] = []
    offsets: list[dict[str, int]] = []
    start = 0
    for (height, width), count in zip(config["pyramid_pooling"]["scales"], config["pyramid_pooling"]["tokens_per_scale"], strict=True):
        pooled = F.adaptive_avg_pool2d(patch_map, (int(height), int(width))).permute(0, 2, 3, 1).reshape(hidden.shape[0], int(count), hidden_size)
        if pooled.shape != (hidden.shape[0], int(count), hidden_size):
            raise RuntimeError("E17 adaptive-pooling token shape contract failed.")
        parts.append(pooled)
        offsets.append({"height": int(height), "width": int(width), "start": start, "stop": start + int(count), "tokens": int(count)})
        start += int(count)
    spatial = torch.cat(parts, dim=1)
    if spatial.shape != (hidden.shape[0], int(config["pyramid_pooling"]["total_spatial_tokens"]), hidden_size) or not bool(torch.isfinite(spatial).all()):
        raise RuntimeError("E17 50-token pyramid output contract failed.")
    return spatial, {"last_hidden_state": list(hidden.shape), "cls": list(cls.shape), "patch_tokens": list(patches.shape), "patch_grid": [side, side], "patch_map": list(patch_map.shape), "scale_offsets": offsets, "token_order": config["pyramid_pooling"]["token_order"]}


def _e15_tokens_from_hidden(hidden: Tensor) -> Tensor:
    if hidden.ndim != 3 or hidden.shape[1:] != (1370, 768):
        raise RuntimeError(f"E17 E15 baseline expected [B,1370,768], got {tuple(hidden.shape)}")
    patch_map = hidden[:, 1:, :].reshape(hidden.shape[0], 37, 37, 768).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(patch_map, (4, 4)).permute(0, 2, 3, 1).reshape(hidden.shape[0], 16, 768)
    return torch.cat([hidden[:, :1, :], pooled], dim=1)


class ScaleAttentionPool(nn.Module):
    """Independent spatial attention pool to prevent scale descriptors collapsing to a shared mean."""
    def __init__(self, hidden_size: int, attention_hidden: int, token_count: int) -> None:
        super().__init__()
        self.hidden_size, self.token_count = hidden_size, token_count
        self.norm = nn.LayerNorm(hidden_size)
        self.score = nn.Sequential(nn.Linear(hidden_size, attention_hidden), nn.Tanh(), nn.Linear(attention_hidden, 1))

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 3 or tokens.shape[1:] != (self.token_count, self.hidden_size):
            raise RuntimeError(f"E17 scale pool expected [B,{self.token_count},{self.hidden_size}], got {tuple(tokens.shape)}")
        normalized = self.norm(tokens)
        attention = torch.softmax(self.score(normalized).squeeze(-1), dim=1)
        descriptor = torch.sum(normalized * attention.unsqueeze(-1), dim=1)
        if descriptor.shape != (tokens.shape[0], self.hidden_size) or attention.shape != (tokens.shape[0], self.token_count) or not torch.allclose(attention.sum(1), torch.ones(tokens.shape[0], device=tokens.device, dtype=attention.dtype), atol=1e-5):
            raise RuntimeError("E17 per-scale attention pooling contract failed.")
        return descriptor, attention


class MultiScaleAttentionHead(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden = int(config["model"]["expected_hidden_size"])
        attention_hidden = int(config["multiscale_head"]["attention_hidden_dimension"])
        counts = [int(value) for value in config["pyramid_pooling"]["tokens_per_scale"]]
        self.hidden, self.counts = hidden, counts
        self.pools = nn.ModuleList([ScaleAttentionPool(hidden, attention_hidden, count) for count in counts])
        self.classifier = nn.Sequential(
            nn.LayerNorm(int(config["multiscale_head"]["feature_dimension"])),
            nn.Linear(int(config["multiscale_head"]["feature_dimension"]), int(config["multiscale_head"]["classifier_hidden_dimension"])),
            nn.GELU(), nn.Dropout(float(config["multiscale_head"]["dropout"])),
            nn.Linear(int(config["multiscale_head"]["classifier_hidden_dimension"]), int(config["multiscale_head"]["num_classes"])),
        )

    def forward(self, cls_features: Tensor, spatial_tokens: Tensor) -> tuple[Tensor, list[Tensor]]:
        if cls_features.ndim != 2 or cls_features.shape[1] != self.hidden or spatial_tokens.ndim != 3 or spatial_tokens.shape != (cls_features.shape[0], sum(self.counts), self.hidden):
            raise RuntimeError(f"E17 head inputs invalid: cls={tuple(cls_features.shape)}, spatial={tuple(spatial_tokens.shape)}")
        descriptors = [cls_features]
        attention: list[Tensor] = []
        offset = 0
        for count, pool in zip(self.counts, self.pools, strict=True):
            descriptor, weights = pool(spatial_tokens[:, offset:offset + count])
            descriptors.append(descriptor)
            attention.append(weights)
            offset += count
        features = torch.cat(descriptors, dim=1)
        logits = self.classifier(features)
        if features.shape != (cls_features.shape[0], 5 * self.hidden) or logits.shape != (cls_features.shape[0], 2) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("E17 5x768 multi-scale classifier contract failed.")
        return logits, attention


def _count_parameters(module: nn.Module) -> dict[str, int]:
    values = list(module.parameters())
    return {"parameter_tensors": len(values), "parameters": int(sum(value.numel() for value in values)), "trainable_parameter_tensors": int(sum(value.requires_grad for value in values)), "trainable_parameters": int(sum(value.numel() for value in values if value.requires_grad))}


def static_smoke_check(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_e17_contract(config)
    train, validation = _frames(config)
    synthetic = torch.zeros((2, 1370, 768), dtype=torch.float32)
    spatial, shape_info = _pyramid_tokens(synthetic, config)
    head = MultiScaleAttentionHead(config).eval()
    with torch.inference_mode():
        logits, attention = head(synthetic[:, 0, :], spatial)
    if logits.shape != (2, 2) or [tuple(value.shape) for value in attention] != [(2, 1), (2, 4), (2, 9), (2, 36)]:
        raise RuntimeError("E17 CPU synthetic pyramid/head shape validation failed.")
    prerequisites = {key: Path(config["resolved"][key]).is_file() for key in ("e15_final", "e15_predictions", "e15_metrics", "e10_predictions")}
    return {
        "experiment": "e17", "static_check": "passed", "train_bv_samples": len(train), "validation_samples": len(validation),
        "train_validation_group_overlap": 0, "train_manifest_sha256": _sha256(Path(config["resolved"]["train_manifest"])),
        "validation_manifest_sha256": _sha256(Path(config["resolved"]["validation_manifest"])), "pyramid_shapes": shape_info,
        "attention_shapes": [list(value.shape) for value in attention], "e17_head_parameter_counts": _count_parameters(head),
        "e15_e10_prerequisites_available": prerequisites, "artifacts_written": False,
    }


def _processor_identity(processor: Any, model_info: Mapping[str, Any]) -> dict[str, Any]:
    size = getattr(processor, "size", None)
    return {"class": type(processor).__name__, "size": size, "image_mean": getattr(processor, "image_mean", None), "image_std": getattr(processor, "image_std", None), "requested_model_revision": model_info.get("requested_revision"), "resolved_model_commit_hash": model_info.get("resolved_commit_hash")}


def _cache_paths(config: Mapping[str, Any], split: str) -> dict[str, Path]:
    artifacts = config["artifacts"]
    cache_dir = Path(config["resolved"]["output_dir"]) / artifacts["cache_dir"]
    prefix = "train" if split == "train" else "validation"
    return {"features": cache_dir / artifacts[f"{prefix}_features"], "index": cache_dir / artifacts[f"{prefix}_index"], "metadata": cache_dir / artifacts[f"{prefix}_metadata"]}


def _cache_signature(config: Mapping[str, Any], frame: pd.DataFrame, manifest: Path, split: str, label_column: str, processor: Any, model_info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cache_version": _CACHE_VERSION, "split": split, "manifest_sha256": _sha256(manifest), "sample_count": len(frame),
        "sample_id_hash": _ordered_hash(frame["sample_id"]), "image_path_hash": _ordered_hash(frame["image_path"]),
        "label_hash": _ordered_hash(frame[label_column]), "label_column": label_column,
        "feature_shape": [len(frame), 50, 768], "cls_shape": [len(frame), 768], "feature_dtype": "float16",
        "model_checkpoint_identity": model_info["e15_checkpoint_identity"], "processor": _processor_identity(processor, model_info),
        "pooling_configuration": dict(config["pyramid_pooling"]),
    }


def _load_cache(config: Mapping[str, Any], split: str, frame: pd.DataFrame, manifest: Path, label_column: str, processor: Any, model_info: Mapping[str, Any]) -> PyramidFeatureSet | None:
    paths = _cache_paths(config, split)
    present = {key: path.is_file() for key, path in paths.items()}
    if not any(present.values()):
        return None
    if not all(present.values()):
        raise RuntimeError(f"E17 {split} cache is incomplete; reuse is blocked and existing cache files are not overwritten: {paths}")
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        index = pd.read_csv(paths["index"])
        with np.load(paths["features"], allow_pickle=False) as archive:
            spatial = np.asarray(archive["spatial_tokens"], dtype=np.float16)
            cls = np.asarray(archive["cls_features"], dtype=np.float16)
            labels = np.asarray(archive["labels"], dtype=np.int64)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError(f"E17 {split} cache cannot be parsed; reuse is blocked.") from error
    expected = _cache_signature(config, frame, manifest, split, label_column, processor, model_info)
    required_columns = ["sample_id", "image_path", "group_key", "source_identifier", "class_name", "canonical_label", "binary_label", "cache_label"]
    valid = metadata == expected and list(index.columns) == required_columns and index["sample_id"].tolist() == frame["sample_id"].tolist() and index["image_path"].tolist() == frame["image_path"].tolist() and index["cache_label"].astype(int).tolist() == frame[label_column].astype(int).tolist() and spatial.shape == (len(frame), 50, 768) and cls.shape == (len(frame), 768) and labels.shape == (len(frame),) and np.array_equal(labels, frame[label_column].to_numpy(dtype=np.int64)) and np.isfinite(spatial).all() and np.isfinite(cls).all()
    if not valid:
        raise RuntimeError(f"E17 {split} cache integrity mismatch; reuse is blocked and no cache files were overwritten.")
    return PyramidFeatureSet(split, spatial, cls, labels, index, True)


def _save_cache(config: Mapping[str, Any], values: PyramidFeatureSet, manifest: Path, label_column: str, processor: Any, model_info: Mapping[str, Any]) -> None:
    paths = _cache_paths(config, values.split)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"E17 {values.split} cache destination exists; automatic overwrite is forbidden.")
    frame = values.index.copy()
    index = frame[["sample_id", "image_path", "group_key", "source_identifier", "class_name", "canonical_label", "binary_label"]].copy()
    index["cache_label"] = frame[label_column].astype(int)
    _atomic_npz(paths["features"], spatial_tokens=values.spatial_tokens, cls_features=values.cls_features, labels=values.labels)
    _atomic_csv(paths["index"], index)
    _atomic_json(paths["metadata"], _cache_signature(config, frame, manifest, values.split, label_column, processor, model_info))


def _is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def _extract_features(config: Mapping[str, Any], split: str, frame: pd.DataFrame, label_column: str, processor: Any, encoder: nn.Module, device: torch.device) -> tuple[PyramidFeatureSet, dict[str, Any]]:
    if device.type == "cuda" and device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    loader = DataLoader(RawManifestDataset(frame, Path(config["resolved"]["project_root"]), label_column), batch_size=int(config["runtime"]["feature_batch_size"]), shuffle=False, num_workers=int(config["runtime"]["num_workers"]), pin_memory=bool(config["runtime"]["pin_memory"]) and device.type == "cuda", collate_fn=_collate)
    spatial_parts: list[np.ndarray] = []
    cls_parts: list[np.ndarray] = []
    seen: list[str] = []
    shape_info: dict[str, Any] | None = None
    amp = bool(config["runtime"]["amp"]) and device.type == "cuda"
    for batch_index, (images, labels, identifiers) in enumerate(loader):
        try:
            inputs = {key: value.to(device, non_blocking=device.type == "cuda") for key, value in processor(images=images, return_tensors="pt").items()}
            pixel_values = inputs.get("pixel_values")
            if pixel_values is None or pixel_values.device != device or not pixel_values.is_floating_point() or pixel_values.dtype != torch.float32:
                raise RuntimeError(f"E17 processor input dtype/device contract failed: {None if pixel_values is None else (pixel_values.dtype, pixel_values.device)}")
            if next(encoder.parameters()).device != device or next(encoder.parameters()).dtype != torch.float32:
                raise RuntimeError("E17 encoder parameter dtype/device contract failed before extraction.")
            autocast = torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
            with torch.inference_mode(), autocast:
                hidden = encoder(**inputs).last_hidden_state
                spatial, current_info = _pyramid_tokens(hidden, config)
                cls = hidden[:, 0, :]
            if hidden.device != device or cls.device != device or spatial.device != device or not bool(torch.isfinite(cls).all() and torch.isfinite(spatial).all()):
                raise FloatingPointError(f"E17 invalid device/non-finite feature at {split} batch={batch_index}.")
            spatial_parts.append(spatial.float().cpu().to(torch.float16).numpy())
            cls_parts.append(cls.float().cpu().to(torch.float16).numpy())
            seen.extend(identifiers)
            expected_labels = frame.iloc[len(seen) - len(labels):len(seen)][label_column].astype(int).tolist()
            if labels.tolist() != expected_labels:
                raise RuntimeError(f"E17 {split} image/label ordering mismatch at batch={batch_index}.")
            shape_info = current_info
        except RuntimeError as error:
            if _is_oom(error):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(f"CUDA OOM during E17 {split} feature extraction at batch={batch_index}, feature_batch_size={config['runtime']['feature_batch_size']}. No cache was written; reduce feature_batch_size and rerun.") from error
            raise
    spatial = np.concatenate(spatial_parts, axis=0).astype(np.float16, copy=False)
    cls = np.concatenate(cls_parts, axis=0).astype(np.float16, copy=False)
    if seen != frame["sample_id"].tolist() or spatial.shape != (len(frame), 50, 768) or cls.shape != (len(frame), 768) or not np.isfinite(spatial).all() or not np.isfinite(cls).all() or shape_info is None:
        raise RuntimeError(f"E17 {split} feature extraction contract failed.")
    return PyramidFeatureSet(split, spatial, cls, frame[label_column].to_numpy(dtype=np.int64), frame.copy(), False), shape_info


def _features(config: Mapping[str, Any], split: str, frame: pd.DataFrame, manifest: Path, label_column: str, processor: Any, encoder: nn.Module, device: torch.device, model_info: Mapping[str, Any], *, cache: bool) -> tuple[PyramidFeatureSet, dict[str, Any]]:
    if cache:
        cached = _load_cache(config, split, frame, manifest, label_column, processor, model_info)
        if cached is not None:
            return cached, {"source": "cache"}
    values, shape_info = _extract_features(config, split, frame, label_column, processor, encoder, device)
    if cache:
        _save_cache(config, values, manifest, label_column, processor, model_info)
    return values, {"source": "fresh", "pyramid_shapes": shape_info}


def _e15_baseline(config: Mapping[str, Any], validation: pd.DataFrame, processor: Any, encoder: nn.Module, head: PatchAttentionHead, device: torch.device) -> dict[str, Any]:
    if device.type == "cuda" and device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    saved = pd.read_csv(Path(config["resolved"]["e15_predictions"]))
    required = {"image_path", "e10_e6_pred", "e15_binary_pred", "e15_prob_bacterial", "e15_prob_viral", "e15_final_pred"}
    if required - set(saved.columns) or saved["image_path"].duplicated().any():
        raise ValueError("E17 requires E15 validation predictions with the locked baseline columns.")
    saved = saved.set_index("image_path").reindex(validation["image_path"]).reset_index()
    if saved.isna().any().any():
        raise ValueError("E17 E15 predictions do not exactly cover the current validation manifest.")
    loader = DataLoader(RawManifestDataset(validation.assign(binary_label=validation["label"].map({1: 0, 2: 1}).fillna(0).astype(int)), Path(config["resolved"]["project_root"]), "binary_label"), batch_size=int(config["runtime"]["feature_batch_size"]), shuffle=False, num_workers=0, pin_memory=bool(config["runtime"]["pin_memory"]) and device.type == "cuda", collate_fn=_collate)
    probabilities: list[np.ndarray] = []
    seen: list[str] = []
    amp = bool(config["runtime"]["amp"]) and device.type == "cuda"
    for images, _, identifiers in loader:
        inputs = {key: value.to(device, non_blocking=device.type == "cuda") for key, value in processor(images=images, return_tensors="pt").items()}
        autocast = torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
        with torch.inference_mode(), autocast:
            hidden = encoder(**inputs).last_hidden_state
            logits, _ = head(_e15_tokens_from_hidden(hidden))
            batch_probabilities = torch.softmax(logits, dim=1)
        if logits.device != device or batch_probabilities.shape != (len(images), 2) or not bool(torch.isfinite(batch_probabilities).all()):
            raise RuntimeError("E17 E15 baseline device/logit contract failed.")
        probabilities.append(batch_probabilities.float().cpu().numpy())
        seen.extend(identifiers)
    probs = np.concatenate(probabilities, axis=0)
    prediction = probs.argmax(1)
    saved_probs = saved[["e15_prob_bacterial", "e15_prob_viral"]].to_numpy(dtype=np.float64)
    e6 = saved["e10_e6_pred"].astype(int).to_numpy()
    final = np.where(e6 == 0, 0, prediction + 1).astype(np.int64)
    probability_error = float(np.abs(probs - saved_probs).max())
    if seen != validation["sample_id"].tolist() or not np.array_equal(prediction, saved["e15_binary_pred"].astype(int).to_numpy()) or not np.allclose(probs, saved_probs, rtol=0.0, atol=float(config["e15_initialization"]["global_baseline_probability_atol"])) or not np.array_equal(final, saved["e15_final_pred"].astype(int).to_numpy()):
        raise RuntimeError("E15 global baseline reproduction failed (binary prediction, probability tolerance, or E10 routing mismatch); E17 training is blocked.")
    return {"e15_binary_prediction_exact": True, "e15_probability_rtol": 0.0, "e15_probability_atol": float(config["e15_initialization"]["global_baseline_probability_atol"]), "max_probability_absolute_error": probability_error, "e15_final_routing_exact": True, "e15_final_predictions": final, "e15_binary_predictions": prediction}


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_head(config: Mapping[str, Any], train: PyramidFeatureSet, device: torch.device) -> tuple[MultiScaleAttentionHead, list[dict[str, Any]]]:
    if device.type == "cuda" and device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    _seed(int(config["training"]["seed"]))
    head = MultiScaleAttentionHead(config).to(device)
    first_parameter = next(head.parameters())
    if first_parameter.device != device or first_parameter.dtype != torch.float32:
        raise RuntimeError("E17 trainable head dtype/device contract failed before training.")
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(train.cls_features.astype(np.float32)), torch.from_numpy(train.spatial_tokens.astype(np.float32)), torch.from_numpy(train.labels)), batch_size=int(config["runtime"]["head_batch_size"]), shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config["training"]["fixed_epochs"]) + 1):
        head.train()
        total_loss, total = 0.0, 0
        targets: list[int] = []
        predictions: list[int] = []
        for cls, spatial, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            cls = cls.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            spatial = spatial.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            labels = labels.to(device=device, dtype=torch.long, non_blocking=device.type == "cuda")
            logits, _ = head(cls, spatial)
            loss = criterion(logits, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"E17 non-finite training loss at epoch={epoch}.")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(labels)
            total += len(labels)
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(1).cpu().tolist())
        metrics = _binary_metrics(np.asarray(targets), np.asarray(predictions))
        history.append({"epoch": epoch, "train_loss": total_loss / total, "train_macro_f1": metrics["macro_f1"], "train_accuracy": metrics["accuracy"], "train_confusion_matrix": metrics["confusion_matrix"], "train_bacterial_to_viral": metrics["bacterial_to_viral"], "train_viral_to_bacterial": metrics["viral_to_bacterial"], "checkpoint_selection": "none_fixed_final_epoch"})
    return head.eval(), history


def _prepare_output(config: Mapping[str, Any]) -> dict[str, Path]:
    output, artifacts = Path(config["resolved"]["output_dir"]), config["artifacts"]
    paths = {"output": output, "final": output / artifacts["final_checkpoint"], "history": output / artifacts["history"], "metrics": output / artifacts["metrics"], "predictions": output / artifacts["predictions"], "metadata": output / artifacts["run_metadata"]}
    existing = [path for key, path in paths.items() if key != "output" and path.exists()]
    if existing:
        raise FileExistsError("E17 final results already exist; automatic overwrite is forbidden: " + ", ".join(map(str, existing)))
    output.mkdir(parents=True, exist_ok=True)
    return paths


def _full_metrics(targets: np.ndarray, final_predictions: np.ndarray) -> dict[str, Any]:
    metrics = classification_metrics(targets, np.eye(3, dtype=np.float64)[final_predictions])
    metrics["macro_auroc_ovr"] = None
    for values in metrics["per_class"].values():
        values["auroc_ovr"] = None
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    metrics.update({"bacterial_to_viral": int(matrix[1, 2]), "viral_to_bacterial": int(matrix[2, 1]), "bacterial_viral_misclassifications": int(matrix[1, 2] + matrix[2, 1]), "auroc_not_probability_comparable": True})
    return metrics


def _validation(config: Mapping[str, Any], validation: pd.DataFrame, trace: pd.DataFrame, features: PyramidFeatureSet, head: MultiScaleAttentionHead, device: torch.device, baseline: Mapping[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    cls = torch.from_numpy(features.cls_features.astype(np.float32)).to(device=device, dtype=torch.float32)
    spatial = torch.from_numpy(features.spatial_tokens.astype(np.float32)).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        probabilities = torch.softmax(head(cls, spatial)[0], dim=1).cpu().numpy()
    binary_predictions = probabilities.argmax(1)
    targets = validation["label"].to_numpy(dtype=np.int64)
    e6 = trace["e6_pred"].astype(int).to_numpy()
    final_predictions = np.where(e6 == 0, 0, binary_predictions + 1).astype(np.int64)
    e15_final = np.asarray(baseline["e15_final_predictions"], dtype=np.int64)
    e17_metrics, e15_metrics = _full_metrics(targets, final_predictions), _full_metrics(targets, e15_final)
    bv = targets >= 1
    e17_bv = _binary_metrics(targets[bv] - 1, binary_predictions[bv], probabilities[bv])
    e15_bv = _binary_metrics(targets[bv] - 1, np.asarray(baseline["e15_binary_predictions"], dtype=np.int64)[bv])
    e15_correct, e17_correct = e15_final == targets, final_predictions == targets
    comparison = {
        "e15": {"macro_f1": e15_metrics["macro_f1"], "accuracy": e15_metrics["accuracy"], "bacterial_f1": e15_metrics["per_class"]["BACTERIAL_PNEUMONIA"]["f1"], "viral_f1": e15_metrics["per_class"]["VIRAL_PNEUMONIA"]["f1"], "viral_recall": e15_metrics["per_class"]["VIRAL_PNEUMONIA"]["recall"], "bv_confusion_matrix": e15_bv["confusion_matrix"], "bacterial_to_viral": e15_bv["bacterial_to_viral"], "viral_to_bacterial": e15_bv["viral_to_bacterial"]},
        "e17": {"macro_f1": e17_metrics["macro_f1"], "accuracy": e17_metrics["accuracy"], "bacterial_f1": e17_metrics["per_class"]["BACTERIAL_PNEUMONIA"]["f1"], "viral_f1": e17_metrics["per_class"]["VIRAL_PNEUMONIA"]["f1"], "viral_recall": e17_metrics["per_class"]["VIRAL_PNEUMONIA"]["recall"], "bv_confusion_matrix": e17_bv["confusion_matrix"], "bacterial_to_viral": e17_bv["bacterial_to_viral"], "viral_to_bacterial": e17_bv["viral_to_bacterial"]},
        "delta_e17_minus_e15": {"macro_f1": float(e17_metrics["macro_f1"] - e15_metrics["macro_f1"]), "accuracy": float(e17_metrics["accuracy"] - e15_metrics["accuracy"]), "bacterial_f1": float(e17_metrics["per_class"]["BACTERIAL_PNEUMONIA"]["f1"] - e15_metrics["per_class"]["BACTERIAL_PNEUMONIA"]["f1"]), "viral_f1": float(e17_metrics["per_class"]["VIRAL_PNEUMONIA"]["f1"] - e15_metrics["per_class"]["VIRAL_PNEUMONIA"]["f1"]), "viral_recall": float(e17_metrics["per_class"]["VIRAL_PNEUMONIA"]["recall"] - e15_metrics["per_class"]["VIRAL_PNEUMONIA"]["recall"])},
        "e15_only_correct_count": int((e15_correct & ~e17_correct).sum()), "e17_only_correct_count": int((e17_correct & ~e15_correct).sum()),
    }
    predictions = validation[["sample_id", "image_path", "group_key", "source_identifier", "label", "class_name"]].copy().rename(columns={"label": "true_label", "class_name": "true_class_name"})
    predictions["e10_e6_pred"] = e6
    predictions["e10_final_pred"] = trace["final_pred"].astype(int).to_numpy()
    predictions["e15_final_pred"] = e15_final
    predictions["e17_binary_pred"] = binary_predictions
    predictions["e17_prob_bacterial"], predictions["e17_prob_viral"] = probabilities[:, 0], probabilities[:, 1]
    predictions["e17_final_pred"] = final_predictions
    predictions["e17_final_pred_name"] = [CANONICAL_CLASSES[value] for value in final_predictions]
    predictions["e10_normal_kept"] = e6 == 0
    metrics = {"experiment": "e17", "e15_baseline_reproduction": {key: value for key, value in baseline.items() if not isinstance(value, np.ndarray)}, "e17_binary_specialist_on_true_bv_validation": e17_bv, "e17_normal_preserving_full_validation": e17_metrics, "e15_vs_e17": comparison, "validation_used_for_model_selection": False}
    return metrics, predictions


def gpu_smoke_check(config: Mapping[str, Any], *, device_request: str | None = None) -> dict[str, Any]:
    static = static_smoke_check(config)
    train, _ = _frames(config)
    selected = pd.concat([train[train["binary_label"] == label].head(int(config["smoke_test"]["samples_per_class"])) for label in (0, 1)], ignore_index=True)
    device = _device(device_request or str(config["runtime"]["device"]))
    processor, encoder, e15_head, model_info = _restore_e15(config, device)
    try:
        features, extraction = _features(config, "gpu_smoke", selected, Path(config["resolved"]["train_manifest"]), "binary_label", processor, encoder, device, model_info, cache=False)
        head = MultiScaleAttentionHead(config).to(device).train()
        cls = torch.from_numpy(features.cls_features.astype(np.float32)).to(device=device, dtype=torch.float32)
        spatial = torch.from_numpy(features.spatial_tokens.astype(np.float32)).to(device=device, dtype=torch.float32)
        labels = torch.from_numpy(features.labels).to(device=device, dtype=torch.long)
        logits, attention = head(cls, spatial)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        if logits.shape != (len(selected), 2) or [tuple(weights.shape) for weights in attention] != [(len(selected), 1), (len(selected), 4), (len(selected), 9), (len(selected), 36)] or any(parameter.grad is not None for parameter in encoder.parameters()) or any(parameter.grad is not None for parameter in e15_head.parameters()):
            raise RuntimeError("E17 GPU smoke frozen-model, pyramid, or classifier-gradient contract failed.")
        memory = {"allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 2), "reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024**2, 2)} if device.type == "cuda" else None
        return {**static, "gpu_smoke": "passed", "device": str(device), "samples": len(selected), "loss": float(loss.detach().cpu()), "extraction": extraction, "attention_shapes": [list(value.shape) for value in attention], "parameter_counts": {"encoder": _count_parameters(encoder), "e15_head": _count_parameters(e15_head), "e17_head": _count_parameters(head)}, "gpu_memory": memory, "artifacts_written": False}
    finally:
        del encoder, e15_head
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_e17(config: Mapping[str, Any]) -> dict[str, Any]:
    static_smoke_check(config)
    train, validation = _frames(config)
    trace = _load_e10_trace(config, validation)
    device = _device(str(config["runtime"]["device"]))
    processor, encoder, e15_head, model_info = _restore_e15(config, device)
    baseline = _e15_baseline(config, validation, processor, encoder, e15_head, device)
    paths = _prepare_output(config)
    cache = bool(config["runtime"]["cache_features"])
    train_features, train_extraction = _features(config, "train", train, Path(config["resolved"]["train_manifest"]), "binary_label", processor, encoder, device, model_info, cache=cache)
    validation_features, validation_extraction = _features(config, "validation", validation, Path(config["resolved"]["validation_manifest"]), "label", processor, encoder, device, model_info, cache=cache)
    del encoder, e15_head
    if device.type == "cuda":
        torch.cuda.empty_cache()
    head, history = _train_head(config, train_features, device)
    counts = {"frozen_e15": model_info["parameter_counts"], "e17_head": _count_parameters(head)}
    metrics, predictions = _validation(config, validation, trace, validation_features, head, device, baseline)
    _atomic_torch(paths["final"], {"experiment": "e17", "run_kind": "e17_frozen_e15_pyramid_pooling_fixed_final_epoch", "epoch": int(config["training"]["fixed_epochs"]), "checkpoint_policy": "final_epoch_only", "multiscale_head_state_dict": head.state_dict(), "config": config, "e15_restore": model_info, "parameter_counts": counts})
    _atomic_json(paths["history"], history)
    _atomic_json(paths["metrics"], metrics)
    _atomic_csv(paths["predictions"], predictions)
    _atomic_json(paths["metadata"], {"experiment": "e17", "encoder": model_info, "e15_head_frozen": True, "encoder_frozen": True, "trainable_modules": ["per_scale_attention_pooling", "multiscale_classifier"], "feature_cache": {"train": {"source": train_extraction["source"], "shape": list(train_features.spatial_tokens.shape), "cls_shape": list(train_features.cls_features.shape)}, "validation": {"source": validation_extraction["source"], "shape": list(validation_features.spatial_tokens.shape), "cls_shape": list(validation_features.cls_features.shape)}}, "training_policy": "B/V train only; fixed five epochs; final epoch checkpoint only; no holdout, early stopping, threshold tuning, validation selection, or hyperparameter search", "validation_used_for_model_selection": False})
    return {"output_dir": str(paths["output"]), "final_epoch": int(config["training"]["fixed_epochs"]), "e17_macro_f1": metrics["e17_normal_preserving_full_validation"]["macro_f1"], "e15_macro_f1": metrics["e15_vs_e17"]["e15"]["macro_f1"], "result_files": {key: str(value) for key, value in paths.items() if key != "output"}}
