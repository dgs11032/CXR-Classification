"""E18: frozen E15 RAD-DINO plus frozen ImageNet ResNet18 residual fusion.

The learned head predicts a residual added to frozen E15 binary logits. Its final
residual layer is initialized to zero, so initial logits reproduce E15 exactly.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .e14_patch_specialist import (
    BINARY_CLASSES, CANONICAL_CLASSES, RawManifestDataset, _binary_metrics,
    _binary_train_frame, _collate, _load_e10_trace, _manifest,
)
from .e17_pyramid_pooling import (
    _count_parameters, _device, _e15_baseline, _e15_tokens_from_hidden,
    _frames, _restore_e15,
)
from .metrics import classification_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_VERSION = 1


@dataclass(frozen=True)
class FusionFeatureSet:
    split: str
    e15_cls: np.ndarray
    e15_logits: np.ndarray
    cnn_features: np.ndarray
    labels: np.ndarray
    index: pd.DataFrame
    from_cache: bool


def _inside(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"E18 path leaves project root: {value}")
    return resolved


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"E18 YAML must be a mapping: {path}")
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


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def _npz(path: Path, values: FusionFeatureSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temp, e15_cls=values.e15_cls, e15_logits=values.e15_logits, cnn_features=values.cnn_features, labels=values.labels)
    os.replace(temp, path)


def _torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def validate_e18_contract(config: Mapping[str, Any]) -> None:
    required = {"experiment": "e18", "status": "frozen_e15_rad_dino_resnet18_residual_feature_fusion_bv_classifier", "data_config": "configs/data_config.yaml", "output_dir": "artifacts/experiments/e18"}
    errors = [{"key": key, "expected": value, "actual": config.get(key)} for key, value in required.items() if config.get(key) != value]
    model = config.get("model", {})
    expected_model = {"hf_model_id": "microsoft/rad-dino", "revision": "main", "trust_remote_code": False, "required_transformers_version": "4.57.6", "expected_hidden_size": 768, "expected_num_hidden_layers": 12, "patch_size": 14, "expected_image_size": 518, "e15_restore": "base_rad_dino_plus_strict_final_last_block_and_patch_attention_head", "encoder_trainable": False, "e15_head_trainable": False}
    errors.extend({"key": f"model.{key}", "expected": value, "actual": model.get(key)} for key, value in expected_model.items() if model.get(key) != value)
    e15_head = config.get("e15_head", {})
    expected_e15 = {"architecture": "cls_plus_4x4_patch_attention_pooling_binary_head", "attention_hidden_dimension": 128, "classifier_hidden_dimension": 256, "dropout": 0.1, "num_classes": 2}
    errors.extend({"key": f"e15_head.{key}", "expected": value, "actual": e15_head.get(key)} for key, value in expected_e15.items() if e15_head.get(key) != value)
    cnn = config.get("cnn", {})
    expected_cnn = {"selection": "torchvision_resnet18_imagenet1k_v1_no_valid_existing_cnn_checkpoint", "architecture": "resnet18", "weights": "IMAGENET1K_V1", "pooled_feature_dimension": 512, "preprocessing": "ResNet18_Weights.IMAGENET1K_V1.transforms", "trainable": False, "eval_mode": True}
    errors.extend({"key": f"cnn.{key}", "expected": value, "actual": cnn.get(key)} for key, value in expected_cnn.items() if cnn.get(key) != value)
    if cnn.get("rejected_existing_checkpoints") != ["artifacts/experiments/e0_dryrun/dryrun_checkpoint.pt", "artifacts/e0/smoke/best.pt"]:
        errors.append({"key": "cnn.rejected_existing_checkpoints", "actual": cnn.get("rejected_existing_checkpoints")})
    fusion = config.get("fusion_head", {})
    expected_fusion = {"architecture": "e15_logits_plus_zero_initialized_residual_mlp", "e15_cls_dimension": 768, "e15_binary_logit_dimension": 2, "cnn_pooled_dimension": 512, "fusion_input_dimension": 1282, "hidden_dimension": 128, "dropout": 0.2, "residual_output_dimension": 2, "residual_final_layer_initialization": "zero_weight_and_bias"}
    errors.extend({"key": f"fusion_head.{key}", "expected": value, "actual": fusion.get(key)} for key, value in expected_fusion.items() if fusion.get(key) != value)
    runtime, training, split = config.get("runtime", {}), config.get("training", {}), config.get("split_contract", {})
    expected_runtime = {"device": "cuda", "feature_batch_size": 1, "fusion_batch_size": 64, "num_workers": 0, "pin_memory": True, "amp": True, "cache_features": True}
    errors.extend({"key": f"runtime.{key}", "expected": value, "actual": runtime.get(key)} for key, value in expected_runtime.items() if runtime.get(key) != value)
    expected_training = {"seed": 42, "fixed_epochs": 3, "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001, "checkpoint_policy": "final_epoch_only", "internal_holdout": "disabled", "early_stopping": False, "validation_weight_selection": False, "validation_threshold_selection": False, "hyperparameter_search": False}
    errors.extend({"key": f"training.{key}", "expected": value, "actual": training.get(key)} for key, value in expected_training.items() if training.get(key) != value)
    if split.get("binary_class_order") != list(BINARY_CLASSES) or split.get("canonical_to_binary") != {1: 0, 2: 1} or bool(split.get("no_test_access")) is not True:
        errors.append({"key": "split_contract", "actual": split})
    if config.get("e10_routing_input", {}).get("normal_rule") != "e6_pred_zero_kept_else_e18_binary_plus_one":
        errors.append({"key": "routing", "actual": config.get("e10_routing_input", {})})
    if float(config.get("e15_initialization", {}).get("global_baseline_probability_atol", -1)) != 0.0005:
        errors.append({"key": "e15 baseline tolerance", "actual": config.get("e15_initialization", {})})
    if errors:
        raise ValueError(f"E18 configuration contract drift: {errors}")


def load_e18_config(path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config_path = _inside(path, root)
    if not config_path.is_file():
        raise FileNotFoundError(f"E18 config unavailable: {config_path}")
    config = _yaml(config_path)
    validate_e18_contract(config)
    data = _yaml(_inside(config["data_config"], root))
    if set(data.get("manifests", {})) != {"train", "validation"}:
        raise ValueError("E18 requires train/validation-only shared data config; test access is forbidden.")
    split, e15, e10 = config["split_contract"], config["e15_initialization"], config["e10_routing_input"]
    if _inside(data["manifests"]["train"], root) != _inside(split["train_manifest"], root) or _inside(data["manifests"]["validation"], root) != _inside(split["validation_manifest"], root):
        raise ValueError("E18 must reuse the locked shared manifests exactly.")
    config["resolved"] = {"project_root": str(root), "config_path": str(config_path), "train_manifest": str(_inside(split["train_manifest"], root)), "validation_manifest": str(_inside(split["validation_manifest"], root)), "e15_final": str(_inside(e15["final_checkpoint"], root)), "e15_predictions": str(_inside(e15["predictions"], root)), "e15_metrics": str(_inside(e15["metrics"], root)), "e10_predictions": str(_inside(e10["predictions"], root)), "output_dir": str(_inside(config["output_dir"], root))}
    return config


def _load_cnn(config: Mapping[str, Any], device: torch.device, *, pretrained: bool) -> tuple[nn.Module, Any, dict[str, Any]]:
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as error:
        raise RuntimeError("E18 requires torchvision ResNet18 in the runtime.") from error
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    cnn = resnet18(weights=weights)
    cnn.fc = nn.Identity()
    cnn.to(device)
    for parameter in cnn.parameters():
        parameter.requires_grad_(False)
    cnn.eval()
    if cnn.training or any(parameter.requires_grad for parameter in cnn.parameters()):
        raise RuntimeError("E18 requires a frozen eval-mode CNN.")
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    return cnn, transform, {"selection": config["cnn"]["selection"], "architecture": "torchvision.models.resnet18", "weights": "IMAGENET1K_V1" if pretrained else "NONE_STATIC_SHAPE_ONLY", "pooled_feature_dimension": 512, "preprocessing": config["cnn"]["preprocessing"], "parameter_counts": _count_parameters(cnn)}


class ResidualFusionHead(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        fusion = config["fusion_head"]
        dimension, hidden = int(fusion["fusion_input_dimension"]), int(fusion["hidden_dimension"])
        self.residual = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, hidden), nn.GELU(), nn.Dropout(float(fusion["dropout"])), nn.Linear(hidden, int(fusion["residual_output_dimension"])))
        final = self.residual[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, e15_cls: Tensor, e15_logits: Tensor, cnn_features: Tensor) -> tuple[Tensor, Tensor]:
        if e15_cls.ndim != 2 or e15_cls.shape[1] != 768 or e15_logits.shape != (e15_cls.shape[0], 2) or cnn_features.shape != (e15_cls.shape[0], 512):
            raise RuntimeError(f"E18 fusion input shapes invalid: cls={tuple(e15_cls.shape)}, logits={tuple(e15_logits.shape)}, cnn={tuple(cnn_features.shape)}")
        correction = self.residual(torch.cat([e15_cls, e15_logits, cnn_features], dim=1))
        logits = e15_logits + correction
        if correction.shape != e15_logits.shape or logits.shape != e15_logits.shape or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("E18 residual fusion output contract failed.")
        return logits, correction


def _cache_paths(config: Mapping[str, Any], split: str) -> dict[str, Path]:
    cache = Path(config["resolved"]["output_dir"]) / config["artifacts"]["cache_dir"]
    prefix = "train" if split == "train" else "validation"
    return {"features": cache / config["artifacts"][f"{prefix}_features"], "index": cache / config["artifacts"][f"{prefix}_index"], "metadata": cache / config["artifacts"][f"{prefix}_metadata"]}


def _signature(config: Mapping[str, Any], frame: pd.DataFrame, manifest: Path, split: str, label_column: str, e15_info: Mapping[str, Any], cnn_info: Mapping[str, Any]) -> dict[str, Any]:
    return {"cache_version": _CACHE_VERSION, "split": split, "manifest_sha256": _sha256(manifest), "sample_count": len(frame), "sample_id_hash": _ordered_hash(frame["sample_id"]), "image_path_hash": _ordered_hash(frame["image_path"]), "label_hash": _ordered_hash(frame[label_column]), "label_column": label_column, "e15_cls_shape": [len(frame), 768], "e15_logits_shape": [len(frame), 2], "cnn_features_shape": [len(frame), 512], "e15_cls_dtype": "float16", "e15_logits_dtype": "float32", "cnn_features_dtype": "float16", "e15_checkpoint_identity": e15_info["e15_checkpoint_identity"], "cnn_identity": dict(cnn_info), "fusion_configuration": dict(config["fusion_head"])}


def _load_cache(config: Mapping[str, Any], split: str, frame: pd.DataFrame, manifest: Path, label_column: str, e15_info: Mapping[str, Any], cnn_info: Mapping[str, Any]) -> FusionFeatureSet | None:
    paths = _cache_paths(config, split)
    present = {name: path.is_file() for name, path in paths.items()}
    if not any(present.values()):
        return None
    if not all(present.values()):
        raise RuntimeError(f"E18 {split} cache is incomplete; reuse is blocked without overwriting existing files.")
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        index = pd.read_csv(paths["index"])
        with np.load(paths["features"], allow_pickle=False) as archive:
            e15_cls = np.asarray(archive["e15_cls"], dtype=np.float16)
            e15_logits = np.asarray(archive["e15_logits"], dtype=np.float32)
            cnn_features = np.asarray(archive["cnn_features"], dtype=np.float16)
            labels = np.asarray(archive["labels"], dtype=np.int64)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError(f"E18 {split} cache cannot be parsed; reuse is blocked.") from error
    expected_columns = ["sample_id", "image_path", "group_key", "source_identifier", "class_name", "canonical_label", "binary_label", "cache_label"]
    valid = metadata == _signature(config, frame, manifest, split, label_column, e15_info, cnn_info) and list(index.columns) == expected_columns and index["sample_id"].tolist() == frame["sample_id"].tolist() and index["image_path"].tolist() == frame["image_path"].tolist() and index["cache_label"].astype(int).tolist() == frame[label_column].astype(int).tolist() and e15_cls.shape == (len(frame), 768) and e15_logits.shape == (len(frame), 2) and cnn_features.shape == (len(frame), 512) and labels.shape == (len(frame),) and np.array_equal(labels, frame[label_column].to_numpy(dtype=np.int64)) and np.isfinite(e15_cls).all() and np.isfinite(e15_logits).all() and np.isfinite(cnn_features).all()
    if not valid:
        raise RuntimeError(f"E18 {split} cache integrity mismatch; reuse is blocked and cache is not overwritten.")
    return FusionFeatureSet(split, e15_cls, e15_logits, cnn_features, labels, index, True)


def _save_cache(config: Mapping[str, Any], values: FusionFeatureSet, manifest: Path, label_column: str, e15_info: Mapping[str, Any], cnn_info: Mapping[str, Any]) -> None:
    paths = _cache_paths(config, values.split)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"E18 {values.split} cache exists; automatic overwrite is forbidden.")
    frame = values.index
    index = frame[["sample_id", "image_path", "group_key", "source_identifier", "class_name", "canonical_label", "binary_label"]].copy()
    index["cache_label"] = frame[label_column].astype(int)
    _npz(paths["features"], values)
    _csv(paths["index"], index)
    _json(paths["metadata"], _signature(config, frame, manifest, values.split, label_column, e15_info, cnn_info))


def _oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def _extract(config: Mapping[str, Any], split: str, frame: pd.DataFrame, label_column: str, processor: Any, encoder: nn.Module, e15_head: nn.Module, cnn: nn.Module, cnn_transform: Any, device: torch.device) -> FusionFeatureSet:
    if device.type == "cuda" and device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    root = Path(config["resolved"]["project_root"])
    loader = DataLoader(RawManifestDataset(frame, root, label_column), batch_size=int(config["runtime"]["feature_batch_size"]), shuffle=False, num_workers=int(config["runtime"]["num_workers"]), pin_memory=bool(config["runtime"]["pin_memory"]) and device.type == "cuda", collate_fn=_collate)
    cls_parts: list[np.ndarray] = []
    logits_parts: list[np.ndarray] = []
    cnn_parts: list[np.ndarray] = []
    seen: list[str] = []
    amp = bool(config["runtime"]["amp"]) and device.type == "cuda"
    for batch_index, (images, labels, identifiers) in enumerate(loader):
        try:
            rad_inputs = {key: value.to(device, non_blocking=device.type == "cuda") for key, value in processor(images=images, return_tensors="pt").items()}
            pixels = rad_inputs.get("pixel_values")
            cnn_inputs = torch.stack([cnn_transform(image.convert("RGB")) for image in images]).to(device, non_blocking=device.type == "cuda")
            if pixels is None or pixels.device != device or pixels.dtype != torch.float32 or cnn_inputs.device != device or cnn_inputs.dtype != torch.float32 or cnn_inputs.shape[1:] != (3, 224, 224) or next(encoder.parameters()).device != device or next(encoder.parameters()).dtype != torch.float32 or next(cnn.parameters()).device != device or next(cnn.parameters()).dtype != torch.float32:
                raise RuntimeError("E18 input/model device or dtype contract failed.")
            autocast = torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
            with torch.inference_mode(), autocast:
                hidden = encoder(**rad_inputs).last_hidden_state
                e15_logits, _ = e15_head(_e15_tokens_from_hidden(hidden))
                cnn_features = cnn(cnn_inputs)
            e15_cls = hidden[:, 0, :]
            if hidden.device != device or e15_logits.shape != (len(images), 2) or cnn_features.shape != (len(images), 512) or not bool(torch.isfinite(e15_cls).all() and torch.isfinite(e15_logits).all() and torch.isfinite(cnn_features).all()):
                raise FloatingPointError(f"E18 non-finite or invalid feature at {split} batch={batch_index}.")
            cls_parts.append(e15_cls.float().cpu().to(torch.float16).numpy())
            logits_parts.append(e15_logits.float().cpu().numpy())
            cnn_parts.append(cnn_features.float().cpu().to(torch.float16).numpy())
            seen.extend(identifiers)
            if labels.tolist() != frame.iloc[len(seen) - len(labels):len(seen)][label_column].astype(int).tolist():
                raise RuntimeError(f"E18 {split} sample/label order mismatch at batch={batch_index}.")
        except RuntimeError as error:
            if _oom(error):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(f"CUDA OOM during E18 {split} extraction at batch={batch_index}; feature_batch_size={config['runtime']['feature_batch_size']}. No cache for this split was written; lower the batch size and rerun.") from error
            raise
    values = FusionFeatureSet(split, np.concatenate(cls_parts).astype(np.float16, copy=False), np.concatenate(logits_parts).astype(np.float32, copy=False), np.concatenate(cnn_parts).astype(np.float16, copy=False), frame[label_column].to_numpy(dtype=np.int64), frame.copy(), False)
    if seen != frame["sample_id"].tolist() or values.e15_cls.shape != (len(frame), 768) or values.e15_logits.shape != (len(frame), 2) or values.cnn_features.shape != (len(frame), 512) or not np.isfinite(values.e15_cls).all() or not np.isfinite(values.e15_logits).all() or not np.isfinite(values.cnn_features).all():
        raise RuntimeError(f"E18 {split} extraction output contract failed.")
    return values


def _features(config: Mapping[str, Any], split: str, frame: pd.DataFrame, manifest: Path, label_column: str, processor: Any, encoder: nn.Module, e15_head: nn.Module, cnn: nn.Module, cnn_transform: Any, device: torch.device, e15_info: Mapping[str, Any], cnn_info: Mapping[str, Any], *, cache: bool) -> FusionFeatureSet:
    if cache:
        cached = _load_cache(config, split, frame, manifest, label_column, e15_info, cnn_info)
        if cached is not None:
            return cached
    values = _extract(config, split, frame, label_column, processor, encoder, e15_head, cnn, cnn_transform, device)
    if cache:
        _save_cache(config, values, manifest, label_column, e15_info, cnn_info)
    return values


def _verify_cached_e15_validation(config: Mapping[str, Any], validation: pd.DataFrame, values: FusionFeatureSet, baseline: Mapping[str, Any]) -> None:
    saved = pd.read_csv(Path(config["resolved"]["e15_predictions"]))
    saved = saved.set_index("image_path").reindex(validation["image_path"]).reset_index()
    probabilities = np.exp(values.e15_logits - values.e15_logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    prediction = probabilities.argmax(axis=1)
    e6 = saved["e10_e6_pred"].astype(int).to_numpy()
    final = np.where(e6 == 0, 0, prediction + 1)
    if not np.array_equal(prediction, np.asarray(baseline["e15_binary_predictions"], dtype=np.int64)) or not np.allclose(probabilities, saved[["e15_prob_bacterial", "e15_prob_viral"]].to_numpy(float), rtol=0.0, atol=float(config["e15_initialization"]["global_baseline_probability_atol"])) or not np.array_equal(final, np.asarray(baseline["e15_final_predictions"], dtype=np.int64)):
        raise RuntimeError("E18 cached E15 logits do not reproduce the verified E15 baseline; training is blocked.")


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_features(values: FusionFeatureSet, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    cls = torch.from_numpy(values.e15_cls.astype(np.float32)).to(device=device, dtype=torch.float32)
    logits = torch.from_numpy(values.e15_logits).to(device=device, dtype=torch.float32)
    cnn = torch.from_numpy(values.cnn_features.astype(np.float32)).to(device=device, dtype=torch.float32)
    return cls, logits, cnn


def _verify_zero_residual(config: Mapping[str, Any], values: FusionFeatureSet, device: torch.device, baseline: Mapping[str, Any]) -> dict[str, Any]:
    _seed(int(config["training"]["seed"]))
    if device.type == "cuda" and device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    head = ResidualFusionHead(config).to(device).eval()
    if next(head.parameters()).device != device or next(head.parameters()).dtype != torch.float32:
        raise RuntimeError("E18 fusion head dtype/device contract failed.")
    cls, e15_logits, cnn = _tensor_features(values, device)
    with torch.inference_mode():
        initial_logits, correction = head(cls, e15_logits, cnn)
        probabilities = torch.softmax(initial_logits, 1).cpu().numpy()
    if not torch.equal(initial_logits.cpu(), e15_logits.cpu()) or not torch.equal(correction.cpu(), torch.zeros_like(correction.cpu())) or not np.array_equal(probabilities.argmax(1), np.asarray(baseline["e15_binary_predictions"], dtype=np.int64)):
        raise RuntimeError("E18 zero-initialized residual does not exactly reproduce E15 initial binary logits/predictions.")
    return {"zero_correction_exact": True, "initial_e15_binary_prediction_exact": True, "head_parameter_counts": _count_parameters(head)}


def _train(config: Mapping[str, Any], train: FusionFeatureSet, device: torch.device) -> tuple[ResidualFusionHead, list[dict[str, Any]]]:
    _seed(int(config["training"]["seed"]))
    head = ResidualFusionHead(config).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    loader = DataLoader(TensorDataset(torch.from_numpy(train.e15_cls.astype(np.float32)), torch.from_numpy(train.e15_logits), torch.from_numpy(train.cnn_features.astype(np.float32)), torch.from_numpy(train.labels)), batch_size=int(config["runtime"]["fusion_batch_size"]), shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    criterion, history = nn.CrossEntropyLoss(), []
    for epoch in range(1, int(config["training"]["fixed_epochs"]) + 1):
        head.train(); total_loss = 0.0; total = 0; targets: list[int] = []; predictions: list[int] = []
        for cls, logits, cnn, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            fused, _ = head(cls.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda"), logits.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda"), cnn.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda"))
            labels = labels.to(device=device, dtype=torch.long, non_blocking=device.type == "cuda")
            loss = criterion(fused, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"E18 non-finite training loss at epoch={epoch}.")
            loss.backward(); optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(labels); total += len(labels)
            targets.extend(labels.cpu().tolist()); predictions.extend(fused.detach().argmax(1).cpu().tolist())
        metrics = _binary_metrics(np.asarray(targets), np.asarray(predictions))
        history.append({"epoch": epoch, "train_loss": total_loss / total, "train_macro_f1": metrics["macro_f1"], "train_accuracy": metrics["accuracy"], "train_confusion_matrix": metrics["confusion_matrix"], "train_bacterial_to_viral": metrics["bacterial_to_viral"], "train_viral_to_bacterial": metrics["viral_to_bacterial"], "checkpoint_selection": "none_fixed_final_epoch"})
    return head.eval(), history


def _full_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    metrics = classification_metrics(targets, np.eye(3)[predictions])
    metrics["macro_auroc_ovr"] = None
    for value in metrics["per_class"].values(): value["auroc_ovr"] = None
    matrix = np.asarray(metrics["confusion_matrix"])
    metrics.update({"bacterial_to_viral": int(matrix[1, 2]), "viral_to_bacterial": int(matrix[2, 1]), "bacterial_viral_misclassifications": int(matrix[1, 2] + matrix[2, 1])})
    return metrics


def _prepare(config: Mapping[str, Any]) -> dict[str, Path]:
    output, names = Path(config["resolved"]["output_dir"]), config["artifacts"]
    paths = {"output": output, "final": output / names["final_checkpoint"], "history": output / names["history"], "metrics": output / names["metrics"], "predictions": output / names["predictions"], "metadata": output / names["run_metadata"]}
    existing = [path for key, path in paths.items() if key != "output" and path.exists()]
    if existing:
        raise FileExistsError("E18 final results exist; automatic overwrite is forbidden: " + ", ".join(map(str, existing)))
    output.mkdir(parents=True, exist_ok=True)
    return paths


def _validate(config: Mapping[str, Any], validation: pd.DataFrame, trace: pd.DataFrame, values: FusionFeatureSet, head: ResidualFusionHead, device: torch.device, baseline: Mapping[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    cls, logits, cnn = _tensor_features(values, device)
    with torch.inference_mode(): probabilities = torch.softmax(head(cls, logits, cnn)[0], 1).cpu().numpy()
    binary = probabilities.argmax(1); targets = validation["label"].to_numpy(dtype=np.int64); e6 = trace["e6_pred"].astype(int).to_numpy(); final = np.where(e6 == 0, 0, binary + 1)
    e15_final = np.asarray(baseline["e15_final_predictions"], dtype=np.int64)
    e18_metrics, e15_metrics = _full_metrics(targets, final), _full_metrics(targets, e15_final)
    bv = targets >= 1
    e18_bv = _binary_metrics(targets[bv] - 1, binary[bv], probabilities[bv])
    e15_bv = _binary_metrics(targets[bv] - 1, np.asarray(baseline["e15_binary_predictions"])[bv])
    comparison = {"e15": {"macro_f1": e15_metrics["macro_f1"], "accuracy": e15_metrics["accuracy"], "bacterial_f1": e15_metrics["per_class"]["BACTERIAL_PNEUMONIA"]["f1"], "viral_f1": e15_metrics["per_class"]["VIRAL_PNEUMONIA"]["f1"], "viral_recall": e15_metrics["per_class"]["VIRAL_PNEUMONIA"]["recall"], "bv_confusion_matrix": e15_bv["confusion_matrix"]}, "e18": {"macro_f1": e18_metrics["macro_f1"], "accuracy": e18_metrics["accuracy"], "bacterial_f1": e18_metrics["per_class"]["BACTERIAL_PNEUMONIA"]["f1"], "viral_f1": e18_metrics["per_class"]["VIRAL_PNEUMONIA"]["f1"], "viral_recall": e18_metrics["per_class"]["VIRAL_PNEUMONIA"]["recall"], "bv_confusion_matrix": e18_bv["confusion_matrix"]}, "e15_only_correct_count": int(((e15_final == targets) & (final != targets)).sum()), "e18_only_correct_count": int(((final == targets) & (e15_final != targets)).sum())}
    predictions = validation[["sample_id", "image_path", "group_key", "source_identifier", "label", "class_name"]].copy().rename(columns={"label": "true_label", "class_name": "true_class_name"})
    predictions["e10_e6_pred"] = e6; predictions["e15_final_pred"] = e15_final; predictions["e18_binary_pred"] = binary; predictions["e18_prob_bacterial"], predictions["e18_prob_viral"] = probabilities[:, 0], probabilities[:, 1]; predictions["e18_final_pred"] = final; predictions["e18_final_pred_name"] = [CANONICAL_CLASSES[value] for value in final]; predictions["e10_normal_kept"] = e6 == 0
    return {"experiment": "e18", "e15_baseline_reproduction": {key: value for key, value in baseline.items() if not isinstance(value, np.ndarray)}, "e18_binary_specialist_on_true_bv_validation": e18_bv, "e18_normal_preserving_full_validation": e18_metrics, "e15_vs_e18": comparison, "validation_used_for_model_selection": False}, predictions


def static_smoke_check(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_e18_contract(config)
    train, validation = _frames(config)
    head = ResidualFusionHead(config).eval()
    with torch.inference_mode():
        initial, correction = head(torch.zeros((2, 768)), torch.tensor([[1.0, -1.0], [-2.0, 2.0]]), torch.zeros((2, 512)))
    if not torch.equal(initial, torch.tensor([[1.0, -1.0], [-2.0, 2.0]])) or not torch.equal(correction, torch.zeros((2, 2))):
        raise RuntimeError("E18 CPU zero-residual contract failed.")
    cnn, _, cnn_info = _load_cnn(config, torch.device("cpu"), pretrained=False)
    with torch.inference_mode(): cnn_out = cnn(torch.zeros((1, 3, 224, 224)))
    if cnn_out.shape != (1, 512): raise RuntimeError("E18 CPU ResNet18 pooled feature shape contract failed.")
    prerequisites = {key: Path(config["resolved"][key]).is_file() for key in ("e15_final", "e15_predictions", "e15_metrics", "e10_predictions")}
    return {"experiment": "e18", "static_check": "passed", "train_bv_samples": len(train), "validation_samples": len(validation), "train_validation_group_overlap": 0, "train_manifest_sha256": _sha256(Path(config["resolved"]["train_manifest"])), "validation_manifest_sha256": _sha256(Path(config["resolved"]["validation_manifest"])), "cnn_static_shape": list(cnn_out.shape), "cnn_static_identity": cnn_info, "fusion_head_parameter_counts": _count_parameters(head), "zero_residual_exact": True, "e15_e10_prerequisites_available": prerequisites, "artifacts_written": False}


def gpu_smoke_check(config: Mapping[str, Any], *, device_request: str | None = None) -> dict[str, Any]:
    static = static_smoke_check(config); train, _ = _frames(config); n = int(config["smoke_test"]["samples_per_class"])
    selected = pd.concat([train[train["binary_label"] == label].head(n) for label in (0, 1)], ignore_index=True)
    device = _device(device_request or str(config["runtime"]["device"])); processor, encoder, e15_head, e15_info = _restore_e15(config, device); cnn, cnn_transform, cnn_info = _load_cnn(config, device, pretrained=True)
    try:
        values = _extract(config, "gpu_smoke", selected, "binary_label", processor, encoder, e15_head, cnn, cnn_transform, device)
        smoke_probabilities = np.exp(values.e15_logits - values.e15_logits.max(axis=1, keepdims=True))
        smoke_probabilities /= smoke_probabilities.sum(axis=1, keepdims=True)
        baseline_like = {"e15_binary_predictions": smoke_probabilities.argmax(1)}
        zero = _verify_zero_residual(config, values, device, baseline_like)
        return {**static, "gpu_smoke": "passed", "device": str(device), "samples": len(selected), "zero_residual": zero, "parameter_counts": {"frozen_e15": e15_info["parameter_counts"], "frozen_cnn": cnn_info["parameter_counts"], "fusion_head": zero["head_parameter_counts"]}, "artifacts_written": False}
    finally:
        del encoder, e15_head, cnn
        if device.type == "cuda": torch.cuda.empty_cache()


def run_e18(config: Mapping[str, Any]) -> dict[str, Any]:
    static_smoke_check(config); train, validation = _frames(config); trace = _load_e10_trace(config, validation); device = _device(str(config["runtime"]["device"])); processor, encoder, e15_head, e15_info = _restore_e15(config, device); cnn, cnn_transform, cnn_info = _load_cnn(config, device, pretrained=True)
    baseline = _e15_baseline(config, validation, processor, encoder, e15_head, device)
    paths = _prepare(config); use_cache = bool(config["runtime"]["cache_features"])
    train_values = _features(config, "train", train, Path(config["resolved"]["train_manifest"]), "binary_label", processor, encoder, e15_head, cnn, cnn_transform, device, e15_info, cnn_info, cache=use_cache)
    validation_values = _features(config, "validation", validation, Path(config["resolved"]["validation_manifest"]), "label", processor, encoder, e15_head, cnn, cnn_transform, device, e15_info, cnn_info, cache=use_cache)
    _verify_cached_e15_validation(config, validation, validation_values, baseline)
    zero = _verify_zero_residual(config, validation_values, device, baseline)
    del encoder, e15_head, cnn
    if device.type == "cuda": torch.cuda.empty_cache()
    head, history = _train(config, train_values, device)
    metrics, predictions = _validate(config, validation, trace, validation_values, head, device, baseline)
    counts = {"frozen_e15": e15_info["parameter_counts"], "frozen_cnn": cnn_info["parameter_counts"], "fusion_head": _count_parameters(head)}
    _torch(paths["final"], {"experiment": "e18", "run_kind": "e18_frozen_e15_resnet18_residual_fusion_fixed_final_epoch", "epoch": int(config["training"]["fixed_epochs"]), "checkpoint_policy": "final_epoch_only", "fusion_head_state_dict": head.state_dict(), "config": config, "e15_restore": e15_info, "cnn": cnn_info, "parameter_counts": counts, "zero_residual_baseline": zero})
    _json(paths["history"], history); _json(paths["metrics"], metrics); _csv(paths["predictions"], predictions)
    _json(paths["metadata"], {"experiment": "e18", "e15": e15_info, "cnn": cnn_info, "frozen_modules": ["E15 RAD-DINO encoder", "E15 PatchAttentionHead", "ImageNet ResNet18"], "trainable_modules": ["zero_initialized_residual_fusion_head"], "zero_residual_baseline": zero, "cache_shapes": {"train": {"e15_cls": list(train_values.e15_cls.shape), "e15_logits": list(train_values.e15_logits.shape), "cnn": list(train_values.cnn_features.shape)}, "validation": {"e15_cls": list(validation_values.e15_cls.shape), "e15_logits": list(validation_values.e15_logits.shape), "cnn": list(validation_values.cnn_features.shape)}}, "training_policy": "B/V train only; fixed 3 epochs; final checkpoint only; no holdout, early stopping, validation selection, threshold tuning, or hyperparameter search", "validation_used_for_model_selection": False})
    return {"output_dir": str(paths["output"]), "final_epoch": 3, "e18_macro_f1": metrics["e18_normal_preserving_full_validation"]["macro_f1"], "result_files": {key: str(path) for key, path in paths.items() if key != "output"}}
