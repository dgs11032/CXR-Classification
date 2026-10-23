"""E14: frozen RAD-DINO spatial patch-token Bacterial/Viral specialist.

E14 never changes an E10 Normal decision. It consumes E10's read-only validation
trace and replaces only the binary Bacterial/Viral specialist decision for rows where
E10's E6 classifier predicted non-Normal. Raw manifest images are processed only by
RAD-DINO's AutoImageProcessor; ConvNeXt transforms are intentionally not imported.
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
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .metrics import classification_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CLASSES = ("NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA")
BINARY_CLASSES = ("BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA")
CLASS_TO_LABEL = {name: index for index, name in enumerate(CANONICAL_CLASSES)}
REQUIRED_COLUMNS = {"image_path", "label", "class_name", "source_split", "group_key", "source_identifier"}


@dataclass(frozen=True)
class PatchFeatureSet:
    split: str
    tokens: np.ndarray
    labels: np.ndarray
    index: pd.DataFrame
    from_cache: bool


class RawManifestDataset(Dataset[tuple[Image.Image, int, str]]):
    def __init__(self, frame: pd.DataFrame, root: Path, label_column: str) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.root = root
        self.label_column = label_column
        self.paths = [_resolve_image_path(value, root) for value in self.frame["image_path"]]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Image.Image, int, str]:
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                image.load()
                source = image.copy()
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise RuntimeError(f"Cannot load E14 original image: index={index}, path={path}, {type(error).__name__}: {error}") from error
        row = self.frame.iloc[index]
        return source, int(row[self.label_column]), str(row["sample_id"])


def _collate(batch: list[tuple[Image.Image, int, str]]) -> tuple[list[Image.Image], Tensor, list[str]]:
    images, labels, identifiers = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), list(identifiers)


def _inside_project(value: str | Path, root: Path) -> Path:
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"E14 path leaves project root: {value}")
    return resolved


def _resolve_image_path(value: object, root: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        raise ValueError(f"E14 manifest image_path must be relative: {value}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise FileNotFoundError(f"E14 manifest source image unavailable: {resolved}")
    return resolved


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"E14 YAML must be a mapping: {path}")
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
        digest.update(str(value).encode("utf-8")); digest.update(b"\0")
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


def _atomic_npz(path: Path, *, tokens: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, tokens=tokens, labels=labels)
    os.replace(temporary, path)


def validate_e14_contract(config: Mapping[str, Any]) -> None:
    expected = {"experiment": "e14", "status": "rad_dino_frozen_spatial_patch_attention_bv_specialist", "data_config": "configs/data_config.yaml", "output_dir": "artifacts/experiments/e14"}
    errors = [{"key": key, "expected": value, "actual": config.get(key)} for key, value in expected.items() if config.get(key) != value]
    model = config.get("model", {})
    expected_model = {"hf_model_id": "microsoft/rad-dino", "trust_remote_code": False, "encoder_mode": "frozen_eval_inference_mode", "expected_hidden_size": 768, "patch_size": 14, "expected_image_size": 518, "spatial_pool_grid": [4, 4], "cache_tokens": "cls_plus_spatially_pooled_patch_tokens"}
    errors.extend({"key": f"model.{key}", "expected": value, "actual": model.get(key)} for key, value in expected_model.items() if model.get(key) != value)
    head = config.get("head", {})
    if head.get("architecture") != "cls_plus_4x4_patch_attention_pooling_binary_head" or int(head.get("num_classes", -1)) != 2:
        errors.append({"key": "head", "expected": "fixed E14 4x4 binary attention head", "actual": head})
    runtime = config.get("runtime", {})
    if int(runtime.get("feature_batch_size", -1)) != 2 or bool(runtime.get("amp")) is not True:
        errors.append({"key": "runtime", "expected": "feature_batch_size=2 and amp=true", "actual": runtime})
    training = config.get("training", {})
    fixed_training = {"seed": 42, "epochs": 10, "optimizer": "AdamW", "internal_holdout": "disabled", "early_stopping": False, "validation_weight_selection": False}
    errors.extend({"key": f"training.{key}", "expected": value, "actual": training.get(key)} for key, value in fixed_training.items() if training.get(key) != value)
    split = config.get("split_contract", {})
    if split.get("binary_class_order") != list(BINARY_CLASSES) or split.get("canonical_to_binary") != {1: 0, 2: 1} or bool(split.get("no_test_access")) is not True:
        errors.append({"key": "split_contract", "expected": "fixed B/V mapping and no test", "actual": split})
    routing = config.get("e10_routing_input", {})
    if routing.get("normal_rule") != "e6_pred_zero_kept_else_e14_binary_plus_one":
        errors.append({"key": "e10_routing_input.normal_rule", "expected": "e6_pred_zero_kept_else_e14_binary_plus_one", "actual": routing.get("normal_rule")})
    if errors:
        raise ValueError(f"E14 configuration contract drift: {errors}")


def _validate_shared_data_config(config: Mapping[str, Any], root: Path) -> None:
    data = _read_yaml(_inside_project(config["data_config"], root))
    manifests = data.get("manifests", {})
    if set(manifests) != {"train", "validation"}:
        raise ValueError("E14 requires shared train/validation-only data configuration.")
    e14 = config["split_contract"]
    if _inside_project(manifests["train"], root) != _inside_project(e14["train_manifest"], root) or _inside_project(manifests["validation"], root) != _inside_project(e14["validation_manifest"], root):
        raise ValueError("E14 manifests must exactly match configs/data_config.yaml.")
    if data.get("class_mapping") != CLASS_TO_LABEL:
        raise ValueError("E14 canonical label mapping differs from shared data config.")


def load_e14_config(path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config_path = _inside_project(path, root)
    if not config_path.is_file():
        raise FileNotFoundError(f"E14 config not found: {config_path}")
    config = _read_yaml(config_path)
    validate_e14_contract(config); _validate_shared_data_config(config, root)
    contract, routing = config["split_contract"], config["e10_routing_input"]
    config["resolved"] = {
        "project_root": str(root), "config_path": str(config_path),
        "train_manifest": str(_inside_project(contract["train_manifest"], root)),
        "validation_manifest": str(_inside_project(contract["validation_manifest"], root)),
        "e10_predictions": str(_inside_project(routing["predictions"], root)),
        "e10_metrics": str(_inside_project(routing["metrics"], root)),
        "output_dir": str(_inside_project(config["output_dir"], root)),
    }
    return config


def _manifest(path: Path, root: Path, split: str, expected: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if REQUIRED_COLUMNS - set(frame.columns) or frame.empty or frame["image_path"].duplicated().any():
        raise ValueError(f"E14 {split} manifest schema/identity contract failed.")
    numeric = pd.to_numeric(frame["label"], errors="coerce")
    if numeric.isna().any() or not (numeric % 1 == 0).all():
        raise ValueError(f"E14 {split} has non-integer labels.")
    frame["label"] = numeric.astype(int)
    if any(CLASS_TO_LABEL.get(str(row.class_name)) != int(row.label) for row in frame[["class_name", "label"]].itertuples(index=False)):
        raise ValueError(f"E14 {split} has class/label inconsistency.")
    actual = frame["class_name"].value_counts().to_dict()
    expected_counts = {key: int(value) for key, value in expected.items() if key != "total"}
    if len(frame) != int(expected["total"]) or actual != expected_counts:
        raise ValueError(f"E14 {split} count contract drift: actual={len(frame)}/{actual}, expected={expected}")
    for value in frame["image_path"]: _resolve_image_path(value, root)
    frame.insert(0, "sample_id", frame["image_path"].astype(str))
    frame["canonical_label"] = frame["label"].astype(int)
    frame["binary_label"] = frame["canonical_label"].map({1: 0, 2: 1}).fillna(-1).astype(int)
    return frame


def _binary_train_frame(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    selected = frame[frame["class_name"].isin(BINARY_CLASSES)].copy().reset_index(drop=True)
    mapping = config["split_contract"]["canonical_to_binary"]
    selected["canonical_label"] = selected["label"].astype(int)
    selected["binary_label"] = selected["canonical_label"].map(mapping).astype(int)
    expected = config["split_contract"]["expected_counts"]["train"]
    counts = selected["class_name"].value_counts().to_dict()
    if len(selected) != int(expected["total"]) or counts != {name: int(expected[name]) for name in BINARY_CLASSES} or set(selected["binary_label"]) != {0, 1}:
        raise ValueError("E14 fixed B/V train subset contract failed.")
    return selected


def static_smoke_check(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_e14_contract(config)
    root, contract = Path(config["resolved"]["project_root"]), config["split_contract"]
    train_full = _manifest(Path(config["resolved"]["train_manifest"]), root, "train", {"total": 4448, "NORMAL": 1148, "BACTERIAL_PNEUMONIA": 2153, "VIRAL_PNEUMONIA": 1147})
    train = _binary_train_frame(train_full, config)
    validation = _manifest(Path(config["resolved"]["validation_manifest"]), root, "validation", contract["expected_counts"]["validation"])
    overlap_ids, overlap_groups = set(train["sample_id"]) & set(validation["sample_id"]), set(train["group_key"]) & set(validation["group_key"])
    if overlap_ids or overlap_groups:
        raise ValueError(f"E14 train/validation contamination: sample_ids={len(overlap_ids)}, group_keys={len(overlap_groups)}")
    synthetic_hidden = torch.zeros((1, 1 + 37 * 37, 768), dtype=torch.float32)
    pooled_tokens, static_grid = _pool_patch_tokens(synthetic_hidden, config)
    static_head = PatchAttentionHead(config).eval()
    with torch.inference_mode():
        static_logits, static_attention = static_head(pooled_tokens)
    if static_logits.shape != (1, 2) or static_attention.shape != (1, 16) or not torch.allclose(static_attention.sum(dim=1), torch.ones(1), atol=1e-6):
        raise RuntimeError("E14 CPU patch-pooling/attention-head shape contract failed.")
    routing = Path(config["resolved"]["e10_predictions"])
    return {"experiment": "e14", "static_check": "passed", "train_bv_samples": len(train), "validation_samples": len(validation), "train_bv_groups": int(train["group_key"].nunique()), "validation_groups": int(validation["group_key"].nunique()), "train_validation_group_overlap": 0, "train_manifest_sha256": _sha256(Path(config["resolved"]["train_manifest"])), "validation_manifest_sha256": _sha256(Path(config["resolved"]["validation_manifest"])), "e10_routing_trace_available": routing.is_file(), "e10_routing_trace": str(routing), "output_dir_exists": Path(config["resolved"]["output_dir"]).exists(), "preprocessing": "AutoImageProcessor only; ConvNeXt transforms excluded"}


def _device(request: str) -> torch.device:
    if request == "cuda":
        if not torch.cuda.is_available(): raise RuntimeError("E14 config requires CUDA but CUDA is unavailable.")
        return torch.device("cuda")
    if request == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if request == "cpu": return torch.device("cpu")
    raise ValueError(f"Unsupported E14 device: {request}")


def _load_encoder(config: Mapping[str, Any], device: torch.device) -> tuple[Any, nn.Module, dict[str, Any]]:
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        raise RuntimeError("E14 needs transformers installed in the AWS runtime before GPU smoke/full execution.") from error
    model_config = config["model"]
    processor = AutoImageProcessor.from_pretrained(model_config["hf_model_id"], revision=model_config["revision"], trust_remote_code=False)
    encoder = AutoModel.from_pretrained(model_config["hf_model_id"], revision=model_config["revision"], trust_remote_code=False).to(device)
    encoder.requires_grad_(False); encoder.eval()
    if any(parameter.requires_grad for parameter in encoder.parameters()): raise RuntimeError("E14 RAD-DINO encoder freeze contract failed.")
    if int(getattr(encoder.config, "hidden_size", -1)) != int(model_config["expected_hidden_size"]): raise RuntimeError("E14 hidden-size contract failed.")
    return processor, encoder, {"model_id": model_config["hf_model_id"], "requested_revision": model_config["revision"], "resolved_commit_hash": getattr(encoder.config, "_commit_hash", None), "model_class": type(encoder).__name__, "processor_class": type(processor).__name__, "processor_size": getattr(processor, "size", None)}


def _pool_patch_tokens(hidden: Tensor, config: Mapping[str, Any]) -> tuple[Tensor, dict[str, int]]:
    if hidden.ndim != 3 or hidden.shape[-1] != int(config["model"]["expected_hidden_size"]): raise RuntimeError(f"E14 invalid last_hidden_state shape: {tuple(hidden.shape)}")
    cls, patch = hidden[:, 0, :], hidden[:, 1:, :]
    patch_count = int(patch.shape[1]); side = math.isqrt(patch_count)
    if side * side != patch_count: raise RuntimeError(f"E14 patch token count is not a square grid: {patch_count}")
    expected_side = int(config["model"]["expected_image_size"]) // int(config["model"]["patch_size"])
    if side != expected_side: raise RuntimeError(f"E14 patch grid drift: expected {expected_side}x{expected_side}, got {side}x{side}")
    batch, hidden_size = cls.shape
    grid_h, grid_w = (int(value) for value in config["model"]["spatial_pool_grid"])
    patch_map = patch.reshape(batch, side, side, hidden_size).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(patch_map, (grid_h, grid_w)).permute(0, 2, 3, 1).reshape(batch, grid_h * grid_w, hidden_size)
    tokens = torch.cat([cls.unsqueeze(1), pooled], dim=1)
    if tokens.shape != (batch, 1 + grid_h * grid_w, hidden_size): raise RuntimeError("E14 CLS/pooled-patch reshape contract failed.")
    return tokens, {"raw_patch_tokens": patch_count, "raw_grid_height": side, "raw_grid_width": side, "pooled_grid_height": grid_h, "pooled_grid_width": grid_w, "cached_tokens": 1 + grid_h * grid_w}


class PatchAttentionHead(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden = int(config["model"]["expected_hidden_size"]); head = config["head"]
        attention_hidden = int(head["attention_hidden_dimension"]); classifier_hidden = int(head["classifier_hidden_dimension"])
        self.hidden = hidden; self.tokens = 1 + int(config["model"]["spatial_pool_grid"][0]) * int(config["model"]["spatial_pool_grid"][1])
        self.cls_norm, self.patch_norm = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.attention = nn.Sequential(nn.Linear(hidden, attention_hidden), nn.Tanh(), nn.Linear(attention_hidden, 1))
        self.classifier = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, classifier_hidden), nn.GELU(), nn.Dropout(float(head["dropout"])), nn.Linear(classifier_hidden, int(head["num_classes"])))

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 3 or tokens.shape[1:] != (self.tokens, self.hidden): raise RuntimeError(f"E14 head expected [B,{self.tokens},{self.hidden}], got {tuple(tokens.shape)}")
        cls, patches = self.cls_norm(tokens[:, 0]), self.patch_norm(tokens[:, 1:])
        attention = torch.softmax(self.attention(patches).squeeze(-1), dim=1)
        pooled = torch.sum(patches * attention.unsqueeze(-1), dim=1)
        logits = self.classifier(torch.cat([cls, pooled], dim=1))
        if logits.shape != (tokens.shape[0], 2) or not bool(torch.isfinite(logits).all()): raise RuntimeError("E14 head output contract failed.")
        return logits, attention


def _cache_paths(config: Mapping[str, Any], split: str) -> dict[str, Path]:
    output, artifacts = Path(config["resolved"]["output_dir"]), config["artifacts"]
    cache = output / artifacts["cache_dir"]; prefix = "train" if split == "train" else "validation"
    return {"features": cache / artifacts[f"{prefix}_features"], "index": cache / artifacts[f"{prefix}_index"], "metadata": cache / artifacts["cache_metadata"]}


def _cache_signature(config: Mapping[str, Any], frame: pd.DataFrame, manifest: Path, split: str, label_column: str) -> dict[str, Any]:
    return {"split": split, "manifest_sha256": _sha256(manifest), "sample_id_hash": _ordered_hash(frame["sample_id"]), "label_hash": _ordered_hash(frame[label_column]), "n_samples": len(frame), "model_id": config["model"]["hf_model_id"], "requested_revision": config["model"]["revision"], "feature_kind": "cls_plus_adaptive_avg_pooled_4x4_patch_tokens", "expected_token_shape": [17, 768]}


def _load_cache(config: Mapping[str, Any], split: str, frame: pd.DataFrame, manifest: Path, label_column: str) -> PatchFeatureSet | None:
    paths = _cache_paths(config, split)
    if not all(path.is_file() for path in paths.values()): return None
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8")); index = pd.read_csv(paths["index"])
        with np.load(paths["features"], allow_pickle=False) as archive: tokens, labels = np.asarray(archive["tokens"], dtype=np.float16), np.asarray(archive["labels"], dtype=np.int64)
    except (OSError, ValueError, KeyError, json.JSONDecodeError): return None
    required = ["sample_id", "image_path", "group_key", "source_identifier", "canonical_label", "binary_label"]
    if metadata.get("splits", {}).get(split) != _cache_signature(config, frame, manifest, split, label_column) or list(index.columns) != required or index["sample_id"].tolist() != frame["sample_id"].tolist() or not np.array_equal(labels, frame[label_column].to_numpy(dtype=np.int64)):
        return None
    if tokens.shape != (len(frame), 17, 768) or not np.isfinite(tokens).all(): return None
    return PatchFeatureSet(split, tokens, labels, index, True)


def _save_cache(config: Mapping[str, Any], values: PatchFeatureSet, manifest: Path, label_column: str, encoder_info: Mapping[str, Any], grid: Mapping[str, int]) -> None:
    paths = _cache_paths(config, values.split)
    index = values.index[["sample_id", "image_path", "group_key", "source_identifier", "canonical_label", "binary_label"]].copy()
    _atomic_npz(paths["features"], tokens=values.tokens, labels=values.labels); _atomic_csv(paths["index"], index)
    metadata: dict[str, Any] = {"version": 1, "splits": {}}
    if paths["metadata"].is_file():
        try: metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): pass
    metadata.setdefault("splits", {})[values.split] = _cache_signature(config, values.index, manifest, values.split, label_column)
    metadata["encoder"] = dict(encoder_info); metadata["patch_grid"] = dict(grid); _atomic_json(paths["metadata"], metadata)


def _oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def _extract(config: Mapping[str, Any], split: str, frame: pd.DataFrame, label_column: str, processor: Any, encoder: nn.Module, device: torch.device) -> tuple[PatchFeatureSet, dict[str, int]]:
    loader = DataLoader(RawManifestDataset(frame, Path(config["resolved"]["project_root"]), label_column), batch_size=int(config["runtime"]["feature_batch_size"]), shuffle=False, num_workers=int(config["runtime"]["num_workers"]), pin_memory=bool(config["runtime"]["pin_memory"]) and device.type == "cuda", collate_fn=_collate)
    parts: list[np.ndarray] = []; seen: list[str] = []; grid: dict[str, int] | None = None; amp = bool(config["runtime"]["amp"]) and device.type == "cuda"
    for batch_index, (images, labels, sample_ids) in enumerate(loader):
        try:
            inputs = {key: value.to(device, non_blocking=device.type == "cuda") for key, value in processor(images=images, return_tensors="pt").items()}
            autocast = torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
            with torch.inference_mode(), autocast:
                tokens, current_grid = _pool_patch_tokens(encoder(**inputs).last_hidden_state, config)
            if not bool(torch.isfinite(tokens).all()): raise FloatingPointError(f"E14 non-finite tokens at {split} batch {batch_index}.")
            parts.append(tokens.cpu().to(torch.float16).numpy()); seen.extend(sample_ids); grid = current_grid
            expected = frame.iloc[len(seen)-len(labels):len(seen)][label_column].astype(int).tolist()
            if labels.tolist() != expected: raise RuntimeError(f"E14 {split} sample/label alignment failed at batch {batch_index}.")
        except RuntimeError as error:
            if _oom(error):
                if device.type == "cuda": torch.cuda.empty_cache()
                raise RuntimeError(f"CUDA OOM extracting E14 {split} at batch={batch_index}, feature_batch_size={config['runtime']['feature_batch_size']}. No cache for this split was written; reduce feature_batch_size and rerun.") from error
            raise
    tokens = np.concatenate(parts, axis=0).astype(np.float16, copy=False)
    if seen != frame["sample_id"].tolist() or tokens.shape != (len(frame), 17, 768) or grid is None: raise RuntimeError(f"E14 {split} extraction contract failed.")
    return PatchFeatureSet(split, tokens, frame[label_column].to_numpy(dtype=np.int64), frame, False), grid


def _features(config: Mapping[str, Any], split: str, frame: pd.DataFrame, manifest: Path, label_column: str, processor: Any, encoder: nn.Module, device: torch.device, encoder_info: Mapping[str, Any]) -> PatchFeatureSet:
    if bool(config["runtime"]["cache_features"]):
        cached = _load_cache(config, split, frame, manifest, label_column)
        if cached is not None: return cached
    values, grid = _extract(config, split, frame, label_column, processor, encoder, device)
    if bool(config["runtime"]["cache_features"]): _save_cache(config, values, manifest, label_column, encoder_info, grid)
    return values


def gpu_smoke_check(config: Mapping[str, Any], *, device_request: str | None = None) -> dict[str, Any]:
    static = static_smoke_check(config); root = Path(config["resolved"]["project_root"])
    train_full = _manifest(Path(config["resolved"]["train_manifest"]), root, "train", {"total": 4448, "NORMAL": 1148, "BACTERIAL_PNEUMONIA": 2153, "VIRAL_PNEUMONIA": 1147})
    train = _binary_train_frame(train_full, config); n = int(config["smoke_test"]["samples_per_class"])
    selected = pd.concat([train[train["binary_label"] == label].head(n) for label in (0, 1)], ignore_index=True)
    device = _device(device_request or str(config["runtime"]["device"])); processor, encoder, info = _load_encoder(config, device); head = PatchAttentionHead(config).to(device).eval()
    try:
        features, grid = _extract(config, "gpu_smoke", selected, "binary_label", processor, encoder, device)
        with torch.inference_mode(): logits, attention = head(torch.from_numpy(features.tokens.astype(np.float32)).to(device))
        if attention.shape != (2*n, 16) or not torch.allclose(attention.sum(dim=1), torch.ones(2*n, device=device), atol=1e-5): raise RuntimeError("E14 attention pooling smoke contract failed.")
    finally:
        del encoder, head
        if device.type == "cuda": torch.cuda.empty_cache()
    return {**static, "gpu_smoke_check": "passed", "device": str(device), "encoder": info, "input_samples": len(selected), "raw_patch_grid": grid, "cached_token_shape": list(features.tokens.shape), "logit_shape": list(logits.shape), "artifacts_written": False}


def _binary_metrics(targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(targets, predictions, labels=[0, 1], zero_division=0)
    auroc = float(roc_auc_score(targets, probabilities[:, 1])) if probabilities is not None and len(np.unique(targets)) == 2 else None
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    return {"n_samples": int(len(targets)), "accuracy": float(accuracy_score(targets, predictions)), "macro_f1": float(f1_score(targets, predictions, labels=[0, 1], average="macro")), "auroc_viral_positive": auroc, "confusion_matrix": matrix.tolist(), "bacterial_to_viral": int(matrix[0, 1]), "viral_to_bacterial": int(matrix[1, 0]), "bacterial_viral_misclassifications": int(matrix[0, 1]+matrix[1, 0]), "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, name in enumerate(BINARY_CLASSES)}}


def _load_e10_trace(config: Mapping[str, Any], validation: pd.DataFrame) -> pd.DataFrame:
    path = Path(config["resolved"]["e10_predictions"])
    if not path.is_file(): raise FileNotFoundError(f"E14 requires existing E10 routing trace to preserve Normal decisions: {path}")
    trace = pd.read_csv(path); required = {"image_path", "group_key", "true_label", "e6_pred", "e9_pred", "final_pred", "e9_prob_bacterial", "e9_prob_viral"}
    if required - set(trace.columns) or trace["image_path"].duplicated().any(): raise ValueError("E14 E10 routing trace schema/identity contract failed.")
    trace = trace.set_index("image_path").reindex(validation["image_path"]).reset_index()
    if trace.isna().any().any() or trace["true_label"].astype(int).tolist() != validation["label"].astype(int).tolist() or trace["group_key"].tolist() != validation["group_key"].tolist(): raise ValueError("E14 E10 routing trace does not exactly align with current validation manifest.")
    e6, e9, final = trace["e6_pred"].astype(int).to_numpy(), trace["e9_pred"].astype(int).to_numpy(), trace["final_pred"].astype(int).to_numpy()
    if not np.array_equal(final, np.where(e6 == 0, 0, e9 + 1)) or set(e6)-{0,1,2} or set(e9)-{0,1}: raise ValueError("E14 E10 hard-routing invariant failed.")
    return trace


def _output_paths(config: Mapping[str, Any], overwrite: bool) -> dict[str, Path]:
    output, names = Path(config["resolved"]["output_dir"]), config["artifacts"]
    paths = {"output": output, "head": output/names["head_checkpoint"], "history": output/names["history"], "metrics": output/names["metrics"], "predictions": output/names["predictions"], "metadata": output/names["run_metadata"]}
    existing = [path for key,path in paths.items() if key != "output" and path.exists()]
    if existing and not overwrite: raise FileExistsError("E14 final output exists; use --overwrite-results to replace only E14 final files: " + ", ".join(map(str, existing)))
    output.mkdir(parents=True, exist_ok=True); return paths


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _train_head(config: Mapping[str, Any], train: PatchFeatureSet, device: torch.device) -> tuple[PatchAttentionHead, list[dict[str, Any]]]:
    _seed(int(config["training"]["seed"])); head = PatchAttentionHead(config).to(device); optimizer = torch.optim.AdamW(head.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"])); criterion = nn.CrossEntropyLoss(); loader = DataLoader(TensorDataset(torch.from_numpy(train.tokens.astype(np.float32)), torch.from_numpy(train.labels)), batch_size=int(config["runtime"]["head_batch_size"]), shuffle=True, num_workers=0, pin_memory=device.type=="cuda"); amp = bool(config["runtime"]["amp"]) and device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp) if device.type=="cuda" else None; history=[]
    for epoch in range(1, int(config["training"]["epochs"])+1):
        head.train(); loss_sum=0.0; correct=0; total=0
        for tokens, labels in loader:
            tokens, labels = tokens.to(device, non_blocking=device.type=="cuda"), labels.to(device, non_blocking=device.type=="cuda"); optimizer.zero_grad(set_to_none=True); autocast=torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
            with autocast: logits,_=head(tokens); loss=criterion(logits, labels)
            if not bool(torch.isfinite(loss)): raise FloatingPointError(f"E14 non-finite head loss at epoch {epoch}.")
            if scaler is None: loss.backward(); optimizer.step()
            else: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += float(loss.detach().cpu())*len(labels); correct += int((logits.argmax(1)==labels).sum().detach().cpu()); total += len(labels)
        history.append({"epoch": epoch, "train_loss": loss_sum/total, "train_accuracy": correct/total, "selection": "fixed_final_epoch_no_validation"})
    return head.eval(), history


def run_e14(config: Mapping[str, Any], *, overwrite_results: bool = False) -> dict[str, Any]:
    validate_e14_contract(config); static_smoke_check(config)
    root, contract = Path(config["resolved"]["project_root"]), config["split_contract"]
    train_full = _manifest(Path(config["resolved"]["train_manifest"]), root, "train", {"total":4448,"NORMAL":1148,"BACTERIAL_PNEUMONIA":2153,"VIRAL_PNEUMONIA":1147}); train_frame = _binary_train_frame(train_full, config)
    validation = _manifest(Path(config["resolved"]["validation_manifest"]), root, "validation", contract["expected_counts"]["validation"]); trace = _load_e10_trace(config, validation)
    paths = _output_paths(config, overwrite_results)
    device = _device(str(config["runtime"]["device"])); processor, encoder, encoder_info = _load_encoder(config, device)
    train = _features(config,"train",train_frame,Path(config["resolved"]["train_manifest"]),"binary_label",processor,encoder,device,encoder_info); valid = _features(config,"validation",validation,Path(config["resolved"]["validation_manifest"]),"label",processor,encoder,device,encoder_info)
    del encoder
    if device.type == "cuda": torch.cuda.empty_cache()
    head, history = _train_head(config, train, device)
    with torch.inference_mode(): logits, attention = head(torch.from_numpy(valid.tokens.astype(np.float32)).to(device)); probabilities=torch.softmax(logits,dim=1).cpu().numpy()
    e14_binary = probabilities.argmax(1); e6_pred=trace["e6_pred"].astype(int).to_numpy(); final=np.where(e6_pred==0,0,e14_binary+1).astype(np.int64); targets=validation["label"].to_numpy(dtype=np.int64)
    hard_probs=np.eye(3,dtype=np.float64)[final]; final_metrics=classification_metrics(targets,hard_probs); final_metrics["macro_auroc_ovr"]=None
    for values in final_metrics["per_class"].values(): values["auroc_ovr"]=None
    matrix=np.asarray(final_metrics["confusion_matrix"],dtype=np.int64); final_metrics.update({"bacterial_to_viral":int(matrix[1,2]),"viral_to_bacterial":int(matrix[2,1]),"bacterial_viral_misclassifications":int(matrix[1,2]+matrix[2,1]),"auroc_not_probability_comparable":True})
    bv=targets>=1; e14_bv=_binary_metrics(targets[bv]-1,e14_binary[bv],probabilities[bv]); e9_bv=_binary_metrics(targets[bv]-1,trace.loc[bv,"e9_pred"].astype(int).to_numpy(),trace.loc[bv,["e9_prob_bacterial","e9_prob_viral"]].to_numpy(dtype=float)); e10_final=trace["final_pred"].astype(int).to_numpy(); e10_metrics=classification_metrics(targets,np.eye(3)[e10_final]); e10_metrics["macro_auroc_ovr"]=None
    metrics={"experiment":"e14","e14_binary_specialist_on_true_bv_validation":e14_bv,"e9_binary_reference_from_e10_trace":e9_bv,"e10_full_reference_from_trace":{"macro_f1":e10_metrics["macro_f1"],"accuracy":e10_metrics["accuracy"],"configured_reference":float(config["e10_routing_input"]["expected_e10_macro_f1"])},"e14_normal_preserving_full_validation":final_metrics,"macro_f1_delta_vs_e10_trace":float(final_metrics["macro_f1"]-e10_metrics["macro_f1"])}
    predictions=validation[["sample_id","image_path","group_key","source_identifier","label","class_name"]].copy().rename(columns={"label":"true_label","class_name":"true_class_name"}); predictions["e10_e6_pred"]=e6_pred; predictions["e10_e9_pred"]=trace["e9_pred"].astype(int); predictions["e10_final_pred"]=e10_final; predictions["e14_binary_pred"]=e14_binary; predictions["e14_prob_bacterial"],predictions["e14_prob_viral"]=probabilities[:,0],probabilities[:,1]; predictions["e14_final_pred"]=final; predictions["e14_final_pred_name"]=[CANONICAL_CLASSES[x] for x in final]; predictions["e10_normal_kept"]=e6_pred==0
    checkpoint = {
        "experiment": "e14",
        "head_state_dict": head.state_dict(),
        "head_config": config["head"],
        "model_config": config["model"],
        "train_manifest_sha256": _sha256(Path(config["resolved"]["train_manifest"])),
        "validation_manifest_sha256": _sha256(Path(config["resolved"]["validation_manifest"])),
    }
    torch.save(checkpoint, paths["head"])
    _atomic_json(paths["history"],history); _atomic_json(paths["metrics"],metrics); _atomic_csv(paths["predictions"],predictions); _atomic_json(paths["metadata"],{"experiment":"e14","encoder":encoder_info,"encoder_frozen":True,"training":"B/V train only, fixed 10 epochs, no internal holdout, no validation selection/early stopping","feature_source":{"train":"cache" if train.from_cache else "fresh","validation":"cache" if valid.from_cache else "fresh"},"cached_tensor_shape":{"train":list(train.tokens.shape),"validation":list(valid.tokens.shape)},"cache_disk_note":"float16 cache stores 17x768 CLS+4x4 pooled tokens; full 37x37 patch tokens are intentionally not persisted.","e10_routing_trace":str(Path(config["resolved"]["e10_predictions"]))})
    return {"output_dir":str(paths["output"]),"e14_macro_f1":final_metrics["macro_f1"],"e10_macro_f1":e10_metrics["macro_f1"],"macro_f1_delta":metrics["macro_f1_delta_vs_e10_trace"],"result_files":{key:str(value) for key,value in paths.items() if key!="output"}}
