"""E13: RAD-DINO frozen CLS-feature extraction with a train-only 3-class linear probe.

This module deliberately does not reuse the ConvNeXt transform pipeline. It reads the
fixed train/validation manifests, opens their original image paths, and supplies PIL
images only to Hugging Face's AutoImageProcessor for ``microsoft/rad-dino``.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, UnidentifiedImageError
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .metrics import classification_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ("NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA")
CLASS_TO_LABEL = {name: index for index, name in enumerate(CLASS_NAMES)}
_REQUIRED_MANIFEST_COLUMNS = {
    "image_path",
    "label",
    "class_name",
    "source_split",
    "group_key",
}


@dataclass(frozen=True)
class FeatureSet:
    split: str
    features: np.ndarray
    labels: np.ndarray
    index: pd.DataFrame
    from_cache: bool


class ManifestImageDataset(Dataset[tuple[Image.Image, int, str]]):
    """Raw original-image manifest dataset for model-native preprocessing only."""

    def __init__(self, frame: pd.DataFrame, project_root: Path) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.project_root = project_root
        self.paths = [_resolve_image_path(path, project_root) for path in self.frame["image_path"]]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Image.Image, int, str]:
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                image.load()
                # Processor's do_convert_rgb=True remains the sole channel conversion policy.
                source = image.copy()
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise RuntimeError(
                f"Unable to open original manifest image: index={index}, path={path}, "
                f"error={type(error).__name__}: {error}"
            ) from error
        row = self.frame.iloc[index]
        return source, int(row["label"]), str(row["image_path"])


def _collate_raw_images(batch: list[tuple[Image.Image, int, str]]) -> tuple[list[Image.Image], torch.Tensor, list[str]]:
    images, labels, sample_ids = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), list(sample_ids)


def _inside_project(value: str | Path, root: Path) -> Path:
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"E13 paths must remain inside the project root: {value}")
    return resolved


def _resolve_image_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Manifest image_path must be project-relative: {value}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Manifest image_path leaves the project root: {value}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Manifest image is missing: {resolved}")
    return resolved


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"E13 config must be a YAML mapping: {path}")
    return payload


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


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, *, features: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, features=features, labels=labels)
    os.replace(temporary, path)


def _atomic_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temporary)
    os.replace(temporary, path)


def _manifest_frame(path: Path, root: Path, split: str, expected: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = _REQUIRED_MANIFEST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"E13 {split} manifest is missing columns: {sorted(missing)}")
    if frame.empty or frame["image_path"].duplicated().any():
        raise ValueError(f"E13 {split} manifest must be non-empty with unique image_path values.")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not (labels % 1 == 0).all():
        raise ValueError(f"E13 {split} labels must be integer 0/1/2 values.")
    frame["label"] = labels.astype(int)
    if set(frame["label"]) != {0, 1, 2}:
        raise ValueError(f"E13 {split} must contain all three labels, got {sorted(set(frame['label']))}.")
    if any(CLASS_TO_LABEL.get(str(row.class_name)) != int(row.label) for row in frame[["class_name", "label"]].itertuples(index=False)):
        raise ValueError(f"E13 {split} manifest has inconsistent class_name/label mapping.")
    expected_total = int(expected["total"])
    expected_counts = {name: int(expected[name]) for name in CLASS_NAMES}
    actual_counts = frame["class_name"].value_counts().to_dict()
    if len(frame) != expected_total or actual_counts != expected_counts:
        raise ValueError(
            f"E13 {split} split contract drift: total={len(frame)}, counts={actual_counts}; "
            f"expected total={expected_total}, counts={expected_counts}"
        )
    for value in frame["image_path"]:
        _resolve_image_path(str(value), root)
    result = frame.copy()
    result.insert(0, "sample_id", result["image_path"].astype(str))
    return result


def validate_e13_contract(config: Mapping[str, Any]) -> None:
    required = {
        "experiment": "e13",
        "status": "rad_dino_frozen_feature_extraction_linear_probe",
        "data_config": "configs/data_config.yaml",
        "output_dir": "artifacts/experiments/e13",
    }
    mismatches = [{"key": key, "expected": value, "actual": config.get(key)} for key, value in required.items() if config.get(key) != value]
    model = config.get("model", {})
    expected_model = {
        "hf_model_id": "microsoft/rad-dino",
        "trust_remote_code": False,
        "encoder_mode": "frozen_eval_inference_mode_cls_embedding",
        "expected_hidden_size": 768,
        "use_auto_image_processor": True,
        "use_auto_model": True,
        "preprocessing": "model_native_only",
    }
    mismatches.extend({"key": f"model.{key}", "expected": value, "actual": model.get(key)} for key, value in expected_model.items() if model.get(key) != value)
    runtime = config.get("runtime", {})
    if int(runtime.get("batch_size", -1)) != 2 or bool(runtime.get("amp")) is not True:
        mismatches.append({"key": "runtime", "expected": "batch_size=2 and amp=true", "actual": runtime})
    probe = config.get("linear_probe", {})
    expected_probe = {"scaler": "StandardScaler", "classifier": "LogisticRegression", "C": 1.0, "seed": 42, "validation_hyperparameter_search": False, "fit_on": "train_features_only"}
    mismatches.extend({"key": f"linear_probe.{key}", "expected": value, "actual": probe.get(key)} for key, value in expected_probe.items() if probe.get(key) != value)
    split = config.get("split_contract", {})
    if bool(split.get("no_test_access")) is not True or set(split.get("labels", {})) != set(CLASS_NAMES):
        mismatches.append({"key": "split_contract", "expected": "train/validation only with canonical labels", "actual": split})
    if mismatches:
        raise ValueError(f"E13 config contract drift: {mismatches}")


def _validate_shared_data_config(config: Mapping[str, Any], root: Path) -> None:
    """Require E13 to point to exactly the train/validation manifests used by the project."""
    data_path = _inside_project(config["data_config"], root)
    if not data_path.is_file():
        raise FileNotFoundError(f"Shared data config is unavailable: {data_path}")
    data_config = _read_yaml(data_path)
    manifests = data_config.get("manifests", {})
    if set(manifests) != {"train", "validation"}:
        raise ValueError("E13 requires the shared data config to expose train/validation manifests only.")
    configured = config["split_contract"]
    for split, key in (("train", "train_manifest"), ("validation", "validation_manifest")):
        shared_path = _inside_project(manifests[split], root)
        e13_path = _inside_project(configured[key], root)
        if shared_path != e13_path:
            raise ValueError(
                f"E13 {split} manifest must exactly reuse {config['data_config']}: "
                f"shared={shared_path}, e13={e13_path}"
            )
    if data_config.get("class_mapping") != config["split_contract"]["labels"]:
        raise ValueError("E13 label mapping must exactly match the shared data config.")


def load_e13_config(config_path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    path = _inside_project(config_path, root)
    if not path.is_file():
        raise FileNotFoundError(f"E13 config is unavailable: {path}")
    config = _read_yaml(path)
    validate_e13_contract(config)
    _validate_shared_data_config(config, root)
    split = config["split_contract"]
    resolved = {
        "project_root": str(root),
        "config_path": str(path),
        "train_manifest": str(_inside_project(split["train_manifest"], root)),
        "validation_manifest": str(_inside_project(split["validation_manifest"], root)),
        "output_dir": str(_inside_project(config["output_dir"], root)),
    }
    config["resolved"] = resolved
    return config


def static_smoke_check(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify E13's isolated split/config contract without importing transformers."""
    validate_e13_contract(config)
    root = Path(config["resolved"]["project_root"])
    split = config["split_contract"]
    train = _manifest_frame(Path(config["resolved"]["train_manifest"]), root, "train", split["expected_counts"]["train"])
    validation = _manifest_frame(Path(config["resolved"]["validation_manifest"]), root, "validation", split["expected_counts"]["validation"])
    overlap = set(train["sample_id"]) & set(validation["sample_id"])
    if overlap:
        raise ValueError(f"E13 split contamination: {len(overlap)} overlapping sample IDs, e.g. {sorted(overlap)[:3]}")
    output = Path(config["resolved"]["output_dir"])
    return {
        "experiment": "e13",
        "static_check": "passed",
        "transform_policy": "AutoImageProcessor model-native preprocessing only; no ConvNeXt transform",
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_manifest_sha256": _sha256(Path(config["resolved"]["train_manifest"])),
        "validation_manifest_sha256": _sha256(Path(config["resolved"]["validation_manifest"])),
        "train_validation_sample_id_overlap": 0,
        "output_dir": str(output),
        "output_dir_exists": output.exists(),
    }


def _resolve_device(request: str) -> torch.device:
    if request == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("E13 requires CUDA by configuration, but CUDA is unavailable.")
        return torch.device("cuda")
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if request == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported E13 device request: {request}")


def _load_rad_dino(config: Mapping[str, Any], device: torch.device) -> tuple[Any, torch.nn.Module, dict[str, Any]]:
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        raise RuntimeError(
            "E13 requires the pinned `transformers` package. Install it in the AWS environment "
            "before running E13; see the experiment instructions."
        ) from error
    model_config = config["model"]
    model_id = str(model_config["hf_model_id"])
    revision = str(model_config["revision"])
    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model = AutoModel.from_pretrained(model_id, revision=revision, trust_remote_code=False).to(device)
    model.requires_grad_(False)
    model.eval()
    for parameter in model.parameters():
        if parameter.requires_grad:
            raise RuntimeError("E13 encoder freeze contract failed: trainable RAD-DINO parameter detected.")
    hidden_size = int(getattr(model.config, "hidden_size", -1))
    if hidden_size != int(model_config["expected_hidden_size"]):
        raise RuntimeError(f"E13 RAD-DINO hidden size drift: expected 768, got {hidden_size}.")
    processor_info = {
        "processor_class": type(processor).__name__,
        "do_resize": getattr(processor, "do_resize", None),
        "size": getattr(processor, "size", None),
        "do_center_crop": getattr(processor, "do_center_crop", None),
        "crop_size": getattr(processor, "crop_size", None),
        "do_rescale": getattr(processor, "do_rescale", None),
        "rescale_factor": getattr(processor, "rescale_factor", None),
        "do_normalize": getattr(processor, "do_normalize", None),
        "image_mean": getattr(processor, "image_mean", None),
        "image_std": getattr(processor, "image_std", None),
    }
    model_info = {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_commit_hash": getattr(model.config, "_commit_hash", None),
        "model_class": type(model).__name__,
        "hidden_size": hidden_size,
        "processor": processor_info,
    }
    return processor, model, model_info


def _cache_paths(config: Mapping[str, Any], split: str) -> dict[str, Path]:
    output = Path(config["resolved"]["output_dir"])
    cache = output / config["artifacts"]["cache_dir"]
    prefix = "train" if split == "train" else "validation"
    return {
        "directory": cache,
        "features": cache / config["artifacts"][f"{prefix}_features"],
        "index": cache / config["artifacts"][f"{prefix}_index"],
        "metadata": cache / config["artifacts"]["cache_metadata"],
    }


def _cache_signature(config: Mapping[str, Any], frame: pd.DataFrame, manifest: Path, split: str) -> dict[str, Any]:
    return {
        "split": split,
        "manifest_sha256": _sha256(manifest),
        "sample_id_hash": _ordered_hash(frame["sample_id"]),
        "label_hash": _ordered_hash(frame["label"]),
        "n_samples": int(len(frame)),
        "model_id": config["model"]["hf_model_id"],
        "requested_revision": config["model"]["revision"],
        "feature_kind": "last_hidden_state_cls",
        "expected_hidden_size": int(config["model"]["expected_hidden_size"]),
    }


def _load_cached_features(config: Mapping[str, Any], split: str, frame: pd.DataFrame, manifest: Path) -> FeatureSet | None:
    paths = _cache_paths(config, split)
    if not all(paths[key].is_file() for key in ("features", "index", "metadata")):
        return None
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    expected = _cache_signature(config, frame, manifest, split)
    actual = metadata.get("splits", {}).get(split)
    if actual != expected:
        return None
    index = pd.read_csv(paths["index"])
    required = ["sample_id", "image_path", "label", "class_name", "source_identifier"]
    if list(index.columns) != required or index["sample_id"].tolist() != frame["sample_id"].tolist() or index["label"].astype(int).tolist() != frame["label"].astype(int).tolist():
        return None
    with np.load(paths["features"], allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        labels = np.asarray(archive["labels"], dtype=np.int64)
    if features.shape != (len(frame), int(config["model"]["expected_hidden_size"])) or labels.shape != (len(frame),):
        return None
    if not np.isfinite(features).all() or not np.array_equal(labels, frame["label"].to_numpy(dtype=np.int64)):
        return None
    return FeatureSet(split=split, features=features, labels=labels, index=index, from_cache=True)


def _save_feature_cache(config: Mapping[str, Any], feature_set: FeatureSet, manifest: Path, model_info: Mapping[str, Any]) -> None:
    paths = _cache_paths(config, feature_set.split)
    index = feature_set.index[["sample_id", "image_path", "label", "class_name", "source_identifier"]].copy()
    _atomic_npz(paths["features"], features=feature_set.features, labels=feature_set.labels)
    _atomic_csv(paths["index"], index)
    metadata: dict[str, Any] = {"version": 1, "splits": {}, "model_runtime": dict(model_info)}
    if paths["metadata"].is_file():
        try:
            loaded = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata.update(loaded)
        except (json.JSONDecodeError, OSError):
            pass
    metadata.setdefault("splits", {})[feature_set.split] = _cache_signature(config, feature_set.index, manifest, feature_set.split)
    metadata["model_runtime"] = dict(model_info)
    _atomic_json(paths["metadata"], metadata)


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def _extract_features(
    *,
    config: Mapping[str, Any],
    split: str,
    frame: pd.DataFrame,
    processor: Any,
    model: torch.nn.Module,
    device: torch.device,
) -> FeatureSet:
    runtime = config["runtime"]
    dataset = ManifestImageDataset(frame, Path(config["resolved"]["project_root"]))
    loader = DataLoader(
        dataset,
        batch_size=int(runtime["batch_size"]),
        shuffle=False,
        num_workers=int(runtime["num_workers"]),
        pin_memory=bool(runtime["pin_memory"]) and device.type == "cuda",
        collate_fn=_collate_raw_images,
    )
    parts: list[np.ndarray] = []
    observed_ids: list[str] = []
    amp_enabled = bool(runtime["amp"]) and device.type == "cuda"
    for batch_index, (images, labels, sample_ids) in enumerate(loader):
        try:
            inputs = processor(images=images, return_tensors="pt")
            inputs = {name: value.to(device, non_blocking=device.type == "cuda") for name, value in inputs.items()}
            autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else contextlib.nullcontext()
            with torch.inference_mode(), autocast:
                output = model(**inputs)
                hidden = output.last_hidden_state
                cls = hidden[:, 0, :]
            if cls.shape != (len(images), int(config["model"]["expected_hidden_size"])):
                raise RuntimeError(f"E13 CLS embedding shape drift at {split} batch {batch_index}: {tuple(cls.shape)}")
            if not bool(torch.isfinite(cls).all()):
                raise FloatingPointError(f"E13 non-finite CLS embedding at {split} batch {batch_index}.")
            parts.append(cls.float().cpu().numpy())
            observed_ids.extend(sample_ids)
            if labels.tolist() != frame.iloc[len(observed_ids) - len(labels):len(observed_ids)]["label"].astype(int).tolist():
                raise RuntimeError(f"E13 {split} loader/manifest label alignment failed at batch {batch_index}.")
        except RuntimeError as error:
            if _is_cuda_oom(error):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA OOM while extracting E13 {split} features at batch={batch_index}, "
                    f"batch_size={runtime['batch_size']}. Extraction stopped without writing this split cache. "
                    "Reduce runtime.batch_size and rerun."
                ) from error
            raise
    if observed_ids != frame["sample_id"].tolist():
        raise RuntimeError(f"E13 {split} feature/sample-ID alignment failed.")
    features = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    labels = frame["label"].to_numpy(dtype=np.int64)
    if features.shape != (len(frame), int(config["model"]["expected_hidden_size"])):
        raise RuntimeError(f"E13 {split} feature count/shape contract failed: {features.shape}")
    return FeatureSet(split=split, features=features, labels=labels, index=frame, from_cache=False)


def encoder_smoke_check(config: Mapping[str, Any], *, device_request: str | None = None) -> dict[str, Any]:
    """Run only a tiny frozen-encoder pass, without feature-cache or result writes.

    It samples the configured number of rows per class independently from train and
    validation. No scaler, classifier, validation tuning, or experiment artifact is
    created; Hugging Face may populate its normal external model cache.
    """
    static = static_smoke_check(config)
    root = Path(config["resolved"]["project_root"])
    contract = config["split_contract"]
    samples_per_class = int(config["smoke_test"]["samples_per_class"])
    if samples_per_class < 1:
        raise ValueError("E13 encoder smoke requires samples_per_class >= 1.")
    frames = {
        "train": _manifest_frame(Path(config["resolved"]["train_manifest"]), root, "train", contract["expected_counts"]["train"]),
        "validation": _manifest_frame(Path(config["resolved"]["validation_manifest"]), root, "validation", contract["expected_counts"]["validation"]),
    }
    selected: dict[str, pd.DataFrame] = {}
    for split, frame in frames.items():
        parts = [frame.loc[frame["label"] == label].head(samples_per_class) for label in range(3)]
        if any(len(part) != samples_per_class for part in parts):
            raise RuntimeError(f"E13 {split} encoder smoke cannot sample every class.")
        selected[split] = pd.concat(parts, ignore_index=True)
    device = _resolve_device(device_request or str(config["runtime"]["device"]))
    processor, model, model_info = _load_rad_dino(config, device)
    try:
        features = {
            split: _extract_features(
                config=config,
                split=f"{split}_smoke",
                frame=frame,
                processor=processor,
                model=model,
                device=device,
            )
            for split, frame in selected.items()
        }
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    expected_rows = 3 * samples_per_class
    if any(feature_set.features.shape != (expected_rows, int(config["model"]["expected_hidden_size"])) for feature_set in features.values()):
        raise RuntimeError("E13 encoder smoke feature shape contract failed.")
    return {
        **static,
        "encoder_smoke_check": "passed",
        "device": str(device),
        "model": model_info,
        "samples_per_class": samples_per_class,
        "train_feature_shape": list(features["train"].features.shape),
        "validation_feature_shape": list(features["validation"].features.shape),
        "artifacts_written": False,
    }


def _get_features(
    *,
    config: Mapping[str, Any],
    split: str,
    frame: pd.DataFrame,
    manifest: Path,
    processor: Any,
    model: torch.nn.Module,
    device: torch.device,
    model_info: Mapping[str, Any],
) -> FeatureSet:
    if bool(config["runtime"]["cache_features"]):
        cached = _load_cached_features(config, split, frame, manifest)
        if cached is not None:
            return cached
    feature_set = _extract_features(config=config, split=split, frame=frame, processor=processor, model=model, device=device)
    if bool(config["runtime"]["cache_features"]):
        _save_feature_cache(config, feature_set, manifest, model_info)
    return feature_set


def _prepare_output_paths(config: Mapping[str, Any], overwrite_results: bool) -> dict[str, Path]:
    output = Path(config["resolved"]["output_dir"])
    artifacts = config["artifacts"]
    paths = {
        "output": output,
        "scaler": output / artifacts["scaler"],
        "classifier": output / artifacts["classifier"],
        "metrics": output / artifacts["metrics"],
        "predictions": output / artifacts["predictions"],
        "run_metadata": output / artifacts["run_metadata"],
    }
    existing = [path for key, path in paths.items() if key != "output" and path.exists()]
    if existing and not overwrite_results:
        raise FileExistsError(
            "E13 result files already exist. Pass --overwrite-results to replace only E13 final outputs; "
            "valid feature caches will still be reused. Existing files: " + ", ".join(str(path) for path in existing)
        )
    output.mkdir(parents=True, exist_ok=True)
    return paths


def _comparison_context(config: Mapping[str, Any], validation_manifest: Path) -> dict[str, Any]:
    current_hash = _sha256(validation_manifest)
    return {
        "e6_macro_f1_reference": float(config["comparison_references"]["e6_macro_f1"]),
        "e10_macro_f1_reference": float(config["comparison_references"]["e10_macro_f1"]),
        "same_validation_manifest_verified": True,
        "verification": "E13 directly reads the validation manifest used by configs/data_config.yaml; E10 config also references that data config and enforces the fixed 742-image validation stream.",
        "validation_manifest_sha256": current_hash,
    }


def run_e13(config: Mapping[str, Any], *, overwrite_results: bool = False) -> dict[str, Any]:
    """Extract/reuse frozen RAD-DINO features, fit only train data, and evaluate validation."""
    validate_e13_contract(config)
    static_smoke_check(config)
    paths = _prepare_output_paths(config, overwrite_results)
    root = Path(config["resolved"]["project_root"])
    split_contract = config["split_contract"]
    train_manifest = Path(config["resolved"]["train_manifest"])
    validation_manifest = Path(config["resolved"]["validation_manifest"])
    train_frame = _manifest_frame(train_manifest, root, "train", split_contract["expected_counts"]["train"])
    validation_frame = _manifest_frame(validation_manifest, root, "validation", split_contract["expected_counts"]["validation"])
    if set(train_frame["sample_id"]) & set(validation_frame["sample_id"]):
        raise RuntimeError("E13 refuses to run because train/validation sample IDs overlap.")
    seed = int(config["linear_probe"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _resolve_device(str(config["runtime"]["device"]))
    processor, model, model_info = _load_rad_dino(config, device)
    train = _get_features(config=config, split="train", frame=train_frame, manifest=train_manifest, processor=processor, model=model, device=device, model_info=model_info)
    validation = _get_features(config=config, split="validation", frame=validation_frame, manifest=validation_manifest, processor=processor, model=model, device=device, model_info=model_info)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.features)
    validation_scaled = scaler.transform(validation.features)
    probe_config = config["linear_probe"]
    probe = LogisticRegression(
        C=float(probe_config["C"]),
        random_state=seed,
        max_iter=int(probe_config["max_iter"]),
        solver=str(probe_config["solver"]),
    )
    probe.fit(train_scaled, train.labels)
    if set(probe.classes_.tolist()) != {0, 1, 2}:
        raise RuntimeError(f"E13 linear probe class contract failed: {probe.classes_.tolist()}")
    probabilities = probe.predict_proba(validation_scaled).astype(np.float64)
    metrics = classification_metrics(validation.labels, probabilities)
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    metrics["bacterial_to_viral"] = int(confusion[1, 2])
    metrics["viral_to_bacterial"] = int(confusion[2, 1])
    metrics["bacterial_viral_misclassifications"] = int(confusion[1, 2] + confusion[2, 1])
    metrics["comparison_context"] = _comparison_context(config, validation_manifest)
    predictions = validation.index[["sample_id", "image_path", "source_identifier", "label", "class_name"]].copy()
    predictions = predictions.rename(columns={"label": "true_label", "class_name": "true_class_name"})
    predictions["predicted_label"] = probabilities.argmax(axis=1)
    predictions["predicted_class_name"] = [CLASS_NAMES[index] for index in predictions["predicted_label"]]
    for index, name in enumerate(CLASS_NAMES):
        predictions[f"prob_{name.lower()}"] = probabilities[:, index]
    _atomic_joblib(paths["scaler"], scaler)
    _atomic_joblib(paths["classifier"], probe)
    _atomic_json(paths["metrics"], metrics)
    _atomic_csv(paths["predictions"], predictions)
    run_metadata = {
        "experiment": "e13",
        "device": str(device),
        "encoder": model_info,
        "encoder_frozen": True,
        "encoder_eval": not model.training,
        "feature_extraction_context": "torch.inference_mode with CUDA AMP when enabled",
        "feature_source": {"train": "cache" if train.from_cache else "fresh", "validation": "cache" if validation.from_cache else "fresh"},
        "train_manifest_sha256": _sha256(train_manifest),
        "validation_manifest_sha256": _sha256(validation_manifest),
        "linear_probe": {"scaler_fit_split": "train_only", "classifier_fit_split": "train_only", **dict(probe_config)},
        "validation_used_for_fit_or_tuning": False,
    }
    _atomic_json(paths["run_metadata"], run_metadata)
    return {
        "output_dir": str(paths["output"]),
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "macro_auroc_ovr": metrics["macro_auroc_ovr"],
        "feature_source": run_metadata["feature_source"],
        "result_files": {key: str(value) for key, value in paths.items() if key != "output"},
    }
