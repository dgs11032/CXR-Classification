"""Read-only diagnosis for completed E18 results.

The diagnostic reads only manifests, existing E15/E18 CSV/JSON/NPZ artifacts, and
prints a report. It never loads a model, writes files, trains, tunes, or reads test
data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .e15_e18_analysis import _align, _history, _metrics, _read_metrics, _verify_reported_metrics, load_config as load_analysis_config
from .e18_residual_fusion import load_e18_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_COLUMNS = ["sample_id", "image_path", "group_key", "source_identifier", "class_name", "canonical_label", "binary_label", "cache_label"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_hash(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _frames(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(Path(config["resolved"]["train_manifest"]))
    validation = pd.read_csv(Path(config["resolved"]["validation_manifest"]))
    required = {"image_path", "label", "class_name", "group_key", "source_identifier"}
    if required - set(train.columns) or required - set(validation.columns) or len(train) != 4448 or len(validation) != 742 or train["image_path"].duplicated().any() or validation["image_path"].duplicated().any():
        raise ValueError("E18 diagnostic manifest schema/count/image identity contract failed.")
    train = train[train["label"].isin([1, 2])].copy().reset_index(drop=True)
    if len(train) != 3300 or train["class_name"].value_counts().to_dict() != {"BACTERIAL_PNEUMONIA": 2153, "VIRAL_PNEUMONIA": 1147}:
        raise ValueError("E18 diagnostic B/V train subset contract failed.")
    train.insert(0, "sample_id", train["image_path"].astype(str))
    train["canonical_label"] = train["label"].astype(int)
    train["binary_label"] = train["canonical_label"] - 1
    validation = validation.copy().reset_index(drop=True)
    validation.insert(0, "sample_id", validation["image_path"].astype(str))
    validation["canonical_label"] = validation["label"].astype(int)
    validation["binary_label"] = validation["canonical_label"].map({1: 0, 2: 1}).fillna(-1).astype(int)
    if set(train["group_key"]) & set(validation["group_key"]):
        raise ValueError("E18 diagnostic detected train/validation group overlap.")
    return train, validation


def _cache_paths(config: Mapping[str, Any], split: str) -> dict[str, Path]:
    prefix = "train" if split == "train" else "validation"
    cache = Path(config["resolved"]["output_dir"]) / config["artifacts"]["cache_dir"]
    return {"features": cache / config["artifacts"][f"{prefix}_features"], "index": cache / config["artifacts"][f"{prefix}_index"], "metadata": cache / config["artifacts"][f"{prefix}_metadata"]}


def _cache_report(config: Mapping[str, Any], split: str, frame: pd.DataFrame, label_column: str) -> dict[str, Any]:
    paths = _cache_paths(config, split)
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"E18 {split} cache unavailable: missing={missing}")
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        index = pd.read_csv(paths["index"])
        with np.load(paths["features"], allow_pickle=False) as archive:
            e15_cls = archive["e15_cls"]
            e15_logits = archive["e15_logits"]
            cnn_features = archive["cnn_features"]
            labels = archive["labels"]
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"E18 {split} cache cannot be parsed read-only.") from error
    expected_shapes = {"e15_cls": (len(frame), 768), "e15_logits": (len(frame), 2), "cnn_features": (len(frame), 512), "labels": (len(frame),)}
    actual = {"e15_cls": tuple(e15_cls.shape), "e15_logits": tuple(e15_logits.shape), "cnn_features": tuple(cnn_features.shape), "labels": tuple(labels.shape)}
    if actual != expected_shapes or e15_cls.dtype != np.float16 or e15_logits.dtype != np.float32 or cnn_features.dtype != np.float16 or labels.dtype != np.int64 or not np.isfinite(e15_cls).all() or not np.isfinite(e15_logits).all() or not np.isfinite(cnn_features).all():
        raise ValueError(f"E18 {split} cache tensor shape/dtype/finiteness contract failed: {actual}")
    if list(index.columns) != _CACHE_COLUMNS or index["sample_id"].tolist() != frame["sample_id"].tolist() or index["image_path"].tolist() != frame["image_path"].tolist() or index["group_key"].tolist() != frame["group_key"].tolist() or index["cache_label"].astype(int).tolist() != frame[label_column].astype(int).tolist() or not np.array_equal(labels, frame[label_column].to_numpy(dtype=np.int64)):
        raise ValueError(f"E18 {split} cache index/manifest/label alignment failed.")
    manifest_path = Path(config["resolved"]["train_manifest"] if split == "train" else config["resolved"]["validation_manifest"])
    expected_hashes = {"manifest_sha256": _sha256(manifest_path), "sample_id_hash": _ordered_hash(frame["sample_id"]), "image_path_hash": _ordered_hash(frame["image_path"]), "label_hash": _ordered_hash(frame[label_column])}
    if any(metadata.get(key) != value for key, value in expected_hashes.items()) or metadata.get("sample_count") != len(frame) or metadata.get("split") != split or not metadata.get("e15_checkpoint_identity") or not metadata.get("cnn_identity"):
        raise ValueError(f"E18 {split} cache metadata identity/hash contract failed.")
    checkpoint = Path(config["resolved"]["e15_final"])
    checkpoint_hash_verified = checkpoint.is_file() and metadata["e15_checkpoint_identity"].get("sha256") == _sha256(checkpoint)
    logits_prediction = e15_logits.argmax(axis=1)
    return {"split": split, "tensor_shapes": {key: list(value) for key, value in actual.items()}, "e15_logit_train_metrics": {"accuracy": float(accuracy_score(labels, logits_prediction)), "macro_f1": float(f1_score(labels, logits_prediction, labels=[0, 1], average="macro", zero_division=0)), "confusion_matrix": confusion_matrix(labels, logits_prediction, labels=[0, 1]).tolist()}, "metadata_hashes_exact": True, "e15_checkpoint_hash_verified": checkpoint_hash_verified, "e15_checkpoint_file_present": checkpoint.is_file(), "cache_feature_values_finite": True}


def _binary_direction(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    return {"confusion_matrix": matrix.tolist(), "bacterial_to_viral": int(matrix[0, 1]), "viral_to_bacterial": int(matrix[1, 0]), "total_bv_errors": int(matrix[0, 1] + matrix[1, 0])}


def _prediction_report(analysis_config: Mapping[str, Any]) -> dict[str, Any]:
    manifest, e15, e18, alignment = _align(analysis_config)
    targets = manifest["label"].to_numpy(dtype=np.int64)
    e15_final = e15["e15_final_pred"].astype(int).to_numpy()
    e18_final = e18["e18_final_pred"].astype(int).to_numpy()
    e15_metrics, _ = _metrics(targets, e15_final)
    e18_metrics, _ = _metrics(targets, e18_final)
    bv = targets >= 1
    e15_binary = e15["e15_binary_pred"].astype(int).to_numpy()
    e18_binary = e18["e18_binary_pred"].astype(int).to_numpy()
    e15_bv = _binary_direction(targets[bv] - 1, e15_binary[bv])
    e18_bv = _binary_direction(targets[bv] - 1, e18_binary[bv])
    e15_correct, e18_correct = e15_final == targets, e18_final == targets
    changes = {"e15_only_correct": int((e15_correct & ~e18_correct).sum()), "e18_only_correct": int((~e15_correct & e18_correct).sum()), "both_correct": int((e15_correct & e18_correct).sum()), "both_wrong": int((~e15_correct & ~e18_correct).sum()), "changed_final_prediction": int((e15_final != e18_final).sum())}
    if sum(changes[key] for key in ("e15_only_correct", "e18_only_correct", "both_correct", "both_wrong")) != 742:
        raise RuntimeError("Prediction correctness partition does not sum to 742.")
    source_metrics = {"e15": _read_metrics(Path(analysis_config["resolved"]["e15_metrics"]), experiment="e15"), "e18": _read_metrics(Path(analysis_config["resolved"]["e18_metrics"]), experiment="e18")}
    _verify_reported_metrics(source_metrics["e15"], e15_metrics, experiment="e15")
    _verify_reported_metrics(source_metrics["e18"], e18_metrics, experiment="e18")
    return {"alignment": alignment, "e15_full_validation": e15_metrics, "e18_full_validation": e18_metrics, "e15_true_bv": e15_bv, "e18_true_bv": e18_bv, "direction_delta_e18_minus_e15": {"bacterial_to_viral": e18_bv["bacterial_to_viral"] - e15_bv["bacterial_to_viral"], "viral_to_bacterial": e18_bv["viral_to_bacterial"] - e15_bv["viral_to_bacterial"]}, "correctness_partition": changes, "source_metric_experiments": {key: value["experiment"] for key, value in source_metrics.items()}, "reported_metrics_match_prediction_csv": True}


def _history_report(analysis_config: Mapping[str, Any]) -> dict[str, Any]:
    history = _history(Path(analysis_config["resolved"]["e18_history"]), analysis_config)
    return {"definition": "online, shuffled, dropout-active training-pass metrics; not a separate post-epoch eval-mode train inference", **history}


def static_smoke_check(e18_config: Mapping[str, Any], analysis_config: Mapping[str, Any]) -> dict[str, Any]:
    train, validation = _frames(e18_config)
    expected = {"e15_predictions": Path(analysis_config["resolved"]["e15_predictions"]).is_file(), "e18_predictions": Path(analysis_config["resolved"]["e18_predictions"]).is_file(), "e18_history": Path(analysis_config["resolved"]["e18_history"]).is_file(), "train_cache": all(path.is_file() for path in _cache_paths(e18_config, "train").values()), "validation_cache": all(path.is_file() for path in _cache_paths(e18_config, "validation").values())}
    return {"diagnostic": "e18_readonly_result_diagnosis", "static_check": "passed", "train_bv_samples": len(train), "validation_samples": len(validation), "train_validation_group_overlap": 0, "source_artifacts_available": expected, "writes_files": False, "loads_models": False, "accesses_test_data": False}


def run_diagnosis(e18_config: Mapping[str, Any], analysis_config: Mapping[str, Any]) -> dict[str, Any]:
    static_smoke_check(e18_config, analysis_config)
    train, validation = _frames(e18_config)
    train_cache = _cache_report(e18_config, "train", train, "binary_label")
    validation_cache = _cache_report(e18_config, "validation", validation, "label")
    return {"diagnostic": "e18_readonly_result_diagnosis", "source_files_read_only": True, "writes_files": False, "loads_models": False, "accesses_test_data": False, "online_train_metric_interpretation": "E18 history predictions are collected batchwise under head.train() with active Dropout(0.2), while parameters change over the epoch; they are not post-epoch train inference metrics.", "cache_alignment": {"train": train_cache, "validation": validation_cache}, "predictions": _prediction_report(analysis_config), "e18_train_history": _history_report(analysis_config)}
