"""E10-only hard E6/E9 hierarchical validation evaluation with traceable sample alignment."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataloader import build_dataset, build_loader, load_data_config

from .checkpoints import load_checkpoint
from .constants import CLASS_NAMES
from .e6_models import build_e6_model
from .e6_training import load_e6_training_config
from .e8_data import BINARY_CLASS_ORDER, EXPECTED_COUNTS, build_e8_dataset
from .e9_models import build_e9_model
from .e9_training import load_e9_training_config
from .metrics import classification_metrics
from .training import PROJECT_ROOT, file_sha256, resolve_device

E6_CLASS_ORDER = tuple(CLASS_NAMES)
E9_CLASS_ORDER = tuple(BINARY_CLASS_ORDER)


def _inside_project(path: str | Path, root: Path) -> Path:
    candidate = Path(path).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"E10 path must remain inside project root: {path}")
    return resolved


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"E10 config must be a YAML mapping: {path}")
    return payload


def validate_e10_evaluation_contract(config: Mapping[str, Any]) -> None:
    expected = {
        "experiment": "e10", "status": "hard_hierarchical_e6_e9_validation_evaluation",
        "mode": "hard_routing_no_training", "data_config": "configs/data_config.yaml",
        "e6_config": "configs/e6_training.yaml", "e9_config": "configs/e9_training.yaml",
        "e6_checkpoint": "artifacts/experiments/e6/best.pt", "e9_checkpoint": "artifacts/experiments/e9/best.pt",
        "output_dir": "artifacts/experiments/e10",
    }
    mismatches = [{"key": key, "expected": value, "actual": config.get(key)} for key, value in expected.items() if config.get(key) != value]
    routing = config.get("routing", {})
    expected_routing = {
        "e6_class_order": list(E6_CLASS_ORDER), "e9_class_order": list(E9_CLASS_ORDER),
        "rule": "e6_normal_kept_else_e9_binary_plus_one", "allow_threshold_tuning": False,
        "allow_probability_averaging": False, "allow_calibration": False, "allow_soft_fusion": False,
    }
    mismatches.extend({"key": f"routing.{key}", "expected": value, "actual": routing.get(key)} for key, value in expected_routing.items() if routing.get(key) != value)
    validation = config.get("validation", {})
    expected_validation = {"expected_total_samples": 742, "expected_bacterial_viral_samples": 550, "expected_bacterial_count": 359, "expected_viral_count": 191, "deterministic_single_view": True, "no_test_access": True}
    mismatches.extend({"key": f"validation.{key}", "expected": value, "actual": validation.get(key)} for key, value in expected_validation.items() if validation.get(key) != value)
    if mismatches:
        raise ValueError(f"E10 evaluation contract drift: {mismatches}")


def load_e10_evaluation_config(config_path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    path = _inside_project(config_path, root)
    if not path.is_file():
        raise FileNotFoundError(f"E10 config is unavailable: {path}")
    config = _read_yaml(path)
    validate_e10_evaluation_contract(config)
    resolved = {key: _inside_project(config[key], root) for key in ("data_config", "e6_config", "e9_config", "e6_checkpoint", "e9_checkpoint", "output_dir")}
    config["resolved"] = {"project_root": str(root), "config_path": str(path), **{key: str(value) for key, value in resolved.items()}}
    return config


def hard_route(e6_predictions: np.ndarray, e9_predictions: np.ndarray) -> np.ndarray:
    """Keep E6 Normal=0; map E9 Bacterial/Viral=0/1 to canonical 1/2 otherwise."""
    e6_predictions = np.asarray(e6_predictions, dtype=np.int64)
    e9_predictions = np.asarray(e9_predictions, dtype=np.int64)
    if e6_predictions.ndim != 1 or e9_predictions.shape != e6_predictions.shape:
        raise ValueError("E10 routing requires aligned one-dimensional E6/E9 predictions.")
    if set(e6_predictions.tolist()) - {0, 1, 2} or set(e9_predictions.tolist()) - {0, 1}:
        raise ValueError("E10 routing received invalid class indices.")
    return np.where(e6_predictions == 0, 0, e9_predictions + 1).astype(np.int64)


def _validate_checkpoint_contracts(
    e6_payload: Mapping[str, Any],
    e9_payload: Mapping[str, Any],
    *,
    e6_checkpoint: Path,
    validation_fingerprint: str,
) -> dict[str, Any]:
    if e6_payload.get("run_kind") != "e6_full_epoch" or e6_payload.get("config", {}).get("experiment") != "e6":
        raise ValueError("E10 requires an E6 best checkpoint with run_kind=e6_full_epoch.")
    if e9_payload.get("run_kind") != "e9_binary_full_epoch" or e9_payload.get("config", {}).get("experiment") != "e9":
        raise ValueError("E10 requires an E9 best checkpoint with run_kind=e9_binary_full_epoch.")
    e6_control, e9_control = e6_payload.get("control_state", {}), e9_payload.get("control_state", {})
    e6_fingerprint = e6_control.get("manifest_fingerprints", {}).get("validation")
    e9_fingerprint = e9_control.get("manifest_fingerprints", {}).get("validation")
    if e6_fingerprint != validation_fingerprint or e9_fingerprint != validation_fingerprint or e6_fingerprint != e9_fingerprint:
        raise ValueError("E6/E9/current validation manifest fingerprints must match exactly.")
    if e9_control.get("binary_class_order") != list(E9_CLASS_ORDER) or e9_control.get("canonical_to_binary_labels") != {"1": 0, "2": 1}:
        raise ValueError("E9 binary class order or canonical mapping contract mismatch.")
    if e9_control.get("filtered_source_counts") != EXPECTED_COUNTS:
        raise ValueError("E9 fixed Bacterial/Viral source-count contract mismatch.")
    transfer = e9_control.get("e6_encoder_transfer", {})
    if not transfer.get("all_encoder_tensors_equal") or transfer.get("transferred_encoder_tensor_count") != 180:
        raise ValueError("E9 checkpoint lacks verified E6 180-tensor encoder-transfer provenance.")
    saved_source = Path(str(transfer.get("source_checkpoint", ""))).expanduser().resolve()
    if saved_source != e6_checkpoint.resolve():
        raise ValueError(f"E9 source checkpoint provenance differs from selected E6 checkpoint: {saved_source}")
    if int(transfer.get("source_epoch", -1)) != int(e6_payload.get("epoch", -2)):
        raise ValueError("E9 transfer source epoch does not match selected E6 checkpoint epoch.")
    if not np.isclose(float(transfer.get("source_best_validation_macro_f1", np.nan)), float(e6_payload.get("best_validation_macro_f1", np.nan)), rtol=0.0, atol=1e-12):
        raise ValueError("E9 transfer source best Macro F1 does not match selected E6 checkpoint.")
    return {"validation_manifest_fingerprint": validation_fingerprint, "e6_epoch": int(e6_payload["epoch"]), "e9_epoch": int(e9_payload["epoch"]), "e6_e9_checkpoint_provenance_match": True}


def load_e10_models(config: Mapping[str, Any], *, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    root = Path(config["resolved"]["project_root"])
    e6_checkpoint, e9_checkpoint = Path(config["resolved"]["e6_checkpoint"]), Path(config["resolved"]["e9_checkpoint"])
    if not e6_checkpoint.is_file() or not e9_checkpoint.is_file():
        raise FileNotFoundError(f"E10 requires both best checkpoints: E6={e6_checkpoint.is_file()}, E9={e9_checkpoint.is_file()}")
    e6_config = load_e6_training_config(config["resolved"]["e6_config"], project_root=root)
    e9_config = load_e9_training_config(config["resolved"]["e9_config"], project_root=root)
    e6_model, e9_model = build_e6_model(e6_config).to(device), build_e9_model(e9_config).to(device)
    e6_payload = load_checkpoint(e6_checkpoint, model=e6_model, map_location=device)
    e9_payload = load_checkpoint(e9_checkpoint, model=e9_model, map_location=device)
    data_config = load_data_config(config["resolved"]["data_config"], project_root=root)
    fingerprint = file_sha256(data_config["resolved"]["validation_manifest"])
    contracts = _validate_checkpoint_contracts(e6_payload, e9_payload, e6_checkpoint=e6_checkpoint, validation_fingerprint=fingerprint)
    return e6_model.eval(), e9_model.eval(), {"e6_payload": e6_payload, "e9_payload": e9_payload, "contracts": contracts}


def validation_alignment(config: Mapping[str, Any]) -> tuple[Any, DataLoader, pd.DataFrame, dict[str, Any]]:
    root = Path(config["resolved"]["project_root"])
    data_config = load_data_config(config["resolved"]["data_config"], project_root=root)
    if data_config["views"]["validation"] != "single" or bool(data_config["loader"]["validation_shuffle"]) or "test" in data_config.get("manifests", {}):
        raise ValueError("E10 requires deterministic single-view, unshuffled validation only, with no test manifest.")
    dataset = build_dataset(data_config, "validation", view_mode="single")
    loader = build_loader(dataset, data_config, training=False)
    frame = dataset.frame.copy().reset_index(drop=True)
    if len(frame) != 742 or frame["image_path"].duplicated().any():
        raise ValueError("E10 requires the fixed 742-image validation manifest with unique image_path identifiers.")
    subtype = frame[frame["label"].isin([1, 2])].copy().reset_index(drop=True)
    expected_counts = {"BACTERIAL_PNEUMONIA": 359, "VIRAL_PNEUMONIA": 191}
    counts = subtype["class_name"].value_counts().to_dict()
    if len(subtype) != 550 or counts != expected_counts:
        raise ValueError(f"E10 B/V subset count drift: {counts}")
    e9_subset = build_e8_dataset(data_config, "validation").frame
    if subtype["image_path"].tolist() != e9_subset["image_path"].tolist() or (subtype["label"].to_numpy() - 1 != e9_subset["label"].to_numpy()).any():
        raise ValueError("E10 full-manifest B/V filter does not exactly align with E9's fixed subset order/mapping.")
    return dataset, loader, frame, {"full_samples": len(frame), "subtype_samples": len(subtype), "subtype_counts": counts, "e9_subset_alignment": True}


def _hard_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    probabilities = np.eye(3, dtype=np.float64)[predictions]
    metrics = classification_metrics(targets, probabilities)
    metrics["macro_auroc_ovr"] = None
    metrics["auroc_not_probability_comparable"] = True
    for values in metrics["per_class"].values():
        values["auroc_ovr"] = None
    return metrics


def _subtype_error_counts(metrics: Mapping[str, Any]) -> dict[str, int]:
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    return {"bacterial_to_viral": int(confusion[1, 2]), "viral_to_bacterial": int(confusion[2, 1]), "bv_total_errors": int(confusion[1, 2] + confusion[2, 1])}


def _analysis(targets: np.ndarray, e6_pred: np.ndarray, e9_pred: np.ndarray, final_pred: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    e6_correct, final_correct = e6_pred == targets, final_pred == targets
    changed = e6_pred != final_pred
    corrected = (~e6_correct) & final_correct
    damaged = e6_correct & (~final_correct)
    changed_wrong = changed & (~e6_correct) & (~final_correct)
    status = np.where(corrected, "corrected", np.where(damaged, "damaged", np.where(changed_wrong, "changed_wrong_to_wrong", np.where(e6_correct, "unchanged_correct", "unchanged_wrong"))))
    normal_misrouted = (targets == 0) & (e6_pred != 0)
    analysis = {
        "changed_final_predictions": int(changed.sum()), "e6_wrong_to_e10_correct": int(corrected.sum()),
        "e6_correct_to_e10_wrong": int(damaged.sum()), "e6_wrong_to_e10_still_wrong_different_class": int(changed_wrong.sum()),
        "true_bacterial_e6_viral_corrected": int(((targets == 1) & (e6_pred == 2) & (final_pred == 1)).sum()),
        "true_bacterial_e6_correct_damaged_to_viral": int(((targets == 1) & (e6_pred == 1) & (final_pred == 2)).sum()),
        "true_viral_e6_bacterial_corrected": int(((targets == 2) & (e6_pred == 1) & (final_pred == 2)).sum()),
        "true_viral_e6_correct_damaged_to_bacterial": int(((targets == 2) & (e6_pred == 2) & (final_pred == 1)).sum()),
        "true_normal_e6_predicted_pneumonia_cannot_recover_normal": int(normal_misrouted.sum()),
        "true_normal_misrouted_subtype_changed_by_e9": int((normal_misrouted & changed).sum()),
        "routed_to_e9": int((e6_pred != 0).sum()), "normal_bypassed_e9": int((e6_pred == 0).sum()),
        "net_correction": int(corrected.sum() - damaged.sum()),
        "correct_count_delta": int(final_correct.sum() - e6_correct.sum()),
    }
    if analysis["net_correction"] != analysis["correct_count_delta"]:
        raise RuntimeError("E10 net correction identity failed.")
    return analysis, status


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def evaluate_e10_hierarchical(config: Mapping[str, Any], *, device_request: str = "auto", overwrite: bool = False) -> dict[str, Any]:
    validate_e10_evaluation_contract(config)
    device = resolve_device(device_request)
    output_dir = Path(config["resolved"]["output_dir"])
    report_path = output_dir / config["output"]["report_filename"]
    trace_path = output_dir / config["output"]["predictions_filename"]
    if not overwrite and (report_path.exists() or trace_path.exists()):
        raise FileExistsError("E10 output exists; pass --overwrite to replace E10-only report/trace files.")
    e6_model, e9_model, checkpoint_info = load_e10_models(config, device=device)
    _, loader, frame, alignment = validation_alignment(config)
    e6_parts: list[torch.Tensor] = []
    e9_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    offset = 0
    with torch.inference_mode():
        for images, labels in loader:
            expected = frame.iloc[offset:offset + len(labels)]
            if len(expected) != len(labels) or labels.tolist() != expected["label"].astype(int).tolist():
                raise RuntimeError(f"E10 manifest/batch alignment failed at offset={offset}.")
            e6_logits, e9_logits = e6_model(images.to(device)), e9_model(images.to(device))
            if e6_logits.shape != (len(labels), 3) or e9_logits.shape != (len(labels), 2) or not bool(torch.isfinite(e6_logits).all()) or not bool(torch.isfinite(e9_logits).all()):
                raise FloatingPointError(f"Invalid E10 logits at offset={offset}: E6={tuple(e6_logits.shape)}, E9={tuple(e9_logits.shape)}")
            e6_parts.append(e6_logits.float().cpu()); e9_parts.append(e9_logits.float().cpu()); target_parts.append(labels.cpu())
            offset += len(labels)
    if offset != len(frame): raise RuntimeError("E10 validation did not consume every manifest row.")
    targets = torch.cat(target_parts).numpy()
    e6_probabilities = torch.softmax(torch.cat(e6_parts), dim=1).numpy()
    e9_probabilities = torch.softmax(torch.cat(e9_parts), dim=1).numpy()
    e6_pred, e9_pred = e6_probabilities.argmax(axis=1), e9_probabilities.argmax(axis=1)
    final_pred = hard_route(e6_pred, e9_pred)
    if not bool(np.all(final_pred[e6_pred == 0] == 0)) or not bool(np.all(np.isin(final_pred[e6_pred != 0], [1, 2]))):
        raise RuntimeError("E10 hard routing invariant failed.")
    e6_metrics, e10_metrics = classification_metrics(targets, e6_probabilities), _hard_metrics(targets, final_pred)
    e6_errors, e10_errors = _subtype_error_counts(e6_metrics), _subtype_error_counts(e10_metrics)
    baseline = config["baseline_sanity"]
    baseline_ok = abs(float(e6_metrics["macro_f1"]) - float(baseline["expected_macro_f1"])) <= float(baseline["macro_f1_tolerance"]) and e6_errors["bacterial_to_viral"] == int(baseline["expected_bacterial_to_viral"]) and e6_errors["viral_to_bacterial"] == int(baseline["expected_viral_to_bacterial"])
    if bool(baseline["require_match"]) and not baseline_ok:
        raise RuntimeError(f"E6 baseline sanity mismatch: metrics={e6_metrics['macro_f1']}, errors={e6_errors}")
    analysis, status = _analysis(targets, e6_pred, e9_pred, final_pred)
    trace = frame[["image_path", "group_key", "source_split", "class_name", "label"]].copy()
    trace = trace.rename(columns={"label": "true_label", "class_name": "true_class_name"})
    trace["e6_pred"], trace["e9_pred"], trace["final_pred"] = e6_pred, e9_pred, final_pred
    trace["e6_pred_name"] = [E6_CLASS_ORDER[index] for index in e6_pred]
    trace["e9_pred_name"] = [E9_CLASS_ORDER[index] for index in e9_pred]
    trace["final_pred_name"] = [E6_CLASS_ORDER[index] for index in final_pred]
    trace["e6_correct"], trace["final_correct"], trace["changed_by_e9"], trace["correction_status"] = e6_pred == targets, final_pred == targets, e6_pred != final_pred, status
    trace["e6_prob_normal"], trace["e6_prob_bacterial"], trace["e6_prob_viral"] = e6_probabilities[:, 0], e6_probabilities[:, 1], e6_probabilities[:, 2]
    trace["e9_prob_bacterial"], trace["e9_prob_viral"] = e9_probabilities[:, 0], e9_probabilities[:, 1]
    report = {"experiment": "e10", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "routing": dict(config["routing"]), "checkpoint_contracts": checkpoint_info["contracts"], "alignment": alignment, "e6_baseline": {"metrics": e6_metrics, "subtype_errors": e6_errors, "sanity_passed": baseline_ok}, "e10_hierarchical": {"metrics": e10_metrics, "subtype_errors": e10_errors}, "change_analysis": analysis, "hard_routing_note": "E10 AUROC is intentionally not reported because final predictions are hard routed labels, not calibrated 3-class probabilities."}
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    _write_csv(trace_path, trace)
    return {"console_summary": {"e6_accuracy": e6_metrics["accuracy"], "e6_macro_f1": e6_metrics["macro_f1"], "e6_bv_total_errors": e6_errors["bv_total_errors"], "e10_accuracy": e10_metrics["accuracy"], "e10_macro_f1": e10_metrics["macro_f1"], "e10_bv_total_errors": e10_errors["bv_total_errors"], "change_analysis": analysis, "report": str(report_path), "predictions": str(trace_path)}, "report": report, "trace": trace}
