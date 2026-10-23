"""E11-only exploratory confidence-gated E6/E9 subtype-routing sensitivity analysis."""

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

from .e10_hierarchical import (
    E6_CLASS_ORDER,
    E9_CLASS_ORDER,
    _analysis,
    _hard_metrics,
    _subtype_error_counts,
    _write_csv,
    _write_json,
    hard_route,
    load_e10_evaluation_config,
    load_e10_models,
    validation_alignment,
)
from .metrics import classification_metrics
from .training import PROJECT_ROOT, resolve_device

THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
EXPLORATORY_NOTE = (
    "Threshold analysis is exploratory and performed on the validation set. "
    "The best validation threshold is not an unbiased final performance estimate."
)


def _inside_project(path: str | Path, root: Path) -> Path:
    candidate = Path(path).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"E11 path must remain inside project root: {path}")
    return resolved


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"E11 config must be a YAML mapping: {path}")
    return payload


def validate_e11_analysis_contract(config: Mapping[str, Any]) -> None:
    expected = {"experiment": "e11", "status": "exploratory_validation_confidence_gated_e6_e9_routing", "mode": "threshold_sensitivity_no_training", "e10_evaluation_config": "configs/e10_evaluation.yaml", "output_dir": "artifacts/experiments/e11", "exploratory_note": EXPLORATORY_NOTE}
    mismatches = [{"key": key, "expected": value, "actual": config.get(key)} for key, value in expected.items() if config.get(key) != value]
    routing = config.get("routing", {})
    expected_routing = {"e6_class_order": list(E6_CLASS_ORDER), "e9_class_order": list(E9_CLASS_ORDER), "rule": "override_e6_subtype_only_on_disagreement_and_e9_confidence_gte_threshold", "confidence": "max_softmax_e9_binary_logits", "thresholds": list(THRESHOLDS), "allow_threshold_optimization": False, "allow_probability_averaging": False, "allow_calibration": False, "allow_temperature_scaling": False, "allow_soft_fusion": False}
    mismatches.extend({"key": f"routing.{key}", "expected": value, "actual": routing.get(key)} for key, value in expected_routing.items() if routing.get(key) != value)
    validation = config.get("validation", {})
    for key, value in {"expected_total_samples": 742, "expected_bacterial_viral_samples": 550, "deterministic_single_view": True, "no_test_access": True}.items():
        if validation.get(key) != value: mismatches.append({"key": f"validation.{key}", "expected": value, "actual": validation.get(key)})
    if mismatches: raise ValueError(f"E11 analysis contract drift: {mismatches}")


def load_e11_analysis_config(config_path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    path = _inside_project(config_path, root)
    if not path.is_file(): raise FileNotFoundError(f"E11 config unavailable: {path}")
    config = _read_yaml(path)
    validate_e11_analysis_contract(config)
    e10_path, output_path = _inside_project(config["e10_evaluation_config"], root), _inside_project(config["output_dir"], root)
    e10_config = load_e10_evaluation_config(e10_path, project_root=root)
    config["resolved"] = {"project_root": str(root), "config_path": str(path), "e10_evaluation_config_path": str(e10_path), "output_dir": str(output_path), "e10_config": e10_config}
    return config


def confidence_gate(e6_predictions: np.ndarray, e9_predictions: np.ndarray, e9_confidence: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Override only E6/E9 subtype disagreements whose E9 max-softmax reaches threshold."""
    e6_predictions, e9_predictions, e9_confidence = np.asarray(e6_predictions, dtype=np.int64), np.asarray(e9_predictions, dtype=np.int64), np.asarray(e9_confidence, dtype=np.float64)
    if e6_predictions.ndim != 1 or e9_predictions.shape != e6_predictions.shape or e9_confidence.shape != e6_predictions.shape: raise ValueError("E11 gate requires aligned one-dimensional vectors.")
    if not 0.5 <= threshold <= 1.0 or set(e6_predictions.tolist()) - {0, 1, 2} or set(e9_predictions.tolist()) - {0, 1}: raise ValueError("E11 gate received invalid threshold or class values.")
    if not np.isfinite(e9_confidence).all() or (e9_confidence < 0.5 - 1e-7).any() or (e9_confidence > 1.0).any(): raise ValueError("E11 E9 argmax confidence must be finite within [0.5,1].")
    e9_canonical = e9_predictions + 1
    disagreement = (e6_predictions != 0) & (e6_predictions != e9_canonical)
    override = disagreement & (e9_confidence >= threshold)
    return np.where(override, e9_canonical, e6_predictions).astype(np.int64), override


def _confidence_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0: return {"count": 0, "mean": None, "median": None, "min": None, "q25": None, "q75": None, "max": None}
    return {"count": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)), "min": float(values.min()), "q25": float(np.quantile(values, 0.25)), "q75": float(np.quantile(values, 0.75)), "max": float(values.max())}


def _collect_once(config: Mapping[str, Any], *, device: torch.device) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    e10_config = config["resolved"]["e10_config"]
    e6_model, e9_model, checkpoint_info = load_e10_models(e10_config, device=device)
    _, loader, frame, alignment = validation_alignment(e10_config)
    e6_parts: list[torch.Tensor] = []; e9_parts: list[torch.Tensor] = []; target_parts: list[torch.Tensor] = []; offset = 0
    with torch.inference_mode():
        for images, labels in loader:
            expected = frame.iloc[offset:offset + len(labels)]
            if len(expected) != len(labels) or labels.tolist() != expected["label"].astype(int).tolist(): raise RuntimeError(f"E11 manifest/batch alignment failed at offset={offset}.")
            e6_logits, e9_logits = e6_model(images.to(device)), e9_model(images.to(device))
            if e6_logits.shape != (len(labels), 3) or e9_logits.shape != (len(labels), 2) or not bool(torch.isfinite(e6_logits).all()) or not bool(torch.isfinite(e9_logits).all()): raise FloatingPointError(f"Invalid E11 logits at offset={offset}.")
            e6_parts.append(e6_logits.float().cpu()); e9_parts.append(e9_logits.float().cpu()); target_parts.append(labels.cpu()); offset += len(labels)
    if offset != len(frame): raise RuntimeError("E11 did not consume all aligned validation rows.")
    return frame, torch.cat(target_parts).numpy(), torch.softmax(torch.cat(e6_parts), dim=1).numpy(), torch.softmax(torch.cat(e9_parts), dim=1).numpy(), {"checkpoint_contracts": checkpoint_info["contracts"], "alignment": alignment}


def _threshold_row(targets: np.ndarray, e6_pred: np.ndarray, e9_pred: np.ndarray, confidence: np.ndarray, threshold: float, hard_corrected: np.ndarray, hard_damaged: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    final, override = confidence_gate(e6_pred, e9_pred, confidence, threshold)
    metrics, errors = _hard_metrics(targets, final), _subtype_error_counts(_hard_metrics(targets, final))
    analysis, _ = _analysis(targets, e6_pred, e9_pred, final)
    row = {"threshold": threshold, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "bacterial_to_viral": errors["bacterial_to_viral"], "viral_to_bacterial": errors["viral_to_bacterial"], "bv_total_errors": errors["bv_total_errors"], "changed_from_e6": analysis["changed_final_predictions"], "corrected": analysis["e6_wrong_to_e10_correct"], "damaged": analysis["e6_correct_to_e10_wrong"], "changed_wrong_to_wrong": analysis["e6_wrong_to_e10_still_wrong_different_class"], "net_correction": analysis["net_correction"], "true_bacterial_corrected": analysis["true_bacterial_e6_viral_corrected"], "true_bacterial_damaged": analysis["true_bacterial_e6_correct_damaged_to_viral"], "true_viral_corrected": analysis["true_viral_e6_bacterial_corrected"], "true_viral_damaged": analysis["true_viral_e6_correct_damaged_to_bacterial"], "override_count": int(override.sum()), "corrected_retained_from_e10": int((hard_corrected & (final == targets)).sum()), "damage_avoided_from_e10": int((hard_damaged & (final == e6_pred)).sum())}
    if row["net_correction"] != analysis["correct_count_delta"]: raise RuntimeError("E11 threshold net-correction identity failed.")
    return row, final, override


def evaluate_e11_confidence_routing(config: Mapping[str, Any], *, device_request: str = "auto", overwrite: bool = False) -> dict[str, Any]:
    validate_e11_analysis_contract(config)
    output_dir = Path(config["resolved"]["output_dir"]); output = config["output"]
    paths = {key: output_dir / output[key] for key in ("threshold_csv", "confidence_json", "analysis_json", "trace_csv")}
    if not overwrite and any(path.exists() for path in paths.values()): raise FileExistsError("E11 output exists; pass --overwrite to replace E11-only analysis files.")
    frame, targets, e6_prob, e9_prob, provenance = _collect_once(config, device=resolve_device(device_request))
    e6_pred, e9_pred, e9_canonical = e6_prob.argmax(axis=1), e9_prob.argmax(axis=1), e9_prob.argmax(axis=1) + 1
    confidence = e9_prob.max(axis=1)
    hard_pred = hard_route(e6_pred, e9_pred)
    threshold_rows: list[dict[str, Any]] = []
    hard_analysis, hard_status = _analysis(targets, e6_pred, e9_pred, hard_pred)
    hard_corrected = (e6_pred != targets) & (hard_pred == targets)
    hard_damaged = (e6_pred == targets) & (hard_pred != targets)
    threshold_predictions: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    previous_override: int | None = None
    for threshold in THRESHOLDS:
        row, final, override = _threshold_row(targets, e6_pred, e9_pred, confidence, threshold, hard_corrected, hard_damaged)
        if previous_override is not None and row["override_count"] > previous_override: raise RuntimeError("E11 override count increased at a higher threshold.")
        previous_override = int(row["override_count"]); threshold_rows.append(row); threshold_predictions[threshold] = (final, override)
    final_050, override_050 = threshold_predictions[0.50]
    hard_equivalent = bool(np.array_equal(final_050, hard_pred))
    if not hard_equivalent: raise RuntimeError("E11 threshold=0.50 must exactly equal E10 hard routing.")
    sanity = config["e10_hard_sanity"]
    hard_sanity_ok = hard_analysis["changed_final_predictions"] == int(sanity["expected_changed_predictions"]) and hard_analysis["e6_wrong_to_e10_correct"] == int(sanity["expected_corrected"]) and hard_analysis["e6_correct_to_e10_wrong"] == int(sanity["expected_damaged"]) and hard_analysis["net_correction"] == int(sanity["expected_net_correction"])
    if bool(sanity["require_match"]) and not hard_sanity_ok: raise RuntimeError(f"E11 E10 hard-routing sanity mismatch: {hard_analysis}")
    disagreement = (e6_pred != 0) & (e6_pred != e9_canonical)
    confidence_groups = {"corrected": _confidence_summary(confidence[hard_corrected]), "damaged": _confidence_summary(confidence[hard_damaged]), "all_e10_changed": _confidence_summary(confidence[e6_pred != hard_pred]), "eligible_subtype_disagreements": _confidence_summary(confidence[disagreement])}
    trace = frame[["image_path", "group_key", "source_split", "class_name", "label"]].copy().rename(columns={"class_name": "true_class_name", "label": "true_label"})
    trace["e6_pred"], trace["e9_pred"], trace["e9_canonical_pred"], trace["e10_hard_pred"] = e6_pred, e9_pred, e9_canonical, hard_pred
    trace["e6_correct"], trace["e10_hard_correct"], trace["e10_hard_changed"], trace["e10_hard_status"] = e6_pred == targets, hard_pred == targets, hard_pred != e6_pred, hard_status
    trace["e9_confidence"], trace["e6_e9_disagree"] = confidence, disagreement
    trace["e6_prob_normal"], trace["e6_prob_bacterial"], trace["e6_prob_viral"] = e6_prob[:, 0], e6_prob[:, 1], e6_prob[:, 2]
    trace["e9_prob_bacterial"], trace["e9_prob_viral"] = e9_prob[:, 0], e9_prob[:, 1]
    e6_metrics, e6_errors = classification_metrics(targets, e6_prob), _subtype_error_counts(classification_metrics(targets, e6_prob))
    analysis = {"experiment": "e11", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "exploratory_note": EXPLORATORY_NOTE, "diagnostic_only_no_threshold_selection": True, "checkpoint_provenance": provenance["checkpoint_contracts"], "alignment": provenance["alignment"], "e6_baseline": {"metrics": e6_metrics, "subtype_errors": e6_errors}, "e10_hard_reference": {"equivalent_to_threshold_050": hard_equivalent, "sanity_passed": hard_sanity_ok, "change_analysis": hard_analysis, "override_count": int(override_050.sum())}, "threshold_grid": threshold_rows, "confidence_groups": confidence_groups}
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(paths["threshold_csv"], pd.DataFrame(threshold_rows))
    _write_json(paths["confidence_json"], {"exploratory_note": EXPLORATORY_NOTE, "groups": confidence_groups})
    _write_json(paths["analysis_json"], analysis)
    _write_csv(paths["trace_csv"], trace)
    return {"console_summary": {"e10_threshold_050_equivalent": hard_equivalent, "e10_hard_sanity_passed": hard_sanity_ok, "threshold_rows": threshold_rows, "confidence_groups": confidence_groups, "outputs": {key: str(path) for key, path in paths.items()}}, "analysis": analysis, "trace": trace}
