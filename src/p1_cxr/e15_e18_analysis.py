"""Read-only E15 versus E18 validation-result analysis.

This module never loads models or checkpoints and never writes beneath either
experiment directory. It accepts only completed CSV/JSON results, verifies their
contract against the locked validation manifest, then writes derived analysis files
into a new timestamped artifacts/analysis directory.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ("NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA")
CLASS_TO_LABEL = {name: index for index, name in enumerate(CLASS_NAMES)}

_E15_REQUIRED = {
    "sample_id", "image_path", "group_key", "source_identifier", "true_label", "true_class_name",
    "e10_e6_pred", "e15_binary_pred", "e15_prob_bacterial", "e15_prob_viral", "e15_final_pred",
    "e15_final_pred_name", "e10_normal_kept",
}
_E18_REQUIRED = {
    "sample_id", "image_path", "group_key", "source_identifier", "true_label", "true_class_name",
    "e10_e6_pred", "e15_final_pred", "e18_binary_pred", "e18_prob_bacterial", "e18_prob_viral",
    "e18_final_pred", "e18_final_pred_name", "e10_normal_kept",
}


def _inside(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Analysis path leaves project root: {value}")
    return resolved


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Analysis YAML must be a mapping: {path}")
    return value


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


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {"version": 1, "analysis": "e15_e18_readonly_validation_comparison", "data_config": "configs/data_config.yaml", "validation_manifest": "artifacts/splits/val_manifest.csv"}
    errors = [{"key": key, "expected": value, "actual": config.get(key)} for key, value in expected.items() if config.get(key) != value]
    contract = config.get("contract", {})
    expected_contract = {"validation_samples": 742, "class_counts": {"NORMAL": 192, "BACTERIAL_PNEUMONIA": 359, "VIRAL_PNEUMONIA": 191}, "class_order": list(CLASS_NAMES), "e10_normal_rule": "e6_pred_zero_kept_else_binary_plus_one", "e18_fixed_epochs": 3, "source_files_read_only": True, "validation_tuning_forbidden": True, "test_access_forbidden": True}
    errors.extend({"key": f"contract.{key}", "expected": value, "actual": contract.get(key)} for key, value in expected_contract.items() if contract.get(key) != value)
    output = config.get("output", {})
    expected_output = {"base_dir": "artifacts/analysis", "directory_prefix": "e15_e18_readonly", "summary": "comparison_summary.json", "full_sample_comparison": "sample_comparison.csv", "changed_samples": "e18_improvements_and_regressions.csv", "class_metrics": "class_metric_comparison.csv", "train_history_summary": "e18_train_history_summary.json"}
    errors.extend({"key": f"output.{key}", "expected": value, "actual": output.get(key)} for key, value in expected_output.items() if output.get(key) != value)
    if errors:
        raise ValueError(f"E15/E18 read-only analysis config drift: {errors}")


def load_config(path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config_path = _inside(path, root)
    if not config_path.is_file():
        raise FileNotFoundError(f"Analysis config unavailable: {config_path}")
    config = _yaml(config_path)
    validate_config(config)
    data = _yaml(_inside(config["data_config"], root))
    if set(data.get("manifests", {})) != {"train", "validation"}:
        raise ValueError("Analysis requires shared train/validation-only config; test access is prohibited.")
    if _inside(data["manifests"]["validation"], root) != _inside(config["validation_manifest"], root):
        raise ValueError("Analysis validation manifest must exactly match shared data configuration.")
    config["resolved"] = {
        "project_root": str(root), "config_path": str(config_path),
        "validation_manifest": str(_inside(config["validation_manifest"], root)),
        "e15_predictions": str(_inside(config["e15"]["predictions"], root)),
        "e15_metrics": str(_inside(config["e15"]["metrics"], root)),
        "e18_predictions": str(_inside(config["e18"]["predictions"], root)),
        "e18_metrics": str(_inside(config["e18"]["metrics"], root)),
        "e18_history": str(_inside(config["e18"]["train_history"], root)),
        "output_base": str(_inside(config["output"]["base_dir"], root)),
    }
    return config


def _validation_manifest(config: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(Path(config["resolved"]["validation_manifest"]))
    required = {"image_path", "label", "class_name", "group_key", "source_identifier"}
    if required - set(frame.columns) or len(frame) != int(config["contract"]["validation_samples"]) or frame["image_path"].duplicated().any():
        raise ValueError("Locked validation manifest schema, count, or image identity failed.")
    numeric = pd.to_numeric(frame["label"], errors="coerce")
    if numeric.isna().any() or not (numeric % 1 == 0).all():
        raise ValueError("Validation manifest labels are not integral.")
    frame["label"] = numeric.astype(int)
    if any(CLASS_TO_LABEL.get(row.class_name) != row.label for row in frame[["class_name", "label"]].itertuples(index=False)):
        raise ValueError("Validation manifest class/label mapping failed.")
    if frame["class_name"].value_counts().to_dict() != config["contract"]["class_counts"]:
        raise ValueError("Validation manifest class counts drifted.")
    frame.insert(0, "sample_id", frame["image_path"].astype(str))
    return frame


def _prediction_frame(path: Path, *, experiment: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Read-only comparison requires {experiment} prediction file: {path}")
    frame = pd.read_csv(path)
    required = _E15_REQUIRED if experiment == "e15" else _E18_REQUIRED
    if required - set(frame.columns) or len(frame) != 742 or frame["image_path"].duplicated().any() or frame["sample_id"].duplicated().any():
        raise ValueError(f"{experiment.upper()} prediction schema, count, or unique identity contract failed.")
    return frame


def _validate_probabilities(frame: pd.DataFrame, *, experiment: str) -> dict[str, float]:
    columns = [f"{experiment}_prob_bacterial", f"{experiment}_prob_viral"]
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    sums = values.sum(axis=1)
    diagnostics = {
        "sum_minimum": float(sums.min()),
        "sum_maximum": float(sums.max()),
        "sum_maximum_absolute_deviation": float(np.max(np.abs(sums - 1.0))),
    }
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or not np.allclose(sums, 1.0, rtol=0.0, atol=5e-4):
        raise ValueError(
            f"{experiment.upper()} B/V probabilities are non-finite, outside [0,1], or exceed "
            f"the FP16 sum tolerance (rtol=0, atol=5e-4): {diagnostics}"
        )
    return diagnostics


def _bool_column(frame: pd.DataFrame, column: str, *, name: str) -> np.ndarray:
    values = frame[column]
    if values.dtype == bool:
        return values.to_numpy(dtype=bool)
    mapped = values.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise ValueError(f"{name} contains non-boolean values.")
    return mapped.to_numpy(dtype=bool)


def _read_metrics(path: Path, *, experiment: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Read-only comparison requires {experiment.upper()} metrics file: {path}")
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{experiment.upper()} metrics JSON is invalid.") from error
    if not isinstance(values, dict) or values.get("experiment") != experiment:
        raise ValueError(f"{experiment.upper()} metrics identity contract failed.")
    return values


def _align(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = _validation_manifest(config)
    e15 = _prediction_frame(Path(config["resolved"]["e15_predictions"]), experiment="e15")
    e18 = _prediction_frame(Path(config["resolved"]["e18_predictions"]), experiment="e18")
    e15_probability_diagnostics = _validate_probabilities(e15, experiment="e15")
    e18_probability_diagnostics = _validate_probabilities(e18, experiment="e18")
    ordered_paths = manifest["image_path"].tolist()
    if set(e15["image_path"]) != set(ordered_paths) or set(e18["image_path"]) != set(ordered_paths):
        raise ValueError("E15/E18 image_path sets do not exactly equal the locked validation manifest.")
    e15 = e15.set_index("image_path").reindex(ordered_paths).reset_index()
    e18 = e18.set_index("image_path").reindex(ordered_paths).reset_index()
    for name, frame in (("E15", e15), ("E18", e18)):
        if frame.isna().any().any() or frame["sample_id"].tolist() != manifest["sample_id"].tolist() or frame["group_key"].tolist() != manifest["group_key"].tolist() or frame["source_identifier"].tolist() != manifest["source_identifier"].tolist() or frame["true_label"].astype(int).tolist() != manifest["label"].tolist() or frame["true_class_name"].tolist() != manifest["class_name"].tolist():
            raise ValueError(f"{name} does not align exactly with validation manifest identities or labels.")
    e6_15, e6_18 = e15["e10_e6_pred"].astype(int).to_numpy(), e18["e10_e6_pred"].astype(int).to_numpy()
    e15_normal_kept = _bool_column(e15, "e10_normal_kept", name="E15 e10_normal_kept")
    e18_normal_kept = _bool_column(e18, "e10_normal_kept", name="E18 e10_normal_kept")
    if not np.array_equal(e6_15, e6_18) or not np.array_equal(e15_normal_kept, e6_15 == 0) or not np.array_equal(e18_normal_kept, e6_18 == 0):
        raise ValueError("E15/E18 E10 Normal routing gate differs or violates e6_pred==0.")
    e15_final, e18_final = e15["e15_final_pred"].astype(int).to_numpy(), e18["e18_final_pred"].astype(int).to_numpy()
    if set(e15_final) - {0, 1, 2} or set(e18_final) - {0, 1, 2} or not np.array_equal(e15_final, np.where(e6_15 == 0, 0, e15["e15_binary_pred"].astype(int).to_numpy() + 1)) or not np.array_equal(e18_final, np.where(e6_18 == 0, 0, e18["e18_binary_pred"].astype(int).to_numpy() + 1)):
        raise ValueError("E15/E18 final prediction violates the locked E10 Normal-preserving routing contract.")
    if not np.array_equal(e18["e15_final_pred"].astype(int).to_numpy(), e15_final):
        raise ValueError("E18 embedded E15 final prediction does not exactly match the E15 result file.")
    alignment = {"samples": len(manifest), "manifest_sha256": _sha256(Path(config["resolved"]["validation_manifest"])), "e15_predictions_sha256": _sha256(Path(config["resolved"]["e15_predictions"])), "e18_predictions_sha256": _sha256(Path(config["resolved"]["e18_predictions"])), "ordered_image_path_hash": _ordered_hash(manifest["image_path"]), "ordered_sample_id_hash": _ordered_hash(manifest["sample_id"]), "e10_gate_exact": True, "e18_embedded_e15_exact": True, "probability_sum_diagnostics": {"e15": e15_probability_diagnostics, "e18": e18_probability_diagnostics}}
    return manifest, e15, e18, alignment


def _metrics(targets: np.ndarray, predictions: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    labels = np.arange(3)
    precision, recall, f1, support = precision_recall_fscore_support(targets, predictions, labels=labels, zero_division=0)
    rows = [{"class_name": name, "precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])} for index, name in enumerate(CLASS_NAMES)]
    matrix = confusion_matrix(targets, predictions, labels=labels)
    return {"accuracy": float(accuracy_score(targets, predictions)), "macro_f1": float(np.mean(f1)), "confusion_matrix": matrix.tolist(), "bacterial_to_viral": int(matrix[1, 2]), "viral_to_bacterial": int(matrix[2, 1]), "bacterial_viral_misclassifications": int(matrix[1, 2] + matrix[2, 1])}, pd.DataFrame(rows)


def _history(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Read-only comparison requires E18 history: {path}")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("E18 train_history.json is not valid JSON.") from error
    required = {"epoch", "train_loss", "train_macro_f1", "train_accuracy", "train_confusion_matrix", "train_bacterial_to_viral", "train_viral_to_bacterial", "checkpoint_selection"}
    if not isinstance(rows, list) or len(rows) != int(config["contract"]["e18_fixed_epochs"]) or any(not isinstance(row, dict) or required - set(row) for row in rows):
        raise ValueError("E18 history does not match the fixed three-epoch contract.")
    frame = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    if frame["epoch"].astype(int).tolist() != [1, 2, 3] or (frame["checkpoint_selection"] != "none_fixed_final_epoch").any():
        raise ValueError("E18 history epoch or no-selection policy drifted.")
    numeric = frame[["train_loss", "train_macro_f1", "train_accuracy"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("E18 history contains non-finite loss/F1/accuracy.")
    return {"epochs": frame.to_dict("records"), "loss": {"first": float(numeric.loc[0, "train_loss"]), "last": float(numeric.loc[2, "train_loss"]), "minimum": float(numeric["train_loss"].min()), "delta_last_minus_first": float(numeric.loc[2, "train_loss"] - numeric.loc[0, "train_loss"])}, "macro_f1": {"first": float(numeric.loc[0, "train_macro_f1"]), "last": float(numeric.loc[2, "train_macro_f1"]), "maximum": float(numeric["train_macro_f1"].max()), "delta_last_minus_first": float(numeric.loc[2, "train_macro_f1"] - numeric.loc[0, "train_macro_f1"])}, "accuracy": {"first": float(numeric.loc[0, "train_accuracy"]), "last": float(numeric.loc[2, "train_accuracy"]), "delta_last_minus_first": float(numeric.loc[2, "train_accuracy"] - numeric.loc[0, "train_accuracy"])}}


def _verify_reported_metrics(source: Mapping[str, Any], calculated: Mapping[str, Any], *, experiment: str) -> None:
    key = f"{experiment}_normal_preserving_full_validation"
    reported = source.get(key)
    if not isinstance(reported, Mapping):
        raise ValueError(f"{experiment.upper()} metrics lack {key}.")
    for metric in ("accuracy", "macro_f1"):
        if not np.isclose(float(reported.get(metric, np.nan)), float(calculated[metric]), rtol=0.0, atol=1e-12):
            raise ValueError(f"{experiment.upper()} reported {metric} disagrees with its prediction CSV.")


def _hypotheses(e15: Mapping[str, Any], e18: Mapping[str, Any], counts: Mapping[str, int], history: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    hypotheses: list[dict[str, Any]] = []
    macro_delta = float(e18["macro_f1"] - e15["macro_f1"])
    if macro_delta < 0 and counts["e15_only_correct"] > counts["e18_only_correct"]:
        hypotheses.append({"hypothesis": "frozen CNN residual fusion introduced harmful validation prediction changes", "evidence": {"macro_f1_delta_e18_minus_e15": macro_delta, "e15_only_correct": counts["e15_only_correct"], "e18_only_correct": counts["e18_only_correct"]}, "interpretation": "The residual head changed more previously correct E15 decisions into errors than it corrected."})
    directional = {"bacterial_to_viral_delta": int(e18["bacterial_to_viral"] - e15["bacterial_to_viral"]), "viral_to_bacterial_delta": int(e18["viral_to_bacterial"] - e15["viral_to_bacterial"])}
    for key, delta in directional.items():
        if delta > 0:
            hypotheses.append({"hypothesis": "directional B/V class bias after residual fusion", "evidence": {key: delta}, "interpretation": f"E18 increased {key.replace('_delta', '').replace('_', '→')} errors relative to E15."})
    if history["loss"]["delta_last_minus_first"] < 0 and macro_delta < 0:
        hypotheses.append({"hypothesis": "train-only optimization did not transfer to fixed external validation", "evidence": {"train_loss_delta": history["loss"]["delta_last_minus_first"], "train_macro_f1_delta": history["macro_f1"]["delta_last_minus_first"], "validation_macro_f1_delta": macro_delta}, "interpretation": "Lower train loss alongside lower validation Macro F1 is consistent with overfitting or a non-complementary frozen CNN feature."})
    if not hypotheses:
        hypotheses.append({"hypothesis": "no single degradation mechanism is established by aggregate validation outcomes", "evidence": {"macro_f1_delta_e18_minus_e15": macro_delta, **directional}, "interpretation": "Inspect the generated per-sample change CSV before proposing a preregistered future experiment; do not tune on this validation set."})
    if macro_delta < 0:
        recommendation = {"recommendation": "retain E15 as the current candidate", "rationale": "E18 is worse on this fixed validation comparison. Do not tune E18 on these validation results; any next architecture should be specified independently and evaluated once."}
    elif macro_delta > 0:
        recommendation = {"recommendation": "E18 merits a separate, preregistered confirmation experiment", "rationale": "The observed validation gain is descriptive only. Do not use this report for threshold, epoch, or hyperparameter tuning."}
    else:
        recommendation = {"recommendation": "retain E15 pending independent evidence", "rationale": "No Macro F1 improvement is established by this fixed validation comparison."}
    return hypotheses, recommendation


def static_smoke_check(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    manifest = _validation_manifest(config)
    expected = {key: Path(config["resolved"][key]).is_file() for key in ("e15_predictions", "e15_metrics", "e18_predictions", "e18_metrics", "e18_history")}
    synthetic_targets = np.array([0, 1, 2, 1], dtype=np.int64)
    synthetic_e15 = np.array([0, 1, 1, 2], dtype=np.int64)
    synthetic_e18 = np.array([0, 2, 2, 1], dtype=np.int64)
    e15_metrics, _ = _metrics(synthetic_targets, synthetic_e15)
    e18_metrics, _ = _metrics(synthetic_targets, synthetic_e18)
    if e15_metrics["confusion_matrix"] != [[1, 0, 0], [0, 1, 1], [0, 1, 0]] or e18_metrics["confusion_matrix"] != [[1, 0, 0], [0, 1, 1], [0, 0, 1]]:
        raise RuntimeError("Read-only analysis synthetic confusion-matrix contract failed.")
    return {"analysis": config["analysis"], "static_check": "passed", "validation_samples": len(manifest), "manifest_sha256": _sha256(Path(config["resolved"]["validation_manifest"])), "source_artifacts_available": expected, "expected_e15_columns": sorted(_E15_REQUIRED), "expected_e18_columns": sorted(_E18_REQUIRED), "artifacts_written": False}


def run_analysis(config: Mapping[str, Any], *, timestamp: str | None = None) -> dict[str, Any]:
    static_smoke_check(config)
    manifest, e15, e18, alignment = _align(config)
    source_metrics = {"e15": _read_metrics(Path(config["resolved"]["e15_metrics"]), experiment="e15"), "e18": _read_metrics(Path(config["resolved"]["e18_metrics"]), experiment="e18")}
    history = _history(Path(config["resolved"]["e18_history"]), config)
    targets = manifest["label"].to_numpy(dtype=np.int64)
    e15_predictions = e15["e15_final_pred"].astype(int).to_numpy()
    e18_predictions = e18["e18_final_pred"].astype(int).to_numpy()
    e15_metrics, e15_class = _metrics(targets, e15_predictions)
    e18_metrics, e18_class = _metrics(targets, e18_predictions)
    _verify_reported_metrics(source_metrics["e15"], e15_metrics, experiment="e15")
    _verify_reported_metrics(source_metrics["e18"], e18_metrics, experiment="e18")
    class_report = e15_class.merge(e18_class, on=["class_name", "support"], suffixes=("_e15", "_e18"))
    for metric in ("precision", "recall", "f1"):
        class_report[f"{metric}_delta_e18_minus_e15"] = class_report[f"{metric}_e18"] - class_report[f"{metric}_e15"]
    e15_correct, e18_correct = e15_predictions == targets, e18_predictions == targets
    status = np.select([e15_correct & ~e18_correct, ~e15_correct & e18_correct, e15_correct & e18_correct], ["e15_only_correct_e18_regression", "e18_only_correct_e18_improvement", "both_correct"], default="both_wrong")
    comparison = manifest[["sample_id", "image_path", "group_key", "source_identifier", "label", "class_name"]].copy()
    comparison["e10_e6_pred"] = e15["e10_e6_pred"].astype(int)
    comparison["e10_normal_kept"] = _bool_column(e15, "e10_normal_kept", name="E15 e10_normal_kept")
    comparison["e15_final_pred"] = e15_predictions
    comparison["e15_final_pred_name"] = [CLASS_NAMES[value] for value in e15_predictions]
    comparison["e18_final_pred"] = e18_predictions
    comparison["e18_final_pred_name"] = [CLASS_NAMES[value] for value in e18_predictions]
    comparison["e15_prob_bacterial"] = e15["e15_prob_bacterial"].astype(float)
    comparison["e15_prob_viral"] = e15["e15_prob_viral"].astype(float)
    comparison["e18_prob_bacterial"] = e18["e18_prob_bacterial"].astype(float)
    comparison["e18_prob_viral"] = e18["e18_prob_viral"].astype(float)
    comparison["e15_correct"] = e15_correct
    comparison["e18_correct"] = e18_correct
    comparison["comparison_status"] = status
    comparison["e18_changed_final_prediction"] = e15_predictions != e18_predictions
    counts = {"e15_only_correct": int((status == "e15_only_correct_e18_regression").sum()), "e18_only_correct": int((status == "e18_only_correct_e18_improvement").sum()), "both_correct": int((status == "both_correct").sum()), "both_wrong": int((status == "both_wrong").sum())}
    if sum(counts.values()) != 742:
        raise RuntimeError("E15/E18 correctness partition does not cover exactly 742 samples.")
    hypotheses, recommendation = _hypotheses(e15_metrics, e18_metrics, counts, history)
    summary = {"analysis": config["analysis"], "source_files_read_only": True, "validation_tuning_performed": False, "test_data_used": False, "alignment": alignment, "source_metric_sha256": {"e15": _sha256(Path(config["resolved"]["e15_metrics"])), "e18": _sha256(Path(config["resolved"]["e18_metrics"])), "e18_history": _sha256(Path(config["resolved"]["e18_history"]))}, "correctness_partition": counts, "e15": e15_metrics, "e18": e18_metrics, "delta_e18_minus_e15": {"macro_f1": float(e18_metrics["macro_f1"] - e15_metrics["macro_f1"]), "accuracy": float(e18_metrics["accuracy"] - e15_metrics["accuracy"]), "bacterial_to_viral": int(e18_metrics["bacterial_to_viral"] - e15_metrics["bacterial_to_viral"]), "viral_to_bacterial": int(e18_metrics["viral_to_bacterial"] - e15_metrics["viral_to_bacterial"])}, "e18_training_history": history, "hypotheses": hypotheses, "recommendation": recommendation}
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(config["resolved"]["output_base"]) / f"{config['output']['directory_prefix']}_{stamp}"
    if output.exists():
        raise FileExistsError(f"Read-only analysis output already exists; no overwrite: {output}")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / config["output"]["summary"], summary)
    _write_csv(output / config["output"]["full_sample_comparison"], comparison)
    _write_csv(output / config["output"]["changed_samples"], comparison[comparison["comparison_status"].isin(["e15_only_correct_e18_regression", "e18_only_correct_e18_improvement"])].reset_index(drop=True))
    _write_csv(output / config["output"]["class_metrics"], class_report)
    _write_json(output / config["output"]["train_history_summary"], history)
    return {"output_dir": str(output), "correctness_partition": counts, "macro_f1_delta_e18_minus_e15": summary["delta_e18_minus_e15"]["macro_f1"], "recommendation": recommendation}
