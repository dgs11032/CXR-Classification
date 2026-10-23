#!/usr/bin/env python3
"""Read-only data-centric audit for E10 versus E15 validation predictions.

The script never trains, changes labels/images/manifests, or deletes samples. It reads
only development train/validation images, writes a new analysis directory, and refuses
to overwrite an existing audit. Test manifests and test images are never referenced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA")
OUTPUT_DEFAULT = PROJECT_ROOT / "artifacts" / "analysis" / "e15_data_audit"


@dataclass(frozen=True)
class AuditPaths:
    train_manifest: Path
    validation_manifest: Path
    e10_predictions: Path
    e15_predictions: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=PROJECT_ROOT / "artifacts" / "splits" / "train_manifest.csv")
    parser.add_argument("--validation-manifest", type=Path, default=PROJECT_ROOT / "artifacts" / "splits" / "val_manifest.csv")
    parser.add_argument("--e10-predictions", type=Path, default=PROJECT_ROOT / "artifacts" / "experiments" / "e10" / "hierarchical_predictions.csv")
    parser.add_argument("--e15-predictions", type=Path, default=PROJECT_ROOT / "artifacts" / "experiments" / "e15" / "validation_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--phash-distance", type=int, default=5, help="Maximum 64-bit dHash Hamming distance for train/validation candidates.")
    parser.add_argument("--contact-sheet-limit", type=int, default=24)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.90)
    return parser.parse_args()


def resolve_inside_project(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path must stay inside project root: {path}")
    return resolved


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(image: Image.Image) -> int:
    """Dependency-free 64-bit difference hash; candidates require visual review."""
    array = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (array[:, 1:] > array[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def image_features(image_path: Path, expected_sha256: str) -> dict[str, Any]:
    actual_sha256 = sha256(image_path)
    try:
        with Image.open(image_path) as image:
            image.load()
            gray = np.asarray(image.convert("L"), dtype=np.float32)
            height, width = gray.shape
            border = max(1, min(height, width) // 50)
            edge = np.concatenate((gray[:border, :].ravel(), gray[-border:, :].ravel(), gray[:, :border].ravel(), gray[:, -border:].ravel()))
            interior = gray[border:-border, border:-border]
            interior_mean = float(interior.mean()) if interior.size else float(gray.mean())
            return {
                "width": int(width), "height": int(height), "aspect_ratio": float(width / height),
                "brightness_mean": float(gray.mean()), "brightness_std": float(gray.std()),
                "border_mean": float(edge.mean()), "interior_mean": interior_mean,
                "border_minus_interior": float(edge.mean() - interior_mean),
                "dark_border_fraction": float((edge < 16).mean()),
                "dhash64": f"{dhash(image):016x}", "computed_sha256": actual_sha256,
                "manifest_sha256_matches_file": actual_sha256 == str(expected_sha256),
            }
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise RuntimeError(f"Cannot inspect image: {image_path}; {type(error).__name__}: {error}") from error


def read_manifest(path: Path, split: str) -> pd.DataFrame:
    required = {"image_path", "label", "class_name", "source_split", "group_key", "sha256", "source_identifier"}
    frame = pd.read_csv(path)
    if required - set(frame.columns) or frame.empty or frame["image_path"].duplicated().any():
        raise ValueError(f"Invalid {split} manifest schema/identity: {path}")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or set(labels.astype(int)) != {0, 1, 2}:
        raise ValueError(f"Invalid {split} manifest labels: {path}")
    frame = frame.copy(); frame["label"] = labels.astype(int); frame["audit_split"] = split
    for image_path in frame["image_path"]:
        candidate = (PROJECT_ROOT / str(image_path)).resolve()
        if not candidate.is_relative_to(PROJECT_ROOT) or not candidate.is_file():
            raise FileNotFoundError(f"Missing development image: {candidate}")
    return frame


def read_and_align_predictions(validation: pd.DataFrame, e10_path: Path, e15_path: Path) -> pd.DataFrame:
    e10_required = {"image_path", "group_key", "true_label", "final_pred", "e6_pred", "e9_pred"}
    e15_required = {"sample_id", "image_path", "group_key", "true_label", "e15_final_pred", "e15_binary_pred", "e15_prob_bacterial", "e15_prob_viral", "e10_normal_kept"}
    e10, e15 = pd.read_csv(e10_path), pd.read_csv(e15_path)
    if e10_required - set(e10.columns) or e15_required - set(e15.columns):
        raise ValueError("E10/E15 prediction CSV does not match the required writer schema.")
    if e10["image_path"].duplicated().any() or e15["image_path"].duplicated().any():
        raise ValueError("E10/E15 prediction CSV has duplicate image_path values.")
    base = validation[["image_path", "group_key", "label", "class_name", "sha256", "source_identifier"]].copy()
    e10 = e10.set_index("image_path").reindex(base["image_path"]).reset_index()
    e15 = e15.set_index("image_path").reindex(base["image_path"]).reset_index()
    if e10.isna().all(axis=1).any() or e15.isna().all(axis=1).any():
        raise ValueError("E10/E15 predictions do not cover every validation manifest image_path.")
    for name, frame in (("E10", e10), ("E15", e15)):
        if frame["true_label"].astype(int).tolist() != base["label"].tolist() or frame["group_key"].tolist() != base["group_key"].tolist():
            raise ValueError(f"{name} prediction identity/label/group alignment drift.")
    aligned = base.rename(columns={"label": "true_label", "class_name": "true_class_name"})
    aligned["e10_pred"] = e10["final_pred"].astype(int)
    aligned["e10_e6_pred"] = e10["e6_pred"].astype(int)
    aligned["e10_e9_pred"] = e10["e9_pred"].astype(int)
    aligned["e15_pred"] = e15["e15_final_pred"].astype(int)
    aligned["e15_binary_pred"] = e15["e15_binary_pred"].astype(int)
    aligned["e15_prob_bacterial"] = e15["e15_prob_bacterial"].astype(float)
    aligned["e15_prob_viral"] = e15["e15_prob_viral"].astype(float)
    aligned["e10_normal_kept"] = e15["e10_normal_kept"].astype(bool)
    probabilities = aligned[["e15_prob_bacterial", "e15_prob_viral"]].to_numpy(dtype=float)
    probability_sums = probabilities.sum(axis=1)
    probability_sum_abs_error = np.abs(probability_sums - 1.0)
    # E15 CSV probabilities can be FP16-rounded; allow only the documented bounded error.
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probability_sums, 1.0, rtol=0, atol=5e-4)
    ):
        raise ValueError("E15 B/V probabilities are invalid.")
    aligned["e15_probability_sum"] = probability_sums
    aligned["e15_probability_sum_abs_error"] = probability_sum_abs_error
    if not np.array_equal(aligned["e15_pred"].to_numpy(), np.where(aligned["e10_e6_pred"].to_numpy() == 0, 0, aligned["e15_binary_pred"].to_numpy() + 1)):
        raise ValueError("E15 Normal-preserving routing invariant failed.")
    return aligned


def class_change_table(aligned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, name in enumerate(CLASS_NAMES):
        subset = aligned[aligned["true_label"] == label]
        for predicted in range(3):
            rows.append({"true_label": label, "true_class_name": name, "predicted_label": predicted, "predicted_class_name": CLASS_NAMES[predicted], "e10_count": int((subset["e10_pred"] == predicted).sum()), "e15_count": int((subset["e15_pred"] == predicted).sum())})
    result = pd.DataFrame(rows); result["count_delta_e15_minus_e10"] = result["e15_count"] - result["e10_count"]
    return result


def outcome_table(aligned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    e10_correct = aligned["e10_pred"] == aligned["true_label"]
    e15_correct = aligned["e15_pred"] == aligned["true_label"]
    aligned = aligned.copy()
    aligned["comparison_outcome"] = np.select(
        [e10_correct & ~e15_correct, ~e10_correct & e15_correct, e10_correct & e15_correct],
        ["e10_only_correct", "e15_only_correct", "both_correct"], default="both_wrong",
    )
    counts = aligned.groupby("comparison_outcome", sort=False).size().reindex(["e10_only_correct", "e15_only_correct", "both_correct", "both_wrong"], fill_value=0).rename("samples").reset_index()
    return aligned, counts


def group_patterns(aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, frame in aligned.groupby("group_key", sort=False):
        rows.append({"group_key": group, "samples": len(frame), "true_classes": ";".join(sorted(frame["true_class_name"].unique())), "e10_errors": int((frame["e10_pred"] != frame["true_label"]).sum()), "e15_errors": int((frame["e15_pred"] != frame["true_label"]).sum()), "e15_only_correct": int((frame["comparison_outcome"] == "e15_only_correct").sum()), "e10_only_correct": int((frame["comparison_outcome"] == "e10_only_correct").sum())})
    return pd.DataFrame(rows).sort_values(["e15_errors", "e10_errors", "samples"], ascending=False)


def validate_feature_hash_columns(features: pd.DataFrame) -> None:
    required = {"audit_split", "image_path", "group_key", "class_name", "manifest_sha256", "computed_sha256", "dhash64"}
    missing = required - set(features.columns)
    if missing or features.empty:
        raise ValueError(f"Image feature schema is invalid: missing={sorted(missing)}, empty={features.empty}")
    for column, length in (("manifest_sha256", 64), ("computed_sha256", 64), ("dhash64", 16)):
        values = features[column]
        if values.isna().any() or not values.map(lambda value: isinstance(value, str) and len(value) == length and all(character in "0123456789abcdef" for character in value)).all():
            raise ValueError(f"Image feature column {column} must contain lowercase hexadecimal values of length {length}.")
    if not set(features["audit_split"].unique()).issubset({"train", "validation"}):
        raise ValueError("Image feature audit_split contains an unsupported partition.")


def phash_candidates(features: pd.DataFrame, distance: int) -> pd.DataFrame:
    if distance < 0 or distance > 64:
        raise ValueError(f"Invalid dHash Hamming distance: {distance}")
    validate_feature_hash_columns(features)
    train = features[features["audit_split"] == "train"]
    validation = features[features["audit_split"] == "validation"]
    if train.empty or validation.empty:
        raise ValueError("dHash candidate search requires non-empty train and validation feature rows.")
    train_records = train[["image_path", "dhash64", "manifest_sha256", "computed_sha256", "class_name", "group_key"]].to_dict("records")
    rows: list[dict[str, Any]] = []
    for validation_record in validation[["image_path", "dhash64", "manifest_sha256", "computed_sha256", "class_name", "group_key"]].to_dict("records"):
        validation_hash = int(validation_record["dhash64"], 16)
        for train_record in train_records:
            # Exact duplicate exclusion uses the SHA-256 recomputed from actual image bytes.
            if validation_record["computed_sha256"] == train_record["computed_sha256"]:
                continue
            hamming = (validation_hash ^ int(train_record["dhash64"], 16)).bit_count()
            if hamming <= distance:
                rows.append({
                    "validation_image_path": validation_record["image_path"],
                    "validation_class_name": validation_record["class_name"],
                    "validation_group_key": validation_record["group_key"],
                    "validation_manifest_sha256": validation_record["manifest_sha256"],
                    "validation_computed_sha256": validation_record["computed_sha256"],
                    "train_image_path": train_record["image_path"],
                    "train_class_name": train_record["class_name"],
                    "train_group_key": train_record["group_key"],
                    "train_manifest_sha256": train_record["manifest_sha256"],
                    "train_computed_sha256": train_record["computed_sha256"],
                    "dhash_hamming_distance": hamming,
                })
    columns = ["validation_image_path", "validation_class_name", "validation_group_key", "validation_manifest_sha256", "validation_computed_sha256", "train_image_path", "train_class_name", "train_group_key", "train_manifest_sha256", "train_computed_sha256", "dhash_hamming_distance"]
    return pd.DataFrame(rows, columns=columns).sort_values(["dhash_hamming_distance", "validation_image_path", "train_image_path"]) if rows else pd.DataFrame(columns=columns)


def thumbnail(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image.load(); result = ImageOps.contain(image.convert("RGB"), size)
    canvas = Image.new("RGB", size, "white"); canvas.paste(result, ((size[0] - result.width) // 2, (size[1] - result.height) // 2)); return canvas


def contact_sheet(items: Iterable[tuple[Path, str]], title: str, output: Path, limit: int, columns: int = 4) -> int:
    chosen = list(items)[:limit]
    if not chosen: return 0
    thumb_size, text_height = (240, 180), 54
    rows = int(np.ceil(len(chosen) / columns)); sheet = Image.new("RGB", (columns * thumb_size[0], 34 + rows * (thumb_size[1] + text_height)), "white")
    draw = ImageDraw.Draw(sheet); draw.text((8, 8), title, fill="black")
    for index, (path, label) in enumerate(chosen):
        x, y = (index % columns) * thumb_size[0], 34 + (index // columns) * (thumb_size[1] + text_height)
        sheet.paste(thumbnail(path, thumb_size), (x, y)); draw.multiline_text((x + 4, y + thumb_size[1] + 3), label[:150], fill="black", spacing=2)
    output.parent.mkdir(parents=True, exist_ok=True); sheet.save(output); return len(chosen)


def e16_note(outcomes: pd.DataFrame, high_confidence: pd.DataFrame, candidates: pd.DataFrame) -> str:
    return "\n".join([
        "# E16 Global+Local readiness note", "",
        "This audit is exploratory only: no sample, image, or label was modified.",
        f"- E10-only correct: {int((outcomes.comparison_outcome == 'e10_only_correct').sum())}",
        f"- E15-only correct: {int((outcomes.comparison_outcome == 'e15_only_correct').sum())}",
        f"- Both wrong: {int((outcomes.comparison_outcome == 'both_wrong').sum())}",
        f"- High-confidence routed E15 errors: {len(high_confidence)}", f"- Train/validation near-duplicate candidates: {len(candidates)}", "",
        "Global+Local modeling is a hypothesis when errors correlate with border, brightness, aspect ratio, or local image structure. It must not be treated as evidence that labels are wrong.",
        "Risk: group_key is filename-derived proxy metadata, not a verified patient ID. Zero group overlap does not establish patient-level independence; near-duplicate candidates require visual review before any future split or data policy decision.",
        "Any E16 architecture or data intervention must be pre-specified without tuning on this validation audit.",
    ]) + "\n"


def main() -> None:
    args = parse_args()
    paths = AuditPaths(*(resolve_inside_project(path) for path in (args.train_manifest, args.validation_manifest, args.e10_predictions, args.e15_predictions, args.output_dir)))
    if paths.output_dir.exists(): raise FileExistsError(f"Audit output already exists; refusing overwrite: {paths.output_dir}")
    if args.phash_distance < 0 or args.phash_distance > 64 or args.contact_sheet_limit < 1 or not 0.0 <= args.high_confidence_threshold <= 1.0: raise ValueError("Invalid audit thresholds.")
    for required in (paths.train_manifest, paths.validation_manifest, paths.e10_predictions, paths.e15_predictions):
        if not required.is_file(): raise FileNotFoundError(f"Required read-only input is unavailable: {required}")
    train, validation = read_manifest(paths.train_manifest, "train"), read_manifest(paths.validation_manifest, "validation")
    if set(train["group_key"]) & set(validation["group_key"]): raise ValueError("Manifest train/validation group_key overlap detected; audit stopped.")
    aligned = read_and_align_predictions(validation, paths.e10_predictions, paths.e15_predictions)
    aligned, outcome_counts = outcome_table(aligned)
    routed = ~aligned["e10_normal_kept"]
    aligned["e15_binary_confidence"] = aligned[["e15_prob_bacterial", "e15_prob_viral"]].max(axis=1)
    high_confidence = aligned[(aligned["e15_pred"] != aligned["true_label"]) & routed & (aligned["e15_binary_confidence"] >= args.high_confidence_threshold)].sort_values("e15_binary_confidence", ascending=False)
    all_images = pd.concat([train, validation], ignore_index=True)
    feature_rows = []
    for row in all_images.itertuples(index=False):
        values = image_features((PROJECT_ROOT / str(row.image_path)).resolve(), str(row.sha256))
        feature_rows.append({"audit_split": row.audit_split, "image_path": row.image_path, "group_key": row.group_key, "class_name": row.class_name, "label": int(row.label), "manifest_sha256": row.sha256, "source_identifier": row.source_identifier, **values})
    features = pd.DataFrame(feature_rows)
    validate_feature_hash_columns(features)
    exact_hash = features.groupby("computed_sha256", dropna=False).agg(samples=("image_path", "size"), splits=("audit_split", lambda values: ";".join(sorted(set(values)))), image_paths=("image_path", lambda values: ";".join(values))).reset_index()
    exact_hash = exact_hash[exact_hash["samples"] > 1].copy(); exact_hash["cross_split"] = exact_hash["splits"].str.contains(";")
    candidates = phash_candidates(features, args.phash_distance)
    class_changes = class_change_table(aligned); groups = group_patterns(aligned)
    paths.output_dir.mkdir(parents=True); sheets = paths.output_dir / "contact_sheets"
    error_items = [((PROJECT_ROOT / str(row.image_path)).resolve(), f"{row.comparison_outcome}\ntrue={row.true_class_name} e10={CLASS_NAMES[int(row.e10_pred)]} e15={CLASS_NAMES[int(row.e15_pred)]}\nconfidence={row.e15_binary_confidence:.3f}") for row in high_confidence.itertuples(index=False)]
    error_sheet_count = contact_sheet(error_items, "High-confidence routed E15 errors", sheets / "high_confidence_e15_errors.png", args.contact_sheet_limit)
    pair_items: list[tuple[Path, str]] = []
    for row in candidates.head(max(1, args.contact_sheet_limit // 2)).itertuples(index=False):
        pair_items.extend([((PROJECT_ROOT / str(row.validation_image_path)).resolve(), f"VAL d={row.dhash_hamming_distance}\n{row.validation_class_name}\n{row.validation_image_path}"), ((PROJECT_ROOT / str(row.train_image_path)).resolve(), f"TRAIN d={row.dhash_hamming_distance}\n{row.train_class_name}\n{row.train_image_path}")])
    pair_sheet_count = contact_sheet(pair_items, "Train/validation dHash candidates (visual review required)", sheets / "perceptual_train_validation_candidates.png", args.contact_sheet_limit)
    atomic_csv(paths.output_dir / "aligned_predictions.csv", aligned)
    atomic_csv(paths.output_dir / "comparison_outcome_counts.csv", outcome_counts)
    atomic_csv(paths.output_dir / "class_error_changes.csv", class_changes)
    atomic_csv(paths.output_dir / "high_confidence_e15_errors.csv", high_confidence)
    atomic_csv(paths.output_dir / "image_features_train_validation.csv", features)
    atomic_csv(paths.output_dir / "exact_hash_duplicates_train_validation.csv", exact_hash)
    atomic_csv(paths.output_dir / "perceptual_hash_candidates_train_validation.csv", candidates)
    atomic_csv(paths.output_dir / "group_key_error_patterns.csv", groups)
    (paths.output_dir / "e16_readiness.md").write_text(e16_note(aligned, high_confidence, candidates), encoding="utf-8")
    summary = {"analysis":"e15_data_audit", "timestamp_utc":datetime.now(timezone.utc).isoformat(), "read_only_inputs":asdict(paths), "safety":{"test_data_used":False,"training_performed":False,"labels_or_images_modified":False,"automatic_deletion_or_relabeling":False}, "prediction_alignment":{"validation_samples":len(aligned),"e10_e15_image_path_order_verified":True}, "e15_probability_validation":{"rtol":0,"atol":0.0005,"max_absolute_sum_error":float(aligned["e15_probability_sum_abs_error"].max())}, "outcomes":outcome_counts.set_index("comparison_outcome")["samples"].to_dict(), "high_confidence_threshold":args.high_confidence_threshold,"high_confidence_routed_e15_errors":len(high_confidence), "exact_hash":{"duplicate_groups":len(exact_hash),"cross_split_groups":int(exact_hash["cross_split"].sum()),"manifest_hash_mismatches":int((~features["manifest_sha256_matches_file"]).sum())}, "perceptual_hash":{"algorithm":"64-bit dHash via Pillow/NumPy","max_hamming_distance":args.phash_distance,"train_validation_candidates":len(candidates),"visual_review_required":True}, "group_key_limitation":"group_key is a filename-derived proxy group, not a verified patient identifier; zero overlap is not patient-level independence.", "contact_sheets":{"high_confidence_items":error_sheet_count,"perceptual_items":pair_sheet_count}, "environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__}}
    atomic_json(paths.output_dir / "audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
