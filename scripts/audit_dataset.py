#!/usr/bin/env python3
"""CXR 원본 데이터의 3클래스 라벨과 파일명 식별자 누수 위험을 읽기 전용으로 감사한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "chest_xray" / "chest_xray"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "data_audit"
SPLITS = ("train", "val", "test")
FOLDERS = ("NORMAL", "PNEUMONIA")
SUPPORTED_SUFFIXES = {".jpeg", ".jpg", ".png"}
CLASS_INFO = {
    "NORMAL": (0, "NORMAL"),
    "BACTERIAL_PNEUMONIA": (1, "BACTERIAL_PNEUMONIA"),
    "VIRAL_PNEUMONIA": (2, "VIRAL_PNEUMONIA"),
}
PNEUMONIA_ID_PATTERN = re.compile(r"(person\d+)", re.IGNORECASE)
NORMAL_ID_PATTERN = re.compile(r"(IM-\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-image-verify",
        action="store_true",
        help="PIL 이미지 무결성 검사를 생략합니다. 파일 해시는 계속 계산합니다.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_image(path: Path) -> tuple[int | None, str | None, str | None]:
    """요청된 폴더·파일명 규칙으로 3클래스 라벨을 판별한다."""
    folder = path.parent.name.upper()
    filename = path.name.lower()
    if folder == "NORMAL":
        return 0, "NORMAL", None
    if folder != "PNEUMONIA":
        return None, None, f"지원하지 않는 클래스 폴더: {path.parent.name}"

    has_bacteria = "bacteria" in filename
    has_virus = "virus" in filename
    if has_bacteria and not has_virus:
        return 1, "BACTERIAL_PNEUMONIA", None
    if has_virus and not has_bacteria:
        return 2, "VIRAL_PNEUMONIA", None
    if has_bacteria and has_virus:
        return None, None, "파일명에 bacteria와 virus가 모두 포함됨"
    return None, None, "PNEUMONIA 파일명에 bacteria 또는 virus가 없음"


def extract_identifier(path: Path) -> tuple[str | None, str | None, str | None]:
    """요청된 원문 식별자 personXXXX 또는 IM-XXXX를 추출한다."""
    folder = path.parent.name.upper()
    pattern = PNEUMONIA_ID_PATTERN if folder == "PNEUMONIA" else NORMAL_ID_PATTERN
    match = pattern.search(path.name)
    if match is None:
        expected = "personXXXX" if folder == "PNEUMONIA" else "IM-XXXX"
        return None, None, f"파일명에서 {expected} 식별자를 찾지 못함"
    raw = match.group(1)
    if folder == "PNEUMONIA":
        return raw.lower(), "person", None
    return raw.upper(), "IM", None


def inspect_image(path: Path, verify: bool) -> tuple[dict[str, Any], str | None]:
    """이미지를 수정하지 않고 크기·모드·무결성을 확인한다."""
    try:
        if verify:
            with Image.open(path) as image:
                image.verify()
        with Image.open(path) as image:
            metadata = {"width": image.width, "height": image.height, "mode": image.mode}
        return metadata, None
    except (OSError, UnidentifiedImageError, ValueError) as error:
        return {"width": None, "height": None, "mode": None}, str(error)


def scan_dataset(data_root: Path, verify_images: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """여섯 원본 폴더를 순회해 이미지 목록과 비이미지 파일 목록을 만든다."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"데이터 루트가 없습니다: {data_root}")

    records: list[dict[str, Any]] = []
    non_image_records: list[dict[str, Any]] = []
    for split in SPLITS:
        for folder in FOLDERS:
            directory = data_root / split / folder
            if not directory.is_dir():
                raise FileNotFoundError(f"필수 데이터 폴더가 없습니다: {directory}")
            for path in sorted(item for item in directory.rglob("*") if item.is_file()):
                relative_path = path.relative_to(data_root).as_posix()
                if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    non_image_records.append(
                        {
                            "split": split,
                            "folder": folder,
                            "relative_path": relative_path,
                            "suffix": path.suffix,
                            "bytes": path.stat().st_size,
                        }
                    )
                    continue

                label, class_name, classification_issue = classify_image(path)
                identifier, identifier_type, identifier_issue = extract_identifier(path)
                image_metadata, image_issue = inspect_image(path, verify_images)
                records.append(
                    {
                        "split": split,
                        "folder": folder,
                        "relative_path": relative_path,
                        "filename": path.name,
                        "label": label,
                        "class_name": class_name,
                        "identifier": identifier,
                        "identifier_type": identifier_type,
                        "classification_issue": classification_issue,
                        "identifier_issue": identifier_issue,
                        "image_issue": image_issue,
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        **image_metadata,
                    }
                )

    images = pd.DataFrame.from_records(records).sort_values(
        ["split", "folder", "relative_path"]
    ).reset_index(drop=True)
    non_images = pd.DataFrame.from_records(
        non_image_records,
        columns=["split", "folder", "relative_path", "suffix", "bytes"],
    )
    return images, non_images


def class_distribution(images: pd.DataFrame) -> pd.DataFrame:
    classified = images[images["label"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for split in (*SPLITS, "전체"):
        subset = classified if split == "전체" else classified[classified["split"] == split]
        denominator = len(subset)
        for class_name, (label, _) in CLASS_INFO.items():
            count = int((subset["class_name"] == class_name).sum())
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "class_name": class_name,
                    "count": count,
                    "ratio": count / denominator if denominator else 0.0,
                    "ratio_percent": 100.0 * count / denominator if denominator else 0.0,
                }
            )
    return pd.DataFrame(rows)


def identifier_summary(images: pd.DataFrame) -> pd.DataFrame:
    valid = images[images["identifier"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for split in (*SPLITS, "전체"):
        subset = valid if split == "전체" else valid[valid["split"] == split]
        groups = subset.groupby("identifier", sort=True)
        sizes = groups.size()
        label_counts = groups["class_name"].nunique(dropna=True)
        repeated = sizes[sizes > 1]
        rows.append(
            {
                "split": split,
                "images_with_identifier": int(len(subset)),
                "unique_identifiers": int(sizes.size),
                "identifiers_with_multiple_images": int(len(repeated)),
                "images_in_repeated_identifiers": int(repeated.sum()),
                "extra_images_beyond_first": int((repeated - 1).sum()),
                "identifiers_with_multiple_labels": int((label_counts > 1).sum()),
                "images_without_identifier": int(
                    images["identifier"].isna().sum()
                    if split == "전체"
                    else images.loc[images["split"] == split, "identifier"].isna().sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def repeated_identifiers(images: pd.DataFrame) -> pd.DataFrame:
    valid = images[images["identifier"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for (split, identifier), group in valid.groupby(["split", "identifier"], sort=True):
        if len(group) <= 1:
            continue
        labels = sorted(group["class_name"].dropna().unique())
        rows.append(
            {
                "split": split,
                "identifier": identifier,
                "identifier_type": group["identifier_type"].iloc[0],
                "image_count": int(len(group)),
                "label_count": len(labels),
                "class_names": "|".join(labels),
                "has_bacterial_and_viral": {
                    "BACTERIAL_PNEUMONIA",
                    "VIRAL_PNEUMONIA",
                }.issubset(labels),
                "file_examples": " | ".join(group["relative_path"].head(8)),
            }
        )
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "split",
            "identifier",
            "identifier_type",
            "image_count",
            "label_count",
            "class_names",
            "has_bacterial_and_viral",
            "file_examples",
        ],
    )


def cross_split_overlaps(images: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    valid = images[images["identifier"].notna()].copy()
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for left, right in combinations(SPLITS, 2):
        left_frame = valid[valid["split"] == left]
        right_frame = valid[valid["split"] == right]
        shared = sorted(set(left_frame["identifier"]) & set(right_frame["identifier"]))
        summaries.append({"left_split": left, "right_split": right, "overlap_count": len(shared)})
        for identifier in shared:
            left_group = left_frame[left_frame["identifier"] == identifier]
            right_group = right_frame[right_frame["identifier"] == identifier]
            rows.append(
                {
                    "left_split": left,
                    "right_split": right,
                    "identifier": identifier,
                    "left_image_count": int(len(left_group)),
                    "right_image_count": int(len(right_group)),
                    "left_classes": "|".join(sorted(left_group["class_name"].dropna().unique())),
                    "right_classes": "|".join(sorted(right_group["class_name"].dropna().unique())),
                    "left_file_examples": " | ".join(left_group["relative_path"].head(4)),
                    "right_file_examples": " | ".join(right_group["relative_path"].head(4)),
                }
            )
    columns = [
        "left_split",
        "right_split",
        "identifier",
        "left_image_count",
        "right_image_count",
        "left_classes",
        "right_classes",
        "left_file_examples",
        "right_file_examples",
    ]
    return pd.DataFrame.from_records(rows, columns=columns), summaries


def label_conflicts(images: pd.DataFrame) -> pd.DataFrame:
    valid = images[images["identifier"].notna() & images["class_name"].notna()].copy()
    rows: list[dict[str, Any]] = []
    scopes: Iterable[tuple[str, pd.DataFrame]] = [
        *( (split, valid[valid["split"] == split]) for split in SPLITS ),
        ("전체", valid),
    ]
    for scope, subset in scopes:
        for identifier, group in subset.groupby("identifier", sort=True):
            labels = sorted(group["class_name"].unique())
            if len(labels) <= 1:
                continue
            rows.append(
                {
                    "scope": scope,
                    "identifier": identifier,
                    "identifier_type": group["identifier_type"].iloc[0],
                    "image_count": int(len(group)),
                    "split_count": int(group["split"].nunique()),
                    "splits": "|".join(sorted(group["split"].unique())),
                    "label_count": len(labels),
                    "class_names": "|".join(labels),
                    "has_bacterial_and_viral": {
                        "BACTERIAL_PNEUMONIA",
                        "VIRAL_PNEUMONIA",
                    }.issubset(labels),
                    "file_examples": " | ".join(group["relative_path"].head(10)),
                }
            )
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "scope",
            "identifier",
            "identifier_type",
            "image_count",
            "split_count",
            "splits",
            "label_count",
            "class_names",
            "has_bacterial_and_viral",
            "file_examples",
        ],
    )


def exact_duplicates(images: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for digest, group in images.groupby("sha256", sort=True):
        if len(group) <= 1:
            continue
        labels = sorted(group["class_name"].dropna().unique())
        rows.append(
            {
                "sha256": digest,
                "image_count": int(len(group)),
                "split_count": int(group["split"].nunique()),
                "splits": "|".join(sorted(group["split"].unique())),
                "identifier_count": int(group["identifier"].nunique(dropna=True)),
                "identifiers": "|".join(sorted(group["identifier"].dropna().unique())),
                "label_count": len(labels),
                "class_names": "|".join(labels),
                "cross_split": group["split"].nunique() > 1,
                "label_conflict": len(labels) > 1,
                "file_examples": " | ".join(group["relative_path"].head(10)),
            }
        )
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "sha256",
            "image_count",
            "split_count",
            "splits",
            "identifier_count",
            "identifiers",
            "label_count",
            "class_names",
            "cross_split",
            "label_conflict",
            "file_examples",
        ],
    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def print_summary(summary: dict[str, Any], distribution: pd.DataFrame) -> None:
    print("\n=== CXR 데이터셋 감사 요약 ===")
    print(f"데이터 루트: {summary['data_root']}")
    print(f"전체 이미지: {summary['total_images']:,}장")
    print("\n[Split별 3클래스 분포]")
    for split in (*SPLITS, "전체"):
        print(f"- {split}")
        subset = distribution[distribution["split"] == split]
        for row in subset.itertuples(index=False):
            print(f"  {row.label}: {row.class_name} = {row.count:,}장 ({row.ratio_percent:.2f}%)")

    print("\n[식별자 요약]")
    for row in summary["identifier_summary"]:
        print(
            f"- {row['split']}: 고유 {row['unique_identifiers']:,}개, "
            f"2장 이상 {row['identifiers_with_multiple_images']:,}개, "
            f"다중 라벨 {row['identifiers_with_multiple_labels']:,}개"
        )

    print("\n[Split 간 raw 식별자 중복]")
    for row in summary["cross_split_overlap_summary"]:
        print(f"- {row['left_split']} ↔ {row['right_split']}: {row['overlap_count']:,}개")

    conflicts = summary["label_conflict_summary"]
    print("\n[동일 식별자 라벨 충돌]")
    print(f"- 전체 다중 라벨 식별자: {conflicts['global_conflicting_identifiers']:,}개")
    print(f"- BACTERIAL과 VIRAL이 함께 나타난 식별자: {conflicts['global_bacterial_viral_identifiers']:,}개")
    print(f"- 라벨 판별 실패 이미지: {summary['quality']['unclassified_images']:,}장")
    print(f"- 식별자 추출 실패 이미지: {summary['quality']['missing_identifier_images']:,}장")
    print(f"- 손상 이미지: {summary['quality']['invalid_images']:,}장")
    print(f"- 정확한 중복 해시 그룹: {summary['quality']['exact_duplicate_hash_groups']:,}개")
    print(f"\n결과 저장 위치: {summary['output_dir']}")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images, non_images = scan_dataset(data_root, verify_images=not args.skip_image_verify)
    distribution = class_distribution(images)
    id_summary = identifier_summary(images)
    repeated = repeated_identifiers(images)
    overlaps, overlap_summary = cross_split_overlaps(images)
    conflicts = label_conflicts(images)
    duplicates = exact_duplicates(images)

    unclassified = images[images["classification_issue"].notna()].copy()
    missing_identifiers = images[images["identifier_issue"].notna()].copy()
    invalid_images = images[images["image_issue"].notna()].copy()

    images.to_csv(output_dir / "image_inventory.csv", index=False)
    distribution.to_csv(output_dir / "class_distribution.csv", index=False)
    id_summary.to_csv(output_dir / "identifier_summary.csv", index=False)
    repeated.to_csv(output_dir / "repeated_identifiers.csv", index=False)
    overlaps.to_csv(output_dir / "cross_split_identifier_overlaps.csv", index=False)
    conflicts.to_csv(output_dir / "label_conflicts.csv", index=False)
    duplicates.to_csv(output_dir / "exact_duplicates.csv", index=False)
    unclassified.to_csv(output_dir / "unclassified_images.csv", index=False)
    missing_identifiers.to_csv(output_dir / "missing_identifiers.csv", index=False)
    invalid_images.to_csv(output_dir / "invalid_images.csv", index=False)
    non_images.to_csv(output_dir / "non_image_files.csv", index=False)

    global_conflicts = conflicts[conflicts["scope"] == "전체"]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "audit_policy": {
            "dataset_modified": False,
            "new_split_created": False,
            "identifier_definition": "PNEUMONIA=person숫자, NORMAL=IM-숫자",
            "identifier_limitation": (
                "파일명에서 추출한 문자열이며 실제 환자 ID 또는 전역적으로 유일한 ID임을 보장하지 않음"
            ),
        },
        "total_images": int(len(images)),
        "class_distribution": distribution.to_dict(orient="records"),
        "identifier_summary": id_summary.to_dict(orient="records"),
        "cross_split_overlap_summary": overlap_summary,
        "label_conflict_summary": {
            "global_conflicting_identifiers": int(len(global_conflicts)),
            "global_bacterial_viral_identifiers": int(
                global_conflicts["has_bacterial_and_viral"].sum()
            ),
            "within_split_conflicting_identifiers": {
                split: int(len(conflicts[conflicts["scope"] == split])) for split in SPLITS
            },
        },
        "quality": {
            "unclassified_images": int(len(unclassified)),
            "missing_identifier_images": int(len(missing_identifiers)),
            "invalid_images": int(len(invalid_images)),
            "zero_byte_images": int((images["bytes"] == 0).sum()),
            "non_image_files": int(len(non_images)),
            "exact_duplicate_hash_groups": int(len(duplicates)),
            "exact_duplicate_rows": int(duplicates["image_count"].sum()) if len(duplicates) else 0,
            "cross_split_exact_duplicate_groups": int(duplicates["cross_split"].sum()) if len(duplicates) else 0,
            "conflicting_label_exact_duplicate_groups": int(duplicates["label_conflict"].sum()) if len(duplicates) else 0,
            "image_modes": dict(Counter(images["mode"].dropna())),
            "minimum_width": int(images["width"].dropna().min()),
            "maximum_width": int(images["width"].dropna().max()),
            "minimum_height": int(images["height"].dropna().min()),
            "maximum_height": int(images["height"].dropna().max()),
        },
    }
    write_json(output_dir / "audit_summary.json", summary)
    print_summary(summary, distribution)


if __name__ == "__main__":
    main()
