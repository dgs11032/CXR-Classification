#!/usr/bin/env python3
"""감사된 CXR 원본 train에서 E0~E3 공통 Train/Validation manifest를 생성한다.

원본 이미지는 읽기 전용으로 다루며 이동·복사·삭제·수정하지 않는다. group_key는
실제 환자 ID가 아니라 클래스 네임스페이스를 포함한 파일명 기반 proxy group이다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "chest_xray" / "chest_xray"
DEFAULT_AUDIT_INVENTORY = PROJECT_ROOT / "artifacts" / "data_audit" / "image_inventory.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "splits"
CLASS_NAMES = ("NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA")
EXPECTED_TEST_IMAGES = 624
EXPECTED_EXCLUDED_SOURCE_VAL_IMAGES = 16


class UnionFind:
    """정확한 중복 이미지가 연결한 proxy group을 하나의 split 단위로 묶는다."""

    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--audit-inventory", type=Path, default=DEFAULT_AUDIT_INVENTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def dataset_state(data_root: Path) -> dict[str, tuple[int, int]]:
    """원본 변경 여부를 검증하기 위한 경로별 (크기, mtime_ns) 스냅샷을 만든다."""
    return {
        path.relative_to(data_root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(data_root.rglob("*"))
        if path.is_file()
    }


def proxy_group_key(row: pd.Series) -> str:
    """클래스별 번호 네임스페이스를 분리한 파일명 기반 proxy key를 만든다."""
    identifier = str(row["identifier"]).upper().replace("-", "_")
    label = int(row["label"])
    if label == 0:
        return f"NORMAL_{identifier}"
    if label == 1:
        return f"BACTERIAL_{identifier}"
    if label == 2:
        return f"VIRAL_{identifier}"
    raise ValueError(f"지원하지 않는 라벨: {label}")


def add_proxy_groups(source_train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """proxy key를 만들고 정확한 중복으로 연결된 key를 동일 component로 병합한다."""
    frame = source_train.copy()
    frame["proxy_group_key"] = frame.apply(proxy_group_key, axis=1)
    proxy_keys = sorted(frame["proxy_group_key"].unique())
    union_find = UnionFind(proxy_keys)

    for _, duplicate_group in frame.groupby("sha256", sort=True):
        keys = sorted(duplicate_group["proxy_group_key"].unique())
        for key in keys[1:]:
            union_find.union(keys[0], key)

    components: dict[str, list[str]] = defaultdict(list)
    for key in proxy_keys:
        components[union_find.find(key)].append(key)

    final_keys: dict[str, str] = {}
    linked_components: dict[str, list[str]] = {}
    for members in components.values():
        members = sorted(members)
        if len(members) == 1:
            final_key = members[0]
        else:
            component_hash = hashlib.sha256("|".join(members).encode("utf-8")).hexdigest()[:12]
            final_key = f"DUPLICATE_COMPONENT_{component_hash}"
            linked_components[final_key] = members
        for member in members:
            final_keys[member] = final_key

    frame["group_key"] = frame["proxy_group_key"].map(final_keys)
    if frame["group_key"].isna().any():
        raise RuntimeError("일부 proxy group을 최종 group_key로 변환하지 못했습니다.")
    return frame, linked_components


def deduplicate_development(
    source_train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SHA-256별 경로가 가장 빠른 대표 1장만 남기고 제외 대응표를 만든다."""
    representatives: list[pd.Series] = []
    exclusions: list[dict[str, Any]] = []
    for digest, group in source_train.groupby("sha256", sort=True):
        ordered = group.sort_values("relative_path").reset_index(drop=True)
        labels = sorted(int(value) for value in ordered["label"].unique())
        if len(labels) != 1:
            raise ValueError(
                f"동일 SHA-256에 서로 다른 라벨이 있어 임의 처리할 수 없습니다: {digest}"
            )
        representative = ordered.iloc[0]
        representatives.append(representative)
        for _, excluded in ordered.iloc[1:].iterrows():
            exclusions.append(
                {
                    "sha256": digest,
                    "representative_path": representative["relative_path"],
                    "excluded_path": excluded["relative_path"],
                    "label": int(representative["label"]),
                    "class_name": representative["class_name"],
                    "representative_proxy_group_key": representative["proxy_group_key"],
                    "excluded_proxy_group_key": excluded["proxy_group_key"],
                    "final_group_key": representative["group_key"],
                    "reason": "동일 SHA-256 중복 — 대표 이미지 1장만 개발용 분할에 포함",
                }
            )

    development = pd.DataFrame(representatives).sort_values("relative_path").reset_index(drop=True)
    exclusion_columns = [
        "sha256",
        "representative_path",
        "excluded_path",
        "label",
        "class_name",
        "representative_proxy_group_key",
        "excluded_proxy_group_key",
        "final_group_key",
        "reason",
    ]
    return development, pd.DataFrame.from_records(exclusions, columns=exclusion_columns)


def split_score(
    frame: pd.DataFrame,
    validation_indices: np.ndarray,
    target_fraction: float,
) -> tuple[float, float, float]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    overall_ratio = np.bincount(labels, minlength=len(CLASS_NAMES)) / len(labels)
    validation_labels = labels[validation_indices]
    validation_ratio = np.bincount(
        validation_labels, minlength=len(CLASS_NAMES)
    ) / len(validation_labels)
    size_error = abs(len(validation_indices) / len(frame) - target_fraction)
    distribution_error = float(np.abs(overall_ratio - validation_ratio).sum())
    return 5.0 * size_error + distribution_error, size_error, distribution_error


def create_group_split(
    development: pd.DataFrame,
    *,
    seed: int,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """StratifiedGroupKFold 후보 중 목표 비율·클래스 분포가 가장 가까운 fold를 선택한다."""
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction은 0과 0.5 사이여야 합니다.")
    n_splits = max(2, round(1.0 / validation_fraction))
    labels = development["label"].to_numpy(dtype=np.int64)
    groups = development["group_key"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    candidates: list[tuple[tuple[float, float, float], int, np.ndarray, np.ndarray]] = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(development)), labels, groups)
    ):
        candidates.append(
            (
                split_score(development, validation_indices, validation_fraction),
                fold,
                train_indices,
                validation_indices,
            )
        )
    score, selected_fold, train_indices, validation_indices = min(
        candidates, key=lambda candidate: (candidate[0], candidate[1])
    )
    train = development.iloc[train_indices].copy().sort_values("relative_path").reset_index(drop=True)
    validation = (
        development.iloc[validation_indices]
        .copy()
        .sort_values("relative_path")
        .reset_index(drop=True)
    )
    metadata = {
        "method": "StratifiedGroupKFold 후보 중 목표 크기·클래스 분포 오차 최소 fold 선택",
        "n_splits": n_splits,
        "selected_fold_zero_based": selected_fold,
        "objective_score": score[0],
        "size_absolute_error": score[1],
        "class_distribution_l1_error": score[2],
    }
    return train, validation, metadata


def manifest_frame(frame: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    """실험 공통 manifest의 최소 필수 컬럼과 감사 컬럼을 정리한다."""
    result = pd.DataFrame(
        {
            "image_path": frame["relative_path"].map(
                lambda value: (Path("chest_xray") / "chest_xray" / str(value)).as_posix()
            ),
            "label": frame["label"].astype(int),
            "class_name": frame["class_name"],
            "source_split": frame["split"],
            "group_key": frame["group_key"],
            "sha256": frame["sha256"],
            "source_identifier": frame["identifier"],
        }
    )
    return result.sort_values("image_path").reset_index(drop=True)


def class_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for label, class_name in enumerate(CLASS_NAMES):
        count = int((frame["label"] == label).sum())
        rows[class_name] = {
            "label": label,
            "count": count,
            "ratio": count / len(frame) if len(frame) else 0.0,
            "ratio_percent": 100.0 * count / len(frame) if len(frame) else 0.0,
        }
    return rows


def validate_manifests(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    source_state_before: dict[str, tuple[int, int]],
    source_state_after: dict[str, tuple[int, int]],
    reproducible: bool,
    target_fraction: float,
) -> dict[str, Any]:
    train_classes = set(train["label"])
    validation_classes = set(validation["label"])
    test_classes = set(test["label"])
    overall_ratios = pd.concat([train, validation])["label"].value_counts(normalize=True)
    validation_ratios = validation["label"].value_counts(normalize=True)
    ratio_differences = {
        CLASS_NAMES[label]: float(
            abs(validation_ratios.get(label, 0.0) - overall_ratios.get(label, 0.0))
        )
        for label in range(len(CLASS_NAMES))
    }

    checks = {
        "train_validation_path_overlap_zero": {
            "passed": len(set(train.image_path) & set(validation.image_path)) == 0,
            "overlap_count": len(set(train.image_path) & set(validation.image_path)),
        },
        "train_validation_sha256_overlap_zero": {
            "passed": len(set(train.sha256) & set(validation.sha256)) == 0,
            "overlap_count": len(set(train.sha256) & set(validation.sha256)),
        },
        "train_validation_proxy_group_overlap_zero": {
            "passed": len(set(train.group_key) & set(validation.group_key)) == 0,
            "overlap_count": len(set(train.group_key) & set(validation.group_key)),
        },
        "all_three_classes_present": {
            "passed": all(
                classes == set(range(len(CLASS_NAMES)))
                for classes in (train_classes, validation_classes, test_classes)
            ),
            "train_labels": sorted(train_classes),
            "validation_labels": sorted(validation_classes),
            "test_labels": sorted(test_classes),
        },
        "validation_distribution_acceptable": {
            "passed": max(ratio_differences.values()) <= 0.02,
            "max_absolute_ratio_difference": max(ratio_differences.values()),
            "per_class_absolute_ratio_difference": ratio_differences,
            "acceptance_threshold": 0.02,
        },
        "test_624_preserved": {
            "passed": len(test) == EXPECTED_TEST_IMAGES,
            "test_images": len(test),
            "expected": EXPECTED_TEST_IMAGES,
        },
        "source_dataset_unchanged": {
            "passed": source_state_before == source_state_after,
            "file_count_before": len(source_state_before),
            "file_count_after": len(source_state_after),
            "bytes_before": sum(value[0] for value in source_state_before.values()),
            "bytes_after": sum(value[0] for value in source_state_after.values()),
        },
        "same_seed_reproducible": {
            "passed": reproducible,
            "seed": 42,
        },
        "validation_size_close_to_target": {
            "passed": abs(len(validation) / (len(train) + len(validation)) - target_fraction) <= 0.02,
            "actual_fraction": len(validation) / (len(train) + len(validation)),
            "target_fraction": target_fraction,
            "acceptance_threshold": 0.02,
        },
    }
    failed = [name for name, result in checks.items() if not result["passed"]]
    return {
        "status": "성공" if not failed else "실패",
        "all_checks_passed": not failed,
        "failed_checks": failed,
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    audit_inventory = args.audit_inventory.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"데이터 루트가 없습니다: {data_root}")
    if not audit_inventory.is_file():
        raise FileNotFoundError(f"감사 inventory가 없습니다: {audit_inventory}")

    source_state_before = dataset_state(data_root)
    inventory = pd.read_csv(audit_inventory)
    required_columns = {
        "split",
        "relative_path",
        "label",
        "class_name",
        "identifier",
        "sha256",
        "classification_issue",
        "identifier_issue",
        "image_issue",
    }
    missing_columns = required_columns - set(inventory.columns)
    if missing_columns:
        raise ValueError(f"감사 inventory 필수 컬럼 누락: {sorted(missing_columns)}")
    if inventory[list(required_columns & {"classification_issue", "identifier_issue", "image_issue"})].notna().any().any():
        raise ValueError("감사 inventory에 라벨·식별자·이미지 오류가 있어 임의 분할하지 않습니다.")
    if len(inventory) != len(source_state_before):
        raise ValueError(
            f"감사 이후 원본 파일 수가 달라졌습니다: inventory={len(inventory)}, source={len(source_state_before)}"
        )
    if not all((data_root / path).is_file() for path in inventory["relative_path"]):
        raise FileNotFoundError("감사 inventory의 일부 이미지가 원본 데이터 루트에 없습니다.")

    source_train = inventory[inventory["split"] == "train"].copy()
    source_validation = inventory[inventory["split"] == "val"].copy()
    source_test = inventory[inventory["split"] == "test"].copy()
    if len(source_train) != 5216:
        raise ValueError(f"기존 train은 5,216장이어야 합니다: {len(source_train)}")
    if len(source_validation) != EXPECTED_EXCLUDED_SOURCE_VAL_IMAGES:
        raise ValueError(f"기존 val은 16장이어야 합니다: {len(source_validation)}")
    if len(source_test) != EXPECTED_TEST_IMAGES:
        raise ValueError(f"기존 test는 624장이어야 합니다: {len(source_test)}")

    grouped_train, linked_duplicate_components = add_proxy_groups(source_train)
    development, exclusions = deduplicate_development(grouped_train)
    train, validation, split_metadata = create_group_split(
        development,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    # 동일 입력·동일 Seed로 두 번째 계산을 수행해 경로 단위 재현성을 확인한다.
    train_again, validation_again, _ = create_group_split(
        development,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    reproducible = (
        train["relative_path"].tolist() == train_again["relative_path"].tolist()
        and validation["relative_path"].tolist()
        == validation_again["relative_path"].tolist()
    )

    test_with_groups, test_linked_components = add_proxy_groups(source_test)
    train_manifest = manifest_frame(train, PROJECT_ROOT)
    validation_manifest = manifest_frame(validation, PROJECT_ROOT)
    test_manifest = manifest_frame(test_with_groups, PROJECT_ROOT)

    source_state_after = dataset_state(data_root)
    validation_report = validate_manifests(
        train_manifest,
        validation_manifest,
        test_manifest,
        source_state_before=source_state_before,
        source_state_after=source_state_after,
        reproducible=reproducible,
        target_fraction=args.validation_fraction,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest.to_csv(output_dir / "train_manifest.csv", index=False)
    validation_manifest.to_csv(output_dir / "val_manifest.csv", index=False)
    test_manifest.to_csv(output_dir / "test_manifest.csv", index=False)
    exclusions.to_csv(output_dir / "duplicate_exclusions.csv", index=False)

    summary = {
        "status": validation_report["status"],
        "seed": args.seed,
        "target_train_fraction": 1.0 - args.validation_fraction,
        "target_validation_fraction": args.validation_fraction,
        "data_root": str(data_root),
        "audit_inventory": str(audit_inventory),
        "group_key_policy": {
            "type": "파일명 기반 클래스별 proxy group",
            "examples": [
                "NORMAL_IM_0001",
                "BACTERIAL_PERSON1000",
                "VIRAL_PERSON1000",
            ],
            "actual_patient_id": False,
            "limitation": (
                "personXXXX와 IM-XXXX의 전역 유일성 및 실제 환자 대응이 확인되지 않아 "
                "환자 단위 독립성을 보장하지 않음"
            ),
            "duplicate_priority": (
                "정확한 SHA-256 중복이 여러 proxy key를 연결하면 동일 component group으로 묶음"
            ),
            "development_linked_duplicate_components": linked_duplicate_components,
            "test_linked_duplicate_components": test_linked_components,
        },
        "duplicate_policy": {
            "source_train_images": int(len(source_train)),
            "unique_development_images": int(len(development)),
            "excluded_duplicate_images": int(len(exclusions)),
            "representative_selection": "동일 SHA-256 그룹에서 relative_path 사전순 첫 이미지",
            "original_files_deleted_or_modified": False,
        },
        "source_data": {
            "existing_train_images": int(len(source_train)),
            "excluded_existing_validation_images": int(len(source_validation)),
            "preserved_test_images": int(len(source_test)),
        },
        "split_method": split_metadata,
        "manifests": {
            "train": {
                "images": int(len(train_manifest)),
                "groups": int(train_manifest["group_key"].nunique()),
                "classes": class_statistics(train_manifest),
            },
            "validation": {
                "images": int(len(validation_manifest)),
                "groups": int(validation_manifest["group_key"].nunique()),
                "classes": class_statistics(validation_manifest),
            },
            "test": {
                "images": int(len(test_manifest)),
                "groups": int(test_manifest["group_key"].nunique()),
                "classes": class_statistics(test_manifest),
            },
        },
        "test_usage_policy": "최종 평가용으로 보존하며 모델 선택에 사용하지 않음",
    }
    write_json(output_dir / "split_summary.json", summary)
    write_json(output_dir / "split_validation_report.json", validation_report)

    print("\n=== E0~E3 공통 분할 생성 결과 ===")
    print(f"상태: {validation_report['status']}")
    print(f"기존 train: {len(source_train):,}장")
    print(f"정확한 중복 제외: {len(exclusions):,}장")
    print(f"개발용 고유 이미지: {len(development):,}장")
    for name, manifest in (
        ("새 Train", train_manifest),
        ("새 Validation", validation_manifest),
        ("보존 Test", test_manifest),
    ):
        print(f"\n{name}: {len(manifest):,}장 / proxy group {manifest['group_key'].nunique():,}개")
        statistics = class_statistics(manifest)
        for class_name in CLASS_NAMES:
            row = statistics[class_name]
            print(f"- {row['label']} {class_name}: {row['count']:,}장 ({row['ratio_percent']:.2f}%)")
    print("\n[검증]")
    for name, result in validation_report["checks"].items():
        print(f"- {'통과' if result['passed'] else '실패'}: {name}")
    print(f"\n결과 저장 위치: {output_dir}")

    if not validation_report["all_checks_passed"]:
        print(
            "필수 검증 조건을 만족하지 못해 성공으로 처리하지 않습니다: "
            + ", ".join(validation_report["failed_checks"]),
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
