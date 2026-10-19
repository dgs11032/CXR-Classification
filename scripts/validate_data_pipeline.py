#!/usr/bin/env python3
"""STEP 6 공통 Dataset/Transform/DataLoader를 실제 CXR로 검증한다.

학습, 모델, Test manifest 접근 없이 Train/Validation 입력 파이프라인만 검사한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataloader import (  # noqa: E402
    build_dataloaders,
    build_dataset,
    build_loader,
    label_statistics,
    load_data_config,
    print_label_statistics,
    seed_everything,
)
from data.dataset import CXRManifestDataset  # noqa: E402
from data.transforms import build_transform, convnext_weights_metadata, denormalize  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data_config.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "data_pipeline"
EXPECTED_TRAIN_COUNTS = {
    "NORMAL": 1148,
    "BACTERIAL_PNEUMONIA": 2153,
    "VIRAL_PNEUMONIA": 1147,
}
EXPECTED_VALIDATION_COUNTS = {
    "NORMAL": 192,
    "BACTERIAL_PNEUMONIA": 359,
    "VIRAL_PNEUMONIA": 191,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_state(paths: Iterable[Path]) -> dict[str, tuple[int, int]]:
    return {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(set(paths))
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def configure_plot_font() -> str:
    for family in ("AppleGothic", "Arial Unicode MS", "Noto Sans CJK KR"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            plt.rcParams["font.family"] = family
            plt.rcParams["axes.unicode_minus"] = False
            return family
        except ValueError:
            continue
    plt.rcParams["font.family"] = "DejaVu Sans"
    return "DejaVu Sans"


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "minimum": float(tensor.min()),
        "maximum": float(tensor.max()),
        "mean": float(tensor.mean()),
        "standard_deviation": float(tensor.std()),
        "finite": bool(torch.isfinite(tensor).all()),
    }


def select_class_indices(dataset: CXRManifestDataset, seed: int) -> list[int]:
    generator = random.Random(seed)
    indices: list[int] = []
    for class_name in EXPECTED_TRAIN_COUNTS:
        candidates = dataset.frame.index[dataset.frame["class_name"] == class_name].tolist()
        indices.append(generator.choice(candidates))
    return indices


def save_transform_samples(
    dataset: CXRManifestDataset,
    indices: list[int],
    mean: list[float],
    std: list[float],
    destination: Path,
) -> list[dict[str, Any]]:
    seed_everything(42)
    figure, axes = plt.subplots(len(indices), 3, figsize=(10.5, 10.0))
    selections: list[dict[str, Any]] = []
    for row_index, index in enumerate(indices):
        views, label = dataset[index]
        if views.shape != (2, 3, 224, 224):
            raise ValueError(f"시각화 two-view shape 오류: {tuple(views.shape)}")
        source = dataset._load_image(index).convert("L")
        view_1 = denormalize(views[0], mean, std).permute(1, 2, 0).numpy()
        view_2 = denormalize(views[1], mean, std).permute(1, 2, 0).numpy()
        record = dataset.record(index)
        displays = (source, view_1, view_2)
        titles = (
            f"원본\n{Path(str(record['image_path'])).name[:28]}",
            "Train View 1\n역정규화",
            "Train View 2\n역정규화",
        )
        for column_index, (display, title) in enumerate(zip(displays, titles)):
            if column_index == 0:
                axes[row_index, column_index].imshow(display, cmap="gray", vmin=0, vmax=255)
            else:
                axes[row_index, column_index].imshow(display)
            axes[row_index, column_index].set_title(title, fontsize=9)
            axes[row_index, column_index].axis("off")
        axes[row_index, 0].set_ylabel(str(record["class_name"]), fontsize=9)
        selections.append(
            {
                "dataset_index": index,
                "image_path": record["image_path"],
                "label": int(label),
                "class_name": record["class_name"],
                "two_views_bitwise_equal": bool(torch.equal(views[0], views[1])),
            }
        )
    figure.suptitle("동일 원본 CXR에서 독립적으로 생성한 Two Augmented Views", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(destination, dpi=170)
    plt.close(figure)
    return selections


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_font = configure_plot_font()
    config = load_data_config(args.config, project_root=PROJECT_ROOT)
    seed = int(config["seed"])
    seed_everything(seed)

    train_manifest_path = Path(config["resolved"]["train_manifest"])
    validation_manifest_path = Path(config["resolved"]["validation_manifest"])
    manifest_hashes_before = {
        "train": sha256_file(train_manifest_path),
        "validation": sha256_file(validation_manifest_path),
    }

    train_single = build_dataset(config, "train", view_mode="single")
    train_two = build_dataset(config, "train", view_mode="two")
    validation_single = build_dataset(config, "validation", view_mode="single")
    all_source_paths = [
        *(train_single.resolve_path(index) for index in range(len(train_single))),
        *(
            validation_single.resolve_path(index)
            for index in range(len(validation_single))
        ),
    ]
    state_before = source_state(all_source_paths)

    single_image, single_label = train_single[0]
    two_images, two_label = train_two[0]
    validation_first, validation_label = validation_single[0]
    validation_second, _ = validation_single[0]

    # Two-view가 같은 원본을 받는지 확인하기 위해 랜덤 증강이 없는 transform을 두 번 적용한다.
    deterministic_transform = build_transform(config["transform"], training=False)
    same_source_probe = CXRManifestDataset(
        train_manifest_path,
        PROJECT_ROOT,
        deterministic_transform,
        view_mode="two",
    )
    deterministic_views, _ = same_source_probe[0]
    deterministic_two_views_equal = bool(
        torch.equal(deterministic_views[0], deterministic_views[1])
    )

    # 실제 Train transform이 호출마다 독립 RNG를 사용해 일부 View를 다르게 만드는지 확인한다.
    seed_everything(seed)
    two_view_difference_count = 0
    probed_two_view_samples = min(32, len(train_two))
    for index in range(probed_two_view_samples):
        views, _ = train_two[index]
        two_view_difference_count += int(not torch.equal(views[0], views[1]))

    seed_everything(seed)
    train_loader, validation_loader = build_dataloaders(config)
    train_batch_images, train_batch_labels = next(iter(train_loader))
    validation_batch_images, validation_batch_labels = next(iter(validation_loader))

    # 동일 Seed의 독립 loader 두 개에서 sampler 인덱스 순서가 같은지 확인한다.
    sampler_dataset_a = build_dataset(config, "train", view_mode="two")
    sampler_dataset_b = build_dataset(config, "train", view_mode="two")
    sampler_loader_a = build_loader(sampler_dataset_a, config, training=True)
    sampler_loader_b = build_loader(sampler_dataset_b, config, training=True)
    sampler_order_a = list(iter(sampler_loader_a.sampler))
    sampler_order_b = list(iter(sampler_loader_b.sampler))
    shuffle_order_reproducible = sampler_order_a == sampler_order_b

    train_labels = set(train_single.frame["label"].astype(int))
    validation_labels = set(validation_single.frame["label"].astype(int))
    train_distribution = label_statistics(train_single)
    validation_distribution = label_statistics(validation_single)
    metadata = convnext_weights_metadata()
    mean = [float(value) for value in config["transform"]["normalization"]["mean"]]
    std = [float(value) for value in config["transform"]["normalization"]["std"]]
    denormalized_single = denormalize(single_image, mean, std)

    sample_indices = select_class_indices(train_two, seed)
    sample_records = save_transform_samples(
        train_two,
        sample_indices,
        mean,
        std,
        output_dir / "transform_samples.png",
    )

    state_after = source_state(all_source_paths)
    manifest_hashes_after = {
        "train": sha256_file(train_manifest_path),
        "validation": sha256_file(validation_manifest_path),
    }
    config_text = Path(config["resolved"]["config_path"]).read_text(encoding="utf-8").lower()

    checks = {
        "train_dataset_length_4448": {
            "passed": len(train_single) == 4448 and len(train_two) == 4448,
            "single": len(train_single),
            "two": len(train_two),
        },
        "validation_dataset_length_742": {
            "passed": len(validation_single) == 742,
            "actual": len(validation_single),
        },
        "label_range_0_1_2": {
            "passed": train_labels == {0, 1, 2} and validation_labels == {0, 1, 2},
            "train": sorted(train_labels),
            "validation": sorted(validation_labels),
        },
        "single_view_shape": {
            "passed": tuple(single_image.shape) == (3, 224, 224),
            "actual": list(single_image.shape),
        },
        "two_view_shape": {
            "passed": tuple(two_images.shape) == (2, 3, 224, 224),
            "actual": list(two_images.shape),
        },
        "dataloader_batch_shapes": {
            "passed": (
                tuple(train_batch_images.shape[1:]) == (2, 3, 224, 224)
                and tuple(validation_batch_images.shape[1:]) == (3, 224, 224)
                and len(train_batch_labels) == int(config["loader"]["batch_size"])
                and len(validation_batch_labels) == int(config["loader"]["batch_size"])
            ),
            "train_images": list(train_batch_images.shape),
            "train_labels": list(train_batch_labels.shape),
            "validation_images": list(validation_batch_images.shape),
            "validation_labels": list(validation_batch_labels.shape),
        },
        "no_nan_or_inf": {
            "passed": all(
                bool(torch.isfinite(tensor).all())
                for tensor in (
                    single_image,
                    two_images,
                    validation_first,
                    train_batch_images,
                    validation_batch_images,
                )
            ),
        },
        "validation_transform_deterministic": {
            "passed": bool(torch.equal(validation_first, validation_second)),
            "maximum_absolute_difference": float(
                (validation_first - validation_second).abs().max()
            ),
        },
        "two_views_same_source": {
            "passed": deterministic_two_views_equal,
            "method": (
                "CXRManifestDataset이 원본을 한 번 로드해 두 transform 호출에 복사하고, "
                "랜덤 증강을 제거한 probe에서 두 View의 bitwise equality 확인"
            ),
        },
        "two_view_augmentation_independent": {
            "passed": two_view_difference_count > 0,
            "different_pairs": two_view_difference_count,
            "probed_pairs": probed_two_view_samples,
            "note": "증강이 확률적으로 미적용되면 일부 pair는 같을 수 있음",
        },
        "all_image_paths_valid": {
            "passed": all(path.is_file() for path in all_source_paths),
            "checked_paths": len(all_source_paths),
        },
        "class_counts_match_manifests": {
            "passed": (
                train_distribution["counts"] == EXPECTED_TRAIN_COUNTS
                and validation_distribution["counts"] == EXPECTED_VALIDATION_COUNTS
            ),
            "train": train_distribution["counts"],
            "validation": validation_distribution["counts"],
        },
        "test_data_not_accessed": {
            "passed": (
                "test_manifest.csv" not in config_text
                and set(config["manifests"]) == {"train", "validation"}
            ),
            "configured_manifests": sorted(config["manifests"]),
            "test_images_loaded": 0,
        },
        "source_images_unchanged": {
            "passed": state_before == state_after,
            "checked_images": len(state_before),
        },
        "manifests_unchanged": {
            "passed": manifest_hashes_before == manifest_hashes_after,
            "before": manifest_hashes_before,
            "after": manifest_hashes_after,
        },
        "train_shuffle_reproducible": {
            "passed": shuffle_order_reproducible,
            "seed": seed,
            "compared_indices": len(sampler_order_a),
        },
        "denormalized_range_valid": {
            "passed": (
                float(denormalized_single.min()) >= 0.0
                and float(denormalized_single.max()) <= 1.0
            ),
            "minimum": float(denormalized_single.min()),
            "maximum": float(denormalized_single.max()),
        },
    }
    failed_checks = [name for name, result in checks.items() if not result["passed"]]
    validation_report = {
        "status": "성공" if not failed_checks else "실패",
        "all_checks_passed": not failed_checks,
        "failed_checks": failed_checks,
        "checks": checks,
    }
    dataloader_summary = {
        "config_path": config["resolved"]["config_path"],
        "seed": seed,
        "convnext_weights_metadata": metadata,
        "transform": {
            "ordered_steps": [
                "이미지 로드",
                "Grayscale(L) 통일 후 RGB 3채널 복제",
                "Direct Resize 224×224 (BILINEAR, antialias=True)",
                "Train만 Rotation ±5° p=0.5",
                "Train만 Brightness/Contrast ±10% p=0.5",
                "ToTensor",
                "ImageNet Normalize",
            ],
            "validation_random_augmentation": False,
            "horizontal_flip": False,
            "random_crop": False,
            "scale_jitter": False,
            "single_view_summary": tensor_summary(single_image),
            "two_view_summary": tensor_summary(two_images),
            "validation_view_summary": tensor_summary(validation_first),
        },
        "datasets": {
            "train_single": len(train_single),
            "train_two": len(train_two),
            "validation_single": len(validation_single),
        },
        "label_statistics": {
            "train": train_distribution,
            "validation": validation_distribution,
        },
        "loader": {
            "batch_size": int(config["loader"]["batch_size"]),
            "num_workers": int(config["loader"]["num_workers"]),
            "train_shuffle": bool(config["loader"]["train_shuffle"]),
            "validation_shuffle": bool(config["loader"]["validation_shuffle"]),
            "train_batch_shape": list(train_batch_images.shape),
            "validation_batch_shape": list(validation_batch_images.shape),
            "shuffle_order_reproducible": shuffle_order_reproducible,
        },
        "two_view_validation": {
            "same_source_probe_equal_without_random_augmentation": deterministic_two_views_equal,
            "different_augmented_pairs": two_view_difference_count,
            "probed_pairs": probed_two_view_samples,
        },
        "visualization_samples": sample_records,
        "test_access": {
            "manifest_loaded": False,
            "images_loaded": 0,
        },
        "model_or_training": {
            "model_created": False,
            "loss_created": False,
            "optimizer_created": False,
            "backward_called": False,
        },
        "plot_font": plot_font,
    }
    write_json(output_dir / "validation_report.json", validation_report)
    write_json(output_dir / "dataloader_summary.json", dataloader_summary)

    print("\n=== STEP 6 공통 데이터 파이프라인 검증 ===")
    print_label_statistics("Train", train_single)
    print_label_statistics("Validation", validation_single)
    print(f"Train batch: {tuple(train_batch_images.shape)}")
    print(f"Validation batch: {tuple(validation_batch_images.shape)}")
    print(
        f"Validation 반복 입력 최대 차이: "
        f"{checks['validation_transform_deterministic']['maximum_absolute_difference']:.1f}"
    )
    print(
        f"실제 Two-view 차이 발생: {two_view_difference_count}/{probed_two_view_samples} pair"
    )
    for name, result in checks.items():
        print(f"- {'통과' if result['passed'] else '실패'}: {name}")
    print(f"결과 저장 위치: {output_dir}")
    if failed_checks:
        print(
            "검증 실패로 성공 처리하지 않습니다: " + ", ".join(failed_checks),
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
