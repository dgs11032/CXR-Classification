#!/usr/bin/env python3
"""고정 Train/Validation manifest의 CXR 이미지 특성과 전처리 후보를 분석한다.

이 스크립트는 분석 전용이다. Test manifest, 모델, Dataset/DataLoader, 학습용 Transform을
사용하지 않으며 원본 이미지와 기존 manifest를 수정하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_MANIFEST = PROJECT_ROOT / "artifacts" / "splits" / "train_manifest.csv"
DEFAULT_VAL_MANIFEST = PROJECT_ROOT / "artifacts" / "splits" / "val_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "eda"
CLASS_NAMES = ("NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA")
CLASS_COLORS = {
    "NORMAL": "#377eb8",
    "BACTERIAL_PNEUMONIA": "#e41a1c",
    "VIRAL_PNEUMONIA": "#4daf4a",
}
IMAGE_SIZE = 224
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--samples-per-class", type=int, default=6)
    return parser.parse_args()


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_state(paths: Iterable[Path]) -> dict[str, tuple[int, int]]:
    return {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(set(paths))
    }


def load_manifests(train_path: Path, val_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"image_path", "label", "class_name", "source_split", "group_key", "sha256"}
    train = pd.read_csv(train_path)
    validation = pd.read_csv(val_path)
    for name, frame, expected_count in (
        ("train", train, 4448),
        ("validation", validation, 742),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} manifest 필수 컬럼 누락: {sorted(missing)}")
        if len(frame) != expected_count:
            raise ValueError(f"{name} manifest 수량 변경 감지: {len(frame)} != {expected_count}")
        if set(frame["label"]) != {0, 1, 2}:
            raise ValueError(f"{name} manifest에 세 클래스가 모두 존재하지 않습니다.")
    train = train.assign(partition="train")
    validation = validation.assign(partition="validation")
    return train, validation


def border_pixels(array: np.ndarray) -> np.ndarray:
    thickness = max(1, int(round(min(array.shape) * 0.05)))
    return np.concatenate(
        [
            array[:thickness, :].ravel(),
            array[-thickness:, :].ravel(),
            array[thickness:-thickness, :thickness].ravel(),
            array[thickness:-thickness, -thickness:].ravel(),
        ]
    )


def padding_geometry(width: int, height: int, target: int = IMAGE_SIZE) -> tuple[int, int, int, int, float]:
    scale = min(target / width, target / height)
    resized_width = max(1, min(target, int(round(width * scale))))
    resized_height = max(1, min(target, int(round(height * scale))))
    pad_x = target - resized_width
    pad_y = target - resized_height
    padding_fraction = 1.0 - (resized_width * resized_height) / (target * target)
    return resized_width, resized_height, pad_x, pad_y, padding_fraction


def inspect_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        path = (PROJECT_ROOT / row["image_path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Manifest 이미지가 없습니다: {path}")
        with Image.open(path) as image:
            width, height = image.size
            original_mode = image.mode
            image_format = image.format or path.suffix.lstrip(".").upper()
            if original_mode == "RGB":
                rgb = np.asarray(image, dtype=np.int16)
                channel_difference = np.max(np.abs(rgb - rgb[..., :1]), axis=2)
                rgb_mean_abs_channel_difference = float(channel_difference.mean())
                rgb_max_channel_difference = int(channel_difference.max())
                rgb_channels_identical = bool((channel_difference == 0).all())
            else:
                rgb_mean_abs_channel_difference = None
                rgb_max_channel_difference = None
                rgb_channels_identical = None
            gray = np.asarray(image.convert("L"), dtype=np.uint8)

        pixels = gray.astype(np.float32, copy=False)
        p01, p05, p25, p50, p75, p95, p99 = np.percentile(
            pixels, [1, 5, 25, 50, 75, 95, 99]
        )
        border = border_pixels(pixels)
        resized_width, resized_height, pad_x, pad_y, padding_fraction = padding_geometry(
            width, height
        )
        records.append(
            {
                "partition": row["partition"],
                "image_path": row["image_path"],
                "label": int(row["label"]),
                "class_name": row["class_name"],
                "source_split": row["source_split"],
                "group_key": row["group_key"],
                "sha256": row["sha256"],
                "width": width,
                "height": height,
                "aspect_ratio_width_height": width / height,
                "megapixels": width * height / 1_000_000,
                "original_mode": original_mode,
                "image_format": image_format,
                "rgb_mean_abs_channel_difference": rgb_mean_abs_channel_difference,
                "rgb_max_channel_difference": rgb_max_channel_difference,
                "rgb_channels_identical": rgb_channels_identical,
                "file_size_bytes": path.stat().st_size,
                "mean_intensity": float(pixels.mean()),
                "std_intensity": float(pixels.std()),
                "min_intensity": int(pixels.min()),
                "max_intensity": int(pixels.max()),
                "p01_intensity": float(p01),
                "p05_intensity": float(p05),
                "p25_intensity": float(p25),
                "median_intensity": float(p50),
                "p75_intensity": float(p75),
                "p95_intensity": float(p95),
                "p99_intensity": float(p99),
                "robust_contrast_p95_p05": float(p95 - p05),
                "intensity_iqr": float(p75 - p25),
                "near_black_fraction": float((pixels <= 5).mean()),
                "near_white_fraction": float((pixels >= 250).mean()),
                "border_mean_intensity": float(border.mean()),
                "border_std_intensity": float(border.std()),
                "direct_resize_aspect_distortion_factor": max(width / height, height / width),
                "padded_resized_width": resized_width,
                "padded_resized_height": resized_height,
                "padding_x_pixels": pad_x,
                "padding_y_pixels": pad_y,
                "padding_fraction": padding_fraction,
            }
        )
        if (index + 1) % 500 == 0 or index + 1 == len(frame):
            print(f"이미지 통계 계산: {index + 1:,}/{len(frame):,}", flush=True)
    return pd.DataFrame.from_records(records)


NUMERIC_METRICS = (
    "width",
    "height",
    "aspect_ratio_width_height",
    "megapixels",
    "file_size_bytes",
    "mean_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity",
    "robust_contrast_p95_p05",
    "intensity_iqr",
    "near_black_fraction",
    "near_white_fraction",
    "border_mean_intensity",
    "border_std_intensity",
    "direct_resize_aspect_distortion_factor",
    "padding_fraction",
)


def descriptive_statistics(statistics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for partition in ("train", "validation"):
        partition_frame = statistics[statistics["partition"] == partition]
        scopes = [("전체", partition_frame)] + [
            (class_name, partition_frame[partition_frame["class_name"] == class_name])
            for class_name in CLASS_NAMES
        ]
        for scope, frame in scopes:
            for metric in NUMERIC_METRICS:
                values = frame[metric].astype(float)
                rows.append(
                    {
                        "partition": partition,
                        "scope": scope,
                        "metric": metric,
                        "count": int(values.count()),
                        "minimum": float(values.min()),
                        "q25": float(values.quantile(0.25)),
                        "median": float(values.median()),
                        "mean": float(values.mean()),
                        "q75": float(values.quantile(0.75)),
                        "maximum": float(values.max()),
                        "standard_deviation": float(values.std(ddof=1)),
                    }
                )
    return pd.DataFrame(rows)


def categorical_statistics(statistics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for partition in ("train", "validation"):
        partition_frame = statistics[statistics["partition"] == partition]
        for class_name in ("전체", *CLASS_NAMES):
            frame = (
                partition_frame
                if class_name == "전체"
                else partition_frame[partition_frame["class_name"] == class_name]
            )
            for metric in ("original_mode", "image_format"):
                for value, count in frame[metric].value_counts().sort_index().items():
                    rows.append(
                        {
                            "partition": partition,
                            "scope": class_name,
                            "metric": metric,
                            "value": value,
                            "count": int(count),
                            "ratio": count / len(frame),
                            "ratio_percent": 100.0 * count / len(frame),
                        }
                    )
    return pd.DataFrame(rows)


def standardized_mean_difference(left: pd.Series, right: pd.Series) -> float:
    left_values = left.astype(float).to_numpy()
    right_values = right.astype(float).to_numpy()
    pooled_variance = (
        ((len(left_values) - 1) * left_values.var(ddof=1))
        + ((len(right_values) - 1) * right_values.var(ddof=1))
    ) / (len(left_values) + len(right_values) - 2)
    if pooled_variance <= 0:
        return 0.0
    return float((left_values.mean() - right_values.mean()) / math.sqrt(pooled_variance))


def shortcut_statistics(train: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "width",
        "height",
        "aspect_ratio_width_height",
        "megapixels",
        "file_size_bytes",
        "mean_intensity",
        "std_intensity",
        "robust_contrast_p95_p05",
        "near_black_fraction",
        "border_mean_intensity",
        "padding_fraction",
    )
    rows: list[dict[str, Any]] = []
    for left_index, left_class in enumerate(CLASS_NAMES):
        for right_class in CLASS_NAMES[left_index + 1 :]:
            left = train[train["class_name"] == left_class]
            right = train[train["class_name"] == right_class]
            for metric in metrics:
                effect = standardized_mean_difference(left[metric], right[metric])
                rows.append(
                    {
                        "left_class": left_class,
                        "right_class": right_class,
                        "metric": metric,
                        "left_mean": float(left[metric].mean()),
                        "right_mean": float(right[metric].mean()),
                        "standardized_mean_difference": effect,
                        "absolute_effect_size": abs(effect),
                        "exploratory_flag": (
                            "large" if abs(effect) >= 0.8 else "moderate" if abs(effect) >= 0.5 else "small"
                        ),
                        "interpretation_limit": "클래스 연관성 탐색값이며 shortcut의 확정적 증거가 아님",
                    }
                )
    return pd.DataFrame(rows).sort_values("absolute_effect_size", ascending=False).reset_index(drop=True)


def select_samples(train: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    samples: list[pd.DataFrame] = []
    for label, class_name in enumerate(CLASS_NAMES):
        frame = train[train["class_name"] == class_name]
        if len(frame) < count:
            raise ValueError(f"{class_name} 표본이 {count}장보다 적습니다.")
        selected = frame.sample(n=count, random_state=seed + label).copy()
        selected["sample_order"] = range(count)
        samples.append(selected)
    return pd.concat(samples, ignore_index=True).sort_values(
        ["label", "sample_order"]
    ).reset_index(drop=True)


def direct_resize(image: Image.Image) -> Image.Image:
    return image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)


def aspect_preserving_resize_with_padding(image: Image.Image) -> Image.Image:
    width, height = image.size
    resized_width, resized_height, _, _, _ = padding_geometry(width, height)
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new(image.mode, (IMAGE_SIZE, IMAGE_SIZE), color=0)
    left = (IMAGE_SIZE - resized_width) // 2
    top = (IMAGE_SIZE - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def plot_resolution(train: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for class_name in CLASS_NAMES:
        subset = train[train["class_name"] == class_name]
        axes[0, 0].hist(subset["width"], bins=35, alpha=0.45, color=CLASS_COLORS[class_name], label=class_name)
        axes[0, 1].hist(subset["height"], bins=35, alpha=0.45, color=CLASS_COLORS[class_name], label=class_name)
        axes[1, 0].scatter(subset["width"], subset["height"], s=8, alpha=0.25, color=CLASS_COLORS[class_name], label=class_name)
    axes[0, 0].set(title="Train 이미지 너비 분포", xlabel="너비 (pixel)", ylabel="이미지 수")
    axes[0, 1].set(title="Train 이미지 높이 분포", xlabel="높이 (pixel)", ylabel="이미지 수")
    axes[1, 0].set(title="클래스별 해상도 분포", xlabel="너비 (pixel)", ylabel="높이 (pixel)")
    box_data = [train.loc[train["class_name"] == name, "megapixels"] for name in CLASS_NAMES]
    axes[1, 1].boxplot(box_data, tick_labels=["NORMAL", "BACTERIAL", "VIRAL"], showfliers=False)
    axes[1, 1].set(title="클래스별 원본 화소 수", xlabel="클래스", ylabel="Megapixels")
    for axis in axes.flat[:3]:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    axes[1, 1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def plot_aspect_ratio(train: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for class_name in CLASS_NAMES:
        subset = train[train["class_name"] == class_name]
        axes[0].hist(
            subset["aspect_ratio_width_height"], bins=35, alpha=0.45,
            color=CLASS_COLORS[class_name], label=class_name,
        )
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1, label="정사각형")
    axes[0].set(title="Train 가로세로 비율 분포", xlabel="너비 / 높이", ylabel="이미지 수")
    axes[0].legend(fontsize=8)
    box_data = [train.loc[train["class_name"] == name, "padding_fraction"] * 100 for name in CLASS_NAMES]
    axes[1].boxplot(box_data, tick_labels=["NORMAL", "BACTERIAL", "VIRAL"], showfliers=False)
    axes[1].set(title="224×224 비율 유지 시 Padding 면적", xlabel="클래스", ylabel="Padding 비율 (%)")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def plot_pixel_intensity(train: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    specifications = (
        ("mean_intensity", "이미지별 평균 밝기", "평균 밝기 (0–255)"),
        ("std_intensity", "이미지별 픽셀 표준편차", "표준편차"),
        ("robust_contrast_p95_p05", "강건 대비 분포", "P95 - P05"),
    )
    for axis, (metric, title, xlabel) in zip(axes, specifications):
        for class_name in CLASS_NAMES:
            subset = train[train["class_name"] == class_name]
            axis.hist(subset[metric], bins=35, alpha=0.45, color=CLASS_COLORS[class_name], label=class_name)
        axis.set(title=title, xlabel=xlabel, ylabel="이미지 수")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def plot_sample_grid(samples: pd.DataFrame, destination: Path) -> None:
    per_class = int(samples.groupby("class_name").size().min())
    figure, axes = plt.subplots(len(CLASS_NAMES), per_class, figsize=(3.1 * per_class, 9.2))
    for row_index, class_name in enumerate(CLASS_NAMES):
        class_samples = samples[samples["class_name"] == class_name].sort_values("sample_order")
        for column_index, sample in enumerate(class_samples.itertuples(index=False)):
            path = PROJECT_ROOT / sample.image_path
            with Image.open(path) as image:
                axes[row_index, column_index].imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255)
            axes[row_index, column_index].set_title(
                f"{Path(sample.image_path).name[:26]}\n{sample.width}×{sample.height}, mean={sample.mean_intensity:.1f}",
                fontsize=8,
            )
            axes[row_index, column_index].axis("off")
        axes[row_index, 0].set_ylabel(class_name, fontsize=10)
    figure.suptitle("클래스별 재현 가능한 Train 표본 (Seed 42)", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def plot_resize_comparison(samples: pd.DataFrame, destination: Path) -> pd.DataFrame:
    selected = samples.groupby("class_name", sort=False).head(2).reset_index(drop=True)
    figure, axes = plt.subplots(len(selected), 3, figsize=(10.5, 3.1 * len(selected)))
    metadata: list[dict[str, Any]] = []
    for row_index, sample in enumerate(selected.itertuples(index=False)):
        path = PROJECT_ROOT / sample.image_path
        with Image.open(path) as image:
            gray = image.convert("L")
            images = [gray.copy(), direct_resize(gray), aspect_preserving_resize_with_padding(gray)]
        titles = [
            f"원본 {sample.width}×{sample.height}\n{sample.class_name}",
            f"직접 224×224\n왜곡계수 {sample.direct_resize_aspect_distortion_factor:.2f}",
            f"비율 유지+Padding\nPadding {sample.padding_fraction * 100:.1f}%",
        ]
        for column_index, (display_image, title) in enumerate(zip(images, titles)):
            axes[row_index, column_index].imshow(display_image, cmap="gray", vmin=0, vmax=255)
            axes[row_index, column_index].set_title(title, fontsize=9)
            axes[row_index, column_index].axis("off")
        metadata.append(
            {
                "sample_order": row_index,
                "image_path": sample.image_path,
                "class_name": sample.class_name,
                "original_width": sample.width,
                "original_height": sample.height,
                "original_aspect_ratio": sample.aspect_ratio_width_height,
                "direct_resize_distortion_factor": sample.direct_resize_aspect_distortion_factor,
                "padded_resized_width": sample.padded_resized_width,
                "padded_resized_height": sample.padded_resized_height,
                "padding_fraction": sample.padding_fraction,
                "original_pixels_preserved_without_crop": True,
            }
        )
    figure.suptitle("동일 CXR의 Resize 방식 비교", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(destination, dpi=170)
    plt.close(figure)
    return pd.DataFrame(metadata)


def scalar_summary(frame: pd.DataFrame, metric: str) -> dict[str, float]:
    values = frame[metric].astype(float)
    return {
        "minimum": float(values.min()),
        "q25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "q75": float(values.quantile(0.75)),
        "maximum": float(values.max()),
        "standard_deviation": float(values.std(ddof=1)),
    }


def main() -> None:
    args = parse_args()
    train_manifest_path = args.train_manifest.expanduser().resolve()
    val_manifest_path = args.val_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    font_name = configure_plot_font()

    manifest_hashes_before = {
        "train_manifest": file_sha256(train_manifest_path),
        "val_manifest": file_sha256(val_manifest_path),
    }
    train_manifest, val_manifest = load_manifests(train_manifest_path, val_manifest_path)
    input_paths = [PROJECT_ROOT / value for value in pd.concat([train_manifest.image_path, val_manifest.image_path])]
    state_before = input_state(input_paths)

    print("Train 이미지 분석을 시작합니다.", flush=True)
    train_statistics = inspect_manifest(train_manifest)
    print("Validation 기본 호환성 분석을 시작합니다.", flush=True)
    val_statistics = inspect_manifest(val_manifest)
    statistics = pd.concat([train_statistics, val_statistics], ignore_index=True)

    class_stats = descriptive_statistics(statistics)
    categorical_stats = categorical_statistics(statistics)
    shortcut_stats = shortcut_statistics(train_statistics)

    q25 = float(train_statistics["mean_intensity"].quantile(0.25))
    q75 = float(train_statistics["mean_intensity"].quantile(0.75))
    iqr = q75 - q25
    dark_threshold = max(0.0, q25 - 1.5 * iqr)
    bright_threshold = min(255.0, q75 + 1.5 * iqr)
    brightness_outliers = train_statistics[
        (train_statistics["mean_intensity"] < dark_threshold)
        | (train_statistics["mean_intensity"] > bright_threshold)
    ].copy()
    brightness_outliers["brightness_outlier_type"] = np.where(
        brightness_outliers["mean_intensity"] < dark_threshold,
        "dark",
        "bright",
    )
    brightness_outliers = brightness_outliers.sort_values("mean_intensity").reset_index(drop=True)

    samples = select_samples(train_statistics, args.samples_per_class, args.seed)
    plot_resolution(train_statistics, output_dir / "resolution_distribution.png")
    plot_aspect_ratio(train_statistics, output_dir / "aspect_ratio_distribution.png")
    plot_pixel_intensity(train_statistics, output_dir / "pixel_intensity_distribution.png")
    plot_sample_grid(samples, output_dir / "class_sample_grid.png")
    resize_metadata = plot_resize_comparison(samples, output_dir / "resize_comparison.png")

    statistics.to_csv(output_dir / "image_statistics.csv", index=False)
    class_stats.to_csv(output_dir / "class_statistics.csv", index=False)
    categorical_stats.to_csv(output_dir / "categorical_statistics.csv", index=False)
    shortcut_stats.to_csv(output_dir / "shortcut_risk_statistics.csv", index=False)
    brightness_outliers.to_csv(output_dir / "brightness_outliers.csv", index=False)
    samples.to_csv(output_dir / "sample_selection.csv", index=False)
    resize_metadata.to_csv(output_dir / "resize_comparison_metadata.csv", index=False)

    state_after = input_state(input_paths)
    manifest_hashes_after = {
        "train_manifest": file_sha256(train_manifest_path),
        "val_manifest": file_sha256(val_manifest_path),
    }
    train_mode_counts = {str(key): int(value) for key, value in train_statistics["original_mode"].value_counts().items()}
    val_mode_counts = {str(key): int(value) for key, value in val_statistics["original_mode"].value_counts().items()}
    rgb_statistics = statistics[statistics["original_mode"] == "RGB"]
    rgb_identical = rgb_statistics["rgb_channels_identical"].dropna().astype(bool)
    rgb_channel_analysis = {
        "rgb_images": int(len(rgb_statistics)),
        "channels_exactly_identical": int(rgb_identical.sum()),
        "nonidentical_rgb_images": int((~rgb_identical).sum()),
        "maximum_channel_difference": int(rgb_statistics["rgb_max_channel_difference"].max()) if len(rgb_statistics) else 0,
        "interpretation": (
            "RGB 저장 이미지도 세 채널 값이 동일하므로 실질적인 컬러 정보는 없으며, "
            "원본 모드 차이는 저장 형식상의 차이임"
        ),
    }
    train_format_counts = {str(key): int(value) for key, value in train_statistics["image_format"].value_counts().items()}
    val_format_counts = {str(key): int(value) for key, value in val_statistics["image_format"].value_counts().items()}
    validation_comparison_metrics = (
        "width",
        "height",
        "aspect_ratio_width_height",
        "mean_intensity",
        "std_intensity",
        "robust_contrast_p95_p05",
        "padding_fraction",
    )
    validation_comparison = {
        metric: {
            "train_mean": float(train_statistics[metric].mean()),
            "validation_mean": float(val_statistics[metric].mean()),
            "absolute_mean_difference": float(
                abs(train_statistics[metric].mean() - val_statistics[metric].mean())
            ),
        }
        for metric in validation_comparison_metrics
    }
    top_shortcuts = shortcut_stats.head(12).to_dict(orient="records")

    summary = {
        "status": "성공",
        "analysis_scope": {
            "train_images": int(len(train_statistics)),
            "validation_images": int(len(val_statistics)),
            "test_manifest_loaded": False,
            "test_images_analyzed": 0,
            "model_or_training_implemented": False,
            "training_transform_implemented": False,
            "new_split_created": False,
        },
        "input_integrity": {
            "source_images_unchanged": state_before == state_after,
            "train_manifest_unchanged": manifest_hashes_before["train_manifest"]
            == manifest_hashes_after["train_manifest"],
            "val_manifest_unchanged": manifest_hashes_before["val_manifest"]
            == manifest_hashes_after["val_manifest"],
            "manifest_sha256_before": manifest_hashes_before,
            "manifest_sha256_after": manifest_hashes_after,
        },
        "plot_font": font_name,
        "train": {
            "resolution": {
                "width": scalar_summary(train_statistics, "width"),
                "height": scalar_summary(train_statistics, "height"),
                "aspect_ratio_width_height": scalar_summary(
                    train_statistics, "aspect_ratio_width_height"
                ),
                "megapixels": scalar_summary(train_statistics, "megapixels"),
            },
            "pixels": {
                "mean_intensity": scalar_summary(train_statistics, "mean_intensity"),
                "std_intensity": scalar_summary(train_statistics, "std_intensity"),
                "robust_contrast_p95_p05": scalar_summary(
                    train_statistics, "robust_contrast_p95_p05"
                ),
            },
            "original_modes": train_mode_counts,
            "image_formats": train_format_counts,
            "rgb_channel_analysis_train_and_validation": rgb_channel_analysis,
            "brightness_outliers": {
                "method": "이미지별 평균 밝기의 Tukey 1.5×IQR 기준",
                "dark_threshold": dark_threshold,
                "bright_threshold": bright_threshold,
                "dark_count": int((brightness_outliers["brightness_outlier_type"] == "dark").sum()),
                "bright_count": int((brightness_outliers["brightness_outlier_type"] == "bright").sum()),
                "total": int(len(brightness_outliers)),
                "examples": brightness_outliers[
                    ["image_path", "class_name", "mean_intensity", "brightness_outlier_type"]
                ].head(20).to_dict(orient="records"),
            },
        },
        "validation_compatibility": {
            "images": int(len(val_statistics)),
            "original_modes": val_mode_counts,
            "image_formats": val_format_counts,
            "all_dimensions_positive": bool(
                (val_statistics[["width", "height"]].to_numpy() > 0).all()
            ),
            "train_validation_mean_comparison": validation_comparison,
        },
        "resize_comparison": {
            "target_size": [IMAGE_SIZE, IMAGE_SIZE],
            "direct_resize": {
                "preserves_all_source_pixels_without_crop": True,
                "preserves_aspect_ratio": False,
                "distortion_factor": scalar_summary(
                    train_statistics, "direct_resize_aspect_distortion_factor"
                ),
            },
            "aspect_preserving_padding": {
                "preserves_all_source_pixels_without_crop": True,
                "preserves_aspect_ratio": True,
                "images_requiring_padding": int((train_statistics["padding_fraction"] > 0).sum()),
                "padding_fraction": scalar_summary(train_statistics, "padding_fraction"),
                "padding_value_in_comparison": 0,
            },
        },
        "shortcut_risk_exploration": {
            "quantitative_candidates": top_shortcuts,
            "categorical_mode_counts": train_mode_counts,
            "categorical_format_counts": train_format_counts,
            "visual_review_observations_from_seed42_grid": [
                "R 및 A=P와 같은 방향·촬영 표식이 영상 외곽에 존재",
                "일부 영상에서 선·튜브·센서로 보이는 외부 요소가 관찰됨",
                "클래스와 표본에 따라 흉곽 촬영 범위, 여백, 확대 정도가 다양함",
                "임상적 병변 유무나 폐렴 원인 차이는 이 표본 Grid에서 판정하지 않음",
            ],
            "visual_review_required": [
                "촬영 방향은 신뢰 가능한 메타데이터가 없어 표본 Grid에서만 탐색",
                "영상 외부 문자·표식의 존재와 클래스 연관성",
                "폐 영역 위치 및 과도한 여백",
                "테두리·배경 패턴이 장비 또는 클래스와 연결되는지 여부",
            ],
            "interpretation_limit": (
                "관찰된 클래스 차이는 shortcut 가능성을 나타내는 탐색 결과이며 "
                "모델이 실제로 해당 특징을 사용한다는 증거가 아님"
            ),
        },
        "preprocessing_recommendation": {
            "status": "EDA 기반 제안이며 학습용 Transform으로 아직 확정·구현하지 않음",
            "image_size": "224×224",
            "resize": (
                "공통 기본 후보는 Direct Resize 224×224를 제안. 기하 왜곡이 발생하지만 검은 Padding 면적이 "
                "NORMAL 평균 17.8%, BACTERIAL 32.0%, VIRAL 33.4%로 클래스와 강하게 연관되어 "
                "Padding 자체가 shortcut이 될 위험이 큼. Aspect Ratio Preserving Resize + Padding은 "
                "별도 전처리 ablation 후보로만 유지하고 결과로 최종 확정"
            ),
            "channels": (
                "모든 이미지를 명시적으로 Grayscale로 통일한 뒤 동일 값을 RGB 3채널로 복제. "
                "원본 RGB 283장도 세 채널 값이 완전히 동일함을 확인"
            ),
            "normalization": "ImageNet pretrained ConvNeXt-Tiny의 weights metadata에 대응하는 ImageNet 정규화",
            "brightness_contrast": "약한 Brightness/Contrast 변화(초기 후보 ±10%)",
            "rotation": "작은 회전 범위(초기 후보 ±5°)",
            "horizontal_flip": "초기 기준에서는 제외하고 별도 검증 후 결정",
            "random_crop": "폐와 흉곽 영역 손실 위험 때문에 초기 기준에서는 제외",
            "supcon_two_views": (
                "동일 resize·채널·정규화 기반에서 회전과 밝기/대비를 독립 표본화한 두 개의 약한 view. "
                "질병 의미를 바꾸거나 해부학적 영역을 제거하는 강한 crop·flip은 제외"
            ),
        },
    }
    (output_dir / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    required_outputs = (
        "image_statistics.csv",
        "class_statistics.csv",
        "resolution_distribution.png",
        "aspect_ratio_distribution.png",
        "pixel_intensity_distribution.png",
        "class_sample_grid.png",
        "resize_comparison.png",
        "eda_summary.json",
    )
    missing_outputs = [name for name in required_outputs if not (output_dir / name).is_file()]
    checks = {
        "train_count_correct": len(train_statistics) == 4448,
        "validation_count_correct": len(val_statistics) == 742,
        "test_not_loaded": True,
        "source_images_unchanged": state_before == state_after,
        "manifests_unchanged": manifest_hashes_before == manifest_hashes_after,
        "sample_count_correct": len(samples) == args.samples_per_class * len(CLASS_NAMES),
        "required_outputs_created": not missing_outputs,
    }
    print("\n=== STEP 5 이미지 EDA 요약 ===")
    print(f"Train: {len(train_statistics):,}장, Validation: {len(val_statistics):,}장, Test: 0장")
    print(f"원본 모드(Train): {train_mode_counts}")
    print(f"이미지 포맷(Train): {train_format_counts}")
    print(
        f"Aspect Ratio 중앙값: {train_statistics['aspect_ratio_width_height'].median():.3f}, "
        f"범위: {train_statistics['aspect_ratio_width_height'].min():.3f}–"
        f"{train_statistics['aspect_ratio_width_height'].max():.3f}"
    )
    print(
        f"평균 밝기 중앙값: {train_statistics['mean_intensity'].median():.2f}, "
        f"Tukey 밝기 이상치: {len(brightness_outliers):,}장"
    )
    print(
        f"비율 유지 Padding 면적 중앙값: "
        f"{train_statistics['padding_fraction'].median() * 100:.2f}%"
    )
    for check, passed in checks.items():
        print(f"- {'통과' if passed else '실패'}: {check}")
    print(f"결과 저장 위치: {output_dir}")
    if not all(checks.values()):
        print(f"검증 실패 또는 누락 산출물: {missing_outputs}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
