"""재현 가능한 공통 CXR Dataset/DataLoader 생성 함수."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import CXRManifestDataset, ViewMode
from .transforms import build_transform

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    if not resolved.is_relative_to(project_root):
        raise ValueError(f"프로젝트 루트 밖의 경로는 허용하지 않습니다: {value}")
    return resolved


def load_data_config(
    config_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Test 경로가 없는 STEP 6 YAML 설정을 읽고 검증한다."""
    root = Path(project_root).expanduser().resolve()
    path = _resolve_project_path(config_path, root)
    if not path.is_file():
        raise FileNotFoundError(f"데이터 설정 파일이 없습니다: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("데이터 설정은 YAML mapping이어야 합니다.")
    manifests = config.get("manifests", {})
    if set(manifests) != {"train", "validation"}:
        raise ValueError("STEP 6 manifests에는 train과 validation만 있어야 합니다.")
    serialized = json_like_text(config).lower()
    if "test_manifest" in serialized or "test_manifest.csv" in serialized:
        raise ValueError("STEP 6 설정에는 Test manifest를 포함할 수 없습니다.")
    config["resolved"] = {
        "project_root": str(root),
        "config_path": str(path),
        "train_manifest": str(_resolve_project_path(manifests["train"], root)),
        "validation_manifest": str(_resolve_project_path(manifests["validation"], root)),
    }
    return config


def json_like_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {json_like_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(json_like_text(item) for item in value)
    return str(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_dataset(
    config: Mapping[str, Any],
    partition: str,
    *,
    view_mode: ViewMode | None = None,
    validate_paths: bool = True,
) -> CXRManifestDataset:
    if partition not in ("train", "validation"):
        raise ValueError("STEP 6에서는 train 또는 validation Dataset만 생성할 수 있습니다.")
    resolved = config["resolved"]
    training = partition == "train"
    selected_view_mode = view_mode or str(config["views"][partition])
    return CXRManifestDataset(
        resolved[f"{partition}_manifest"],
        resolved["project_root"],
        build_transform(config["transform"], training=training),
        view_mode=selected_view_mode,  # type: ignore[arg-type]
        validate_paths=validate_paths,
    )


def build_loader(
    dataset: CXRManifestDataset,
    config: Mapping[str, Any],
    *,
    training: bool,
) -> DataLoader:
    loader_config = config["loader"]
    seed = int(config["seed"])
    generator = torch.Generator().manual_seed(seed)
    num_workers = int(loader_config["num_workers"])
    shuffle = bool(
        loader_config["train_shuffle"] if training else loader_config["validation_shuffle"]
    )
    if training and not shuffle:
        raise ValueError("Train DataLoader는 shuffle=True여야 합니다.")
    if not training and shuffle:
        raise ValueError("Validation DataLoader는 shuffle=False여야 합니다.")
    persistent_workers = bool(loader_config["persistent_workers"])
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers=True에는 num_workers > 0이 필요합니다.")
    return DataLoader(
        dataset,
        batch_size=int(loader_config["batch_size"]),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(loader_config["pin_memory"]),
        persistent_workers=persistent_workers,
        drop_last=bool(loader_config["drop_last"]),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def build_dataloaders(config: Mapping[str, Any]) -> tuple[DataLoader, DataLoader]:
    """기본 two-view Train과 deterministic single-view Validation loader를 만든다."""
    seed_everything(int(config["seed"]))
    train_dataset = build_dataset(config, "train")
    validation_dataset = build_dataset(config, "validation", view_mode="single")
    train_loader = build_loader(train_dataset, config, training=True)
    validation_loader = build_loader(validation_dataset, config, training=False)
    return train_loader, validation_loader


def label_statistics(dataset: CXRManifestDataset) -> dict[str, Any]:
    counts = dataset.class_counts()
    total = len(dataset)
    return {
        "total": total,
        "counts": counts,
        "ratios": {name: count / total for name, count in counts.items()},
        "ratio_percent": {name: 100.0 * count / total for name, count in counts.items()},
    }


def print_label_statistics(name: str, dataset: CXRManifestDataset) -> None:
    statistics = label_statistics(dataset)
    print(f"{name}: {statistics['total']:,}장")
    for class_name, count in statistics["counts"].items():
        print(
            f"- {class_name}: {count:,}장 "
            f"({statistics['ratio_percent'][class_name]:.2f}%)"
        )
