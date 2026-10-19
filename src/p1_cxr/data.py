"""Dataset inventory, leakage-aware splitting, and image loading."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .config import write_json
from .constants import CLASS_NAMES, CLASS_TO_INDEX, IMAGENET_MEAN, IMAGENET_STD

SUPPORTED_SUFFIXES = {".jpeg", ".jpg", ".png"}
SOURCE_SPLITS = ("train", "val", "test")
_PNEUMONIA_PATIENT_PATTERN = re.compile(r"^(person\d+)", re.IGNORECASE)
_NORMAL_PATIENT_PATTERN = re.compile(r"(IM-\d+)", re.IGNORECASE)


def infer_label(path: Path) -> tuple[str, int]:
    """Infer the three-class target without exposing the filename to the model."""
    parent = path.parent.name.upper()
    name = path.name.lower()
    if parent == "NORMAL":
        class_name = "NORMAL"
    elif parent == "PNEUMONIA" and "bacteria" in name:
        class_name = "BACTERIAL_PNEUMONIA"
    elif parent == "PNEUMONIA" and "virus" in name:
        class_name = "VIRAL_PNEUMONIA"
    else:
        raise ValueError(f"Cannot infer a three-class label from: {path}")
    return class_name, CLASS_TO_INDEX[class_name]


def infer_patient_id(path: Path) -> str:
    """Extract the dataset's patient identifier from known filename conventions."""
    if path.parent.name.upper() == "PNEUMONIA":
        match = _PNEUMONIA_PATIENT_PATTERN.search(path.name)
    else:
        match = _NORMAL_PATIENT_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Cannot infer patient ID from filename: {path.name}")
    return match.group(1).lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def build_inventory(data_root: str | Path, *, verify_images: bool = True) -> pd.DataFrame:
    """Create an image-level inventory with labels, groups, dimensions, and hashes."""
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Data root does not exist: {root}. Sync the S3 dataset or pass --data-root."
        )

    paths: list[Path] = []
    for split in SOURCE_SPLITS:
        split_root = root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Required source split does not exist: {split_root}")
        paths.extend(
            path
            for path in split_root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
    if not paths:
        raise ValueError(f"No supported images found under {root}")

    records: list[dict[str, object]] = []
    errors: list[str] = []
    for path in sorted(paths):
        try:
            class_name, label = infer_label(path)
            patient_id = infer_patient_id(path)
            if verify_images:
                with Image.open(path) as verification_image:
                    verification_image.verify()
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
                dhash = _difference_hash(image)
            records.append(
                {
                    "key": path.relative_to(root).as_posix(),
                    "source_split": path.relative_to(root).parts[0],
                    "class_name": class_name,
                    "label": label,
                    "patient_id": patient_id,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "dhash": dhash,
                }
            )
        except (OSError, UnidentifiedImageError, ValueError) as error:
            errors.append(f"{path}: {error}")

    if errors:
        preview = "\n".join(errors[:10])
        raise ValueError(f"Invalid images or labels found ({len(errors)}):\n{preview}")

    inventory = pd.DataFrame.from_records(records).sort_values("key").reset_index(drop=True)
    unknown_labels = set(inventory["class_name"]) - set(CLASS_NAMES)
    if unknown_labels:
        raise ValueError(f"Unexpected labels: {sorted(unknown_labels)}")

    # The public filenames reuse numeric IDs across bacterial/viral classes, and
    # NORMAL2 identifiers overlap the IM-* namespace. Treat these as distinct
    # namespaces rather than incorrectly merging unrelated children.
    normal2 = inventory["key"].str.contains(r"/NORMAL/NORMAL2-", case=False, regex=True)
    namespace = inventory["class_name"].str.lower()
    namespace = namespace.mask(normal2, "normal2")
    inventory["group_id"] = namespace + ":" + inventory["patient_id"]
    return inventory


def duplicate_report(inventory: pd.DataFrame) -> pd.DataFrame:
    """Report exact duplicates and perceptual-hash candidates without deleting data."""
    pieces: list[pd.DataFrame] = []
    for column, kind in (("sha256", "exact_sha256"), ("dhash", "dhash_candidate")):
        counts = inventory[column].value_counts()
        duplicate_values = counts[counts > 1].index
        if len(duplicate_values) == 0:
            continue
        frame = inventory[inventory[column].isin(duplicate_values)][
            ["key", "source_split", "class_name", "patient_id", "group_id", column]
        ].copy()
        frame.insert(0, "duplicate_kind", kind)
        frame = frame.rename(columns={column: "hash"})
        pieces.append(frame)
    if not pieces:
        return pd.DataFrame(
            columns=[
                "duplicate_kind",
                "key",
                "source_split",
                "class_name",
                "patient_id",
                "group_id",
                "hash",
            ]
        )
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["duplicate_kind", "hash", "key"]
    ).reset_index(drop=True)


def _split_score(
    labels: np.ndarray,
    validation_indices: np.ndarray,
    target_fraction: float,
) -> float:
    overall = np.bincount(labels, minlength=len(CLASS_NAMES)) / len(labels)
    validation = np.bincount(
        labels[validation_indices], minlength=len(CLASS_NAMES)
    ) / len(validation_indices)
    size_error = abs(len(validation_indices) / len(labels) - target_fraction)
    distribution_error = float(np.abs(overall - validation).sum())
    return 5.0 * size_error + distribution_error


def stratified_group_split(
    development: pd.DataFrame,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the best fold from a shuffled StratifiedGroupKFold partition."""
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    n_splits = max(2, round(1.0 / validation_fraction))
    labels = development["label"].to_numpy(dtype=np.int64)
    groups = development["group_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    candidates = list(splitter.split(np.zeros(len(development)), labels, groups))
    train_indices, validation_indices = min(
        candidates,
        key=lambda pair: _split_score(labels, pair[1], validation_fraction),
    )
    train = development.iloc[train_indices].copy().sort_values("key").reset_index(drop=True)
    validation = (
        development.iloc[validation_indices].copy().sort_values("key").reset_index(drop=True)
    )
    overlap = set(train["group_id"]) & set(validation["group_id"])
    if overlap:
        raise RuntimeError(f"Group leakage between train and validation: {sorted(overlap)[:10]}")
    return train, validation


def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["class_name"].value_counts()
    return {name: int(counts.get(name, 0)) for name in CLASS_NAMES}


def prepare_manifests(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    validation_fraction: float,
    seed: int,
    verify_images: bool = True,
) -> dict[str, object]:
    """Inventory data, create leakage-aware manifests, and persist Day 1 artifacts."""
    root = Path(data_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory(root, verify_images=verify_images)
    duplicates = duplicate_report(inventory)
    exact_label_counts = inventory.groupby("sha256")["class_name"].nunique()
    conflicting_hashes = exact_label_counts[exact_label_counts > 1]
    if not conflicting_hashes.empty:
        raise ValueError(
            f"Exact duplicate pixels have conflicting labels: {len(conflicting_hashes)} hashes"
        )

    source_development = inventory[inventory["source_split"] == "train"].copy()
    # Preserve the full inventory, but avoid weighting identical pixels more than once
    # during optimization and model selection.
    development = source_development.drop_duplicates("sha256", keep="first").copy()
    original_validation = inventory[inventory["source_split"] == "val"].copy()
    test = inventory[inventory["source_split"] == "test"].copy()
    test_deduplicated = test.drop_duplicates("sha256", keep="first").copy()
    train, validation = stratified_group_split(
        development,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    for subset_name, frame in (("train", train), ("validation", validation), ("test", test)):
        frame.insert(0, "subset", subset_name)
        frame.to_csv(output / f"{subset_name}.csv", index=False)
    test_deduplicated.insert(0, "subset", "test_deduplicated")
    test_deduplicated.to_csv(output / "test_deduplicated.csv", index=False)
    inventory.to_csv(output / "inventory.csv", index=False)
    duplicates.to_csv(output / "duplicate_report.csv", index=False)
    original_validation.to_csv(output / "excluded_original_validation.csv", index=False)

    group_overlap = {
        "train_validation": sorted(set(train.group_id) & set(validation.group_id)),
        "development_test": sorted(set(development.group_id) & set(test.group_id)),
    }
    raw_id_overlap = sorted(set(development.patient_id) & set(test.patient_id))
    exact_cross_split = duplicates[
        (duplicates["duplicate_kind"] == "exact_sha256")
        & duplicates.duplicated("hash", keep=False)
    ]
    exact_hash_split_counts = exact_cross_split.groupby("hash")["source_split"].nunique()
    exact_cross_split_hashes = int((exact_hash_split_counts > 1).sum())

    eda_summary = {
        "data_root": str(root),
        "total_images": int(len(inventory)),
        "total_bytes": int(inventory["bytes"].sum()),
        "source_split_counts": {
            str(key): int(value) for key, value in inventory["source_split"].value_counts().items()
        },
        "class_counts": _class_counts(inventory),
        "class_counts_by_source_split": {
            split: _class_counts(inventory[inventory["source_split"] == split])
            for split in SOURCE_SPLITS
        },
        "raw_patient_id_count": int(inventory["patient_id"].nunique()),
        "namespaced_group_count": int(inventory["group_id"].nunique()),
        "image_width": inventory["width"].describe().round(2).to_dict(),
        "image_height": inventory["height"].describe().round(2).to_dict(),
        "image_modes": {
            str(key): int(value) for key, value in inventory["mode"].value_counts().items()
        },
        "exact_duplicate_rows": int(
            (duplicates["duplicate_kind"] == "exact_sha256").sum()
        ),
        "dhash_candidate_rows": int(
            (duplicates["duplicate_kind"] == "dhash_candidate").sum()
        ),
        "exact_cross_source_split_hashes": exact_cross_split_hashes,
    }
    split_summary = {
        "seed": seed,
        "requested_validation_fraction": validation_fraction,
        "actual_validation_fraction": len(validation) / len(development),
        "source_development_images": int(len(source_development)),
        "modeling_development_images": int(len(development)),
        "development_exact_duplicates_removed": int(len(source_development) - len(development)),
        "train_images": int(len(train)),
        "validation_images": int(len(validation)),
        "test_images_locked": int(len(test)),
        "test_images_deduplicated": int(len(test_deduplicated)),
        "test_exact_duplicates_retained_for_official_metric": int(len(test) - len(test_deduplicated)),
        "excluded_original_validation_images": int(len(original_validation)),
        "train_class_counts": _class_counts(train),
        "validation_class_counts": _class_counts(validation),
        "test_class_counts": _class_counts(test),
        "test_deduplicated_class_counts": _class_counts(test_deduplicated),
        "train_groups": int(train["group_id"].nunique()),
        "validation_groups": int(validation["group_id"].nunique()),
        "test_groups": int(test["group_id"].nunique()),
        "raw_development_test_id_overlap_count": len(raw_id_overlap),
        "raw_id_overlap_note": (
            "Raw numeric IDs are reused across label and NORMAL2 namespaces; "
            "group_id is the leakage-control key."
        ),
        "group_overlap_counts": {key: len(value) for key, value in group_overlap.items()},
        "group_overlap_preview": {key: value[:20] for key, value in group_overlap.items()},
    }
    write_json(output / "eda_summary.json", eda_summary)
    write_json(output / "split_summary.json", split_summary)

    if group_overlap["train_validation"]:
        raise RuntimeError("Train/validation group leakage detected after writing diagnostics")
    return {"eda": eda_summary, "split": split_summary}


def build_transforms(image_size: int) -> tuple[Callable, Callable]:
    """Return weak, anatomy-preserving train transforms and deterministic evaluation transforms."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomAffine(
                degrees=5.0,
                translate=(0.02, 0.02),
                scale=(0.98, 1.02),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


class CXRDataset(Dataset):
    """Manifest-backed dataset that can produce one or more stochastic views."""

    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: str | Path,
        transform: Callable,
        *,
        num_views: int = 1,
    ) -> None:
        if num_views < 1:
            raise ValueError("num_views must be at least 1")
        self.frame = frame.reset_index(drop=True).copy()
        self.data_root = Path(data_root).expanduser().resolve()
        self.transform = transform
        self.num_views = num_views

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.frame.iloc[index]
        with Image.open(self.data_root / row["key"]) as image:
            rgb = image.convert("RGB")
            views = torch.stack([self.transform(rgb.copy()) for _ in range(self.num_views)])
        return views, torch.tensor(int(row["label"]), dtype=torch.long), str(row["key"])


def make_loader(
    frame: pd.DataFrame,
    data_root: str | Path,
    transform: Callable,
    *,
    batch_size: int,
    num_views: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    """Construct a deterministic DataLoader for a manifest subset."""
    generator = torch.Generator().manual_seed(seed)
    dataset = CXRDataset(frame, data_root, transform, num_views=num_views)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )
