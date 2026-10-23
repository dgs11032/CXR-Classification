"""E8-only fixed-manifest Bacterial/Viral binary specialist data boundary."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from data.dataloader import build_dataset, build_loader, seed_everything
from data.dataset import CXRManifestDataset

BINARY_CLASS_ORDER = ("BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA")
CANONICAL_TO_BINARY = {1: 0, 2: 1}
EXPECTED_COUNTS = {
    "train": {"BACTERIAL_PNEUMONIA": 2153, "VIRAL_PNEUMONIA": 1147},
    "validation": {"BACTERIAL_PNEUMONIA": 359, "VIRAL_PNEUMONIA": 191},
}


class E8BinarySpecialistDataset(Dataset[tuple[Tensor, Tensor]]):
    """Filter an already-validated fixed three-class manifest without rewriting it."""

    def __init__(self, base_dataset: CXRManifestDataset, *, partition: str) -> None:
        if partition not in EXPECTED_COUNTS:
            raise ValueError(f"E8 partition must be train or validation: {partition}")
        self.base_dataset = base_dataset
        self.partition = partition
        source_frame = base_dataset.frame
        selected = source_frame[source_frame["class_name"].isin(BINARY_CLASS_ORDER)].copy()
        if len(selected) == 0:
            raise ValueError(f"E8 {partition} binary specialist selection is empty.")
        if set(selected["class_name"].unique()) != set(BINARY_CLASS_ORDER):
            raise ValueError(f"E8 {partition} must include both Bacterial and Viral classes.")
        canonical_labels = set(int(value) for value in selected["label"].unique())
        if canonical_labels != set(CANONICAL_TO_BINARY):
            raise ValueError(f"E8 {partition} canonical labels must be {{1, 2}}: {canonical_labels}")
        self._base_indices = [int(index) for index in selected.index.tolist()]
        self.frame = selected.reset_index(drop=True)
        self.frame["canonical_label"] = self.frame["label"].astype(int)
        self.frame["label"] = self.frame["canonical_label"].map(CANONICAL_TO_BINARY).astype(int)
        counts = self.class_counts()
        if counts != EXPECTED_COUNTS[partition]:
            raise ValueError(
                f"E8 {partition} fixed-manifest counts drift: expected={EXPECTED_COUNTS[partition]}, actual={counts}"
            )
        if set(int(value) for value in self.frame["label"].unique()) != {0, 1}:
            raise ValueError(f"E8 {partition} binary labels must be {{0, 1}}.")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        images, canonical_label = self.base_dataset[self._base_indices[index]]
        expected_canonical = int(self.frame.iloc[index]["canonical_label"])
        if int(canonical_label) != expected_canonical:
            raise RuntimeError("E8 base dataset label no longer matches its fixed manifest record.")
        return images, torch.tensor(int(self.frame.iloc[index]["label"]), dtype=torch.long)

    def class_counts(self) -> dict[str, int]:
        counts = self.frame["class_name"].value_counts()
        return {class_name: int(counts.get(class_name, 0)) for class_name in BINARY_CLASS_ORDER}


def build_e8_dataset(data_config: Mapping[str, Any], partition: str) -> E8BinarySpecialistDataset:
    base = build_dataset(
        data_config,
        partition,
        view_mode="two" if partition == "train" else "single",
    )
    return E8BinarySpecialistDataset(base, partition=partition)


def build_e8_dataloaders(data_config: Mapping[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Reuse shared fixed manifests/transforms/loader settings after E8-local filtering."""
    if data_config["views"]["train"] != "two" or data_config["views"]["validation"] != "single":
        raise ValueError("E8 requires shared two-view train and deterministic single-view validation.")
    seed_everything(int(data_config["seed"]))
    train_dataset = build_e8_dataset(data_config, "train")
    validation_dataset = build_e8_dataset(data_config, "validation")
    return (
        build_loader(train_dataset, data_config, training=True),
        build_loader(validation_dataset, data_config, training=False),
    )
