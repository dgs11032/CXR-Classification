"""고정 CSV manifest를 읽는 single-view/two-view CXR Dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset

ViewMode = Literal["single", "two"]
CLASS_TO_LABEL = {
    "NORMAL": 0,
    "BACTERIAL_PNEUMONIA": 1,
    "VIRAL_PNEUMONIA": 2,
}
_REQUIRED_COLUMNS = {
    "image_path",
    "label",
    "class_name",
    "source_split",
    "group_key",
}


class CXRManifestDataset(Dataset[tuple[Tensor, Tensor]]):
    """Manifest의 이미지 Tensor와 정수 라벨만 반환한다.

    파일명, 경로, class_name, group_key는 검증과 추적에만 사용되며 __getitem__의
    반환값에 포함되지 않으므로 모델 입력 Feature로 전달되지 않는다.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        project_root: str | Path,
        transform: Callable[[Image.Image], Tensor],
        *,
        view_mode: ViewMode = "single",
        validate_paths: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.project_root = Path(project_root).expanduser().resolve()
        self.transform = transform
        if view_mode not in ("single", "two"):
            raise ValueError(f"view_mode는 single 또는 two여야 합니다: {view_mode}")
        self.view_mode: ViewMode = view_mode
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest 파일이 없습니다: {self.manifest_path}")

        frame = pd.read_csv(self.manifest_path)
        missing = _REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Manifest 필수 컬럼 누락: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"Manifest가 비어 있습니다: {self.manifest_path}")
        if frame["image_path"].duplicated().any():
            examples = frame.loc[frame["image_path"].duplicated(False), "image_path"].head(5)
            raise ValueError(f"Manifest에 중복 image_path가 있습니다: {examples.tolist()}")

        numeric_labels = pd.to_numeric(frame["label"], errors="coerce")
        if numeric_labels.isna().any() or not (numeric_labels % 1 == 0).all():
            raise ValueError("Manifest label은 0, 1, 2 중 하나인 정수여야 합니다.")
        frame["label"] = numeric_labels.astype(int)
        invalid_labels = sorted(set(frame["label"]) - set(CLASS_TO_LABEL.values()))
        if invalid_labels:
            raise ValueError(f"지원하지 않는 라벨: {invalid_labels}")
        mismatched = frame[
            frame.apply(
                lambda row: CLASS_TO_LABEL.get(str(row["class_name"])) != int(row["label"]),
                axis=1,
            )
        ]
        if not mismatched.empty:
            raise ValueError(
                "label과 class_name 매핑이 일치하지 않습니다: "
                f"{mismatched[['image_path', 'label', 'class_name']].head(5).to_dict('records')}"
            )

        self.frame = frame.reset_index(drop=True)
        self._resolved_paths = [self._resolve_image_path(value) for value in self.frame["image_path"]]
        if validate_paths:
            missing_paths = [str(path) for path in self._resolved_paths if not path.is_file()]
            if missing_paths:
                raise FileNotFoundError(
                    f"Manifest 이미지 {len(missing_paths)}개가 없습니다. 예: {missing_paths[:5]}"
                )

    def _resolve_image_path(self, image_path: str) -> Path:
        relative = Path(str(image_path))
        if relative.is_absolute():
            raise ValueError(f"image_path는 프로젝트 루트 상대경로여야 합니다: {image_path}")
        resolved = (self.project_root / relative).resolve()
        if not resolved.is_relative_to(self.project_root):
            raise ValueError(f"프로젝트 루트 밖을 가리키는 image_path입니다: {image_path}")
        return resolved

    def __len__(self) -> int:
        return len(self.frame)

    def resolve_path(self, index: int) -> Path:
        """검증·시각화에서만 사용하는 원본 경로 조회 함수."""
        return self._resolved_paths[index]

    def record(self, index: int) -> dict[str, object]:
        """검증용 manifest 행을 반환하며 모델 입력에는 사용하지 않는다."""
        return self.frame.iloc[index].to_dict()

    def class_counts(self) -> dict[str, int]:
        counts = self.frame["class_name"].value_counts()
        return {name: int(counts.get(name, 0)) for name in CLASS_TO_LABEL}

    def _load_image(self, index: int) -> Image.Image:
        path = self._resolved_paths[index]
        try:
            with Image.open(path) as image:
                image.load()
                return image.copy()
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise RuntimeError(
                f"CXR 이미지를 열거나 디코딩할 수 없습니다: index={index}, path={path}, "
                f"error={type(error).__name__}: {error}"
            ) from error

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        source = self._load_image(index)
        label = torch.tensor(int(self.frame.iloc[index]["label"]), dtype=torch.long)
        try:
            if self.view_mode == "single":
                image_tensor = self.transform(source.copy())
            else:
                # 같은 source를 사용하되 transform RNG는 호출마다 독립적으로 표본화된다.
                image_tensor = torch.stack(
                    [self.transform(source.copy()), self.transform(source.copy())], dim=0
                )
        except Exception as error:
            path = self._resolved_paths[index]
            raise RuntimeError(
                f"CXR Transform 적용 실패: index={index}, path={path}, "
                f"error={type(error).__name__}: {error}"
            ) from error
        if not isinstance(image_tensor, Tensor):
            raise TypeError("Transform 출력은 torch.Tensor여야 합니다.")
        return image_tensor, label
