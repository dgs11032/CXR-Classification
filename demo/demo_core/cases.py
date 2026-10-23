"""Anonymous in-memory case handling for the presentation demo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .model_adapter import E6ModelAdapter, Prediction

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class DemoCase:
    case_id: str
    image: Image.Image
    prediction: Prediction | None
    inference_note: str | None = None


class CaseStore:
    """Keeps only anonymous, in-memory presentation cases for one UI session."""

    def __init__(self) -> None:
        self._cases: list[DemoCase] = []

    @property
    def cases(self) -> tuple[DemoCase, ...]:
        return tuple(self._cases)

    def get(self, case_id: str | None) -> DemoCase | None:
        return next((case for case in self._cases if case.case_id == case_id), None)

    def add_paths(self, paths: list[str] | None, adapter: E6ModelAdapter) -> tuple[int, list[str]]:
        added = 0
        notes: list[str] = []
        for raw_path in paths or []:
            path = Path(raw_path)
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                notes.append(f"Skipped unsupported file type: {path.name}")
                continue
            try:
                with Image.open(path) as source:
                    image = source.convert("RGB").copy()
            except (OSError, ValueError):
                notes.append("Skipped an unreadable image file.")
                continue

            prediction = None
            inference_note = None
            if adapter.is_connected:
                try:
                    prediction = adapter.predict(image)
                except Exception as error:  # noqa: BLE001 - retain per-case UI availability on inference failure.
                    inference_note = f"Inference unavailable for this case: {error}"
            else:
                inference_note = "Full-training model checkpoint has not been connected yet."

            case_id = f"Case #{len(self._cases) + 1:03d}"
            self._cases.append(DemoCase(case_id, image, prediction, inference_note))
            added += 1
        return added, notes


def discover_asset_paths(asset_directory: Path) -> list[str]:
    """Return only approved local demo assets; no dataset discovery or copying occurs."""
    if not asset_directory.is_dir():
        return []
    return [
        str(path)
        for path in sorted(asset_directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
