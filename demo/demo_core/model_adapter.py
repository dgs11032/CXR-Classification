"""E6-only model adapter for the isolated research demo.

The UI depends only on Prediction and E6ModelAdapter.  Future model changes should
be isolated here rather than requiring a rewrite of the Gradio presentation layer.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.dataloader import load_data_config
from data.transforms import build_transform
from p1_cxr.checkpoints import load_checkpoint
from p1_cxr.e6_models import ConvNeXtTinyE6, build_e6_model
from p1_cxr.e6_training import load_e6_training_config

from .priority import PriorityAssessment, assess_priority

CANONICAL_CLASSES: Final[tuple[str, str, str]] = (
    "NORMAL",
    "BACTERIAL_PNEUMONIA",
    "VIRAL_PNEUMONIA",
)
DEFAULT_CHECKPOINT_RELATIVE_PATH: Final[Path] = Path("artifacts/experiments/e6/best.pt")
CHECKPOINT_ENVIRONMENT_VARIABLE: Final[str] = "P1_CXR_DEMO_CHECKPOINT"


@dataclass(frozen=True)
class ModelStatus:
    connected: bool
    message: str
    checkpoint_path: Path


@dataclass(frozen=True)
class Prediction:
    probabilities: dict[str, float]
    predicted_class: str
    confidence: float
    priority: PriorityAssessment


class E6ModelAdapter:
    """Loads only a provenance-compatible full-training E6 checkpoint.

    E0 smoke and dry-run artifacts are rejected explicitly.  When no accepted
    checkpoint is present, the adapter remains available as a status object but
    never produces fabricated probabilities or Grad-CAM outputs.
    """

    def __init__(
        self,
        *,
        project_root: Path = PROJECT_ROOT,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.device = self._resolve_device(device)
        self.model: ConvNeXtTinyE6 | None = None
        self.transform = None
        self.status = ModelStatus(
            connected=False,
            message="Full-training model checkpoint has not been connected yet.",
            checkpoint_path=self.checkpoint_path,
        )
        self._connect_if_valid()

    @classmethod
    def from_environment(cls, *, project_root: Path = PROJECT_ROOT) -> E6ModelAdapter:
        configured = os.environ.get(CHECKPOINT_ENVIRONMENT_VARIABLE)
        return cls(project_root=project_root, checkpoint_path=configured)

    def _resolve_checkpoint_path(self, configured: str | Path | None) -> Path:
        raw = Path(configured).expanduser() if configured else self.project_root / DEFAULT_CHECKPOINT_RELATIVE_PATH
        return raw.resolve() if raw.is_absolute() else (self.project_root / raw).resolve()

    @staticmethod
    def _resolve_device(requested: str | torch.device | None) -> torch.device:
        if requested is not None:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _is_rejected_artifact(path: Path) -> bool:
        normalized = {part.lower() for part in path.parts}
        return "smoke" in normalized or "dryrun" in normalized or "e0_dryrun" in normalized

    def _unavailable(self, message: str) -> None:
        self.status = ModelStatus(False, message, self.checkpoint_path)
        self.model = None
        self.transform = None

    def _connect_if_valid(self) -> None:
        if self._is_rejected_artifact(self.checkpoint_path):
            self._unavailable("Rejected checkpoint: smoke and dry-run artifacts cannot be used for demo prediction.")
            return
        if not self.checkpoint_path.is_file():
            self._unavailable("Full-training model checkpoint has not been connected yet.")
            return
        try:
            payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            config = payload.get("config", {})
            model_config = config.get("model", {})
            if payload.get("checkpoint_version") != 2:
                raise ValueError("checkpoint_version must be 2")
            if payload.get("run_kind") != "e6_full_epoch":
                raise ValueError("run_kind must be e6_full_epoch")
            if config.get("experiment") != "e6":
                raise ValueError("checkpoint config experiment must be e6")
            if model_config.get("num_classes") != 3:
                raise ValueError("checkpoint must use the E6 three-class contract")

            training_config = load_e6_training_config(
                self.project_root / "configs" / "e6_training.yaml",
                project_root=self.project_root,
            )
            model = build_e6_model(training_config).to(self.device)
            load_checkpoint(self.checkpoint_path, model=model, map_location="cpu")
            data_config = load_data_config(
                self.project_root / "configs" / "data_config.yaml",
                project_root=self.project_root,
            )
            self.model = model.eval()
            self.transform = build_transform(data_config["transform"], training=False)
            self.status = ModelStatus(True, "Model connected: validated E6 full-training checkpoint.", self.checkpoint_path)
        except Exception as error:  # noqa: BLE001 - adapter must leave the UI usable for any incompatible artifact.
            self._unavailable(f"Model checkpoint is unavailable or incompatible: {error}")

    @property
    def is_connected(self) -> bool:
        return self.status.connected and self.model is not None and self.transform is not None

    def require_model(self) -> ConvNeXtTinyE6:
        if not self.is_connected or self.model is None:
            raise RuntimeError(self.status.message)
        return self.model

    def prepare_image(self, image: Image.Image) -> torch.Tensor:
        if not self.is_connected or self.transform is None:
            raise RuntimeError(self.status.message)
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image) -> Prediction:
        model = self.require_model()
        inputs = self.prepare_image(image)
        with torch.inference_mode():
            probabilities = torch.softmax(model(inputs), dim=1)[0].detach().cpu().tolist()
        values = {name: float(value) for name, value in zip(CANONICAL_CLASSES, probabilities, strict=True)}
        predicted_class = max(values, key=values.get)
        return Prediction(
            probabilities=values,
            predicted_class=predicted_class,
            confidence=values[predicted_class],
            priority=assess_priority(values["NORMAL"]),
        )
