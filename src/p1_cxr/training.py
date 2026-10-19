"""STEP 6 데이터 파이프라인과 호환되는 E0 FP32 학습·평가 engine."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data.dataloader import build_dataloaders, load_data_config

from .checkpoints import config_fingerprint, load_checkpoint, save_checkpoint
from .metrics import classification_metrics, prediction_frame
from .models import ConvNeXtTinyE0, build_e0_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class NumericalStabilityError(FloatingPointError):
    def __init__(self, stage: str, *, batch_index: int, detail: str) -> None:
        self.stage = stage
        self.batch_index = batch_index
        self.detail = detail
        super().__init__(f"수치 오류 stage={stage}, batch={batch_index}: {detail}")


@dataclass
class TrainEpochResult:
    loss: float
    processed_source_images: int
    processed_views: int
    batches: int
    gradients_finite: bool
    maximum_gradient_norm_before_clip: float


@dataclass
class EvaluationResult:
    loss: float
    metrics: dict[str, Any]
    targets: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    logits: np.ndarray
    processed_images: int
    batches: int


def _nested_value(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        value = value[key]
    return value


def validate_final_training_contract(config: Mapping[str, Any]) -> None:
    """사용자가 확정한 E0 최종 값을 fail-closed 방식으로 검증한다."""
    expected = {
        ("version",): 2,
        ("experiment",): "e0",
        ("status",): "final_pretraining_configuration",
        ("model", "architecture"): "flat",
        ("model", "backbone"): "convnext_tiny",
        ("model", "weights"): "IMAGENET1K_V1",
        ("model", "pretrained"): True,
        ("model", "allow_weight_download"): False,
        ("model", "feature_dimension"): 768,
        ("model", "num_classes"): 3,
        ("model", "trainable_scope"): "full",
        ("loss", "name"): "CrossEntropyLoss",
        ("loss", "label_smoothing"): 0.0,
        ("loss", "class_weighting"): "none_unweighted_e0_baseline",
        ("optimizer", "name"): "AdamW",
        ("optimizer", "strategy"): "differential_lr",
        ("optimizer", "backbone_learning_rate"): 1e-5,
        ("optimizer", "head_learning_rate"): 1e-4,
        ("optimizer", "weight_decay"): 1e-4,
        ("scheduler", "name"): "CosineAnnealingLR",
        ("scheduler", "t_max"): 15,
        ("scheduler", "eta_min"): 1e-6,
        ("training", "seed"): 42,
        ("training", "epochs"): 15,
        ("training", "early_stopping_patience"): 5,
        ("training", "selection_metric"): "macro_f1",
        ("training", "device"): "cuda",
        ("training", "precision"): "fp32",
        ("training", "amp"): False,
        ("training", "gradient_clip_norm"): 1.0,
        ("training", "detect_nonfinite"): True,
        ("training", "checkpoint", "best_filename"): "best.pt",
        ("training", "checkpoint", "last_filename"): "last.pt",
        ("training", "checkpoint", "save_rng_state"): True,
    }
    mismatches = []
    for path, expected_value in expected.items():
        try:
            actual = _nested_value(config, path)
        except (KeyError, TypeError):
            actual = "__MISSING__"
        if actual != expected_value:
            mismatches.append(
                {"path": ".".join(path), "expected": expected_value, "actual": actual}
            )
    if mismatches:
        raise ValueError(f"최종 E0 설정 drift 감지: {mismatches}")


def validate_final_data_contract(data_config: Mapping[str, Any]) -> None:
    expected = {
        ("seed",): 42,
        ("views", "train"): "two",
        ("views", "validation"): "single",
        ("loader", "batch_size"): 16,
        ("loader", "num_workers"): 0,
        ("loader", "train_shuffle"): True,
        ("loader", "validation_shuffle"): False,
    }
    mismatches = []
    for path, expected_value in expected.items():
        try:
            actual = _nested_value(data_config, path)
        except (KeyError, TypeError):
            actual = "__MISSING__"
        if actual != expected_value:
            mismatches.append(
                {"path": ".".join(path), "expected": expected_value, "actual": actual}
            )
    if mismatches:
        raise ValueError(f"최종 E0 데이터 설정 drift 감지: {mismatches}")


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_resume_checkpoint(
    payload: Mapping[str, Any],
    *,
    training_config: Mapping[str, Any],
    seed: int,
    manifest_fingerprints: Mapping[str, str],
) -> None:
    if payload.get("run_kind") != "full_epoch":
        raise ValueError(
            f"full E0 resume에는 run_kind=full_epoch checkpoint만 허용합니다: "
            f"{payload.get('run_kind')}"
        )
    if payload.get("config_fingerprint") != config_fingerprint(training_config):
        raise ValueError("현재 최종 설정과 checkpoint config fingerprint가 다릅니다.")
    if payload.get("random_seed") != seed:
        raise ValueError("현재 seed와 checkpoint seed가 다릅니다.")
    saved_manifests = payload.get("control_state", {}).get("manifest_fingerprints")
    if saved_manifests != dict(manifest_fingerprints):
        raise ValueError("Train/Validation manifest fingerprint가 checkpoint와 다릅니다.")


def load_training_config(
    config_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    path = Path(config_path).expanduser()
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("학습 설정은 프로젝트 루트 내부에 있어야 합니다.")
    if not path.is_file():
        raise FileNotFoundError(f"학습 설정이 없습니다: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("학습 설정은 YAML mapping이어야 합니다.")
    validate_final_training_contract(config)
    if str(config.get("experiment", "")).lower() != "e0":
        raise ValueError("현재 학습 engine은 E0만 허용합니다.")
    if config.get("loss", {}).get("name") != "CrossEntropyLoss":
        raise ValueError("E0 Loss는 CrossEntropyLoss여야 합니다.")
    if config.get("training", {}).get("precision") != "fp32":
        raise ValueError("최종 E0 설정은 precision=fp32여야 합니다.")
    if bool(config.get("training", {}).get("amp")):
        raise ValueError("최종 E0 설정은 AMP를 비활성화해야 합니다.")
    output_dir = (root / config["output_dir"]).resolve()
    data_config_path = (root / config["data_config"]).resolve()
    if not output_dir.is_relative_to(root) or not data_config_path.is_relative_to(root):
        raise ValueError("설정 경로는 프로젝트 루트 내부여야 합니다.")
    config["resolved"] = {
        "project_root": str(root),
        "config_path": str(path),
        "data_config_path": str(data_config_path),
        "output_dir": str(output_dir),
        "dry_run_output_dir": str((root / config["dry_run"]["output_dir"]).resolve()),
    }
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA를 요청했지만 사용할 수 없습니다.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS를 요청했지만 사용할 수 없습니다.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def expand_labels_for_views(labels: Tensor, num_views: int) -> Tensor:
    """[y0,y1]을 sample-contiguous [y0,y0,y1,y1] 순서로 확장한다."""
    if labels.ndim != 1:
        raise ValueError(f"labels는 [B]여야 합니다: {tuple(labels.shape)}")
    if num_views < 1:
        raise ValueError("num_views는 1 이상이어야 합니다.")
    return labels[:, None].expand(-1, num_views).reshape(-1)


def flatten_two_view_batch(images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    if images.ndim != 5:
        raise ValueError(f"Two-view images는 [B,V,C,H,W]여야 합니다: {tuple(images.shape)}")
    batch_size, num_views, channels, height, width = images.shape
    if len(labels) != batch_size:
        raise ValueError("이미지 batch와 label batch 크기가 다릅니다.")
    return (
        images.reshape(batch_size * num_views, channels, height, width),
        expand_labels_for_views(labels, num_views),
    )


def create_criterion(config: Mapping[str, Any]) -> nn.CrossEntropyLoss:
    if config["loss"].get("class_weighting") != "none_unweighted_e0_baseline":
        raise ValueError("E0는 unweighted CrossEntropyLoss만 사용합니다.")
    return nn.CrossEntropyLoss(
        label_smoothing=float(config["loss"].get("label_smoothing", 0.0)),
        reduction="mean",
    )


def _trainable_parameters(modules: tuple[nn.Module, ...]) -> list[nn.Parameter]:
    return [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def create_optimizer(
    model: ConvNeXtTinyE0,
    config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    """Backbone 1e-5와 Head 1e-4의 중복 없는 AdamW parameter group을 만든다."""
    optimizer_config = config["optimizer"]
    if optimizer_config["name"] != "AdamW":
        raise ValueError("E0 Optimizer는 AdamW여야 합니다.")
    if optimizer_config.get("strategy") != "differential_lr":
        raise ValueError("최종 E0 Optimizer는 differential_lr 전략이어야 합니다.")

    backbone = _trainable_parameters((model.features, model.avgpool, model.feature_norm))
    head = _trainable_parameters((model.classifier,))
    if not backbone or not head:
        raise ValueError("Full fine-tuning에는 non-empty backbone/head group이 모두 필요합니다.")
    backbone_ids = {id(parameter) for parameter in backbone}
    head_ids = {id(parameter) for parameter in head}
    duplicated = backbone_ids & head_ids
    all_trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    grouped_ids = backbone_ids | head_ids
    if duplicated:
        raise RuntimeError(f"Optimizer group parameter 중복: {len(duplicated)}개")
    if grouped_ids != all_trainable_ids:
        raise RuntimeError(
            "Optimizer group 누락/초과: "
            f"missing={len(all_trainable_ids - grouped_ids)}, extra={len(grouped_ids - all_trainable_ids)}"
        )

    return torch.optim.AdamW(
        [
            {
                "params": backbone,
                "lr": float(optimizer_config["backbone_learning_rate"]),
                "name": "backbone",
            },
            {
                "params": head,
                "lr": float(optimizer_config["head_learning_rate"]),
                "name": "head",
            },
        ],
        weight_decay=float(optimizer_config["weight_decay"]),
    )


def optimizer_group_summary(optimizer: torch.optim.Optimizer) -> dict[str, dict[str, Any]]:
    return {
        str(group.get("name", index)): {
            "learning_rate": float(group["lr"]),
            "parameter_tensors": len(group["params"]),
            "parameters": int(sum(parameter.numel() for parameter in group["params"])),
        }
        for index, group in enumerate(optimizer.param_groups)
    }


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    scheduler_config = config["scheduler"]
    if scheduler_config["name"] != "CosineAnnealingLR":
        raise ValueError("E0 Scheduler는 CosineAnnealingLR여야 합니다.")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(scheduler_config["t_max"]),
        eta_min=float(scheduler_config["eta_min"]),
    )


def create_grad_scaler(*, amp_enabled: bool, device: torch.device) -> torch.amp.GradScaler | None:
    if not amp_enabled:
        return None
    if device.type != "cuda":
        raise ValueError("AMP GradScaler는 CUDA에서만 사용할 수 있습니다.")
    return torch.amp.GradScaler("cuda", enabled=True)


def _autocast_context(device: torch.device, enabled: bool):
    if enabled:
        if device.type != "cuda":
            raise ValueError("autocast는 현재 CUDA에서만 허용합니다.")
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _require_finite(tensor: Tensor, *, stage: str, batch_index: int) -> None:
    if not bool(torch.isfinite(tensor).all()):
        nonfinite = int((~torch.isfinite(tensor)).sum().detach().cpu())
        raise NumericalStabilityError(
            stage,
            batch_index=batch_index,
            detail=f"nonfinite elements={nonfinite}, shape={tuple(tensor.shape)}",
        )


def _first_nonfinite_gradient(model: nn.Module) -> str | None:
    found_any = False
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        found_any = True
        if not bool(torch.isfinite(parameter.grad).all()):
            return name
    return None if found_any else "__NO_GRADIENT__"


def _first_nonfinite_parameter(model: nn.Module) -> str | None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not bool(torch.isfinite(parameter).all()):
            return name
    return None


def train_one_epoch(
    model: ConvNeXtTinyE0,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    amp_enabled: bool,
    gradient_clip_norm: float | None,
    scaler: torch.amp.GradScaler | None = None,
    max_batches: int | None = None,
) -> TrainEpochResult:
    """Two-view CE를 학습하고 단계별 nonfinite를 즉시 중단한다."""
    if amp_enabled and scaler is None:
        raise ValueError("AMP 활성화 시 run-scoped GradScaler가 필요합니다.")
    if not amp_enabled and scaler is not None:
        raise ValueError("AMP 비활성화 시 GradScaler를 전달하지 않아야 합니다.")
    model.train()
    loss_sum = 0.0
    processed_views = 0
    processed_source_images = 0
    batch_count = 0
    max_gradient_norm = 0.0

    for batch_index, (views, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        flattened, expanded_labels = flatten_two_view_batch(views, labels)
        flattened = flattened.to(device)
        expanded_labels = expanded_labels.to(device)
        _require_finite(flattened, stage="train_input", batch_index=batch_index)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_enabled):
            logits = model(flattened)
            _require_finite(logits, stage="train_logits", batch_index=batch_index)
            loss = criterion(logits, expanded_labels)
            _require_finite(loss, stage="train_loss", batch_index=batch_index)

        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        offending_gradient = _first_nonfinite_gradient(model)
        if offending_gradient == "__NO_GRADIENT__":
            raise NumericalStabilityError(
                "train_gradient",
                batch_index=batch_index,
                detail="학습 가능한 gradient가 생성되지 않음",
            )
        if offending_gradient is not None:
            raise NumericalStabilityError(
                "train_gradient",
                batch_index=batch_index,
                detail=f"first nonfinite parameter={offending_gradient}",
            )

        if gradient_clip_norm is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
                error_if_nonfinite=False,
            )
            _require_finite(total_norm, stage="train_gradient_norm", batch_index=batch_index)
            max_gradient_norm = max(max_gradient_norm, float(total_norm.detach().cpu()))
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        offending_parameter = _first_nonfinite_parameter(model)
        if offending_parameter is not None:
            raise NumericalStabilityError(
                "post_optimizer_parameter",
                batch_index=batch_index,
                detail=f"first nonfinite parameter={offending_parameter}",
            )

        view_count = len(expanded_labels)
        loss_sum += float(loss.detach().cpu()) * view_count
        processed_views += view_count
        processed_source_images += len(labels)
        batch_count += 1

    if batch_count == 0:
        raise ValueError("처리된 Train batch가 없습니다.")
    return TrainEpochResult(
        loss=loss_sum / processed_views,
        processed_source_images=processed_source_images,
        processed_views=processed_views,
        batches=batch_count,
        gradients_finite=True,
        maximum_gradient_norm_before_clip=max_gradient_norm,
    )


@torch.inference_mode()
def evaluate(
    model: ConvNeXtTinyE0,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    *,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None = None,
) -> EvaluationResult:
    """전체 single-view validation에서 단계별 finite와 공통 지표를 계산한다."""
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    batch_count = 0
    logits_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if images.ndim != 4:
            raise ValueError(
                f"Validation images는 [B,C,H,W] single-view여야 합니다: {tuple(images.shape)}"
            )
        images = images.to(device)
        device_labels = labels.to(device)
        _require_finite(images, stage="validation_input", batch_index=batch_index)
        with _autocast_context(device, amp_enabled):
            logits = model(images)
            _require_finite(logits, stage="validation_logits", batch_index=batch_index)
            loss = criterion(logits, device_labels)
            _require_finite(loss, stage="validation_loss", batch_index=batch_index)
        probabilities = torch.softmax(logits.float(), dim=1)
        _require_finite(
            probabilities, stage="validation_probabilities", batch_index=batch_index
        )
        loss_sum += float(loss.cpu()) * len(labels)
        sample_count += len(labels)
        batch_count += 1
        logits_parts.append(logits.float().cpu())
        target_parts.append(labels.cpu())

    if batch_count == 0:
        raise ValueError("처리된 Validation batch가 없습니다.")
    logits_tensor = torch.cat(logits_parts)
    targets_tensor = torch.cat(target_parts)
    probabilities_tensor = torch.softmax(logits_tensor, dim=1)
    targets = targets_tensor.numpy()
    probabilities = probabilities_tensor.numpy()
    predictions = probabilities.argmax(axis=1)
    metrics = classification_metrics(targets, probabilities)
    return EvaluationResult(
        loss=loss_sum / sample_count,
        metrics=metrics,
        targets=targets,
        predictions=predictions,
        probabilities=probabilities,
        logits=logits_tensor.numpy(),
        processed_images=sample_count,
        batches=batch_count,
    )


def configure_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"p1_cxr.e0.{output_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _validation_artifacts(
    output_dir: Path,
    prefix: str,
    *,
    epoch: int,
    train_loss: float,
    result: EvaluationResult,
    image_paths: list[str],
) -> None:
    metrics = {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": result.loss,
        **result.metrics,
    }
    _write_json_atomic(output_dir / f"{prefix}_validation_metrics.json", metrics)
    _write_csv_atomic(
        output_dir / f"{prefix}_validation_predictions.csv",
        prediction_frame(image_paths, result.targets, result.probabilities),
    )
    _write_csv_atomic(
        output_dir / f"{prefix}_validation_confusion_matrix.csv",
        pd.DataFrame(
            result.metrics["confusion_matrix"],
            index=["true_normal", "true_bacterial", "true_viral"],
            columns=["pred_normal", "pred_bacterial", "pred_viral"],
        ).reset_index(names="ground_truth"),
    )


def _epoch_row(
    *,
    epoch: int,
    train_result: TrainEpochResult,
    validation_result: EvaluationResult,
    lrs: Mapping[str, Mapping[str, Any]],
    train_seconds: float,
    validation_seconds: float,
    epoch_seconds: float,
    improved: bool,
    best_metric: float,
    stale_epochs: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": epoch,
        "train_loss": train_result.loss,
        "validation_loss": validation_result.loss,
        "validation_accuracy": validation_result.metrics["accuracy"],
        "validation_macro_f1": validation_result.metrics["macro_f1"],
        "validation_macro_auroc": validation_result.metrics["macro_auroc_ovr"],
        "backbone_lr": lrs["backbone"]["learning_rate"],
        "head_lr": lrs["head"]["learning_rate"],
        "train_seconds": train_seconds,
        "validation_seconds": validation_seconds,
        "epoch_seconds": epoch_seconds,
        "train_batches": train_result.batches,
        "train_source_images": train_result.processed_source_images,
        "train_views": train_result.processed_views,
        "maximum_gradient_norm_before_clip": train_result.maximum_gradient_norm_before_clip,
        "numerical_checks_passed": train_result.gradients_finite,
        "is_best": improved,
        "best_validation_macro_f1": best_metric,
        "stale_epochs": stale_epochs,
    }
    for class_name, class_metrics in validation_result.metrics["per_class"].items():
        prefix = class_name.lower()
        row[f"{prefix}_precision"] = class_metrics["precision"]
        row[f"{prefix}_recall"] = class_metrics["recall"]
        row[f"{prefix}_f1"] = class_metrics["f1"]
    return row


def fit_e0(
    training_config: Mapping[str, Any],
    *,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """최종 FP32 E0 전체 학습 loop. 명시적 CLI 승인 후에만 호출한다."""
    seed = int(training_config["training"]["seed"])
    seed_everything(seed)
    root = Path(training_config["resolved"]["project_root"])
    data_config = load_data_config(
        training_config["resolved"]["data_config_path"], project_root=root
    )
    validate_final_data_contract(data_config)
    if int(data_config["loader"]["batch_size"]) != 16:
        raise ValueError("최종 E0 원본 Batch Size는 16이어야 합니다.")
    train_loader, validation_loader = build_dataloaders(data_config)
    if len(validation_loader.dataset) != 742:
        raise ValueError("Validation은 전체 742장을 사용해야 합니다.")

    model = build_e0_model(training_config)
    device = resolve_device(str(training_config["training"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("최종 E0 전체 학습 장치는 CUDA여야 합니다.")
    model.to(device)
    amp_enabled = bool(training_config["training"]["amp"])
    if amp_enabled:
        raise ValueError("최종 E0는 FP16 AMP를 사용할 수 없습니다.")
    scaler = create_grad_scaler(amp_enabled=amp_enabled, device=device)
    criterion = create_criterion(training_config)
    optimizer = create_optimizer(model, training_config)
    scheduler = create_scheduler(optimizer, training_config)

    output_dir = Path(training_config["resolved"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(output_dir)
    _write_json_atomic(output_dir / "training_config.json", training_config)
    logger.info("optimizer_groups=%s", json.dumps(optimizer_group_summary(optimizer)))

    best_metric = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    train_generator = train_loader.generator
    manifest_fingerprints = {
        "train": file_sha256(data_config["resolved"]["train_manifest"]),
        "validation": file_sha256(data_config["resolved"]["validation_manifest"]),
    }
    if resume_from is not None:
        payload = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_loader_generator=train_generator,
            restore_rng=True,
            map_location=device,
        )
        validate_resume_checkpoint(
            payload,
            training_config=training_config,
            seed=seed,
            manifest_fingerprints=manifest_fingerprints,
        )
        start_epoch = int(payload["next_epoch"])
        best_metric = float(payload["best_validation_macro_f1"])
        control = payload.get("control_state", {})
        best_epoch = int(control.get("best_epoch", 0))
        stale_epochs = int(control.get("stale_epochs", 0))
        history = list(payload.get("history", []))
        logger.info("resumed_from=%s next_epoch=%d", resume_from, start_epoch)

    epochs = int(training_config["training"]["epochs"])
    patience = int(training_config["training"]["early_stopping_patience"])
    checkpoint_config = training_config["training"]["checkpoint"]
    best_path = output_dir / checkpoint_config["best_filename"]
    last_path = output_dir / checkpoint_config["last_filename"]
    validation_paths = validation_loader.dataset.frame["image_path"].tolist()

    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        used_lrs = optimizer_group_summary(optimizer)
        try:
            train_started = time.perf_counter()
            train_result = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device=device,
                amp_enabled=amp_enabled,
                scaler=scaler,
                gradient_clip_norm=float(
                    training_config["training"]["gradient_clip_norm"]
                ),
            )
            train_seconds = time.perf_counter() - train_started
            validation_started = time.perf_counter()
            validation_result = evaluate(
                model,
                validation_loader,
                criterion,
                device=device,
                amp_enabled=amp_enabled,
            )
            validation_seconds = time.perf_counter() - validation_started
        except NumericalStabilityError as error:
            failure = {
                "epoch": epoch,
                "stage": error.stage,
                "batch_index": error.batch_index,
                "detail": error.detail,
                "amp_enabled": amp_enabled,
                "optimizer_groups": optimizer_group_summary(optimizer),
            }
            _write_json_atomic(output_dir / "numerical_failure.json", failure)
            logger.exception("numerical_failure=%s", json.dumps(failure))
            raise

        metric = float(validation_result.metrics["macro_f1"])
        improved = metric > best_metric
        if improved:
            best_metric = metric
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        epoch_seconds = time.perf_counter() - epoch_started
        row = _epoch_row(
            epoch=epoch,
            train_result=train_result,
            validation_result=validation_result,
            lrs=used_lrs,
            train_seconds=train_seconds,
            validation_seconds=validation_seconds,
            epoch_seconds=epoch_seconds,
            improved=improved,
            best_metric=best_metric,
            stale_epochs=stale_epochs,
        )
        history.append(row)
        _write_csv_atomic(output_dir / "history.csv", pd.DataFrame(history))
        logger.info(json.dumps(row, ensure_ascii=False))

        _validation_artifacts(
            output_dir,
            "last",
            epoch=epoch,
            train_loss=train_result.loss,
            result=validation_result,
            image_paths=validation_paths,
        )
        if improved:
            _validation_artifacts(
                output_dir,
                "best",
                epoch=epoch,
                train_loss=train_result.loss,
                result=validation_result,
                image_paths=validation_paths,
            )

        # Checkpoint는 다음 Epoch에 사용할 두 group LR을 포함하도록 step 이후 저장한다.
        scheduler.step()
        control_state = {
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "next_epoch_learning_rates": optimizer_group_summary(optimizer),
            "manifest_fingerprints": manifest_fingerprints,
        }
        checkpoint_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "epoch": epoch,
            "metrics": validation_result.metrics,
            "config": training_config,
            "best_validation_macro_f1": best_metric,
            "random_seed": seed,
            "control_state": control_state,
            "history": history,
            "train_loader_generator": train_generator,
            "save_rng_state": bool(checkpoint_config["save_rng_state"]),
            "run_kind": "full_epoch",
        }
        save_checkpoint(last_path, **checkpoint_kwargs)
        if improved:
            save_checkpoint(best_path, **checkpoint_kwargs)
        if stale_epochs >= patience:
            logger.info("early_stopping epoch=%d best_epoch=%d", epoch, best_epoch)
            break

    return {
        "best_validation_macro_f1": best_metric,
        "best_epoch": best_epoch,
        "last_epoch": history[-1]["epoch"] if history else start_epoch - 1,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history": history,
    }
