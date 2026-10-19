#!/usr/bin/env python3
"""STEP 8-D: Tesla T4 CUDA 소규모 벤치마크.

E0 ConvNeXt-Tiny 모델과 공통 DataLoader를 재사용하여 Batch Size(8/16)와
Precision(FP32 / CUDA FP16 AMP) 조합의 학습 Step 처리량·메모리를 측정한다.

제약:
- 전체 Epoch 학습, Test 접근, AWS 작업, 원본/Manifest 수정은 수행하지 않는다.
- 각 설정은 Warm-up 1회 + 측정 5회의 소규모 Step 벤치마크로만 제한한다.
- 모델 초기화·사전학습 가중치 로딩 시간은 Step 측정에서 제외한다(한 번만 수행).
- CUDA가 없는 환경(예: macOS)에서는 벤치마크를 실행하지 않고 종료한다.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import platform
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataloader import build_dataloaders, load_data_config  # noqa: E402
from p1_cxr.checkpoints import save_checkpoint  # noqa: E402
from p1_cxr.models import build_e0_model, pretrained_weight_cache_path  # noqa: E402
from p1_cxr.training import (  # noqa: E402
    create_criterion,
    create_optimizer,
    create_scheduler,
    flatten_two_view_batch,
    load_training_config,
    resolve_device,
    seed_everything,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "e0_training.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "benchmarks" / "cuda"

# 원본 이미지 기준 Batch Size. Two-view이므로 실제 Forward 입력은 2배이다.
BATCH_SIZES = (8, 16)
PRECISIONS = ("fp32", "fp16_amp")
WARMUP_STEPS = 1
MEASURED_STEPS = 5

TRAIN_IMAGES = 4448
VALIDATION_IMAGES = 742
EPOCHS = 15
# E0~E3 공정 비교를 위해 SupCon 실험 시 확보하고 싶은 메모리 안전 상한.
MEMORY_SAFETY_FRACTION = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--measured-steps", type=int, default=MEASURED_STEPS)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(BATCH_SIZES),
        help="원본 이미지 기준 Batch Size 목록 (기본 8 16)",
    )
    parser.add_argument(
        "--allow-non-cuda",
        action="store_true",
        help="CUDA가 없어도 종료하지 않고 환경 리포트만 기록 (디버깅 전용, 벤치마크는 실행 안 함)",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# CUDA 유틸리티
# --------------------------------------------------------------------------- #
def synchronize() -> None:
    """CUDA 비동기 연산을 완료시켜 정확한 벽시계 시간을 측정한다."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def empty_cuda_cache() -> None:
    """다음 실험이 안전하게 시작되도록 CUDA allocator cache를 비운다."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_bytes(device: torch.device) -> dict[str, int | None]:
    if device.type == "cuda":
        return {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        }
    return {"peak_allocated_bytes": None, "peak_reserved_bytes": None, "total_bytes": None}


def is_oom_error(error: Exception) -> bool:
    message = str(error).lower()
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in message


def autocast_context(device: torch.device, enabled: bool):
    """FP16 AMP일 때만 CUDA autocast를 활성화한다."""
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


# --------------------------------------------------------------------------- #
# 환경 리포트
# --------------------------------------------------------------------------- #
def collect_environment(device: torch.device) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu: dict[str, Any] = {"available": cuda_available}
    if cuda_available:
        properties = torch.cuda.get_device_properties(device)
        gpu.update(
            {
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "total_memory_gib": round(properties.total_memory / 1024**3, 2),
                "multi_processor_count": properties.multi_processor_count,
                "capability": f"{properties.major}.{properties.minor}",
            }
        )
    return {
        "operating_system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        },
        "compute": {
            "selected_device": str(device),
            "cuda_available": cuda_available,
            "gpu": gpu,
        },
        "benchmark_policy": {
            "batch_sizes_source_images": list(BATCH_SIZES),
            "precisions": list(PRECISIONS),
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "two_view_forward_multiplier": 2,
            "timing": "torch.cuda.synchronize() 기반 벽시계 측정",
            "peak_memory": "torch.cuda.max_memory_allocated / max_memory_reserved",
            "model_load_excluded_from_step_timing": True,
            "full_epochs_executed": 0,
            "test_images_accessed": 0,
            "aws_resources_created": 0,
            "pretrained_cache_path": str(pretrained_weight_cache_path()),
        },
    }


# --------------------------------------------------------------------------- #
# 단일 Train Step 측정
# --------------------------------------------------------------------------- #
def measure_train_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scaler: torch.amp.GradScaler,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    amp_enabled: bool,
    gradient_clip_norm: float,
) -> dict[str, Any]:
    """단계별 finite를 검사하고 실패 시 optimizer update 없이 반환한다."""
    views, labels = batch
    images, expanded_labels = flatten_two_view_batch(views, labels)

    synchronize()
    started = time.perf_counter()
    images = images.to(device, non_blocking=True)
    expanded_labels = expanded_labels.to(device, non_blocking=True)
    synchronize()
    transfer_seconds = time.perf_counter() - started
    input_finite = bool(torch.isfinite(images).all())

    optimizer.zero_grad(set_to_none=True)
    synchronize()
    started = time.perf_counter()
    with autocast_context(device, amp_enabled):
        logits = model(images)
        loss = criterion(logits, expanded_labels)
    synchronize()
    forward_seconds = time.perf_counter() - started
    logits_finite = bool(torch.isfinite(logits).all())
    loss_value = float(loss.detach().float().cpu())
    loss_finite = math.isfinite(loss_value)

    failure_stage: str | None = None
    if not input_finite:
        failure_stage = "input"
    elif not logits_finite:
        failure_stage = "forward_logits"
    elif not loss_finite:
        failure_stage = "loss"

    backward_seconds = 0.0
    optimizer_seconds = 0.0
    grad_finite: bool | None = None
    clip_norm_finite: bool | None = None
    parameters_finite_after_step: bool | None = None
    optimizer_step_applied = False
    gradient_norm: float | None = None

    if failure_stage is None:
        synchronize()
        started = time.perf_counter()
        scaler.scale(loss).backward()
        synchronize()
        backward_seconds = time.perf_counter() - started

        synchronize()
        started = time.perf_counter()
        scaler.unscale_(optimizer)
        grad_finite = gradients_are_finite(model)
        if not grad_finite:
            failure_stage = "backward_gradient"
        else:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
                error_if_nonfinite=False,
            )
            gradient_norm = float(total_norm.detach().float().cpu())
            clip_norm_finite = math.isfinite(gradient_norm)
            if not clip_norm_finite:
                failure_stage = "gradient_clip_norm"
            else:
                scaler.step(optimizer)
                scaler.update()
                optimizer_step_applied = True
                parameters_finite_after_step = all(
                    bool(torch.isfinite(parameter).all())
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                if not parameters_finite_after_step:
                    failure_stage = "post_optimizer_parameters"
        synchronize()
        optimizer_seconds = time.perf_counter() - started

    nan_or_inf = failure_stage is not None
    batch_total_seconds = (
        transfer_seconds + forward_seconds + backward_seconds + optimizer_seconds
    )
    return {
        "source_images": int(len(labels)),
        "forward_images": int(len(expanded_labels)),
        "transfer_seconds": transfer_seconds,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_step_seconds": optimizer_seconds,
        "batch_total_seconds": batch_total_seconds,
        "loss": loss_value,
        "input_finite": input_finite,
        "logits_finite": logits_finite,
        "loss_finite": loss_finite,
        "grad_finite": grad_finite,
        "gradient_norm": gradient_norm,
        "clip_norm_finite": clip_norm_finite,
        "parameters_finite_after_step": parameters_finite_after_step,
        "optimizer_step_applied": optimizer_step_applied,
        "failure_stage": failure_stage,
        "nan_or_inf": nan_or_inf,
    }


def gradients_are_finite(model: torch.nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        return False
    return all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


def aggregate(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return mean(values), (pstdev(values) if len(values) > 1 else 0.0)


# --------------------------------------------------------------------------- #
# 단일 설정(Batch Size × Precision) 벤치마크
# --------------------------------------------------------------------------- #
def benchmark_setting(
    *,
    batch_size: int,
    precision: str,
    base_data_config: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    gradient_clip_norm: float,
    warmup_steps: int,
    measured_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    amp_enabled = precision == "fp16_amp"
    config = copy.deepcopy(base_data_config)
    config["loader"]["batch_size"] = batch_size
    train_loader, _ = build_dataloaders(config)

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")

    empty_cuda_cache()
    reset_peak_memory(device)

    raw_rows: list[dict[str, Any]] = []
    measured_rows: list[dict[str, Any]] = []
    failure_stages: list[str] = []
    train_iterator = iter(train_loader)
    total_steps = warmup_steps + measured_steps
    for step_index in range(total_steps):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        measurement = measure_train_step(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            batch=batch,
            device=device,
            amp_enabled=amp_enabled,
            gradient_clip_norm=gradient_clip_norm,
        )
        is_warmup = step_index < warmup_steps
        row = {
            "batch_size_source": batch_size,
            "actual_forward_batch": batch_size * 2,
            "precision": precision,
            "amp_enabled": amp_enabled,
            "step_index": step_index,
            "is_warmup": is_warmup,
            "repeat": None if is_warmup else step_index - warmup_steps + 1,
            **{
                key: value
                for key, value in measurement.items()
                if key not in {"source_images", "forward_images"}
            },
        }
        raw_rows.append(row)
        if measurement["nan_or_inf"]:
            failure_stages.append(str(measurement["failure_stage"]))
            # 수치적으로 오염된 설정에서 추가 update를 수행하지 않는다.
            break
        if not is_warmup:
            measured_rows.append(row)

    memory = peak_memory_bytes(device)
    frame = pd.DataFrame(measured_rows)
    timing_columns = [
        "transfer_seconds",
        "forward_seconds",
        "backward_seconds",
        "optimizer_step_seconds",
        "batch_total_seconds",
    ]
    numerical_valid = not failure_stages and len(measured_rows) == measured_steps
    status = "success" if numerical_valid else "numerical_failure"
    aggregate_row: dict[str, Any] = {
        "batch_size_source": batch_size,
        "actual_forward_batch": batch_size * 2,
        "precision": precision,
        "amp_enabled": amp_enabled,
        "status": status,
        "oom": False,
        "warmup_steps": warmup_steps,
        "measured_steps_requested": measured_steps,
        "measured_steps_completed": len(measured_rows),
        "numerically_valid": numerical_valid,
        "nan_or_inf": not numerical_valid,
        "failure_stages": "|".join(sorted(set(failure_stages))),
    }
    for column in timing_columns:
        values = frame[column].astype(float).tolist() if not frame.empty else []
        avg, std = aggregate(values)
        aggregate_row[f"{column}_mean"] = avg
        aggregate_row[f"{column}_std"] = std
    batch_total_mean = aggregate_row["batch_total_seconds_mean"]
    aggregate_row["source_images_per_second"] = (
        batch_size / batch_total_mean if numerical_valid and batch_total_mean > 0 else 0.0
    )
    aggregate_row["forward_images_per_second"] = (
        batch_size * 2 / batch_total_mean
        if numerical_valid and batch_total_mean > 0
        else 0.0
    )
    aggregate_row["peak_allocated_bytes"] = memory["peak_allocated_bytes"]
    aggregate_row["peak_reserved_bytes"] = memory["peak_reserved_bytes"]
    aggregate_row["total_memory_bytes"] = memory["total_bytes"]
    if memory["peak_reserved_bytes"] is not None and memory["total_bytes"]:
        aggregate_row["peak_memory_fraction"] = (
            memory["peak_reserved_bytes"] / memory["total_bytes"]
        )
    else:
        aggregate_row["peak_memory_fraction"] = None
    return aggregate_row, raw_rows


# --------------------------------------------------------------------------- #
# 학습 시간 추정
# --------------------------------------------------------------------------- #
def measure_checkpoint_seconds(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[float, int]:
    temporary = output_dir / "_checkpoint_benchmark_temporary.pt"
    started = time.perf_counter()
    save_checkpoint(
        temporary,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        metrics={"benchmark_only": True},
        config=config,
    )
    elapsed = time.perf_counter() - started
    size = temporary.stat().st_size
    temporary.unlink()
    return elapsed, size


def estimate_training_times(
    aggregate_rows: list[dict[str, Any]],
    checkpoint_seconds: float,
) -> dict[str, Any]:
    """측정한 Train Step 평균을 Train/Validation 전체로 선형 외삽한다.

    Validation 추론은 측정한 FP16/FP32 forward 시간을 기반으로 보수적으로 추정한다.
    실제 측정하지 않은 시간(checkpoint 외삽, metric 집계 등)은 추정치로 표시한다.
    """
    estimates: list[dict[str, Any]] = []
    for row in aggregate_rows:
        if row["status"] != "success":
            continue
        batch_size = int(row["batch_size_source"])
        train_steps = math.ceil(TRAIN_IMAGES / batch_size)
        # Validation은 single-view이므로 forward 입력이 Train(two-view)의 절반이다.
        validation_steps = math.ceil(VALIDATION_IMAGES / batch_size)
        train_epoch_seconds = train_steps * row["batch_total_seconds_mean"]
        # Validation step: transfer + forward만 발생(backward/optimizer 없음),
        # single-view라 forward 비용을 two-view 대비 0.5배로 보수 추정.
        validation_step_seconds = (
            row["transfer_seconds_mean"] + 0.5 * row["forward_seconds_mean"]
        )
        validation_seconds = validation_steps * validation_step_seconds
        epoch_seconds = train_epoch_seconds + validation_seconds
        estimates.append(
            {
                "batch_size_source": batch_size,
                "precision": row["precision"],
                "train_steps_per_epoch": train_steps,
                "validation_steps_per_epoch": validation_steps,
                "estimated_train_epoch_seconds": train_epoch_seconds,
                "estimated_validation_seconds_per_epoch": validation_seconds,
                "estimated_epoch_seconds_without_checkpoint": epoch_seconds,
                "estimated_15_epochs_seconds_without_checkpoint": epoch_seconds * EPOCHS,
                "estimated_15_epochs_seconds_with_5_checkpoints": (
                    epoch_seconds * EPOCHS + checkpoint_seconds * 5
                ),
                "estimated_15_epochs_seconds_worst_case_15_checkpoints": (
                    epoch_seconds * EPOCHS + checkpoint_seconds * EPOCHS
                ),
                "estimation_basis": "warm-up 후 측정 Step 평균의 선형 외삽(추정치)",
            }
        )
    return {
        "train_images": TRAIN_IMAGES,
        "validation_images": VALIDATION_IMAGES,
        "candidate_epochs": EPOCHS,
        "checkpoint_save_seconds_measured_once": checkpoint_seconds,
        "per_setting": estimates,
        "caveats": [
            "소수 Step 선형 외삽이라 실제 epoch 중 캐시/열 상태에 따라 달라질 수 있음(추정치)",
            "Validation forward는 single-view 가정으로 Train의 0.5배로 보수 추정(추정치)",
            "Validation 후 metric 집계·CSV 저장 시간은 미포함",
            "Checkpoint 1회만 실측, 나머지 저장 횟수는 외삽(추정치)",
            "Early stopping 발생 시 실제 시간은 감소",
        ],
    }


# --------------------------------------------------------------------------- #
# 추천 로직
# --------------------------------------------------------------------------- #
def build_recommendation(aggregate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """OOM과 어느 단계든 nonfinite인 설정을 최종 추천 후보에서 제외한다."""
    valid = [
        row
        for row in aggregate_rows
        if row.get("status") == "success"
        and row.get("numerically_valid", not row.get("nan_or_inf", True))
        and not row.get("nan_or_inf", True)
    ]
    excluded = [
        {
            "batch_size_source": row.get("batch_size_source"),
            "precision": row.get("precision"),
            "status": row.get("status"),
            "oom": row.get("oom"),
            "nan_or_inf": row.get("nan_or_inf"),
            "failure_stages": row.get("failure_stages", ""),
        }
        for row in aggregate_rows
        if row not in valid
    ]
    if not valid:
        return {
            "available": False,
            "reason": "OOM 없이 모든 numerical validity 검사를 통과한 설정이 없음",
            "excluded_settings": excluded,
        }

    fastest = max(valid, key=lambda row: row["source_images_per_second"])
    safe = [
        row
        for row in valid
        if row.get("peak_memory_fraction") is not None
        and row["peak_memory_fraction"] <= MEMORY_SAFETY_FRACTION
    ]
    safe_pool = safe or valid
    recommended = max(safe_pool, key=lambda row: row["source_images_per_second"])

    valid_amp_rows = [row for row in valid if row["amp_enabled"]]
    valid_fp32_rows = [row for row in valid if not row["amp_enabled"]]
    invalid_amp_rows = [
        row for row in aggregate_rows if row.get("amp_enabled") and row not in valid
    ]

    def speedup(amp_row: dict[str, Any]) -> float | None:
        match = next(
            (
                row
                for row in valid_fp32_rows
                if row["batch_size_source"] == amp_row["batch_size_source"]
            ),
            None,
        )
        if match is None or match["source_images_per_second"] == 0:
            return None
        return amp_row["source_images_per_second"] / match["source_images_per_second"]

    amp_speedups = {
        str(row["batch_size_source"]): speedup(row) for row in valid_amp_rows
    }
    amp_recommended = bool(valid_amp_rows) and not invalid_amp_rows and any(
        value is not None and value > 1.0 for value in amp_speedups.values()
    )
    return {
        "available": True,
        "memory_safety_fraction": MEMORY_SAFETY_FRACTION,
        "valid_candidate_count": len(valid),
        "excluded_settings": excluded,
        "fastest_numerically_valid_setting": {
            "batch_size_source": fastest["batch_size_source"],
            "precision": fastest["precision"],
            "source_images_per_second": fastest["source_images_per_second"],
            "peak_memory_fraction": fastest["peak_memory_fraction"],
        },
        "recommended_setting": {
            "batch_size_source": recommended["batch_size_source"],
            "actual_forward_batch": recommended["actual_forward_batch"],
            "precision": recommended["precision"],
            "source_images_per_second": recommended["source_images_per_second"],
            "peak_memory_fraction": recommended["peak_memory_fraction"],
            "numerically_valid": True,
            "policy": (
                "OOM 없음 + NaN/Inf 없음 + peak reserved <= "
                f"{int(MEMORY_SAFETY_FRACTION * 100)}% 후보 중 원본 img/s 최대"
            ),
        },
        "amp_recommendation": {
            "use_fp16_amp": amp_recommended,
            "reason": (
                "모든 AMP 설정이 numerical validity를 통과하고 동일 Batch FP32보다 빠름"
                if amp_recommended
                else (
                    "AMP 설정이 NaN/Inf 또는 다른 numerical validity 검사에 실패해 추천 후보에서 제외"
                    if invalid_amp_rows
                    else (
                        "AMP는 numerical validity를 통과했지만 동일 Batch FP32보다 빠른 설정이 없음"
                        if valid_amp_rows
                        else "numerically valid한 AMP 측정이 없음"
                    )
                )
            ),
            "valid_amp_speedup_by_batch_size": amp_speedups,
            "invalid_amp_settings": [
                {
                    "batch_size_source": row.get("batch_size_source"),
                    "failure_stages": row.get("failure_stages", "unknown"),
                }
                for row in invalid_amp_rows
            ],
        },
        "supcon_memory_note": (
            "E1/E3 추가 메모리를 고려하되 numerical validity가 메모리·속도보다 우선함"
        ),
        "note": "소규모 벤치마크 추천이며 전체 학습은 수행하지 않음",
    }


# --------------------------------------------------------------------------- #
# 메인
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_config = load_training_config(args.config, project_root=PROJECT_ROOT)
    data_config = load_data_config(
        training_config["resolved"]["data_config_path"], project_root=PROJECT_ROOT
    )
    device = resolve_device(str(training_config["training"]["device"]))
    environment = collect_environment(device)

    if device.type != "cuda":
        message = (
            f"CUDA를 사용할 수 없습니다(선택된 device={device}). "
            "이 벤치마크는 CUDA GPU(예: AWS JupyterLab의 Tesla T4)에서만 실행하세요."
        )
        print(message, flush=True)
        environment["aborted"] = {"reason": message, "benchmark_executed": False}
        (output_dir / "cuda_benchmark_summary.json").write_text(
            json.dumps(
                {"environment": environment, "benchmark_executed": False},
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        if not args.allow_non_cuda:
            sys.exit(1)
        return

    # 모델/가중치 로딩은 Step 측정 전에 1회만 수행(측정에서 제외).
    seed_everything(int(training_config["training"]["seed"]))
    model_load_started = time.perf_counter()
    model = build_e0_model(training_config).to(device)
    synchronize()
    model_load_seconds = time.perf_counter() - model_load_started
    environment["model_load_seconds_excluded_from_step_timing"] = model_load_seconds
    environment["model_parameters"] = model.parameter_summary()

    criterion = create_criterion(training_config)
    baseline_model_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    gradient_clip_norm = float(training_config["training"]["gradient_clip_norm"])
    environment["benchmark_policy"].update(
        {
            "batch_sizes_source_images": list(args.batch_sizes),
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
        }
    )

    aggregate_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []

    for batch_size in args.batch_sizes:
        for precision in PRECISIONS:
            # 각 설정은 동일한 model weight, 새 AdamW state, 동일 seed에서 시작한다.
            seed_everything(int(training_config["training"]["seed"]))
            model.load_state_dict(baseline_model_state, strict=True)
            optimizer = create_optimizer(model, training_config)
            model.train()
            label = f"Batch {batch_size} (Forward {batch_size * 2}) / {precision}"
            print(f"측정 시작: {label}", flush=True)
            empty_cuda_cache()
            try:
                aggregate_row, setting_raw_rows = benchmark_setting(
                    batch_size=batch_size,
                    precision=precision,
                    base_data_config=data_config,
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    gradient_clip_norm=gradient_clip_norm,
                    warmup_steps=args.warmup_steps,
                    measured_steps=args.measured_steps,
                )
                aggregate_rows.append(aggregate_row)
                raw_rows.extend(setting_raw_rows)
                if aggregate_row["status"] == "success":
                    print(
                        f"  성공: {aggregate_row['batch_total_seconds_mean']:.4f}s/batch, "
                        f"{aggregate_row['source_images_per_second']:.2f} src img/s, "
                        f"peak reserved {(aggregate_row['peak_reserved_bytes'] or 0) / 1024**3:.2f} GiB, "
                        "numerically_valid=True",
                        flush=True,
                    )
                else:
                    print(
                        "  수치 실패: "
                        f"stage={aggregate_row['failure_stages'] or 'unknown'}, "
                        "추천 후보에서 제외",
                        flush=True,
                    )
            except Exception as error:  # noqa: BLE001
                if not is_oom_error(error):
                    raise
                aggregate_rows.append(
                    {
                        "batch_size_source": batch_size,
                        "actual_forward_batch": batch_size * 2,
                        "precision": precision,
                        "amp_enabled": precision == "fp16_amp",
                        "status": "oom",
                        "oom": True,
                        "nan_or_inf": None,
                        "error": str(error),
                    }
                )
                print(f"  OOM 발생: {error}", flush=True)
            finally:
                # OOM 여부와 무관하게 다음 실험이 안전하도록 메모리를 정리한다.
                optimizer.zero_grad(set_to_none=True)
                empty_cuda_cache()

    # Checkpoint I/O 측정도 오염되지 않은 기준 상태와 새 optimizer/scheduler로 수행한다.
    model.load_state_dict(baseline_model_state, strict=True)
    optimizer = create_optimizer(model, training_config)
    scheduler = create_scheduler(optimizer, training_config)
    checkpoint_seconds, checkpoint_bytes = measure_checkpoint_seconds(
        model, optimizer, scheduler, training_config, output_dir
    )
    estimates = estimate_training_times(aggregate_rows, checkpoint_seconds)
    estimates["checkpoint_file_bytes"] = checkpoint_bytes
    recommendation = build_recommendation(aggregate_rows)

    # CSV 저장
    pd.DataFrame(aggregate_rows).to_csv(output_dir / "cuda_benchmark.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(output_dir / "cuda_benchmark_raw.csv", index=False)

    summary = {
        "environment": environment,
        "benchmark_executed": True,
        "aggregate_results": aggregate_rows,
        "training_time_estimate": estimates,
        "recommendation": recommendation,
        "constraints_respected": {
            "full_epochs_executed": 0,
            "test_images_accessed": 0,
            "aws_resources_created": 0,
            "manifest_modified": False,
            "original_images_modified": False,
            "e0_model_modified": False,
        },
    }
    (output_dir / "cuda_benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print("\n=== 벤치마크 완료 ===", flush=True)
    print(f"결과 저장: {output_dir}", flush=True)
    if recommendation.get("available"):
        rec = recommendation["recommended_setting"]
        print(
            f"추천 설정: Batch {rec['batch_size_source']} "
            f"(Forward {rec['actual_forward_batch']}) / {rec['precision']}, "
            f"{rec['source_images_per_second']:.2f} src img/s",
            flush=True,
        )
        print(
            f"FP16 AMP 사용 권장: {recommendation['amp_recommendation']['use_fp16_amp']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
