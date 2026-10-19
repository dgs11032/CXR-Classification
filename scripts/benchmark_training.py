#!/usr/bin/env python3
"""E0 전체 학습 전 macOS/MPS 환경과 Batch 4·8·16 처리량을 소규모 측정한다.

전체 epoch, Test, AWS 리소스, E1/E2/E3는 실행하지 않는다.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "benchmarks"
BATCH_SIZES = (4, 8, 16)
WARMUP_STEPS = 1
MEASURED_STEPS = 3
TRAIN_IMAGES = 4448
VALIDATION_IMAGES = 742
EPOCHS = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--measured-steps", type=int, default=MEASURED_STEPS)
    return parser.parse_args()


def run_text(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_snapshot(device: torch.device) -> dict[str, int | None]:
    if device.type == "cuda":
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "driver_or_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "recommended_max_bytes": int(
                torch.cuda.get_device_properties(device).total_memory
            ),
        }
    if device.type == "mps":
        return {
            "allocated_bytes": int(torch.mps.current_allocated_memory()),
            "driver_or_reserved_bytes": int(torch.mps.driver_allocated_memory()),
            "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
        }
    return {
        "allocated_bytes": None,
        "driver_or_reserved_bytes": None,
        "recommended_max_bytes": None,
    }


def max_memory_snapshots(snapshots: list[dict[str, int | None]]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for key in snapshots[0]:
        values = [snapshot[key] for snapshot in snapshots if snapshot[key] is not None]
        result[key] = max(values) if values else None
    return result


def vm_available_bytes_approx() -> int | None:
    try:
        output = run_text(["vm_stat"])
        page_match = re.search(r"page size of (\d+) bytes", output)
        if page_match is None:
            return None
        page_size = int(page_match.group(1))
        pages = 0
        for name in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
            match = re.search(rf"{re.escape(name)}:\s+(\d+)\.", output)
            if match:
                pages += int(match.group(1))
        return pages * page_size
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def collect_environment(device: torch.device) -> dict[str, Any]:
    total_ram = int(run_text(["sysctl", "-n", "hw.memsize"]))
    disk = shutil.disk_usage(PROJECT_ROOT)
    cpu = run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
    gpu_cores: int | None = None
    try:
        display_data = json.loads(
            run_text(["system_profiler", "SPDisplaysDataType", "-json"])
        )["SPDisplaysDataType"][0]
        gpu_cores = int(display_data.get("sppci_cores"))
    except (KeyError, IndexError, TypeError, ValueError, subprocess.SubprocessError):
        pass
    return {
        "operating_system": {
            "name": "macOS",
            "version": platform.mac_ver()[0],
            "architecture": platform.machine(),
        },
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
        },
        "compute": {
            "selected_device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "chip_or_cpu": cpu,
            "gpu": "Apple M4 integrated GPU" if device.type == "mps" else str(device),
            "gpu_cores": gpu_cores,
            "memory_model": "Apple unified memory" if device.type == "mps" else "discrete/CPU",
            "device_memory_initial": memory_snapshot(device),
        },
        "system_memory": {
            "total_bytes": total_ram,
            "available_bytes_approx": vm_available_bytes_approx(),
            "note": (
                "macOS available은 free+inactive+speculative+purgeable page 근사값이며 "
                "MPS GPU와 CPU가 unified memory를 공유함"
            ),
        },
        "disk": {
            "path": str(PROJECT_ROOT),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "device_limitations": {
            "cuda_amp_available": torch.cuda.is_available(),
            "benchmark_amp_enabled": device.type == "cuda",
            "mps_peak_memory_limit": (
                "CUDA max_memory_allocated와 같은 정확한 peak counter가 없어 단계별 "
                "current/driver allocated의 최대 관측값을 보고"
                if device.type == "mps"
                else None
            ),
            "mps_operation_note": (
                "일부 CUDA 전용 연산·AMP API를 사용할 수 없고 미지원 연산은 CPU fallback 또는 오류 가능"
                if device.type == "mps"
                else None
            ),
        },
        "aws_sagemaker_readiness": {
            "resources_created_or_queried": False,
            "credentials_reported": False,
            "training_job_ready": False,
            "portable_components": [
                "CSV manifests",
                "YAML configs",
                "ConvNeXt E0 module",
                "common training engine",
            ],
            "missing_before_training_job": [
                "SageMaker training entrypoint",
                "/opt/ml input·model·output channel contract",
                "pinned container/runtime specification",
                "job submission configuration and IAM role validation",
            ],
        },
    }


def timed_data_batch(iterator) -> tuple[tuple[torch.Tensor, torch.Tensor], float]:
    started = time.perf_counter()
    batch = next(iterator)
    return batch, time.perf_counter() - started


def measure_train_step(
    model,
    optimizer,
    criterion,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    gradient_clip_norm: float,
) -> dict[str, Any]:
    views, labels = batch
    images, expanded_labels = flatten_two_view_batch(views, labels)
    memory_samples = [memory_snapshot(device)]

    synchronize(device)
    started = time.perf_counter()
    images = images.to(device)
    expanded_labels = expanded_labels.to(device)
    synchronize(device)
    transfer_seconds = time.perf_counter() - started
    memory_samples.append(memory_snapshot(device))

    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    started = time.perf_counter()
    logits = model(images)
    loss = criterion(logits, expanded_labels)
    synchronize(device)
    forward_seconds = time.perf_counter() - started
    memory_samples.append(memory_snapshot(device))

    synchronize(device)
    started = time.perf_counter()
    loss.backward()
    synchronize(device)
    backward_seconds = time.perf_counter() - started
    memory_samples.append(memory_snapshot(device))

    synchronize(device)
    started = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
    optimizer.step()
    synchronize(device)
    optimizer_seconds = time.perf_counter() - started
    memory_samples.append(memory_snapshot(device))

    return {
        "source_images": len(labels),
        "actual_forward_images": len(expanded_labels),
        "transfer_seconds": transfer_seconds,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_step_seconds": optimizer_seconds,
        "compute_total_seconds": (
            transfer_seconds + forward_seconds + backward_seconds + optimizer_seconds
        ),
        "loss": float(loss.detach().cpu()),
        "memory": max_memory_snapshots(memory_samples),
    }


@torch.inference_mode()
def measure_validation_step(
    model,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict[str, float | int]:
    images, labels = batch
    synchronize(device)
    started = time.perf_counter()
    images = images.to(device)
    synchronize(device)
    transfer_seconds = time.perf_counter() - started

    synchronize(device)
    started = time.perf_counter()
    logits = model(images)
    probabilities = torch.softmax(logits, dim=1)
    synchronize(device)
    forward_seconds = time.perf_counter() - started
    if not bool(torch.isfinite(probabilities).all()):
        raise FloatingPointError("Validation probability에 NaN/Inf가 있습니다.")
    return {
        "images": len(labels),
        "transfer_seconds": transfer_seconds,
        "forward_seconds": forward_seconds,
        "compute_total_seconds": transfer_seconds + forward_seconds,
    }


def aggregate(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def benchmark_batch_size(
    *,
    batch_size: int,
    base_data_config: dict[str, Any],
    model,
    optimizer,
    criterion,
    device: torch.device,
    gradient_clip_norm: float,
    warmup_steps: int,
    measured_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = copy.deepcopy(base_data_config)
    config["loader"]["batch_size"] = batch_size
    train_loader, validation_loader = build_dataloaders(config)
    train_iterator = iter(train_loader)
    validation_iterator = iter(validation_loader)
    raw_rows: list[dict[str, Any]] = []

    for _ in range(warmup_steps):
        batch, _ = timed_data_batch(train_iterator)
        measure_train_step(
            model, optimizer, criterion, batch, device, gradient_clip_norm
        )
        validation_batch, _ = timed_data_batch(validation_iterator)
        measure_validation_step(model, validation_batch, device)

    for repeat in range(1, measured_steps + 1):
        batch, data_seconds = timed_data_batch(train_iterator)
        train_measurement = measure_train_step(
            model, optimizer, criterion, batch, device, gradient_clip_norm
        )
        validation_batch, validation_data_seconds = timed_data_batch(validation_iterator)
        validation_measurement = measure_validation_step(
            model, validation_batch, device
        )
        raw_rows.append(
            {
                "batch_size_source": batch_size,
                "actual_forward_batch": batch_size * 2,
                "repeat": repeat,
                "train_data_loading_seconds": data_seconds,
                **{
                    key: value
                    for key, value in train_measurement.items()
                    if key not in {"memory"}
                },
                "peak_allocated_bytes_observed": train_measurement["memory"]["allocated_bytes"],
                "peak_driver_or_reserved_bytes_observed": train_measurement["memory"]["driver_or_reserved_bytes"],
                "recommended_max_bytes": train_measurement["memory"]["recommended_max_bytes"],
                "validation_data_loading_seconds": validation_data_seconds,
                "validation_transfer_seconds": validation_measurement["transfer_seconds"],
                "validation_forward_seconds": validation_measurement["forward_seconds"],
                "validation_compute_total_seconds": validation_measurement["compute_total_seconds"],
            }
        )

    frame = pd.DataFrame(raw_rows)
    aggregate_row: dict[str, Any] = {
        "batch_size_source": batch_size,
        "actual_forward_batch": batch_size * 2,
        "status": "success",
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "oom": False,
    }
    timing_columns = [
        "train_data_loading_seconds",
        "transfer_seconds",
        "forward_seconds",
        "backward_seconds",
        "optimizer_step_seconds",
        "compute_total_seconds",
        "validation_data_loading_seconds",
        "validation_transfer_seconds",
        "validation_forward_seconds",
        "validation_compute_total_seconds",
    ]
    for column in timing_columns:
        average, deviation = aggregate(frame[column].astype(float).tolist())
        aggregate_row[f"{column}_mean"] = average
        aggregate_row[f"{column}_std"] = deviation
    aggregate_row["batch_total_seconds_mean"] = (
        aggregate_row["train_data_loading_seconds_mean"]
        + aggregate_row["compute_total_seconds_mean"]
    )
    aggregate_row["source_images_per_second"] = (
        batch_size / aggregate_row["batch_total_seconds_mean"]
    )
    aggregate_row["view_images_per_second"] = (
        batch_size * 2 / aggregate_row["batch_total_seconds_mean"]
    )
    aggregate_row["peak_allocated_bytes_observed"] = int(
        frame["peak_allocated_bytes_observed"].max()
    )
    aggregate_row["peak_driver_or_reserved_bytes_observed"] = int(
        frame["peak_driver_or_reserved_bytes_observed"].max()
    )
    aggregate_row["recommended_max_bytes"] = int(frame["recommended_max_bytes"].max())
    aggregate_row["observed_driver_memory_ratio"] = (
        aggregate_row["peak_driver_or_reserved_bytes_observed"]
        / aggregate_row["recommended_max_bytes"]
    )
    return aggregate_row, raw_rows


def measure_checkpoint_seconds(model, optimizer, scheduler, config, output_dir: Path) -> tuple[float, int]:
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
    estimates: list[dict[str, Any]] = []
    for row in aggregate_rows:
        if row["status"] != "success":
            continue
        batch_size = int(row["batch_size_source"])
        train_steps = math.ceil(TRAIN_IMAGES / batch_size)
        validation_steps = math.ceil(VALIDATION_IMAGES / batch_size)
        train_epoch_seconds = train_steps * row["batch_total_seconds_mean"]
        validation_step_seconds = (
            row["validation_data_loading_seconds_mean"]
            + row["validation_compute_total_seconds_mean"]
        )
        validation_seconds = validation_steps * validation_step_seconds
        epoch_seconds = train_epoch_seconds + validation_seconds
        estimates.append(
            {
                "batch_size_source": batch_size,
                "actual_forward_batch": batch_size * 2,
                "train_steps_per_epoch": train_steps,
                "validation_steps_per_epoch": validation_steps,
                "estimated_train_epoch_seconds": train_epoch_seconds,
                "estimated_validation_seconds": validation_seconds,
                "estimated_epoch_seconds_without_checkpoint": epoch_seconds,
                "estimated_15_epochs_seconds_without_checkpoint": epoch_seconds * EPOCHS,
                "estimated_15_epochs_seconds_with_5_checkpoints": (
                    epoch_seconds * EPOCHS + checkpoint_seconds * 5
                ),
                "estimated_15_epochs_seconds_worst_case_15_checkpoints": (
                    epoch_seconds * EPOCHS + checkpoint_seconds * EPOCHS
                ),
                "basis": "소수 warm-up 후 batch 평균의 선형 외삽",
            }
        )
    return {
        "train_images": TRAIN_IMAGES,
        "validation_images": VALIDATION_IMAGES,
        "candidate_epochs": EPOCHS,
        "checkpoint_save_seconds_measured_once": checkpoint_seconds,
        "per_batch_size": estimates,
        "caveats": [
            "소수 batch의 선형 외삽으로 실제 epoch 중 캐시·열·메모리 압력에 따라 달라질 수 있음",
            "num_workers=0이며 데이터 로딩과 증강 시간이 포함됨",
            "Validation 후 전체 metric 집계·CSV 저장 시간은 포함하지 않음",
            "Early stopping이 발생하면 실제 시간은 감소",
        ],
    }


def choose_batch(aggregate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        row
        for row in aggregate_rows
        if row["status"] == "success"
        and row["observed_driver_memory_ratio"] <= 0.80
    ]
    if not successful:
        successful = [row for row in aggregate_rows if row["status"] == "success"]
    if not successful:
        return {"batch_size": None, "reason": "성공한 Batch Size가 없음"}
    selected = max(successful, key=lambda row: row["source_images_per_second"])
    return {
        "batch_size": int(selected["batch_size_source"]),
        "actual_forward_batch": int(selected["actual_forward_batch"]),
        "source_images_per_second": selected["source_images_per_second"],
        "observed_driver_memory_ratio": selected["observed_driver_memory_ratio"],
        "policy": "관측 메모리 비율 80% 이하 후보 중 원본 이미지 처리량 최대",
    }


def seconds_text(value: float) -> str:
    if value < 60:
        return f"{value:.1f}초"
    if value < 3600:
        return f"{value / 60:.1f}분"
    return f"{value / 3600:.2f}시간"


def write_markdown_summary(
    path: Path,
    environment: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
    estimates: dict[str, Any],
    recommendation: dict[str, Any],
    checkpoint_seconds: float,
    experiment_estimate: dict[str, Any],
) -> None:
    lines = [
        "# E0 학습 환경 및 Batch Size 벤치마크",
        "",
        "> 소수 Batch 측정 결과이며 전체 Epoch 학습 결과가 아니다. Test와 AWS는 사용하지 않았다.",
        "",
        "## 환경",
        "",
        f"- 장치: {environment['compute']['gpu']} ({environment['compute']['gpu_cores']} GPU cores)",
        f"- 메모리: {environment['system_memory']['total_bytes'] / 1024**3:.1f} GiB unified memory",
        f"- PyTorch: {environment['software']['pytorch']}",
        f"- CUDA: {environment['compute']['cuda_available']}, MPS: {environment['compute']['mps_available']}",
        "",
        "## Batch Size 측정",
        "",
        "| 원본 Batch | Forward 입력 | Forward | Backward | Optimizer | 총 Batch | 원본 img/s | 관측 driver memory |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        if row["status"] != "success":
            lines.append(f"| {row['batch_size_source']} | {row['actual_forward_batch']} | - | - | - | OOM | - | - |")
            continue
        lines.append(
            f"| {row['batch_size_source']} | {row['actual_forward_batch']} | "
            f"{row['forward_seconds_mean']:.3f}s | {row['backward_seconds_mean']:.3f}s | "
            f"{row['optimizer_step_seconds_mean']:.3f}s | {row['batch_total_seconds_mean']:.3f}s | "
            f"{row['source_images_per_second']:.2f} | "
            f"{row['peak_driver_or_reserved_bytes_observed'] / 1024**3:.2f} GiB |"
        )
    lines.extend(
        [
            "",
            f"- Checkpoint 1회 저장: {seconds_text(checkpoint_seconds)}",
            f"- 추천 원본 Batch Size: {recommendation['batch_size']} (실제 Forward {recommendation.get('actual_forward_batch')})",
            "",
            "## 15 Epoch 예상",
            "",
            "| Batch | Train steps | Val steps | Train/Epoch | Validation | 15 Epoch(Checkpoint 5회) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in estimates["per_batch_size"]:
        lines.append(
            f"| {row['batch_size_source']} | {row['train_steps_per_epoch']} | "
            f"{row['validation_steps_per_epoch']} | {seconds_text(row['estimated_train_epoch_seconds'])} | "
            f"{seconds_text(row['estimated_validation_seconds'])} | "
            f"{seconds_text(row['estimated_15_epochs_seconds_with_5_checkpoints'])} |"
        )
    lines.extend(
        [
            "",
            "## 권고",
            "",
            "- Fine-tuning: Backbone 1e-5 / Head 1e-4 differential LR를 1순위 후보로 권고.",
            "- Head-only warm-up: 필수는 아니며 초기 loss가 불안정할 때 1 epoch 선택 적용.",
            "- Rotation fill: 현재 fill=0은 유지 중. 본 학습 전 border-median fill 시각 검증 후 승인해 변경 여부 결정.",
            "- 불확실성: E1~E3는 아직 구현하지 않아 E0 실측치에 구조별 가정 배수를 적용한 범위 추정임.",
            f"- E0~E3 15 Epoch 총 예상: {seconds_text(experiment_estimate['lower_seconds'])}–{seconds_text(experiment_estimate['upper_seconds'])}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_config = load_training_config(args.config, project_root=PROJECT_ROOT)
    data_config = load_data_config(
        training_config["resolved"]["data_config_path"], project_root=PROJECT_ROOT
    )
    device = resolve_device(str(training_config["training"]["device"]))
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    environment = collect_environment(device)

    seed_everything(int(training_config["training"]["seed"]))
    model = build_e0_model(training_config).to(device)
    criterion = create_criterion(training_config)
    optimizer = create_optimizer(model, training_config)
    scheduler = create_scheduler(optimizer, training_config)
    model.train()

    aggregate_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    stop_after_oom = False
    for batch_size in BATCH_SIZES:
        if stop_after_oom:
            aggregate_rows.append(
                {
                    "batch_size_source": batch_size,
                    "actual_forward_batch": batch_size * 2,
                    "status": "skipped_after_previous_oom",
                    "oom": None,
                }
            )
            continue
        # 이전 Batch 크기의 비활성 allocator cache가 다음 memory 관측값에 섞이지 않게 한다.
        optimizer.zero_grad(set_to_none=True)
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Batch Size {batch_size} 측정 시작 (Forward {batch_size * 2})", flush=True)
        try:
            aggregate_row, batch_raw_rows = benchmark_batch_size(
                batch_size=batch_size,
                base_data_config=data_config,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                gradient_clip_norm=float(
                    training_config["training"]["gradient_clip_norm"]
                ),
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
            )
            aggregate_rows.append(aggregate_row)
            raw_rows.extend(batch_raw_rows)
            print(
                f"  성공: {aggregate_row['batch_total_seconds_mean']:.3f}s/batch, "
                f"{aggregate_row['source_images_per_second']:.2f} source img/s, "
                f"driver {aggregate_row['peak_driver_or_reserved_bytes_observed'] / 1024**3:.2f} GiB",
                flush=True,
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            aggregate_rows.append(
                {
                    "batch_size_source": batch_size,
                    "actual_forward_batch": batch_size * 2,
                    "status": "oom",
                    "oom": True,
                    "error": str(error),
                }
            )
            stop_after_oom = True
            optimizer.zero_grad(set_to_none=True)
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  OOM 발생, 더 큰 Batch를 중단합니다: {error}", flush=True)

    checkpoint_seconds, checkpoint_bytes = measure_checkpoint_seconds(
        model, optimizer, scheduler, training_config, output_dir
    )
    estimates = estimate_training_times(aggregate_rows, checkpoint_seconds)
    recommendation = choose_batch(aggregate_rows)
    selected_estimate = next(
        (
            row
            for row in estimates["per_batch_size"]
            if row["batch_size_source"] == recommendation["batch_size"]
        ),
        None,
    )
    if selected_estimate is None:
        experiment_estimate = {
            "lower_seconds": 0.0,
            "upper_seconds": 0.0,
            "available": False,
        }
    else:
        e0_seconds = selected_estimate["estimated_15_epochs_seconds_with_5_checkpoints"]
        multipliers = {
            "E0_flat_ce": [1.00, 1.00],
            "E1_flat_ce_supcon_assumption": [1.10, 1.30],
            "E2_hierarchical_assumption": [1.00, 1.10],
            "E3_hierarchical_supcon_assumption": [1.15, 1.40],
        }
        experiment_estimate = {
            "available": True,
            "selected_batch_size": recommendation["batch_size"],
            "e0_15_epochs_seconds": e0_seconds,
            "assumption_multipliers": multipliers,
            "lower_seconds": e0_seconds * sum(value[0] for value in multipliers.values()),
            "upper_seconds": e0_seconds * sum(value[1] for value in multipliers.values()),
            "warning": "E1~E3 미구현 상태의 가정 범위이며 실측값이 아님",
        }

    environment["benchmark_policy"] = {
        "batch_sizes_source_images": list(BATCH_SIZES),
        "actual_forward_batches": [value * 2 for value in BATCH_SIZES],
        "warmup_steps_per_batch_size": args.warmup_steps,
        "measured_steps_per_batch_size": args.measured_steps,
        "amp_enabled": device.type == "cuda",
        "synchronization": "torch.mps.synchronize" if device.type == "mps" else "torch.cuda.synchronize" if device.type == "cuda" else "none",
        "full_epochs_executed": 0,
        "test_images_accessed": 0,
        "aws_resources_created": 0,
        "pretrained_cache_path": str(pretrained_weight_cache_path()),
    }
    environment["model_memory_theoretical_fp32"] = {
        "parameters": model.parameter_summary(),
        "weights_bytes": model.parameter_summary()["total_parameters"] * 4,
        "gradients_bytes": model.parameter_summary()["trainable_parameters"] * 4,
        "adam_first_second_moments_bytes": model.parameter_summary()["trainable_parameters"] * 8,
        "note": "activation, allocator overhead, DataLoader tensors 제외",
    }
    estimates["recommended_batch"] = recommendation
    estimates["checkpoint_file_bytes"] = checkpoint_bytes
    estimates["e0_to_e3_total_estimate"] = experiment_estimate
    estimates["fine_tuning_review"] = {
        "single_lr": {
            "backbone_lr": 1e-4,
            "head_lr": 1e-4,
            "advantages": ["설정 단순", "현재 engine과 config로 바로 지원"],
            "risks": ["작은 의료영상 데이터에서 pretrained feature를 빠르게 훼손할 가능성"],
        },
        "differential_lr": {
            "backbone_lr": 1e-5,
            "head_lr": 1e-4,
            "advantages": ["pretrained feature 보존", "새 3-class head는 빠르게 적응"],
            "risks": ["optimizer parameter group 구현 및 기록 필요", "수렴이 느릴 수 있음"],
        },
        "recommendation": (
            "Differential LR를 1순위 후보로 권고. Head-only warm-up은 필수로 두지 않고 "
            "초기 validation/loss가 불안정한 경우 1 epoch 선택 적용"
        ),
        "performance_training_executed": False,
    }
    estimates["rotation_fill_review"] = {
        "current": "fill=0 유지, 설정 미변경",
        "fill_zero": {
            "advantages": ["단순하고 X-ray 외곽의 검은 배경과 유사", "모든 클래스 동일 적용"],
            "risks": ["정규화 후 극단값의 삼각형 모서리", "회전 적용 여부 shortcut 가능"],
        },
        "image_border_brightness_fill": {
            "advantages": ["각 영상 배경과 자연스럽게 연결", "검은 삼각형 대비 감소"],
            "risks": ["custom transform 필요", "영상별 fill 통계 계산과 재현성 기록 필요"],
        },
        "reflect_padding_rotation": {
            "advantages": ["상수색 경계 제거", "연속적인 가장자리"],
            "risks": ["외곽 문자·해부 구조를 거울 복제", "새로운 반복 패턴 생성 가능"],
        },
        "recommendation": (
            "본 학습 전 Train 표본에서 border-median fill을 우선 시각 검토. "
            "Reflect는 문자·표식 복제 위험 때문에 2순위. 사용자 승인 전 변경하지 않음"
        ),
    }

    pd.DataFrame(aggregate_rows).to_csv(output_dir / "batch_benchmark.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(output_dir / "batch_benchmark_raw.csv", index=False)
    (output_dir / "environment_report.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "training_time_estimate.json").write_text(
        json.dumps(estimates, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    write_markdown_summary(
        output_dir / "benchmark_summary.md",
        environment,
        aggregate_rows,
        estimates,
        recommendation,
        checkpoint_seconds,
        experiment_estimate,
    )

    print("\n=== STEP 8-A 벤치마크 완료 ===")
    print(f"장치: {device}")
    print(f"Checkpoint 저장 측정: {checkpoint_seconds:.3f}s, {checkpoint_bytes / 1024**2:.1f} MiB")
    print(f"추천 원본 Batch: {recommendation['batch_size']}")
    for estimate in estimates["per_batch_size"]:
        print(
            f"Batch {estimate['batch_size_source']}: "
            f"Epoch {seconds_text(estimate['estimated_epoch_seconds_without_checkpoint'])}, "
            f"15 Epoch {seconds_text(estimate['estimated_15_epochs_seconds_with_5_checkpoints'])}"
        )
    print(f"결과 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()
