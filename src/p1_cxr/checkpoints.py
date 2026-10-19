"""E0 학습을 정확히 재개하기 위한 atomic checkpoint와 RNG 상태 관리."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

CHECKPOINT_VERSION = 2


def config_fingerprint(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(tuple(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([tensor.cpu() for tensor in state["torch_cuda"]])


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    epoch: int,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    best_validation_macro_f1: float | None = None,
    random_seed: int | None = None,
    control_state: Mapping[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    train_loader_generator: torch.Generator | None = None,
    save_rng_state: bool = True,
    run_kind: str = "utility",
) -> Path:
    """완료 Epoch 경계의 전체 상태를 임시 파일 후 atomic replace로 저장한다."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "run_kind": run_kind,
        "config_fingerprint": config_fingerprint(config),
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "state_contract": "scheduler stepped; optimizer LR is for next epoch",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "metrics": dict(metrics),
        "best_validation_macro_f1": best_validation_macro_f1,
        "random_seed": random_seed,
        "control_state": dict(control_state or {}),
        "history": list(history or []),
        "train_loader_generator_state": (
            train_loader_generator.get_state()
            if train_loader_generator is not None
            else None
        ),
        "rng_state": capture_rng_state() if save_rng_state else None,
        "config": dict(config),
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    train_loader_generator: torch.Generator | None = None,
    restore_rng: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """이 프로젝트가 생성한 신뢰 가능한 v2 checkpoint 전체 상태를 복원한다."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint 파일이 없습니다: {source}")
    # RNG와 DataLoader Generator state는 CPU ByteTensor여야 하므로 payload는 항상 CPU로 읽는다.
    # model.load_state_dict와 optimizer.load_state_dict가 대상 parameter device로 복사한다.
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"지원하지 않는 checkpoint version: {payload.get('checkpoint_version')}"
        )
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    if (
        train_loader_generator is not None
        and payload.get("train_loader_generator_state") is not None
    ):
        train_loader_generator.set_state(
            payload["train_loader_generator_state"].cpu()
        )
    if restore_rng and payload.get("rng_state") is not None:
        restore_rng_state(payload["rng_state"])
    return payload
