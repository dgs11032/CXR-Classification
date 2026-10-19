#!/usr/bin/env python3
"""ConvNeXt-Tiny E0와 공통 engine을 작은 batch로만 검증한다.

전체 epoch 학습, Test 접근, AWS 작업, E1/E2/E3 구현은 수행하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataloader import build_dataloaders, build_dataset, load_data_config  # noqa: E402
from p1_cxr.checkpoints import load_checkpoint, save_checkpoint  # noqa: E402
from p1_cxr.metrics import prediction_frame  # noqa: E402
from p1_cxr.models import (  # noqa: E402
    ConvNeXtTinyE0,
    build_e0_model,
    pretrained_weight_cache_path,
)
from p1_cxr.training import (  # noqa: E402
    create_criterion,
    create_grad_scaler,
    create_optimizer,
    create_scheduler,
    evaluate,
    flatten_two_view_batch,
    load_training_config,
    optimizer_group_summary,
    resolve_device,
    seed_everything,
    train_one_epoch,
    validate_final_data_contract,
    validate_resume_checkpoint,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "e0_training.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_state(paths: Iterable[Path]) -> dict[str, tuple[int, int]]:
    return {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(set(paths))
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def nested_state_equal(left: Any, right: Any) -> bool:
    """Tensor를 포함한 optimizer/scheduler state dict를 재귀적으로 비교한다."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and bool(torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            nested_state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            nested_state_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def balanced_validation_indices(dataset, samples_per_class: int) -> list[int]:
    indices: list[int] = []
    for label in (0, 1, 2):
        candidates = dataset.frame.index[dataset.frame["label"] == label].tolist()
        if len(candidates) < samples_per_class:
            raise ValueError(f"Validation label {label} 표본이 부족합니다.")
        indices.extend(candidates[:samples_per_class])
    return indices


def summarize_model(model: ConvNeXtTinyE0, device: torch.device) -> str:
    parameter_summary = model.parameter_summary()
    lines = [
        "E0 DRY-RUN MODEL SUMMARY",
        "========================",
        "주의: 전체 학습 결과가 아닌 구조 검증용 모델입니다.",
        "",
        "Architecture: ConvNeXt-Tiny flat 3-class classification",
        f"Pretrained weights: {model.weights_name}",
        f"Device: {device}",
        f"Feature dimension: {model.feature_dim}",
        f"Classification head: {model.classifier}",
        f"Total parameters: {parameter_summary['total_parameters']:,}",
        f"Backbone parameters: {parameter_summary['backbone_parameters']:,}",
        f"Head parameters: {parameter_summary['classification_head_parameters']:,}",
        f"Trainable parameters: {parameter_summary['trainable_parameters']:,}",
        f"Trainable scope: {parameter_summary['trainable_scope']}",
        "",
        "Forward contract:",
        "- forward_features(images) -> [B, 768]",
        "- forward(images) -> logits [B, 3]",
        "- logits are passed directly to CrossEntropyLoss",
        "- Softmax is used only in evaluation/inference",
        "",
        repr(model),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    training_config = load_training_config(args.config, project_root=PROJECT_ROOT)
    data_config = load_data_config(
        training_config["resolved"]["data_config_path"], project_root=PROJECT_ROOT
    )
    validate_final_data_contract(data_config)
    output_dir = Path(training_config["resolved"]["dry_run_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(training_config["training"]["seed"])
    seed_everything(seed)

    train_manifest = Path(data_config["resolved"]["train_manifest"])
    validation_manifest = Path(data_config["resolved"]["validation_manifest"])
    manifest_hashes_before = {
        "train": sha256_file(train_manifest),
        "validation": sha256_file(validation_manifest),
    }

    train_dataset = build_dataset(data_config, "train", view_mode="two")
    validation_dataset = build_dataset(data_config, "validation", view_mode="single")
    input_paths = [
        *(train_dataset.resolve_path(index) for index in range(len(train_dataset))),
        *(
            validation_dataset.resolve_path(index)
            for index in range(len(validation_dataset))
        ),
    ]
    source_state_before = path_state(input_paths)

    weights_path = pretrained_weight_cache_path()
    if not weights_path.is_file():
        raise FileNotFoundError(
            "Dry-run은 pretrained weight를 자동 다운로드하지 않습니다: "
            f"{weights_path}"
        )
    weights_state_before = (weights_path.stat().st_size, weights_path.stat().st_mtime_ns)

    device = resolve_device(
        str(training_config["dry_run"].get("device", training_config["training"]["device"]))
    )
    model = build_e0_model(training_config).to(device)
    criterion = create_criterion(training_config)
    optimizer = create_optimizer(model, training_config)
    scheduler = create_scheduler(optimizer, training_config)
    optimizer_groups_initial = optimizer_group_summary(optimizer)
    amp_enabled = bool(training_config["training"]["amp"]) and device.type == "cuda"
    configured_scaler = create_grad_scaler(amp_enabled=amp_enabled, device=device)

    # 별도 probe loader로 실제 STEP 6 two-view batch의 shape와 label 순서를 확인한다.
    probe_train_loader, _ = build_dataloaders(data_config)
    probe_views, probe_labels = next(iter(probe_train_loader))
    flattened_images, expanded_labels = flatten_two_view_batch(probe_views, probe_labels)
    expected_labels = probe_labels.repeat_interleave(probe_views.shape[1])
    label_mapping_correct = bool(torch.equal(expanded_labels, expected_labels))

    model.eval()
    with torch.inference_mode():
        probe_features = model.forward_features(flattened_images.to(device))
        probe_logits = model(flattened_images.to(device))
    initial_ce_loss = criterion(probe_logits, expanded_labels.to(device))

    # 실제 학습 engine은 동일 Seed의 첫 Train batch 딱 1개만 처리한다.
    seed_everything(seed)
    dry_train_loader, _ = build_dataloaders(data_config)
    train_result = train_one_epoch(
        model,
        dry_train_loader,
        criterion,
        optimizer,
        device=device,
        amp_enabled=amp_enabled,
        scaler=configured_scaler,
        gradient_clip_norm=float(training_config["training"]["gradient_clip_norm"]),
        max_batches=int(training_config["dry_run"]["train_batches"]),
    )
    scheduler.step()
    optimizer_groups_after_scheduler = optimizer_group_summary(optimizer)

    # Macro AUROC를 포함한 지표 함수 검증을 위해 세 클래스 각 2장만 평가한다.
    samples_per_class = int(training_config["dry_run"]["validation_samples_per_class"])
    validation_indices = balanced_validation_indices(
        validation_dataset, samples_per_class
    )
    validation_subset = Subset(validation_dataset, validation_indices)
    validation_loader = DataLoader(
        validation_subset,
        batch_size=len(validation_subset),
        shuffle=False,
        num_workers=0,
    )
    validation_result = evaluate(
        model,
        validation_loader,
        criterion,
        device=device,
        amp_enabled=amp_enabled,
    )
    validation_paths = [
        str(validation_dataset.frame.iloc[index]["image_path"])
        for index in validation_indices
    ]
    predictions = prediction_frame(
        validation_paths,
        validation_result.targets,
        validation_result.probabilities,
    )
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)

    probability_sums = validation_result.probabilities.sum(axis=1)
    checkpoint_path = output_dir / "dryrun_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        metrics=validation_result.metrics,
        config=training_config,
        scaler=configured_scaler,
        best_validation_macro_f1=float(validation_result.metrics["macro_f1"]),
        random_seed=seed,
        control_state={"best_epoch": 0, "stale_epochs": 0},
        history=[{"epoch": 0, "dry_run": True}],
        train_loader_generator=dry_train_loader.generator,
        save_rng_state=True,
        run_kind="dry_run",
    )

    # Random initialization 동형 모델과 새 optimizer/scheduler에 전체 상태를 복원한다.
    reloaded_model = ConvNeXtTinyE0(
        pretrained=False,
        trainable_scope=str(training_config["model"]["trainable_scope"]),  # type: ignore[arg-type]
    ).to(device)
    reloaded_optimizer = create_optimizer(reloaded_model, training_config)
    reloaded_scheduler = create_scheduler(reloaded_optimizer, training_config)
    reloaded_train_generator = torch.Generator()
    loaded_payload = load_checkpoint(
        checkpoint_path,
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        scheduler=reloaded_scheduler,
        scaler=configured_scaler,
        train_loader_generator=reloaded_train_generator,
        restore_rng=True,
        map_location=device,
    )
    optimizer_state_equal = nested_state_equal(
        optimizer.state_dict(), reloaded_optimizer.state_dict()
    )
    scheduler_state_equal = nested_state_equal(
        scheduler.state_dict(), reloaded_scheduler.state_dict()
    )
    generator_state_equal = torch.equal(
        reloaded_train_generator.get_state(),
        loaded_payload["train_loader_generator_state"].cpu(),
    )
    dryrun_resume_rejected = False
    try:
        validate_resume_checkpoint(
            loaded_payload,
            training_config=training_config,
            seed=seed,
            manifest_fingerprints=manifest_hashes_before,
        )
    except ValueError as error:
        dryrun_resume_rejected = "run_kind=full_epoch" in str(error)
    validation_images, _ = next(iter(validation_loader))
    model.eval()
    reloaded_model.eval()
    with torch.inference_mode():
        original_logits = model(validation_images.to(device)).float().cpu()
        reloaded_logits = reloaded_model(validation_images.to(device)).float().cpu()
    checkpoint_max_difference = float(
        (original_logits - reloaded_logits).abs().max()
    )
    reloaded_model.set_trainable_scope("head_only")
    head_only_summary = reloaded_model.parameter_summary()
    reloaded_model.set_trainable_scope("full")
    full_summary = reloaded_model.parameter_summary()

    source_state_after = path_state(input_paths)
    manifest_hashes_after = {
        "train": sha256_file(train_manifest),
        "validation": sha256_file(validation_manifest),
    }
    weights_state_after = (weights_path.stat().st_size, weights_path.stat().st_mtime_ns)

    checks = {
        "final_batch_size_16": {
            "passed": int(data_config["loader"]["batch_size"]) == 16,
            "actual": int(data_config["loader"]["batch_size"]),
        },
        "fp32_amp_disabled": {
            "passed": (
                training_config["training"]["precision"] == "fp32"
                and not bool(training_config["training"]["amp"])
                and configured_scaler is None
            ),
            "precision": training_config["training"]["precision"],
            "amp": bool(training_config["training"]["amp"]),
            "grad_scaler_created": configured_scaler is not None,
        },
        "differential_learning_rates": {
            "passed": (
                optimizer_groups_initial["backbone"]["learning_rate"] == 1e-5
                and optimizer_groups_initial["head"]["learning_rate"] == 1e-4
                and optimizer_groups_initial["backbone"]["parameters"] == 27820128
                and optimizer_groups_initial["head"]["parameters"] == 2307
            ),
            "groups": optimizer_groups_initial,
        },
        "scheduler_adjusts_both_groups": {
            "passed": (
                optimizer_groups_after_scheduler["backbone"]["learning_rate"]
                < optimizer_groups_initial["backbone"]["learning_rate"]
                and optimizer_groups_after_scheduler["head"]["learning_rate"]
                < optimizer_groups_initial["head"]["learning_rate"]
            ),
            "before": optimizer_groups_initial,
            "after": optimizer_groups_after_scheduler,
        },
        "full_validation_configured_742": {
            "passed": len(validation_dataset) == 742,
            "configured_images": len(validation_dataset),
            "dry_run_evaluated_images": validation_result.processed_images,
        },
        "best_last_checkpoint_policy": {
            "passed": training_config["training"]["checkpoint"] == {
                "best_filename": "best.pt",
                "last_filename": "last.pt",
                "save_rng_state": True,
            },
            "config": training_config["training"]["checkpoint"],
        },
        "dryrun_checkpoint_rejected_for_full_resume": {
            "passed": dryrun_resume_rejected,
            "checkpoint_run_kind": loaded_payload["run_kind"],
        },
        "model_created": {
            "passed": isinstance(model, ConvNeXtTinyE0),
        },
        "trainable_scope_configuration": {
            "passed": (
                head_only_summary["trainable_parameters"] == 2307
                and full_summary["trainable_parameters"]
                == full_summary["total_parameters"]
            ),
            "head_only_trainable_parameters": head_only_summary["trainable_parameters"],
            "full_trainable_parameters": full_summary["trainable_parameters"],
            "note": "scope별 requires_grad 구성 검증이며 head-only optimizer step은 아직 수행하지 않음",
        },
        "input_shape": {
            "passed": tuple(flattened_images.shape[1:]) == (3, 224, 224),
            "actual": list(flattened_images.shape),
        },
        "logits_shape": {
            "passed": tuple(probe_logits.shape) == (len(flattened_images), 3),
            "actual": list(probe_logits.shape),
        },
        "feature_vector_shape": {
            "passed": tuple(probe_features.shape) == (len(flattened_images), 768),
            "actual": list(probe_features.shape),
        },
        "two_view_label_mapping": {
            "passed": label_mapping_correct,
            "source_labels_preview": probe_labels[:6].tolist(),
            "expanded_labels_preview": expanded_labels[:12].tolist(),
        },
        "cross_entropy_loss": {
            "passed": bool(torch.isfinite(initial_ce_loss)) and float(initial_ce_loss) > 0,
            "value_before_update": float(initial_ce_loss.detach().cpu()),
        },
        "backpropagation": {
            "passed": train_result.batches == 1 and train_result.processed_views > 0,
            "processed_batches": train_result.batches,
            "processed_source_images": train_result.processed_source_images,
            "processed_views": train_result.processed_views,
            "train_loss": train_result.loss,
        },
        "gradients_finite": {
            "passed": train_result.gradients_finite,
        },
        "validation_forward": {
            "passed": (
                validation_result.processed_images == samples_per_class * 3
                and validation_result.logits.shape == (samples_per_class * 3, 3)
            ),
            "processed_images": validation_result.processed_images,
            "logits_shape": list(validation_result.logits.shape),
            "validation_loss": validation_result.loss,
        },
        "softmax_probability_sum": {
            "passed": bool(np.allclose(probability_sums, 1.0, atol=1e-6)),
            "minimum_sum": float(probability_sums.min()),
            "maximum_sum": float(probability_sums.max()),
        },
        "evaluation_metrics": {
            "passed": (
                validation_result.metrics["macro_auroc_ovr"] is not None
                and set(validation_result.metrics["per_class"])
                == {"NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA"}
            ),
            "metrics": validation_result.metrics,
        },
        "checkpoint_round_trip": {
            "passed": (
                loaded_payload["checkpoint_version"] == 2
                and loaded_payload["run_kind"] == "dry_run"
                and loaded_payload["epoch"] == 0
                and loaded_payload["best_validation_macro_f1"] is not None
                and loaded_payload["random_seed"] == seed
                and loaded_payload["rng_state"] is not None
                and loaded_payload["train_loader_generator_state"] is not None
                and checkpoint_max_difference <= 1e-6
                and optimizer_state_equal
                and scheduler_state_equal
                and generator_state_equal
            ),
            "checkpoint_path": str(checkpoint_path),
            "maximum_logit_difference": checkpoint_max_difference,
            "optimizer_state_equal": optimizer_state_equal,
            "scheduler_state_equal": scheduler_state_equal,
            "train_loader_generator_state_equal": generator_state_equal,
            "scheduler_last_epoch": scheduler.state_dict()["last_epoch"],
            "reloaded_scheduler_last_epoch": reloaded_scheduler.state_dict()["last_epoch"],
        },
        "pretrained_weights_from_cache": {
            "passed": (
                model.weights_name == "IMAGENET1K_V1"
                and weights_state_before == weights_state_after
            ),
            "cache_path": str(weights_path),
            "cache_bytes": weights_state_before[0],
            "cache_mtime_unchanged": weights_state_before[1]
            == weights_state_after[1],
            "automatic_download_allowed": False,
        },
        "source_images_unchanged": {
            "passed": source_state_before == source_state_after,
            "checked_images": len(source_state_before),
        },
        "manifests_unchanged": {
            "passed": manifest_hashes_before == manifest_hashes_after,
            "before": manifest_hashes_before,
            "after": manifest_hashes_after,
        },
        "test_not_accessed": {
            "passed": set(data_config["manifests"]) == {"train", "validation"},
            "configured_manifests": sorted(data_config["manifests"]),
            "test_images_loaded": 0,
        },
        "dry_run_scope_only": {
            "passed": train_result.batches == int(training_config["dry_run"]["train_batches"]),
            "full_epochs_executed": 0,
            "aws_resources_created": 0,
        },
    }
    failed_checks = [name for name, result in checks.items() if not result["passed"]]
    dryrun_report = {
        "status": "성공" if not failed_checks else "실패",
        "artifact_type": "DRY_RUN_ONLY_NOT_A_TRAINING_RESULT",
        "all_checks_passed": not failed_checks,
        "failed_checks": failed_checks,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "model": model.parameter_summary(),
        "checks": checks,
        "constraints": {
            "full_epoch_training_executed": False,
            "test_data_accessed": False,
            "aws_resource_created": False,
            "e1_e2_e3_implemented": False,
            "source_images_modified": False,
            "manifests_modified": False,
        },
        "loss_policy": {
            "criterion": "unweighted CrossEntropyLoss",
            "purpose": "E0 비교 기준선",
            "class_imbalance_note": (
                "BACTERIAL 표본이 각 다른 클래스의 약 1.9배이므로 전체 학습에서는 "
                "클래스별 Precision/Recall/F1과 혼동행렬을 반드시 함께 보고"
            ),
            "weighted_loss_or_sampler": "E0에는 미적용; 필요 시 별도 승인된 ablation으로 검토",
        },
        "rotation_fill_review": {
            "current_setting": 0,
            "common_transform_modified_in_step7": False,
            "observation": (
                "±5° 회전에서 모서리에 검은 삼각형이 생길 수 있고, 0은 정규화 후 극단값이 됨"
            ),
            "alternatives_requiring_user_approval": [
                "이미지별 border median을 fill로 사용하는 custom rotation",
                "reflect/edge padding 후 회전하고 224×224로 복귀",
                "회전 확률 또는 각도를 더 줄임",
                "fill=0을 유지하되 모든 클래스에서 동일 적용하고 후속 오류 분석",
            ],
            "recommendation": (
                "현재 설정을 임의 변경하지 않음. E0 본 학습 전 border-median fill 또는 "
                "reflect padding의 소규모 시각 검증을 권장"
            ),
        },
    }
    validation_report = {
        "artifact_type": "DRY_RUN_BALANCED_VALIDATION_SUBSET",
        "warning": "세 클래스 각 2장, 총 6장에 대한 코드 검증값이며 모델 성능으로 해석하지 않음",
        "validation_loss": validation_result.loss,
        "metrics": validation_result.metrics,
        "predictions": predictions.to_dict(orient="records"),
        "probability_sum_minimum": float(probability_sums.min()),
        "probability_sum_maximum": float(probability_sums.max()),
    }
    write_json(output_dir / "dryrun_report.json", dryrun_report)
    write_json(output_dir / "validation_report.json", validation_report)
    (output_dir / "model_summary.txt").write_text(
        summarize_model(model, device), encoding="utf-8"
    )

    print("\n=== E0 ConvNeXt-Tiny Dry-run ===")
    print(f"Device: {device}, pretrained: {model.weights_name} (cache only)")
    print(f"Feature shape: {tuple(probe_features.shape)}")
    print(f"Logits shape: {tuple(probe_logits.shape)}")
    print(
        f"Train: {train_result.batches} batch, "
        f"{train_result.processed_source_images} source / {train_result.processed_views} views"
    )
    print(
        f"Dry-run validation: {validation_result.processed_images} images, "
        f"loss={validation_result.loss:.4f}, "
        f"macro_f1={validation_result.metrics['macro_f1']:.4f}"
    )
    for name, result in checks.items():
        print(f"- {'통과' if result['passed'] else '실패'}: {name}")
    print(f"결과 저장 위치: {output_dir}")
    if failed_checks:
        print(
            "Dry-run 검증 실패: " + ", ".join(failed_checks),
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
