#!/usr/bin/env python3
"""E10 hard-routing smoke validator; full mode requires actual E6/E9 best checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from p1_cxr.e10_hierarchical import (  # noqa: E402
    E6_CLASS_ORDER,
    E9_CLASS_ORDER,
    hard_route,
    load_e10_evaluation_config,
    load_e10_models,
    validation_alignment,
)
from p1_cxr.training import resolve_device  # noqa: E402

E10_ARTIFACT = PROJECT_ROOT / "artifacts" / "experiments" / "e10"


def snapshot() -> dict[str, str]:
    protected: set[Path] = set()
    for directory, pattern in ((PROJECT_ROOT / "configs", "*.yaml"), (PROJECT_ROOT / "src" / "data", "*.py"), (PROJECT_ROOT / "src" / "p1_cxr", "*.py"), (PROJECT_ROOT / "scripts", "*.py")):
        if directory.is_dir():
            protected.update(path for path in directory.rglob(pattern) if "e10" not in path.name.lower())
    for name in ("train_manifest.csv", "val_manifest.csv"):
        path = PROJECT_ROOT / "artifacts" / "splits" / name
        if path.is_file(): protected.add(path)
    for name in ("e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9"):
        directory = PROJECT_ROOT / "artifacts" / "experiments" / name
        if directory.is_dir(): protected.update(path for path in directory.rglob("*") if path.is_file())
    return {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(protected)}


def check_routing() -> None:
    e6 = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    e9 = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.int64)
    final = hard_route(e6, e9)
    expected = np.asarray([0, 1, 2, 0, 2, 1], dtype=np.int64)
    if not np.array_equal(final, expected): raise RuntimeError(f"E10 synthetic routing mismatch: {final} != {expected}")
    if not np.all(final[e6 == 0] == 0) or not np.all(final[e6 != 0] == e9[e6 != 0] + 1):
        raise RuntimeError("E10 routing invariants failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e10_evaluation.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--static-only", action="store_true", help="Validate config/routing/alignment without best checkpoints.")
    args = parser.parse_args()
    if E10_ARTIFACT.exists(): raise RuntimeError("E10 artifact directory already exists; smoke test must not write or reuse it.")
    before = snapshot()
    config = load_e10_evaluation_config(args.config, project_root=PROJECT_ROOT)
    check_routing()
    _, loader, frame, alignment = validation_alignment(config)
    if list(E6_CLASS_ORDER) != ["NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA"] or list(E9_CLASS_ORDER) != ["BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA"]:
        raise RuntimeError("E10 E6/E9 class-order contract failed.")
    e6_path, e9_path = Path(config["resolved"]["e6_checkpoint"]), Path(config["resolved"]["e9_checkpoint"])
    if args.static_only:
        after = snapshot()
        if before != after or E10_ARTIFACT.exists(): raise RuntimeError("Static E10 smoke modified protected files or created an artifact.")
        print(f"static config/routing/alignment validation: passed; alignment={alignment}")
        print(f"class orders: E6={list(E6_CLASS_ORDER)}; E9={list(E9_CLASS_ORDER)}")
        print(f"E6/E9 checkpoint presence: E6={e6_path.is_file()}, E9={e9_path.is_file()} (full checkpoint smoke skipped)")
        print(f"E0-E9/common/train-val protection hashes: passed; protected_files={len(after)}")
        return
    if not e6_path.is_file() or not e9_path.is_file():
        raise FileNotFoundError(f"E10 full smoke requires both best checkpoints: E6={e6_path.is_file()}, E9={e9_path.is_file()}")
    device = resolve_device(args.device or "auto")
    e6_model, e9_model, checkpoint_info = load_e10_models(config, device=device)
    images, labels = next(iter(loader))
    expected_labels = frame.iloc[:len(labels)]["label"].astype(int).tolist()
    if labels.tolist() != expected_labels: raise RuntimeError("E10 first-batch manifest alignment failed.")
    with torch.inference_mode():
        e6_logits, e9_logits = e6_model(images.to(device)), e9_model(images.to(device))
    if e6_logits.shape != (len(labels), 3) or e9_logits.shape != (len(labels), 2) or not bool(torch.isfinite(e6_logits).all()) or not bool(torch.isfinite(e9_logits).all()):
        raise RuntimeError(f"E10 checkpoint inference shape/finiteness failed: E6={tuple(e6_logits.shape)}, E9={tuple(e9_logits.shape)}")
    e6_pred = torch.softmax(e6_logits, dim=1).argmax(dim=1).cpu().numpy()
    e9_pred = torch.softmax(e9_logits, dim=1).argmax(dim=1).cpu().numpy()
    final = hard_route(e6_pred, e9_pred)
    if not np.all(final[e6_pred == 0] == 0) or not np.all(final[e6_pred != 0] == e9_pred[e6_pred != 0] + 1): raise RuntimeError("E10 checkpoint routing invariant failed.")
    after = snapshot()
    if before != after or E10_ARTIFACT.exists(): raise RuntimeError("E10 full smoke modified protected files or created an artifact.")
    print(f"device: {device}; checkpoint contracts: {checkpoint_info['contracts']}")
    print(f"alignment: {alignment}; first-batch input={tuple(images.shape)}; E6 logits={tuple(e6_logits.shape)}; E9 logits={tuple(e9_logits.shape)}")
    print(f"E6 Normal preserved={bool(np.all(final[e6_pred == 0] == 0))}; E6 B/V remapped by E9={bool(np.all(final[e6_pred != 0] == e9_pred[e6_pred != 0] + 1))}")
    print(f"E0-E9/common/train-val protection hashes: passed; protected_files={len(after)}")
    print("artifacts/experiments/e10 absent before/after smoke: passed=True")


if __name__ == "__main__": main()
