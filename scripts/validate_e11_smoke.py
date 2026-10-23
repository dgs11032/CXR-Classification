#!/usr/bin/env python3
"""E11 confidence-routing smoke validator; full mode requires E6/E9 best checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from p1_cxr.e10_hierarchical import E6_CLASS_ORDER, E9_CLASS_ORDER, hard_route, load_e10_models, validation_alignment  # noqa: E402
from p1_cxr.e11_confidence_routing import THRESHOLDS, confidence_gate, load_e11_analysis_config  # noqa: E402
from p1_cxr.training import resolve_device  # noqa: E402

E11_ARTIFACT = PROJECT_ROOT / "artifacts" / "experiments" / "e11"


def snapshot() -> dict[str, str]:
    protected: set[Path] = set()
    for directory, pattern in ((PROJECT_ROOT / "configs", "*.yaml"), (PROJECT_ROOT / "src" / "data", "*.py"), (PROJECT_ROOT / "src" / "p1_cxr", "*.py"), (PROJECT_ROOT / "scripts", "*.py")):
        if directory.is_dir(): protected.update(path for path in directory.rglob(pattern) if "e11" not in path.name.lower())
    for name in ("train_manifest.csv", "val_manifest.csv"):
        path = PROJECT_ROOT / "artifacts" / "splits" / name
        if path.is_file(): protected.add(path)
    for name in ("e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10"):
        directory = PROJECT_ROOT / "artifacts" / "experiments" / name
        if directory.is_dir(): protected.update(path for path in directory.rglob("*") if path.is_file())
    return {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(protected)}


def synthetic_gate_checks() -> None:
    e6 = np.asarray([0, 1, 2, 1, 2, 1], dtype=np.int64)
    e9 = np.asarray([1, 0, 1, 1, 0, 0], dtype=np.int64)
    confidence = np.asarray([0.99, 0.92, 0.88, 0.71, 0.66, 0.54], dtype=np.float64)
    hard = hard_route(e6, e9)
    final_050, override_050 = confidence_gate(e6, e9, confidence, 0.50)
    if not np.array_equal(final_050, hard): raise RuntimeError("E11 threshold=0.50 synthetic route differs from E10 hard route.")
    if not np.all(final_050[e6 == 0] == 0): raise RuntimeError("E11 E6 Normal was not preserved.")
    agreement = (e6 != 0) & (e6 == e9 + 1)
    if not np.all(final_050[agreement] == e6[agreement]): raise RuntimeError("E11 subtype agreement changed unexpectedly.")
    counts = []
    for threshold in THRESHOLDS:
        final, override = confidence_gate(e6, e9, confidence, threshold)
        if not np.all(final[e6 == 0] == 0) or not np.all(final[agreement] == e6[agreement]): raise RuntimeError("E11 gate changed a protected sample.")
        counts.append(int(override.sum()))
    if any(later > earlier for earlier, later in zip(counts, counts[1:])): raise RuntimeError(f"E11 override count increased: {counts}")
    if not bool(override_050.sum()): raise RuntimeError("E11 synthetic disagreement gate was not exercised.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e11_analysis.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--static-only", action="store_true", help="Validate config, threshold rules, and alignment without checkpoints.")
    args = parser.parse_args()
    if E11_ARTIFACT.exists(): raise RuntimeError("E11 artifact directory already exists; smoke test must not write or reuse it.")
    before = snapshot()
    config = load_e11_analysis_config(args.config, project_root=PROJECT_ROOT)
    synthetic_gate_checks()
    _, loader, frame, alignment = validation_alignment(config["resolved"]["e10_config"])
    e10_config = config["resolved"]["e10_config"]
    e6_path, e9_path = Path(e10_config["resolved"]["e6_checkpoint"]), Path(e10_config["resolved"]["e9_checkpoint"])
    if list(E6_CLASS_ORDER) != ["NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA"] or list(E9_CLASS_ORDER) != ["BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA"]: raise RuntimeError("E11 class-order contract failed.")
    if args.static_only:
        after = snapshot()
        if before != after or E11_ARTIFACT.exists(): raise RuntimeError("Static E11 smoke modified protected files or created an artifact.")
        print(f"static config/gating/alignment validation: passed; threshold_grid={list(THRESHOLDS)}")
        print(f"alignment={alignment}; class_orders=E6:{list(E6_CLASS_ORDER)}, E9:{list(E9_CLASS_ORDER)}")
        print(f"E6/E9 checkpoint presence: E6={e6_path.is_file()}, E9={e9_path.is_file()} (full checkpoint smoke skipped)")
        print(f"E0-E10/common/train-val protection hashes: passed; protected_files={len(after)}")
        return
    if not e6_path.is_file() or not e9_path.is_file(): raise FileNotFoundError(f"E11 full smoke requires E6/E9 best checkpoints: E6={e6_path.is_file()}, E9={e9_path.is_file()}")
    device = resolve_device(args.device or "auto")
    e6_model, e9_model, checkpoint_info = load_e10_models(e10_config, device=device)
    images, labels = next(iter(loader))
    if labels.tolist() != frame.iloc[:len(labels)]["label"].astype(int).tolist(): raise RuntimeError("E11 first-batch manifest alignment failed.")
    with torch.inference_mode(): e6_logits, e9_logits = e6_model(images.to(device)), e9_model(images.to(device))
    if e6_logits.shape != (len(labels), 3) or e9_logits.shape != (len(labels), 2) or not bool(torch.isfinite(e6_logits).all()) or not bool(torch.isfinite(e9_logits).all()): raise RuntimeError("E11 checkpoint inference shape/finiteness failed.")
    e6_pred = torch.softmax(e6_logits, dim=1).argmax(dim=1).cpu().numpy(); e9_prob = torch.softmax(e9_logits, dim=1).cpu().numpy(); e9_pred, confidence = e9_prob.argmax(axis=1), e9_prob.max(axis=1)
    hard = hard_route(e6_pred, e9_pred); final_050, _ = confidence_gate(e6_pred, e9_pred, confidence, 0.50)
    if not np.array_equal(hard, final_050): raise RuntimeError("E11 real threshold=0.50 does not equal E10 hard routing.")
    counts = [int(confidence_gate(e6_pred, e9_pred, confidence, threshold)[1].sum()) for threshold in THRESHOLDS]
    if any(later > earlier for earlier, later in zip(counts, counts[1:])): raise RuntimeError("E11 real override monotonicity failed.")
    after = snapshot()
    if before != after or E11_ARTIFACT.exists(): raise RuntimeError("E11 full smoke modified protected files or created an artifact.")
    print(f"device: {device}; checkpoint contracts={checkpoint_info['contracts']}")
    print(f"input={tuple(images.shape)}; E6 logits={tuple(e6_logits.shape)}; E9 logits={tuple(e9_logits.shape)}; override_counts={counts}")
    print(f"threshold=0.50 equals E10 hard routing: passed=True; manifest alignment={alignment}")
    print(f"E0-E10/common/train-val protection hashes: passed; protected_files={len(after)}")
    print("artifacts/experiments/e11 absent before/after smoke: passed=True")


if __name__ == "__main__": main()
