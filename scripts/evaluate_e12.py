#!/usr/bin/env python3
"""Read-only deterministic validation evaluation of an E12 weighted-focal checkpoint."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/"src"))
from data.dataloader import build_loader, load_data_config  # noqa: E402
from p1_cxr.checkpoints import load_checkpoint  # noqa: E402
from p1_cxr.e8_data import BINARY_CLASS_ORDER, build_e8_dataset  # noqa: E402
from p1_cxr.e8_metrics import binary_classification_metrics  # noqa: E402
from p1_cxr.e12_models import build_e12_model  # noqa: E402
from p1_cxr.e12_training import load_e12_training_config  # noqa: E402
from p1_cxr.training import resolve_device  # noqa: E402


def inside(path: Path) -> Path:
    value=path.expanduser(); value=value.resolve() if value.is_absolute() else (PROJECT_ROOT/value).resolve()
    if not value.is_relative_to(PROJECT_ROOT): raise ValueError(f"Path must remain inside project root: {path}")
    return value


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs"/"e12_training.yaml"); parser.add_argument("--checkpoint",type=Path,default=PROJECT_ROOT/"artifacts"/"experiments"/"e12"/"best.pt"); parser.add_argument("--device",default="auto"); args=parser.parse_args()
    config=load_e12_training_config(inside(args.config),project_root=PROJECT_ROOT); checkpoint=inside(args.checkpoint)
    if not checkpoint.is_file(): raise FileNotFoundError(f"E12 checkpoint not found: {checkpoint}")
    data=load_data_config(config["resolved"]["data_config_path"],project_root=PROJECT_ROOT)
    if data["views"]["validation"]!="single" or "test" in data.get("manifests",{}): raise ValueError("E12 evaluation permits deterministic validation only.")
    dataset=build_e8_dataset(data,"validation"); loader=build_loader(dataset,data,training=False)
    if len(dataset)!=550: raise ValueError("E12 evaluation requires fixed 550-image B/V validation subset.")
    device=resolve_device(args.device); model=build_e12_model(config).to(device); payload=load_checkpoint(checkpoint,model=model,map_location=device); control=payload.get("control_state",{})
    if payload.get("run_kind")!="e12_inverse_frequency_focal_e6_transfer_full_epoch" or payload.get("config",{}).get("experiment")!="e12": raise ValueError("Checkpoint is not E12.")
    if control.get("binary_class_order")!=list(BINARY_CLASS_ORDER) or control.get("class_weights")!=config["loss"]["class_weights"]: raise ValueError("E12 class-order/weight checkpoint provenance mismatch.")
    logits=[]; targets=[]; model.eval()
    with torch.inference_mode():
        for images,labels in loader:
            output=model(images.to(device))
            if output.ndim!=2 or output.shape[1]!=2 or not bool(torch.isfinite(output).all()): raise FloatingPointError("Invalid E12 validation logits.")
            logits.append(output.float().cpu()); targets.append(labels.cpu())
    probabilities=torch.softmax(torch.cat(logits),dim=1).numpy(); metrics=binary_classification_metrics(torch.cat(targets).numpy(),probabilities); confusion=np.asarray(metrics["confusion_matrix"],dtype=np.int64)
    print(f"device: {device}"); print(f"Accuracy: {metrics['accuracy']:.6f}"); print(f"Macro F1: {metrics['macro_f1']:.6f}"); print(f"AUROC (Viral positive): {metrics['auroc']}")
    print(f"Bacterial -> Viral: {int(confusion[0,1])}"); print(f"Viral -> Bacterial: {int(confusion[1,0])}"); print(f"Total B/V errors: {int(confusion[0,1]+confusion[1,0])}"); print("confusion matrix [true rows, predicted columns; Bacterial, Viral]:"); print(confusion)
    for name in BINARY_CLASS_ORDER:
        value=metrics["per_class"][name]; print(f"{name}: precision={value['precision']:.6f}, recall={value['recall']:.6f}, f1={value['f1']:.6f}, support={value['support']}")
    print("prediction source: softmax(binary logits) -> argmax; class_weights are training-loss-only")


if __name__=="__main__": main()
