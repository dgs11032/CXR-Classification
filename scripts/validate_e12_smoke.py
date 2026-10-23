#!/usr/bin/env python3
"""E12 weighted-focal smoke validator; full mode requires E6 best checkpoint."""
from __future__ import annotations
import argparse
import hashlib
import sys
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/"src"))
from data.dataloader import load_data_config  # noqa: E402
from p1_cxr.e8_data import BINARY_CLASS_ORDER, EXPECTED_COUNTS, build_e8_dataloaders  # noqa: E402
from p1_cxr.e8_training import evaluate_e8  # noqa: E402
from p1_cxr.e9_models import transfer_e6_encoder  # noqa: E402
from p1_cxr.e12_models import build_e12_model  # noqa: E402
from p1_cxr.e12_training import CLASS_WEIGHTS, create_e12_criterion, load_e12_training_config  # noqa: E402
from p1_cxr.training import create_optimizer, flatten_two_view_batch, resolve_device, seed_everything  # noqa: E402
E12_ARTIFACT=PROJECT_ROOT/"artifacts"/"experiments"/"e12"

def snapshot() -> dict[str,str]:
    protected:set[Path]=set()
    for directory,pattern in ((PROJECT_ROOT/"configs","*.yaml"),(PROJECT_ROOT/"src"/"data","*.py"),(PROJECT_ROOT/"src"/"p1_cxr","*.py"),(PROJECT_ROOT/"scripts","*.py")):
        if directory.is_dir(): protected.update(path for path in directory.rglob(pattern) if "e12" not in path.name.lower())
    for name in ("train_manifest.csv","val_manifest.csv"):
        path=PROJECT_ROOT/"artifacts"/"splits"/name
        if path.is_file(): protected.add(path)
    for name in ("e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","e10","e11"):
        directory=PROJECT_ROOT/"artifacts"/"experiments"/name
        if directory.is_dir(): protected.update(path for path in directory.rglob("*") if path.is_file())
    return {str(path.relative_to(PROJECT_ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(protected)}

def balanced(dataset: object, per_class:int)->DataLoader:
    frame=dataset.frame; indices=[]
    for label in (0,1): indices.extend(frame.index[frame["label"]==label].tolist()[:per_class])
    return DataLoader(Subset(dataset,indices),batch_size=len(indices),shuffle=False)

def has_grad(module:nn.Module)->bool:
    return any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in module.parameters() if p.requires_grad)

def main()->None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs"/"e12_training.yaml"); parser.add_argument("--device",default=None); parser.add_argument("--static-only",action="store_true"); args=parser.parse_args()
    if E12_ARTIFACT.exists(): raise RuntimeError("E12 artifact exists; smoke must not reuse/write it.")
    before=snapshot(); config=load_e12_training_config(args.config,project_root=PROJECT_ROOT); source=Path(config["resolved"]["source_checkpoint_path"]); model=build_e12_model(config); criterion=create_e12_criterion(config)
    weights_ok=torch.allclose(criterion.class_weights.cpu(),torch.tensor(CLASS_WEIGHTS),rtol=0.0,atol=1e-7) and criterion.class_weights[1]>criterion.class_weights[0] and criterion.alpha is None and criterion.gamma==2.0
    if tuple(model.classifier.weight.shape)!=(2,768) or tuple(model.classifier.bias.shape)!=(2,) or not weights_ok: raise RuntimeError("E12 static model/loss contract failed.")
    synthetic_logits=torch.tensor([[1.2,0.0],[0.0,1.2]],dtype=torch.float32); synthetic_targets=torch.tensor([0,1],dtype=torch.long); synthetic=criterion.components(synthetic_logits,synthetic_targets)
    weighted_ratio_ok=torch.allclose(synthetic.per_sample_ce[0],synthetic.per_sample_ce[1]) and torch.allclose(synthetic.modulation[0],synthetic.modulation[1]) and synthetic.per_sample_focal[1]>synthetic.per_sample_focal[0]
    if not weighted_ratio_ok: raise RuntimeError("E12 synthetic Viral weighting check failed.")
    if args.static_only:
        after=snapshot()
        if before!=after or E12_ARTIFACT.exists(): raise RuntimeError("Static E12 smoke modified protected files or created artifact.")
        print(f"static config/model/loss validation: passed; class_order={list(BINARY_CLASS_ORDER)}; weights={criterion.class_weights.tolist()}; synthetic_viral_weighted_more={weighted_ratio_ok}")
        print(f"E9/E12 parity: {config['resolved']['e9_parity']}"); print(f"E6 source checkpoint present: {source.is_file()} (full transfer smoke skipped)"); print(f"E0-E11/common/train-val protection hashes: passed; protected_files={len(after)}"); return
    if not source.is_file(): raise FileNotFoundError(f"E12 full smoke requires E6 best checkpoint: {source}")
    data=load_data_config(config["resolved"]["data_config_path"],project_root=PROJECT_ROOT)
    if "test" in data.get("manifests",{}): raise RuntimeError("E12 smoke must not use test.")
    seed_everything(int(config["training"]["seed"])); device=resolve_device(args.device or str(config["smoke_test"]["device"])); train_loader,val_loader=build_e8_dataloaders(data)
    counts_ok=len(train_loader.dataset)==3300 and len(val_loader.dataset)==550 and train_loader.dataset.class_counts()==EXPECTED_COUNTS['train'] and val_loader.dataset.class_counts()==EXPECTED_COUNTS['validation']
    if not counts_ok: raise RuntimeError("E12 fixed subset contract failed.")
    report=transfer_e6_encoder(model,checkpoint_path=source,project_root=PROJECT_ROOT)
    if not(report['all_encoder_tensors_equal'] and report['fresh_binary_classifier_preserved'] and report['transferred_encoder_tensor_count']==180 and report['missing_keys']==[] and report['unexpected_keys']==[]): raise RuntimeError(f"E12 transfer report failed: {report}")
    # Equal binary logits and swapped targets give equal CE/modulation; Viral has larger alpha-weighted contribution.
    synthetic_logits=torch.tensor([[1.2,0.0],[0.0,1.2]],dtype=torch.float32); synthetic_targets=torch.tensor([0,1],dtype=torch.long); synthetic=criterion.components(synthetic_logits,synthetic_targets)
    weighted_ratio_ok=torch.allclose(synthetic.per_sample_ce[0],synthetic.per_sample_ce[1]) and torch.allclose(synthetic.modulation[0],synthetic.modulation[1]) and synthetic.per_sample_focal[1]>synthetic.per_sample_focal[0]
    if not weighted_ratio_ok: raise RuntimeError("E12 synthetic Viral weighting check failed.")
    model=model.to(device); criterion=criterion.to(device); optimizer=create_optimizer(model,config); views,labels=next(iter(train_loader)); flattened,expanded=flatten_two_view_batch(views,labels); flattened,expanded=flattened.to(device),expanded.to(device); optimizer.zero_grad(set_to_none=True); logits=model(flattened); components=criterion.components(logits,expanded); loss=components.per_sample_focal.mean()
    manual=torch.mean(components.class_weight_t*components.modulation*components.per_sample_ce); focal_ok=bool(torch.isfinite(loss)) and torch.allclose(loss,manual) and bool((components.pt>=0).all() and (components.pt<=1).all())
    if logits.shape[1]!=2 or not focal_ok: raise RuntimeError("E12 weighted focal/logit contract failed.")
    loss.backward(); gradients={'encoder':has_grad(model.features) and has_grad(model.feature_norm),'classifier':has_grad(model.classifier)}
    if not all(gradients.values()): raise RuntimeError(f"E12 gradients failed: {gradients}")
    head=model.classifier.weight.detach().clone(); optimizer.step(); step_ok=not torch.equal(head,model.classifier.weight.detach())
    if not step_ok: raise RuntimeError("E12 optimizer step failed.")
    validation=evaluate_e8(model,balanced(val_loader.dataset,int(config['smoke_test']['validation_samples_per_class'])),device=device); metrics_ok=validation.metrics['macro_f1'] is not None and validation.metrics['auroc'] is not None and not validation.focal_used_for_prediction
    if not metrics_ok: raise RuntimeError("E12 validation prediction path failed.")
    after=snapshot()
    if before!=after or E12_ARTIFACT.exists(): raise RuntimeError("E12 full smoke modified protected files or created artifact.")
    print(f"device: {device}; E9/E12 parity={config['resolved']['e9_parity']}; subset_counts passed={counts_ok}")
    print(f"transfer={report}; weights={criterion.class_weights.detach().cpu().tolist()}; synthetic_viral_weighted_more={weighted_ratio_ok}")
    print(f"input/flattened/logits={tuple(views.shape)}->{tuple(flattened.shape)}->{tuple(logits.shape)}; focal finite/manual={focal_ok}; gradients={gradients}; optimizer_step={step_ok}")
    print(f"validation logits={tuple(validation.logits.shape)}; metrics/prediction path passed={metrics_ok}")
    print(f"E0-E11/common/train-val protection hashes: passed; protected_files={len(after)}"); print("artifacts/experiments/e12 absent before/after smoke: passed=True")
if __name__=='__main__': main()
