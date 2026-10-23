"""E12-only E9-parity E6-transfer training with inverse-frequency weighted focal loss."""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from data.dataloader import load_data_config
from .checkpoints import save_checkpoint
from .e8_data import BINARY_CLASS_ORDER, EXPECTED_COUNTS, build_e8_dataloaders
from .e8_training import evaluate_e8, train_one_epoch_e8
from .e9_models import transfer_e6_encoder
from .e12_models import ConvNeXtTinyE12, build_e12_model
from .training import PROJECT_ROOT, create_optimizer, create_scheduler, file_sha256, resolve_device, seed_everything

GAMMA = 2.0
CLASS_WEIGHTS = (3300 / (2 * 2153), 3300 / (2 * 1147))


@dataclass
class WeightedFocalComponents:
    per_sample_ce: Tensor
    pt: Tensor
    modulation: Tensor
    class_weight_t: Tensor
    per_sample_focal: Tensor


class InverseFrequencyBinaryFocalLoss(nn.Module):
    """Mean(class_weight[target] * (1-pt)^2 * unweighted CE); alpha remains disabled."""
    def __init__(self, *, gamma: float = GAMMA, alpha: None = None, class_weights: tuple[float, float] = CLASS_WEIGHTS, reduction: str = "mean") -> None:
        super().__init__()
        weights = torch.tensor(class_weights, dtype=torch.float32)
        if gamma != GAMMA or alpha is not None or reduction != "mean" or weights.shape != (2,) or not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
            raise ValueError("E12 requires gamma=2.0, alpha=None, two positive class_weights, and reduction='mean'.")
        if not torch.allclose(weights, torch.tensor(CLASS_WEIGHTS), rtol=0.0, atol=1e-7):
            raise ValueError(f"E12 class_weights must be inverse-frequency {CLASS_WEIGHTS}.")
        self.gamma, self.alpha, self.reduction = gamma, alpha, reduction
        self.register_buffer("class_weights", weights)

    def components(self, logits: Tensor, targets: Tensor) -> WeightedFocalComponents:
        if logits.ndim != 2 or logits.shape[1] != 2 or targets.ndim != 1 or len(logits) != len(targets) or not bool(torch.isin(targets, torch.tensor([0, 1], device=targets.device)).all()):
            raise ValueError("E12 expects logits [N,2] and binary targets [N].")
        per_sample_ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=0.0)
        pt = torch.exp(-per_sample_ce)
        modulation = (1.0 - pt).pow(self.gamma)
        class_weight_t = self.class_weights[targets]
        return WeightedFocalComponents(per_sample_ce, pt, modulation, class_weight_t, class_weight_t * modulation * per_sample_ce)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.components(logits, targets).per_sample_focal.mean()


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _value(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path: value = value[key]
    return value


def validate_e12_fairness_contract(config: Mapping[str, Any]) -> None:
    if config.get("experiment") != "e12" or config.get("data_config") != "configs/data_config.yaml" or config.get("output_dir") != "artifacts/experiments/e12": raise ValueError("E12 identity/data/output contract drift.")
    expected = {("binary_specialist", "class_order"): list(BINARY_CLASS_ORDER), ("binary_specialist", "expected_source_counts"): EXPECTED_COUNTS, ("model", "architecture"): "flat_binary_specialist", ("model", "backbone"): "convnext_tiny", ("model", "weights"): "IMAGENET1K_V1", ("model", "pretrained"): False, ("model", "feature_dimension"): 768, ("model", "num_classes"): 2, ("model", "classification_head"): "Linear(768, 2)", ("model", "trainable_scope"): "full", ("loss", "name"): "InverseFrequencyBinaryFocalLoss", ("loss", "gamma"): GAMMA, ("loss", "alpha"): None, ("loss", "class_weights"): list(CLASS_WEIGHTS), ("loss", "class_weight_class_order"): list(BINARY_CLASS_ORDER), ("loss", "class_weight_counts"): [2153, 1147], ("loss", "label_smoothing"): 0.0, ("loss", "reduction"): "mean", ("optimizer", "name"): "AdamW", ("optimizer", "strategy"): "differential_lr", ("optimizer", "backbone_learning_rate"): 1e-5, ("optimizer", "head_learning_rate"): 1e-4, ("optimizer", "weight_decay"): 1e-4, ("scheduler", "name"): "CosineAnnealingLR", ("scheduler", "t_max"): 15, ("scheduler", "eta_min"): 1e-6, ("training", "seed"): 42, ("training", "epochs"): 15, ("training", "early_stopping_patience"): 5, ("training", "selection_metric"): "macro_f1", ("training", "device"): "cuda", ("training", "precision"): "fp32", ("training", "amp"): False, ("training", "gradient_clip_norm"): 1.0}
    mismatches=[]
    for path, expected_value in expected.items():
        try: actual = _value(config, path)
        except (KeyError, TypeError): actual = "__MISSING__"
        if actual != expected_value: mismatches.append({"path": ".".join(path), "expected": expected_value, "actual": actual})
    if config.get("loss", {}).get("class_weights") is None or config.get("loss", {}).get("alpha") is not None: mismatches.append({"path": "loss_weighting", "expected": "class_weights only, alpha None"})
    if any(key in config for key in {"resume_from", "checkpoint_path", "load_checkpoint", "initial_checkpoint"}): mismatches.append({"path": "checkpoint_loading", "expected": "absent"})
    if mismatches: raise ValueError(f"E12 fairness contract drift: {mismatches}")


def e9_e12_control_comparison(e9: Mapping[str, Any], e12: Mapping[str, Any]) -> dict[str, Any]:
    paths = [("data_config",), ("initialization",), ("binary_specialist",), ("model",), ("optimizer",), ("scheduler",), ("training",)]
    mismatches=[]
    for path in paths:
        if _value(e9, path) != _value(e12, path): mismatches.append(".".join(path))
    e9_loss, e12_loss = e9["loss"], e12["loss"]
    common_loss = (e9_loss["gamma"] == e12_loss["gamma"] and e9_loss["alpha"] is None and e12_loss["alpha"] is None and e9_loss["label_smoothing"] == e12_loss["label_smoothing"] and e9_loss["reduction"] == e12_loss["reduction"])
    if mismatches or not common_loss: raise ValueError(f"E9/E12 non-weighting parity drift: paths={mismatches}, common_loss={common_loss}")
    return {"controls_match": True, "only_difference": "E12 applies inverse-frequency class_weights to focal training contributions; E9 class_weights=None", "class_weights": list(CLASS_WEIGHTS)}


def load_e12_training_config(config_path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root=Path(project_root).expanduser().resolve(); path=Path(config_path).expanduser(); path=path.resolve() if path.is_absolute() else (root/path).resolve()
    if not path.is_relative_to(root) or not path.is_file(): raise FileNotFoundError(f"E12 config unavailable: {path}")
    config=_read_yaml(path); validate_e12_fairness_contract(config)
    data=(root/config["data_config"]).resolve(); output=(root/config["output_dir"]).resolve(); e9_ref=(root/config["e9_reference_config"]).resolve(); source=(root/config["initialization"]["source_checkpoint"]).resolve()
    if not all(item.is_relative_to(root) for item in (data,output,e9_ref,source)): raise ValueError("E12 paths must remain inside project root.")
    config["resolved"]={"project_root":str(root),"config_path":str(path),"data_config_path":str(data),"output_dir":str(output),"source_checkpoint_path":str(source),"e9_parity":e9_e12_control_comparison(_read_yaml(e9_ref),config)}
    return config


def create_e12_criterion(config: Mapping[str, Any]) -> InverseFrequencyBinaryFocalLoss:
    validate_e12_fairness_contract(config)
    return InverseFrequencyBinaryFocalLoss(gamma=float(config["loss"]["gamma"]), alpha=config["loss"]["alpha"], class_weights=tuple(float(x) for x in config["loss"]["class_weights"]), reduction=str(config["loss"]["reduction"]))


def _write_json(path: Path, payload: Any) -> None:
    temporary=path.with_suffix(path.suffix+".tmp"); temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8"); os.replace(temporary,path)


def fit_e12(training_config: Mapping[str, Any]) -> dict[str, Any]:
    validate_e12_fairness_contract(training_config); seed=int(training_config["training"]["seed"]); seed_everything(seed); root=Path(training_config["resolved"]["project_root"])
    data_config=load_data_config(training_config["resolved"]["data_config_path"],project_root=root)
    if data_config["views"]["train"]!="two" or data_config["views"]["validation"]!="single" or int(data_config["loader"]["batch_size"])!=16 or "test" in data_config.get("manifests",{}): raise ValueError("E12 requires E9 data contract.")
    train_loader,validation_loader=build_e8_dataloaders(data_config)
    if len(train_loader.dataset)!=3300 or len(validation_loader.dataset)!=550: raise ValueError("E12 fixed subset count drift.")
    device=resolve_device(str(training_config["training"]["device"]));
    if device.type!="cuda": raise RuntimeError("Full E12 training requires CUDA, matching E9.")
    model=build_e12_model(training_config); transfer=transfer_e6_encoder(model,checkpoint_path=training_config["resolved"]["source_checkpoint_path"],project_root=root); model=model.to(device)
    criterion=create_e12_criterion(training_config).to(device); optimizer=create_optimizer(model,training_config); scheduler=create_scheduler(optimizer,training_config); output=Path(training_config["resolved"]["output_dir"]); output.mkdir(parents=True,exist_ok=True); _write_json(output/"training_config.json",training_config)
    fingerprints={"train":file_sha256(data_config["resolved"]["train_manifest"]),"validation":file_sha256(data_config["resolved"]["validation_manifest"])}; history=[]; best,best_epoch,stale=float("-inf"),0,0
    for epoch in range(1,int(training_config["training"]["epochs"])+1):
        train=train_one_epoch_e8(model,train_loader,criterion=criterion,optimizer=optimizer,device=device,gradient_clip_norm=float(training_config["training"]["gradient_clip_norm"])); validation=evaluate_e8(model,validation_loader,device=device); metric=float(validation.metrics["macro_f1"]); improved=metric>best
        if improved: best,best_epoch,stale=metric,epoch,0
        else: stale+=1
        row={"epoch":epoch,"train_weighted_focal_loss":train.loss,"validation_unweighted_ce":validation.loss,"validation_accuracy":validation.metrics["accuracy"],"validation_macro_f1":metric,"validation_auroc":validation.metrics["auroc"],"is_best":improved,"stale_epochs":stale}; history.append(row); _write_json(output/"history.json",history); scheduler.step()
        payload={"model":model,"epoch":epoch,"metrics":row,"config":training_config,"optimizer":optimizer,"scheduler":scheduler,"best_validation_macro_f1":best,"random_seed":seed,"control_state":{"best_epoch":best_epoch,"stale_epochs":stale,"manifest_fingerprints":fingerprints,"initialization":"e6_best_encoder_transfer_fresh_binary_linear_768_2","e6_encoder_transfer":transfer,"binary_class_order":list(BINARY_CLASS_ORDER),"canonical_to_binary_labels":{"1":0,"2":1},"filtered_source_counts":EXPECTED_COUNTS,"loss":"inverse_frequency_weighted_binary_focal_gamma_2_alpha_none","class_weights":list(CLASS_WEIGHTS)},"history":history,"train_loader_generator":train_loader.generator,"save_rng_state":bool(training_config["training"]["checkpoint"]["save_rng_state"]),"run_kind":"e12_inverse_frequency_focal_e6_transfer_full_epoch"}
        save_checkpoint(output/training_config["training"]["checkpoint"]["last_filename"],**payload)
        if improved: save_checkpoint(output/training_config["training"]["checkpoint"]["best_filename"],**payload)
        if stale>=int(training_config["training"]["early_stopping_patience"]): break
    return {"best_validation_macro_f1":best,"best_epoch":best_epoch,"epochs_completed":len(history),"manifest_fingerprints":fingerprints,"e6_encoder_transfer":transfer,"output_dir":str(output)}
