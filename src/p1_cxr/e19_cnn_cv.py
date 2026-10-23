"""E19: fold-clean ConvNeXt-Tiny CNN-only B/V group-CV OOF experiment.

E19 deliberately has no E15/RAD-DINO/external-validation/test input. Every fold
creates an independently ImageNet-initialized CNN and trains only on fold-train.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from data.transforms import build_transform

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BINARY_NAMES = ("BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA")


def _inside(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"E19 path leaves project root: {value}")
    return resolved


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"E19 YAML must be a mapping: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _ordered_hash(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8")); digest.update(b"\0")
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def _torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def validate_contract(config: Mapping[str, Any]) -> None:
    required = {"version": 1, "experiment": "e19", "status": "cnn_only_bv_stratified_group_kfold_oof", "output_dir": "artifacts/experiments/e19_cnn_bv_cv", "data_config": "configs/data_config.yaml"}
    errors = [{"key": key, "expected": value, "actual": config.get(key)} for key, value in required.items() if config.get(key) != value]
    data = config.get("data_contract", {})
    expected_data = {"development_manifest": "artifacts/splits/train_manifest.csv", "source_samples": 3300, "source_class_counts": {"BACTERIAL_PNEUMONIA": 2153, "VIRAL_PNEUMONIA": 1147}, "canonical_to_binary": {1: 0, 2: 1}, "binary_class_order": list(BINARY_NAMES), "external_validation_access": "forbidden", "test_access": "forbidden", "e15_feature_or_logit_access": "forbidden", "group_key_policy": "proxy_group_plus_exact_sha256_duplicate_components_not_verified_patient_id"}
    errors.extend({"key": f"data_contract.{key}", "expected": value, "actual": data.get(key)} for key, value in expected_data.items() if data.get(key) != value)
    cv = config.get("cv", {})
    expected_cv = {"splitter": "StratifiedGroupKFold", "n_splits": 3, "seed": 42, "fold_model_initialization": "independent_imagenet1k_v1_per_fold", "fold_validation_usage": "final_evaluation_only_after_fixed_train_only_epochs", "oof_policy": "every_development_image_exactly_once_as_fold_validation"}
    errors.extend({"key": f"cv.{key}", "expected": value, "actual": cv.get(key)} for key, value in expected_cv.items() if cv.get(key) != value)
    model = config.get("model", {})
    expected_model = {"backbone": "convnext_tiny", "weights": "IMAGENET1K_V1", "pretrained": True, "allow_weight_download": False, "classifier": "Linear(768, 2)", "trainable_scope": "final_stage_and_classifier", "frozen_modules": ["features.0", "features.1", "features.2", "features.3", "features.4", "features.5"], "trainable_modules": ["features.6", "features.7", "feature_norm", "classifier"]}
    errors.extend({"key": f"model.{key}", "expected": value, "actual": model.get(key)} for key, value in expected_model.items() if model.get(key) != value)
    train = config.get("training", {})
    expected_train = {"seed": 42, "epochs": 10, "optimizer": "AdamW", "backbone_learning_rate": 0.00001, "classifier_learning_rate": 0.0001, "weight_decay": 0.0001, "scheduler": "CosineAnnealingLR", "eta_min": 0.000001, "loss": "CrossEntropyLoss", "amp": True, "source_batch_size": 8, "effective_train_forward_batch_size": 16, "epoch_policy": "fixed_train_only_epochs_no_outer_fold_selection", "gradient_clip_norm": 1.0, "max_consecutive_skipped_steps": 8, "early_stopping": False, "dynamic_oom_batch_fallback": "forbidden_restart_fold_with_explicit_config_change"}
    errors.extend({"key": f"training.{key}", "expected": value, "actual": train.get(key)} for key, value in expected_train.items() if train.get(key) != value)
    if errors: raise ValueError(f"E19 configuration contract drift: {errors}")


def load_config(path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve(); config_path = _inside(path, root)
    config = _yaml(config_path); validate_contract(config)
    data_config = _yaml(_inside(config["data_config"], root))
    if set(data_config.get("manifests", {})) != {"train", "validation"} or "test" in json.dumps(data_config).lower():
        raise ValueError("E19 requires the shared train/validation-only data config and forbids test access.")
    config["resolved"] = {"project_root": str(root), "config_path": str(config_path), "development_manifest": str(_inside(config["data_contract"]["development_manifest"], root)), "data_config": str(_inside(config["data_config"], root)), "output_dir": str(_inside(config["output_dir"], root))}
    return config


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _device(request: str) -> torch.device:
    if request == "cuda":
        if not torch.cuda.is_available(): raise RuntimeError("E19 CUDA was requested but is unavailable.")
        return torch.device("cuda")
    if request == "cpu": return torch.device("cpu")
    if request == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unsupported E19 device: {request}")


class BVDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, frame: pd.DataFrame, root: Path, transform: Any, *, views: int) -> None:
        self.frame, self.root, self.transform, self.views = frame.reset_index(drop=True).copy(), root, transform, views
        self.paths = []
        for value in self.frame["image_path"]:
            path = (root / str(value)).resolve()
            if not path.is_relative_to(root) or not path.is_file(): raise FileNotFoundError(f"E19 image unavailable: {path}")
            self.paths.append(path)

    def __len__(self) -> int: return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        try:
            with Image.open(self.paths[index]) as image:
                image.load(); source = image.copy()
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise RuntimeError(f"E19 cannot load image index={index}: {self.paths[index]}") from error
        values = [self.transform(source.copy()) for _ in range(self.views)]
        image_tensor = values[0] if self.views == 1 else torch.stack(values)
        return image_tensor, torch.tensor(int(self.frame.iloc[index]["binary_label"]), dtype=torch.long)


class ConvNeXtTinyBV(nn.Module):
    def __init__(self, *, pretrained: bool, allow_weight_download: bool) -> None:
        super().__init__()
        if pretrained:
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            filename = weights.url.rsplit("/", 1)[-1]
            cache = Path(torch.hub.get_dir()) / "checkpoints" / filename
            if not cache.is_file() and not allow_weight_download: raise FileNotFoundError(f"E19 requires cached ConvNeXt IMAGENET1K_V1 weights: {cache}")
        else: weights = None
        base = convnext_tiny(weights=weights)
        self.features, self.avgpool, self.feature_norm = base.features, base.avgpool, base.classifier[0]
        self.classifier = nn.Linear(int(base.classifier[-1].in_features), 2)
        self.set_final_stage_trainable()

    def set_final_stage_trainable(self) -> None:
        for parameter in self.parameters(): parameter.requires_grad_(False)
        for module in (self.features[6], self.features[7], self.feature_norm, self.classifier):
            for parameter in module.parameters(): parameter.requires_grad_(True)

    def enforce_frozen_modes(self) -> None:
        for index in range(6): self.features[index].eval()
        self.features[6].train(); self.features[7].train(); self.feature_norm.train(); self.classifier.train()

    def forward(self, images: Tensor) -> Tensor:
        feature_map = self.features(images); pooled = self.avgpool(feature_map); normalized = self.feature_norm(pooled)
        return self.classifier(torch.flatten(normalized, 1))

    def parameter_counts(self) -> dict[str, int]:
        return {"total": int(sum(p.numel() for p in self.parameters())), "trainable": int(sum(p.numel() for p in self.parameters() if p.requires_grad)), "frozen": int(sum(p.numel() for p in self.parameters() if not p.requires_grad))}


def _development(config: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(Path(config["resolved"]["development_manifest"]))
    required = {"image_path", "label", "class_name", "source_split", "group_key", "sha256", "source_identifier"}
    if required - set(frame.columns) or frame["image_path"].duplicated().any() or frame["sha256"].duplicated().any(): raise ValueError("E19 development manifest identity/duplicate contract failed.")
    frame = frame[frame["class_name"].isin(BINARY_NAMES)].copy().reset_index(drop=True)
    expected = config["data_contract"]
    if len(frame) != int(expected["source_samples"]) or frame["class_name"].value_counts().to_dict() != expected["source_class_counts"]: raise ValueError("E19 B/V development count contract failed.")
    if (frame["label"].astype(int) - 1).isin([0, 1]).all() is False: raise ValueError("E19 canonical-to-binary mapping failed.")
    frame.insert(0, "sample_id", frame["image_path"].astype(str)); frame["binary_label"] = frame["label"].astype(int) - 1
    return frame


def _folds(config: Mapping[str, Any], frame: pd.DataFrame) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    cv = config["cv"]; splitter = StratifiedGroupKFold(n_splits=int(cv["n_splits"]), shuffle=True, random_state=int(cv["seed"]))
    folds=[]; oof=[]
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(frame)), frame["binary_label"], frame["group_key"])):
        train, valid = frame.iloc[train_idx].copy().reset_index(drop=True), frame.iloc[valid_idx].copy().reset_index(drop=True)
        checks = [set(train["image_path"]) & set(valid["image_path"]), set(train["sha256"]) & set(valid["sha256"]), set(train["group_key"]) & set(valid["group_key"])]
        if any(checks) or set(train["binary_label"]) != {0, 1} or set(valid["binary_label"]) != {0, 1}: raise RuntimeError(f"E19 fold {fold} leakage/class feasibility contract failed.")
        folds.append((fold, train, valid)); oof.extend(valid["image_path"].tolist())
    if sorted(oof) != sorted(frame["image_path"].tolist()) or len(oof) != len(set(oof)): raise RuntimeError("E19 OOF partition is not exactly once per development image.")
    return folds


def _loader(frame: pd.DataFrame, config: Mapping[str, Any], *, training: bool, fold_seed: int) -> DataLoader:
    data = _yaml(Path(config["resolved"]["data_config"]))
    transform = build_transform(data["transform"], training=training)
    dataset = BVDataset(frame, Path(config["resolved"]["project_root"]), transform, views=2 if training else 1)
    return DataLoader(dataset, batch_size=int(config["training"]["source_batch_size"]), shuffle=training, num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False, generator=torch.Generator().manual_seed(fold_seed))


def _binary_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = probabilities.argmax(1); matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(targets, predictions, labels=[0, 1], zero_division=0)
    return {"samples": int(len(targets)), "accuracy": float(accuracy_score(targets, predictions)), "macro_f1": float(f1_score(targets, predictions, labels=[0, 1], average="macro", zero_division=0)), "confusion_matrix": matrix.tolist(), "bacterial_recall": float(recall[0]), "viral_recall": float(recall[1]), "bacterial_to_viral": int(matrix[0, 1]), "viral_to_bacterial": int(matrix[1, 0]), "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, name in enumerate(BINARY_NAMES)}}


def _amp(device: torch.device, enabled: bool): return torch.autocast("cuda", dtype=torch.float16) if enabled and device.type == "cuda" else contextlib.nullcontext()
def _oom(error: BaseException) -> bool: return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def _train_epoch(model: ConvNeXtTinyBV, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: Any, device: torch.device, config: Mapping[str, Any]) -> dict[str, Any]:
    model.train(); model.enforce_frozen_modes(); criterion=nn.CrossEntropyLoss(); total=0.; seen=0; amp=bool(config["training"]["amp"])
    scale_start=float(scaler.get_scale()); scale_min=scale_start; scale_max=scale_start; scale_change_events=0; optimizer_updates=0; skipped_steps=0; nonfinite_grad_after_unscale_batches=0; consecutive_skips=0; max_consecutive_skips=0
    for batch_index, (views, labels) in enumerate(loader):
        try:
            images=views.reshape(-1,*views.shape[2:]).to(device, non_blocking=device.type=="cuda"); expanded=labels[:,None].expand(-1,views.shape[1]).reshape(-1).to(device)
            optimizer.zero_grad(set_to_none=True)
            with _amp(device,amp): logits=model(images); loss=criterion(logits,expanded)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"E19 non-finite training loss at batch={batch_index}; no optimizer or scheduler update was executed.")
            scale_before=float(scaler.get_scale())
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            gradients_finite=bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
            if gradients_finite:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(config["training"]["gradient_clip_norm"]), error_if_nonfinite=True)
                scaler.step(optimizer); optimizer_updates+=1; consecutive_skips=0
            else:
                nonfinite_grad_after_unscale_batches+=1; skipped_steps+=1; consecutive_skips+=1; max_consecutive_skips=max(max_consecutive_skips,consecutive_skips)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                else:
                    optimizer.zero_grad(set_to_none=True)
            scaler.update()
            scale_after=float(scaler.get_scale()); scale_min=min(scale_min,scale_after); scale_max=max(scale_max,scale_after)
            if scale_after != scale_before: scale_change_events+=1
            if not gradients_finite and consecutive_skips > int(config["training"]["max_consecutive_skipped_steps"]):
                raise FloatingPointError(f"E19 repeated non-finite gradients after GradScaler unscale at batch={batch_index}; consecutive_skips={consecutive_skips}, limit={config['training']['max_consecutive_skipped_steps']}, loss={float(loss.detach().cpu())}, scale_before={scale_before}, scale_after={scale_after}. Optimizer updates were skipped for those batches.")
            if gradients_finite:
                total += float(loss.detach().cpu())*len(expanded); seen += len(expanded)
        except RuntimeError as error:
            if _oom(error):
                if device.type=="cuda": torch.cuda.empty_cache()
                raise RuntimeError(f"E19 CUDA OOM; source_batch_size={config['training']['source_batch_size']} means two-view forward batch={config['training']['effective_train_forward_batch_size']}. Do not dynamically alter a fold; change config and restart all folds.") from error
            raise
    if seen==0 or optimizer_updates==0: raise RuntimeError("E19 train epoch completed without a successful optimizer update.")
    return {"train_loss":total/seen,"finite_loss_samples":seen,"optimizer_updates":optimizer_updates,"skipped_optimizer_steps":skipped_steps,"nonfinite_grad_after_unscale_batches":nonfinite_grad_after_unscale_batches,"max_consecutive_skipped_steps":max_consecutive_skips,"gradscaler_scale_start":scale_start,"gradscaler_scale_end":float(scaler.get_scale()),"gradscaler_scale_min":scale_min,"gradscaler_scale_max":scale_max,"gradscaler_scale_change_events":scale_change_events}


@torch.inference_mode()
def _evaluate(model: ConvNeXtTinyBV, loader: DataLoader, device: torch.device, config: Mapping[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval(); probs=[]; targets=[]; amp=bool(config["training"]["amp"])
    for images, labels in loader:
        try:
            images=images.to(device, non_blocking=device.type=="cuda")
            with _amp(device,amp): logits=model(images)
            if logits.shape!=(len(labels),2) or not bool(torch.isfinite(logits).all()): raise FloatingPointError("E19 invalid validation logits.")
            probs.append(torch.softmax(logits.float(),1).cpu().numpy()); targets.append(labels.numpy())
        except RuntimeError as error:
            if _oom(error):
                if device.type=="cuda": torch.cuda.empty_cache()
                raise RuntimeError("E19 CUDA OOM during fold validation; reduce source_batch_size in config and restart all folds.") from error
            raise
    probability=np.concatenate(probs); target=np.concatenate(targets); return _binary_metrics(target, probability), target, probability


def _paths(config: Mapping[str, Any], fold: int) -> dict[str, Path]:
    root=Path(config["resolved"]["output_dir"]); prefix=config["artifacts"]["fold_dir_prefix"]; directory=root/f"{prefix}{fold:02d}"; a=config["artifacts"]
    return {"directory":directory,"checkpoint":directory/a["fold_checkpoint"],"history":directory/a["fold_history"],"metrics":directory/a["fold_metrics"],"predictions":directory/a["fold_predictions"]}


def _run_fold(config: Mapping[str, Any], fold: int, train: pd.DataFrame, valid: pd.DataFrame, device: torch.device, *, write: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    fold_seed=int(config["training"]["seed"])+fold; _seed(fold_seed)
    model=ConvNeXtTinyBV(pretrained=True,allow_weight_download=bool(config["model"]["allow_weight_download"])).to(device)
    train_loader=_loader(train,config,training=True,fold_seed=fold_seed); valid_loader=_loader(valid,config,training=False,fold_seed=fold_seed)
    backbone=[p for p in list(model.features[6].parameters())+list(model.features[7].parameters())+list(model.feature_norm.parameters()) if p.requires_grad]; head=[p for p in model.classifier.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW([{"params":backbone,"lr":float(config["training"]["backbone_learning_rate"])},{"params":head,"lr":float(config["training"]["classifier_learning_rate"])}],weight_decay=float(config["training"]["weight_decay"]))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=int(config["training"]["epochs"]),eta_min=float(config["training"]["eta_min"])); scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and bool(config["training"]["amp"]))
    final_state=None; history=[]
    for epoch in range(1,int(config["training"]["epochs"])+1):
        train_stats=_train_epoch(model,train_loader,optimizer,scaler,device,config)
        scheduler_advanced=train_stats["optimizer_updates"] > 0
        lr_before=[group["lr"] for group in optimizer.param_groups]
        if scheduler_advanced: scheduler.step()
        history.append({"epoch":epoch,**train_stats,"learning_rates_before_scheduler":lr_before,"learning_rates_after_scheduler":[group["lr"] for group in optimizer.param_groups],"scheduler_advanced":scheduler_advanced,"outer_fold_validation_evaluated":False,"epoch_selection":"fixed_train_only"})
    final_epoch=int(config["training"]["epochs"])
    final_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    model.load_state_dict(final_state,strict=True)
    metrics,targets,probs=_evaluate(model,valid_loader,device,config); predictions=probs.argmax(1)
    prediction=valid[["sample_id","image_path","group_key","sha256","source_identifier","class_name","label","binary_label"]].copy(); prediction.insert(0,"fold",fold); prediction["cnn_binary_pred"]=predictions; prediction["cnn_prob_bacterial"],prediction["cnn_prob_viral"]=probs[:,0],probs[:,1]
    info={"fold":fold,"seed":fold_seed,"final_epoch":final_epoch,"epoch_policy":"fixed_train_only_epochs_no_outer_fold_selection","outer_fold_validation_evaluations":1,"metrics":metrics,"train_samples":len(train),"validation_samples":len(valid),"group_overlap":0,"sha256_overlap":0,"model_parameter_counts":model.parameter_counts(),"initialization":"fresh_imagenet1k_v1_no_project_checkpoint"}
    if write:
        paths=_paths(config,fold); paths["directory"].mkdir(parents=True,exist_ok=False); _torch(paths["checkpoint"],{"experiment":"e19","fold":fold,"epoch":final_epoch,"checkpoint_policy":"fixed_final_epoch_no_outer_fold_selection","model_state_dict":final_state,"config":config,"fold_seed":fold_seed,"fold_train_image_hash":_ordered_hash(train["image_path"]),"fold_validation_image_hash":_ordered_hash(valid["image_path"]),"run_kind":"e19_fold_clean_cnn_only_fixed_final_epoch"}); _json(paths["history"],history); _json(paths["metrics"],info); _csv(paths["predictions"],prediction)
    del model
    if device.type=="cuda": torch.cuda.empty_cache()
    return prediction,info


def static_smoke_check(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_contract(config); development=_development(config); folds=_folds(config,development)
    model=ConvNeXtTinyBV(pretrained=False,allow_weight_download=False).eval(); output=model(torch.zeros((2,3,224,224)))
    if output.shape!=(2,2) or any(p.requires_grad for module in [model.features[i] for i in range(6)] for p in module.parameters()) or not any(p.requires_grad for p in model.features[7].parameters()): raise RuntimeError("E19 model scope/shape static contract failed.")
    return {"experiment":"e19","static_check":"passed","development_bv_samples":len(development),"development_groups":int(development.group_key.nunique()),"folds":[{"fold":f,"train":len(t),"validation":len(v),"group_overlap":0,"sha256_overlap":0,"validation_class_counts":v.class_name.value_counts().to_dict()} for f,t,v in folds],"model_logits_shape":list(output.shape),"model_parameter_counts":model.parameter_counts(),"external_validation_access":False,"e15_access":False,"test_access":False,"artifacts_written":False}


def gpu_smoke_check(config: Mapping[str, Any], *, device_request: str | None=None) -> dict[str, Any]:
    static=static_smoke_check(config); development=_development(config); fold,train,_=_folds(config,development)[0]; device=_device(device_request or "cuda"); sample=pd.concat([train[train.binary_label==label].head(1) for label in (0,1)],ignore_index=True); _seed(int(config["training"]["seed"])+fold); model=ConvNeXtTinyBV(pretrained=True,allow_weight_download=bool(config["model"]["allow_weight_download"])).to(device); loader=_loader(sample,config,training=True,fold_seed=int(config["training"]["seed"])+fold); views,labels=next(iter(loader)); images=views.reshape(-1,*views.shape[2:]).to(device); expanded=labels[:,None].expand(-1,2).reshape(-1).to(device); optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=float(config["training"]["classifier_learning_rate"])); scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and bool(config["training"]["amp"])); model.train(); model.enforce_frozen_modes(); optimizer.zero_grad(set_to_none=True)
    try:
        with _amp(device,bool(config["training"]["amp"])): logits=model(images); loss=nn.CrossEntropyLoss()(logits,expanded)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); frozen_grad=any(p.grad is not None for index in range(6) for p in model.features[index].parameters()); trainable_grad=any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad)
        if logits.shape!=(4,2) or frozen_grad or not trainable_grad: raise RuntimeError("E19 GPU smoke gradient/scope contract failed.")
        scale_before=float(scaler.get_scale()); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(config["training"]["gradient_clip_norm"]),error_if_nonfinite=True); scaler.step(optimizer); scaler.update(); scale_after=float(scaler.get_scale())
        return {**static,"gpu_smoke":"passed","device":str(device),"loss":float(loss.detach().cpu()),"frozen_early_stage_gradient":False,"final_stage_or_classifier_gradient":True,"optimizer_update_executed":True,"gradscaler_scale_before":scale_before,"gradscaler_scale_after":scale_after,"artifacts_written":False}
    finally:
        del model
        if device.type=="cuda": torch.cuda.empty_cache()


def run_e19(config: Mapping[str, Any]) -> dict[str, Any]:
    static_smoke_check(config); output=Path(config["resolved"]["output_dir"])
    if output.exists(): raise FileExistsError(f"E19 output exists; automatic overwrite is forbidden: {output}")
    development=_development(config); folds=_folds(config,development); device=_device("cuda"); output.mkdir(parents=True,exist_ok=False); predictions=[]; reports=[]
    try:
        _json(output/config["artifacts"]["cv_manifest"], {"experiment":"e19", "development_manifest":config["resolved"]["development_manifest"], "development_manifest_sha256":_sha(Path(config["resolved"]["development_manifest"])), "development_image_hash":_ordered_hash(development["image_path"]), "development_group_key_note":"proxy group plus exact duplicate components; not verified patient identity", "outer_fold_validation_policy":"final_evaluation_only_after_fixed_train_only_epochs", "fixed_epochs":int(config["training"]["epochs"]), "folds":[{"fold":f,"train_image_hash":_ordered_hash(t["image_path"]),"validation_image_hash":_ordered_hash(v["image_path"]),"train_groups":int(t.group_key.nunique()),"validation_groups":int(v.group_key.nunique())} for f,t,v in folds]})
        for fold,train,valid in folds:
            pred,info=_run_fold(config,fold,train,valid,device,write=True); predictions.append(pred); reports.append(info)
        oof=pd.concat(predictions,ignore_index=True).sort_values("image_path").reset_index(drop=True)
        if oof.image_path.duplicated().any() or sorted(oof.image_path.tolist())!=sorted(development.image_path.tolist()): raise RuntimeError("E19 OOF identity contract failed.")
        oof_metrics=_binary_metrics(oof.binary_label.to_numpy(int),oof[["cnn_prob_bacterial","cnn_prob_viral"]].to_numpy(float)); _csv(output/config["artifacts"]["oof_predictions"],oof); _json(output/config["artifacts"]["oof_metrics"],{"experiment":"e19","oof":oof_metrics,"fold_metrics":reports,"fold_macro_f1_mean":float(np.mean([report["metrics"]["macro_f1"] for report in reports])),"fold_macro_f1_std":float(np.std([report["metrics"]["macro_f1"] for report in reports]))}); _json(output/config["artifacts"]["run_metadata"],{"experiment":"e19","external_validation_used":False,"test_used":False,"e15_used":False,"future_fusion_interface":config["future_fold_clean_fusion_interface"],"oom_policy":config["training"]["dynamic_oom_batch_fallback"]})
        return {"output_dir":str(output),"oof_macro_f1":oof_metrics["macro_f1"],"oof_accuracy":oof_metrics["accuracy"]}
    except Exception:
        raise
