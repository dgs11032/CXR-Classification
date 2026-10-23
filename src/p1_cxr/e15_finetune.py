"""E15: three fixed epochs on full B/V train with RAD-DINO final-block tuning.

The E14 head is initialization only. Because it previously saw all B/V train images,
E15 deliberately has no internal holdout, early stopping, or model selection. The
third epoch is always the final checkpoint; external validation is evaluated once.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .e14_patch_specialist import (
    BINARY_CLASSES, CANONICAL_CLASSES, PatchAttentionHead, RawManifestDataset,
    _binary_metrics, _binary_train_frame, _collate, _load_e10_trace, _manifest,
    _pool_patch_tokens,
)
from .metrics import classification_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _inside_project(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser(); resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root): raise ValueError(f"E15 path leaves project root: {value}")
    return resolved


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"E15 YAML must be a mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"); os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False); os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary); os.replace(temporary, path)


def validate_e15_contract(config: Mapping[str, Any]) -> None:
    required = {"experiment":"e15", "status":"rad_dino_last_transformer_block_finetune_e14_patch_head", "data_config":"configs/data_config.yaml", "output_dir":"artifacts/experiments/e15"}
    errors = [{"key":key,"expected":value,"actual":config.get(key)} for key,value in required.items() if config.get(key)!=value]
    expected_model = {"hf_model_id":"microsoft/rad-dino", "trust_remote_code":False, "required_transformers_version":"4.57.6", "expected_hidden_size":768, "expected_num_hidden_layers":12, "patch_size":14, "expected_image_size":518, "spatial_pool_grid":[4,4], "trainable_encoder_scope":"last_transformer_block_only", "gradient_checkpointing":"disabled_after_runtime_capability_inspection"}
    errors.extend({"key":f"model.{k}","expected":v,"actual":config.get("model",{}).get(k)} for k,v in expected_model.items() if config.get("model",{}).get(k)!=v)
    head=config.get("head",{}); runtime=config.get("runtime",{}); training=config.get("training",{}); split=config.get("split_contract",{})
    if head.get("architecture")!="cls_plus_4x4_patch_attention_pooling_binary_head" or int(head.get("num_classes",-1))!=2: errors.append({"key":"head","actual":head})
    if int(runtime.get("batch_size",-1))!=1 or int(runtime.get("gradient_accumulation_steps",-1))!=8 or bool(runtime.get("amp")) is not True: errors.append({"key":"runtime","actual":runtime})
    expected_training={"seed":42,"fixed_epochs":3,"optimizer":"AdamW","encoder_learning_rate":0.00001,"head_learning_rate":0.0001,"weight_decay":0.0001,"checkpoint_policy":"final_epoch_only","checkpoint_selection_data":"none","validation_weight_selection":False,"validation_threshold_selection":False,"internal_holdout":"disabled","early_stopping":False}
    errors.extend({"key":f"training.{k}","expected":v,"actual":training.get(k)} for k,v in expected_training.items() if training.get(k)!=v)
    if "max_epochs" in training or "early_stopping_patience" in training or "selection_metric" in training: errors.append({"key":"training","expected":"no holdout/early-stopping/selection fields","actual":training})
    if split.get("binary_class_order")!=list(BINARY_CLASSES) or split.get("canonical_to_binary")!={1:0,2:1} or bool(split.get("no_test_access")) is not True: errors.append({"key":"split_contract","actual":split})
    if config.get("e10_routing_input",{}).get("normal_rule")!="e6_pred_zero_kept_else_e15_binary_plus_one": errors.append({"key":"e10_routing_input.normal_rule","actual":config.get("e10_routing_input",{}).get("normal_rule")})
    if errors: raise ValueError(f"E15 configuration contract drift: {errors}")


def load_e15_config(path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root=Path(project_root).expanduser().resolve(); config_path=_inside_project(path,root)
    if not config_path.is_file(): raise FileNotFoundError(f"E15 config unavailable: {config_path}")
    config=_read_yaml(config_path); validate_e15_contract(config)
    data=_read_yaml(_inside_project(config["data_config"],root)); manifests=data.get("manifests",{})
    if set(manifests)!={"train","validation"}: raise ValueError("E15 requires shared train/validation-only data config.")
    split,e14,e10=config["split_contract"],config["e14_initialization"],config["e10_routing_input"]
    if _inside_project(manifests["train"],root)!=_inside_project(split["train_manifest"],root) or _inside_project(manifests["validation"],root)!=_inside_project(split["validation_manifest"],root): raise ValueError("E15 must exactly reuse shared manifests.")
    config["resolved"]={"project_root":str(root),"config_path":str(config_path),"train_manifest":str(_inside_project(split["train_manifest"],root)),"validation_manifest":str(_inside_project(split["validation_manifest"],root)),"e14_checkpoint":str(_inside_project(e14["checkpoint"],root)),"e14_history":str(_inside_project(e14["history"],root)),"e14_metrics":str(_inside_project(e14["metrics"],root)),"e14_predictions":str(_inside_project(e14["predictions"],root)),"e10_predictions":str(_inside_project(e10["predictions"],root)),"e10_metrics":str(_inside_project(e10["metrics"],root)),"output_dir":str(_inside_project(config["output_dir"],root))}
    return config


def _frames(config: Mapping[str, Any]) -> tuple[pd.DataFrame,pd.DataFrame]:
    root=Path(config["resolved"]["project_root"]); split=config["split_contract"]
    train_full=_manifest(Path(config["resolved"]["train_manifest"]),root,"train",{"total":4448,"NORMAL":1148,"BACTERIAL_PNEUMONIA":2153,"VIRAL_PNEUMONIA":1147})
    return _binary_train_frame(train_full,config), _manifest(Path(config["resolved"]["validation_manifest"]),root,"validation",split["expected_counts"]["validation"])


def static_smoke_check(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_e15_contract(config); train_bv,validation=_frames(config)
    overlap=set(train_bv["group_key"]) & set(validation["group_key"])
    if overlap: raise RuntimeError(f"E15 train/validation group contamination: {len(overlap)}")
    head=PatchAttentionHead(config).eval()
    with torch.inference_mode(): logits,attention=head(torch.zeros((1,17,768)))
    if logits.shape!=(1,2) or attention.shape!=(1,16) or not torch.allclose(attention.sum(1),torch.ones(1),atol=1e-6): raise RuntimeError("E15 E14-head compatibility contract failed.")
    prerequisites={key:Path(config["resolved"][key]).is_file() for key in ("e14_checkpoint","e14_history","e14_metrics","e14_predictions","e10_predictions","e10_metrics")}
    return {"experiment":"e15","static_check":"passed","policy":"full_3300_bv_train_fixed_3_epochs_final_epoch_only","train_bv_samples":len(train_bv),"train_bv_groups":int(train_bv["group_key"].nunique()),"validation_samples":len(validation),"outer_train_validation_group_overlap":0,"internal_holdout_used":False,"early_stopping_used":False,"head_parameter_tensors":sum(1 for _ in head.parameters()),"head_parameters":sum(p.numel() for p in head.parameters()),"e14_e10_prerequisites":prerequisites,"train_manifest_sha256":_sha256(Path(config["resolved"]["train_manifest"])),"validation_manifest_sha256":_sha256(Path(config["resolved"]["validation_manifest"])),"output_dir_exists":Path(config["resolved"]["output_dir"]).exists()}


def _device(request: str) -> torch.device:
    if request=="cuda":
        if not torch.cuda.is_available(): raise RuntimeError("E15 requires CUDA by configuration but CUDA is unavailable.")
        return torch.device("cuda")
    if request=="auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if request=="cpu": return torch.device("cpu")
    raise ValueError(f"Unsupported E15 device: {request}")


def _load_encoder_and_last_block(config: Mapping[str, Any], device: torch.device) -> tuple[Any,nn.Module,nn.Module,dict[str,Any]]:
    try:
        import transformers
        from transformers import AutoImageProcessor,AutoModel
    except ImportError as error: raise RuntimeError("E15 requires existing transformers==4.57.6; it is not installed.") from error
    if transformers.__version__!=str(config["model"]["required_transformers_version"]): raise RuntimeError(f"E15 requires transformers=={config['model']['required_transformers_version']}; found {transformers.__version__}. Do not downgrade automatically.")
    model_config=config["model"]; processor=AutoImageProcessor.from_pretrained(model_config["hf_model_id"],revision=model_config["revision"],trust_remote_code=False); encoder=AutoModel.from_pretrained(model_config["hf_model_id"],revision=model_config["revision"],trust_remote_code=False).to(device)
    if int(getattr(encoder.config,"hidden_size",-1))!=768 or int(getattr(encoder.config,"num_hidden_layers",-1))!=12: raise RuntimeError("Loaded RAD-DINO architecture violates E15 contract.")
    candidates=[(name,module) for name,module in encoder.named_modules() if isinstance(module,nn.ModuleList) and len(module)==12]
    if len(candidates)!=1: raise RuntimeError(f"Unable to uniquely identify 12 RAD-DINO transformer blocks: {[name for name,_ in candidates]}")
    block_path,blocks=candidates[0]
    for parameter in encoder.parameters(): parameter.requires_grad_(False)
    last_block=blocks[-1]
    for parameter in last_block.parameters(): parameter.requires_grad_(True)
    if any(parameter.requires_grad for block in blocks[:-1] for parameter in block.parameters()) or not any(parameter.requires_grad for parameter in last_block.parameters()): raise RuntimeError("E15 last-block-only freeze contract failed.")
    supports_gc=bool(getattr(encoder,"supports_gradient_checkpointing",False)) and callable(getattr(encoder,"gradient_checkpointing_enable",None))
    return processor,encoder,last_block,{"model_id":model_config["hf_model_id"],"requested_revision":model_config["revision"],"resolved_commit_hash":getattr(encoder.config,"_commit_hash",None),"transformers_version":transformers.__version__,"model_class":type(encoder).__name__,"transformer_block_module_path":block_path,"transformer_block_count":len(blocks),"gradient_checkpointing_supported":supports_gc,"gradient_checkpointing_enabled":False}


def _load_e14_head(config: Mapping[str, Any], train_hash: str, validation_hash: str, device: torch.device) -> tuple[PatchAttentionHead,dict[str,Any]]:
    paths={key:Path(config["resolved"][key]) for key in ("e14_checkpoint","e14_history","e14_metrics","e14_predictions")}; missing=[str(path) for path in paths.values() if not path.is_file()]
    if missing: raise FileNotFoundError("E15 requires completed E14 checkpoint/history/metrics/predictions: "+", ".join(missing))
    payload=torch.load(paths["e14_checkpoint"],map_location="cpu",weights_only=True)
    if payload.get("experiment")!="e14" or payload.get("head_config")!=config["head"] or payload.get("train_manifest_sha256")!=train_hash or payload.get("validation_manifest_sha256")!=validation_hash: raise ValueError("E14 checkpoint provenance/config/manifest mismatch.")
    history=json.loads(paths["e14_history"].read_text(encoding="utf-8")); metrics=json.loads(paths["e14_metrics"].read_text(encoding="utf-8"))
    if not isinstance(history,list) or len(history)!=int(config["e14_initialization"]["required_history_epochs"]) or not all(np.isfinite(float(row.get("train_loss",np.nan))) for row in history): raise ValueError("Cannot verify complete finite E14 history.")
    if "e14_binary_specialist_on_true_bv_validation" not in metrics: raise ValueError("E14 metrics lack B/V specialist result.")
    head=PatchAttentionHead(config).to(device); head.load_state_dict(payload["head_state_dict"],strict=True)
    return head,{"checkpoint":str(paths["e14_checkpoint"]),"history_epochs":len(history),"final_train_loss":float(history[-1]["train_loss"]),"e14_binary_metrics":metrics["e14_binary_specialist_on_true_bv_validation"]}


def _forward(images: list[Any],processor: Any,encoder: nn.Module,head: PatchAttentionHead,config: Mapping[str,Any],device: torch.device,*,training: bool)->tuple[Tensor,Tensor]:
    inputs={key:value.to(device,non_blocking=device.type=="cuda") for key,value in processor(images=images,return_tensors="pt").items()}; autocast=torch.autocast("cuda",dtype=torch.float16) if bool(config["runtime"]["amp"]) and device.type=="cuda" else contextlib.nullcontext(); context=contextlib.nullcontext() if training else torch.inference_mode()
    with context,autocast:
        tokens,_=_pool_patch_tokens(encoder(**inputs).last_hidden_state,config); return head(tokens)


def _loader(frame: pd.DataFrame,root: Path,*,shuffle: bool,config: Mapping[str,Any])->DataLoader:
    return DataLoader(RawManifestDataset(frame,root,"binary_label"),batch_size=int(config["runtime"]["batch_size"]),shuffle=shuffle,num_workers=int(config["runtime"]["num_workers"]),pin_memory=bool(config["runtime"]["pin_memory"]),collate_fn=_collate)


def _count_parameters(module: nn.Module)->dict[str,int]:
    values=list(module.parameters()); return {"parameter_tensors":len(values),"parameters":int(sum(p.numel() for p in values)),"trainable_parameter_tensors":sum(p.requires_grad for p in values),"trainable_parameters":int(sum(p.numel() for p in values if p.requires_grad))}


def _seed(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _checkpoint_payload(config: Mapping[str,Any],epoch:int,encoder:nn.Module,last_block:nn.Module,head:PatchAttentionHead,optimizer:torch.optim.Optimizer,counts:Mapping[str,Any],e14_info:Mapping[str,Any])->dict[str,Any]:
    return {"experiment":"e15","run_kind":"e15_last_block_finetune_fixed_final_epoch","epoch":epoch,"checkpoint_policy":"final_epoch_only","last_block_state_dict":last_block.state_dict(),"head_state_dict":head.state_dict(),"optimizer":optimizer.state_dict(),"config":config,"parameter_counts":dict(counts),"e14_initialization":dict(e14_info),"encoder_model_class":type(encoder).__name__}


def _prepare_output(config:Mapping[str,Any],overwrite:bool)->dict[str,Path]:
    output,names=Path(config["resolved"]["output_dir"]),config["artifacts"]; paths={"output":output,"final":output/names["final_checkpoint"],"history":output/names["history"],"metrics":output/names["metrics"],"predictions":output/names["predictions"],"metadata":output/names["run_metadata"]}; existing=[path for key,path in paths.items() if key!="output" and path.exists()]
    if existing and not overwrite: raise FileExistsError("E15 output exists; use --overwrite-results only for E15 files: "+", ".join(map(str,existing)))
    output.mkdir(parents=True,exist_ok=True); return paths


def gpu_backward_smoke(config:Mapping[str,Any],*,device_request:str|None=None)->dict[str,Any]:
    static=static_smoke_check(config); train_bv,_=_frames(config); device=_device(device_request or str(config["runtime"]["device"])); train_hash,val_hash=_sha256(Path(config["resolved"]["train_manifest"])),_sha256(Path(config["resolved"]["validation_manifest"]))
    head,e14_info=_load_e14_head(config,train_hash,val_hash,device); processor,encoder,last_block,model_info=_load_encoder_and_last_block(config,device); sample=train_bv.iloc[:2].reset_index(drop=True); images,labels,_=next(iter(_loader(sample,Path(config["resolved"]["project_root"]),shuffle=False,config=config))); encoder.train(); head.train(); logits,attention=_forward(images,processor,encoder,head,config,device,training=True); loss=nn.CrossEntropyLoss()(logits,labels.to(device)); loss.backward()
    last_grad=any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in last_block.parameters() if p.requires_grad); head_grad=any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters() if p.requires_grad); frozen_grad=any(p.grad is not None for p in encoder.parameters() if not p.requires_grad)
    memory={"allocated_mb":round(torch.cuda.max_memory_allocated(device)/1024**2,2),"reserved_mb":round(torch.cuda.max_memory_reserved(device)/1024**2,2)} if device.type=="cuda" else None
    if not last_grad or not head_grad or frozen_grad or attention.shape!=(len(labels),16): raise RuntimeError(f"E15 backward smoke failed: last_grad={last_grad}, head_grad={head_grad}, frozen_grad={frozen_grad}, attention={tuple(attention.shape)}")
    counts={"encoder":_count_parameters(encoder),"last_block":_count_parameters(last_block),"head":_count_parameters(head)}; del encoder,head
    if device.type=="cuda": torch.cuda.empty_cache()
    return {**static,"gpu_backward_smoke":"passed","device":str(device),"input_samples":len(sample),"loss":float(loss.detach().cpu()),"last_block_gradient":last_grad,"head_gradient":head_grad,"frozen_encoder_gradient":frozen_grad,"parameter_counts":counts,"gpu_memory":memory,"encoder":model_info,"e14_initialization":e14_info,"artifacts_written":False}


def run_e15(config:Mapping[str,Any],*,overwrite_results:bool=False)->dict[str,Any]:
    validate_e15_contract(config); static_smoke_check(config); train_bv,validation=_frames(config); trace=_load_e10_trace(config,validation); train_hash,val_hash=_sha256(Path(config["resolved"]["train_manifest"])),_sha256(Path(config["resolved"]["validation_manifest"])); device=_device(str(config["runtime"]["device"])); head,e14_info=_load_e14_head(config,train_hash,val_hash,device); processor,encoder,last_block,model_info=_load_encoder_and_last_block(config,device); paths=_prepare_output(config,overwrite_results)
    _seed(int(config["training"]["seed"])); encoder.train(); head.train(); counts={"encoder":_count_parameters(encoder),"last_block":_count_parameters(last_block),"head":_count_parameters(head)}; optimizer=torch.optim.AdamW([{"params":[p for p in last_block.parameters() if p.requires_grad],"lr":float(config["training"]["encoder_learning_rate"]),"name":"last_transformer_block"},{"params":[p for p in head.parameters() if p.requires_grad],"lr":float(config["training"]["head_learning_rate"]),"name":"patch_attention_head"}],weight_decay=float(config["training"]["weight_decay"]))
    if any(not p.requires_grad for group in optimizer.param_groups for p in group["params"]): raise RuntimeError("E15 optimizer contains frozen parameters.")
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and bool(config["runtime"]["amp"])) if device.type=="cuda" else None; criterion=nn.CrossEntropyLoss(); history=[]; loader=_loader(train_bv,Path(config["resolved"]["project_root"]),shuffle=True,config=config); accumulation=int(config["runtime"]["gradient_accumulation_steps"])
    for epoch in range(1,int(config["training"]["fixed_epochs"])+1):
        encoder.train(); head.train(); optimizer.zero_grad(set_to_none=True); loss_sum=0.0; seen=0; targets:list[int]=[]; predictions:list[int]=[]
        for step,(images,labels,_) in enumerate(loader,start=1):
            logits,_=_forward(images,processor,encoder,head,config,device,training=True); loss=criterion(logits,labels.to(device))
            if not bool(torch.isfinite(loss)): raise FloatingPointError(f"E15 non-finite training loss at epoch {epoch}, step {step}.")
            if scaler is None: (loss/accumulation).backward()
            else: scaler.scale(loss/accumulation).backward()
            if step%accumulation==0 or step==len(loader):
                if scaler is None: optimizer.step()
                else: scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_sum+=float(loss.detach().cpu())*len(labels); seen+=len(labels); targets.extend(labels.tolist()); predictions.extend(logits.detach().argmax(1).cpu().tolist())
        train_metrics=_binary_metrics(np.asarray(targets),np.asarray(predictions)); history.append({"epoch":epoch,"train_loss":loss_sum/seen,"train_macro_f1":train_metrics["macro_f1"],"train_accuracy":train_metrics["accuracy"],"train_confusion_matrix":train_metrics["confusion_matrix"],"train_bacterial_to_viral":train_metrics["bacterial_to_viral"],"train_viral_to_bacterial":train_metrics["viral_to_bacterial"],"checkpoint_selection":"none_fixed_final_epoch"})
    _atomic_torch_save(paths["final"],_checkpoint_payload(config,int(config["training"]["fixed_epochs"]),encoder,last_block,head,optimizer,counts,e14_info))
    encoder.eval(); head.eval(); full_frame=validation.copy(); full_frame["binary_label"]=full_frame["label"].map({1:0,2:1}).fillna(0).astype(int); probabilities=[]
    for images,_,_ in _loader(full_frame,Path(config["resolved"]["project_root"]),shuffle=False,config=config): logits,_=_forward(images,processor,encoder,head,config,device,training=False); probabilities.append(torch.softmax(logits,1).float().cpu().numpy())
    probabilities=np.concatenate(probabilities); binary_pred=probabilities.argmax(1); targets=validation["label"].to_numpy(dtype=np.int64); e6=trace["e6_pred"].astype(int).to_numpy(); final=np.where(e6==0,0,binary_pred+1).astype(np.int64); final_metrics=classification_metrics(targets,np.eye(3)[final]); final_metrics["macro_auroc_ovr"]=None
    for values in final_metrics["per_class"].values(): values["auroc_ovr"]=None
    matrix=np.asarray(final_metrics["confusion_matrix"],dtype=np.int64); final_metrics.update({"bacterial_to_viral":int(matrix[1,2]),"viral_to_bacterial":int(matrix[2,1]),"bacterial_viral_misclassifications":int(matrix[1,2]+matrix[2,1]),"auroc_not_probability_comparable":True})
    bv=targets>=1; e15_bv=_binary_metrics(targets[bv]-1,binary_pred[bv],probabilities[bv]); e9_bv=_binary_metrics(targets[bv]-1,trace.loc[bv,"e9_pred"].astype(int).to_numpy(),trace.loc[bv,["e9_prob_bacterial","e9_prob_viral"]].to_numpy(dtype=float)); e10_final=trace["final_pred"].astype(int).to_numpy(); e10_metrics=classification_metrics(targets,np.eye(3)[e10_final]); e10_metrics["macro_auroc_ovr"]=None
    metrics={"experiment":"e15","training_policy":"full_3300_bv_train_fixed_3_epochs_final_epoch_only","e14_initialization":e14_info,"e9_binary_reference_from_e10_trace":e9_bv,"e14_binary_reference":e14_info["e14_binary_metrics"],"e15_binary_specialist_on_true_bv_validation":e15_bv,"e10_full_reference_from_trace":{"macro_f1":e10_metrics["macro_f1"],"accuracy":e10_metrics["accuracy"],"configured_reference":float(config["e10_routing_input"]["expected_e10_macro_f1"])},"e15_normal_preserving_full_validation":final_metrics,"macro_f1_delta_vs_e10_trace":float(final_metrics["macro_f1"]-e10_metrics["macro_f1"])}
    predictions=validation[["sample_id","image_path","group_key","source_identifier","label","class_name"]].copy().rename(columns={"label":"true_label","class_name":"true_class_name"}); predictions["e10_e6_pred"]=e6; predictions["e10_e9_pred"]=trace["e9_pred"].astype(int); predictions["e10_final_pred"]=e10_final; predictions["e15_binary_pred"]=binary_pred; predictions["e15_prob_bacterial"],predictions["e15_prob_viral"]=probabilities[:,0],probabilities[:,1]; predictions["e15_final_pred"]=final; predictions["e15_final_pred_name"]=[CANONICAL_CLASSES[value] for value in final]; predictions["e10_normal_kept"]=e6==0
    _atomic_json(paths["history"],history); _atomic_json(paths["metrics"],metrics); _atomic_csv(paths["predictions"],predictions); _atomic_json(paths["metadata"],{"experiment":"e15","encoder":model_info,"parameter_counts":counts,"gradient_accumulation_steps":accumulation,"amp":bool(config["runtime"]["amp"]),"e14_feature_cache_reused_for_training":False,"internal_holdout_used":False,"early_stopping_used":False,"checkpoint_policy":"final_epoch_only","validation_used_for_training_or_selection":False,"gpu_memory_final":{"allocated_mb":round(torch.cuda.max_memory_allocated(device)/1024**2,2),"reserved_mb":round(torch.cuda.max_memory_reserved(device)/1024**2,2)} if device.type=="cuda" else None})
    return {"output_dir":str(paths["output"]),"final_epoch":int(config["training"]["fixed_epochs"]),"e15_macro_f1":final_metrics["macro_f1"],"e10_macro_f1":e10_metrics["macro_f1"],"macro_f1_delta":metrics["macro_f1_delta_vs_e10_trace"],"result_files":{key:str(value) for key,value in paths.items() if key!="output"}}
