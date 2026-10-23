"""E16 frozen E15 global/local feature fusion with conservative non-segmentation ROI."""
from __future__ import annotations
import contextlib, hashlib, json, os, random
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw, ImageOps
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from .e14_patch_specialist import (BINARY_CLASSES, CANONICAL_CLASSES, PatchAttentionHead, RawManifestDataset, _binary_metrics, _binary_train_frame, _collate, _load_e10_trace, _manifest, _pool_patch_tokens)
from .e15_finetune import _load_encoder_and_last_block
from .metrics import classification_metrics

PROJECT_ROOT=Path(__file__).resolve().parents[2]


def _inside(value:str|Path,root:Path)->Path:
    path=Path(value).expanduser(); resolved=path.resolve() if path.is_absolute() else (root/path).resolve()
    if not resolved.is_relative_to(root): raise ValueError(f"E16 path leaves project root: {value}")
    return resolved

def _yaml(path:Path)->dict[str,Any]:
    value=yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict): raise ValueError(f"E16 YAML must be mapping: {path}")
    return value

def _sha(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def _json(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+".tmp"); temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8"); os.replace(temp,path)
def _csv(path:Path,frame:pd.DataFrame)->None:
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+".tmp"); frame.to_csv(temp,index=False); os.replace(temp,path)
def _npz(path:Path,features:np.ndarray,labels:np.ndarray)->None:
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_name(path.name+".tmp.npz"); np.savez_compressed(temp,features=features,labels=labels); os.replace(temp,path)
def _torch(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+".tmp"); torch.save(value,temp); os.replace(temp,path)

def validate_e16_contract(config:Mapping[str,Any])->None:
    required={"experiment":"e16","status":"frozen_e15_global_local_dual_view_fusion","data_config":"configs/data_config.yaml","output_dir":"artifacts/experiments/e16"}
    errors=[{"key":k,"expected":v,"actual":config.get(k)} for k,v in required.items() if config.get(k)!=v]
    model=config.get("model",{}); expected={"hf_model_id":"microsoft/rad-dino","revision":"main","required_transformers_version":"4.57.6","expected_hidden_size":768,"expected_num_hidden_layers":12,"patch_size":14,"expected_image_size":518,"spatial_pool_grid":[4,4],"e15_restore":"base_rad_dino_plus_final_last_block_and_head","encoder_trainable":False,"e15_head_trainable":False}
    errors.extend({"key":f"model.{k}","expected":v,"actual":model.get(k)} for k,v in expected.items() if model.get(k)!=v)
    roi=config.get("roi",{}); expected_roi={"method":"conservative_foreground_bounds_no_segmentation","foreground_threshold":5,"expansion_fraction":0.02,"minimum_width_fraction":0.78,"minimum_height_fraction":0.78,"minimum_area_fraction":0.60,"fallback":"original_image","quality_contact_sheet_train_only":True,"seed":42}
    errors.extend({"key":f"roi.{k}","expected":v,"actual":roi.get(k)} for k,v in expected_roi.items() if roi.get(k)!=v)
    train=config.get("training",{}); expected_train={"seed":42,"fixed_epochs":5,"optimizer":"AdamW","learning_rate":0.001,"weight_decay":0.0001,"checkpoint_policy":"final_epoch_only","internal_holdout":"disabled","early_stopping":False,"validation_weight_selection":False,"validation_threshold_selection":False}
    errors.extend({"key":f"training.{k}","expected":v,"actual":train.get(k)} for k,v in expected_train.items() if train.get(k)!=v)
    if config.get("features",{}).get("dimension_per_view")!=770 or config.get("fusion_head",{}).get("num_classes")!=2: errors.append({"key":"features/fusion_head","actual":config.get("features")})
    if config.get("e10_routing_input",{}).get("normal_rule")!="e6_pred_zero_kept_else_e16_binary_plus_one": errors.append({"key":"routing","actual":config.get("e10_routing_input",{})})
    if errors: raise ValueError(f"E16 config contract drift: {errors}")

def load_e16_config(path:str|Path,*,project_root:str|Path=PROJECT_ROOT)->dict[str,Any]:
    root=Path(project_root).expanduser().resolve(); config_path=_inside(path,root)
    config=_yaml(config_path); validate_e16_contract(config)
    data=_yaml(_inside(config["data_config"],root)); manifests=data.get("manifests",{})
    if set(manifests)!={"train","validation"}: raise ValueError("E16 requires shared train/validation manifests only.")
    split,e15,e10=config["split_contract"],config["e15_initialization"],config["e10_routing_input"]
    if _inside(split["train_manifest"],root)!=_inside(manifests["train"],root) or _inside(split["validation_manifest"],root)!=_inside(manifests["validation"],root): raise ValueError("E16 manifest reuse contract failed.")
    config["resolved"]={"project_root":str(root),"config_path":str(config_path),"train_manifest":str(_inside(split["train_manifest"],root)),"validation_manifest":str(_inside(split["validation_manifest"],root)),"e15_final":str(_inside(e15["final_checkpoint"],root)),"e15_predictions":str(_inside(e15["predictions"],root)),"e15_metrics":str(_inside(e15["metrics"],root)),"e10_predictions":str(_inside(e10["predictions"],root)),"output_dir":str(_inside(config["output_dir"],root))}
    return config

def _frames(config:Mapping[str,Any])->tuple[pd.DataFrame,pd.DataFrame]:
    root=Path(config["resolved"]["project_root"]); split=config["split_contract"]
    full=_manifest(Path(config["resolved"]["train_manifest"]),root,"train",{"total":4448,"NORMAL":1148,"BACTERIAL_PNEUMONIA":2153,"VIRAL_PNEUMONIA":1147})
    return _binary_train_frame(full,config),_manifest(Path(config["resolved"]["validation_manifest"]),root,"validation",split["expected_counts"]["validation"])

def _roi(image:Image.Image,config:Mapping[str,Any])->tuple[Image.Image,dict[str,Any]]:
    gray=np.asarray(image.convert("L")); height,width=gray.shape; threshold=int(config["roi"]["foreground_threshold"]); ys,xs=np.where(gray>threshold)
    original={"roi_left":0,"roi_top":0,"roi_right":width,"roi_bottom":height,"roi_width_fraction":1.0,"roi_height_fraction":1.0,"roi_area_fraction":1.0,"roi_fallback":True}
    if len(xs)==0: return image.copy(),original
    left,right,top,bottom=int(xs.min()),int(xs.max()+1),int(ys.min()),int(ys.max()+1); pad_x=round(width*float(config["roi"]["expansion_fraction"])); pad_y=round(height*float(config["roi"]["expansion_fraction"])); left=max(0,left-pad_x); right=min(width,right+pad_x); top=max(0,top-pad_y); bottom=min(height,bottom+pad_y); wf=(right-left)/width; hf=(bottom-top)/height; af=wf*hf
    if wf<float(config["roi"]["minimum_width_fraction"]) or hf<float(config["roi"]["minimum_height_fraction"]) or af<float(config["roi"]["minimum_area_fraction"]): return image.copy(),original
    return image.crop((left,top,right,bottom)),{"roi_left":left,"roi_top":top,"roi_right":right,"roi_bottom":bottom,"roi_width_fraction":wf,"roi_height_fraction":hf,"roi_area_fraction":af,"roi_fallback":False}

def static_smoke_check(config:Mapping[str,Any])->dict[str,Any]:
    train,validation=_frames(config)
    if set(train.group_key)&set(validation.group_key): raise RuntimeError("E16 train/validation proxy-group overlap.")
    wide=Image.new("L",(1200,600),0); from PIL import ImageDraw as _Draw; _Draw.Draw(wide).rectangle((40,25,1160,575),fill=128); local,quality=_roi(wide,config)
    if quality["roi_fallback"] or local.size[0]>=wide.size[0]: raise RuntimeError("E16 conservative ROI synthetic check failed.")
    required={key:Path(config["resolved"][key]).is_file() for key in ("e15_final","e15_predictions","e15_metrics","e10_predictions")}
    return {"experiment":"e16","static_check":"passed","train_bv_samples":len(train),"validation_samples":len(validation),"train_validation_group_overlap":0,"roi_synthetic_quality":quality,"roi_uses_validation_errors":False,"e15_e10_prerequisites":required,"output_dir_exists":Path(config["resolved"]["output_dir"]).exists()}

def _device(value:str)->torch.device:
    if value=="cuda":
        if not torch.cuda.is_available(): raise RuntimeError("E16 requires CUDA.")
        return torch.device("cuda")
    return torch.device("cpu") if value=="cpu" else torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _restore_e15(config:Mapping[str,Any],device:torch.device)->tuple[Any,nn.Module,PatchAttentionHead,dict[str,Any]]:
    final,pred,metric=Path(config["resolved"]["e15_final"]),Path(config["resolved"]["e15_predictions"]),Path(config["resolved"]["e15_metrics"])
    missing=[str(path) for path in (final,pred,metric) if not path.is_file()]
    if missing: raise FileNotFoundError("E16 requires E15 final checkpoint/predictions/metrics: "+", ".join(missing))
    processor,encoder,last_block,info=_load_encoder_and_last_block(config,device); payload=torch.load(final,map_location="cpu",weights_only=True)
    if payload.get("experiment")!="e15" or payload.get("checkpoint_policy")!="final_epoch_only" or payload.get("config",{}).get("head")!=config["head"]: raise ValueError("E15 final checkpoint contract mismatch.")
    last_block.load_state_dict(payload["last_block_state_dict"],strict=True); head=PatchAttentionHead(config).to(device); head.load_state_dict(payload["head_state_dict"],strict=True)
    for module in (encoder,head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
        module.eval()
    if any(parameter.requires_grad for parameter in encoder.parameters()) or any(parameter.requires_grad for parameter in head.parameters()): raise RuntimeError("E16 requires fully frozen restored E15 model.")
    info["e15_final_checkpoint"]=str(final); info["e15_last_block_restored"]=True; info["e15_head_restored"]=True
    return processor,encoder,head,info

def _view_feature(image:Image.Image,processor:Any,encoder:nn.Module,head:PatchAttentionHead,config:Mapping[str,Any],device:torch.device)->np.ndarray:
    inputs={k:v.to(device) for k,v in processor(images=[image],return_tensors="pt").items()}; autocast=torch.autocast("cuda",dtype=torch.float16) if bool(config["runtime"]["amp"]) and device.type=="cuda" else contextlib.nullcontext()
    with torch.inference_mode(),autocast:
        hidden=encoder(**inputs).last_hidden_state; logits,_=head(_pool_patch_tokens(hidden,config)[0]); vector=torch.cat([hidden[:,0,:],logits],1)
    if vector.shape!=(1,770) or not bool(torch.isfinite(vector).all()): raise RuntimeError("E16 frozen E15 view feature contract failed.")
    return vector.float().cpu().numpy()[0]

def _extract(frame:pd.DataFrame,label_col:str,view:str,processor:Any,encoder:nn.Module,head:PatchAttentionHead,config:Mapping[str,Any],device:torch.device)->tuple[np.ndarray,list[dict[str,Any]]]:
    root=Path(config["resolved"]["project_root"]); values=[]; roi_rows=[]
    for row in frame.itertuples(index=False):
        path=(root/str(row.image_path)).resolve()
        with Image.open(path) as source:
            source.load(); global_image=source.copy()
        image,quality=(global_image,_roi(global_image,config)[1]) if view=="global" else _roi(global_image,config)
        values.append(_view_feature(image,processor,encoder,head,config,device)); roi_rows.append({"sample_id":str(row.image_path),"image_path":str(row.image_path),"class_name":str(row.class_name),"label":int(getattr(row,label_col)),**quality})
    return np.asarray(values,dtype=np.float16),roi_rows
class FusionHead(nn.Module):
    def __init__(self,config:Mapping[str,Any])->None:
        super().__init__(); hidden=int(config["fusion_head"]["hidden_dimension"]); self.net=nn.Sequential(nn.LayerNorm(1540),nn.Linear(1540,hidden),nn.GELU(),nn.Dropout(float(config["fusion_head"]["dropout"])),nn.Linear(hidden,2))
    def forward(self,global_features:Tensor,local_features:Tensor)->Tensor:
        logits=self.net(torch.cat([global_features,local_features],1))
        if logits.ndim!=2 or logits.shape[1]!=2: raise RuntimeError("E16 fusion logits shape invalid.")
        return logits

def _paths(config:Mapping[str,Any],split:str)->tuple[Path,Path]:
    root=Path(config["resolved"]["output_dir"])/config["artifacts"]["cache_dir"]; art=config["artifacts"]; prefix="train" if split=="train" else "validation"; return root/art[f"{prefix}_global"],root/art[f"{prefix}_local"]
def _save_cache(config:Mapping[str,Any],split:str,global_f:np.ndarray,local_f:np.ndarray,labels:np.ndarray,frame:pd.DataFrame)->None:
    gp,lp=_paths(config,split); _npz(gp,global_f,labels); _npz(lp,local_f,labels); index=Path(config["resolved"]["output_dir"])/config["artifacts"][f"{'train' if split=='train' else 'validation'}_index"]; _csv(index,frame[["image_path","group_key","class_name","label","binary_label"]].copy())
def _load_cache(config:Mapping[str,Any],split:str,frame:pd.DataFrame,label_col:str)->tuple[np.ndarray,np.ndarray,np.ndarray]|None:
    gp,lp=_paths(config,split); art=config["artifacts"]; output=Path(config["resolved"]["output_dir"]); index=output/art[f"{'train' if split=='train' else 'validation'}_index"]; metadata_path=output/art["cache_metadata"]
    if not gp.is_file() or not lp.is_file() or not index.is_file() or not metadata_path.is_file(): return None
    try: metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): return None
    manifest_key="train_manifest_sha256" if split=="train" else "validation_manifest_sha256"; manifest_path=Path(config["resolved"]["train_manifest"] if split=="train" else config["resolved"]["validation_manifest"])
    if metadata.get("cache_version")!=1 or metadata.get("feature_kind")!="cls_plus_frozen_e15_binary_logits" or metadata.get(manifest_key)!=_sha(manifest_path): return None
    with np.load(gp) as g,np.load(lp) as l: global_f,local_f,labels=g["features"],l["features"],g["labels"]
    saved=pd.read_csv(index)
    if global_f.shape!=(len(frame),770) or local_f.shape!=(len(frame),770) or labels.shape!=(len(frame),) or not np.array_equal(labels,frame[label_col].to_numpy(dtype=np.int64)) or saved["image_path"].tolist()!=frame["image_path"].tolist() or not np.isfinite(global_f).all() or not np.isfinite(local_f).all(): return None
    return global_f.astype(np.float32),local_f.astype(np.float32),labels.astype(np.int64)
def _features(config:Mapping[str,Any],split:str,frame:pd.DataFrame,label_col:str,processor:Any,encoder:nn.Module,head:PatchAttentionHead,device:torch.device,*,cache:bool=True)->tuple[np.ndarray,np.ndarray,np.ndarray,list[dict[str,Any]]]:
    cached=_load_cache(config,split,frame,label_col) if cache else None
    if cached is not None: return *cached,[]
    global_f,_=_extract(frame,label_col,"global",processor,encoder,head,config,device); local_f,roi_rows=_extract(frame,label_col,"local",processor,encoder,head,config,device); labels=frame[label_col].to_numpy(dtype=np.int64)
    if cache: _save_cache(config,split,global_f,local_f,labels,frame)
    return global_f.astype(np.float32),local_f.astype(np.float32),labels,roi_rows
def _sheet(frame:pd.DataFrame,config:Mapping[str,Any],output:Path)->int:
    selected=[]
    for label in (0,1): selected.extend(frame[frame.binary_label==label].head(int(config["roi"]["contact_sheet_samples_per_class"])).to_dict("records"))
    if not selected:return 0
    thumb=(220,160); sheet=Image.new("RGB",(len(selected)*thumb[0],220),"white"); draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(selected):
        with Image.open((PROJECT_ROOT/row["image_path"]).resolve()) as image: local,q=_roi(image.copy(),config); global_thumb=ImageOps.contain(image.convert("RGB"),thumb); local_thumb=ImageOps.contain(local.convert("RGB"),thumb)
        pane=Image.new("RGB",thumb,"white"); pane.paste(global_thumb,((thumb[0]-global_thumb.width)//2,0)); sheet.paste(pane,(i*thumb[0],0)); draw.text((i*thumb[0]+3,164),f"{row['class_name']}\nroi={q['roi_width_fraction']:.2f}x{q['roi_height_fraction']:.2f}\nfallback={q['roi_fallback']}",fill="black")
    output.parent.mkdir(parents=True,exist_ok=True); sheet.save(output); return len(selected)
def _baseline(config:Mapping[str,Any],validation:pd.DataFrame,processor:Any,encoder:nn.Module,head:PatchAttentionHead,device:torch.device)->dict[str,Any]:
    path=Path(config["resolved"]["e15_predictions"]); saved=pd.read_csv(path).set_index("image_path").reindex(validation.image_path).reset_index(); values=[]
    for row in validation.itertuples(index=False):
        with Image.open((Path(config["resolved"]["project_root"])/str(row.image_path)).resolve()) as image: values.append(_view_feature(image.copy(),processor,encoder,head,config,device)[-2:])
    logits=np.asarray(values); probs=np.exp(logits-logits.max(1,keepdims=True)); probs/=probs.sum(1,keepdims=True); pred=probs.argmax(1)
    if not np.array_equal(pred,saved.e15_binary_pred.astype(int).to_numpy()) or not np.allclose(probs,saved[["e15_prob_bacterial","e15_prob_viral"]].to_numpy(float),rtol=0,atol=5e-4) or not np.array_equal(np.where(saved.e10_e6_pred.astype(int).to_numpy()==0,0,pred+1),saved.e15_final_pred.astype(int).to_numpy()): raise RuntimeError("E15 global baseline reproduction failed; E16 fusion is blocked.")
    return {"e15_binary_prediction_exact":True,"e15_probability_atol":5e-4,"max_probability_absolute_error":float(np.abs(probs-saved[["e15_prob_bacterial","e15_prob_viral"]].to_numpy(float)).max())}
def gpu_smoke_check(config:Mapping[str,Any],*,device_request:str|None=None)->dict[str,Any]:
    static=static_smoke_check(config); train,_=_frames(config); device=_device(device_request or str(config["runtime"]["device"])); processor,encoder,head,info=_restore_e15(config,device); sample=train.iloc[:2].reset_index(drop=True); g,l,_,roi=_features(config,"smoke",sample,"binary_label",processor,encoder,head,device,cache=False); fusion=FusionHead(config).to(device); global_tensor=torch.from_numpy(g).to(device=device,dtype=torch.float32); local_tensor=torch.from_numpy(l).to(device=device,dtype=torch.float32); logits=fusion(global_tensor,local_tensor);
    if logits.shape!=(2,2) or any(p.requires_grad for p in encoder.parameters()) or any(p.requires_grad for p in head.parameters()): raise RuntimeError("E16 smoke frozen/fusion contract failed.")
    return {**static,"gpu_smoke":"passed","encoder":info,"feature_shapes":{"global":list(g.shape),"local":list(l.shape)},"fusion_logits":list(logits.shape),"roi_rows":len(roi),"artifacts_written":False}
def run_e16(config:Mapping[str,Any],*,overwrite_results:bool=False)->dict[str,Any]:
    static_smoke_check(config); train,validation=_frames(config); trace=_load_e10_trace(config,validation); device=_device(str(config["runtime"]["device"])); processor,encoder,head,info=_restore_e15(config,device); baseline=_baseline(config,validation,processor,encoder,head,device); output=Path(config["resolved"]["output_dir"]); art=config["artifacts"]; final=output/art["final_checkpoint"]
    if output.exists() and not overwrite_results: raise FileExistsError(f"E16 output exists: {output}")
    output.mkdir(parents=True,exist_ok=True); sheet_count=_sheet(train,config,output/art["roi_contact_sheet"]); gtr,ltr,ytr,roi_rows=_features(config,"train",train,"binary_label",processor,encoder,head,device); gva,lva,yva,_=_features(config,"validation",validation,"label",processor,encoder,head,device); _csv(output/art["roi_quality"],pd.DataFrame(roi_rows)); _json(output/art["cache_metadata"],{"cache_version":1,"feature_kind":"cls_plus_frozen_e15_binary_logits","global_local_feature_dimension":770,"train_manifest_sha256":_sha(Path(config["resolved"]["train_manifest"])),"validation_manifest_sha256":_sha(Path(config["resolved"]["validation_manifest"])),"sample_id":"image_path","e15":info})
    fusion=FusionHead(config).to(device); opt=torch.optim.AdamW(fusion.parameters(),lr=float(config["training"]["learning_rate"]),weight_decay=float(config["training"]["weight_decay"])); loader=DataLoader(TensorDataset(torch.from_numpy(gtr),torch.from_numpy(ltr),torch.from_numpy(ytr)),batch_size=int(config["runtime"]["fusion_batch_size"]),shuffle=True); hist=[]
    for epoch in range(1,int(config["training"]["fixed_epochs"])+1):
        fusion.train(); losses=[]; targets=[]; preds=[]
        for g,l,y in loader: opt.zero_grad(); global_tensor=g.to(device=device,dtype=torch.float32); local_tensor=l.to(device=device,dtype=torch.float32); logits=fusion(global_tensor,local_tensor); loss=nn.CrossEntropyLoss()(logits,y.to(device)); loss.backward(); opt.step(); losses.append(float(loss.detach())); targets.extend(y.tolist()); preds.extend(logits.argmax(1).detach().cpu().tolist())
        m=_binary_metrics(np.asarray(targets),np.asarray(preds)); hist.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"train_macro_f1":m["macro_f1"],"train_confusion_matrix":m["confusion_matrix"],"checkpoint_selection":"none_fixed_final_epoch"})
    _torch(final,{"experiment":"e16","fusion_state_dict":fusion.state_dict(),"config":config,"e15_restore":info}); fusion.eval(); validation_global=torch.from_numpy(gva).to(device=device,dtype=torch.float32); validation_local=torch.from_numpy(lva).to(device=device,dtype=torch.float32); probs=torch.softmax(fusion(validation_global,validation_local),1).detach().cpu().numpy(); bpred=probs.argmax(1); e6=trace.e6_pred.astype(int).to_numpy(); final_pred=np.where(e6==0,0,bpred+1); metrics=classification_metrics(yva,np.eye(3)[final_pred]); metrics["macro_auroc_ovr"]=None; cm=np.asarray(metrics["confusion_matrix"]); metrics.update({"bacterial_to_viral":int(cm[1,2]),"viral_to_bacterial":int(cm[2,1]),"bacterial_viral_misclassifications":int(cm[1,2]+cm[2,1])}); saved=pd.read_csv(Path(config["resolved"]["e15_predictions"])); e15_metric={"available":True}
    pred=validation[["image_path","group_key","label","class_name","source_identifier"]].copy(); pred["sample_id"]=pred.image_path; pred["e16_binary_pred"]=bpred; pred["e16_prob_bacterial"],pred["e16_prob_viral"]=probs[:,0],probs[:,1]; pred["e16_final_pred"]=final_pred; pred["e10_normal_kept"]=e6==0
    _json(output/art["history"],hist); _json(output/art["metrics"],{"experiment":"e16","baseline_reproduction":baseline,"e15_reference":e15_metric,"e16_full_validation":metrics}); _csv(output/art["predictions"],pred); _json(output/art["run_metadata"],{"encoder":info,"encoder_frozen":True,"e15_head_frozen":True,"roi":"conservative image-only; train-only quality sheet; fallback original","validation_used_for_selection":False,"roi_contact_sheet_samples":sheet_count}); return {"output_dir":str(output),"macro_f1":metrics["macro_f1"],"baseline":baseline}
