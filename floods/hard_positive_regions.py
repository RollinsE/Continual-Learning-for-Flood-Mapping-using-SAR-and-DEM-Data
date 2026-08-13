"""Audit-guided hard-positive region mining and crop supervision.

Mines false-negative regions from a labelled training split and replays crops
centred on the missed flood pixels.  Validation/test mining is diagnostic only
and must not be used to construct training manifests.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from floods.evaluation import BinaryThresholdSweep, load_checkpoint_state
from floods.hard_examples import _tile_keys
from floods.hard_negative_regions import (_IndexedDataset, _box_iou, _clamp_crop,
    _connected_components, _event_id, _safe_div)
from floods.utils.common import get_logger
from floods.utils.console import progress_iter
LOG=get_logger(__name__)
MODE_NORMAL=0
MODE_AUDIT_HARD_POSITIVE=4


def _positive_candidate(component: Dict[str, Any], prob: np.ndarray, target: np.ndarray,
                        threshold: float, crop_sizes: Sequence[int], min_fn_pixels: int,
                        min_label_fg_ratio: float, min_valid_ratio: float) -> Optional[Dict[str, Any]]:
    height,width=target.shape
    valid=target!=255
    flood=(target==1)&valid
    missed=(prob < threshold)&flood
    labels=component['labels']
    component_mask=labels==int(component['label'])
    eligible=[int(s) for s in crop_sizes if int(s)<=min(height,width)]
    if not eligible: return None
    preferred=max(int(component['w']),int(component['h']),1)
    size=next((s for s in sorted(eligible) if s>=preferred),max(eligible))
    component_prob=np.where(component_mask,prob,2.0)
    min_index=int(np.argmin(component_prob))
    peak_y,peak_x=np.unravel_index(min_index,prob.shape)
    center_y=(float(component['cy'])+float(peak_y))/2.0
    center_x=(float(component['cx'])+float(peak_x))/2.0
    y0,x0=_clamp_crop(center_y,center_x,size,height,width)
    crop_valid=valid[y0:y0+size,x0:x0+size]
    valid_pixels=int(np.count_nonzero(crop_valid))
    valid_ratio=_safe_div(valid_pixels,size*size)
    if valid_pixels<=0 or valid_ratio<float(min_valid_ratio): return None
    crop_target=target[y0:y0+size,x0:x0+size]
    crop_missed=missed[y0:y0+size,x0:x0+size]
    crop_prob=prob[y0:y0+size,x0:x0+size]
    fn_pixels=int(np.count_nonzero(crop_missed))
    if fn_pixels<int(min_fn_pixels): return None
    fg_pixels=int(np.count_nonzero((crop_target==1)&crop_valid))
    fg_ratio=_safe_div(fg_pixels,valid_pixels)
    if fg_ratio<float(min_label_fg_ratio): return None
    missed_probs=crop_prob[crop_missed]
    mean_fn_probability=float(missed_probs.mean()) if missed_probs.size else 0.0
    min_probability=float(missed_probs.min()) if missed_probs.size else 0.0
    component_pixels=int(component['area'])
    score=float(fn_pixels*max(threshold-mean_fn_probability,1e-4))
    return {'x0':int(x0),'y0':int(y0),'crop_size':int(size),
            'component_area':component_pixels,'fn_pixels':fn_pixels,
            'fn_ratio':_safe_div(fn_pixels,valid_pixels),'fg_pixels':fg_pixels,
            'fg_ratio':fg_ratio,'valid_ratio':valid_ratio,
            'mean_fn_probability':mean_fn_probability,'min_probability':min_probability,
            'score':score}


def _select(candidates: List[Dict[str,Any]], max_regions:int, nms_iou:float):
    selected=[]
    for c in sorted(candidates,key=lambda r:(r['score'],r['fn_pixels']),reverse=True):
        box=(int(c['y0']),int(c['x0']),int(c['crop_size']))
        if any(_box_iou(box,(int(r['y0']),int(r['x0']),int(r['crop_size'])))>nms_iou for r in selected):
            continue
        selected.append(c)
        if len(selected)>=int(max_regions): break
    return selected


def mine_hard_positive_regions(config:Any, checkpoint_path:Path, output_dir:Path,
        split:str='train', threshold:float=0.50, crop_sizes:Sequence[int]=(320,384),
        min_component_area:int=32, min_fn_pixels:int=64, min_label_fg_ratio:float=0.002,
        min_valid_ratio:float=0.50, max_regions_per_tile:int=3, nms_iou:float=0.30,
        max_samples:Optional[int]=None)->Dict[str,Any]:
    from floods.datasets.flood import FloodDataset, RGBFloodDataset
    from floods.eval_collate import pad_segmentation_batch
    from floods.normalization import describe_stats, load_normalization_stats
    from floods.prepare import eval_transforms, prepare_model
    from floods.utils.ml import seed_everything, seed_worker
    if split!='train': LOG.warning('Hard-positive mining is normally run on split=train. Using split=%s as requested.',split)
    threshold=float(threshold)
    if not 0.0<threshold<1.0: raise ValueError('threshold must be between 0 and 1')
    crop_sizes=sorted({int(v) for v in crop_sizes if int(v)>0})
    if not crop_sizes: raise ValueError('crop_sizes must contain at least one positive integer')
    seed_everything(config.seed,deterministic=True)
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    use_cuda=torch.cuda.is_available() and not bool(config.trainer.cpu)
    device=torch.device('cuda' if use_cuda else 'cpu')
    amp_enabled=bool(config.trainer.amp and use_cuda)
    use_rgb=(config.data.in_channels-int(config.data.include_dem))==3
    dataset_cls=RGBFloodDataset if use_rgb else FloodDataset
    modalities=(['r','g','b'] if use_rgb else ['vv','vh'])+(['dem'] if config.data.include_dem else [])
    norm_mode=str(getattr(config.data,'normalization_mode','fixed') or 'fixed').lower()
    if norm_mode in {'stats','robust_percentile','notebook_robust','robust_minmax'} or getattr(config.data,'normalization_stats_path',None):
        if not config.data.normalization_stats_path: raise ValueError(f"normalization_mode='{norm_mode}' requires --normalization-stats-path")
        mean,std,clip_min,clip_max=load_normalization_stats(Path(config.data.normalization_stats_path),modalities,mode=norm_mode)
        LOG.info('Using train-fitted normalization stats (%s): %s',norm_mode,describe_stats(Path(config.data.normalization_stats_path)))
    else:
        mean=dataset_cls.mean()[:config.data.in_channels]; std=dataset_cls.std()[:config.data.in_channels]
        clip_min=tuple([-30.0]*config.data.in_channels); clip_max=tuple([30.0]*config.data.in_channels)
    transform=eval_transforms(mean=mean,std=std,clip_min=clip_min,clip_max=clip_max,normalization_mode=norm_mode)
    dataset=dataset_cls(path=Path(config.data.path),subset=split,include_dem=config.data.include_dem,normalization=transform)
    indexed:Dataset=_IndexedDataset(dataset)
    if max_samples is not None and int(max_samples)>0:
        indexed=torch.utils.data.Subset(indexed,list(range(min(int(max_samples),len(dataset)))))
    loader=DataLoader(indexed,batch_size=config.trainer.batch_size,shuffle=False,num_workers=config.trainer.num_workers,worker_init_fn=seed_worker,collate_fn=pad_segmentation_batch)
    model=prepare_model(config=config,num_classes=1,stage='eval')
    model.load_state_dict(load_checkpoint_state(Path(checkpoint_path)),strict=not config.model.multibranch)
    model=model.to(device); model.eval()
    rows=[]; tiles_with_regions=0
    LOG.info('Mining hard-positive regions from: %s',checkpoint_path)
    LOG.info('Dataset: %s split, %d samples | threshold=%.2f | crop sizes=%s',split,len(indexed),threshold,crop_sizes)
    with torch.no_grad():
      for x,y,index in progress_iter(loader,desc=f'Mine hard positives {split}',unit='batch',colour='green'):
        x=torch.nan_to_num(x.float(),nan=0.0,posinf=30.0,neginf=-30.0).clamp(-30.0,30.0).to(device)
        with torch.autocast(device_type=device.type,enabled=amp_enabled): out=model(x)
        logits=BinaryThresholdSweep._main_prediction(out).detach().float()
        probs=torch.sigmoid(BinaryThresholdSweep._squeeze_logits(logits)).cpu().numpy()
        targets=y.detach().cpu().numpy()
        if targets.ndim==4 and targets.shape[1]==1: targets=targets[:,0]
        indices=index.detach().cpu().numpy().tolist() if isinstance(index,torch.Tensor) else list(index)
        for b,dataset_index in enumerate(indices):
            dataset_index=int(dataset_index); target=targets[b].astype(np.uint8); prob=probs[b].astype(np.float32)
            valid=target!=255; missed=(prob<threshold)&(target==1)&valid
            components=[c for c in _connected_components(missed) if int(c['area'])>=int(min_component_area)]
            candidates=[]
            for c in components:
                candidate=_positive_candidate(c,prob,target,threshold,crop_sizes,min_fn_pixels,min_label_fg_ratio,min_valid_ratio)
                if candidate is not None: candidates.append(candidate)
            selected=_select(candidates,max_regions_per_tile,nms_iou)
            if not selected: continue
            tiles_with_regions+=1
            image_path=Path(dataset.image_files[dataset_index]); mask_path=Path(dataset.label_files[dataset_index])
            dem_path=Path(dataset.dem_files[dataset_index]) if getattr(dataset,'_include_dem',False) else None
            for rank,candidate in enumerate(selected,start=1):
                rows.append({'split':split,'index':dataset_index,'tile_id':image_path.stem,'file':image_path.name,
                    'event_id':_event_id(image_path.name),'image_path':str(image_path),'mask_path':str(mask_path),
                    'dem_path':str(dem_path) if dem_path else '', 'threshold':threshold,'region_rank':rank,**candidate})
    manifest=pd.DataFrame(rows); manifest_path=output_dir/'hard_positive_regions.csv'; manifest.to_csv(manifest_path,index=False)
    summary={'checkpoint':str(checkpoint_path),'split':split,'samples':int(len(indexed)),'threshold':threshold,
        'crop_sizes':crop_sizes,'min_component_area':int(min_component_area),'min_fn_pixels':int(min_fn_pixels),
        'min_label_fg_ratio':float(min_label_fg_ratio),'min_valid_ratio':float(min_valid_ratio),
        'max_regions_per_tile':int(max_regions_per_tile),'nms_iou':float(nms_iou),
        'tiles_with_regions':int(tiles_with_regions),'regions':int(len(manifest)),'manifest':str(manifest_path)}
    with (output_dir/'summary.json').open('w',encoding='utf-8') as f: json.dump(summary,f,indent=2)
    LOG.info('Hard-positive mining written to: %s',output_dir)
    LOG.info('Mined %d regions from %d/%d tiles',len(manifest),tiles_with_regions,len(indexed))
    if manifest.empty: LOG.warning('No hard-positive regions met the criteria. Raise --threshold or lower --min-fn-pixels.')
    return summary


class AuditGuidedHardPositiveCropSupervision:
    def __init__(self,manifest_path:str|Path,target_size:int,probability:float=1.0,ignore_index:int=255)->None:
        self.manifest_path=Path(manifest_path).expanduser()
        if not self.manifest_path.exists(): raise FileNotFoundError(f'Hard-positive region manifest does not exist: {self.manifest_path}')
        frame=pd.read_csv(self.manifest_path); required={'x0','y0','crop_size'}; missing=required-set(frame.columns)
        if missing: raise ValueError(f'Hard-positive manifest is missing columns: {sorted(missing)}')
        if frame.empty: raise ValueError(f'Hard-positive manifest is empty: {self.manifest_path}')
        self.target_size=int(target_size); self.probability=float(probability); self.ignore_index=int(ignore_index)
        if not 0.0<=self.probability<=1.0: raise ValueError('hard-positive crop probability must be between 0 and 1')
        self.regions={}
        for _,row in frame.iterrows():
            record=row.to_dict(); keys=set()
            for column in ('image_path','mask_path','file','tile_id'):
                if column in record: keys.update(_tile_keys(record[column]))
            for key in keys: self.regions.setdefault(key,[]).append(record)
        if not self.regions: raise ValueError(f'No usable tile identifiers found in manifest: {self.manifest_path}')
    def matching_regions(self,sample_path):
        matches=[]; seen=set()
        for key in _tile_keys(sample_path):
            for record in self.regions.get(key,[]):
                marker=(int(record['x0']),int(record['y0']),int(record['crop_size']))
                if marker not in seen: matches.append(record); seen.add(marker)
        return matches
    def has_regions(self,sample_path): return bool(self.matching_regions(sample_path))
    def __call__(self,image,mask,sample_path):
        matches=self.matching_regions(sample_path)
        if not matches or np.random.random()>self.probability: return image,mask,None
        weights=np.asarray([max(float(r.get('score',r.get('fn_pixels',1.0))),1e-8) for r in matches],dtype=np.float64); weights/=weights.sum()
        row=matches[int(np.random.choice(len(matches),p=weights))]
        size=int(row['crop_size']); y0,x0=int(row['y0']),int(row['x0']); h,w=mask.shape
        if size<=0 or y0<0 or x0<0 or y0+size>h or x0+size>w: return image,mask,None
        ci=image[y0:y0+size,x0:x0+size,...]; cm=mask[y0:y0+size,x0:x0+size]
        if ci.size==0 or cm.size==0: return image,mask,None
        if size!=self.target_size:
            target=(self.target_size,self.target_size); ci=cv2.resize(ci,target,interpolation=cv2.INTER_LINEAR)
            if ci.ndim==2: ci=ci[...,None]
            cm=cv2.resize(cm,target,interpolation=cv2.INTER_NEAREST)
        return ci.astype(image.dtype,copy=False),cm.astype(mask.dtype,copy=False),None


def manifest_matching_indices(label_files:Sequence[str],manifest_path:str|Path)->set[int]:
    supervisor=AuditGuidedHardPositiveCropSupervision(manifest_path,target_size=1,probability=0.0)
    return {idx for idx,path in enumerate(label_files) if supervisor.has_regions(path)}


def prepare_hard_positive_region_sampler(dataset:Any,manifest_path:str|Path,weight:float=3.0,max_fraction:float=0.20,samples_multiplier:float=1.0)->WeightedRandomSampler:
    weight=float(weight); max_fraction=float(max_fraction)
    if weight<=1.0: raise ValueError('hard_positive_region_weight must be greater than 1.0')
    if not 0.0<max_fraction<1.0: raise ValueError('hard_positive_region_max_fraction must be between 0 and 1')
    indices=manifest_matching_indices(dataset.label_files,manifest_path)
    if not indices: raise ValueError('Hard-positive manifest does not match any current training tiles')
    weights=np.ones(len(dataset),dtype=np.float64); hard=np.zeros(len(dataset),dtype=bool); hard[list(indices)]=True; weights[hard]*=weight
    hard_mass=float(weights[hard].sum()); normal_mass=float(weights[~hard].sum()); frac=_safe_div(hard_mass,hard_mass+normal_mass)
    if frac>max_fraction and normal_mass>0:
        target=(max_fraction*normal_mass)/(1.0-max_fraction); weights[hard]*=target/hard_mass; hard_mass=float(weights[hard].sum()); frac=_safe_div(hard_mass,hard_mass+normal_mass)
    multiplier=float(samples_multiplier or 1.0)
    if multiplier<=0: raise ValueError('weighted_samples_multiplier must be greater than 0')
    num_samples=max(1,int(round(len(dataset)*multiplier)))
    LOG.info('Audit-guided hard-positive region sampling: %d samples per epoch | region tiles=%d/%d | effective region mass=%.2f | weight=%.2f',num_samples,int(np.count_nonzero(hard)),len(dataset),frac,weight)
    return WeightedRandomSampler(weights=weights,num_samples=num_samples,replacement=True)
