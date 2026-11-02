#!/usr/bin/env python3
"""06 — Independently audit a physically-pruned FP32 Ultralytics YOLO checkpoint.

One shared implementation for YOLOv8, YOLO11, YOLO26 detection models.
Does not mutate checkpoint/weights. Never uses validation images to estimate
importance. Uses actual labeled TRAIN batches for a REAL detection-loss backward.
Validation always uses VAL, optionally a reproducible subset; compares the
original model on precisely that same subset (unless explicitly disabled).

Structural PASS and RECOVERY_NEEDED are deliberately independent: poor mAP
before recovery is expected and must not be reported as a graph failure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from dataset_legacy_bridge import current_content_inventory, validate_legacy_bridge
from pruning_common import (
    SUPPORTED_FAMILIES, contract, dataset_fingerprint, forward_tensors, gate_file,
    hash_object, make_imgsz, model_head_metadata, now_utc, read_dataset,
    read_json, resolved, save_gate, select_device, setup_log, sha256_file, write_json,
)

STAGE, NAME = '06', 'verify_pruned_model'
IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp', '.dng', '.heic'}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='06: new-process YOLO reload, FP32 graph/ABI, true detection loss backward and same-split smoke validation')
    child_mode = '--_child-reload' in sys.argv
    for key in ('model', 'data', 'output'):
        p.add_argument('--' + key, required=not child_mode)
    p.add_argument('--imgsz', nargs='+', type=int, default=[640])
    p.add_argument('--device', default='0')
    p.add_argument('--batch', type=int, default=8, help='UNCHANGED Stage00–05 CLI contract')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--family', choices=('auto', *SUPPORTED_FAMILIES), default='auto')
    p.add_argument('--expected-classes', type=int)
    p.add_argument('--backward-batches', type=int, default=2, help='Number of labeled TRAIN minibatches used to test backward')
    p.add_argument('--backward-batch-size', type=int, default=2, help='Only Stage06 backward minibatch size; does NOT alter Stage00 contract')
    p.add_argument('--val-max-images', type=int, default=256, help='Smoke subset of VAL; 0 = full VAL')
    p.add_argument('--val-seed', type=int, default=42, help='Deterministic VAL subset sampling seed')
    p.add_argument('--val-batch', type=int, default=4, help='Validation minibatch size; may differ from contract --batch')
    p.add_argument('--val-conf', type=float, default=None, help='Default: Stage01 baseline confidence threshold')
    p.add_argument('--val-iou', type=float, default=None, help='Default: Stage01 baseline IoU threshold')
    p.add_argument('--val-max-det', type=int, default=None, help='Default: Stage01 baseline max_det (e.g. 1000)')
    p.add_argument('--nms-free', action=argparse.BooleanOptionalAction, default=None, help='Default: Stage01 validation NMS configuration')
    p.add_argument('--compare-source-val', action=argparse.BooleanOptionalAction, default=True,
                   help='Validate source and pruned on the SAME VAL subset to measure recovery drop')
    p.add_argument('--recovery-map-drop', type=float, default=0.02, help='mAP50-95 drop marking RECOVERY_NEEDED, not graph FAIL')
    p.add_argument('--reload-device', choices=('cpu', 'same'), default='cpu', help='Independent subprocess device')
    p.add_argument('--reload-timeout', type=int, default=300, help='Seconds for fresh-process reload/ABI test')
    p.add_argument('--max-cuda-peak-mib', type=float, default=0, help='Optional HARD CUDA memory ceiling; 0 only records peak')
    p.add_argument('--export-onnx', action='store_true', help='Optional ONNX export + checker; requested failure fails Stage06')
    p.add_argument('--onnx-opset', type=int, default=None)
    p.add_argument('--overwrite', action='store_true', help='Overwrite ONLY Stage06 outputs; invalidate gates 06..10')
    # Internal subprocess protocol. Not user-facing Stage06 configuration.
    p.add_argument('--_child-reload', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--_child-checkpoint', default=None, help=argparse.SUPPRESS)
    p.add_argument('--_child-result', default=None, help=argparse.SUPPRESS)
    p.add_argument('--_child-expected-sha', default=None, help=argparse.SUPPRESS)
    p.add_argument('--_child-imgsz', nargs=2, type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument('--_child-device', default='cpu', help=argparse.SUPPRESS)
    return p


def initialize_cuda_memory_stats(torch, device) -> None:
    """Initialize the selected CUDA context before querying/resetting allocator stats.

    torch.cuda.is_available() and torch.cuda.device_count() may work before CUDA
    has been initialized. In that state, reset_peak_memory_stats(device) can
    raise RuntimeError('Invalid device argument'). This must not bypass an
    invalid GPU or suppress CUDA errors: allocate on the selected device first.
    """
    if device.type != 'cuda':
        return
    torch.cuda.set_device(device)
    torch.cuda.init()
    probe = torch.empty((1,), device=device)
    if probe.device != device:
        raise RuntimeError(f'CUDA probe allocated on {probe.device}, expected {device}')
    del probe
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _load_step(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    if not path.is_file():
        raise FileNotFoundError(f'Stage06 requires unchanged predecessor source: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_stage05_legacy_bridge(out: Path, dataset: dict, gate5: dict) -> dict | None:
    """Load only the Stage05-attested legacy waiver, never create one in Stage06."""
    prov = gate5.get('provenance', {})
    if 'dataset_legacy_bridge_sha256' not in prov:
        return None
    path = out / 'stages/05/05_dataset_legacy_bridge.json'
    if prov['dataset_legacy_bridge_path'] != str(path):
        raise RuntimeError('Stage05 legacy bridge path changed')
    if not path.is_file() or sha256_file(path) != prov['dataset_legacy_bridge_sha256']:
        raise RuntimeError('Stage05 legacy bridge missing or modified')
    if prov.get('artifact_hashes', {}).get('dataset_legacy_bridge') != sha256_file(path):
        raise RuntimeError('Stage05 gate bridge artifact SHA mismatch')
    if gate5.get('artifacts', {}).get('dataset_legacy_bridge') != str(path):
        raise RuntimeError('Stage05 gate bridge artifact path mismatch')
    g0_path = gate_file(out, '00', 'check_environment')
    bridge = read_json(path)
    validate_legacy_bridge(dataset, read_json(g0_path), sha256_file(g0_path), bridge)
    return bridge


def verify_lineage(out: Path, request: dict, data: dict) -> tuple[dict, dict, dict, dict, Path, Path]:
    """Fail closed on stale ancestor gates, checksum mismatches or input changes."""
    step4 = _load_step('04_plan_structured_pruning.py', 'stage04_for06')
    step5 = _load_step('05_apply_structured_pruning.py', 'stage05_for06')
    gate_path = gate_file(out, '05', 'apply_structured_pruning')
    if not gate_path.is_file():
        raise RuntimeError('Stage05 gate missing')
    gate = read_json(gate_path)
    legacy_bridge = load_stage05_legacy_bridge(out, data, gate)
    g04, plan, graph, source, receipts = step5.verify_stage04(
        out, request, data, step4, legacy_bridge=legacy_bridge)
    if gate.get('step') != '05_apply_structured_pruning' or gate.get('status') != 'PASS' or not gate.get('checks') or not all(x is True for x in gate['checks'].values()):
        raise RuntimeError('Stage05 must PASS with all mandatory checks true')
    pr = gate.get('provenance', {})
    if pr.get('contract_sha256') != hash_object(request):
        raise RuntimeError('Stage05 contract does not match Stage06')
    if pr.get('stage04_gate_sha256') != receipts['stage04_gate_sha256'] or pr.get('stage04_plan_sha256') != receipts['stage04_plan_sha256']:
        raise RuntimeError('Stage04 plan/gate changed after Stage05')
    if pr.get('stage04_source_sha256') != sha256_file(Path(__file__).with_name('04_plan_structured_pruning.py')):
        raise RuntimeError('Stage04 source changed since Stage05')
    if pr.get('stage02_source_sha256') != sha256_file(Path(__file__).with_name('02_analyze_pruning_graph.py')):
        raise RuntimeError('Stage02 safety source changed since Stage05')
    art = gate.get('artifacts', {})
    hashes = pr.get('artifact_hashes', {})
    required = {
        'checkpoint': out / 'stages/05/05_pruned_raw.pt',
        'trace': out / 'stages/05/05_pruning_trace.csv',
        'graph_signature': out / 'stages/05/05_graph_signature.json',
        'report': out / 'stages/05/05_report.json',
    }
    for k, path in required.items():
        if not path.is_file() or hashes.get(k) != sha256_file(path):
            raise RuntimeError(f'Stage05 artifact missing or modified: {k}: {path}')
        if art.get(k) != str(path):
            raise RuntimeError(f'Stage05 artifact path mismatch: {k}')
    checkpoint = required['checkpoint']
    if art.get('checkpoint_sha256') != sha256_file(checkpoint):
        raise RuntimeError('Stage05 checkpoint artifact SHA differs')
    report = read_json(required['report'])
    if report.get('output_pruned_raw_sha256') != sha256_file(checkpoint):
        raise RuntimeError('Stage05 report checkpoint SHA differs')
    if report.get('decisions_applied') != len(plan.get('decisions', [])):
        raise RuntimeError('Stage05 actual decision count differs from Stage04 plan')
    if report.get('pruned_params') != plan['projected']['params']:
        raise RuntimeError('Stage05 report parameters differ from Stage04 target')
    if not math.isclose(float(report['pruned_macs']), float(plan['projected']['macs']), rel_tol=1e-6, abs_tol=512):
        raise RuntimeError('Stage05 report MACs differ from Stage04 simulation')
    return gate, plan, read_json(required['graph_signature']), report, source, checkpoint


def check_argument_values(args, hw):
    if hw[0] != hw[1]:
        raise ValueError('Stage06 requires square imgsz for Ultralytics validation and TRAIN loss tests')
    if args.batch < 1 or args.workers < 0 or args.backward_batches < 1 or args.backward_batch_size < 1 or args.val_batch < 1:
        raise ValueError('Invalid train-backward/validation batch and worker arguments')
    if args.val_max_images < 0 or args.val_seed < 0 or args.reload_timeout < 30:
        raise ValueError('Invalid subset seed/count or reload timeout')
    if args.recovery_map_drop < 0 or args.recovery_map_drop > 1:
        raise ValueError('--recovery-map-drop must be between 0 and 1')
    if args.max_cuda_peak_mib < 0:
        raise ValueError('--max-cuda-peak-mib must be >= 0')
    if args.val_max_det is not None and args.val_max_det < 1:
        raise ValueError('--val-max-det must be positive')
    for k in ('val_conf', 'val_iou'):
        v = getattr(args, k)
        if v is not None and not 0 <= v <= 1:
            raise ValueError(f'{k} must be in [0,1]')
    if args.onnx_opset is not None and args.onnx_opset < 11:
        raise ValueError('--onnx-opset must be >=11')


def tensor_dtype_summary(net) -> dict:
    return {'parameter_tensors': dict(Counter(str(p.dtype) for p in net.parameters())),
            'buffer_tensors': dict(Counter(str(b.dtype) for b in net.buffers())),
            'parameter_bytes': sum(p.numel()*p.element_size() for p in net.parameters()),
            'buffer_bytes': sum(b.numel()*b.element_size() for b in net.buffers()),
            'buffer_elements': sum(b.numel() for b in net.buffers())}


def check_tensor_finiteness(net):
    import torch
    count = 0
    for name, tensor in list(net.named_parameters()) + list(net.named_buffers()):
        if tensor.is_floating_point() or tensor.is_complex():
            if not bool(torch.isfinite(tensor).all().item()):
                raise RuntimeError(f'Non-finite tensor in saved pruned model: {name}')
        count += 1
    return count


def eval_output_schema(net, hw, device) -> list[dict]:
    """Native model eval output ABI (not postprocessed Results objects)."""
    import torch
    was_train = net.training
    net.eval()
    try:
        with torch.inference_mode():
            y = net(torch.zeros((1,3,*hw),dtype=torch.float32,device=device))
        items = forward_tensors(y)
        if not items or not all(i['finite'] for i in items):
            raise RuntimeError('Eval output missing or nonfinite')
        return [{'path': x['path'], 'shape': x['shape'], 'dtype': x['dtype']} for x in items]
    finally:
        net.train(was_train)


def verify_graph_shapes(net, source_net, graph_sig, plan, stage5, stage4, stage2, hw, device) -> dict:
    """Compare exact Stage05 graph signature + original/model head and native outputs."""
    conv = stage5.current_conv_shapes(net)
    bn = stage5.current_bn_shapes(net)
    if not conv or not bn:
        raise RuntimeError('Pruned architecture lacks Conv/BN modules')
    if conv != graph_sig.get('pruned_convs'):
        raise RuntimeError('Conv dimensions after reload differ from Stage05 graph signature')
    if bn != graph_sig.get('pruned_batchnorm'):
        raise RuntimeError('BN dimensions after reload differ from Stage05 graph signature')
    source_conv = stage5.current_conv_shapes(source_net)
    if source_conv != graph_sig.get('baseline_convs'):
        raise RuntimeError('Stage05 original Conv signature differs from current unpruned checkpoint')
    if stage5.current_bn_shapes(source_net) != graph_sig.get('baseline_batchnorm'):
        raise RuntimeError('Original BN signature differs from recorded Stage05 baseline')
    if set(conv) != set(source_conv) or set(bn) != set(graph_sig['baseline_batchnorm']):
        raise RuntimeError('Unexpected module removal/creation')
    if model_head_metadata(net) != model_head_metadata(source_net) or model_head_metadata(net) != graph_sig['baseline_head']:
        raise RuntimeError('Detection head task, class count, end2end metadata changed')
    altered = stage5.changed_shapes(source_conv, conv)
    if not altered or altered != graph_sig.get('changed_convs'):
        raise RuntimeError('Physical Conv changes missing / differ from Stage05')
    stage5.verify_stage04_alignment_from_snapshot(source_conv, net, stage2, stage4,
                                                   plan['channel_align'], plan['min_channels'])
    # BN widths may be static under user metadata; ensure no BN became invalid.
    for name, count in bn.items():
        mod = dict(net.named_modules())[name]
        if mod.weight is not None and mod.weight.numel() != count:
            raise RuntimeError(f'BN feature/weight width mismatch: {name}')
        if mod.running_mean is not None and mod.running_mean.numel() != count:
            raise RuntimeError(f'BN running_mean width mismatch: {name}')
    # Check both outputs against SOURCE and Stage05 training native ABI.
    train_out = stage5.detection_output_schema(net, hw, device)
    if train_out != graph_sig.get('baseline_train_output_schema'):
        raise RuntimeError('Pruned native training output ABI changed')
    eval_src = eval_output_schema(source_net, hw, device)
    eval_pruned = eval_output_schema(net, hw, device)
    if eval_src != eval_pruned:
        raise RuntimeError('Pruned native inference output ABI changed')
    return {'conv_layers': len(conv), 'bn_layers': len(bn),
            'conv_modified': len(altered), 'changes': altered,
            'bn_changed': sum(bn[n] != graph_sig['baseline_batchnorm'][n] for n in bn),
            'head': model_head_metadata(net), 'train_output_abi': train_out,
            'eval_output_abi': eval_pruned,
            'channel_align': plan['channel_align'], 'min_channels': plan['min_channels']}


def check_lineage_trace(out: Path, plan: dict, graph_sig: dict, report: dict) -> dict:
    """Audit original/current index mapping without replaying physical pruning."""
    csv_path = out/'stages/05/05_pruning_trace.csv'
    with csv_path.open(newline='', encoding='utf-8') as fh:
        trace = list(csv.DictReader(fh))
    decisions = plan['decisions']
    if len(trace) != len(decisions) or len(trace) != report['decisions_applied']:
        raise RuntimeError('Physical pruning CSV row count does not equal planned decisions')
    maps: dict[str, list[int]] = {}
    for n,(row,d) in enumerate(zip(trace, decisions),1):
        if int(row['decision']) != n or int(row['iterative_stage']) != int(d['iterative_stage']):
            raise RuntimeError(f'Trace decision order/stage wrong at {n}')
        if row['status'] != 'PHYSICAL_PRUNE_VERIFIED' or row['root'] != d['root'] or row['group_id'] != d['group_id']:
            raise RuntimeError(f'Trace decision/root/status wrong at {n}')
        original = [int(x) for x in row['original_indices'].split(',') if x.strip()]
        current = [int(x) for x in row['current_indices'].split(',') if x.strip()]
        if original != d['original_indices'] or current != d['current_indices']:
            raise RuntimeError(f'Trace exact pruning index mismatch at {n}')
        if int(row['params_before']) != int(d['actual_params_before']) or int(row['params_after']) != int(d['actual_params_after']):
            raise RuntimeError(f'Trace resource mismatch at {n}')
        gid = d['group_id']
        surviving = maps.setdefault(gid,list(range(d['channels_before'])))
        if len(surviving) != d['channels_before'] or any(i < 0 or i>=len(surviving) for i in current):
            raise RuntimeError(f'Trace index lineage out of range at {n}')
        if [surviving[i] for i in current] != original:
            raise RuntimeError(f'Trace original->current lineage mapping wrong at {n}')
        original_set = set(original)
        maps[gid] = [o for o in surviving if o not in original_set]
    stage05_lineage = graph_sig.get('root_lineages')
    if stage05_lineage != maps:
        raise RuntimeError('Final channel index lineage does not match Stage05 graph signature')
    return {'decision_count':len(trace),'stage_count':len(set(d['iterative_stage'] for d in decisions)),
            'unique_roots':len(maps),'removed_root_channels':sum(len(d['original_indices']) for d in decisions)}


def profile_resources(net, tp, stage4, hw, device, expected_params:int, expected_macs:float) -> dict:
    measured_params,measured_macs = stage4._profile(net,tp,hw,device)
    actual_params = sum(p.numel() for p in net.parameters())
    if actual_params!=expected_params or measured_params!=actual_params:
        raise RuntimeError(f'Params mismatch measured={actual_params}, profiler={measured_params}, expected={expected_params}')
    if not math.isclose(float(measured_macs), float(expected_macs),rel_tol=1e-6,abs_tol=512):
        raise RuntimeError(f'MAC mismatch measured={measured_macs}, expected={expected_macs}')
    return {'params':actual_params,'macs':measured_macs,'gmacs':measured_macs/1e9,
            'gflops_nominal_2_per_mac':2*measured_macs/1e9,
            'mac_method':'torch_pruning.utils.count_ops_and_params'}


def _train_loader_stage06(source_ckpt, data_path, net, args, dataset, log):
    """Native Ultralytics TRAIN loader; scratch files stay under Stage06."""
    from ultralytics.models.yolo.detect import DetectionTrainer
    settings=dict(model=str(source_ckpt),data=str(data_path),imgsz=args.imgsz[0],
        task='detect',device=str(args.device),batch=args.backward_batch_size,
        workers=args.workers,epochs=1,fraction=1.0,cache=False,rect=False,
        multi_scale=0.0,mosaic=0.0,mixup=0.0,copy_paste=0.0,
        scale=0.0,translate=0.0,degrees=0.0,shear=0.0,perspective=0.0,
        fliplr=0.0,flipud=0.0,hsv_h=0.0,hsv_s=0.0,hsv_v=0.0,
        seed=args.val_seed,deterministic=True,
        project=str(resolved(args.output)/'stages/06'),name='backward_loader',
        exist_ok=True,save=False,plots=False,val=False)
    trainer=DetectionTrainer(overrides=settings)
    trainer.model=net
    trainer.set_model_attributes()
    train_paths=dataset['splits']['train']
    path_arg=train_paths[0] if len(train_paths)==1 else train_paths
    loader=trainer.get_dataloader(path_arg,batch_size=args.backward_batch_size,rank=-1,mode='train')
    if len(loader)<args.backward_batches:
        raise RuntimeError(f'Only {len(loader)} TRAIN batches; requested {args.backward_batches}')
    log.info('Stage06 backward uses TRAIN only; %d batches, batch size %d',args.backward_batches,args.backward_batch_size)
    return trainer,loader


def backward_smoke(net, source_ckpt, data_path, dataset, args, hw, device, log) -> dict:
    """Official labeled TRAIN dataloader, Ultralytics loss and genuine backward."""
    import torch
    stage3 = _load_step('03_collect_gradient_importance.py', 'stage03_for06_backward')
    trainer,loader=_train_loader_stage06(source_ckpt,data_path,net,args,dataset,log)
    net.requires_grad_(True).train()
    changed_roots=[]
    # Selected from changed CONV modules, not all graph roots (no per-root nonzero guarantee).
    nonzero_batches=0
    rows=[]
    for n,batch in enumerate(loader,1):
        if n>args.backward_batches: break
        b=trainer.preprocess_batch(batch)
        if not isinstance(b,dict) or not all(k in b for k in ('img','cls','bboxes','batch_idx')):
            raise RuntimeError('Official train loader returned incomplete labeled batch')
        for k in ('cls','bboxes','batch_idx'):
            if b[k].device != b['img'].device:
                b[k] = b[k].to(b['img'].device)
        if tuple(b['img'].shape[2:])!=tuple(hw) or b['img'].shape[1]!=3:
            raise RuntimeError('TRAIN image shape invalid for native FP32 backward')
        bs=int(b['img'].shape[0]); n_gt=int(b['cls'].numel())
        net.zero_grad(set_to_none=True)
        with torch.enable_grad():
            raw=net(b)
            loss,components=stage3.unpack_detection_loss(raw,torch)
            normalized=loss/bs
            normalized.backward()
        grads=[p.grad for p in net.parameters() if p.requires_grad and p.grad is not None]
        if not grads or any(not bool(torch.isfinite(g).all().item()) for g in grads):
            raise RuntimeError('Detection backward produced missing or nonfinite gradients')
        nnz=sum(bool(torch.any(g != 0).item()) for g in grads)
        if nnz<1:
            raise RuntimeError('All pruned detector gradients are zero')
        nonzero_batches+=1
        rows.append({'batch':n,'images':bs,'gt_objects':n_gt,
                     'loss':float(loss.detach().item()),'loss_per_image':float(normalized.detach().item()),
                     'grad_tensors_nonzero':nnz,'grad_tensors_total':len(grads)})
        log.info('Stage06 TRAIN backward %d/%d: loss/image=%.6f GT=%d nonzero=%d/%d',
                 n,args.backward_batches,rows[-1]['loss_per_image'],n_gt,nnz,len(grads))
    net.zero_grad(set_to_none=True)
    if len(rows)!=args.backward_batches or sum(r['gt_objects'] for r in rows)==0:
        raise RuntimeError('Not enough real labeled TRAIN batches for backward test')
    return {'batches':rows,'images':sum(r['images'] for r in rows),
            'gt_objects':sum(r['gt_objects'] for r in rows),
            'all_nonzero_grad':nonzero_batches==args.backward_batches,
            'loss_source':'REAL_ULTRALYTICS_DETECTION_LOSS_TRAIN_ONLY'}


def _val_images(dataset:dict,limit:int,seed:int) -> tuple[list[Path],int]:
    paths=[]
    for raw in dataset['splits']['val']:
        p=Path(raw)
        if p.is_dir():
            paths.extend(f.resolve() for f in p.rglob('*') if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES)
        else:
            for line in p.read_text(encoding='utf-8').splitlines():
                item=line.strip()
                if not item: continue
                loc=Path(item)
                if loc.is_absolute():
                    candidate=loc.resolve()
                else:
                    opts=[(p.parent/loc).resolve(), (Path(dataset['root'])/loc).resolve()]
                    candidate=next((opt for opt in opts if opt.is_file()),opts[0])
                if not candidate.is_file():
                    raise FileNotFoundError(f'VAL list references nonexistent image: {item} ({p})')
                paths.append(candidate)
    paths=sorted(set(paths),key=str)
    if not paths:
        raise RuntimeError('No VAL images found')
    total=len(paths)
    if 0<limit<total:
        rng=random.Random(seed)
        paths=sorted(rng.sample(paths,limit),key=str)
    return paths,total


def prepare_validation_subset(out:Path,dataset:dict,data_path:Path,args,hw) -> dict:
    """Create deterministic VAL-only smoke yaml/TXT; original YAML is immutable."""
    selected,total=_val_images(dataset,args.val_max_images,args.val_seed)
    stage_dir=out/'stages/06'
    stage_dir.mkdir(parents=True, exist_ok=True)
    import yaml
    subset_file=stage_dir/'06_val_subset.txt'
    subset_file.write_text(''.join(str(p)+'\n' for p in selected),encoding='utf-8')
    if len(selected)==total:
        return {'yaml_path':str(data_path),'manifest':str(subset_file),'total_images':total,
                'selected_images':len(selected),'mode':'FULL_VAL','subset_sha256':sha256_file(subset_file)}
    original=yaml.safe_load(data_path.read_text(encoding='utf-8'))
    if not isinstance(original,dict):
        raise RuntimeError('Malformed original validation YAML')
    payload=dict(original)
    payload['path']=dataset['root']
    payload['val']=str(subset_file.resolve())
    # Preserve train/test keys as supplied by user; do not load them during val.
    smoke_yaml=stage_dir/'06_smoke_val.yaml'
    smoke_yaml.write_text(yaml.safe_dump(payload,sort_keys=False,allow_unicode=True),encoding='utf-8')
    return {'yaml_path':str(smoke_yaml),'manifest':str(subset_file),'total_images':total,
            'selected_images':len(selected),'mode':'VAL_SUBSET',
            'subset_sha256':sha256_file(subset_file),'seed':args.val_seed}


def validation_settings(out:Path,args,profile:dict,stage5_gate:dict,subset:dict) -> dict:
    config=profile.get('input',{})
    # The official Stage01 profile often uses input or validation.validation_config.
    config2=profile.get('validation',{}).get('validation_config',{})
    merged={**config,**config2}
    def pick(name,override,default):
        return override if override is not None else merged.get(name,default)
    conf=float(pick('conf',args.val_conf,0.001))
    iou=float(pick('iou',args.val_iou,0.7))
    maxdet=int(pick('max_det',args.val_max_det,1000))
    nmsfree=bool(pick('nms_free',args.nms_free,False))
    if nmsfree and stage5_gate.get('provenance',{}).get('model_family')!='yolo26':
        raise ValueError('--nms-free is supported only for YOLO26')
    if not (0<=conf<=1 and 0<=iou<=1) or maxdet<1:
        raise RuntimeError('Invalid baseline validation policy')
    return {'data':subset['yaml_path'],'split':'val','imgsz':args.imgsz[0],
            'device':args.device,'workers':args.workers,'batch':args.val_batch,
            'conf':conf,'iou':iou,'max_det':maxdet,'rect':False,'plots':False,
            'half':False,'verbose':True,'nms':nmsfree}


def extract_smoke_metrics(metrics,names:dict)->dict:
    """Allow unrepresented classes in small VAL subset; never invent AP for them."""
    box=metrics.box
    present=[int(v) for v in box.ap_class_index]
    if len(present)!=len(set(present)) or any(str(cid) not in names for cid in present):
        raise RuntimeError('Validation produced invalid class mapping')
    result=[]
    ap_rows=getattr(box,'all_ap',None)
    for pos,cid in enumerate(present):
        p,r,ap50,ap5095=[float(v) for v in box.class_result(pos)]
        row_ap=ap_rows[pos] if ap_rows is not None else None
        ap75=float(row_ap[5]) if row_ap is not None and len(row_ap)>5 else None
        f1=2*p*r/(p+r+1e-12)
        row={'class_id':cid,'class_name':names[str(cid)],'precision':p,'recall':r,
             'f1':f1,'ap50':ap50,'ap75':ap75,'ap50_95':ap5095}
        if any(v is not None and not math.isfinite(v) for v in row.values() if isinstance(v,(int,float))):
            raise RuntimeError(f'Nonfinite val metrics for class {cid}')
        result.append(row)
    if not result:
        raise RuntimeError('No evaluated classes in validation subset; cannot certify useful detection metrics')
    summ={'precision':float(box.mp),'recall':float(box.mr),'map50':float(box.map50),
          'map75':float(box.map75),'map50_95':float(box.map)}
    if any(not math.isfinite(x) or x<0 or x>1.00001 for x in summ.values()):
        raise RuntimeError(f'Invalid overall VAL metrics: {summ}')
    result.sort(key=lambda x:x['class_id'])
    return {'overall':summ,'per_class':result,'classes_observed':len(present),
            'classes_not_observed':[int(k) for k in names if int(k) not in present],
            'validator_speed_ms':getattr(metrics,'speed',{})}


def run_validation(source:Path,pruned:Path,settings:dict,subset:dict,names:dict,args,logger)->dict:
    from ultralytics import YOLO
    import torch
    runconfig=dict(settings)
    nms=runconfig.pop('nms')
    if nms: runconfig['nms']=False
    report={'subset':subset,'settings':{**settings,'nms_free':nms},
            'accuracy_status':'NOT_ASSESSED','accuracy_is_structural_gate':False}
    paths=[('source',source),('pruned',pruned)] if args.compare_source_val else [('pruned',pruned)]
    for label,path in paths:
        runner=YOLO(str(path))
        # Use a newly loaded model each time: no optimizer or artifact modification.
        kw={**runconfig,'project':str(resolved(args.output)/'stages/06'),
            'name':f'validation_{label}','exist_ok':True}
        logger.info('Validation on %s: %d VAL images',label,subset['selected_images'])
        metrics=runner.val(**kw)
        report[label]=extract_smoke_metrics(metrics,names)
        del runner,metrics
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if args.compare_source_val:
        bm=report['source']['overall']['map50_95']
        pm=report['pruned']['overall']['map50_95']
        drop=bm-pm
        report['map50_95_drop_source_minus_pruned']=drop
        report['recovery_threshold_map50_95_drop']=args.recovery_map_drop
        report['accuracy_status']='RECOVERY_NEEDED' if drop>args.recovery_map_drop else 'WITHIN_SMOKE_TOLERANCE'
    else:
        report['accuracy_status']='NOT_ASSESSED_NO_SAME_SUBSET_BASELINE'
    return report


def subprocess_reload(script:Path,ckpt:Path,sha:str,hw,device:str,timeout:int,stage_dir:Path)->dict:
    """New Python interpreter, NOT in-process YOLO reload from Stage05."""
    receipt=stage_dir/'06_reload_receipt.json'
    receipt.unlink(missing_ok=True)
    cmd=[sys.executable,str(script),'--_child-reload','--_child-checkpoint',str(ckpt),
         '--_child-result',str(receipt),'--_child-expected-sha',sha,
         '--_child-imgsz',str(hw[0]),str(hw[1]),'--_child-device',str(device)]
    proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8',timeout=timeout,check=False)
    if proc.returncode!=0 or not receipt.is_file():
        raise RuntimeError(f'Independent process reload failed (returncode={proc.returncode}): stdout={proc.stdout[-2000:]} stderr={proc.stderr[-3000:]}')
    doc=read_json(receipt)
    if doc.get('status')!='PASS' or doc.get('pid')==os.getpid() or doc.get('checkpoint_sha256')!=sha:
        raise RuntimeError('Independent process reload failed attestation')
    return doc


def _child_main(args)->int:
    """A fresh process verifies actual Ultralytics load + both native forwards."""
    result=Path(args._child_result)
    try:
        import torch
        from ultralytics import YOLO
        ckpt=resolved(args._child_checkpoint)
        if sha256_file(ckpt)!=args._child_expected_sha:
            raise RuntimeError('Checkpoint hash changed before independent reload')
        device=torch.device(args._child_device)
        y=YOLO(str(ckpt))
        net=y.model.to(device).float()
        hw=tuple(args._child_imgsz)
        import importlib.util
        mod=_load_step('05_apply_structured_pruning.py','stage05_for06_child')
        train_sig=mod.detection_output_schema(net,hw,device)
        eval_sig=eval_output_schema(net,hw,device)
        doc={'status':'PASS','pid':os.getpid(),'checkpoint_sha256':sha256_file(ckpt),
             'parameter_count':sum(p.numel() for p in net.parameters()),
             'conv_shapes':mod.current_conv_shapes(net),'bn_shapes':mod.current_bn_shapes(net),
             'train_output_abi':train_sig,'eval_output_abi':eval_sig,'metadata':model_head_metadata(net)}
        write_json(result,doc)
        return 0
    except Exception:
        write_json(result,{'status':'FAIL','pid':os.getpid(),'traceback':traceback.format_exc()})
        return 1


def optional_onnx_export(ckpt:Path,stage_dir:Path,hw,args)->dict:
    if not args.export_onnx:
        return {'status':'NOT_REQUESTED'}
    from ultralytics import YOLO
    import onnx
    # Export from a temporary copy placed under Stage06, never emit new files
    # into Stage05's artifact directory.
    temp_pt=stage_dir/'06_onnx_input.pt'
    shutil.copy2(ckpt,temp_pt)
    model=YOLO(str(temp_pt))
    kw={'format':'onnx','imgsz':hw[0],'device':args.device,'half':False,
        'dynamic':False,'simplify':False}
    if args.onnx_opset is not None: kw['opset']=args.onnx_opset
    try:
        result=model.export(**kw)
        source=Path(str(result))
        if not source.is_file() or source.stat().st_size==0:
            raise RuntimeError('Ultralytics ONNX export did not produce a nonempty file')
        target=stage_dir/'06_pruned_pre_recovery.onnx'
        if source.resolve()!=target.resolve():
            shutil.move(str(source),str(target))
    finally:
        temp_pt.unlink(missing_ok=True)
    onnx.checker.check_model(str(target))
    return {'status':'ONNX_CHECKER_PASS','path':str(target),'sha256':sha256_file(target),
            'size_bytes':target.stat().st_size,'scope':'ONNX structural validation only; numerical/runtime equivalence not certified'}


def run(args:argparse.Namespace)->int:
    out=resolved(args.output)
    out.mkdir(parents=True,exist_ok=True)
    stage_dir=out/'stages/06'
    stage_dir.mkdir(parents=True,exist_ok=True)
    logger=setup_log(out,STAGE,NAME)
    gatepath=gate_file(out,STAGE,NAME)
    if gatepath.is_file() and not args.overwrite:
        logger.error('Stage06 gate already exists; use --overwrite to replace: %s',gatepath)
        return 2
    checks=dict.fromkeys([
        'cli_valid','predecessor_gates_current','checkpoint_identity_verified',
        'lineage_and_index_mapping_valid','weights_and_buffers_finite',
        'exact_conv_bn_shapes_and_alignment','native_train_and_eval_abi_preserved',
        'params_macs_match_stage05','true_detection_loss_backward',
        'independent_python_reload','validation_metrics_finite',
        'memory_and_optional_export_policy','artifacts_saved'],False)
    errors=[];warnings=[];provenance={};artifacts={}
    report={'step':f'{STAGE}_{NAME}','status':'PRELIMINARY','created_utc':now_utc()}
    valreport={'status':'NOT_RUN'}
    audit={'step':f'{STAGE}_{NAME}','status':'NOT_VERIFIED','created_utc':now_utc()}
    try:
        if args.overwrite:
            for file in stage_dir.glob('06_*'):
                if file.is_file():file.unlink()
            for path in (out/'gates').glob('*.json'):
                if path.name[:2].isdigit() and int(path.name[:2])>=6:
                    path.unlink()
            logger.warning('--overwrite invalidated gates 06..10; prior stages untouched')
        hw=make_imgsz(args.imgsz)
        check_argument_values(args,hw)
        checks['cli_valid']=True
        data_path=resolved(args.data)
        dataset=read_dataset(data_path,args.expected_classes)
        request=contract(args.model,data_path,hw,args.device,args.batch,args.workers,args.family,args.expected_classes)
        gate5,plan,graphsig,report5,source,checkpoint=verify_lineage(out,request,dataset)
        checks['predecessor_gates_current']=True
        checks['checkpoint_identity_verified']=True
        audit['checkpoint']={'source_path':str(source),'pruned_path':str(checkpoint),
          'source_bytes':source.stat().st_size,'pruned_bytes':checkpoint.stat().st_size,
          'source_sha256':sha256_file(source),'pruned_sha256':sha256_file(checkpoint),
          'size_ratio':checkpoint.stat().st_size/source.stat().st_size}
        legacy_bridge = load_stage05_legacy_bridge(out, dataset, gate5)
        audit['provenance']={'stage05_gate_sha256':sha256_file(gate_file(out,'05','apply_structured_pruning')),
          'stage04_plan_sha256':gate5['provenance']['stage04_plan_sha256'],
          'contract_sha256':hash_object(request),'dataset_fingerprint':dataset_fingerprint(dataset)}
        if legacy_bridge is None:
            if audit['provenance']['dataset_fingerprint']['manifest_sha256']!=gate5['provenance']['dataset_manifest']['manifest_sha256']:
                raise RuntimeError('Dataset inventory changed since Stage05')
        else:
            audit['provenance']['dataset_legacy_bridge_sha256'] = gate5['provenance']['dataset_legacy_bridge_sha256']
            audit['provenance']['non_cache_content_sha256'] = legacy_bridge['non_cache_inventory']['manifest_sha256']
            if legacy_bridge['policy'].startswith('EXPLICIT_LEGACY'):
                warnings.append('Legacy Stage00 dataset stat hash waived by explicit Stage05 attestation; current non-cache bytes fully checked')
        provenance={**gate5['provenance'],'stage05_gate_sha256':audit['provenance']['stage05_gate_sha256'],
                    'stage05_checkpoint_sha256':audit['checkpoint']['pruned_sha256'],
                    'stage06_config':{'backward_batches':args.backward_batches,
                                     'backward_batch_size':args.backward_batch_size,
                                     'val_max_images':args.val_max_images,'val_seed':args.val_seed,
                                     'val_batch':args.val_batch,'compare_source_val':args.compare_source_val,
                                     'recovery_map_drop':args.recovery_map_drop,
                                     'export_onnx':args.export_onnx}}
        lin=check_lineage_trace(out,plan,graphsig,report5)
        checks['lineage_and_index_mapping_valid']=True
        audit['lineage']=lin
        import torch
        import torch_pruning as tp
        from ultralytics import YOLO
        step5=_load_step('05_apply_structured_pruning.py','stage05_for06_verify')
        step4=_load_step('04_plan_structured_pruning.py','stage04_for06_verify')
        step2=_load_step('02_analyze_pruning_graph.py','stage02_for06_verify')
        device=select_device(args.device)
        initialize_cuda_memory_stats(torch, device)
        # Only fresh loads from immutable checkpoints; never prune or optimizer.step.
        original=YOLO(str(source))
        pruned=YOLO(str(checkpoint))
        if original.task!='detect' or pruned.task!='detect':
            raise RuntimeError('Stage06 supports Ultralytics object detection only')
        if len(original.names)!=dataset['nc'] or len(pruned.names)!=dataset['nc']:
            raise RuntimeError('Class count or label metadata altered during pruning')
        source_net=original.model.to(device).float()
        net=pruned.model.to(device).float()
        if sum(p.numel() for p in net.parameters())!=report5['pruned_params']:
            raise RuntimeError('Reloaded pruned model parameter count wrong')
        cnt=check_tensor_finiteness(net)
        checks['weights_and_buffers_finite']=True
        audit['tensor_summary']={'pruned':tensor_dtype_summary(net),
                                  'source_loaded':tensor_dtype_summary(source_net),
                                  'finite_parameter_and_buffer_tensors_checked':cnt}
        structure=verify_graph_shapes(net,source_net,graphsig,plan,step5,step4,step2,hw,device)
        checks['exact_conv_bn_shapes_and_alignment']=True
        checks['native_train_and_eval_abi_preserved']=True
        audit['graph']=structure
        # Reconstruct autograd DepGraph from the serialized, independently loaded
        # pruned architecture. This is a structural audit, NEVER a prune action.
        depgraph=step4.build_graph(net,tp,step2,hw,device)
        dep_nodes=len(depgraph.module2node)
        if dep_nodes<1:
            raise RuntimeError('Empty autograd DepGraph after Stage05')
        audit['graph']['autograd_depgraph']={'status':'REBUILT','module_nodes':dep_nodes,
                 'autograd_enabled':True,'pruning_applied':False}
        del depgraph
        profile=profile_resources(net,tp,step4,hw,device,report5['pruned_params'],report5['pruned_macs'])
        checks['params_macs_match_stage05']=True
        audit['resources']={'source_params':report5['baseline_params'],
            'source_macs':report5['baseline_macs'],'pruned':profile,
            'params_reduction':1-profile['params']/report5['baseline_params'],
            'macs_reduction':1-profile['macs']/report5['baseline_macs']}
        logger.info('Stage06 structural audit PASS: params=%d (%.2f%% less), MACs=%.3fG (%.2f%% less), modified Conv=%d',
             profile['params'],audit['resources']['params_reduction']*100,
             profile['gmacs'],audit['resources']['macs_reduction']*100,structure['conv_modified'])
        # True TRAIN detection loss; full model live and backward, no weight update.
        back=backward_smoke(net,source,data_path,dataset,args,hw,device,logger)
        audit['backward']=back
        checks['true_detection_loss_backward']=True
        with (stage_dir/'06_backward_log.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['batch','images','gt_objects','loss','loss_per_image','grad_tensors_nonzero','grad_tensors_total'])
            writer.writeheader();writer.writerows(back['batches'])
        # Independent process checks exactly the final Stage05 SHA, reloading both train/eval paths.
        del net,source_net,pruned,original
        if device.type=='cuda':torch.cuda.empty_cache()
        childdev=args.device if args.reload_device=='same' else 'cpu'
        receipt=subprocess_reload(Path(__file__).resolve(),checkpoint,
                                   audit['checkpoint']['pruned_sha256'],hw,childdev,
                                   args.reload_timeout,stage_dir)
        if receipt['parameter_count']!=report5['pruned_params'] or receipt['conv_shapes']!=graphsig['pruned_convs'] or receipt['bn_shapes']!=graphsig['pruned_batchnorm']:
            raise RuntimeError('Independent interpreter architecture differs from Stage05 signature')
        if receipt['train_output_abi']!=graphsig['baseline_train_output_schema'] or receipt['eval_output_abi']!=structure['eval_output_abi']:
            raise RuntimeError('Independent interpreter output ABI differs from verified main-process model')
        checks['independent_python_reload']=True
        audit['reload']={'status':'PASS_FRESH_PROCESS','pid':receipt['pid'],
                          'checkpoint_sha256':receipt['checkpoint_sha256'],
                          'module_shapes_match':True,'train_eval_abi_match':True,
                          'python':sys.executable,'device':childdev}
        # Full or deterministic VAL-only subset. We validate SOURCE on same images.
        subset=prepare_validation_subset(out,dataset,data_path,args,hw)
        baseline_profile=read_json(out/'stages/01/01_profile.json')
        valsettings=validation_settings(out,args,baseline_profile,gate5,subset)
        valreport=run_validation(source,checkpoint,valsettings,subset,dataset['names'],args,logger)
        valreport['status']='VAL_COMPLETE'
        logger.info('Stage06 VAL smoke (%s, %d/%d): pruned mAP50-95=%.5f; accuracy_status=%s',
            subset['mode'],subset['selected_images'],subset['total_images'],
            valreport['pruned']['overall']['map50_95'],valreport['accuracy_status'])
        if args.compare_source_val:
            logger.info('Same-subset source mAP50-95=%.5f; source-minus-pruned=%.5f',
                valreport['source']['overall']['map50_95'],
                valreport['map50_95_drop_source_minus_pruned'])
        checks['validation_metrics_finite']=True
        audit['validation_summary']={'accuracy_status':valreport['accuracy_status'],
            'map50_95_drop_source_minus_pruned':valreport.get('map50_95_drop_source_minus_pruned'),
            'subset':subset,'note':'Accuracy does NOT enter Stage06 structural PASS gate'}
        if valreport['accuracy_status']=='RECOVERY_NEEDED':
            warnings.append('RECOVERY_NEEDED: post-pruning accuracy drop exceeds threshold; proceed to FP32 recovery in Stage07')
        # GPU memory max is diagnostic unless user gave explicit limit.
        memory_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None
        if args.max_cuda_peak_mib and (memory_bytes is None or memory_bytes>args.max_cuda_peak_mib*1024**2):
            raise RuntimeError(f'CUDA peak memory {memory_bytes} violates requested hard limit of {args.max_cuda_peak_mib} MiB')
        audit['memory']={'peak_cuda_allocated_bytes':memory_bytes,
            'max_cuda_peak_mib_policy':args.max_cuda_peak_mib,
            'note':'Peak includes model/validation caching and is not deployment memory or activation-only peak'}
        audit['onnx']=optional_onnx_export(checkpoint,stage_dir,hw,args)
        checks['memory_and_optional_export_policy']=True
        if sha256_file(checkpoint)!=audit['checkpoint']['pruned_sha256'] or sha256_file(source)!=audit['checkpoint']['source_sha256']:
            raise RuntimeError('Checkpoint bytes changed during Stage06')
        if legacy_bridge is None:
            if dataset_fingerprint(dataset)['manifest_sha256']!=audit['provenance']['dataset_fingerprint']['manifest_sha256']:
                raise RuntimeError('Dataset image/label fingerprint changed during Stage06 validation')
        else:
            # Ultralytics can legitimately create/update .cache while validating;
            # SHA256 verifies that no *non-cache* dataset contents have changed.
            g0_path = gate_file(out, '00', 'check_environment')
            validate_legacy_bridge(dataset, read_json(g0_path), sha256_file(g0_path), legacy_bridge)
        audit['status']='STRUCTURALLY_VERIFIED'
        report.update({'status':'PASS','accuracy_status':valreport['accuracy_status'],
           'structural_status':audit['status'],'params':profile['params'],
           'params_reduction':audit['resources']['params_reduction'],
           'macs':profile['macs'],'macs_reduction':audit['resources']['macs_reduction']})
        write_json(stage_dir/'06_graph_audit.json',audit)
        write_json(stage_dir/'06_pre_recovery_val.json',valreport)
        write_json(stage_dir/'06_report.json',report)
        artifacts={'graph_audit':str(stage_dir/'06_graph_audit.json'),
                   'pre_recovery_val':str(stage_dir/'06_pre_recovery_val.json'),
                   'report':str(stage_dir/'06_report.json'),
                   'reload_receipt':str(stage_dir/'06_reload_receipt.json'),
                   'backward_log':str(stage_dir/'06_backward_log.csv')}
        if audit['onnx']['status']=='ONNX_CHECKER_PASS':
            artifacts['onnx']=audit['onnx']['path']
        provenance['artifact_hashes']={key:sha256_file(Path(path)) for key,path in artifacts.items()}
        checks['artifacts_saved']=True
    except Exception as exc:
        logger.exception('Stage06 FAIL: %s: %s',type(exc).__name__,exc)
        errors.append(f'{type(exc).__name__}: {exc}')
        (stage_dir/'06_error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        audit['status']='FAIL';audit['errors']=errors
        write_json(stage_dir/'06_graph_audit.json',audit)
        valreport['status']='INCOMPLETE' if valreport.get('status')!='VAL_COMPLETE' else 'VAL_COMPLETE'
        write_json(stage_dir/'06_pre_recovery_val.json',valreport)
        report['status']='FAIL';report['errors']=errors
        write_json(stage_dir/'06_report.json',report)
    gate=save_gate(out,STAGE,NAME,checks=checks,errors=errors,warnings=warnings,
                   provenance=provenance,artifacts=artifacts if not errors else {})
    write_json(stage_dir/'06_gate.json',gate)
    logger.info('[06] %s — %s',gate['status'],gatepath)
    return 0 if gate['status']=='PASS' else 1


if __name__=='__main__':
    args=parser().parse_args()
    sys.exit(_child_main(args) if args._child_reload else run(args))
