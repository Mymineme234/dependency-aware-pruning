#!/usr/bin/env python3
"""07 — FP32 recovery fine-tuning of an *already structurally pruned* YOLO.

IMPORTANT: Ultralytics YOLO(pruned.pt).train() normally rebuilds the model
from its original YAML. This script injects the actual pruned nn.Module into
DetectionTrainer and explicitly forbids get_model() reconstruction.

This is recovery, not pruning or QAT. Accuracy acceptance belongs to Stage08.
Supports Ultralytics YOLOv8 / YOLO11 / YOLO26 detection, with the same
ancestor contract, exact Conv/BN dimensions and checkpoint topology.

Run only on .pt files you trust: Ultralytics checkpoints use pickle.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path

from pruning_common import (
    SUPPORTED_FAMILIES, contract, dataset_fingerprint, gate_file, hash_object,
    make_imgsz, model_head_metadata, now_utc, read_dataset, read_json,
    resolved, save_gate, setup_log, sha256_file, write_json,
)

STAGE, NAME = '07', 'recover_fp32'


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Stage07: train the SAME physically pruned architecture; select by VAL fitness')
    for flag in ('model', 'data', 'output'):
        p.add_argument('--' + flag, required=True)
    p.add_argument('--imgsz', nargs='+', type=int, default=[640])
    p.add_argument('--device', default='0')
    p.add_argument('--batch', type=int, default=8, help='EXACT Stage00 contract, not necessarily recovery batch')
    p.add_argument('--workers', type=int, default=4, help='EXACT Stage00 contract, not necessarily recovery workers')
    p.add_argument('--expected-classes', type=int)
    p.add_argument('--family', choices=('auto', *SUPPORTED_FAMILIES), default='auto')
    p.add_argument('--recovery-epochs', type=int, default=30)
    p.add_argument('--recovery-batch', type=int, default=4, help='Physical train batch; independent of contract --batch')
    p.add_argument('--recovery-workers', type=int, default=2, help='Train/val dataloader workers; independent of contract --workers')
    p.add_argument('--optimizer', choices=('AdamW', 'SGD'), default='AdamW')
    p.add_argument('--lr0', type=float, default=0.0002)
    p.add_argument('--lrf', type=float, default=0.05)
    p.add_argument('--weight-decay', type=float, default=0.0005)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--warmup-epochs', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False,
                   help='Optional mixed-precision compute; exported best/last remain FP32')
    p.add_argument('--mosaic', type=float, default=0.3)
    p.add_argument('--close-mosaic', type=int, default=0)
    p.add_argument('--nbs', type=int, default=64)
    p.add_argument('--scale', type=float, default=0.25)
    p.add_argument('--translate', type=float, default=0.05)
    p.add_argument('--fliplr', type=float, default=0.5)
    p.add_argument('--val-subset', choices=('full', 'stage06'), default='full',
                   help='Default full VAL. stage06 uses SAME fixed subset for fast exploratory recovery')
    p.add_argument('--reload-timeout', type=int, default=300)
    p.add_argument('--overwrite', action='store_true', help='Only replace Stage07 and invalidate 07..10 gates')
    return p


def load_sibling(filename: str, unique_name: str):
    path = Path(__file__).with_name(filename)
    if not path.is_file():
        raise FileNotFoundError(f'Required predecessor source missing: {path}')
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_args(args, hw):
    if hw[0] != hw[1]:
        raise ValueError('Stage07 Ultralytics trainer requires a square --imgsz (one integer, e.g. 640)')
    if args.recovery_epochs < 1 or args.recovery_batch < 1 or args.recovery_workers < 0:
        raise ValueError('Invalid recovery epochs/batch/workers')
    if args.patience < 0 or args.seed < 0 or args.reload_timeout < 1:
        raise ValueError('Invalid patience/seed/timeout')
    if not (0 < args.lr0 < 1 and 0 < args.lrf <= 1 and args.weight_decay >= 0):
        raise ValueError('Invalid LR/weight-decay')
    if not (0 <= args.warmup_epochs <= args.recovery_epochs):
        raise ValueError('Invalid warmup_epochs')
    for name in ('mosaic', 'scale', 'translate', 'fliplr'):
        if not (0 <= float(getattr(args, name)) <= 1):
            raise ValueError(f'--{name} must be in [0,1]')
    if ',' in args.device or len(str(args.device).split()) > 1:
        raise ValueError('Stage07 supports one GPU/CPU device; multi-GPU DDP would reconstruct the pruned model')


def model_structure(net, step5):
    return {
        'params': sum(p.numel() for p in net.parameters()),
        'parameter_shapes': {n: list(p.shape) for n, p in net.named_parameters()},
        'conv_shapes': step5.current_conv_shapes(net),
        'bn_shapes': step5.current_bn_shapes(net),
        'head': model_head_metadata(net),
    }


def assert_unchanged_structure(actual: dict, expected: dict, context: str):
    for key in ('params', 'parameter_shapes', 'conv_shapes', 'bn_shapes', 'head'):
        if actual[key] != expected[key]:
            a, b = actual[key], expected[key]
            if isinstance(a, dict) and isinstance(b, dict):
                names = sorted({*a, *b})
                changed = [n for n in names if a.get(n) != b.get(n)]
                raise RuntimeError(f'TOPOLOGY_CHANGED ({context}): {key}, altered keys={changed[:12]}')
            raise RuntimeError(f'TOPOLOGY_CHANGED ({context}): {key}, expected={b}, actual={a}')


def assert_finite_fp32(net):
    import torch
    for name, tensor in list(net.named_parameters()) + list(net.named_buffers()):
        if tensor.is_floating_point():
            if tensor.dtype != torch.float32:
                raise RuntimeError(f'Not FP32: {name}: {tensor.dtype}')
            if not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f'NaN/Inf in parameter/buffer {name}')


def train_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise RuntimeError(f'Ultralytics did not emit results.csv: {path}')
    with path.open(encoding='utf-8', newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError('No recovery epoch recorded in results.csv')
    for i, row in enumerate(rows):
        if 'epoch' not in row:
            raise RuntimeError('results.csv missing epoch column')
        for k, v in row.items():
            if v is None or not str(v).strip():
                continue
            try:
                number = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                raise RuntimeError(f'Non-finite training/validation metric at epoch={i+1}: {k}={v}')
    if not any('mAP50-95' in k for k in rows[-1]):
        raise RuntimeError('No validation mAP50-95 column: training VAL fitness was not verified')
    return rows


def val_map_from_row(row: dict):
    keys = [k for k in row if 'mAP50-95' in k and 'metrics/' in k]
    if len(keys) != 1:
        raise RuntimeError(f'Ambiguous/missing VAL mAP50-95 metric: {keys}')
    value = float(row[keys[0]])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise RuntimeError(f'Invalid VAL mAP50-95: {value}')
    return value


def parameter_changed(before_net, after_net) -> dict:
    import torch
    before = dict(before_net.named_parameters())
    after = dict(after_net.named_parameters())
    if before.keys() != after.keys():
        raise RuntimeError('Parameter names changed while comparing recovery weights')
    changed = []
    # Comparing FP32 weights exactly is appropriate for verifying an optimization
    # occurred (not for judging its usefulness); all weights must remain finite.
    with torch.no_grad():
        for name in before:
            a = before[name].detach().cpu().float()
            b = after[name].detach().cpu().float()
            if not torch.equal(a, b):
                changed.append(name)
    return {'changed_parameter_tensors': len(changed),
            'total_parameter_tensors': len(before),
            'examples': changed[:15]}


def safe_dataset_lineage(out, dataset, stage06, gate5):
    bridge = stage06.load_stage05_legacy_bridge(out, dataset, gate5)
    if bridge is None:
        current = dataset_fingerprint(dataset)
        ancestor = gate5.get('provenance', {}).get('dataset_manifest', {})
        if current['manifest_sha256'] != ancestor.get('manifest_sha256'):
            raise RuntimeError('Dataset inventory changed since Stage05; no legacy bridge attested')
        return {'mode': 'original_dataset_manifest', 'manifest_sha256': current['manifest_sha256']}
    # The prior-stage verifier has already validated the bridge's full non-cache
    # content hashes; revalidate after training too.
    return {'mode': bridge.get('policy', 'attested_legacy_bridge'),
            'bridge_sha256': sha256_file(out / 'stages/05/05_dataset_legacy_bridge.json'),
            'non_cache_manifest_sha256': bridge['non_cache_inventory']['manifest_sha256']}


def verified_stage06(out, request, dataset):
    stage06 = load_sibling('06_verify_pruned_model.py', 'stage06_for07')
    g06path = gate_file(out, '06', 'verify_pruned_model')
    if not g06path.is_file():
        raise RuntimeError('Stage06 gate missing: run Stage06 before recovery')
    g06 = read_json(g06path)
    if g06.get('step') != '06_verify_pruned_model' or g06.get('status') != 'PASS' or not g06.get('checks') or not all(v is True for v in g06['checks'].values()):
        raise RuntimeError('Stage06 must PASS on structure, reload and backward; accuracy may be RECOVERY_NEEDED')
    if g06.get('provenance', {}).get('contract_sha256') != hash_object(request):
        raise RuntimeError('Stage06 contract differs from Stage07 CLI')
    g05, plan, graph, report5, source, raw = stage06.verify_lineage(out, request, dataset)
    if g06['provenance'].get('stage05_gate_sha256') != sha256_file(gate_file(out,'05','apply_structured_pruning')):
        raise RuntimeError('Stage05 gate has changed since Stage06')
    if g06['provenance'].get('stage05_checkpoint_sha256') != sha256_file(raw):
        raise RuntimeError('Raw pruned checkpoint changed since Stage06')
    art_hashes = g06['provenance'].get('artifact_hashes', {})
    for key, path in g06.get('artifacts', {}).items():
        if not isinstance(path, str) or not Path(path).is_file() or art_hashes.get(key) != sha256_file(Path(path)):
            raise RuntimeError(f'Stage06 artifact changed/missing: {key}: {path}')
    if not g06['artifacts']:
        raise RuntimeError('Stage06 artifact list empty')
    meta = safe_dataset_lineage(out, dataset, stage06, g05)
    return stage06, g06, g05, plan, graph, report5, source, raw, meta


def trainer_kwargs(args, raw: Path, data_path: Path, stage_dir: Path, val_settings: dict):
    return dict(
        model=str(raw), data=str(data_path), task='detect', mode='train',
        project=str(stage_dir), name='trainer', exist_ok=True,
        epochs=args.recovery_epochs, batch=args.recovery_batch,
        imgsz=make_imgsz(args.imgsz)[0], device=args.device,
        workers=args.recovery_workers, optimizer=args.optimizer, lr0=args.lr0,
        lrf=args.lrf, cos_lr=True, weight_decay=args.weight_decay,
        patience=args.patience, warmup_epochs=args.warmup_epochs,
        seed=args.seed, deterministic=True, amp=args.amp,
        mosaic=args.mosaic, scale=args.scale, translate=args.translate, nbs=args.nbs,
        fliplr=args.fliplr, mixup=0.0, copy_paste=0.0,
        close_mosaic=args.close_mosaic, degrees=0.0, shear=0.0, perspective=0.0,
        cache=False, rect=False, multi_scale=0.0,
        pretrained=False, resume=False, save=True, val=True,
        save_period=-1, plots=False, verbose=True,
        quantize=None, compile=False,
        conf=val_settings['conf'], iou=val_settings['iou'],
        max_det=val_settings['max_det'], nms=val_settings['nms'],
    )


def atomic_fp32_checkpoint(path: Path, net, trainargs: dict, meta: dict) -> str:
    """Native, safe-to-reload Ultralytics FP32 checkpoint from a detached EMA copy.

    Avoid trainer's auto-FP16 best.pt serialization; preserve actual FP32 EMA
    values. Never mutate the live trainer EMA or optimizer.
    """
    import torch
    import ultralytics
    path.parent.mkdir(parents=True, exist_ok=True)
    net.cpu().eval().float()
    for name, param in net.named_parameters():
        param.requires_grad_('.dfl' not in name)  # fixed DFL integration conv is never learned
    if hasattr(net, 'criterion'):
        net.criterion = None
    assert_finite_fp32(net)
    payload = {
        'epoch': -1, 'best_fitness': meta['fitness'],
        'model': net, 'ema': None, 'updates': int(meta['optimizer_updates']),
        'optimizer': None, 'train_args': dict(trainargs),
        'train_metrics': {'fitness': meta['fitness']}, 'train_results': {},
        'date': now_utc(), 'version': ultralytics.__version__,
        'pruning_stage': 5, 'recovery_stage': 7, 'recovery_manifest': dict(meta),
    }
    fd, temp = tempfile.mkstemp(prefix='stage07_',suffix='.tmp',dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temp)
        if Path(temp).stat().st_size <= 0:
            raise RuntimeError('Empty Stage07 FP32 checkpoint')
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return sha256_file(path)


def run_training(args, raw: Path, data_yaml: Path, out: Path, expected: dict,
                 step5, val_settings: dict, logger) -> dict:
    """Use native detection train/EMA/optimizer, preserving pruned topology.

    Epoch checkpoints are written as FP32 *before* Ultralytics auto-strips and
    half-converts best.pt/last.pt. Only weights, no optimizer, for Stage08/QAT.
    """
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    class PreservePrunedTopologyTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            raise RuntimeError('RECONSTRUCTION_FORBIDDEN: Stage07 cannot rebuild from original YAML')

    initial = YOLO(str(raw))
    if initial.task != 'detect':
        raise RuntimeError('Recovery only supports Ultralytics detect checkpoints')
    # Never use YOLO(...).train(); that would silently run get_model(cfg=yaml).
    net = initial.model.float()
    net.requires_grad_(True)
    if hasattr(net, 'criterion'):
        net.criterion = None  # initialize the correct YOLO-version detection loss on first train batch
    # The Ultralytics detection trainer may change head.max_det to fit crowded
    # training VAL. Keep that choice for validation, but restore the original
    # inference ABI on published pruned FP32 checkpoint snapshots.
    original_head_max_det = getattr(net.model[-1], 'max_det', None)
    assert_unchanged_structure(model_structure(net, step5), expected, 'pretrain')
    before_tensor = {k: p.detach().cpu().clone() for k,p in net.named_parameters()}
    stage_dir = out / 'stages/07'
    weights_dir = out / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_out = weights_dir / 'pruned_recovered_best.pt'
    last_out = weights_dir / 'pruned_recovered_last.pt'
    kw = trainer_kwargs(args, raw, data_yaml, stage_dir, val_settings)
    trainer = PreservePrunedTopologyTrainer(overrides=kw)
    trainer.model = net  # BaseTrainer.setup_model returns immediately for nn.Module
    assert_unchanged_structure(model_structure(trainer.model, step5), expected, 'trainer model injection')
    snapshots = []
    def save_fp32_epoch(tr):
        if tr.ema is None or getattr(tr.ema, 'ema', None) is None or tr.ema.updates <= 0:
            raise RuntimeError('No optimizer/EMA updates before recovery checkpoint save')
        if tr.fitness is None or not math.isfinite(float(tr.fitness)):
            raise RuntimeError('Non-finite/missing VAL fitness at recovery epoch')
        ema_cpu = deepcopy(tr.ema.ema).cpu().float()
        ema_cpu.eval()
        if hasattr(ema_cpu, 'criterion'):
            ema_cpu.criterion = None
        if original_head_max_det is not None:
            ema_cpu.model[-1].max_det = original_head_max_det
        # Ultralytics 8.4.173 BaseValidator changes trainer.ema.ema.end2end
        # to select NMS / NMS-free validation mode (tasks.DetectionModel
        # forwards this attribute to model[-1].end2end). This toggles an
        # inference-mode flag, NOT the network's physical topology.
        # Restore the original inference selection on this detached EMA copy
        # before comparing the head ABI and saving an FP32 checkpoint.
        original_end2end = bool(expected['head']['head_end2end'])
        if bool(getattr(ema_cpu.model[-1], 'end2end', False)) != original_end2end:
            logger.info('Restoring epoch EMA inference head mode: end2end=%s (validation toggled it)', original_end2end)
            ema_cpu.end2end = original_end2end
        assert_unchanged_structure(model_structure(ema_cpu, step5), expected, 'epoch EMA')
        assert_finite_fp32(ema_cpu)
        metadata = {'stage': 7, 'epoch': int(tr.epoch),
                    'fitness': float(tr.fitness),
                    'raw_pruned_checkpoint_sha256': sha256_file(raw),
                    'optimizer_updates': int(tr.ema.updates),
                    'checkpoint_selection': 'VAL_fitness'}
        trainargs = vars(tr.args).copy()
        trainargs['data'] = str(data_yaml)
        sha = atomic_fp32_checkpoint(last_out, ema_cpu, trainargs, metadata)
        entry = {'epoch': int(tr.epoch)+1, 'fitness': float(tr.fitness),
                 'optimizer_updates': int(tr.ema.updates), 'last_sha256': sha}
        # Ultralytics native trainer selects best using fitness, not last loss.
        if tr.best_fitness == tr.fitness:
            shutil.copy2(last_out, best_out)
            entry['best_sha256'] = sha256_file(best_out)
            entry['selected_best'] = True
        else:
            entry['selected_best'] = False
        snapshots.append(entry)
        logger.info('Stage07 epoch %d: VAL fitness=%.6f, best=%s, FP32 snapshot=%s',
                    entry['epoch'], entry['fitness'], entry['selected_best'], last_out)
        del ema_cpu

    trainer.add_callback('on_model_save', save_fp32_epoch)
    trainer.train()
    if not snapshots or not best_out.is_file() or not last_out.is_file():
        raise RuntimeError('Recovery completed without FP32 best/last checkpoints')
    if len(snapshots) < 1 or max(s['optimizer_updates'] for s in snapshots) < 1:
        raise RuntimeError('No optimizer step occurred in training')
    assert_unchanged_structure(model_structure(trainer.model, step5), expected, 'after training')
    results_csv = Path(trainer.csv)
    rows = train_csv_rows(results_csv)
    if len(rows) < len(snapshots):
        raise RuntimeError('Fewer VAL epoch result rows than saved checkpoints')
    best_map = max(val_map_from_row(row) for row in rows)
    metrics = {'best_val_map50_95_over_epochs': best_map,
               'last_val_map50_95': val_map_from_row(rows[-1]),
               'training_epochs_completed': len(rows),
               'snapshot_epochs': len(snapshots),
               'note': 'Stage07 diagnostic only; accuracy acceptance belongs to Stage08'}
    # Store model state for checkpoint update evidence without reloading raw twice.
    compare_net = YOLO(str(best_out)).model.float()
    before_names = set(before_tensor)
    after_names = set(dict(compare_net.named_parameters()))
    if before_names != after_names:
        raise RuntimeError('FP32 best changed parameter names')
    changed = [k for k,p in compare_net.named_parameters()
               if not torch.equal(before_tensor[k], p.detach().cpu().float())]
    del before_tensor
    if not changed:
        raise RuntimeError('NO_WEIGHT_UPDATE: recovery best equals pruned raw weights')
    metrics['updated_parameter_tensors_in_best'] = len(changed)
    metrics['updated_parameter_tensor_examples'] = changed[:12]
    metrics['effective_recovery_batch'] = int(trainer.batch_size)
    return {'best': best_out, 'last': last_out,
            'training_csv': results_csv, 'snapshots': snapshots,
            'metrics': metrics, 'trainer_path': Path(trainer.save_dir)}


def get_val_settings(out: Path, dataset: dict):
    """Reuse Stage01 validation thresholds; no special output-mode changes."""
    profile = read_json(out/'stages/01/01_profile.json')
    merged = {**profile.get('input', {}),
              **profile.get('validation', {}).get('validation_config', {})}
    conf = float(merged.get('conf', .001))
    iou = float(merged.get('iou', .7))
    max_det = int(merged.get('max_det', 1000))
    nms = bool(merged.get('nms_free', False))
    if not 0 <= conf <= 1 or not 0 <= iou <= 1 or max_det < 1:
        raise RuntimeError('Bad original validation config')
    if nms and dataset['nc'] < 1:
        raise RuntimeError('Invalid NMS-free config')
    return {'conf': conf, 'iou': iou, 'max_det': max_det, 'nms': nms}


def run(args: argparse.Namespace) -> int:
    out = resolved(args.output)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_log(out, STAGE, NAME)
    stage_dir = out/'stages/07'
    stage_dir.mkdir(parents=True, exist_ok=True)
    gatepath = gate_file(out, STAGE, NAME)
    if gatepath.exists() and not args.overwrite:
        logger.error('Stage07 gate already exists; use --overwrite (only Stage07 and successors)')
        return 2
    checks = dict.fromkeys((
        'cli_valid','stage06_structural_pass','lineage_and_dataset_verified',
        'original_checkpoint_untouched','pruned_topology_preserved',
        'native_detection_training_and_optimizer_updated',
        'validation_fitness_finite','fp32_best_last_saved',
        'independent_reload_and_abi_preserved','artifacts_saved'), False)
    errors=[]; warnings=[]; provenance={}; artifacts={}
    report={'step':f'{STAGE}_{NAME}', 'status':'PRELIMINARY', 'created_utc':now_utc()}
    try:
        hw = make_imgsz(args.imgsz)
        check_args(args, hw)
        checks['cli_valid'] = True
        data_path = resolved(args.data)
        dataset = read_dataset(data_path,args.expected_classes)
        request = contract(args.model, data_path, hw, args.device, args.batch,
                           args.workers, args.family, args.expected_classes)
        step06, gate6, gate5, plan, graph, report5, source, raw, dset_meta = verified_stage06(out,request,dataset)
        checks['stage06_structural_pass']=True
        checks['lineage_and_dataset_verified']=True
        step5 = load_sibling('05_apply_structured_pruning.py', 'stage05_for07')
        graph_conv=graph.get('pruned_convs')
        graph_bn=graph.get('pruned_batchnorm')
        if not graph_conv or not graph_bn:
            raise RuntimeError('Stage05 graph signature missing verified pruned Conv/BN map')
        from ultralytics import YOLO
        net = YOLO(str(raw)).model.float()
        expected=model_structure(net,step5)
        if expected['conv_shapes']!=graph_conv or expected['bn_shapes']!=graph_bn:
            raise RuntimeError('Raw checkpoint topology no longer matches Stage05 signature')
        if expected['params']!=int(report5['pruned_params']):
            raise RuntimeError('Raw checkpoint parameter count wrong')
        assert_finite_fp32(net)
        checks['original_checkpoint_untouched']=True
        raw_sha=sha256_file(raw); source_sha=sha256_file(source)
        g06sha=sha256_file(gate_file(out,'06','verify_pruned_model'))
        provenance={**gate6['provenance'],
                    'stage06_gate_sha256':g06sha,
                    'stage06_source_sha256':sha256_file(Path(__file__).with_name('06_verify_pruned_model.py')),
                    'stage05_raw_sha256':raw_sha,
                    'stage05_pruned_parameter_count':expected['params'],
                    'stage07_recovery_config':{
                        'epochs':args.recovery_epochs,'batch':args.recovery_batch,
                        'workers':args.recovery_workers,'optimizer':args.optimizer,
                        'lr0':args.lr0,'lrf':args.lrf,'weight_decay':args.weight_decay,
                        'patience':args.patience,'warmup_epochs':args.warmup_epochs,
                        'seed':args.seed,'amp':args.amp,'val_subset':args.val_subset,
                        'mosaic':args.mosaic,'close_mosaic':args.close_mosaic,'nbs':args.nbs,'scale':args.scale,
                        'translate':args.translate,'fliplr':args.fliplr},
                    'dataset_reverification':dset_meta}
        del net
        if args.overwrite:
            if (stage_dir/'trainer').exists():
                shutil.rmtree(stage_dir/'trainer')
            for p in stage_dir.glob('07_*'):
                if p.is_file():p.unlink()
            for p in (out/'weights/pruned_recovered_best.pt',out/'weights/pruned_recovered_last.pt'):
                p.unlink(missing_ok=True)
            for p in (out/'gates').glob('*.json'):
                if p.name[:2].isdigit() and int(p.name[:2])>=7:p.unlink()
            logger.warning('--overwrite invalidated Stage07..10, preserving Stage00..06 and Stage05 raw')
        data_for_train = data_path
        if args.val_subset == 'stage06':
            alt = out/'stages/06/06_smoke_val.yaml'
            if not alt.is_file():
                raise RuntimeError('--val-subset stage06 requires Stage06 smoke subset yaml (full VAL uses --val-subset full)')
            import yaml
            original=yaml.safe_load(data_path.read_text(encoding='utf-8'))
            candidate=yaml.safe_load(alt.read_text(encoding='utf-8'))
            if candidate.get('train') != original.get('train') or candidate.get('nc') != original.get('nc') or candidate.get('names') != original.get('names'):
                raise RuntimeError('Stage06 smoke YAML differs in train, classes or names')
            data_for_train=stage_dir/'07_train_with_stage06_val.yaml'
            data_for_train.write_text(yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True), encoding='utf-8')
            warnings.append('Recovery checkpoint selected on Stage06 VAL subset, not full validation; confirm using full VAL in Stage08')
        val_settings = get_val_settings(out, dataset)
        if val_settings['nms'] and gate5['provenance']['model_family'] != 'yolo26':
            raise RuntimeError('NMS-free allowed only for YOLO26 in this pipeline')
        logger.info('Stage07 recovery: raw=%s; epochs=%d, train_batch=%d, AMP=%s, val=%s',
                    raw,args.recovery_epochs,args.recovery_batch,args.amp,args.val_subset)
        logger.info('Pruned topology: %d params; %d Conv, %d BN; train native Ultralytics without YAML reconstruction',
                    expected['params'],len(graph_conv),len(graph_bn))
        result = run_training(args,raw,data_for_train,out,expected,step5,val_settings,logger)
        checks['native_detection_training_and_optimizer_updated']=True
        checks['validation_fitness_finite']=True
        checks['fp32_best_last_saved']=True
        for which in ('best','last'):
            pt=result[which]
            reloaded=YOLO(str(pt))
            if reloaded.task!='detect' or len(reloaded.names)!=dataset['nc']:
                raise RuntimeError(f'{which}: task/class-count changed after recovery')
            net=reloaded.model.float()
            assert_unchanged_structure(model_structure(net,step5),expected, f'{which} in-process reload')
            assert_finite_fp32(net)
            # Independently invoke existing Stage06 child verification of exact
            # Conv/BN shape, train/eval ABI and native Ultralytics reload.
            child_dir=stage_dir/f'07_{which}_fresh_process'
            child_dir.mkdir(parents=True,exist_ok=True)
            receipt=step06.subprocess_reload(
                Path(__file__).with_name('06_verify_pruned_model.py'),
                pt,sha256_file(pt),hw,'cpu',args.reload_timeout,child_dir)
            if receipt['parameter_count']!=expected['params'] or receipt['conv_shapes']!=graph_conv or receipt['bn_shapes']!=graph_bn:
                raise RuntimeError(f'{which}: fresh-process Conv/BN/params differ from Stage05')
            if receipt['train_output_abi'] != graph['baseline_train_output_schema']:
                raise RuntimeError(f'{which}: train output ABI differs from baseline')
            pre_reload=read_json(out/'stages/06/06_reload_receipt.json')
            if receipt['eval_output_abi'] != pre_reload['eval_output_abi']:
                raise RuntimeError(f'{which}: eval output ABI differs from Stage06')
            logger.info('Stage07 %s: fresh-process reload PASS, SHA=%s, params=%d',
                        which,sha256_file(pt)[:12],expected['params'])
        checks['pruned_topology_preserved']=True
        checks['independent_reload_and_abi_preserved']=True
        if source_sha != sha256_file(source) or raw_sha != sha256_file(raw) or g06sha != sha256_file(gate_file(out,'06','verify_pruned_model')):
            raise RuntimeError('Ancestor checkpoint or Stage06 gate changed during recovery')
        step06.verify_lineage(out,request,dataset)
        dset_after = safe_dataset_lineage(out,dataset,step06,gate5)
        if dset_after != dset_meta:
            raise RuntimeError('Dataset provenance changed during recovery')
        target_csv=stage_dir/'07_results.csv'
        shutil.copy2(result['training_csv'],target_csv)
        history=stage_dir/'07_epoch_checkpoints.json'
        write_json(history,result['snapshots'])
        summary={'checkpoint_pre':str(raw),'checkpoint_pre_sha256':raw_sha,
                 'best':str(result['best']),'last':str(result['last']),
                 'epoch_metrics':result['metrics'],
                 'resource':{'params':expected['params'],'baseline_params':int(report5['baseline_params']),
                             'params_reduction':float(report5['param_reduction']),
                             'macs':float(report5['pruned_macs']),
                             'macs_reduction':float(report5['mac_reduction'])},
                 'train_data':str(data_for_train),
                 'dataset_provenance':dset_after,
                 'topology_unchanged':True,
                 'accuracy_gate':'DEFERRED_TO_STAGE08',
                 'qat_readiness':'UNVERIFIED_STAGE09_REQUIRED',
                 'notes':'Trainer best chosen on VAL fitness, not test. Full FP32 EMA snapshots saved before Ultralytics half conversion.'}
        report.update({'status':'PASS',**summary})
        write_json(stage_dir/'07_report.json',report)
        artifacts={'best':str(result['best']),'last':str(result['last']),
                   'results_csv':str(target_csv),'epoch_checkpoints':str(history),
                   'report':str(stage_dir/'07_report.json'),
                   'best_reload_receipt':str(stage_dir/'07_best_fresh_process/06_reload_receipt.json'),
                   'last_reload_receipt':str(stage_dir/'07_last_fresh_process/06_reload_receipt.json')}
        if args.val_subset == 'stage06':
            artifacts['recovery_data_yaml']=str(data_for_train)
        provenance['artifact_hashes']={key:sha256_file(Path(path)) for key,path in artifacts.items()}
        checks['artifacts_saved']=True
        logger.info('Recovery checkpoint: %s (VAL best mAP50-95 over epochs %.5f)',
                    result['best'],result['metrics']['best_val_map50_95_over_epochs'])
        logger.info('Accuracy acceptance deferred to Stage08; target resources unchanged from Stage05.')
    except Exception as exc:
        logger.exception('Stage07 FAIL: %s: %s',type(exc).__name__,exc)
        errors.append(f'{type(exc).__name__}: {exc}')
        (stage_dir/'07_error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        report['status']='FAIL'; report['errors']=errors
        write_json(stage_dir/'07_report.json',report)
        # A failing run cannot leave apparently certified recovered weights.
        for p in (out/'weights/pruned_recovered_best.pt',out/'weights/pruned_recovered_last.pt'):
            p.unlink(missing_ok=True)
    gate=save_gate(out,STAGE,NAME,checks=checks,errors=errors,warnings=warnings,
                   provenance=provenance,artifacts=artifacts if not errors else {})
    write_json(stage_dir/'07_gate.json',gate)
    logger.info('[07] %s — %s',gate['status'],gate_file(out,STAGE,NAME))
    return 0 if gate['status']=='PASS' else 1


if __name__=='__main__':
    sys.exit(run(parser().parse_args()))
