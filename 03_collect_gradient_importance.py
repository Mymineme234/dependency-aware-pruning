#!/usr/bin/env python3
"""Stage 03: real YOLO detection-loss gradients and dependency-coupled Taylor scores.

Uses the unmodified FP32 checkpoint and TRAIN data only. No optimizer, pruning,
checkpoint mutation, or test/validation samples. Requires PASS gates 00, 01, 02.

Score = mean_b sum_unique_params |weight * d(loss_per_image)/d(weight)|.
The dependency for every channel is derived from Torch-Pruning DepGraph; Stage 04
MUST re-check exact aligned channel sets and graph constraints before pruning.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from pruning_common import (
    SUPPORTED_FAMILIES, contract, dataset_fingerprint, gate_file, hash_object,
    make_imgsz, model_head_metadata, now_utc, read_dataset, read_json, resolved,
    save_gate, select_device, setup_log, sha256_file, versions, write_json,
)

STAGE = '03'
NAME = 'collect_gradient_importance'


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='03: train-only YOLO detection-loss Group Taylor importance (no pruning)')
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--imgsz', type=int, nargs='+', default=[640])
    p.add_argument('--device', default='0')
    p.add_argument('--batch', type=int, default=8, help='Stage 00/01/02 contract batch')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--expected-classes', type=int, default=None)
    p.add_argument('--family', choices=['auto', *SUPPORTED_FAMILIES], default='auto')
    p.add_argument('--gradient-batches', type=int, default=64, help='Number of DISTINCT train dataloader batches')
    p.add_argument('--gradient-batch-size', type=int, default=None, help='Optional smaller gradient microbatch; contract --batch unchanged')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--min-nonzero-gradient-fraction', type=float, default=0.01,
                   help='Share of sampled batches with a nonzero gradient on eligible roots')
    p.add_argument('--allow-unscored-roots', action='store_true', help='Explicitly protect Stage 02 roots with no complete scoreable aligned block; default FAIL')
    p.add_argument('--overwrite', action='store_true', help='Replace 03 results and invalidate 04..10 gates')
    return p


def ensure_predecessors(out: Path, current_contract: dict, data: dict) -> tuple[dict, dict, dict, Path, dict]:
    """Protect the exact same model/data/software used in stages 00–02."""
    paths = [gate_file(out, '00', 'check_environment'),
             gate_file(out, '01', 'profile_baseline'),
             gate_file(out, '02', 'analyze_pruning_graph')]
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f'Missing predecessor gate: {path}')
    g0, g1, g2 = [read_json(p) for p in paths]
    for i, g in enumerate((g0, g1, g2)):
        if g.get('status') != 'PASS' or not g.get('checks') or not all(x is True for x in g['checks'].values()):
            raise RuntimeError(f'Stage {i:02d} has not PASSed all mandatory checks')
    p0, p1, p2 = (g['provenance'] for g in (g0, g1, g2))
    identity = hash_object(current_contract)
    if any(p.get('contract_sha256') != identity for p in (p0, p1, p2)):
        raise RuntimeError('Input model/data/imgsz/device/batch/workers/family does not match 00–02 contract')
    if p1.get('stage00_gate_sha256') != sha256_file(paths[0]):
        raise RuntimeError('Stage 00 changed since Stage 01')
    if (p2.get('stage00_gate_sha256') != sha256_file(paths[0]) or
            p2.get('stage01_gate_sha256') != sha256_file(paths[1])):
        raise RuntimeError('Stage 00/01 changed since Stage 02')
    profile_path = out / 'stages' / '01' / '01_profile.json'
    if not profile_path.is_file() or p2.get('stage01_profile_sha256') != sha256_file(profile_path):
        raise RuntimeError('Baseline profile changed since Stage 02')
    checkpoint = Path(p0.get('source_checkpoint', ''))
    if not checkpoint.is_file() or sha256_file(checkpoint) != p0.get('checkpoint_sha256'):
        raise RuntimeError('Model checkpoint has changed since Stage 00')
    if data['yaml_sha256'] != p0.get('dataset_yaml_sha256'):
        raise RuntimeError('Dataset YAML changed since Stage 00')
    if dataset_fingerprint(data)['manifest_sha256'] != p0.get('dataset_manifest', {}).get('manifest_sha256'):
        raise RuntimeError('Dataset image/label inventory changed since Stage 00')
    current_versions = versions()
    for pkg in ('torch', 'torchvision', 'ultralytics', 'torch-pruning', 'numpy'):
        if p0.get('dependency_versions', {}).get(pkg) != current_versions.get(pkg):
            raise RuntimeError(f'Dependency version changed since Stage 00: {pkg}')
    graph_path = out / 'stages' / '02' / '02_graph.json'
    if not graph_path.is_file():
        raise RuntimeError(f'Stage 02 graph missing: {graph_path}')
    graph = read_json(graph_path)
    if graph.get('family') != p0.get('model_family') or graph.get('trace_mode') != 'train':
        raise RuntimeError('Stage 02 must have a valid TRAIN autograd dependency graph for gradient collection')
    ids = [g['group_id'] for g in graph.get('groups', [])]
    if len(ids) != len(set(ids)) or not ids:
        raise RuntimeError('Malformed Stage 02 groups')
    eligible = [g for g in graph['groups'] if g['feasibility'] == 'ELIGIBLE_PROBE_ONLY']
    if not eligible:
        raise RuntimeError('Stage 02 contains no eligible pruning roots')
    if len(eligible) != graph.get('summary', {}).get('eligible_probe_roots'):
        raise RuntimeError('Eligible group count differs from Stage 02 summary')
    return g0, g1, g2, checkpoint, graph


def load_stage02_helpers() -> Any:
    """Reuse EXACT Stage 02 hazard checks, without modifying already-PASSed code."""
    filename = Path(__file__).with_name('02_analyze_pruning_graph.py')
    if not filename.is_file():
        raise RuntimeError('03 requires 02_analyze_pruning_graph.py in the same pipeline directory')
    spec = importlib.util.spec_from_file_location('pruning_stage02_analysis', filename)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot import Stage 02 dependency safety rules')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr in ('_safe_probe', '_head_paths', '_attention_paths', '_static_module_paths', '_output_scalar'):
        if not hasattr(module, attr):
            raise RuntimeError(f'Stage 02 is missing required interface: {attr}')
    return module, sha256_file(filename)


def unpack_detection_loss(result, torch):
    """Validate native Ultralytics detection loss and reduce its component vector.

    Ultralytics 8.4.173 v8DetectionLoss.loss returns a differentiable tensor
    of weighted detection-loss components (typically [box, cls, dfl/l1])
    multiplied by batch size; E2ELoss (YOLO26 dual-head) combines the
    branch-wise vectors. The *official trainer* reduces this vector to the
    scalar used for backward. Older versions may return a scalar already.

    Summing the native differentiable components is NOT a synthetic objective.
    Reject detached losses and unexpected higher-dimensional predictions.
    """
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError('Expected Ultralytics model(labelled_batch) -> (detection_loss, loss_items); dummy loss forbidden')
    loss, components = result
    if not torch.is_tensor(loss):
        raise RuntimeError(f'Expected native detection-loss tensor; got {type(loss).__name__}')
    if loss.ndim > 1 or loss.numel() == 0 or not loss.is_floating_point() or not loss.requires_grad:
        raise RuntimeError(
            f'Invalid native detection loss: shape={tuple(loss.shape)}, '
            f'dtype={loss.dtype}, requires_grad={loss.requires_grad}. '
            'Expected differentiable scalar or 1-D vector of native detection loss components.'
        )
    if not bool(torch.isfinite(loss).all().item()):
        raise RuntimeError('Non-finite real detection loss component(s)')
    total = loss.sum() if loss.ndim == 1 else loss
    if total.ndim != 0 or not total.requires_grad or not bool(torch.isfinite(total).item()):
        raise RuntimeError('Failed to reduce real detection-loss components to differentiable scalar')
    return total, components


def _loss_components(obj: Any, torch) -> dict:
    if torch.is_tensor(obj):
        vals = obj.detach().flatten().float().cpu().tolist()
        return {f'component_{i}': float(x) for i, x in enumerate(vals)}
    if isinstance(obj, dict):
        result = {}
        for k, val in obj.items():
            if torch.is_tensor(val) and val.numel() == 1:
                result[str(k)] = float(val.detach().item())
            elif isinstance(val, (float, int)):
                result[str(k)] = float(val)
        return result
    if isinstance(obj, (list, tuple)):
        result = {}
        for i, val in enumerate(obj):
            if torch.is_tensor(val) and val.numel() == 1:
                result[f'component_{i}'] = float(val.detach().item())
            elif isinstance(val, (float, int)):
                result[f'component_{i}'] = float(val)
        return result
    raise RuntimeError(f'Unexpected YOLO loss_items structure: {type(obj).__name__}')


def _dataset_loader(ckpt: Path, yaml_path: Path, net, args, data: dict, device, logger):
    """Official train dataloader + preprocessing. Never call train()/optimizer.step()."""
    from ultralytics.models.yolo.detect import DetectionTrainer
    if len(args.imgsz) == 2 and args.imgsz[0] != args.imgsz[1]:
        raise RuntimeError('Stage 03 Ultralytics DetectionTrainer loader currently requires square --imgsz; rectangular training unverified')
    sz = args.imgsz[0]
    effective_batch = args.gradient_batch_size or args.batch
    settings = dict(
        model=str(ckpt), data=str(yaml_path), imgsz=sz, task='detect',
        device=str(args.device), batch=effective_batch, workers=args.workers,
        epochs=1, fraction=1.0, cache=False, rect=False,
        multi_scale=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0,
        scale=0.0, translate=0.0, degrees=0.0, shear=0.0, perspective=0.0,
        fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        seed=args.seed, deterministic=True,
        project=str(resolved(args.output) / 'stages' / '03'),
        name='loader_state', exist_ok=True, save=False, plots=False, val=False,
    )
    trainer = DetectionTrainer(overrides=settings)
    trainer.model = net
    # This sets net.args needed by model's REAL detection criterion, including
    # differing loss implementations in YOLOv8/11 vs YOLO26.
    trainer.set_model_attributes()
    train_paths = data['splits'].get('train')
    if not train_paths:
        raise RuntimeError('No TRAIN split present in YAML')
    path_arg = train_paths[0] if len(train_paths) == 1 else train_paths
    logger.info('Training split ONLY: %s; minibatch=%d, K=%d', train_paths, effective_batch, args.gradient_batches)
    loader = trainer.get_dataloader(path_arg, batch_size=effective_batch, rank=-1, mode='train')
    if len(loader) < args.gradient_batches:
        raise RuntimeError(f'Insufficient train batches: loader has {len(loader)}, requested {args.gradient_batches}; no recycling/repeats')
    return trainer, loader


def collect_real_gradients(net, trainer, loader, args, logger, stage02_graph: dict):
    """Accumulate signed and |w*g| over K real-labelled train batches in FP32."""
    import torch
    net.requires_grad_(True)
    net.train()
    params = dict(net.named_parameters())
    if not params:
        raise RuntimeError('No model parameters available')
    signed = {name: torch.zeros_like(p, dtype=torch.float32, device='cpu')
              for name, p in params.items() if p.requires_grad}
    absolute = {name: torch.zeros_like(p, dtype=torch.float32, device='cpu')
                for name, p in params.items() if p.requires_grad}
    logs = []
    # Explicit graph input from ensure_predecessors(); not an argparse argument.
    eligible_roots = [g['root'] for g in stage02_graph['groups']
                      if g['feasibility'] == 'ELIGIBLE_PROBE_ONLY']
    if not eligible_roots:
        raise RuntimeError('Stage 02 graph has no eligible roots to collect gradients for')
    nonzero_batch_count = 0
    named_modules = dict(net.named_modules())
    iterator = iter(loader)
    for idx in range(1, args.gradient_batches + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            raise RuntimeError(f'Train dataloader exhausted after {idx - 1}/{args.gradient_batches} batches')
        start = time.monotonic()
        prepared = trainer.preprocess_batch(batch)
        if not isinstance(prepared, dict) or not all(k in prepared for k in ('img', 'cls', 'bboxes', 'batch_idx')):
            raise RuntimeError('Official detection dataloader returned incomplete labelled batch')
        for name in ('img', 'cls', 'bboxes', 'batch_idx'):
            if not torch.is_tensor(prepared[name]):
                raise RuntimeError(f'Labelled batch key {name} is not a tensor')
            if prepared[name].device != prepared['img'].device:
                prepared[name] = prepared[name].to(prepared['img'].device)
        if prepared['img'].ndim != 4 or prepared['img'].shape[1] != 3:
            raise RuntimeError('Expected BCHW RGB image tensor')
        if tuple(prepared['img'].shape[2:]) != tuple(args.imgsz_hw):
            raise RuntimeError(f'Train image shape unexpectedly changed: {list(prepared["img"].shape)}')
        if not prepared['img'].is_floating_point() or not bool(torch.isfinite(prepared['img']).all().item()):
            raise RuntimeError('Invalid normalized training image')
        bs = int(prepared['img'].shape[0])
        nobjects = int(prepared['cls'].numel())
        if bs < 1:
            raise RuntimeError(f'Invalid empty image batch {idx}')
        # An all-background minibatch is valid for YOLO detection; require at
        # least one positive GT across the WHOLE sampling window, not per batch.
        net.zero_grad(set_to_none=True)
        with torch.enable_grad():
            result = net(prepared)  # REAL detection criterion; NOT output.sum().
            loss, components = unpack_detection_loss(result, torch)
            per_image_loss = loss / bs  # Ultralytics often returns bs-scaled loss.
            per_image_loss.backward()
        total_grad_tensors, nonzero_tensors = 0, 0
        elig_nonzero = 0
        for root in eligible_roots:
            module = named_modules.get(root)
            if module is None or not isinstance(module, torch.nn.Conv2d):
                raise RuntimeError(f'Eligible Stage 02 root missing / not Conv2d: {root}')
            grad = module.weight.grad
            if grad is not None and bool(torch.any(grad != 0).item()):
                elig_nonzero += 1
        if elig_nonzero:
            nonzero_batch_count += 1
        for name, p in params.items():
            if not p.requires_grad:
                continue
            grad = p.grad
            if grad is None:
                continue
            total_grad_tensors += 1
            if not bool(torch.isfinite(grad).all().item()):
                raise RuntimeError(f'Nonfinite detection gradient at {name}, batch {idx}')
            if bool(torch.any(grad != 0).item()):
                nonzero_tensors += 1
            prod = (p.detach() * grad.detach()).float().cpu()
            absolute[name].add_(prod.abs_())
            signed[name].add_(prod)
        if nonzero_tensors == 0:
            raise RuntimeError(f'All parameter gradients missing/zero in labelled detection batch {idx}')
        c = _loss_components(components, torch)
        if any(not math.isfinite(float(x)) for x in c.values()):
            raise RuntimeError(f'Nonfinite loss component at batch {idx}')
        row = dict(batch=idx, images=bs, gt_objects=nobjects,
                   detection_loss=float(loss.detach().item()),
                   loss_per_image=float(per_image_loss.detach().item()),
                   nonzero_gradient_tensors=nonzero_tensors,
                   gradient_tensors=total_grad_tensors,
                   eligible_roots_with_nonzero_grad=elig_nonzero,
                   elapsed_seconds=round(time.monotonic() - start, 3),
                   components_json=__import__('json').dumps(c, sort_keys=True))
        logs.append(row)
        if idx == 1 or idx % 5 == 0 or idx == args.gradient_batches:
            logger.info('grad batch %d/%d | loss/img %.6g | gt %d | grad tensors %d/%d | eligible roots %d/%d',
                        idx, args.gradient_batches, row['loss_per_image'], nobjects,
                        nonzero_tensors, total_grad_tensors, elig_nonzero, len(eligible_roots))
    if len(logs) != args.gradient_batches:
        raise RuntimeError(f'Collected only {len(logs)}/{args.gradient_batches} train batches')
    if sum(row['gt_objects'] for row in logs) == 0:
        raise RuntimeError('All sampled train batches had zero GT objects; cannot claim representative detection-loss gradients')
    frac = nonzero_batch_count / len(logs)
    if frac < args.min_nonzero_gradient_fraction:
        raise RuntimeError(f'Only {frac:.1%} batches have gradients on eligible roots; required {args.min_nonzero_gradient_fraction:.1%}')
    for d in (signed, absolute):
        for v in d.values():
            v.div_(len(logs))
    net.zero_grad(set_to_none=True)
    return signed, absolute, logs, frac


def _mask_for(param, indices: list[int], axis: int):
    import torch
    if not indices or axis >= param.ndim:
        raise RuntimeError('Missing/out-of-range pruning axis')
    if min(indices) < 0 or max(indices) >= param.shape[axis]:
        raise RuntimeError(f'Out-of-range indices on shape {list(param.shape)}, axis {axis}: {indices[:10]}')
    mask = torch.zeros_like(param, dtype=torch.bool, device='cpu')
    slices = [slice(None)] * param.ndim
    slices[axis] = indices
    mask[tuple(slices)] = True
    return mask


def dependency_parameter_masks(group, dg, named_modules: dict, named_parameters: dict):
    """Deduplicated parameter index masks induced by a single-channel pruning group.

    Unsupported parameterized dependencies are NOT silently skipped.
    """
    import torch
    reverse = {mod: name for name, mod in named_modules.items()}
    masks = {}
    for dep, indices0 in group:
        mod = dep.target.module
        path = reverse.get(mod)
        indices = list(map(int, indices0))
        if path is None:
            # Virtual operators carry index mappings but no model parameters.
            if any(p is not None for p in getattr(mod, 'parameters', lambda **kwargs: [])(recurse=False)):
                raise RuntimeError(f'Parameterized unnamed virtual op unsupported: {type(mod).__name__}')
            continue
        is_out = dg.is_out_channel_pruning_fn(dep.handler)
        is_in = dg.is_in_channel_pruning_fn(dep.handler)
        if not (is_in or is_out):
            raise RuntimeError(f'Unknown dependency pruning handler: {path}')
        channels_axis = 0 if is_out else 1
        assignments = []
        if isinstance(mod, torch.nn.Conv2d):
            if mod.groups != 1:
                raise RuntimeError(f'Grouped/depthwise Conv scoring not yet certified: {path}')
            assignments.append((f'{path}.weight', channels_axis))
            if is_out and mod.bias is not None:
                assignments.append((f'{path}.bias', 0))
        elif isinstance(mod, torch.nn.Linear):
            assignments.append((f'{path}.weight', channels_axis))
            if is_out and mod.bias is not None:
                assignments.append((f'{path}.bias', 0))
        elif isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            if is_in and not is_out:
                raise RuntimeError(f'BN unexpectedly pruned on input only: {path}')
            if mod.affine:
                assignments += [(f'{path}.weight', 0), (f'{path}.bias', 0)]
        elif isinstance(mod, (torch.nn.ReLU, torch.nn.ReLU6, torch.nn.SiLU, torch.nn.Identity)):
            pass
        elif any(True for _ in mod.parameters(recurse=False)):
            raise RuntimeError(f'Unsupported parameterized module in dependency graph: {path} {type(mod).__name__}')
        for key, axis in assignments:
            if key not in named_parameters:
                raise RuntimeError(f'Missing parameter {key}')
            new = _mask_for(named_parameters[key].detach().cpu(), indices, axis)
            masks[key] = new if key not in masks else (masks[key] | new)
    if not masks:
        raise RuntimeError('Pruning group has no scoreable trainable parameters')
    return masks


def score_parameter_masks(masks: dict, accum_abs: dict, accum_signed: dict) -> dict:
    import torch
    total_abs, total_signed, count = 0.0, 0.0, 0
    for key, mask in masks.items():
        if key not in accum_abs or key not in accum_signed:
            raise RuntimeError(f'Missing gradient accumulation for dependent parameter: {key}')
        if mask.shape != accum_abs[key].shape:
            raise RuntimeError(f'Shape mismatch for dependent parameter: {key}')
        total_abs += float(accum_abs[key][mask].sum(dtype=torch.float64).item())
        total_signed += float(accum_signed[key][mask].sum(dtype=torch.float64).item())
        count += int(mask.sum().item())
    if count <= 0 or not (math.isfinite(total_abs) and math.isfinite(total_signed)):
        raise RuntimeError('Empty or nonfinite coupled Taylor score')
    return {'score_mean_abs_taylor': total_abs,
            'score_abs_mean_signed_taylor': abs(total_signed),
            'score_mean_abs_per_parameter': total_abs / count,
            'parameter_elements': count,
            'parameter_tensors': len(masks)}


def score_depgraph_candidates(net, args, tp, graph: dict, accum_signed: dict, accum_abs: dict, stage2, logger):
    """Score each singleton dependency group, then Stage 04 bundles into aligned 8-channel sets.

    Singletons are ranking probes ONLY, not permission to physically prune 1 channel.
    Requires every scored singleton to be contained in >=1 safe aligned block.
    """
    import torch
    net.train()
    net.requires_grad_(True)
    example = torch.zeros((1, 3, *args.imgsz_hw), device=next(net.parameters()).device, requires_grad=True)
    with torch.enable_grad():
        dg = tp.DependencyGraph().build_dependency(
            net, example_inputs=example,
            forward_fn=lambda model, images: stage2._output_scalar(model(images))[0], verbose=False)
    if not getattr(dg, 'module2node', None):
        raise RuntimeError('DepGraph returned no nodes while rescoring')
    named = dict(net.named_modules())
    params = dict(net.named_parameters())
    reverse = {m: name for name, m in named.items()}
    head_paths, attention_paths, static_paths = (
        stage2._head_paths(net), stage2._attention_paths(net), stage2._static_module_paths(net))
    protected = {'head_paths': head_paths, 'attention_paths': attention_paths, 'static_paths': static_paths}
    cfg = graph.get('_stage02_config')
    if cfg is None:
        raise RuntimeError('Missing Stage 02 configuration for exact hazard checks')
    config = {'head': cfg['protect_detect_head'], 'attention': cfg['protect_attention'],
              'static_splits': cfg['protect_static_splits']}
    align, min_channels = cfg['channel_align'], cfg['min_channels']
    from types import SimpleNamespace
    entries, skipped, arrays, total_candidates = [], [], {}, 0
    for info in graph['groups']:
        if info.get('feasibility') != 'ELIGIBLE_PROBE_ONLY':
            continue
        gid, root, width = info['group_id'], info['root'], int(info['channels_before'])
        mod = named.get(root)
        if not isinstance(mod, torch.nn.Conv2d) or mod.out_channels != width:
            raise RuntimeError(f'Stage 02 eligible root changed: {root}')
        safe_channel_ids = set()
        safe_blocks = []
        for start in range(0, width, align):
            block = list(range(start, min(start + align, width)))
            if len(block) != align:
                continue
            deps, reasons = stage2._safe_probe(dg, mod, block, reverse, protected, align, min_channels, config)
            if deps is not None and not reasons:
                safe_blocks.append(block)
                safe_channel_ids.update(block)
        if not safe_blocks:
            skipped.append({'group_id': gid, 'root': root, 'reason': 'NO_SAFE_ALIGNED_BLOCK_IN_RESCORED_DEPGRAPH'})
            logger.warning('No safe aligned block in %s; group excluded from importance', root)
            continue
        pruner = dg.get_pruner_of_module(mod)
        scores_abs = [float('nan')] * width
        scores_signed = [float('nan')] * width
        scores_norm = [float('nan')] * width
        for i in sorted(safe_channel_ids):
            try:
                group = dg.get_pruning_group(mod, pruner.prune_out_channels, [i])
                if not dg.check_pruning_group(group):
                    raise RuntimeError('DepGraph rejected singleton channel mapping')
                # Singleton is never physically pruned; alignment/min-channel
                # checks occur on safe blocks and Stage 04 exact selected sets.
                for dep, idxs in group:
                    obj = dep.target.module
                    nm = reverse.get(obj, f'<virtual:{type(obj).__name__}>')
                    if nm in head_paths or nm in attention_paths:
                        raise RuntimeError(f'Singleton crosses protected head/attention: {nm}')
                    if stage2._is_virtual(obj, reverse):
                        kind, known = stage2._special_operation(obj, reverse)
                        if not known or any(k in kind.lower() for k in stage2.VIRTUAL_DANGEROUS):
                            raise RuntimeError(f'Unverified singleton virtual mapping {kind}')
                masks = dependency_parameter_masks(group, dg, named, params)
                result = score_parameter_masks(masks, accum_abs, accum_signed)
                scores_abs[i] = result['score_mean_abs_taylor']
                scores_signed[i] = result['score_abs_mean_signed_taylor']
                scores_norm[i] = result['score_mean_abs_per_parameter']
                entries.append({'group_id': gid, 'root': root, 'channel': i, **result})
            except (ValueError, RuntimeError, IndexError) as exc:
                skipped.append({'group_id': gid, 'root': root, 'channel': i,
                                'reason': f'{type(exc).__name__}: {str(exc)[:240]}'})
        good = [i for i in range(width) if math.isfinite(scores_abs[i])]
        complete_blocks = [block for block in safe_blocks if all(math.isfinite(scores_abs[i]) for i in block)]
        if not complete_blocks:
            skipped.append({'group_id': gid, 'root': root, 'reason': 'NO_COMPLETE_SCORED_ALIGNED_BLOCK'})
        if not good:
            logger.warning('No scoreable channel in %s, marking group excluded', root)
        else:
            logger.info('Scored %s: %d/%d channels; %d aligned blocks feasible', root, len(good), width, len(safe_blocks))
        arrays[f'{gid}_abs'] = scores_abs
        arrays[f'{gid}_signed'] = scores_signed
        arrays[f'{gid}_normalized'] = scores_norm
        arrays[f'{gid}_allowed_block_starts'] = [block[0] for block in complete_blocks]
        total_candidates += len(good)
    usable_roots = sum(bool(v) for k, v in arrays.items() if k.endswith('_allowed_block_starts'))
    candidate_roots = len([g for g in graph['groups'] if g['feasibility'] == 'ELIGIBLE_PROBE_ONLY'])
    if not args.allow_unscored_roots and usable_roots != candidate_roots:
        raise RuntimeError(f'Only {usable_roots}/{candidate_roots} roots have a fully scored safe {align}-channel block; refuse partial PASS (use --allow-unscored-roots to explicitly protect remainder)')
    if usable_roots == 0:
        raise RuntimeError('No fully scored aligned pruning blocks')
    if total_candidates < 2:
        raise RuntimeError('Insufficient valid coupled per-channel Taylor scores (<2)')
    scores = [r['score_mean_abs_taylor'] for r in entries]
    if not all(math.isfinite(x) and x >= 0 for x in scores):
        raise RuntimeError('Nonfinite or negative Taylor scores')
    if max(scores) <= 0 or max(scores) - min(scores) <= max(1e-12, max(scores) * 1e-8):
        raise RuntimeError('Taylor scores all zero / undifferentiated, cannot rank')
    net.zero_grad(set_to_none=True)
    return entries, skipped, arrays, {'scored_channels': total_candidates,
                                       'scored_roots': len(set(r['group_id'] for r in entries)),
                                       'roots_with_complete_safe_blocks': usable_roots,
                                       'candidate_roots': candidate_roots,
                                       'skipped_records': len(skipped),
                                       'score_min': min(scores), 'score_max': max(scores)}


def write_csv(path: Path, rows: list, cols: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def run(args) -> int:
    out = resolved(args.output)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_log(out, STAGE, NAME)
    stage_dir = out / 'stages' / STAGE
    gatepath = gate_file(out, STAGE, NAME)
    if gatepath.exists() and not args.overwrite:
        logger.error('Existing 03 gate preserved: %s; use --overwrite to invalidate downstream gates', gatepath)
        return 2
    checks = {key: False for key in (
        'cli_valid', 'predecessor_gates_current', 'model_identity_unchanged',
        'train_only_official_dataloader', 'real_detection_loss_backward',
        'requested_train_batches_collected', 'eligible_root_gradients_nonzero',
        'depgraph_candidate_mapping_rebuilt', 'coupled_scores_finite_distinct',
        'artifacts_saved')}
    provenance, errors, warnings = {}, [], []
    artifacts = {}
    try:
        if args.overwrite:
            for p in (out / 'gates').glob('*.json'):
                if p.name[:2].isdigit() and int(p.name[:2]) >= 3:
                    p.unlink()
            logger.warning('Invalidated Stage 03..10 gates with --overwrite')
        if args.gradient_batches < 1 or args.batch < 1 or args.workers < 0 or args.seed < 0:
            raise ValueError('gradient-batches/batch must >=1, workers/seed >=0')
        if args.gradient_batch_size is not None and args.gradient_batch_size < 1:
            raise ValueError('gradient-batch-size must be >=1')
        if not (0 < args.min_nonzero_gradient_fraction <= 1):
            raise ValueError('min-nonzero-gradient-fraction must be in (0,1]')
        hw = make_imgsz(args.imgsz)
        if hw[0] != hw[1]:
            raise RuntimeError('Non-square imgsz unsupported by Stage 03 data preprocessing; fail safe')
        args.imgsz_hw = hw
        checks['cli_valid'] = True
        yaml_path = resolved(args.data)
        data = read_dataset(yaml_path, args.expected_classes)
        current = contract(args.model, yaml_path, hw, args.device, args.batch, args.workers,
                           args.family, args.expected_classes)
        g0, g1, g2, checkpoint, graph = ensure_predecessors(out, current, data)
        checks['predecessor_gates_current'] = True
        stage02_file, stage02_code_hash = load_stage02_helpers()
        provenance = {**g0['provenance'],
                      'stage00_gate_sha256': sha256_file(gate_file(out,'00','check_environment')),
                      'stage01_gate_sha256': sha256_file(gate_file(out,'01','profile_baseline')),
                      'stage02_gate_sha256': sha256_file(gate_file(out,'02','analyze_pruning_graph')),
                      'stage02_graph_sha256': sha256_file(out/'stages'/'02'/'02_graph.json'),
                      'stage02_source_sha256': stage02_code_hash,
                      'stage03_config': {'gradient_batches': args.gradient_batches,
                                         'gradient_batch_size': args.gradient_batch_size or args.batch,
                                         'seed': args.seed, 'score': 'E_b[sum_unique |w * grad(loss / batch_size)|]',
                                         'augmentation': 'train mode with core geometric/color augmentation flags disabled; Ultralytics default Albumentations may remain',
                                         'allow_unscored_roots': args.allow_unscored_roots}}
        graph['_stage02_config'] = g2['provenance']['stage02_config']
        from ultralytics import YOLO
        import torch
        import numpy as np
        import torch_pruning as tp
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        device = select_device(args.device)
        yolo = YOLO(str(checkpoint))
        net = yolo.model.to(device).float()
        if getattr(yolo,'task',None) != 'detect':
            raise RuntimeError('Only YOLO detection checkpoints are supported')
        meta = model_head_metadata(net)
        if len(getattr(yolo,'names',{})) != data['nc'] or meta.get('head_nc') != data['nc']:
            raise RuntimeError('Model/dataset number of classes mismatch')
        if g0['provenance']['model_family'] != graph['family']:
            raise RuntimeError('Stage 02 family differs from checkpoint')
        checks['model_identity_unchanged'] = True
        trainer, loader = _dataset_loader(checkpoint, yaml_path, net, args, data, device, logger)
        checks['train_only_official_dataloader'] = True
        signed, absolute, gradlog, validfrac = collect_real_gradients(net, trainer, loader, args, logger, graph)
        checks['real_detection_loss_backward'] = True
        checks['requested_train_batches_collected'] = len(gradlog) == args.gradient_batches
        checks['eligible_root_gradients_nonzero'] = validfrac >= args.min_nonzero_gradient_fraction
        logger.info('Rebuilding autograd DepGraph to compute coupled importance indices ...')
        entries, skipped, arrays, summary = score_depgraph_candidates(net,args,tp,graph,signed,absolute,stage02_file,logger)
        checks['depgraph_candidate_mapping_rebuilt'] = (summary['roots_with_complete_safe_blocks'] > 0 and
            (args.allow_unscored_roots or summary['roots_with_complete_safe_blocks'] == summary['candidate_roots']))
        checks['coupled_scores_finite_distinct'] = summary['scored_channels'] >= 2
        result = {'stage':'03_collect_gradient_importance', 'timestamp_utc': now_utc(),
                  'model_family':graph['family'],'model_scale':graph['scale'],
                  'source':'TRAIN_SPLIT_ONLY', 'loss':'REAL_ULTRALYTICS_DETECTION_LOSS',
                  'gradient_batches_requested':args.gradient_batches,
                  'gradient_batches_collected':len(gradlog),
                  'training_images_observed':sum(x['images'] for x in gradlog),
                  'training_objects_observed':sum(x['gt_objects'] for x in gradlog),
                  'nonzero_eligible_root_batch_fraction':validfrac,
                  'score_definition':provenance['stage03_config']['score'],
                  'summary':summary, 'groups':[], 'skipped':skipped,
                  'note':'Singleton scores guide ranking ONLY; Stage 04 must prove exact aligned (C%8) groups via DepGraph'}
        group_info = {g['group_id']:g for g in graph['groups']}
        for gid in sorted({r['group_id'] for r in entries}):
            records = [r for r in entries if r['group_id'] == gid]
            g = group_info[gid]
            result['groups'].append({'group_id':gid, 'root':g['root'],
                                     'channels_before':g['channels_before'],
                                     'scored_channels':len(records),
                                     'score_min':min(r['score_mean_abs_taylor'] for r in records),
                                     'score_max':max(r['score_mean_abs_taylor'] for r in records),
                                     'score_mean':sum(r['score_mean_abs_taylor'] for r in records)/len(records),
                                     'npz_keys':[f'{gid}_abs',f'{gid}_signed',f'{gid}_normalized',f'{gid}_allowed_block_starts']})
        out_json=stage_dir/'03_importance.json'
        out_npz=stage_dir/'03_importance.npz'
        out_grad=stage_dir/'03_grad_log.csv'
        out_scores=stage_dir/'03_channel_scores.csv'
        out_skip=stage_dir/'03_unscored_channels.csv'
        write_json(out_json, result)
        np.savez_compressed(out_npz, **{name:np.asarray(vals,dtype=np.float64 if not name.endswith('allowed_block_starts') else np.int32) for name,vals in arrays.items()})
        write_csv(out_grad, gradlog, ['batch','images','gt_objects','detection_loss','loss_per_image','nonzero_gradient_tensors','gradient_tensors','eligible_roots_with_nonzero_grad','elapsed_seconds','components_json'])
        write_csv(out_scores, entries, ['group_id','root','channel','score_mean_abs_taylor','score_abs_mean_signed_taylor','score_mean_abs_per_parameter','parameter_elements','parameter_tensors'])
        write_csv(out_skip, skipped, ['group_id','root','channel','reason'])
        checks['artifacts_saved']=all(p.exists() and p.stat().st_size>0 for p in (out_json,out_npz,out_grad,out_scores,out_skip))
        artifacts={k:str(v) for k,v in [('importance_json',out_json),('importance_npz',out_npz),('grad_log',out_grad),('scores_csv',out_scores),('unscored_csv',out_skip)]}
        provenance['artifact_hashes']={key:sha256_file(Path(path)) for key,path in artifacts.items()}
        if skipped:
            warnings.append(f'{len(skipped)} singleton/root mappings conservatively unscored; see 03_unscored_channels.csv')
        warnings.append('Stage 03 scores are a ranking heuristic, NOT permission to prune singleton channels or bypass Stage 04 graph/alignment checks')
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
        logger.error('FAIL: %s',errors[-1]); logger.debug('Traceback: %s',traceback.format_exc())
        # save the traceback separately for debugging; gate remains compact
        (stage_dir/'03_error_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        gate=save_gate(out,STAGE,NAME,checks=checks,errors=errors,warnings=warnings,provenance=provenance,artifacts=artifacts)
        write_json(stage_dir/'03_gate.json',gate)
        logger.info('[03] %s — %s',gate['status'],gatepath)
    return 0 if gate['status']=='PASS' else 1


if __name__=='__main__':
    sys.exit(run(make_parser().parse_args()))
