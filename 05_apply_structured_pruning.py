#!/usr/bin/env python3
"""Stage 05 — Replay exact Stage 04 DepGraph decisions on a FP32 YOLO model.

Supports Ultralytics YOLOv8 / YOLO11 / YOLO26 detection models, n/s/m/l,
provided Stage 02-04 previously certified this exact model/run.

No gradient ranking here; no new channel decisions; original checkpoint is
immutable. Uses current_indices from the plan and verifies EVERY coupled
pruning mapping before physically applying it. Writes real, reloadable .pt.

IMPORTANT: Stage 05 PASS is NOT equivalent to QAT-ready or recovery-ready.
Stages 06/07/09 must separately verify serialization, backward, fine-tuning,
and the actual QAT pipeline.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import tempfile
import traceback
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dataset_legacy_bridge import (create_legacy_bridge, create_verified_bridge,
                                   validate_legacy_bridge, scoped_legacy_stage04_guard)
from pruning_common import (
    SUPPORTED_FAMILIES, contract, forward_tensors, gate_file, hash_object, make_imgsz,
    model_head_metadata, now_utc, read_dataset, read_json, resolved,
    save_gate, select_device, setup_log, sha256_file, write_json,
)

STAGE, NAME = '05', 'apply_structured_pruning'
PREDECESSOR = ('04', 'plan_structured_pruning')
TRACE_FIELDS = [
    'decision', 'iterative_stage', 'group_id', 'root', 'status',
    'original_indices', 'current_indices', 'channels_before', 'channels_after',
    'dependency_count', 'affected_modules', 'params_before', 'params_after',
    'params_removed', 'macs_before', 'macs_after', 'macs_removed',
    'weight_shape_changes', 'dependency_mapping_json',
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='05: physically replay Stage04 DepGraph plan; only PASS with exact dependency/resource verification')
    for flag in ('model', 'data', 'output'):
        p.add_argument('--' + flag, required=True)
    p.add_argument('--imgsz', nargs='+', type=int, default=[640])
    p.add_argument('--device', default='0')
    p.add_argument('--batch', type=int, default=8, help='Stage00-04 contract value, not DepGraph batch size')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--expected-classes', type=int, default=None)
    p.add_argument('--family', default='auto', choices=['auto', *SUPPORTED_FAMILIES])
    p.add_argument('--channel-align', type=int, default=None, help='Optional assertion; must match Stage04 plan')
    p.add_argument('--min-channels', type=int, default=None, help='Optional assertion; must match Stage04 plan')
    p.add_argument('--profile-every', type=int, default=1,
                   help='Measure MACs every N decisions; default 1 compares every decision to Stage04 plan')
    p.add_argument('--verify-reload', action=argparse.BooleanOptionalAction, default=True,
                   help='Reload final pt using Ultralytics and compare parameter shapes in-process')
    p.add_argument('--legacy-cache-path', default=None,
                   help='EXPLICIT legacy exception: existing .cache path whose metadata changed since Stage00')
    p.add_argument('--acknowledge-unverifiable-legacy-fingerprint', action='store_true',
                   help='Acknowledge Stage00 global hash cannot prove only cache changed; pin current non-cache content SHA256')
    p.add_argument('--overwrite', action='store_true',
                   help='Replace Stage05 outputs and invalidate downstream gates; never changes Stage00-04')
    return p


def load_local_step(filename: str, stem: str):
    path = Path(__file__).with_name(filename)
    if not path.is_file():
        raise FileNotFoundError(f'Required unchanged predecessor source not found: {path}')
    spec = importlib.util.spec_from_file_location(stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import required stage helper: {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, sha256_file(path)


def verify_stage04(out: Path, request: dict, dataset: dict, stage4,
                   *, legacy_bridge: dict | None = None) -> tuple[dict, dict, dict, Path, dict]:
    """Read ALL ancestors with unchanged Stage04 verifier; explicit cache waiver only."""
    if legacy_bridge is None:
        ancestor_gates, graph, importance, checkpoint, other = stage4.require_ancestors(out, request, dataset)
    else:
        gate0_path = gate_file(out, '00', 'check_environment')
        gate0 = read_json(gate0_path)
        with scoped_legacy_stage04_guard(stage4, dataset, gate0,
                                        sha256_file(gate0_path), legacy_bridge):
            ancestor_gates, graph, importance, checkpoint, other = stage4.require_ancestors(out, request, dataset)
    gate_path = gate_file(out, *PREDECESSOR)
    if not gate_path.is_file():
        raise RuntimeError('Stage04 gate missing; Stage05 cannot proceed')
    gate = read_json(gate_path)
    if gate.get('step') != '04_plan_structured_pruning' or gate.get('status') != 'PASS':
        raise RuntimeError(f'Stage04 must PASS, observed {gate.get("status")}')
    if not gate.get('checks') or not all(v is True for v in gate['checks'].values()):
        raise RuntimeError('Stage04 PASS does not certify every mandatory check')
    g04p = gate.get('provenance', {})
    if g04p.get('contract_sha256') != hash_object(request):
        raise RuntimeError('Stage04 contract differs from Stage05 CLI')
    for n, name in stage4.GATES:
        if g04p.get(f'stage{n}_gate_sha256') != sha256_file(gate_file(out, n, name)):
            raise RuntimeError(f'Ancestor gate {n} modified after Stage04')
    src2 = Path(__file__).with_name('02_analyze_pruning_graph.py')
    if g04p.get('stage02_source_sha256') != sha256_file(src2):
        raise RuntimeError('Stage02 safety source differs from Stage04 version')
    for key, rel in (
        ('stage02_graph_sha256', 'stages/02/02_graph.json'),
        ('stage03_importance_sha256', 'stages/03/03_importance.json'),
        ('stage03_scores_sha256', 'stages/03/03_channel_scores.csv'),
    ):
        path = out / rel
        if not path.is_file() or g04p.get(key) != sha256_file(path):
            raise RuntimeError(f'Changed/missing upstream artifact {rel}')
    plan_path = out / 'stages' / '04' / '04_plan.json'
    change_path = out / 'stages' / '04' / '04_channel_changes.csv'
    recorded = g04p.get('artifact_hashes', {})
    for key, path in [('plan', plan_path), ('channel_changes', change_path)]:
        if not path.is_file() or recorded.get(key) != sha256_file(path):
            raise RuntimeError(f'Stage04 {key} changed since PASS')
    plan = read_json(plan_path)
    if plan.get('step') != '04_plan_structured_pruning':
        raise RuntimeError('Malformed Stage04 plan: wrong step')
    if plan.get('checkpoint_sha256') != sha256_file(checkpoint):
        raise RuntimeError('Stage04 plan checkpoint identity mismatch')
    if plan.get('stage02_graph_sha256') != sha256_file(out / 'stages/02/02_graph.json'):
        raise RuntimeError('Stage04 graph identity mismatch')
    if plan.get('stage03_scores_sha256') != sha256_file(out / 'stages/03/03_channel_scores.csv'):
        raise RuntimeError('Stage04 gradient score identity mismatch')
    if plan.get('source_checkpoint') != str(checkpoint):
        raise RuntimeError('Stage04 plan source path mismatch')
    if plan.get('model_family') != graph.get('family'):
        raise RuntimeError('Stage04 plan model family mismatch')
    if list(plan.get('imgsz_hw', [])) != list(request['imgsz_hw']):
        raise RuntimeError('Stage04 plan input shape mismatch')
    pcfg = g04p.get('stage04_config', {})
    target = plan.get('target', {})
    for a, b in [('target_metric', 'metric'), ('target_reduction', 'reduction'),
                 ('target_tolerance', 'tolerance'), ('allow_partial_target', 'allow_partial_target')]:
        if pcfg.get(a) != target.get(b):
            raise RuntimeError(f'Stage04 gate/plan target mismatch: {a}/{b}')
    if plan.get('target_status') not in ('TARGET_REACHED', 'PARTIAL_TARGET'):
        raise RuntimeError('Stage04 unknown target status')
    if plan['target_status'] == 'PARTIAL_TARGET' and target.get('allow_partial_target') is not True:
        raise RuntimeError('PARTIAL_TARGET cannot pass Stage05 without explicit Stage04 --allow-partial-target')
    if plan.get('channel_align') != pcfg.get('channel_align') or plan.get('min_channels') != pcfg.get('min_channels'):
        raise RuntimeError('Stage04 constraint settings inconsistent')
    if plan.get('stage02_safety_policy') != ancestor_gates[2]['provenance']['stage02_config']:
        raise RuntimeError('Stage02 protective constraints modified in Stage04')
    return gate, plan, graph, checkpoint, {'stage04_gate_sha256': sha256_file(gate_path),
                                          'stage04_plan_sha256': sha256_file(plan_path),
                                          'stage04_csv_sha256': sha256_file(change_path)}


def validate_plan(plan: dict) -> dict:
    """Structural + exact lineage self-consistency before touching the model."""
    decisions = plan.get('decisions', [])
    if not isinstance(decisions, list) or not decisions:
        raise RuntimeError('Stage04 has no pruning decisions')
    if len(decisions) > 100000:
        raise RuntimeError('Unreasonably large pruning decision list')
    align = int(plan['channel_align'])
    min_channels = int(plan['min_channels'])
    if align < 1 or min_channels < 1 or min_channels % align:
        raise RuntimeError('Malformed channel alignment/minimum')
    lineage: dict[str, list[int]] = {}
    gid_to_root: dict[str, str] = {}
    owner: dict[str, str] = {}
    seen_decision = set()
    last_stage = 0
    for n, d in enumerate(decisions, 1):
        if d.get('decision') != n or d.get('status') != 'SIMULATED_VALID':
            raise RuntimeError(f'Decision ordering/status invalid at #{n}')
        stage = int(d.get('iterative_stage', 0))
        if stage < 1 or stage < last_stage:
            raise RuntimeError(f'Non-monotonic iterative_stage at decision {n}')
        last_stage = stage
        root, gid = d.get('root'), d.get('group_id')
        if not isinstance(root, str) or not isinstance(gid, str) or not root or not gid:
            raise RuntimeError(f'Missing root/group_id at decision {n}')
        key = (gid, root)
        if gid in gid_to_root and gid_to_root[gid] != root:
            raise RuntimeError(f'One group_id mapped to two roots: {gid}')
        gid_to_root[gid] = root
        seen_decision.add(key)
        if gid not in lineage:
            old = int(d['channels_before'])
            lineage[gid] = list(range(old))
        old_lineage = lineage[gid]
        oi = d.get('original_indices', [])
        ci = d.get('current_indices', [])
        if not (isinstance(oi, list) and isinstance(ci, list) and oi and len(oi) == len(ci)):
            raise RuntimeError(f'Incomplete exact indices decision {n}')
        if len(oi) % align or len(oi) != len(set(oi)) or len(ci) != len(set(ci)):
            raise RuntimeError(f'Invalid removal block or duplicated indices decision {n}')
        if any(not isinstance(i, int) or i < 0 for i in oi + ci):
            raise RuntimeError(f'Invalid nonnegative channel indices decision {n}')
        if int(d['channels_before']) != len(old_lineage):
            raise RuntimeError(f'Inconsistent root original lineage at decision {n}')
        if any(i >= len(old_lineage) for i in ci) or [old_lineage[i] for i in ci] != oi:
            raise RuntimeError(f'Current→original indices mismatch at decision {n}')
        remain = [x for x in old_lineage if x not in set(oi)]
        if len(remain) != int(d['channels_after']) or len(remain) < min_channels or len(remain) % align:
            raise RuntimeError(f'Illegal root channel count after decision {n}')
        lineage[gid] = remain
        if (int(d['actual_params_before']) - int(d['actual_params_after']) != int(d['actual_param_saving'])
            or int(d['actual_param_saving']) <= 0):
            raise RuntimeError(f'Invalid physical parameters accounting at decision {n}')
        if (float(d['actual_macs_before']) < float(d['actual_macs_after'])
            or not math.isfinite(float(d['actual_macs_after']))):
            raise RuntimeError(f'Invalid MAC accounting at decision {n}')
        if n > 1:
            prev = decisions[n - 2]
            if int(prev['actual_params_after']) != int(d['actual_params_before']):
                raise RuntimeError(f'Noncontiguous parameter trace at decision {n}')
            if not math.isclose(float(prev['actual_macs_after']), float(d['actual_macs_before']), rel_tol=1e-8, abs_tol=64):
                raise RuntimeError(f'Noncontiguous MAC trace at decision {n}')
        if not isinstance(d.get('dependencies'), list) or not d['dependencies']:
            raise RuntimeError(f'Empty dependency list at decision {n}')
        for op in d['dependencies']:
            if not {'module', 'module_type', 'prune_kind', 'indices', 'is_virtual'} <= set(op):
                raise RuntimeError(f'Incomplete dependency specification at decision {n}')
            if not isinstance(op['indices'], list) or not op['indices']:
                raise RuntimeError(f'Empty dependency indices at decision {n}')
            if not op['is_virtual']:
                name = op['module']
                if name in owner and owner[name] != gid:
                    raise RuntimeError(f'Cross-root unsafe overlapping module {name} at decision {n}')
                owner[name] = gid
    base, projected = plan['baseline'], plan['projected']
    if int(decisions[0]['actual_params_before']) != int(base['params']):
        raise RuntimeError('Baseline plan params mismatch')
    if int(decisions[-1]['actual_params_after']) != int(projected['params']):
        raise RuntimeError('Final plan params mismatch')
    if not math.isclose(float(decisions[-1]['actual_macs_after']), float(projected['macs']), rel_tol=1e-8, abs_tol=128):
        raise RuntimeError('Final plan MACs mismatch')
    audit_lineage = plan.get('audit', {}).get('root_original_to_current_remaining', {})
    for gid, remain in lineage.items():
        if gid in audit_lineage and audit_lineage[gid] != remain:
            raise RuntimeError(f'Stage04 plan lineage audit differs: {gid}')
    expected_removed = sum(len(d['original_indices']) for d in decisions)
    if expected_removed != projected['removed_root_channels']:
        raise RuntimeError('Plan removed channels mismatch')
    return {'decisions': len(decisions), 'distinct_roots': len(seen_decision),
            'iterative_stages': last_stage, 'removed_root_channels': expected_removed,
            'root_lineage': lineage, 'module_owners': owner}


def current_conv_shapes(net) -> dict[str, dict]:
    import torch
    return {
        name: {'in': int(module.in_channels), 'out': int(module.out_channels),
               'groups': int(module.groups), 'weight': list(module.weight.shape)}
        for name, module in net.named_modules() if isinstance(module, torch.nn.Conv2d)
    }


def verify_stage04_alignment_from_snapshot(baseline_snapshot: dict, net, stage2, stage4,
                                            align: int, minimum: int):
    """Bridge Stage05's rich Conv snapshots to Stage04's (in, out) contract.

    Stage05 snapshots contain {'in', 'out', 'groups', 'weight'}; Stage04's
    _verify_changed_conv_alignment accepts only (in_channels, out_channels).
    Feeding dictionaries to that function falsely marks *all* Convs as changed,
    including protected head modules and pre-existing RGB/8-channel exceptions.
    This adapter does not relax any safety rule: it passes exactly the original
    widths so Stage04 checks only channels which physically changed.
    """
    widths = {}
    for name, old in baseline_snapshot.items():
        if not isinstance(old, dict) or not {'in', 'out'} <= set(old):
            raise RuntimeError(f'Malformed baseline Conv snapshot for alignment: {name}')
        ci, co = old['in'], old['out']
        if not (isinstance(ci, int) and isinstance(co, int) and ci > 0 and co > 0):
            raise RuntimeError(f'Invalid baseline Conv channels: {name}')
        widths[name] = (ci, co)
    return stage4._verify_changed_conv_alignment(widths, net, stage2, align, minimum)


def current_bn_shapes(net) -> dict[str, int]:
    import torch
    return {name: int(mod.num_features) for name, mod in net.named_modules()
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)}



@contextmanager
def preserve_batchnorm_buffers(net):
    """Ensure synthetic train-mode probes cannot corrupt learned BN statistics.

    Training forwards of zero-filled inputs update running_mean, running_var
    and num_batches_tracked even though no optimizer step occurs.  Stage 05
    replays many such forwards. Snapshot/restore around each *probe* only,
    rather than freezing all BN or altering the actual pruning operation.

    A physical prune must happen OUTSIDE this context because BN buffer shapes
    can change during a prune operation.
    """
    import torch
    snapshots = []
    for module in net.modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        fields = {}
        for field in ('running_mean', 'running_var', 'num_batches_tracked'):
            buf = getattr(module, field, None)
            if buf is not None:
                fields[field] = buf.detach().clone()
        snapshots.append((module, fields))
    try:
        yield
    finally:
        with torch.no_grad():
            for module, fields in snapshots:
                for field, before in fields.items():
                    current = getattr(module, field, None)
                    if current is None or current.shape != before.shape:
                        raise RuntimeError(
                            f'BN buffer shape unexpectedly changed during synthetic probe: '
                            f'{field}; do not perform physical pruning inside '
                            f'preserve_batchnorm_buffers()')
                    current.copy_(before)


def detection_output_schema(net, hw, device) -> list[dict]:
    """Shape-only signature of native training-mode detector outputs."""
    import torch
    was_training = net.training
    net.train()
    try:
        with torch.no_grad():
            raw = net(torch.zeros((1, 3, *hw), device=device, dtype=torch.float32))
        ts = forward_tensors(raw)
        if any(not item['finite'] for item in ts):
            raise RuntimeError('Detector produced non-finite tensor during ABI schema check')
        return [{'path': item['path'], 'shape': item['shape'], 'dtype': item['dtype']} for item in ts]
    finally:
        net.train(was_training)


def changed_shapes(before: dict, after: dict) -> dict:
    changes = {}
    for name, old in before.items():
        new = after.get(name)
        if new is None:
            raise RuntimeError(f'Conv disappeared after pruning: {name}')
        if old != new:
            changes[name] = {'before': old, 'after': new}
    return changes


def exact_dependency_replay(dg, net, stage2, stage4, decision: dict, align: int, minimum: int,
                            protected_config: dict):
    """Verify exact DepGraph *including virtual Concat/Add indices*, no silent tolerance."""
    import torch
    named = dict(net.named_modules())
    root = named.get(decision['root'])
    if not isinstance(root, torch.nn.Conv2d):
        raise RuntimeError(f'Missing Conv2d root: {decision["root"]}')
    if int(root.out_channels) != int(decision['channels_before']):
        raise RuntimeError(f'Pruning root width differs from plan {decision["root"]}: '
                           f'{root.out_channels} vs {decision["channels_before"]}')
    context, flags = stage4.protected_context(net, stage2, protected_config)
    deps, reasons = stage2._safe_probe(
        dg, root, list(decision['current_indices']),
        {m: name for name, m in named.items()}, context,
        align, minimum, flags)
    if deps is None or reasons:
        raise RuntimeError(f'Exact graph probe rejected #{decision["decision"]}: {reasons}')
    actual_ops = stage4.dependency_ops(deps)
    if actual_ops != decision['dependencies']:
        # Preserve correctness over compression: even a changed virtual Concat index
        # may silently select the wrong consumer weight slice.
        raise RuntimeError(f'DepGraph coupled indices differ from Stage04 plan at decision '
                           f'{decision["decision"]}, root={decision["root"]}. '
                           f'Expected={decision["dependencies"]}, observed={actual_ops}')
    pruner = dg.get_pruner_of_module(root)
    group = dg.get_pruning_group(root, pruner.prune_out_channels, list(decision['current_indices']))
    if not dg.check_pruning_group(group):
        raise RuntimeError(f'DepGraph rejects decision {decision["decision"]}')
    return group, actual_ops


def copy_module_dims(module) -> dict:
    import torch
    if isinstance(module, torch.nn.Conv2d):
        return {'type':'Conv2d','in':int(module.in_channels),'out':int(module.out_channels)}
    if isinstance(module, torch.nn.Linear):
        return {'type':'Linear','in':int(module.in_features),'out':int(module.out_features)}
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        return {'type':'BatchNorm','out':int(module.num_features)}
    raise RuntimeError(f'Unverified parameter dependency type {type(module).__name__}')


def verify_dependency_width_snapshots(before: dict, after: dict, deps: list[dict]):
    """Verify actual channel dimension shrank by number of exact unique indices."""
    edits: dict[tuple[str,str],set[int]] = defaultdict(set)
    for op in deps:
        if not op['is_virtual']:
            edits[(op['module'],op['prune_kind'])].update(int(i) for i in op['indices'])
    if set(before)!=set(after):
        raise RuntimeError('Physical pruning removed/renamed a protected dependency module')
    for (name,axis),idx in edits.items():
        old,new=before[name],after[name]
        if old['type']!=new['type'] or axis not in old or axis not in new:
            raise RuntimeError(f'Invalid coupled dependency dimension: {name}.{axis}')
        if old[axis]-new[axis]!=len(idx):
            raise RuntimeError(f'Coupled channel count incorrect {name}.{axis}: '
                               f'{old[axis]}->{new[axis]} but expected -{len(idx)}')


def verify_resources(expected: float, actual: float, what: str, rel_tol: float = 1e-7, abs_tol: float = 256):
    if not math.isfinite(float(actual)) or not math.isclose(float(expected), float(actual), rel_tol=rel_tol, abs_tol=abs_tol):
        raise RuntimeError(f'Stage04 planned {what}={expected}, Stage05 measured {actual}')


def atomic_pt_save(path: Path, net, train_args: dict, metadata: dict):
    """Write a normal Ultralytics .pt with the *full structurally pruned model*.

    The original model object remains Ultralytics-native; no __main__ custom
    modules or state_dict-only checkpoint. Reload testing is separate.
    """
    import torch
    import ultralytics
    path.parent.mkdir(parents=True, exist_ok=True)
    # Move the live model to CPU only for checkpoint serialization, then return
    # it to the same GPU; no extra full model in GPU VRAM needed.
    device = next(net.parameters()).device
    old_training = net.training
    net.cpu().eval()
    try:
        checkpoint = {
            'epoch': -1, 'best_fitness': None, 'model': net,
            'ema': None, 'updates': 0, 'optimizer': None,
            'train_args': dict(train_args), 'train_metrics': {}, 'train_results': {},
            'date': now_utc(), 'version': ultralytics.__version__,
            'pruning_stage': 5, 'pruning_manifest': metadata,
        }
        fd, temp_name = tempfile.mkstemp(prefix=path.stem + '_', suffix='.tmp', dir=path.parent)
        os.close(fd)
        try:
            torch.save(checkpoint, temp_name)
            if not Path(temp_name).is_file() or Path(temp_name).stat().st_size == 0:
                raise RuntimeError('Empty output checkpoint')
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    finally:
        net.to(device)
        net.train(old_training)
    return sha256_file(path)


def write_trace(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=TRACE_FIELDS, extrasaction='ignore')
        w.writeheader()
        for row in records:
            rec = dict(row)
            for k in ('original_indices', 'current_indices'):
                rec[k] = ','.join(map(str, rec[k]))
            rec['affected_modules'] = ';'.join(rec['affected_modules'])
            rec['weight_shape_changes'] = json.dumps(rec['weight_shape_changes'], sort_keys=True)
            rec['dependency_mapping_json'] = json.dumps(rec['dependency_mapping_json'], separators=(',', ':'))
            w.writerow(rec)


def run(args: argparse.Namespace) -> int:
    out = resolved(args.output)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_log(out, STAGE, NAME)
    stage_dir = out / 'stages' / STAGE
    stage_dir.mkdir(parents=True, exist_ok=True)
    gatepath = gate_file(out, STAGE, NAME)
    if gatepath.exists() and not args.overwrite:
        logger.error('Existing Stage05 gate preserved; add --overwrite to replace: %s', gatepath)
        return 2
    checks = dict.fromkeys([
        'cli_valid', 'predecessor_gates_current', 'plan_identity_and_lineage_valid',
        'model_identity_unchanged', 'all_exact_depgraph_groups_verified',
        'physical_weight_shapes_changed', 'resources_match_plan',
        'graph_and_alignment_valid', 'staged_checkpoints_saved',
        'final_checkpoint_replay_complete', 'artifacts_saved'], False)
    errors: list[str] = []
    warnings: list[str] = []
    provenance: dict = {}
    artifacts: dict = {}
    plan = None
    try:
        if args.overwrite:
            for stale in stage_dir.glob('05_*'):
                if stale.is_file():
                    stale.unlink()
            # Do not touch any earlier stage. Do not remove old Stage05 gate until
            # the new inputs have been validated (but invalidate downstream now).
            for gate in (out / 'gates').glob('*.json'):
                if gate.name[:2].isdigit() and int(gate.name[:2]) >= 5:
                    gate.unlink()
            logger.warning('--overwrite invalidated Stage05..10 gates, kept Stage00..04 intact')
        hw = make_imgsz(args.imgsz)
        if args.batch <= 0 or args.workers < 0 or args.profile_every < 1:
            raise ValueError('batch >0, workers >=0, profile-every >=1 required')
        checks['cli_valid'] = True
        data_path = resolved(args.data)
        data = read_dataset(data_path, args.expected_classes)
        request = contract(args.model, data_path, hw, args.device, args.batch,
                           args.workers, args.family, args.expected_classes)
        step4, step4_sha = load_local_step('04_plan_structured_pruning.py', 'pruning_stage04_replay')
        step2, step2_sha = load_local_step('02_analyze_pruning_graph.py', 'pruning_stage02_replay')
        # Historical Stage00 persisted just one aggregate stat fingerprint, not
        # a per-file inventory. Do NOT silently assume a changed hash means only
        # a cache changed. Require two explicit user options and a complete
        # current content-hash baseline before narrowly waiving that one check.
        gate0_path = gate_file(out, '00', 'check_environment')
        gate0 = read_json(gate0_path)
        original_fp = gate0.get('provenance', {}).get('dataset_manifest', {})
        import pruning_common as common
        now_fp = common.dataset_fingerprint(data)
        legacy_bridge = None
        if now_fp['manifest_sha256'] != original_fp.get('manifest_sha256'):
            if not (args.legacy_cache_path and args.acknowledge_unverifiable_legacy_fingerprint):
                raise RuntimeError('Dataset inventory changed since Stage 00. Legacy cache migration requires BOTH --legacy-cache-path PATH and --acknowledge-unverifiable-legacy-fingerprint; otherwise original ancestry remains strictly enforced.')
            legacy_bridge = create_legacy_bridge(data, gate0, sha256_file(gate0_path), args.legacy_cache_path)
            bridge_path = stage_dir / '05_dataset_legacy_bridge.json'
            write_json(bridge_path, legacy_bridge)
            logger.warning('LEGACY DATASET PROVENANCE GAP acknowledged: original aggregate hash cannot prove only cache changed. Non-cache file contents pinned by SHA256; Stage00..04 remain untouched.')
            warnings.append('EXPLICIT_LEGACY_DATASET_CACHE_WAIVER: historical non-cache content identity is unprovable from Stage00 global stat hash; new SHA256 baseline pinned')
        else:
            if args.legacy_cache_path or args.acknowledge_unverifiable_legacy_fingerprint:
                raise RuntimeError('Legacy waiver flags provided but dataset matches Stage00; remove both flags')
            legacy_bridge = create_verified_bridge(data, gate0, sha256_file(gate0_path))
            write_json(stage_dir / '05_dataset_legacy_bridge.json', legacy_bridge)
        gate4, plan, graph, source, receipts = verify_stage04(out, request, data, step4,
                                                          legacy_bridge=legacy_bridge)
        checks['predecessor_gates_current'] = True
        checks['model_identity_unchanged'] = True
        if args.channel_align is not None and int(args.channel_align) != int(plan['channel_align']):
            raise RuntimeError('Stage05 --channel-align must equal Stage04 plan')
        if args.min_channels is not None and int(args.min_channels) != int(plan['min_channels']):
            raise RuntimeError('Stage05 --min-channels must equal Stage04 plan')
        plandiag = validate_plan(plan)
        checks['plan_identity_and_lineage_valid'] = True
        logger.info('Plan validated: %d decisions, %d roots, %d stages, %s (target %.2f%% achieved %.2f%%)',
                    plandiag['decisions'], plandiag['distinct_roots'], plandiag['iterative_stages'],
                    plan['target_status'], float(plan['target']['reduction'])*100,
                    float(plan['projected']['target_metric_reduction'])*100)
        provenance = {
            **gate4['provenance'], 'stage04_gate_sha256': receipts['stage04_gate_sha256'],
            'stage04_plan_sha256': receipts['stage04_plan_sha256'],
            'stage04_source_sha256': step4_sha, 'stage02_source_sha256': step2_sha,
            'stage05_config': {'profile_every': args.profile_every,
                               'verify_reload': args.verify_reload},
        }
        if legacy_bridge is not None:
            bridge_path = stage_dir / '05_dataset_legacy_bridge.json'
            provenance['dataset_legacy_bridge_sha256'] = sha256_file(bridge_path)
            provenance['dataset_legacy_bridge_policy'] = legacy_bridge['policy']
            provenance['dataset_legacy_bridge_path'] = str(bridge_path)
        import torch
        import torch_pruning as tp
        from ultralytics import YOLO
        device = select_device(args.device)
        yolo = YOLO(str(source))
        if yolo.task != 'detect' or len(yolo.names) != data['nc']:
            raise RuntimeError('Only detection models with original expected class count supported')
        net = yolo.model.to(device).float().train().requires_grad_(True)
        meta = model_head_metadata(net)
        if meta.get('head_nc') != data['nc']:
            raise RuntimeError('Detection head output classes changed')
        baseline_shapes = current_conv_shapes(net)
        baseline_bn_shapes = current_bn_shapes(net)
        baseline_head = dict(meta)
        with preserve_batchnorm_buffers(net):
            baseline_outputs = detection_output_schema(net, hw, device)
        pbase, mbase = step4._profile(net, tp, hw, device)
        if pbase != int(plan['baseline']['params']):
            raise RuntimeError(f'Stage05 baseline params {pbase} != Stage04 plan {plan["baseline"]["params"]}')
        verify_resources(float(plan['baseline']['macs']), mbase, 'baseline MACs')
        train_args_from_ckpt = dict((yolo.ckpt or {}).get('train_args') or {})
        training_args = dict(train_args_from_ckpt)
        if not training_args:
            training_args = dict(getattr(net, 'args', {}) or {})
        training_args['task'] = 'detect'
        # Preserve source model metadata. Setting it to the UNPRUNED checkpoint
        # could cause an unaware trainer to reconstruct the old architecture.
        training_args['data'] = str(data_path)
        current_params, current_macs = pbase, mbase
        actual_lineage: dict[str, list[int]] = {}
        traces: list[dict] = []
        stages: list[dict] = []
        completed = 0
        stage_last = 0
        for d in plan['decisions']:
            n = int(d['decision'])
            stage = int(d['iterative_stage'])
            if stage != stage_last and stage_last != 0:
                stages[-1]['finished_decision'] = n - 1
                stages[-1]['end_params'] = current_params
                stages[-1]['end_macs'] = current_macs
                with preserve_batchnorm_buffers(net):
                    current_output_schema = detection_output_schema(net, hw, device)
                if current_output_schema != baseline_outputs:
                    raise RuntimeError('Native YOLO output tensor ABI changed after a pruning stage')
                logger.info('Stage %d complete at decision %d', stage_last, n - 1)
                snap_path = stage_dir / f'05_stage_{stage_last:02d}.pt'
                sha = atomic_pt_save(snap_path, net, training_args, {'stage': stage_last,
                                  'last_decision': n-1, 'stage04_plan_sha256': receipts['stage04_plan_sha256']})
                stages[-1]['checkpoint'] = str(snap_path)
                stages[-1]['checkpoint_sha256'] = sha
                write_json(stage_dir/'05_graph_signature.json', {'stages': stages,
                           'current_convs': current_conv_shapes(net), 'baseline_convs': baseline_shapes,
                           'source_checkpoint_sha256': sha256_file(source),
                           'stage04_plan_sha256': receipts['stage04_plan_sha256']})
                write_trace(stage_dir/'05_pruning_trace.csv', traces)
            if stage != stage_last:
                stages.append({'iterative_stage': stage, 'started_decision': n})
                stage_last = stage
            root = d['root']
            gid = d['group_id']
            lineage = actual_lineage.setdefault(gid, list(range(int(d['channels_before']))))
            ci, oi = list(d['current_indices']), list(d['original_indices'])
            if len(lineage) != int(d['channels_before']) or [lineage[i] for i in ci] != oi:
                raise RuntimeError(f'Stage05 current-index lineage mismatch at #{n}')
            with preserve_batchnorm_buffers(net):
                dg = step4.build_graph(net, tp, step2, hw, device)
            group, deps = exact_dependency_replay(dg, net, step2, step4, d,
                               int(plan['channel_align']), int(plan['min_channels']),
                               plan['stage02_safety_policy'])
            before_conv = current_conv_shapes(net)
            before_bn = current_bn_shapes(net)
            before_dependent_modules = {name: copy_module_dims(mod) for name,mod in net.named_modules()
                                        if name in {d['module'] for d in deps if not d['is_virtual']}}
            before = sum(p.numel() for p in net.parameters())
            if before != int(d['actual_params_before']):
                raise RuntimeError(f'Decision {n}: unexpected pre-prune parameter count {before}')
            group.prune()
            del group, dg
            after_conv = current_conv_shapes(net)
            after_bn = current_bn_shapes(net)
            after_dependent_modules = {name: copy_module_dims(mod) for name,mod in net.named_modules()
                                       if name in before_dependent_modules}
            verify_dependency_width_snapshots(before_dependent_modules, after_dependent_modules, deps)
            shape_diff = changed_shapes(before_conv, after_conv)
            bn_diff = {k: {'before': before_bn[k], 'after': v} for k, v in after_bn.items()
                       if k in before_bn and before_bn[k] != v}
            if set(before_bn) != set(after_bn):
                raise RuntimeError(f'Decision {n}: BatchNorm module unexpectedly created/removed')
            if not shape_diff or root not in shape_diff:
                raise RuntimeError(f'Decision {n}: no physical Conv changes at root')
            verify_stage04_alignment_from_snapshot(baseline_shapes, net, step2, step4,
                                                    int(plan['channel_align']), int(plan['min_channels']))
            if model_head_metadata(net) != baseline_head:
                raise RuntimeError(f'Decision {n}: detection head schema changed')
            after_p = sum(p.numel() for p in net.parameters())
            if after_p != int(d['actual_params_after']):
                raise RuntimeError(f'Decision {n}: param mismatch, Stage04 {d["actual_params_after"]}, now {after_p}')
            if after_p >= before:
                raise RuntimeError(f'Decision {n}: params did not decrease')
            # Each new graph and forward tests the *current* pruned model. It is
            # important that even if MAC profiling is sparse, full graph checks
            # are performed on every decision.
            with preserve_batchnorm_buffers(net):
                step4._post_prune_forward(net, step2, hw, device)
            if args.profile_every == 1 or n % args.profile_every == 0 or n == len(plan['decisions']) or (
                    n < len(plan['decisions']) and plan['decisions'][n]['iterative_stage'] != stage):
                current_params, current_macs = step4._profile(net, tp, hw, device)
                verify_resources(float(d['actual_macs_after']), current_macs,
                                 f'MACs at decision {n}', rel_tol=1e-6, abs_tol=512)
                if current_params != after_p:
                    raise RuntimeError(f'Decision {n}: profiler param count mismatch')
            else:
                current_params = after_p
                # Predicted MACs are only a placeholder until next actual profile.
                current_macs = float(d['actual_macs_after'])
            actual_lineage[gid] = [x for x in lineage if x not in set(oi)]
            traces.append({
                'decision': n, 'iterative_stage': stage, 'group_id': gid, 'root': root,
                'status': 'PHYSICAL_PRUNE_VERIFIED',
                'original_indices': oi, 'current_indices': ci,
                'channels_before': d['channels_before'], 'channels_after': d['channels_after'],
                'dependency_count': len(deps), 'affected_modules': d['affected_modules'],
                'params_before': before, 'params_after': after_p,
                'params_removed': before - after_p,
                'macs_before': float(d['actual_macs_before']),
                'macs_after': float(d['actual_macs_after']),
                'macs_removed': float(d['actual_mac_saving']),
                'weight_shape_changes': {'conv': shape_diff, 'batchnorm': bn_diff},
                'dependency_mapping_json': deps,
            })
            completed += 1
            logger.info('[%03d/%03d] stage %d | %-30s | -%d params | total -%.2f%% params, -%.2f%% MACs',
                        n, len(plan['decisions']), stage, root, before-after_p,
                        100*(1-after_p/pbase), 100*(1-current_macs/mbase))
            # Early trail for diagnostic inspection on failure (no success gate).
            if n % 10 == 0:
                write_trace(stage_dir/'05_pruning_trace.csv', traces)
        stages[-1]['finished_decision'] = completed
        stages[-1]['end_params'] = current_params
        stages[-1]['end_macs'] = current_macs
        with preserve_batchnorm_buffers(net):
            final_output_schema = detection_output_schema(net, hw, device)
        if final_output_schema != baseline_outputs:
            raise RuntimeError('Native YOLO output tensor ABI changed at final pruning stage')
        final_stage = stages[-1]['iterative_stage']
        stage_checkpoint = stage_dir/f'05_stage_{final_stage:02d}.pt'
        sha_stage = atomic_pt_save(stage_checkpoint, net, training_args,
                            {'stage': final_stage, 'last_decision': completed,
                             'stage04_plan_sha256': receipts['stage04_plan_sha256']})
        stages[-1]['checkpoint'] = str(stage_checkpoint)
        stages[-1]['checkpoint_sha256'] = sha_stage
        checks['staged_checkpoints_saved'] = len(stages) == plandiag['iterative_stages']
        final_pt = stage_dir / '05_pruned_raw.pt'
        final_sha = atomic_pt_save(final_pt, net, training_args, {
            'source_sha256': sha256_file(source),
            'stage04_plan_sha256': receipts['stage04_plan_sha256'],
            'decisions': completed,
            'params': current_params, 'macs': current_macs,
            'status': 'RAW_PRUNED_NEEDS_STAGE06_07_09',
        })
        checks['final_checkpoint_replay_complete'] = completed == len(plan['decisions'])
        if current_params != int(plan['projected']['params']):
            raise RuntimeError('Final parameter count differs from Stage04')
        verify_resources(float(plan['projected']['macs']), current_macs, 'final MACs', rel_tol=1e-6, abs_tol=512)
        checks['all_exact_depgraph_groups_verified'] = True
        checks['physical_weight_shapes_changed'] = current_params < pbase and any(
            current_conv_shapes(net).get(k) != v for k, v in baseline_shapes.items())
        checks['resources_match_plan'] = True
        checks['graph_and_alignment_valid'] = True
        checks['staged_checkpoints_saved'] = bool(checks['staged_checkpoints_saved'])
        write_trace(stage_dir/'05_pruning_trace.csv', traces)
        graph_sig_path = stage_dir/'05_graph_signature.json'
        write_json(graph_sig_path, {
            'source_checkpoint_sha256': sha256_file(source),
            'stage04_plan_sha256': receipts['stage04_plan_sha256'],
            'model_family': plan['model_family'], 'model_scale': plan['model_scale'],
            'baseline_head': baseline_head, 'baseline_train_output_schema': baseline_outputs,
            'baseline_convs': baseline_shapes, 'baseline_batchnorm': baseline_bn_shapes,
            'pruned_convs': current_conv_shapes(net),
            'pruned_batchnorm': current_bn_shapes(net),
            'changed_convs': changed_shapes(baseline_shapes, current_conv_shapes(net)),
            'stages': stages, 'root_lineages': actual_lineage,
        })
        reload_result = {'status': 'NOT_REQUESTED'}
        if args.verify_reload:
            reloaded = YOLO(str(final_pt))
            if reloaded.task != 'detect' or len(reloaded.names) != data['nc']:
                raise RuntimeError('Serialized checkpoint reloaded with wrong task/class count')
            reloaded_conv = current_conv_shapes(reloaded.model)
            if reloaded_conv != current_conv_shapes(net):
                raise RuntimeError('Serialized checkpoint loaded with changed architecture')
            if sum(p.numel() for p in reloaded.model.parameters()) != current_params:
                raise RuntimeError('Serialized checkpoint reloaded with wrong params')
            reload_result = {'status':'PASS_IN_PROCESS', 'params': current_params,
                             'conv_shapes_identical': True}
        if legacy_bridge is not None:
            validate_legacy_bridge(data, gate0, sha256_file(gate0_path), legacy_bridge)
        elif common.dataset_fingerprint(data)['manifest_sha256'] != original_fp['manifest_sha256']:
            raise RuntimeError('Dataset inventory changed during Stage05')
        if sha256_file(source) != plan['checkpoint_sha256']:
            raise RuntimeError('Source checkpoint changed during pruning')
        report = {
            'step': '05_apply_structured_pruning', 'status': 'PRELIMINARY',
            'plan_status': plan['target_status'], 'source_checkpoint': str(source),
            'source_checkpoint_sha256': plan['checkpoint_sha256'],
            'stage04_plan_sha256': receipts['stage04_plan_sha256'],
            'decisions_applied': completed, 'iterative_stages': len(stages),
            'baseline_params': pbase, 'pruned_params': current_params,
            'param_reduction': 1-current_params/pbase,
            'baseline_macs': mbase, 'pruned_macs': current_macs,
            'mac_reduction': 1-current_macs/mbase,
            'removed_root_channels': sum(len(d['original_indices']) for d in plan['decisions']),
            'channel_align': plan['channel_align'], 'min_channels': plan['min_channels'],
            'reload': reload_result,
            'output_pruned_raw': str(final_pt), 'output_pruned_raw_sha256': final_sha,
            'next': '06_verify_pruned_model.py; recovery and QAT handoff NOT yet certified',
        }
        report_path = stage_dir/'05_report.json'
        write_json(report_path, report)
        artifacts = {'checkpoint': str(final_pt), 'checkpoint_sha256': final_sha,
                     'trace': str(stage_dir/'05_pruning_trace.csv'),
                     'graph_signature': str(graph_sig_path),
                     'report': str(report_path),
                     'intermediate_checkpoints': stages}
        provenance['artifact_hashes'] = {
            'checkpoint': final_sha,
            'trace': sha256_file(stage_dir/'05_pruning_trace.csv'),
            'graph_signature': sha256_file(graph_sig_path),
            'report': sha256_file(report_path),
        }
        if legacy_bridge is not None:
            artifacts['dataset_legacy_bridge'] = str(stage_dir / '05_dataset_legacy_bridge.json')
            provenance['artifact_hashes']['dataset_legacy_bridge'] = sha256_file(
                stage_dir / '05_dataset_legacy_bridge.json')
        checks['artifacts_saved'] = True
    except Exception as exc:
        logger.exception('Stage05 FAIL: %s: %s', type(exc).__name__, exc)
        errors.append(f'{type(exc).__name__}: {exc}')
        (stage_dir/'05_error_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        # Never leave a seemingly successful raw checkpoint when failure occurs.
        final_pt = stage_dir/'05_pruned_raw.pt'
        if final_pt.is_file():
            final_pt.unlink()
    gate = save_gate(out, STAGE, NAME, checks=checks,
                     errors=errors, warnings=warnings, provenance=provenance, artifacts=artifacts)
    gate_stage = dict(gate)
    write_json(stage_dir/'05_gate.json', gate_stage)
    logger.info('[05] %s — %s', gate['status'], gatepath)
    return 0 if gate['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(run(parser().parse_args()))
