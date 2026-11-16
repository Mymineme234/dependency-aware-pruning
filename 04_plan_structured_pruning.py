#!/usr/bin/env python3
"""04 — Dependency-aware, budget-aware pruning PLAN (NO checkpoint mutation).

Consumes PASS gates 00..03 and the native Stage 02 graph / Stage 03 Taylor scores.
For each accepted removal, reconstructs and checks EXACT DepGraph dependencies,
physically prunes a *throwaway in-memory clone*, re-runs forward and re-profiles
parameters/MACs. Source checkpoint is never saved or changed.

Conservative v1 policy: a parameter-bearing module cannot be edited by two
*different* pruning roots within one plan. Multiple channel-block deletions from
one root are allowed, tracking original <-> current channel indices explicitly.
This costs some compression opportunity but prevents stale cross-root index maps.
Stage 05 MUST replay each action with fresh DepGraph and verify it again.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import math
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from pruning_common import (
    SUPPORTED_FAMILIES, contract, dataset_fingerprint, gate_file, hash_object,
    make_imgsz, model_head_metadata, now_utc, read_dataset, read_json, resolved,
    save_gate, select_device, setup_log, sha256_file, versions, write_json,
)

STAGE, NAME = '04', 'plan_structured_pruning'
GATES = [('00', 'check_environment'), ('01', 'profile_baseline'),
         ('02', 'analyze_pruning_graph'), ('03', 'collect_gradient_importance')]
CSV_FIELDS = ['decision', 'iterative_stage', 'group_id', 'root', 'channels_before',
              'channels_after', 'original_indices', 'current_indices',
              'score_abs_taylor_sum', 'estimated_param_saving',
              'estimated_mac_saving', 'actual_params_before', 'actual_params_after',
              'actual_param_saving', 'actual_macs_before', 'actual_macs_after',
              'actual_mac_saving', 'affected_modules', 'resource_reduction',
              'status']


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='04: exact aligned DepGraph planning + scratch-model compression simulation; never edit source best.pt')
    for name in ('model', 'data', 'output'):
        p.add_argument(f'--{name}', required=True)
    p.add_argument('--imgsz', nargs='+', type=int, default=[640])
    p.add_argument('--device', default='0')
    p.add_argument('--batch', type=int, default=8, help='Must match the Stage 00-03 contract; planning trace always uses batch=1')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--expected-classes', type=int, default=None)
    p.add_argument('--family', default='auto', choices=['auto', *SUPPORTED_FAMILIES])
    p.add_argument('--method', choices=['group-taylor'], default='group-taylor')
    p.add_argument('--target-metric', choices=['channels', 'params', 'macs'], default='channels')
    p.add_argument('--target-reduction', type=float, default=0.10,
                   help='Fraction of chosen target to REMOVE, e.g. 0.33 or 0.50')
    p.add_argument('--target-tolerance', type=float, default=0.005,
                   help='Allowed absolute shortfall from target before PARTIAL_TARGET')
    p.add_argument('--channel-align', type=int, default=8)
    p.add_argument('--min-channels', type=int, default=16)
    p.add_argument('--iterative-steps', type=int, default=5,
                   help='Number of planned pruning stages recorded for Stage 05')
    p.add_argument('--max-decisions', type=int, default=64,
                   help='Hard cap on accepted 8-channel decisions; report PARTIAL_TARGET if cap binds')
    p.add_argument('--ranking', choices=['sensitivity', 'efficiency'], default='sensitivity',
                   help='Lowest Taylor sum (accuracy-first) or Taylor sum / estimated resource saving')
    p.add_argument('--candidate-mode', choices=['hybrid', 'contiguous'], default='hybrid',
                   help='hybrid additionally tries the 8 lowest-score channels per root (noncontiguous only if DepGraph approves)')
    p.add_argument('--reference-model', default=None)
    p.add_argument('--max-original-conv-prune-ratio',
                   type=float, default=1.0)
    p.add_argument('--balance-strength',
                   type=float, default=0.0)
    p.add_argument('--normalize-sensitivity',
                   action='store_true')
    p.add_argument('--allow-partial-target', action='store_true',
                   help='Allow gate PASS with target_status=PARTIAL_TARGET; Stage 05 still must verify exact actions')
    p.add_argument('--overwrite', action='store_true',
                   help='Replace Stage 04 results and invalidate Stage 05..10 gates')
    return p


def load_stage02() -> tuple[Any, str]:
    path = Path(__file__).with_name('02_analyze_pruning_graph.py')
    if not path.is_file():
        raise FileNotFoundError(f'Requires the unchanged Stage 02 source alongside Stage 04: {path}')
    spec = importlib.util.spec_from_file_location('pruning_stage02_for_04', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot dynamically import Stage 02 safety rules')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for method in ('_safe_probe', '_output_scalar', '_head_paths', '_attention_paths',
                   '_static_module_paths', '_group_items'):
        if not hasattr(module, method):
            raise RuntimeError(f'Unrecognized Stage 02 helper API: {method}')
    return module, sha256_file(path)


def require_ancestors(out: Path, request: dict, dataset: dict) -> tuple[list[dict], dict, dict, Path, dict]:
    paths = [gate_file(out, n, name) for n, name in GATES]
    if not all(p.is_file() for p in paths):
        raise RuntimeError(f'Missing required PASS gates: {[str(p) for p in paths if not p.is_file()]}')
    gates = [read_json(p) for p in paths]
    for (n, _), g in zip(GATES, gates):
        if g.get('status') != 'PASS' or not g.get('checks') or not all(v is True for v in g['checks'].values()):
            raise RuntimeError(f'Ancestor Stage {n} has no complete PASS gate')
    p0, p1, p2, p3 = [g.get('provenance', {}) for g in gates]
    identity = hash_object(request)
    if any(p.get('contract_sha256') != identity for p in (p0, p1, p2, p3)):
        raise RuntimeError('Model/data/imgsz/device/batch/workers/family does not match 00..03 contract')
    # Ensure each gate is still linked to the same upstream run.
    if p1.get('stage00_gate_sha256') != sha256_file(paths[0]):
        raise RuntimeError('Stage 00 gate differs from Stage 01 ancestor hash')
    for p, deps in ((p2, [(0, 'stage00_gate_sha256'), (1, 'stage01_gate_sha256')]),
                    (p3, [(0, 'stage00_gate_sha256'), (1, 'stage01_gate_sha256'),
                          (2, 'stage02_gate_sha256')])):
        for idx, key in deps:
            if p.get(key) != sha256_file(paths[idx]):
                raise RuntimeError(f'Ancestor gate changed: {key}; rerun downstream stages')
    if p2.get('stage01_profile_sha256') != sha256_file(out / 'stages' / '01' / '01_profile.json'):
        raise RuntimeError('Stage 01 profile changed since Stage 02')
    graph_path = out / 'stages' / '02' / '02_graph.json'
    if not graph_path.is_file() or p3.get('stage02_graph_sha256') != sha256_file(graph_path):
        raise RuntimeError('Stage 02 graph missing/changed since Stage 03')
    stage02_source = Path(__file__).with_name('02_analyze_pruning_graph.py')
    if not stage02_source.is_file() or p3.get('stage02_source_sha256') != sha256_file(stage02_source):
        raise RuntimeError('Stage 02 source changed since Stage 03; existing safety guarantees invalidated')
    source = Path(p0.get('source_checkpoint', ''))
    if not source.is_file() or sha256_file(source) != p0.get('checkpoint_sha256'):
        raise RuntimeError('FP32 source checkpoint changed since Stage 00')
    if p0.get('dataset_yaml_sha256') != dataset['yaml_sha256']:
        raise RuntimeError('Dataset YAML changed since Stage 00')
    if p0.get('dataset_manifest', {}).get('manifest_sha256') != dataset_fingerprint(dataset)['manifest_sha256']:
        raise RuntimeError('Dataset inventory changed since Stage 00')
    current_ver = versions()
    for pkg in ('torch', 'torchvision', 'ultralytics', 'torch-pruning', 'numpy'):
        if current_ver.get(pkg) != p0.get('dependency_versions', {}).get(pkg):
            raise RuntimeError(f'Dependency {pkg} changed since Stage 00')
    files = p3.get('artifact_hashes', {})
    outputs = gates[3].get('artifacts', {})
    required = ('importance_json', 'importance_npz', 'scores_csv', 'grad_log', 'unscored_csv')
    for key in required:
        if key not in files or key not in outputs:
            raise RuntimeError(f'Stage 03 missing recorded hash/path for {key}')
        artifact_path = Path(outputs[key])
        if not artifact_path.is_file() or sha256_file(artifact_path) != files[key]:
            raise RuntimeError(f'Stage 03 artifact missing or modified: {key}')
    graph = read_json(graph_path)
    imp = read_json(Path(outputs['importance_json']))
    if imp.get('source') != 'TRAIN_SPLIT_ONLY' or imp.get('loss') != 'REAL_ULTRALYTICS_DETECTION_LOSS':
        raise RuntimeError('Stage 03 did not certify train-only real-detection-loss Taylor scoring')
    if graph.get('family') != p0.get('model_family') or imp.get('model_family') != graph.get('family'):
        raise RuntimeError('Stage 00/02/03 architecture identity conflict')
    return gates, graph, imp, source, {'stage03_artifacts': outputs}


def read_channel_scores(path: Path, graph: dict, importance: dict) -> dict[str, list[float]]:
    roots = {g['group_id']: g for g in graph['groups'] if g['feasibility'] == 'ELIGIBLE_PROBE_ONLY'}
    scored = {g['group_id'] for g in importance['groups']}
    if not roots or scored != set(roots):
        raise RuntimeError('All Stage 02 eligible roots must have complete Stage 03 scores; cannot silently prune unscored roots')
    result = {gid: [float('nan')] * int(info['channels_before']) for gid, info in roots.items()}
    with path.open(newline='', encoding='utf-8') as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            gid, root, idx = row['group_id'], row['root'], int(row['channel'])
            if gid not in roots or roots[gid]['root'] != root or not 0 <= idx < len(result[gid]):
                raise RuntimeError(f'Unexpected score group, root or channel: {row}')
            if math.isfinite(result[gid][idx]):
                raise RuntimeError(f'Duplicate channel score: {gid}:{idx}')
            s = float(row['score_mean_abs_taylor'])
            if not math.isfinite(s) or s < 0:
                raise RuntimeError(f'Nonfinite/negative Taylor score for {gid}:{idx}')
            result[gid][idx] = s
    for gid, vals in result.items():
        if any(not math.isfinite(v) for v in vals):
            raise RuntimeError(f'Incomplete channel scores for {gid}')
    return result


def build_graph(net, tp, stage2, hw: tuple[int, int], device):
    import torch
    net.train().requires_grad_(True)
    net.zero_grad(set_to_none=True)
    x = torch.zeros((1, 3, *hw), device=device, dtype=torch.float32, requires_grad=True)
    with torch.enable_grad():
        dg = tp.DependencyGraph().build_dependency(
            net, example_inputs=x,
            forward_fn=lambda model, images: stage2._output_scalar(model(images))[0], verbose=False)
    if not getattr(dg, 'module2node', None):
        raise RuntimeError('Empty DepGraph on scratch model')
    net.zero_grad(set_to_none=True)
    return dg


def protected_context(net, stage2, config: dict) -> tuple[dict, dict]:
    named = dict(net.named_modules())
    return ({'head_paths': stage2._head_paths(net),
             'attention_paths': stage2._attention_paths(net),
             'static_paths': stage2._static_module_paths(net)},
            {'head': bool(config['protect_detect_head']),
             'attention': bool(config['protect_attention']),
             'static_splits': bool(config['protect_static_splits'])})


def dependency_ops(deps: list[dict]) -> list[dict]:
    """Full exact named dependency indexes required for Stage 05 replay."""
    if any(d.get('indices_truncated', False) or d.get('num_indices', 0) != len(d.get('indices', [])) for d in deps):
        raise RuntimeError('Truncated dependency index list cannot be used in a physical pruning plan')
    return [{k: d[k] for k in ('module', 'module_type', 'prune_kind', 'indices', 'is_virtual')}
            for d in deps]


def modified_parameter_modules(deps: list[dict]) -> set[str]:
    """Module overlap guard to prevent stale source->current index mapping."""
    keys = set()
    for d in deps:
        if d['is_virtual']:
            continue
        # Conv, BN, Linear, etc. All nonvirtual modules conservatively count.
        keys.add(d['module'])
    return keys


def estimate_saving(deps: list[dict], named: dict, module_shapes: dict) -> tuple[int, float]:
    """Approximate before mutation; actual params/MACs are MEASURED after it.

    Uses union of input/output index sets on each module to avoid double-counting
    the intersection when a Conv changes both axes.
    """
    import torch
    mods: dict[str, dict] = defaultdict(lambda: {'in': set(), 'out': set()})
    for dep in deps:
        if dep['is_virtual'] or dep['prune_kind'] not in ('in', 'out'):
            continue
        mods[dep['module']][dep['prune_kind']].update(dep['indices'])
    delta_params, delta_macs = 0, 0.0
    for name, axes in mods.items():
        mod = named.get(name)
        if mod is None:
            raise RuntimeError(f'Dependency refers to unresolved module {name}')
        co, ci = len(axes['out']), len(axes['in'])
        if isinstance(mod, torch.nn.Conv2d):
            if mod.groups != 1:
                raise RuntimeError(f'Cannot estimate grouped/depthwise saving precisely without adapter: {name}')
            if co >= mod.out_channels or ci >= mod.in_channels:
                raise RuntimeError(f'Illegal full Conv dimension deletion in candidate {name}')
            kernel = mod.kernel_size[0] * mod.kernel_size[1]
            removed_weights = (mod.in_channels * mod.out_channels -
                               (mod.in_channels - ci) * (mod.out_channels - co)) * kernel
            removed_bias = co if mod.bias is not None else 0
            delta_params += removed_weights + removed_bias
            shape = module_shapes.get(name, {}).get('output_shape')
            if isinstance(shape, list) and len(shape) == 4:
                delta_macs += removed_weights * int(shape[2]) * int(shape[3])
        elif isinstance(mod, torch.nn.Linear):
            removed_weights = (mod.in_features * mod.out_features -
                               (mod.in_features - ci) * (mod.out_features - co))
            delta_params += removed_weights + (co if mod.bias is not None else 0)
            delta_macs += removed_weights
        elif isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            if mod.affine:
                delta_params += 2 * co
        elif any(True for _ in mod.parameters(recurse=False)):
            raise RuntimeError(f'Unhandled parameterized dependency: {name} ({type(mod).__name__})')
    return delta_params, float(delta_macs)


def valid_candidate_pool(graph: dict, scores: dict, keep: dict,
                         dg, net, stage2, align: int, minimum: int,
                         module_owners: dict, ranking: str, candidate_mode: str,
                         metric: str, pbase: int, mbase: float) -> tuple[list[dict], dict]:
    """Validate EXACT current indices (not the sparse Stage02 sample probes)."""
    import torch
    named = dict(net.named_modules())
    reverse = {m: name for name, m in named.items()}
    protected, config = protected_context(net, stage2, graph['_config'])
    module_shapes = graph.get('module_shapes', {})
    counts = {'scanned': 0, 'blocked_mapping': 0, 'blocked_owner': 0,
              'blocked_hazard': 0, 'blocked_resource': 0}
    choices = []
    for info in graph['groups']:
        if info.get('feasibility') != 'ELIGIBLE_PROBE_ONLY':
            continue
        gid, root = info['group_id'], info['root']
        mod = named.get(root)
        lineage = keep[gid]
        if not isinstance(mod, torch.nn.Conv2d) or int(mod.out_channels) != len(lineage):
            raise RuntimeError(f'Root current-index lineage lost: {gid}:{root}; refuse potentially wrong pruning')
        candidate_blocks = []
        for start in range(0, int(info['channels_before']), align):
            block = list(range(start, start + align))
            if block[-1] < int(info['channels_before']) and set(block).issubset(lineage):
                candidate_blocks.append(block)
        if candidate_mode == 'hybrid' and len(lineage) >= align:
            # Exact DepGraph validation decides if noncontiguous original channel
            # indices are safe; merely having low Taylor score gives no permission.
            ranked = sorted(lineage, key=lambda i: (scores[gid][i], i))
            for offset in (0, align // 2, align):
                block = sorted(ranked[offset:offset + align])
                if len(block) == align:
                    candidate_blocks.append(block)
        seen_blocks = set()
        for orig in candidate_blocks:
            key = tuple(orig)
            if key in seen_blocks:
                continue
            seen_blocks.add(key)
            counts['scanned'] += 1
            original_to_current = {x: i for i, x in enumerate(lineage)}
            current_indices = [original_to_current[c] for c in orig]
            if len(current_indices) != align:
                counts['blocked_mapping'] += 1
                continue
            deps, reasons = stage2._safe_probe(dg, mod, current_indices, reverse, protected, align, minimum, config)
            if deps is None or reasons:
                counts['blocked_hazard'] += 1
                continue
            try:
                ops = dependency_ops(deps)
                touched = modified_parameter_modules(ops)
            except RuntimeError:
                counts['blocked_mapping'] += 1
                continue
            if any(module_owners.get(x, gid) != gid for x in touched):
                counts['blocked_owner'] += 1
                continue
            try:
                dp, dm = estimate_saving(deps, named, module_shapes)
            except RuntimeError:
                counts['blocked_resource'] += 1
                continue
            if dp <= 0 or dm < 0:
                counts['blocked_resource'] += 1
                continue
            cost = sum(scores[gid][x] for x in orig)
            saving = (dp / max(1, pbase) if metric == 'params' else
                      dm / max(1, mbase) if metric == 'macs' else len(orig) / graph['_eligible_channels'])
            priority = cost if ranking == 'sensitivity' else cost / max(saving, 1e-12)
            choices.append({'group_id': gid, 'root': root, 'original_indices': orig,
                            'current_indices': current_indices, 'score_abs_taylor_sum': cost,
                            'estimated_param_saving': dp, 'estimated_mac_saving': dm,
                            'priority': priority, 'dependencies': ops, 'touched': touched})
    choices.sort(key=lambda c: (c['priority'], c['score_abs_taylor_sum'], c['group_id'], c['original_indices']))
    return choices, counts


def _profile(net, tp, hw, device) -> tuple[int, float]:
    import torch
    net.eval()
    x = torch.zeros((1, 3, *hw), device=device, dtype=torch.float32)
    with torch.inference_mode():
        p_tp = tp.utils.count_ops_and_params(net, example_inputs=x)
    if not isinstance(p_tp, (tuple, list)) or len(p_tp) != 2:
        raise RuntimeError('Unrecognized Torch-Pruning MAC profiler output')
    count = sum(p.numel() for p in net.parameters())
    measured_macs, profiler_params = float(p_tp[0]), float(p_tp[1])
    if not all(math.isfinite(v) and v > 0 for v in (count, measured_macs, profiler_params)):
        raise RuntimeError('Nonfinite/zero model resources during pruning simulation')
    if abs(profiler_params - count) > max(1, count * .001):
        raise RuntimeError(f'Profiler params differ from actual model: {profiler_params} vs {count}')
    return count, measured_macs


def _post_prune_forward(net, stage2, hw, device):
    import torch
    net.train()
    x = torch.zeros((1, 3, *hw), device=device, dtype=torch.float32, requires_grad=True)
    with torch.enable_grad():
        out, stats = stage2._output_scalar(net(x))
    if not out.requires_grad or not bool(torch.isfinite(out).item()):
        raise RuntimeError('Post-pruning TRAIN forward not finite/differentiable')
    if int(stats.get('differentiable_tensors', 0)) == 0:
        raise RuntimeError('Post-pruning graph produced zero differentiable outputs')
    net.zero_grad(set_to_none=True)
    return stats


def _verify_changed_conv_alignment(before: dict, net, stage2, align: int, minimum: int):
    """All changed Conv feature widths must meet alignment and min channels.

    Original non-aligned *unchanged* internal widths remain explicit pre-existing
    exceptions. Detection head ABI and RGB input are protected separately.
    """
    import torch
    after_named = dict(net.named_modules())
    heads = stage2._head_paths(net)
    problems = []
    changed = []
    for name, old in before.items():
        mod = after_named.get(name)
        if not isinstance(mod, torch.nn.Conv2d):
            problems.append(f'Conv disappeared: {name}')
            continue
        widths = (int(mod.in_channels), int(mod.out_channels))
        if widths == old:
            continue
        changed.append(name)
        if name in heads:
            problems.append(f'Protected detect head structurally modified: {name}')
            continue
        for axis, val, initial in zip(('in', 'out'), widths, old):
            if val == initial:
                continue
            if axis == 'in' and initial == 3 and val == 3:
                continue
            if val < minimum or val % align:
                problems.append(f'Misaligned changed Conv {name}.{axis}: {initial}->{val}')
    if problems:
        raise RuntimeError('Post-pruning alignment/head violations: ' + '; '.join(problems[:10]))
    return changed


def _candidate_clone_and_measure(net, candidate, tp, stage2, hw, device,
                                 align, minimum, before_convs: dict, expected_params: int,
                                 expected_macs: float):
    """Apply to throwaway cloned scratch model with a FRESH dependency graph.

    Returning net is internal to the planner; it is NEVER serialized. The real
    Stage 05 will replay exact original/current indices from the JSON plan.
    """
    import torch
    clone = copy.deepcopy(net)
    clone.to(device).float().requires_grad_(True)
    dg = build_graph(clone, tp, stage2, hw, device)
    named = dict(clone.named_modules())
    protected, config = protected_context(clone, stage2, {'protect_detect_head': True,
                                                          'protect_attention': True,
                                                          'protect_static_splits': True})
    mod = named.get(candidate['root'])
    if not isinstance(mod, torch.nn.Conv2d):
        raise RuntimeError('Candidate Conv root missing on clone')
    deps, reasons = stage2._safe_probe(dg, mod, candidate['current_indices'],
                                       {m: n for n, m in named.items()}, protected,
                                       align, minimum, config)
    if deps is None or reasons:
        raise RuntimeError(f'Candidate hazard differs on scratch clone: {reasons}')
    if dependency_ops(deps) != candidate['dependencies']:
        raise RuntimeError('Scratch graph dependency index map differs from ranked graph')
    group = dg.get_pruning_group(mod, dg.get_pruner_of_module(mod).prune_out_channels,
                                 candidate['current_indices'])
    if not dg.check_pruning_group(group):
        raise RuntimeError('DepGraph rejected scratch physical pruning group')
    # The *only* place parameters are physically pruned, on this temporary copy.
    group.prune()
    del dg, group
    _verify_changed_conv_alignment(before_convs, clone, stage2, align, minimum)
    _post_prune_forward(clone, stage2, hw, device)
    new_p, new_m = _profile(clone, tp, hw, device)
    if new_p >= expected_params or new_m > expected_macs * (1 + 1e-7):
        raise RuntimeError(f'Not a real non-increasing pruning: params {expected_params}->{new_p}, MACs {expected_macs}->{new_m}')
    return clone, new_p, new_m


def objective_reduction(metric: str, p0: int, m0: float, eligible: int,
                        pnow: int, mnow: float, removed: int) -> float:
    if metric == 'params':
        return 1.0 - pnow / p0
    if metric == 'macs':
        return 1.0 - mnow / m0
    return removed / eligible


def plan_scratch(net, tp, stage2, graph: dict, scores: dict, baseline: dict,
                 args, device, logger) -> dict:
    import torch
    graph = dict(graph)
    graph['_config'] = baseline['stage02_config']
    eligible = [g for g in graph['groups'] if g['feasibility'] == 'ELIGIBLE_PROBE_ONLY']
    eligible_channels = sum(int(g['channels_before']) for g in eligible)
    graph['_eligible_channels'] = eligible_channels
    keep = {g['group_id']: list(range(int(g['channels_before']))) for g in eligible}
    module_owners = {}
    current = net
    hw = args.imgsz_hw
    original_params, original_macs = _profile(current, tp, hw, device)
    report_params = int(baseline['base_params'])
    report_macs = float(baseline['base_macs'])
    if abs(original_params - report_params) > max(1, report_params * .001):
        raise RuntimeError(f'Baseline params changed from Stage 01: {report_params} -> {original_params}')
    if abs(original_macs - report_macs) > max(10, report_macs * 1e-5):
        raise RuntimeError(f'Baseline MACs differ from Stage 01: {report_macs} -> {original_macs}')
    baseline_channels = {n: (m.in_channels, m.out_channels)
                         for n, m in current.named_modules() if isinstance(m, torch.nn.Conv2d)}
    if args.reference_model:
        from ultralytics import YOLO
        reference = YOLO(str(resolved(args.reference_model))).model
        original_widths = {
            n: m.out_channels
            for n, m in reference.named_modules()
            if isinstance(m, torch.nn.Conv2d)
        }
        del reference
    else:
        original_widths = {
            n: v[1] for n, v in baseline_channels.items()
        }
    for name, (_, width) in baseline_channels.items():
        if name not in original_widths or width > original_widths[name]:
            raise RuntimeError(
                f'Conv missing/widened relative to reference: {name}'
            )
    removed_total, decisions, reject_reasons, counts_accum = 0, [], [], defaultdict(int)
    for decision_num in range(1, args.max_decisions + 1):
        ratio = objective_reduction(args.target_metric, original_params, original_macs,
                                    eligible_channels, baseline['current_params'],
                                    baseline['current_macs'], removed_total)
        if ratio + args.target_tolerance >= args.target_reduction:
            break
        dg = build_graph(current, tp, stage2, hw, device)
        choices, counts = valid_candidate_pool(graph, scores, keep, dg, current, stage2,
                                               args.channel_align, args.min_channels,
                                               module_owners, args.ranking, args.candidate_mode, args.target_metric,
                                               original_params, original_macs)
        del dg
        for key, value in counts.items():
            counts_accum[key] += value
        # Balance Taylor importance across roots and enforce
        # cumulative pruning limits relative to the original model.
        filtered = []
        for c in choices:
            gid, root = c['group_id'], c['root']
            orig = original_widths[root]
            after = len(keep[gid]) - len(c['original_indices'])
            if after < orig * (
                1.0 - args.max_original_conv_prune_ratio
            ) - 1e-9:
                counts_accum['blocked_root_cap'] += 1
                continue

            avg = sum(
                scores[gid][i] for i in keep[gid]
            ) / len(keep[gid])

            cost = c['score_abs_taylor_sum']
            if args.normalize_sensitivity:
                cost /= max(
                    len(c['original_indices']) * avg, 1e-12
                )

            pruned = (orig - len(keep[gid])) / orig
            penalty = 1.0 + args.balance_strength * (
                pruned / args.max_original_conv_prune_ratio
            ) ** 2

            savings = (
                c['estimated_param_saving'] / original_params
                if args.target_metric == 'params'
                else c['estimated_mac_saving'] / original_macs
                if args.target_metric == 'macs'
                else len(c['original_indices']) / eligible_channels
            )
            c['priority'] = (
                cost if args.ranking == 'sensitivity'
                else cost / max(savings, 1e-12)
            ) * penalty
            filtered.append(c)

        choices = sorted(
            filtered,
            key=lambda c: (
                c['priority'],
                c['group_id'],
                c['original_indices']
            )
        )
        if not choices:
            logger.warning(
                'No candidates within cumulative original-Conv root cap: %s',
                counts
            )
            break
        accepted = False
        for candidate in choices:
            try:
                clone, new_p, new_m = _candidate_clone_and_measure(
                    current, candidate, tp, stage2, hw, device,
                    args.channel_align, args.min_channels, baseline_channels,
                    baseline['current_params'], baseline['current_macs'])
                # Check ALL coupled Conv output widths.
                clone_convs = {
                    n: m for n, m in clone.named_modules()
                    if isinstance(m, torch.nn.Conv2d)
                }
                for name, original_width in original_widths.items():
                    mod = clone_convs.get(name)
                    if (
                        mod is None or
                        mod.out_channels < original_width * (
                            1.0 - args.max_original_conv_prune_ratio
                        ) - 1e-9
                    ):
                        raise RuntimeError(
                            f'ORIGINAL_CONV_CAP: {name}'
                        )
                # Params are exactly countable from Conv/BN/Linear dependency
                # slices. A mismatch signals untracked graph changes and MUST
                # not become an apparently valid plan.
                actual_saved_params = baseline['current_params'] - new_p
                if abs(actual_saved_params - candidate['estimated_param_saving']) > 1:
                    raise RuntimeError(
                        f'UNTRACKED_PARAMETER_CHANGE: expected to remove '
                        f'{candidate["estimated_param_saving"]} but actually removed '
                        f'{actual_saved_params}')
                new_removed = removed_total + len(candidate['original_indices'])
                new_ratio = objective_reduction(args.target_metric, original_params, original_macs,
                                               eligible_channels, new_p, new_m, new_removed)
                if new_ratio > args.target_reduction + args.target_tolerance:
                    old_gap = abs(args.target_reduction - ratio)
                    new_gap = abs(args.target_reduction - new_ratio)
                    if new_gap >= old_gap:
                        raise RuntimeError('WOULD_OVERSHOOT_TARGET_WITHOUT_IMPROVING_DISTANCE')
                gid = candidate['group_id']
                old_width = len(keep[gid])
                keep[gid] = [x for x in keep[gid] if x not in set(candidate['original_indices'])]
                for name in candidate['touched']:
                    module_owners[name] = gid
                record = {'decision': decision_num, 'iterative_stage': 0,
                          'group_id': gid, 'root': candidate['root'],
                          'channels_before': old_width, 'channels_after': len(keep[gid]),
                          'original_indices': candidate['original_indices'],
                          'current_indices': candidate['current_indices'],
                          'score_abs_taylor_sum': candidate['score_abs_taylor_sum'],
                          'ranking_priority': candidate['priority'],
                          'estimated_param_saving': candidate['estimated_param_saving'],
                          'estimated_mac_saving': candidate['estimated_mac_saving'],
                          'actual_params_before': baseline['current_params'],
                          'actual_params_after': new_p,
                          'actual_param_saving': baseline['current_params'] - new_p,
                          'actual_macs_before': baseline['current_macs'],
                          'actual_macs_after': new_m,
                          'actual_mac_saving': baseline['current_macs'] - new_m,
                          'affected_modules': sorted(candidate['touched']),
                          'dependencies': candidate['dependencies'],
                          'resource_reduction': new_ratio, 'status': 'SIMULATED_VALID'}
                decisions.append(record)
                current = clone
                baseline['current_params'], baseline['current_macs'] = new_p, new_m
                removed_total = new_removed
                logger.info('Decision %d: %-7s %-29s orig=%s | -%d params | -%.3f GMACs | %s reduction %.2f%%',
                            decision_num, gid, candidate['root'], candidate['original_indices'],
                            record['actual_param_saving'], record['actual_mac_saving']/1e9,
                            args.target_metric, new_ratio*100)
                accepted = True
                break
            except (RuntimeError, ValueError, IndexError, AttributeError) as exc:
                reject_reasons.append({'root': candidate['root'],
                                       'original_indices': candidate['original_indices'],
                                       'reason': f'{type(exc).__name__}: {str(exc)[:280]}'})
        if not accepted:
            logger.warning('No scratch simulation succeeded among %d candidates', len(choices))
            break
    if decisions:
        for i, r in enumerate(decisions):
            r['iterative_stage'] = min(args.iterative_steps, 1 + (i * args.iterative_steps // len(decisions)))
    final_reduction = objective_reduction(args.target_metric, original_params, original_macs,
                                          eligible_channels, baseline['current_params'],
                                          baseline['current_macs'], removed_total)
    target_status = ('TARGET_REACHED' if final_reduction + args.target_tolerance >= args.target_reduction
                     else 'PARTIAL_TARGET')
    return {'decisions': decisions, 'target_status': target_status,
            'target': {'metric': args.target_metric, 'reduction': args.target_reduction,
                       'tolerance': args.target_tolerance, 'allow_partial_target': args.allow_partial_target},
            'baseline': {'params': original_params, 'macs': original_macs,
                         'eligible_root_channels': eligible_channels},
            'projected': {'params': baseline['current_params'], 'macs': baseline['current_macs'],
                          'removed_root_channels': removed_total,
                          'param_reduction': 1 - baseline['current_params'] / original_params,
                          'mac_reduction': 1 - baseline['current_macs'] / original_macs,
                          'eligible_root_channel_reduction': removed_total / eligible_channels,
                          'target_metric_reduction': final_reduction},
            'audit': {'candidate_checks': dict(counts_accum), 'rejected_simulations': reject_reasons,
                      'root_original_to_current_remaining': keep,
                      'module_owners': module_owners,
                      'max_decisions_limit': args.max_decisions,
                      'original_conv_cap': args.max_original_conv_prune_ratio,
                      'final_conv_max_removed_ratio': max(
                          (original_widths[n] - m.out_channels) / original_widths[n]
                          for n, m in current.named_modules()
                          if isinstance(m, torch.nn.Conv2d)
                          and n in original_widths)},
            'note': ('This is a reproducible plan validated by physical pruning ONLY on scratch clones. '
                     'NO pruned checkpoint is saved. Stage 05 must replay and reverify every step; '
                     'Stage 06 and QAT/export checks remain mandatory.')}


def write_changes_csv(path: Path, decisions: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction='ignore')
        w.writeheader()
        for item in decisions:
            r = dict(item)
            r['original_indices'] = ','.join(map(str, item['original_indices']))
            r['current_indices'] = ','.join(map(str, item['current_indices']))
            r['affected_modules'] = ';'.join(item['affected_modules'])
            w.writerow(r)


def run(args) -> int:
    out = resolved(args.output)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_log(out, STAGE, NAME)
    gatepath = gate_file(out, STAGE, NAME)
    if gatepath.exists() and not args.overwrite:
        logger.error('Existing Stage 04 gate preserved: %s; use --overwrite to replace it', gatepath)
        return 2
    stage_dir = out / 'stages' / STAGE
    stage_dir.mkdir(parents=True, exist_ok=True)
    checks = {key: False for key in (
        'cli_valid', 'predecessor_gates_current', 'model_identity_unchanged',
        'gradient_scores_complete', 'exact_depgraph_candidates_checked',
        'all_decisions_scratch_simulated', 'alignment_and_protection_respected',
        'budget_policy_satisfied', 'artifacts_saved')}
    errors, warnings, provenance, artifacts = [], [], {}, {}
    plan_path = stage_dir / '04_plan.json'
    csv_path = stage_dir / '04_channel_changes.csv'
    try:
        if args.overwrite:
            for p in (out/'gates').glob('*.json'):
                if p.name[:2].isdigit() and int(p.name[:2]) >= 4:
                    p.unlink()
            for stale in (plan_path, csv_path, stage_dir/'04_gate.json', stage_dir/'04_error_traceback.txt'):
                if stale.is_file():
                    stale.unlink()
            logger.warning('--overwrite invalidated Stage 04..10 gates/artifacts; kept Stage 00..03 untouched')
        args.imgsz_hw = make_imgsz(args.imgsz)
        if args.channel_align < 1 or args.min_channels < 1 or args.min_channels % args.channel_align:
            raise ValueError('Positive channel-align required and min-channels must be its multiple')
        if not 0 < args.max_original_conv_prune_ratio <= 1 or args.balance_strength < 0:
            raise ValueError('Invalid cumulative Conv cap or balance strength')
        if args.max_original_conv_prune_ratio < 1 and not args.reference_model:
            raise ValueError('--reference-model required for cumulative original Conv cap')
        if args.reference_model and not resolved(args.reference_model).is_file():
            raise FileNotFoundError('--reference-model checkpoint missing')
        if args.iterative_steps < 1 or args.max_decisions < 1 or args.batch < 1 or args.workers < 0:
            raise ValueError('iterative-steps/max-decisions/batch >=1 and workers >=0 required')
        if not (0 < args.target_reduction < 1) or not (0 <= args.target_tolerance < .5):
            raise ValueError('target-reduction must be (0,1); target-tolerance must be [0,0.5)')
        checks['cli_valid'] = True
        data_path = resolved(args.data)
        dataset = read_dataset(data_path, args.expected_classes)
        ctx = contract(args.model, data_path, args.imgsz_hw, args.device, args.batch,
                       args.workers, args.family, args.expected_classes)
        gates, graph, importance, checkpoint, other = require_ancestors(out, ctx, dataset)
        checks['predecessor_gates_current'] = True
        stage2, stage02_source_sha = load_stage02()
        g0, g1, g2, g3 = gates
        p2cfg = g2['provenance']['stage02_config']
        if (p2cfg['trace_mode'] != 'train' or args.channel_align != p2cfg['channel_align'] or
                args.min_channels != p2cfg['min_channels']):
            raise RuntimeError('Stage 04 align/min/trace must match Stage 02; rerun earlier stages if changing constraints')
        if not (p2cfg['protect_detect_head'] and p2cfg['protect_attention'] and p2cfg['protect_static_splits']):
            raise RuntimeError('Unprotected detection head, attention or static split unsupported in Stage 04')
        scores = read_channel_scores(Path(other['stage03_artifacts']['scores_csv']), graph, importance)
        checks['gradient_scores_complete'] = True
        baseline_json = read_json(out/'stages'/'01'/'01_profile.json')
        base_params = int(baseline_json['profile']['parameter_count'])
        base_macs = float(baseline_json['profile']['macs'])
        if base_params <= 0 or not math.isfinite(base_macs) or base_macs <= 0:
            raise RuntimeError('Invalid Stage 01 baseline resource metrics')
        provenance = {**g0['provenance'],
                      **{f'stage{n}_gate_sha256': sha256_file(gate_file(out,n,name)) for n,name in GATES},
                      'stage02_source_sha256': stage02_source_sha,
                      'stage02_graph_sha256': sha256_file(out/'stages'/'02'/'02_graph.json'),
                      'stage03_importance_sha256': sha256_file(Path(other['stage03_artifacts']['importance_json'])),
                      'stage03_scores_sha256': sha256_file(Path(other['stage03_artifacts']['scores_csv'])),
                      'stage04_config': {'method': args.method, 'target_metric': args.target_metric,
                                         'target_reduction': args.target_reduction, 'target_tolerance': args.target_tolerance,
                                         'channel_align': args.channel_align, 'min_channels': args.min_channels,
                                         'iterative_steps': args.iterative_steps, 'max_decisions': args.max_decisions,
                                         'ranking': args.ranking, 'candidate_mode': args.candidate_mode,
                                         'allow_partial_target': args.allow_partial_target,
                                         'reference_model': str(resolved(args.reference_model)) if args.reference_model else None,
                                         'reference_model_sha256': sha256_file(resolved(args.reference_model)) if args.reference_model else None,
                                         'max_original_conv_prune_ratio': args.max_original_conv_prune_ratio,
                                         'balance_strength': args.balance_strength,
                                         'normalize_sensitivity': args.normalize_sensitivity}}
        from ultralytics import YOLO
        import torch
        import torch_pruning as tp
        device = select_device(args.device)
        yolo = YOLO(str(checkpoint))
        if getattr(yolo, 'task', None) != 'detect' or len(getattr(yolo, 'names', {})) != dataset['nc']:
            raise RuntimeError('Checkpoint detection task / class mismatch')
        net = yolo.model.to(device).float().train().requires_grad_(True)
        if model_head_metadata(net).get('head_nc') != dataset['nc']:
            raise RuntimeError('Detection head class count changed')
        checks['model_identity_unchanged'] = True
        logger.info('Planning %s reduction %.2f%% using %d gradient-scored roots (CPU/GPU scratch model only)',
                    args.target_metric, args.target_reduction * 100, len(scores))
        info = plan_scratch(net, tp, stage2, graph, scores,
                            {'base_params': base_params, 'base_macs': base_macs,
                             'stage02_config': p2cfg,
                             'current_params': base_params, 'current_macs': base_macs},
                            args, device, logger)
        checks['exact_depgraph_candidates_checked'] = info['audit']['candidate_checks'].get('scanned', 0) > 0
        checks['all_decisions_scratch_simulated'] = bool(info['decisions']) and all(d['status'] == 'SIMULATED_VALID' for d in info['decisions'])
        checks['alignment_and_protection_respected'] = checks['all_decisions_scratch_simulated']
        is_partial = info['target_status'] != 'TARGET_REACHED'
        checks['budget_policy_satisfied'] = not is_partial or args.allow_partial_target
        if is_partial:
            warnings.append(f'PARTIAL_TARGET: achieved {info["projected"]["target_metric_reduction"]:.3%} '
                            f'vs requested {args.target_reduction:.3%}; '
                            f'allow_partial_target={args.allow_partial_target}')
            if not args.allow_partial_target:
                errors.append('PARTIAL_TARGET: target not reached; rerun with --allow-partial-target or lower target')
        if not info['decisions']:
            errors.append('No valid physical-pruning decisions; cannot create an actionable plan')
        info.update({'step': f'{STAGE}_{NAME}', 'created_utc': now_utc(),
                     'model_family': graph['family'], 'model_scale': graph['scale'],
                     'checkpoint_sha256': g0['provenance']['checkpoint_sha256'],
                     'stage02_graph_sha256': provenance['stage02_graph_sha256'],
                     'stage03_scores_sha256': provenance['stage03_scores_sha256'],
                     'channel_align': args.channel_align, 'min_channels': args.min_channels,
                     'imgsz_hw': list(args.imgsz_hw),
                     'source_checkpoint': str(checkpoint),
                     'stage02_safety_policy': p2cfg})
        write_json(plan_path, info)
        write_changes_csv(csv_path, info['decisions'])
        artifacts = {'plan': str(plan_path), 'channel_changes': str(csv_path)}
        checks['artifacts_saved'] = all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in artifacts.values())
        provenance['artifact_hashes'] = {k: sha256_file(Path(v)) for k,v in artifacts.items()}
        logger.info('Stage 04 plan: %d actions, achieved %.3f%% of %s, target_status=%s',
                    len(info['decisions']), 100*info['projected']['target_metric_reduction'],
                    args.target_metric, info['target_status'])
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
        logger.error('FAIL: %s', errors[-1])
        (stage_dir/'04_error_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        gate = save_gate(out, STAGE, NAME, checks=checks, errors=errors,
                         warnings=warnings, provenance=provenance, artifacts=artifacts)
        if 'plan' in artifacts and Path(artifacts['plan']).is_file():
            gate['target_status'] = read_json(Path(artifacts['plan'])).get('target_status')
            write_json(gate_file(out, STAGE, NAME), gate)
        write_json(stage_dir/'04_gate.json', gate)
        logger.info('[04] %s (target=%s) — %s', gate['status'], gate.get('target_status', 'UNKNOWN'), gatepath)
    return 0 if gate['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(run(make_parser().parse_args()))
