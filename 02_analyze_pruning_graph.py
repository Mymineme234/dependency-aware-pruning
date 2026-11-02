#!/usr/bin/env python3
"""02 — Conservative dependency-aware pruning feasibility analysis for YOLO detection.

No weights, original checkpoints, or dataset files are modified. 02 PASS means:
  * valid 00/01 contracts and checkpoint/dataset fingerprint;
  * DepGraph autograd trace succeeded on the requested training graph;
  * every enumerated root has been classified into ELIGIBLE or PROTECTED;
  * at least one output-channel pruning probe with explicit dependent indices is safe
    according to the checks implemented here.

This is NOT a certificate that arbitrary future channel sets, future multi-stage
prunes, serialization, or QAT will work. Stages 04–06 must revalidate the exact
selected indices and physically modified graph. See README_02.md.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import sys
import traceback
from pathlib import Path
from typing import Any

from pruning_common import (
    SUPPORTED_FAMILIES, contract, dataset_fingerprint, gate_file, hash_object,
    jsonable, make_imgsz, model_head_metadata, now_utc, read_dataset,
    read_json, resolved, save_gate, select_device, setup_log, sha256_file,
    versions, write_json,
)

STAGE = "02"
NAME = "analyze_pruning_graph"
SENSITIVE_MODULE_TOKENS = (
    "attention", "attn", "psa", "transformer", "areaattention", "c2psa",
    "a2c2f", "c2fattn", "imagepoolingattn", "maxsigmoidattn", "mha"
)
STATIC_SPLIT_MODULES = ("c2f", "c3k2", "c3k", "c3", "c2", "repncspelan", "elan", "c2psa")
VIRTUAL_KNOWN = ("concat", "cat", "add", "elementwise", "mul", "multiply", "split", "chunk", "slice", "getitem", "unbind", "reshape", "view", "flatten", "permute", "transpose", "expand", "repeat", "upsample", "interpolate", "pad", "pixelshuffle")
VIRTUAL_DANGEROUS = ("split", "chunk", "slice", "getitem", "unbind", "reshape", "view", "flatten", "permute", "expand", "repeat", "pixelshuffle")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="02: dependency graph + aligned output-channel feasibility; read only")
    p.add_argument("--model", required=True, help="Exact checkpoint used in 00 and 01")
    p.add_argument("--data", required=True, help="Same YOLO detection data.yaml")
    p.add_argument("--output", required=True, help="Same pruning run directory")
    p.add_argument("--imgsz", nargs="+", type=int, default=[640])
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--expected-classes", type=int, default=None)
    p.add_argument("--family", default="auto", choices=["auto", *SUPPORTED_FAMILIES])
    p.add_argument("--channel-align", type=int, default=8)
    p.add_argument("--min-channels", type=int, default=16)
    p.add_argument("--probe-block", type=int, default=None, help="Root channel test block; defaults to --channel-align")
    p.add_argument("--trace-mode", choices=["train", "eval"], default="train",
                   help="Default train: expose raw detection branches; no optimizer/update")
    p.add_argument("--protect-attention", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--protect-detect-head", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--protect-static-splits", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true", help="Replace 02 and invalidate 03..10 gates")
    return p


def _validate_ancestors(out: Path, current: dict, dataset: dict) -> tuple[dict, dict, Path]:
    g0p, g1p = gate_file(out, "00", "check_environment"), gate_file(out, "01", "profile_baseline")
    for path in (g0p, g1p):
        if not path.is_file():
            raise RuntimeError(f"Missing required predecessor gate: {path}")
    g0, g1 = read_json(g0p), read_json(g1p)
    for num, gate in (("00", g0), ("01", g1)):
        if gate.get("status") != "PASS" or not gate.get("checks") or not all(x is True for x in gate["checks"].values()):
            raise RuntimeError(f"Stage {num} has not PASSed all checks")
    pv0, pv1 = g0.get("provenance", {}), g1.get("provenance", {})
    if pv0.get("contract_sha256") != hash_object(current):
        raise RuntimeError("Stage 00 input contract differs; --model/data/output-dependent params must match exactly")
    if pv1.get("contract_sha256") != pv0.get("contract_sha256"):
        raise RuntimeError("Stage 01 was run under a different 00 contract")
    if pv1.get("stage00_gate_sha256") != sha256_file(g0p):
        raise RuntimeError("Stage 00 gate changed after Stage 01; rerun 01")
    if pv0.get("dataset_yaml_sha256") != dataset["yaml_sha256"]:
        raise RuntimeError("Dataset YAML changed since Stage 00")
    actual_fingerprint = dataset_fingerprint(dataset)
    if actual_fingerprint.get("manifest_sha256") != pv0.get("dataset_manifest", {}).get("manifest_sha256"):
        raise RuntimeError("Dataset image/label inventory changed since Stage 00")
    ckpt = Path(pv0.get("source_checkpoint", ""))
    if not ckpt.is_file() or sha256_file(ckpt) != pv0.get("checkpoint_sha256"):
        raise RuntimeError("Source checkpoint missing/changed since Stage 00")
    now_versions = versions()
    for pkg in ("torch", "torchvision", "ultralytics", "torch-pruning", "numpy"):
        if pv0.get("dependency_versions", {}).get(pkg) != now_versions.get(pkg):
            raise RuntimeError(f"Dependency version changed since 00: {pkg}")
    stage01 = out / "stages" / "01"
    for fn in ("01_profile.json", "01_baseline_graph.json", "01_baseline_val.json"):
        if not (stage01 / fn).is_file():
            raise RuntimeError(f"Stage 01 artifact missing: {stage01 / fn}")
    report = read_json(stage01 / "01_profile.json")
    if report.get("status") != "PASS":
        raise RuntimeError("Stage 01 report is not PASS")
    if not (report.get("profile", {}).get("parameter_count", 0) > 0):
        raise RuntimeError("Stage 01 contains no valid parameter profile")
    val = read_json(stage01 / "01_baseline_val.json")
    if val.get("checkpoint_sha256") != pv0.get("checkpoint_sha256"):
        raise RuntimeError("Stage 01 val result references another model")
    if val.get("model_family") != pv0.get("model_family"):
        raise RuntimeError("Stage 01 val model family mismatch")
    return g0, g1, ckpt


def _output_scalar(output: Any):
    """For AUTOGRAD DEPGRAPH TRACING ONLY, never used as a pruning/importance loss.

    Tracing the train-mode detection model traverses raw head outputs. Detached
    YOLO26 one-to-one inputs are not represented in the autograd graph, hence
    all detection head tensors and boundary channels are conservatively protected.
    """
    import torch
    items = []
    detached = 0

    def walk(v):
        nonlocal detached
        if torch.is_tensor(v):
            if v.requires_grad and v.is_floating_point():
                items.append(v.sum())
            else:
                detached += 1
        elif isinstance(v, dict):
            for value in v.values():
                walk(value)
        elif isinstance(v, (list, tuple)):
            for value in v:
                walk(value)

    walk(output)
    if not items:
        raise RuntimeError("Autograd graph has no differentiable detection outputs; tracing is unsafe")
    return sum(items), {"differentiable_tensors": len(items), "detached_or_non_float_tensors": detached}



def _enable_trace_gradients(net, logger) -> dict[str, int | bool]:
    """Enable autograd on the in-memory analysis model, not the checkpoint file.

    Ultralytics may load inference weights with all parameters frozen.
    This stage never updates weights or saves this in-memory model.
    """
    params = list(net.parameters())
    if not params:
        raise RuntimeError("Checkpoint has no model parameters to trace")
    frozen_before = sum(not p.requires_grad for p in params)
    if frozen_before:
        logger.info("Temporarily enabling autograd for %d/%d frozen parameter tensors in the in-memory tracing model", frozen_before, len(params))
        net.requires_grad_(True)
    if not any(p.requires_grad for p in net.parameters()):
        raise RuntimeError("Unable to enable parameter gradients for DepGraph")
    return {
        "parameter_tensors": len(params),
        "frozen_on_load": frozen_before,
        "temporarily_enabled": frozen_before,
        "source_checkpoint_unchanged": True,
    }


def _node_name(module: Any, mod_names: dict[Any, str]) -> str:
    if module in mod_names:
        return mod_names[module] or "<root>"
    return f"<virtual:{type(module).__name__}>"


def _classify_node(module: Any, name: str) -> str:
    cls = type(module).__name__.lower()
    text = f"{cls} {name.lower()}"
    for key in ("concat", "split", "chunk", "slice", "getitem", "add", "mul", "reshape", "view", "flatten", "transpose", "permute", "upsample", "interpolate", "pad", "elementwise"):
        if key in text:
            return key
    return type(module).__name__


def _is_virtual(module, module_names) -> bool:
    return module not in module_names


def _special_operation(module, module_names) -> tuple[str, bool]:
    if not _is_virtual(module, module_names):
        return "normal_module", False
    k = type(module).__name__.lower()
    known = any(s in k for s in VIRTUAL_KNOWN)
    return _classify_node(module, ""), known


def _head_paths(net) -> set[str]:
    seq = getattr(net, "model", None)
    if seq is None or not hasattr(seq, "__len__") or not len(seq):
        raise RuntimeError("Cannot locate YOLO detection head module sequence")
    head = seq[-1]
    named = dict(net.named_modules())
    head_name = next((name for name, m in named.items() if m is head), None)
    if not head_name:
        raise RuntimeError("Detection head module has no stable named_modules path")
    return {n for n in named if n == head_name or n.startswith(head_name + ".")}


def _attention_paths(net) -> set[str]:
    named = dict(net.named_modules())
    prefixes = []
    for n, m in named.items():
        if n and any(k in type(m).__name__.lower() for k in SENSITIVE_MODULE_TOKENS):
            prefixes.append(n)
    return {name for name in named if any(name == p or name.startswith(p + ".") for p in prefixes)}


def _static_module_paths(net) -> dict[str, list[str]]:
    named = dict(net.named_modules())
    root_prefixes = []
    for name, module in named.items():
        if name and type(module).__name__.lower() in STATIC_SPLIT_MODULES:
            root_prefixes.append(name)
    return {name: [p for p in root_prefixes if name.startswith(p + ".")] for name in named}


def _probe_blocks(channels: int, block: int, min_channels: int) -> list[list[int]]:
    if channels - block < min_channels:
        return []
    starts = [0, max(0, ((channels - block) // 2 // block) * block), channels - block]
    return [list(range(s, s + block)) for s in sorted(set(starts))]


def _get_prune_type(dg, dep) -> str:
    if dg.is_out_channel_pruning_fn(dep.handler):
        return "out"
    if dg.is_in_channel_pruning_fn(dep.handler):
        return "in"
    return "unknown"


def _group_items(group, dg, named: dict, max_indices: int = 32) -> list[dict]:
    out = []
    for dep, idxs in group:
        m = dep.target.module
        name = _node_name(m, named)
        kind = _get_prune_type(dg, dep)
        indices = [int(i) for i in idxs]
        row = {"module": name, "module_type": type(m).__name__, "operation": _classify_node(m, name),
               "prune_kind": kind, "num_indices": len(indices),
               "indices": indices if len(indices) <= max_indices else indices[:max_indices],
               "indices_truncated": len(indices) > max_indices,
               "is_virtual": _is_virtual(m, named), "semantic_c_axis": (
                   1 if _is_conv_or_bn(m) else "last" if type(m).__name__ == "Linear" else "unknown")}
        out.append(row)
    return out


def _is_conv_or_bn(m) -> bool:
    import torch
    return isinstance(m, (torch.nn.modules.conv._ConvNd, torch.nn.modules.batchnorm._BatchNorm))


def _hazards(group, dg, named: dict, head_paths: set[str], attention_paths: set[str],
             static_paths: dict, align: int, minimum: int, protect: dict) -> list[str]:
    import torch
    reasons = []
    for dep, idxs in group:
        module = dep.target.module
        name = _node_name(module, named)
        idxs = [int(i) for i in idxs]
        if not idxs:
            reasons.append("EMPTY_DEPENDENCY_MAPPING")
            continue
        if len(set(idxs)) != len(idxs) or min(idxs) < 0:
            reasons.append("DUPLICATE_OR_NEGATIVE_DEPENDENCY_INDEX")
        if _is_virtual(module, named):
            op, is_known = _special_operation(module, named)
            if not is_known:
                reasons.append(f"UNKNOWN_VIRTUAL_OPERATOR:{type(module).__name__}")
            elif any(k in op.lower() for k in VIRTUAL_DANGEROUS):
                # Fixed chunk/split/index metadata can be wrong even if DepGraph
                # derives a pruning mapping. No unverified automatic mutation.
                if protect["static_splits"]:
                    reasons.append(f"UNVERIFIED_STATIC_INDEX_OPERATOR:{op}")
            elif op.lower() in ("concat", "cat", "add"):
                # DepGraph's explicit indices are retained in CSV/JSON, but this
                # only proves a candidate probe, not all channel choices.
                pass
            continue
        # Resolve the dependency handler BEFORE inspecting static module paths.
        # Previously `kind` was referenced before assignment in the cv1/cv3
        # protection branch, incorrectly classifying safe probes as failures.
        kind = _get_prune_type(dg, dep)
        if protect["head"] and name in head_paths:
            reasons.append("PROTECTED_DETECTION_HEAD")
        if protect["attention"] and name in attention_paths:
            reasons.append("PROTECTED_ATTENTION")
        if protect["static_splits"]:
            # Python-side hidden C2f/C3k2 metadata can describe internal hidden
            # channels. Protect changing the split producer (usually .cv1 conv).
            for ancestor in static_paths.get(name, ()):
                inner = name[len(ancestor) + 1:].split(".")
                if kind == "out" and inner and inner[0] in ("cv1", "cv3"):
                    reasons.append(f"STATIC_MODULE_INTERNAL_WIDTH:{ancestor}")
        if kind == "unknown":
            reasons.append(f"UNKNOWN_PRUNING_HANDLER:{name}")
            continue
        try:
            width = int(dg.get_out_channels(module) if kind == "out" else dg.get_in_channels(module))
        except Exception:
            width = None
        if width is not None:
            if max(idxs) >= width:
                reasons.append(f"OUT_OF_RANGE_INDEX:{name}:{kind}")
            after = width - len(idxs)
            if after <= 0:
                reasons.append(f"CHANNEL_ZERO_OR_NEGATIVE:{name}")
            # Enforce C%8 for feature widths; explicit exemptions are model RGB
            # input and immutable fixed-size detection head outputs only.
            is_rgb_entry = kind == "in" and isinstance(module, torch.nn.modules.conv._ConvNd) and width == 3
            if not is_rgb_entry and isinstance(module, (torch.nn.modules.conv._ConvNd, torch.nn.Linear)):
                if after < minimum:
                    reasons.append(f"BELOW_MIN_CHANNELS:{name}:{kind}:{after}")
                if after % align:
                    reasons.append(f"MISALIGNED_CHANNELS:{name}:{kind}:{after}")
            if is_rgb_entry:
                reasons.append("RGB_INPUT_CANNOT_BE_PRUNED")
        else:
            if isinstance(module, (torch.nn.modules.conv._ConvNd, torch.nn.Linear)):
                reasons.append(f"MISSING_CHANNEL_COUNT:{name}:{kind}")
        if isinstance(module, torch.nn.modules.conv._ConvNd):
            g = int(module.groups)
            if g > 1:
                # No safe handling of changing grouped convolution partitions
                # without validated group-specific index remapping in v1.
                if g not in (module.in_channels, module.out_channels):
                    reasons.append(f"UNVERIFIED_GROUPED_CONV:{name}:groups={g}")
                elif module.in_channels != module.out_channels:
                    reasons.append(f"UNVERIFIED_DEPTHWISE_MULTIPLIER:{name}")
    return sorted(set(reasons))


def _safe_probe(dg, module, block_indices, named, protected, align, minimum, config):
    pruner = dg.get_pruner_of_module(module)
    if pruner is None or getattr(pruner, "prune_out_channels", None) is None:
        return None, ["NO_ROOT_OUT_CHANNEL_PRUNER"]
    try:
        group = dg.get_pruning_group(module, pruner.prune_out_channels, block_indices)
        if not dg.check_pruning_group(group):
            return None, ["DEPGRAPH_REJECTED_PRUNING_GROUP"]
        issues = _hazards(group, dg, named, protected["head_paths"],
                          protected["attention_paths"], protected["static_paths"],
                          align, minimum, config)
        if issues:
            return None, issues
        items = _group_items(group, dg, named)
        if not items or not any(i["prune_kind"] == "out" and i["module"] == named[module] for i in items):
            return None, ["NO_ROOT_PRUNING_DEPENDENCY"]
        return items, []
    except Exception as exc:
        return None, [f"DEPGRAPH_PROBE_FAILED:{type(exc).__name__}:{str(exc)[:200]}"]


def _write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize_nodes(dg, named) -> tuple[list[dict], dict[str, int]]:
    rows = []
    counts = collections.Counter()
    for m, node in getattr(dg, "module2node", {}).items():
        name = _node_name(m, named)
        op = _classify_node(m, name)
        counts[op] += 1
        deps = []
        for dep in getattr(node, "dependencies", ()):
            try:
                deps.append({"to": _node_name(dep.target.module, named),
                             "handler": str(getattr(dep.handler, "__name__", type(dep.handler).__name__))})
            except Exception:
                deps.append({"to": "<unresolved>", "handler": "unknown"})
        rows.append({"name": name, "operator": op, "class": type(m).__name__,
                     "virtual": _is_virtual(m, named),
                     "semantic_c_axis": (1 if _is_conv_or_bn(m) else "last" if type(m).__name__ == "Linear" else "unknown"),
                     "dependencies": deps})
    return rows, dict(counts)


def analyze(net, x, tp, args, logger) -> dict:
    import torch
    named = dict(net.named_modules())
    reverse = {m: n for n, m in named.items()}
    head = _head_paths(net)
    attention = _attention_paths(net)
    static = _static_module_paths(net)
    protected = {"head_paths": head, "attention_paths": attention, "static_paths": static}
    config = {"head": args.protect_detect_head, "attention": args.protect_attention,
              "static_splits": args.protect_static_splits}
    if not config["head"]:
        # In this version even --no-protect-detect-head is not supported: it
        # would require model-specific output ABI/coupling verification.
        raise ValueError("--no-protect-detect-head is not implemented safely in Stage 02")
    if not config["static_splits"]:
        raise ValueError("--no-protect-static-splits needs an audited YOLO family adapter; refusing unsafe bypass")
    if not config["attention"]:
        raise ValueError("--no-protect-attention needs a tested attention adapter; refusing unsafe bypass")
    if args.trace_mode == "train":
        net.train()
    else:
        net.eval()
    net.zero_grad(set_to_none=True)
    trace_stats = {}
    observed = {}
    # Collect tensor shapes for channel-bearing modules during autograd trace.
    hooks = []
    for path, m in named.items():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear)):
            def make_hook(n):
                def hook(_m, ins, outs):
                    i = ins[0] if ins and torch.is_tensor(ins[0]) else None
                    o = outs if torch.is_tensor(outs) else None
                    observed[n] = {"input_shape": list(i.shape) if i is not None else None,
                                   "output_shape": list(o.shape) if o is not None else None,
                                   "channel_axis": 1 if isinstance(_m, (torch.nn.Conv2d, torch.nn.BatchNorm2d)) else -1}
                return hook
            hooks.append(m.register_forward_hook(make_hook(path)))
    try:
        def forward_for_trace(model, images):
            scalar, stats = _output_scalar(model(images))
            trace_stats.update(stats)
            return scalar
        with torch.enable_grad():
            dg = tp.DependencyGraph().build_dependency(
                net, example_inputs=x, forward_fn=forward_for_trace, verbose=False)
    finally:
        for handle in hooks:
            handle.remove()
        net.zero_grad(set_to_none=True)
    if not getattr(dg, "module2node", {}):
        raise RuntimeError("DepGraph traced zero nodes")
    nodes, opcounts = _summarize_nodes(dg, reverse)
    convs = [(n, m) for n, m in named.items() if isinstance(m, torch.nn.Conv2d)]
    if not convs:
        raise RuntimeError("No torch.nn.Conv2d layers found")
    align = args.channel_align
    minimum = args.min_channels
    block = args.probe_block or align
    eligible_rows, protected_rows, all_groups = [], [], []
    for n, m in convs:
        n_channels = int(m.out_channels)
        groupid = f"g{len(all_groups):04d}"
        base = {"group_id": groupid, "root": n, "root_type": type(m).__name__,
                "channels_before": n_channels, "candidate_block_size": block,
                "root_groups": int(m.groups)}
        reasons = []
        if m not in getattr(dg, "module2node", {}):
            reasons.append("ROOT_MISSING_FROM_DEPGRAPH")
        if n in head:
            reasons.append("ROOT_INSIDE_DETECTION_HEAD")
        if n in attention:
            reasons.append("ROOT_INSIDE_ATTENTION")
        if m.groups != 1:
            reasons.append("ROOT_GROUPED_OR_DEPTHWISE_NOT_YET_SUPPORTED")
        if n_channels % align:
            reasons.append(f"ROOT_CHANNELS_NOT_MULTIPLE_OF_{align}")
        if block % align:
            reasons.append("PROBE_BLOCK_NOT_ALIGNED")
        if n_channels - block < minimum:
            reasons.append("ROOT_BELOW_MIN_AFTER_PROBE")
        if args.trace_mode == "eval" and getattr(net, "end2end", False):
            reasons.append("END2END_EVAL_GRAPH_TRACE_NOT_CERTIFIED")
        probes = []
        if not reasons:
            for ids in _probe_blocks(n_channels, block, minimum):
                deps, fail = _safe_probe(dg, m, ids, reverse, protected, align, minimum, config)
                probes.append({"root_indices": ids, "pass": deps is not None,
                               "dependencies": deps or [], "reasons": fail})
            if not any(p["pass"] for p in probes):
                reasons.append("NO_SAFE_CHANNEL_BLOCK_FOUND")
        base["probes"] = probes
        success = [probe for probe in probes if probe["pass"]]
        base["feasibility"] = "ELIGIBLE_PROBE_ONLY" if success else "PROTECTED"
        base["reason_codes"] = sorted(set(reasons + [r for p in probes for r in p["reasons"]]))
        base["safe_probe_count"] = len(success)
        # This bound ignores cross-group constraints and is not a feasible global ratio.
        base["per_root_upper_bound_prunable_channels"] = (max(0, n_channels - minimum) // align * align) if success else 0
        all_groups.append(base)
        row = {k: v for k, v in base.items() if k not in ("probes", "reason_codes")}
        row["reasons"] = ";".join(base["reason_codes"])
        row["safe_probe_indices"] = ";".join(",".join(str(i) for i in p["root_indices"]) for p in success)
        if success:
            eligible_rows.append(row)
            logger.info("ELIGIBLE %s %s (%d C, %d/%d probe blocks passed)", groupid, n, n_channels, len(success), len(probes))
        else:
            protected_rows.append(row)
    logger.info("DepGraph nodes=%d; roots=%d; eligible=%d; protected=%d", len(nodes), len(all_groups), len(eligible_rows), len(protected_rows))
    if trace_stats.get("differentiable_tensors", 0) == 0:
        raise RuntimeError("DepGraph forward contains no differentiable output")
    return {"graph": dg, "nodes": nodes, "opcounts": opcounts,
            "head_paths": sorted(head), "attention_paths": sorted(attention),
            "static_modules": sorted(set(p for ps in static.values() for p in ps)),
            "groups": all_groups, "eligible": eligible_rows, "protected": protected_rows,
            "trace_stats": trace_stats, "observed_shapes": observed}


def run(args) -> int:
    out = resolved(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_log(out, STAGE, NAME)
    gatepath = gate_file(out, STAGE, NAME)
    if gatepath.exists() and not args.overwrite:
        log.error("Existing Stage 02 gate preserved: %s. Use --overwrite to invalidate later gates.", gatepath)
        return 2
    checks = {k: False for k in (
        "cli_valid", "predecessor_gates_current", "model_identity_unchanged", "depgraph_built_autograd",
        "groups_classified", "eligible_groups_present", "protected_hazards_excluded", "eligible_probe_mapping_complete", "artifacts_saved"
    )}
    errors, warnings, provenance = [], [], {}
    report = {"stage": f"{STAGE}_{NAME}", "started_utc": now_utc(), "status": "FAIL"}
    stage_dir = out / "stages" / STAGE
    stage_dir.mkdir(parents=True, exist_ok=True)
    eligible_csv = stage_dir / "02_eligible_groups.csv"
    protected_csv = stage_dir / "02_protected_groups.csv"
    graph_json = stage_dir / "02_graph.json"
    try:
        if args.overwrite:
            for p in (out / "gates").glob("*.json"):
                if p.name[:2].isdigit() and int(p.name[:2]) >= 2:
                    p.unlink()
            log.warning("--overwrite invalidated 02..10 gates (not 00/01)")
        hw = make_imgsz(args.imgsz)
        if args.batch < 1 or args.workers < 0:
            raise ValueError("batch must be >=1 and workers >=0")
        if args.channel_align <= 0 or args.min_channels <= 0:
            raise ValueError("channel-align and min-channels must be positive")
        if args.min_channels % args.channel_align:
            raise ValueError("--min-channels must be multiple of --channel-align")
        if args.probe_block is not None and (args.probe_block < args.channel_align or args.probe_block % args.channel_align):
            raise ValueError("--probe-block must be a positive multiple of --channel-align")
        if args.expected_classes is not None and args.expected_classes <= 0:
            raise ValueError("--expected-classes must be positive")
        checks["cli_valid"] = True
        data_path = resolved(args.data)
        context = contract(args.model, data_path, hw, args.device, args.batch, args.workers, args.family, args.expected_classes)
        data = read_dataset(data_path, args.expected_classes)
        g0, g1, ckpt = _validate_ancestors(out, context, data)
        checks["predecessor_gates_current"] = True
        provenance = {**g0["provenance"], "stage00_gate_sha256": sha256_file(gate_file(out, "00", "check_environment")),
                      "stage01_gate_sha256": sha256_file(gate_file(out, "01", "profile_baseline")),
                      "stage01_profile_sha256": sha256_file(out / "stages" / "01" / "01_profile.json"),
                      "stage02_config": {"channel_align": args.channel_align, "min_channels": args.min_channels,
                                         "probe_block": args.probe_block or args.channel_align, "trace_mode": args.trace_mode,
                                         "protect_attention": args.protect_attention,
                                         "protect_detect_head": args.protect_detect_head,
                                         "protect_static_splits": args.protect_static_splits}}
        report["inputs"] = {**context, **provenance["stage02_config"]}
        from ultralytics import YOLO
        import torch
        import torch_pruning as tp
        device = select_device(args.device)
        yolo = YOLO(str(ckpt))
        family = g0["provenance"]["model_family"]
        if getattr(yolo, "task", None) != "detect":
            raise RuntimeError("Task changed since Stage 00")
        if len(getattr(yolo, "names", {})) != data["nc"]:
            raise RuntimeError("Class count changed since Stage 00")
        head_meta = model_head_metadata(yolo.model)
        if head_meta.get("head_nc") != data["nc"]:
            raise RuntimeError("Detection head class count mismatch")
        checks["model_identity_unchanged"] = True
        # One scratch model, always FP32 and UNFUSED; no in-place pruning.
        net = yolo.model.to(device).float()
        report["autograd_trace"] = _enable_trace_gradients(net, log)
        x = torch.zeros((1, 3, *hw), device=device, dtype=torch.float32, requires_grad=True)
        # Build graph, classify group boundaries and sample exact dependency index paths.
        result = analyze(net, x, tp, args, log)
        checks["depgraph_built_autograd"] = True
        checks["groups_classified"] = len(result["groups"]) == len(result["eligible"]) + len(result["protected"])
        checks["eligible_groups_present"] = bool(result["eligible"])
        if not result["eligible"]:
            errors.append("No safe aligned pruning groups found; examine 02_protected_groups.csv and topology before changing policy")
        checks["protected_hazards_excluded"] = all(g["feasibility"] != "ELIGIBLE_PROBE_ONLY" or any(p["pass"] for p in g["probes"])
                                                     for g in result["groups"])
        checks["eligible_probe_mapping_complete"] = all(any(p["pass"] and p["dependencies"] for p in g["probes"])
                                                        for g in result["groups"] if g["feasibility"] == "ELIGIBLE_PROBE_ONLY")
        fields = ["group_id", "root", "root_type", "channels_before", "candidate_block_size", "root_groups",
                  "feasibility", "safe_probe_count", "per_root_upper_bound_prunable_channels", "reasons", "safe_probe_indices"]
        _write_csv(eligible_csv, result["eligible"], fields)
        _write_csv(protected_csv, result["protected"], fields)
        graph_report = {"family": family, "scale": g0["provenance"].get("model_scale"),
                        "trace_mode": args.trace_mode, "trace_input": [1, 3, *hw],
                        "trace_outputs": result["trace_stats"], "module_shapes": result["observed_shapes"],
                        "graph_node_count": len(result["nodes"]), "operators": result["opcounts"],
                        "nodes": result["nodes"],
                        "protected_detection_head": result["head_paths"],
                        "protected_attention": result["attention_paths"],
                        "sensitive_static_modules": result["static_modules"],
                        "groups": result["groups"],
                        "summary": {"conv_roots": len(result["groups"]),
                                    "eligible_probe_roots": len(result["eligible"]),
                                    "protected_roots": len(result["protected"]),
                                    "eligible_root_channels_total": sum(row["channels_before"] for row in result["eligible"]),
                                    "single_group_upper_bound_channel_removals": sum(row["per_root_upper_bound_prunable_channels"] for row in result["eligible"]),
                                    "upper_bound_warning": "Sum of per-root limits is NOT a simultaneous feasible or accuracy-safe pruning ratio."},
                        "limits": [
                            "No physical pruning performed; Stage 04/05 must re-check exact selected channel indices after every edit.",
                            "A successful DepGraph test on sampled 8-channel blocks does not certify all blocks or combined pruning operations.",
                            "Semantic channel axis is known only for Conv/BN/Linear module outputs; opaque PyTorch function nodes are classified conservatively.",
                            "Head, static split/chunk and attention dependencies are protected; YOLO26 detached one-to-one head remains outside complete autograd trace.",
                            "No GFLOPs/accuracy claims beyond Stage 01 baseline; compression feasibility requires Stage 04+.",
                        ]}
        write_json(graph_json, graph_report)
        checks["artifacts_saved"] = all(p.exists() for p in (eligible_csv, protected_csv, graph_json))
        report["graph_summary"] = graph_report["summary"]
        report["operator_summary"] = result["opcounts"]
        log.info("Stage 02 result: %d eligible probe roots, %d protected roots", len(result["eligible"]), len(result["protected"]))
        warnings.append("02 PASS proves only conservative group feasibility probes; later stages MUST validate exact chosen channel indices")
        if head_meta.get("head_end2end"):
            warnings.append("End-to-end dual-head: detached one-to-one branch not fully visible to autograd; detection head protected")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        log.error("FAIL: %s", errors[-1])
        report["traceback"] = traceback.format_exc()
    finally:
        gate = save_gate(out, STAGE, NAME, checks=checks, errors=errors, warnings=warnings,
                         provenance=provenance, artifacts={"graph": str(graph_json),
                         "eligible_groups": str(eligible_csv), "protected_groups": str(protected_csv)})
        write_json(stage_dir / "02_gate.json", gate)
        report.update({"status": gate["status"], "checks": checks, "errors": errors,
                       "warnings": warnings, "completed_utc": now_utc(), "gate": str(gatepath)})
        write_json(stage_dir / "02_report.json", report)
        log.info("[02] %s — %s", gate["status"], gatepath)
    return 0 if gate["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(run(parser().parse_args()))
