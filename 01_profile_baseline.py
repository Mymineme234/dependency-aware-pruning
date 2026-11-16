#!/usr/bin/env python3
"""Stage 01: baseline FP32 profile and validation, without pruning.

Requires a current PASS gate from stage 00. Never evaluates the test split.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import statistics
import sys
import time
import traceback
from pathlib import Path

from pruning_common import (
    SUPPORTED_FAMILIES, checkpoint_path, contract, dataset_fingerprint,
    family_from_metadata, gate_file, hash_object, make_imgsz, model_head_metadata,
    now_utc, read_dataset, read_json, resolved, save_gate, select_device,
    setup_log, sha256_file, versions, write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 01: baseline profile + validation with per-class AP")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--imgsz", type=int, nargs="+", default=[640])
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--expected-classes", type=int, default=None)
    p.add_argument("--family", choices=["auto", *SUPPORTED_FAMILIES], default="auto")
    p.add_argument("--max-det", type=int, default=1000)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--profile-repeats", type=int, default=2,
                   help="Repeat MAC profiler; require matching results")
    p.add_argument("--warmup-iters", type=int, default=2, help="Forward warmups for latency only")
    p.add_argument("--bench-iters", type=int, default=5, help="Forward timing runs; 0 disables")
    p.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nms-free", action="store_true", help="YOLO26 ONLY: one-to-one NMS-free validation")
    p.add_argument("--overwrite", action="store_true", help="Overwrite 01 outputs and invalidate 02..10 gates")
    return p.parse_args()


def _check_00(out: Path, context: dict, dataset: dict) -> tuple[dict, Path]:
    prev_path = gate_file(out, "00", "check_environment")
    if not prev_path.is_file():
        raise RuntimeError(f"Missing stage 00 gate: {prev_path}")
    prev = read_json(prev_path)
    if prev.get("status") != "PASS" or not all(prev.get("checks", {}).values()):
        raise RuntimeError("Stage 00 has not PASSed all mandatory checks")
    provenance = prev.get("provenance", {})
    if hash_object(context) != provenance.get("contract_sha256"):
        raise RuntimeError("Stage 00 input contract differs (model/data/imgsz/device/batch/workers/family). Re-run 00.")
    if dataset["yaml_sha256"] != provenance.get("dataset_yaml_sha256"):
        raise RuntimeError("data.yaml changed since 00; re-run stage 00")
    now_manifest = dataset_fingerprint(dataset)
    if now_manifest.get("manifest_sha256") != provenance.get("dataset_manifest", {}).get("manifest_sha256"):
        raise RuntimeError("Dataset image/label stat inventory changed since stage 00; re-run 00")
    checkpoint = Path(provenance.get("source_checkpoint", ""))
    if not checkpoint.is_file() or sha256_file(checkpoint) != provenance.get("checkpoint_sha256"):
        raise RuntimeError("Checkpoint changed or missing since stage 00")
    v0 = provenance.get("dependency_versions", {})
    v1 = versions()
    for package in ("torch", "torchvision", "ultralytics", "torch-pruning", "numpy"):
        if v0.get(package) != v1.get(package):
            raise RuntimeError(f"Dependency version changed since 00: {package}: {v0.get(package)} => {v1.get(package)}")
    return prev, checkpoint


def _inventory(net) -> tuple[list[dict], dict]:
    import torch
    records = []
    histogram = collections.Counter()
    for name, m in net.named_modules():
        classname = type(m).__name__
        histogram[classname] += 1
        rec = {"path": name or "<root>", "module": classname,
               "direct_parameters": sum(p.numel() for p in m.parameters(recurse=False))}
        if isinstance(m, torch.nn.modules.conv._ConvNd):
            rec.update({"input_channels": m.in_channels, "output_channels": m.out_channels,
                        "groups": m.groups, "kernel": list(m.kernel_size), "stride": list(m.stride),
                        "weight_shape": list(m.weight.shape)})
        elif isinstance(m, (torch.nn.Linear, torch.nn.BatchNorm2d)):
            rec["shape"] = list(m.weight.shape) if m.weight is not None else []
        records.append(rec)
    return records, dict(histogram.most_common())


def _macs(net, x, repeats: int):
    import torch_pruning as tp
    runs = []
    for n in range(repeats):
        result = tp.utils.count_ops_and_params(net, example_inputs=x)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise RuntimeError(f"Torch-Pruning returned unsupported profiler result: {type(result).__name__}")
        macs, params = (float(result[0]), float(result[1]))
        if not (math.isfinite(macs) and math.isfinite(params) and macs > 0 and params > 0):
            raise RuntimeError(f"Invalid profiler result on repetition {n + 1}: {(macs, params)}")
        runs.append({"macs": macs, "parameters": params})
    # Exact values should be deterministic for a fixed-shape forward.
    if any(abs(r["macs"] - runs[0]["macs"]) > max(1, runs[0]["macs"] * 1e-6) for r in runs[1:]):
        raise RuntimeError(f"MAC profile inconsistent across repeats: {runs}")
    if any(abs(r["parameters"] - runs[0]["parameters"]) > 1 for r in runs[1:]):
        raise RuntimeError(f"Parameter count inconsistent across repeats: {runs}")
    return runs


def _latency(net, x, device, warmup_iters: int, bench_iters: int) -> dict:
    import torch
    if not bench_iters:
        return {"status": "SKIPPED", "note": "--bench-iters 0"}
    samples = []
    # Wall-time of raw PyTorch NN forward, not end-to-end preprocessing/NMS.
    with torch.inference_mode():
        for _ in range(warmup_iters):
            net(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(bench_iters):
            start = time.perf_counter()
            net(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - start) * 1000)
    return {"status": "MEASURED", "scope": "raw_torch_forward_only_no_preproc_nms",
            "device": str(device), "batch": 1, "n": bench_iters,
            "median_ms": statistics.median(samples), "mean_ms": statistics.mean(samples),
            "min_ms": min(samples), "max_ms": max(samples), "individual_ms": samples}


def _get(seq, idx: int, default=None):
    try:
        return seq[idx]
    except (KeyError, IndexError, TypeError):
        return default


def _metrics_report(metrics, names: dict[str, str]) -> dict:
    """Map Ultralytics AP row indices to class IDs (ap_class_index can be reordered)."""
    box = metrics.box
    indices = [int(v) for v in box.ap_class_index]
    if len(set(indices)) != len(indices) or any(str(i) not in names for i in indices):
        raise RuntimeError(f"Invalid class index mapping in metric rows: {indices}")
    per_class = []
    for row_i, cid in enumerate(indices):
        p, r, ap50, ap5095 = (float(v) for v in box.class_result(row_i))
        row_ap = _get(box.all_ap, row_i, [])
        ap75 = _get(row_ap, 5)
        f1 = _get(box.f1, row_i)
        if f1 is None:
            f1 = 2 * p * r / (p + r + 1e-12)
        nt = getattr(metrics, "nt_per_class", None)
        ni = getattr(metrics, "nt_per_image", None)
        per_class.append({"class_id": cid, "class_name": names[str(cid)],
                          "precision": p, "recall": r, "f1": float(f1),
                          "ap50": ap50, "ap75": float(ap75) if ap75 is not None else None,
                          "ap50_95": ap5095,
                          "gt_instances": int(_get(nt, cid, 0)) if nt is not None else None,
                          "gt_images": int(_get(ni, cid, 0)) if ni is not None else None})
    per_class.sort(key=lambda row: row["class_id"])
    summary = {
        "precision": float(box.mp), "recall": float(box.mr),
        "f1": statistics.mean(row["f1"] for row in per_class) if per_class else None,
        "map50": float(box.map50), "map75": float(box.map75),
        "map50_95": float(box.map),
        "f1_definition": "macro mean of per-class F1 at validator selected operating points",
        "precision_recall_note": "Ultralytics curve-selected P/R; not necessarily conf threshold used for generating PR curve",
    }
    for row in per_class:
        for key in ("precision", "recall", "f1", "ap50", "ap75", "ap50_95"):
            if row[key] is not None and not math.isfinite(row[key]):
                raise ValueError(f"Non-finite per-class metric {row['class_id']} {key}")
    for key in ("precision", "recall", "f1", "map50", "map75", "map50_95"):
        if summary[key] is None or not math.isfinite(summary[key]):
            raise ValueError(f"Non-finite overall metric {key}")
    missing = sorted(set(int(k) for k in names) - set(indices))
    if missing:
        raise ValueError(f"Validation did not calculate AP for class IDs {missing}; cannot PASS complete per-class baseline")
    return {"overall": summary, "per_class": per_class,
            "speed_ms_per_image": getattr(metrics, "speed", {}),
            "evaluated_classes": len(indices)}


def _write_csv(path: Path, metrics: dict):
    keys = ("class_id", "class_name", "precision", "recall", "f1", "ap50", "ap75", "ap50_95", "gt_instances", "gt_images")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerow({"class_id": "ALL", "class_name": "ALL",
                         "precision": metrics["overall"]["precision"], "recall": metrics["overall"]["recall"],
                         "f1": metrics["overall"]["f1"], "ap50": metrics["overall"]["map50"],
                         "ap75": metrics["overall"]["map75"], "ap50_95": metrics["overall"]["map50_95"]})
        for row in metrics["per_class"]:
            writer.writerow(row)


def run(args) -> int:
    out = resolved(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_log(out, "01", "profile_baseline")
    existing_gate = gate_file(out, "01", "profile_baseline")
    if existing_gate.exists() and not args.overwrite:
        log.error("Existing gate preserved: %s. Use --overwrite or choose a new --output.", existing_gate)
        return 2
    checks = {k: False for k in (
        "cli_valid", "stage_00_pass_current", "model_loaded", "fp32_profiler_valid",
        "parameter_count_consistent", "validation_complete", "per_class_metrics_complete", "artifacts_saved"
    )}
    errors, warnings, provenance = [], [], {}
    report = {"stage": "01_profile_baseline", "started_utc": now_utc(), "status": "FAIL"}
    try:
        if args.overwrite:
            for f in (out / "gates").glob("*.json"):
                if f.name[:2].isdigit() and int(f.name[:2]) >= 1:
                    f.unlink()
            log.warning("--overwrite invalidated 01..10 gates; re-run downstream stages")
        hw = make_imgsz(args.imgsz)
        if hw[0] != hw[1]:
            raise ValueError("Stage 01 currently requires square imgsz for exact Ultralytics val geometry; refusing silent H/W coalescing")
        if not (0 <= args.conf <= 1 and 0 <= args.iou <= 1):
            raise ValueError("--conf and --iou must be within [0,1]")
        if args.batch <= 0 or args.workers < 0 or args.max_det <= 0 or args.profile_repeats < 2 or args.warmup_iters < 0 or args.bench_iters < 0:
            raise ValueError("Invalid batch/workers/max-det/profile-repeats/warmup-iters/bench-iters")
        checks["cli_valid"] = True
        data_path = resolved(args.data)
        context = contract(args.model, data_path, hw, args.device, args.batch, args.workers, args.family, args.expected_classes)
        data = read_dataset(data_path, args.expected_classes)
        old_gate, ckpt = _check_00(out, context, data)
        checks["stage_00_pass_current"] = True
        provenance.update(old_gate["provenance"])
        provenance["stage00_gate_sha256"] = sha256_file(gate_file(out, "00", "check_environment"))
        provenance["validation_config"] = {"conf": args.conf, "iou": args.iou,
                                           "max_det": args.max_det, "nms_free": args.nms_free,
                                           "rect": False, "imgsz": hw[0], "split": "val"}
        report["input"] = {**context, **provenance["validation_config"]}
        from ultralytics import YOLO
        import torch
        device = select_device(args.device)
        family = old_gate["provenance"]["model_family"]
        if args.nms_free and family != "yolo26":
            raise ValueError("--nms-free is supported only for YOLO26 in this pipeline")
        if args.nms_free:
            warnings.append("NMS-free YOLO26 may cap outputs at 300; unsuitable for crowded frames >300 detections")
        model = YOLO(str(ckpt))
        net = model.model.to(device).float().eval()
        if getattr(model, "task", None) != "detect":
            raise ValueError("Model must be YOLO object detection")
        if len(getattr(model, "names", {})) != data["nc"]:
            raise ValueError("Loaded model class count changed since 00")
        checks["model_loaded"] = True
        x = torch.zeros((1, 3, *hw), device=device, dtype=torch.float32)
        param_count = sum(p.numel() for p in net.parameters())
        trainable_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
        buf_count = sum(b.numel() for b in net.buffers())
        if param_count <= 0:
            raise RuntimeError("Zero parameters in model")
        modules, histogram = _inventory(net)
        graph = {"module_inventory": modules, "module_histogram": histogram,
                 "note": "Module inventory, NOT a dependency graph. Stage 02 will build DepGraph."}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        runs = _macs(net, x, args.profile_repeats)
        macs = runs[0]["macs"]
        checks["fp32_profiler_valid"] = True
        tp_params = int(round(runs[0]["parameters"]))
        if abs(tp_params - param_count) > max(1, param_count * 0.001):
            raise RuntimeError(f"Torch-Pruning params {tp_params} != torch params {param_count}; profiler result untrusted")
        checks["parameter_count_consistent"] = True
        memory_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        raw_latency = _latency(net, x, device, args.warmup_iters, args.bench_iters)
        report["profile"] = {
            "parameter_count": param_count, "trainable_parameter_count": trainable_count,
            "buffer_elements": buf_count, "macs": macs, "gmacs": macs / 1e9,
            "flops_2_per_mac": 2 * macs, "gflops_2_per_mac": 2 * macs / 1e9,
            "flops_convention": "2 FLOPs per MAC; elementwise ops may not be covered by profiler",
            "macs_method": "torch_pruning.utils.count_ops_and_params",
            "profile_repeats": runs, "input_shape": [1, 3, *hw],
            "checkpoint_size_bytes": ckpt.stat().st_size,
            "cuda_peak_allocated_bytes_during_profiling": memory_peak,
            "cuda_peak_note": "Includes model weights/cached tensors; not a true peak activation-only measure",
            "raw_forward_latency": raw_latency,
            "module_count": len(modules), "graph_inventory_file": str(out / "stages" / "01" / "01_baseline_graph.json"),
        }
        write_json(out / "stages" / "01" / "01_baseline_graph.json", graph)
        log.info("Params: %.3f M | MACs: %.3f GMACs | nominal FLOPs: %.3f GFLOPs", param_count / 1e6, macs / 1e9, 2 * macs / 1e9)
        # Reload to isolate validation from any profiling forward hooks/traces.
        del model, net
        if device.type == "cuda":
            torch.cuda.empty_cache()
        val_model = YOLO(str(ckpt))
        val_kw = dict(data=str(data_path), split="val", imgsz=hw[0], batch=args.batch,
                      device=args.device, workers=args.workers, max_det=args.max_det,
                      conf=args.conf, iou=args.iou, rect=False,
                      plots=args.plots, project=str(out / "stages" / "01"),
                      name="validation", exist_ok=True, verbose=True, half=False)
        # Don't change nms mode for YOLO8/11; preserve native default behavior.
        if args.nms_free:
            val_kw["nms"] = False
        metrics = val_model.val(**val_kw)
        checks["validation_complete"] = True
        result = _metrics_report(metrics, data["names"])
        checks["per_class_metrics_complete"] = True
        report["validation"] = {"overall": result["overall"], "per_class_count": len(result["per_class"]),
                                "validation_config": provenance["validation_config"]}
        metric_path = out / "stages" / "01" / "01_baseline_val.json"
        write_json(metric_path, {**result, "validation_config": provenance["validation_config"],
                                 "model_family": family, "checkpoint_sha256": provenance["checkpoint_sha256"]})
        _write_csv(out / "stages" / "01" / "01_baseline_per_class.csv", result)
        checks["artifacts_saved"] = True
        log.info("Val mAP50-95=%.5f, mAP50=%.5f, P=%.5f, R=%.5f",
                 result["overall"]["map50_95"], result["overall"]["map50"],
                 result["overall"]["precision"], result["overall"]["recall"])
        for row in result["per_class"]:
            log.info("class %-4d %-18s P=%.4f R=%.4f F1=%.4f AP50=%.4f AP75=%.4f AP50-95=%.4f",
                     row["class_id"], row["class_name"], row["precision"], row["recall"],
                     row["f1"], row["ap50"], row["ap75"], row["ap50_95"])
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        errors.append(message)
        log.error("FAIL: %s", message)
        report["traceback"] = traceback.format_exc()
    finally:
        artifact_dir = out / "stages" / "01"
        gate = save_gate(out, "01", "profile_baseline", checks=checks, errors=errors,
                         warnings=warnings, provenance=provenance,
                         artifacts={"profile": str(artifact_dir / "01_profile.json"),
                                    "val_metrics": str(artifact_dir / "01_baseline_val.json"),
                                    "module_inventory": str(artifact_dir / "01_baseline_graph.json")})
        report.update({"status": gate["status"], "checks": checks, "errors": errors,
                       "warnings": warnings, "completed_utc": now_utc(),
                       "gate": str(gate_file(out, "01", "profile_baseline"))})
        write_json(artifact_dir / "01_profile.json", report)
        log.info("[01] %s — %s", gate["status"], gate_file(out, "01", "profile_baseline"))
    return 0 if gate["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(run(parse_args()))
