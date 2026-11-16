#!/usr/bin/env python3
"""Stage 00: validate YOLO checkpoint/dataset/environment and perform real FP32 forward.

No pruning or checkpoint mutation. A valid PASS is required by stage 01.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from pruning_common import (
    SUPPORTED_FAMILIES, checkpoint_path, contract, dataset_fingerprint,
    family_from_metadata, forward_tensors, gate_file, hash_object, make_imgsz,
    model_head_metadata, model_names, now_utc, read_dataset, resolved, save_gate,
    select_device, setup_log, sha256_file, versions, write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 00: strict environment / FP32 model gate")
    p.add_argument("--model", required=True, help="Local best.pt, or a downloadable Ultralytics .pt")
    p.add_argument("--data", required=True, help="YOLO detection data.yaml")
    p.add_argument("--output", required=True, help="Pruning run directory")
    p.add_argument("--imgsz", type=int, nargs="+", default=[640], metavar="SIZE")
    p.add_argument("--device", default="0", help="cpu, 0, cuda:0, ...")
    p.add_argument("--batch", type=int, default=8, help="Reference batch (stage 01 validation)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--expected-classes", type=int, default=None)
    p.add_argument("--family", choices=["auto", *SUPPORTED_FAMILIES], default="auto")
    p.add_argument("--overwrite", action="store_true", help="Re-run 00 and invalidate ALL downstream gates")
    return p.parse_args()


def run(args) -> int:
    out = resolved(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_log(out, "00", "check_environment")
    existing_gate = gate_file(out, "00", "check_environment")
    if existing_gate.exists() and not args.overwrite:
        log.error("Existing gate preserved: %s. Use --overwrite or choose a new --output.", existing_gate)
        return 2
    checks = {k: False for k in (
        "cli_valid", "dependencies_present", "dataset_valid", "checkpoint_loaded",
        "family_supported", "task_detection", "class_contract", "forward_fp32_finite", "fingerprints_complete"
    )}
    errors, warnings = [], []
    report = {"stage": "00_check_environment", "started_utc": now_utc(), "status": "FAIL"}
    provenance = {}
    try:
        if args.overwrite:
            for f in (out / "gates").glob("*.json"):
                if f.name.startswith(("00_", "01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_", "09_", "10_")):
                    f.unlink()
            log.warning("--overwrite removed existing step gates 00..10. Re-run later stages.")
        hw = make_imgsz(args.imgsz)
        if args.batch <= 0 or args.workers < 0 or (args.expected_classes is not None and args.expected_classes < 1):
            raise ValueError("batch > 0, workers >= 0, expected-classes >= 1 required")
        checks["cli_valid"] = True
        data = resolved(args.data)
        context = contract(args.model, data, hw, args.device, args.batch, args.workers, args.family, args.expected_classes)
        provenance["contract"] = context
        provenance["contract_sha256"] = hash_object(context)
        report["input"] = context
        report["versions"] = versions()
        try:
            import torch
            import ultralytics
            import torch_pruning as tp
            if not callable(getattr(getattr(tp, "utils", None), "count_ops_and_params", None)):
                raise RuntimeError("torch_pruning.utils.count_ops_and_params is not available")
            device = select_device(args.device)
            checks["dependencies_present"] = True
            report["torch"] = {
                "cuda_available": torch.cuda.is_available(),
                "cuda_count": torch.cuda.device_count(), "requested_device": str(device),
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            }
        except Exception as exc:
            raise RuntimeError(f"Dependency/device check failed: {exc}") from exc
        dataset = read_dataset(data, args.expected_classes)
        report["dataset"] = dataset
        checks["dataset_valid"] = True
        log.info("Loaded YAML: %s; %d classes; splits: %s", data, dataset["nc"], ", ".join(dataset["splits"]))
        from ultralytics import YOLO
        model = YOLO(args.model)
        ckpt = checkpoint_path(model, args.model)
        checks["checkpoint_loaded"] = True
        family, scale, family_warnings = family_from_metadata(model, args.model, args.family)
        warnings.extend(family_warnings)
        report["identity"] = {"family": family, "scale": scale, "source_checkpoint": str(ckpt)}
        if family not in SUPPORTED_FAMILIES:
            raise RuntimeError("Cannot determine YOLOv8 / YOLO11 / YOLO26 from model metadata. Supply --family explicitly after verifying architecture.")
        checks["family_supported"] = True
        task = getattr(model, "task", None)
        if task != "detect":
            raise RuntimeError(f"Only YOLO object detection is supported; model.task={task!r}")
        checks["task_detection"] = True
        names = model_names(model)
        if len(names) != dataset["nc"]:
            raise ValueError(f"Model nc={len(names)} does not match dataset nc={dataset['nc']}")
        expected_names = {int(k): v for k, v in dataset["names"].items()}
        if names != expected_names:
            raise ValueError(f"Model class names {names} differ from dataset names {expected_names}. Refuse ambiguous class ID mapping.")
        head = model_head_metadata(model.model)
        if head["head_nc"] is not None and head["head_nc"] != dataset["nc"]:
            raise ValueError(f"Detection head nc={head['head_nc']} != dataset nc={dataset['nc']}")
        checks["class_contract"] = True
        if family == "yolo26" and not head["head_end2end"]:
            warnings.append("YOLO26 checkpoint does not expose head.end2end=True; verify it has the expected detection branches")
        report["identity"].update({"task": task, "names": dataset["names"], "head": head})
        # Force FP32 eval without altering original checkpoint on disk.
        model.model = model.model.to(device).float().eval()
        x = torch.zeros((1, 3, hw[0], hw[1]), device=device, dtype=torch.float32)
        with torch.inference_mode():
            y = model.model(x)
        tensors = forward_tensors(y)
        if not all(t["finite"] for t in tensors):
            raise RuntimeError("Forward produced an empty/non-finite tensor")
        report["fp32_forward"] = {"input_shape": list(x.shape), "outputs": tensors,
                                   "parameter_count": sum(p.numel() for p in model.model.parameters())}
        checks["forward_fp32_finite"] = True
        log.info("FP32 forward PASS: input=%s, %d output tensors", list(x.shape), len(tensors))
        manifest = dataset_fingerprint(dataset)
        provenance.update({"source_checkpoint": str(ckpt), "checkpoint_sha256": sha256_file(ckpt),
                           "dataset_yaml_sha256": dataset["yaml_sha256"], "dataset_manifest": manifest,
                           "model_family": family, "model_scale": scale,
                           "dependency_versions": report["versions"]})
        checks["fingerprints_complete"] = True
        report["dataset_inventory"] = manifest
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        errors.append(message)
        log.error("FAIL: %s", message)
        log.debug(traceback.format_exc())
        report["traceback"] = traceback.format_exc()
    finally:
        gate = save_gate(out, "00", "check_environment", checks=checks, errors=errors,
                         warnings=warnings, provenance=provenance,
                         artifacts={"report": str(out / "stages" / "00" / "00_environment.json")})
        report.update({"status": gate["status"], "checks": checks, "errors": errors,
                       "warnings": warnings, "completed_utc": now_utc(), "gate": str(gate_file(out, "00", "check_environment"))})
        write_json(out / "stages" / "00" / "00_environment.json", report)
        log.info("[00] %s — %s", gate["status"], gate_file(out, "00", "check_environment"))
    return 0 if gate["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(run(parse_args()))
