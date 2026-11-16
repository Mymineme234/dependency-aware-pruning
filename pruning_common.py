"""Small shared IO/contract helpers for the numbered YOLO pruning pipeline.

Not a pruning backend. Standard library plus PyYAML; no Ultralytics imports here.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any

import yaml


SUPPORTED_FAMILIES = ("yolov8", "yolo11", "yolo26")
SUPPORTED_SCALES = ("n", "s", "m", "l")
STAGES = ("logs", "gates", "stages")


def make_imgsz(values: list[int]) -> tuple[int, int]:
    if len(values) not in (1, 2) or any(x <= 0 for x in values):
        raise ValueError("--imgsz must be 640 or HEIGHT WIDTH, both positive")
    return (values[0], values[0]) if len(values) == 1 else (values[0], values[1])


def resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(2 ** 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_object(data: Any) -> str:
    raw = json.dumps(jsonable(data), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list, set)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return jsonable(obj.item())
        except (TypeError, ValueError):
            pass
    if hasattr(obj, "tolist"):
        return jsonable(obj.tolist())
    return str(obj)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(jsonable(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def setup_log(out: Path, number: str, name: str) -> logging.Logger:
    for sub in STAGES:
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / "stages" / number).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"pruning-step-{number}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for old in list(logger.handlers):
        old.close()
        logger.removeHandler(old)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(out / "logs" / f"{number}_{name}.log", encoding="utf-8")]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def gate_file(out: Path, number: str, name: str) -> Path:
    return out / "gates" / f"{number}_{name}.json"


def save_gate(out: Path, number: str, name: str, *, checks: dict[str, bool], errors: list[str], warnings: list[str], provenance: dict, artifacts: dict | None = None) -> dict:
    status = "PASS" if not errors and checks and all(v is True for v in checks.values()) else "FAIL"
    data = {
        "step": f"{number}_{name}", "status": status, "timestamp_utc": now_utc(),
        "checks": checks, "errors": errors, "warnings": warnings,
        "provenance": provenance, "artifacts": artifacts or {},
    }
    write_json(gate_file(out, number, name), data)
    return data


def normalize_names(names: object) -> dict[int, str]:
    if isinstance(names, (tuple, list)):
        result = {i: str(n) for i, n in enumerate(names)}
    elif isinstance(names, dict):
        result = {int(k): str(v) for k, v in names.items()}
    else:
        raise ValueError("dataset names must be a YAML list or mapping")
    if not result or sorted(result) != list(range(len(result))):
        raise ValueError(f"Class ids must be consecutive 0..nc-1, got {sorted(result)}")
    if not all(value.strip() for value in result.values()):
        raise ValueError("Empty class name in names")
    if len(set(result.values())) != len(result):
        raise ValueError("Duplicate class names in dataset")
    return result


def _split_locations(value: Any, root: Path) -> list[Path]:
    vals = value if isinstance(value, list) else [value]
    paths = []
    for raw in vals:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"Invalid split entry: {raw!r}")
        p = Path(raw)
        paths.append((root / p).resolve() if not p.is_absolute() else p.resolve())
    return paths


def read_dataset(path: Path, expected_classes: int | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("Dataset YAML must contain a mapping")
    names = normalize_names(doc.get("names"))
    nc = int(doc.get("nc", len(names)))
    if nc != len(names):
        raise ValueError(f"data.yaml nc={nc} != len(names)={len(names)}")
    if expected_classes is not None and nc != expected_classes:
        raise ValueError(f"expected-classes={expected_classes} but YAML nc={nc}")
    # An absent 'path' means paths are relative to the YAML directory.
    root_hint = doc.get("path")
    if root_hint is None or not str(root_hint).strip():
        root = path.parent.resolve()
    else:
        hp = Path(str(root_hint)).expanduser()
        root = hp.resolve() if hp.is_absolute() else (path.parent / hp).resolve()
        if not root.exists() and not hp.is_absolute():
            # Some YOLO configs give project-root-relative path instead.
            cwd_root = hp.resolve()
            if cwd_root.exists():
                root = cwd_root
    splits: dict[str, list[Path]] = {}
    for name in ("train", "val", "test"):
        if name not in doc or not doc[name]:
            if name != "test":
                raise ValueError(f"Mandatory split '{name}' is absent from YAML")
            continue
        locations = _split_locations(doc[name], root)
        for p in locations:
            if not p.exists():
                raise FileNotFoundError(f"Missing {name} path: {p}")
            if not (p.is_dir() or (p.is_file() and p.suffix.lower() == ".txt")):
                raise ValueError(f"Unsupported {name} path: {p}")
        splits[name] = locations
    return {"yaml": str(path), "yaml_sha256": sha256_file(path), "root": str(root),
            "names": {str(k): v for k, v in names.items()}, "nc": nc,
            "splits": {k: [str(p) for p in ps] for k, ps in splits.items()}}


def dataset_fingerprint(dataset: dict) -> dict:
    """Stat inventory of dataset files, excluding regenerated Ultralytics .cache.

    Images and YOLO labels are included even when split dirs point only to images.
    This detects filename, size and mtime changes, not content edits preserving
    size and mtime. Temporary *.cache files are not dataset ground truth.
    """
    root = Path(dataset["root"])
    # Anchor inventories in the YAML root and split directories.
    dirs = []
    for split_paths in dataset["splits"].values():
        for raw in split_paths:
            p = Path(raw)
            if p.is_dir():
                dirs.append(p)
                if p.name.lower() == "images":
                    sibling = p.parent / "labels"
                    if sibling.is_dir():
                        dirs.append(sibling)
                if p.parent.name.lower() == "images":
                    sibling = p.parent.parent / "labels" / p.name
                    if sibling.is_dir():
                        dirs.append(sibling)
    # Include common layout labels/split when split is under root/images/split.
    if (root / "labels").is_dir():
        dirs.append(root / "labels")
    dirs = sorted(set(dirs), key=str)
    file_list = []
    for p in dirs:
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix.lower() != ".cache":
                st = f.stat()
                file_list.append((str(f), st.st_size, st.st_mtime_ns))
    for split_paths in dataset["splits"].values():
        for raw in split_paths:
            p = Path(raw)
            if p.is_file():
                st = p.stat()
                file_list.append((str(p), st.st_size, st.st_mtime_ns, sha256_file(p)))
    file_list.sort()
    return {"mode": "stat_filename_size_mtime_ns", "num_files": len(file_list),
            "manifest_sha256": hash_object(file_list),
            "note": "Excludes Ultralytics .cache files; does not detect byte changes preserving size and mtime."}


def family_from_metadata(model: Any, model_arg: str, override: str = "auto") -> tuple[str, str, list[str]]:
    warnings: list[str] = []
    net = model.model
    meta = getattr(net, "yaml", {})
    meta = meta if isinstance(meta, dict) else {}
    # Priority: architecture metadata, then checkpoint/model name. Avoid guessed family from modules.
    fields = [str(meta.get(k, "")) for k in ("yaml_file", "model", "name")]
    fields += [str(getattr(net, "yaml_file", "")), str(getattr(model, "ckpt_path", "")), str(model_arg)]
    patterns = {
        "yolov8": re.compile(r"yolov8(?:[nsmlx])?", re.I),
        "yolo11": re.compile(r"yolo11(?:[nsmlx])?", re.I),
        "yolo26": re.compile(r"yolo26(?:[nsmlx])?", re.I),
    }
    detected = next((f for field in fields for f, pat in patterns.items() if pat.search(field)), "unknown")
    if override != "auto":
        if detected not in ("unknown", override):
            raise ValueError(f"Family override {override} conflicts with architecture metadata {detected}")
        if detected == "unknown":
            warnings.append("Family supplied by --family override; not independently identified from checkpoint metadata")
        detected = override
    scale = str(meta.get("scale", ""))
    if scale not in SUPPORTED_SCALES:
        scale = "unknown"
        for field in fields:
            m = re.search(r"yolo(?:v8|11|26)([nsml])(?=\W|$)", field, re.I)
            if m:
                scale = m.group(1)
                break
    if scale == "unknown":
        warnings.append("Scale n/s/m/l was not found in model metadata; profile will continue")
    return detected, scale, warnings


def checkpoint_path(model: Any, model_arg: str) -> Path:
    choices = [model_arg, getattr(model, "ckpt_path", "")]
    for raw in choices:
        if raw and Path(str(raw)).is_file():
            return resolved(raw)
    raise FileNotFoundError("Cannot locate downloaded/local .pt file to hash. Use a local --model checkpoint path.")


def contract(model_arg: str, data: Path, imgsz: tuple[int, int], device: str, batch: int, workers: int, family: str, expected_classes: int | None) -> dict:
    return {"model_arg": str(model_arg), "data": str(data), "imgsz_hw": list(imgsz),
            "device": str(device), "batch": int(batch), "workers": int(workers),
            "family_override": family, "expected_classes": expected_classes}


def versions() -> dict:
    import importlib.metadata as md
    packages = ("torch", "torchvision", "ultralytics", "torch-pruning", "numpy", "PyYAML", "Pillow", "thop")
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in packages:
        try:
            result[name] = md.version(name)
        except md.PackageNotFoundError:
            result[name] = None
    return result


def select_device(device: str):
    import torch
    value = str(device).strip().lower()
    if value == "cpu":
        return torch.device("cpu")
    if value.startswith("cuda:"):
        idx = int(value.split(":", 1)[1])
    elif value.isdecimal():
        idx = int(value)
    else:
        raise ValueError(f"Unsupported --device {device!r}; use cpu, 0, 1, or cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({device}), but torch.cuda.is_available() is False")
    if idx >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {idx} missing; count={torch.cuda.device_count()}")
    return torch.device(f"cuda:{idx}")


def forward_tensors(obj: Any, label: str = "root", max_depth: int = 15) -> list[dict]:
    import torch
    tensors = []
    def walk(x, path, depth):
        if depth > max_depth:
            raise ValueError("Model return nesting too deep")
        if torch.is_tensor(x):
            tensors.append({"path": path, "shape": list(x.shape), "dtype": str(x.dtype),
                            "finite": bool(torch.isfinite(x).all().item()) if x.numel() else False,
                            "numel": x.numel()})
        elif isinstance(x, (list, tuple)):
            for i, sub in enumerate(x):
                walk(sub, f"{path}[{i}]", depth + 1)
        elif isinstance(x, dict):
            for key, sub in x.items():
                walk(sub, f"{path}.{key}", depth + 1)
    walk(obj, label, 0)
    if not tensors:
        raise ValueError(f"Forward returned no torch.Tensor: {type(obj).__name__}")
    return tensors


def model_names(model: Any) -> dict[int, str]:
    raw = getattr(model, "names", None)
    if raw is None:
        raw = getattr(model.model, "names", None)
    return normalize_names(raw)


def model_head_metadata(net: Any) -> dict:
    modules = getattr(net, "model", None)
    head = modules[-1] if modules is not None and len(modules) else None
    return {"head_class": type(head).__name__ if head is not None else None,
            "head_end2end": bool(getattr(head, "end2end", False)),
            "one2one_branch_present": bool(hasattr(head, "one2one_cv2") or hasattr(head, "one2one_cv3")),
            "head_nc": int(head.nc) if head is not None and getattr(head, "nc", None) is not None else None,
            "head_reg_max": int(head.reg_max) if head is not None and getattr(head, "reg_max", None) is not None else None}
