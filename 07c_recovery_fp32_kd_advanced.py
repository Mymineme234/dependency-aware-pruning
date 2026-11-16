#!/usr/bin/env python3
"""Advanced FP32 KD recovery: attention + class-logit + regression distillation."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import train as detect_train

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pruning_common import sha256_file, write_json, setup_log


def load(file, name):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s07 = load("07_recover_fp32.py", "kd_original_07")
s05 = load("05_apply_structured_pruning.py", "kd_original_05")
s06 = load("06_verify_pruned_model.py", "kd_original_06")


class AttentionKD:
    def __init__(self, teacher_pt, student, device, strength):
        self.teacher = YOLO(str(teacher_pt)).model.float().to(device).eval()
        self.teacher.requires_grad_(False)

        if dict(self.teacher.names) != dict(student.names):
            raise RuntimeError("Teacher/student class mapping differs")

        self.strength = strength
        self.steps = 0
        self.total = 0.0
        self.sf = None
        self.tf = None

        self.a = student.model[-1].register_forward_pre_hook(
            self.take_student
        )
        self.b = self.teacher.model[-1].register_forward_pre_hook(
            self.take_teacher
        )
        self.c = student.register_forward_hook(self.add_loss)

    @staticmethod
    def pyramid(inputs):
        if not inputs or not isinstance(inputs[0], (list, tuple)):
            raise RuntimeError("Invalid Detect feature prehook input")

        features = tuple(inputs[0])

        if len(features) != 3 or not all(
            isinstance(x, torch.Tensor) and x.ndim == 4
            for x in features
        ):
            raise RuntimeError("KD requires exactly three NCHW features")

        return features

    def take_student(self, module, inputs):
        self.sf = self.pyramid(inputs)

    def take_teacher(self, module, inputs):
        self.tf = tuple(x.detach() for x in self.pyramid(inputs))

    def add_loss(self, model, inputs, result):
        if not model.training or not torch.is_grad_enabled():
            return result

        if (
            len(inputs) != 1
            or not isinstance(inputs[0], dict)
            or "img" not in inputs[0]
        ):
            raise RuntimeError(
                "Expected YOLO training forward(batch_dict)"
            )

        if (
            not isinstance(result, (list, tuple))
            or len(result) != 2
            or self.sf is None
        ):
            raise RuntimeError(
                "Missing YOLO detection loss or student features"
            )

        images = inputs[0]["img"]
        self.tf = None

        with torch.no_grad():
            self.teacher(images)

        if self.tf is None:
            raise RuntimeError("Teacher forward hook not reached")

        terms = []

        for s, t in zip(self.sf, self.tf):
            if (
                s.shape[0] != t.shape[0]
                or s.shape[-2:] != t.shape[-2:]
            ):
                raise RuntimeError(
                    f"Feature resolution mismatch: {s.shape}, {t.shape}"
                )

            s = F.normalize(
                s.float().square().mean(1).flatten(1),
                p=2, dim=1, eps=1e-8
            )
            t = F.normalize(
                t.float().square().mean(1).flatten(1),
                p=2, dim=1, eps=1e-8
            )

            terms.append(
                (s - t).square().sum(1).mean()
            )

        kd = torch.stack(terms).mean()

        if not torch.isfinite(kd):
            raise RuntimeError("KD loss is not finite")

        self.steps += 1
        self.total += float(kd.detach())

        if self.steps % 250 == 0:
            print(
                f"[07c] step={self.steps} "
                f"KD_attention={float(kd.detach()):.6f}",
                flush=True
            )

        self.sf = None
        self.tf = None

        # YOLO detection loss is batch-scaled.
        total_loss = (
            result[0]
            + self.strength * images.shape[0] * kd
        )

        return total_loss, result[1]

    def close(self):
        for hook in (self.a, self.b, self.c):
            hook.remove()



class AdvancedKD(AttentionKD):
    """Add raw-head KD to verified attention KD without reconstructing student.

    YOLO26 raw branch is selected by its *actual* returned structure;
    unsupported ABI must fail, never silently ignore requested losses.
    """
    def __init__(self, teacher_pt, student, device, strength,
                 cls_weight, box_weight, confidence_threshold):
        super().__init__(teacher_pt, student, device, strength)
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.confidence_threshold = confidence_threshold
        self.raw_student = None
        self.raw_teacher = None
        self.cls_total = 0.0
        self.box_total = 0.0
        self.raw_steps = 0
        self.d = student.model[-1].register_forward_hook(self.capture_student)
        self.e = self.teacher.model[-1].register_forward_hook(self.capture_teacher)
        self.nc = len(self.teacher.names)

    def capture_student(self, module, inputs, result):
        if module.training and torch.is_grad_enabled():
            self.raw_student = result

    def capture_teacher(self, module, inputs, result):
        self.raw_teacher = result

    def raw_heads(self, obj):
        """Decode TRAIN raw head tensors, never post-NMS eval predictions."""
        if isinstance(obj, dict) and 'one2many' in obj:
            obj = obj['one2many']
        if isinstance(obj, dict):
            if 'boxes' in obj and 'scores' in obj:
                boxes, scores = obj['boxes'], obj['scores']
                if isinstance(boxes, torch.Tensor): boxes = [boxes]
                if isinstance(scores, torch.Tensor): scores = [scores]
                if len(boxes) != len(scores) or not boxes:
                    raise RuntimeError('KD boxes/scores pyramid mismatch')
                return list(zip(boxes, scores))
            raise RuntimeError(f'Unsupported YOLO raw-head dict: {list(obj)}')
        if isinstance(obj, (tuple, list)) and obj and all(
                isinstance(x, torch.Tensor) and x.ndim == 4 for x in obj):
            # Non-end-to-end Detect: [B,4*reg_max+nc,H,W] per pyramid level.
            return [(x[:,:-self.nc], x[:,-self.nc:]) for x in obj]
        raise RuntimeError(f'Unknown raw detection head ABI: {type(obj).__name__}')

    def add_loss(self, model, inputs, result):
        if not model.training or not torch.is_grad_enabled():
            return result
        if self.raw_student is None and (self.cls_weight or self.box_weight):
            raise RuntimeError('No student raw head before KD loss')
        # Temporarily activate TRAIN raw-output path only for the teacher Detect
        # top module; its BN children remain in eval and no teacher updates occur.
        head = self.teacher.model[-1]
        old_head_train = head.training
        head.training = True
        try:
            attention_result = super().add_loss(model, inputs, result)
        finally:
            head.training = old_head_train
        if not (self.cls_weight or self.box_weight):
            return attention_result
        student_heads = self.raw_heads(self.raw_student)
        teacher_heads = self.raw_heads(self.raw_teacher)
        if len(student_heads) != len(teacher_heads):
            raise RuntimeError('KD pyramid stage count mismatch')
        cls_terms, box_terms = [], []
        for (sr, sc),(tr, tc) in zip(student_heads,teacher_heads):
            if not all(isinstance(x,torch.Tensor) for x in (sr,sc,tr,tc)):
                raise RuntimeError('Non-Tensor raw head for KD')
            if sr.shape != tr.shape or sc.shape != tc.shape or sc.ndim not in (3,4) or sc.shape[1] != self.nc:
                raise RuntimeError(f'Teacher/student raw head shape mismatch: {sr.shape} vs {tr.shape}, {sc.shape} vs {tc.shape}')
            with torch.no_grad():
                teacher_probability = tc.float().sigmoid()
                # Downweight easy background, avoid zero-positive batches.
                weight = teacher_probability.max(dim=1,keepdim=True).values
                weight = torch.where(weight >= self.confidence_threshold,
                                     weight, weight * 0.1).clamp_min(1e-4)
            if self.cls_weight:
                # Bernoulli KL with teacher soft targets, gradient flows to student logits.
                soft_target = teacher_probability.clamp(1e-6,1-1e-6)
                bce = F.binary_cross_entropy_with_logits(sc.float(), soft_target.expand_as(sc), reduction='none')
                entropy = F.binary_cross_entropy(soft_target,soft_target,reduction='none')
                cls_terms.append(((bce-entropy)*weight).sum() / (weight.sum()*sc.shape[1]+1e-6))
            if self.box_weight:
                reg = F.smooth_l1_loss(sr.float(),tr.detach().float(),reduction='none')
                box_terms.append((reg * weight).sum() / (weight.sum()*sr.shape[1]+1e-6))
        if self.cls_weight and not cls_terms or self.box_weight and not box_terms:
            raise RuntimeError('Requested raw-head KD produced no terms')
        cls = torch.stack(cls_terms).mean() if cls_terms else attention_result[0].new_zeros(())
        box = torch.stack(box_terms).mean() if box_terms else attention_result[0].new_zeros(())
        if not torch.isfinite(cls+box):
            raise RuntimeError('Non-finite classification/localization KD loss')
        bs = int(inputs[0]['img'].shape[0])
        self.cls_total += float(cls.detach())
        self.box_total += float(box.detach())
        self.raw_steps += 1
        self.raw_student = None
        self.raw_teacher = None
        return attention_result[0] + bs*(self.cls_weight*cls+self.box_weight*box), attention_result[1]

    def close(self):
        super().close()
        self.d.remove()
        self.e.remove()


def main():
    p = argparse.ArgumentParser()

    for flag in (
        "parent-round", "student", "teacher", "data", "output"
    ):
        p.add_argument("--" + flag, required=True, type=Path)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--kd-weight", type=float, default=0.2)
    p.add_argument("--cls-weight", type=float, default=0.05)
    p.add_argument("--box-weight", type=float, default=0.05)
    p.add_argument("--teacher-score-threshold", type=float, default=0.2)
    p.add_argument("--lr0", type=float, default=0.0001)
    p.add_argument("--lrf", type=float, default=0.1)
    p.add_argument("--device", default="0")

    args = p.parse_args()

    for name in ("parent_round", "student", "teacher", "data"):
        path = getattr(args, name).resolve()

        if not path.exists():
            raise FileNotFoundError(path)

        setattr(args, name, path)

    args.output = args.output.resolve()

    if args.output.exists():
        raise RuntimeError(
            f"Output exists; refusing overwrite: {args.output}"
        )

    if args.cls_weight < 0 or args.box_weight < 0 or not 0 <= args.teacher_score_threshold <= 1:
        raise ValueError('Invalid classification/localization KD parameters')
    if args.kd_weight <= 0 or args.epochs < 1 or args.batch < 1:
        raise ValueError("Invalid KD hyperparameters")

    # Require parent pruning and recovery to be certified.
    for name in (
        "05_apply_structured_pruning",
        "06_verify_pruned_model",
    ):
        gate = json.loads(
            (
                args.parent_round / "gates" / f"{name}.json"
            ).read_text()
        )

        if gate["status"] != "PASS":
            raise RuntimeError(f"Parent gate is not PASS: {name}")

    # Accept either Stage05 raw or Stage07 recovered best.
    raw = (
        args.parent_round / "stages/05/05_pruned_raw.pt"
    ).resolve()

    parent = None

    if args.student == raw:
        report05 = json.loads(
            (
                args.parent_round / "stages/05/05_report.json"
            ).read_text()
        )
        if sha256_file(raw) != report05["output_pruned_raw_sha256"]:
            raise RuntimeError("Stage05 raw checkpoint hash mismatch")
    else:
        gate = json.loads(
            (
                args.parent_round / "gates/07_recover_fp32.json"
            ).read_text()
        )
        if gate["status"] != "PASS":
            raise RuntimeError("Stage07 is not PASS")

        parent = json.loads(
            (
                args.parent_round / "stages/07/07_report.json"
            ).read_text()
        )
        if args.student != Path(parent["best"]).resolve():
            raise RuntimeError(
                "Student must be Stage05 raw or Stage07 best"
            )

    graph = json.loads(
        (
            args.parent_round / "stages/05/05_graph_signature.json"
        ).read_text()
    )
    pruned = json.loads(
        (
            args.parent_round / "stages/05/05_report.json"
        ).read_text()
    )

    student = YOLO(str(args.student)).model.float()
    teacher = YOLO(str(args.teacher)).model.float()
    expected = s07.model_structure(student, s05)

    if dict(student.names) != dict(teacher.names):
        raise RuntimeError("Teacher/student class mismatch")

    if (
        expected["params"] != pruned["pruned_params"]
        or expected["conv_shapes"] != graph["pruned_convs"]
        or expected["bn_shapes"] != graph["pruned_batchnorm"]
    ):
        raise RuntimeError(
            "Student physical topology not identical to Stage05"
        )

    s07.assert_finite_fp32(student)
    del student, teacher

    student_sha = sha256_file(args.student)
    teacher_sha = sha256_file(args.teacher)

    args.output.mkdir(parents=True)

    logger = setup_log(
        args.output, "07c", "recovery_fp32_kd"
    )

    # Reuse original Stage07 training configuration.
    cfg = s07.parser().parse_args([
        "--model", str(args.student),
        "--data", str(args.data),
        "--output", str(args.output),
        "--expected-classes",
        str(len(YOLO(str(args.student)).names)),
        "--imgsz", "640",
        "--device", args.device,
        "--batch", "8",
        "--workers", "8",
        "--recovery-epochs", str(args.epochs),
        "--recovery-batch", str(args.batch),
        "--recovery-workers", str(args.workers),
        "--optimizer", "AdamW",
        "--lr0", str(args.lr0),
        "--lrf", str(args.lrf),
        "--weight-decay", "0.0005",
        "--patience", str(args.epochs),
        "--warmup-epochs", "1",
        "--seed", "42",
        "--no-amp",
        "--mosaic", "0.3",
        "--scale", "0.25",
        "--translate", "0.05",
        "--fliplr", "0.5",
        "--val-subset", "full",
    ])

    s07.check_args(
        cfg, s07.make_imgsz(cfg.imgsz)
    )

    old_kwargs = s07.trainer_kwargs

    def kwargs(*a, **k):
        result = old_kwargs(*a, **k)
        result.update(
            nbs=32,
            close_mosaic=5
        )
        return result

    s07.trainer_kwargs = kwargs

    # Preserve Stage07's get_model() reconstruction prohibition.
    # KD hooks are installed AFTER Ultralytics creates EMA.
    original = detect_train.DetectionTrainer
    active = []

    class TrainerWithKD(original):
        def _setup_train(self):
            super()._setup_train()

            if self.ema is None:
                raise RuntimeError(
                    "EMA must be created before KD hooks"
                )

            self.kd_controller = AdvancedKD(
                args.teacher,
                self.model,
                self.device,
                args.kd_weight,
                args.cls_weight,
                args.box_weight,
                args.teacher_score_threshold,
            )
            active.append(self.kd_controller)

    detect_train.DetectionTrainer = TrainerWithKD

    try:
        val_settings = s07.get_val_settings(
            args.parent_round, {}
        )

        result = s07.run_training(
            cfg,
            args.student,
            args.data,
            args.output,
            expected,
            s05,
            val_settings,
            logger,
        )
    finally:
        detect_train.DetectionTrainer = original

        for controller in active:
            controller.close()

    if not active or active[0].steps < 1:
        raise RuntimeError("KD never executed")

    if (
        sha256_file(args.student) != student_sha
        or sha256_file(args.teacher) != teacher_sha
    ):
        raise RuntimeError(
            "Teacher/student checkpoint unexpectedly changed"
        )

    # Both best and last must reload with exact Stage05 topology.
    for kind in ("best", "last"):
        checkpoint = result[kind]
        net = YOLO(str(checkpoint)).model.float()

        s07.assert_unchanged_structure(
            s07.model_structure(net, s05),
            expected,
            kind,
        )
        s07.assert_finite_fp32(net)

        receipt = s06.subprocess_reload(
            HERE / "06_verify_pruned_model.py",
            checkpoint,
            sha256_file(checkpoint),
            (640, 640),
            "cpu",
            300,
            args.output / "stages/07c" / f"{kind}_reload",
        )

        if (
            receipt["parameter_count"] != expected["params"]
            or receipt["conv_shapes"] != expected["conv_shapes"]
            or receipt["bn_shapes"] != expected["bn_shapes"]
        ):
            raise RuntimeError(
                f"{kind}: fresh-process topology mismatch"
            )

    baseline = (
        parent["epoch_metrics"]["best_val_map50_95_over_epochs"]
        if parent is not None else None
    )
    kdmap = result["metrics"][
        "best_val_map50_95_over_epochs"
    ]

    summary = {
        "status": "PASS",
        "kd_type": "attention_plus_raw_classification_localization",
        "cls_weight":args.cls_weight,
        "box_weight":args.box_weight,
        "teacher_score_threshold":args.teacher_score_threshold,
        "raw_kd_steps":active[0].raw_steps,
        "mean_cls_kd":active[0].cls_total/max(active[0].raw_steps,1),
        "mean_box_kd":active[0].box_total/max(active[0].raw_steps,1),
        "kd_weight": args.kd_weight,
        "kd_steps": active[0].steps,
        "kd_loss_avg": active[0].total / active[0].steps,
        "input": str(args.student),
        "input_sha256": student_sha,
        "teacher": str(args.teacher),
        "teacher_sha256": teacher_sha,
        "params": expected["params"],
        "parent_map": baseline,
        "kd_map": kdmap,
        "delta_map": None if baseline is None else kdmap - baseline,
        "best": str(result["best"]),
        "last": str(result["last"]),
        "csv": str(result["training_csv"]),
        "topology_unchanged": True,
        "note": (
            "KD experiment only, not Stage08 accuracy gate. "
            "No additional inference parameters."
        ),
    }

    report = (
        args.output / "stages/07c/07c_report.json"
    )
    write_json(report, summary)

    write_json(
        args.output / "gates/07c_recovery_fp32_kd_advanced.json",
        {
            "status": "PASS",
            "step": "07c_recovery_fp32_kd_advanced",
            "report": str(report),
        },
    )

    if baseline is None:
        print(
            f"07c PASS | KD mAP={kdmap:.5f} "
            f"| params={expected['params']}"
        )
    else:
        print(
            f"07c PASS | mAP {baseline:.5f} -> {kdmap:.5f} "
            f"| params={expected['params']}"
        )
    print("Checkpoint:", result["best"])


if __name__ == "__main__":
    main()
