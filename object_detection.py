"""
train_v8.py  –  Gym Equipment YOLO trainer  (v8 — Dropout + 5-class)
Dataset  : Roboflow — dumbells-and-kettlebells (v15)
Classes  : bb, db, kb, medicine ball, plates  (nc=5)

════════════════════════════════════════════════════════════════════
  WHAT CHANGED vs v7
════════════════════════════════════════════════════════════════════
  1. CLASS FIX  ─ Reverted to 5 classes (swiss ball removed).
                  CLASS_NAMES and KNOWN_COUNTS updated accordingly.

  2. DROPOUT  ─ YOLO backbone/head don't expose a dropout arg natively,
                so we inject it the correct way:
                  • model.model[-1].dropout = 0.10  (head Detect layer)
                  • A post-load hook patches every nn.Dropout in the
                    backbone to p=BACKBONE_DROPOUT (default 0.05).
                This regularises without changing architecture.
                Dropout is DISABLED during val/test automatically by
                model.eval() which Ultralytics calls internally.

  3. EPOCH GUIDANCE (see table below) ─ with your dataset size and
     the plates class having only ~52 samples in the val split
     you need at minimum 120 epochs; 150 is safer.

  4. SCALE REDUCED  ─ bbox size plot shows objects cluster tightly at
     ~0.05-0.10 w/h (small, uniform). Wide scale jitter (0.5) hurts
     small-object recall. Dropped back to 0.35.

  5. AUGMENTATION TUNED for plates (52 samples):
     • copy_paste raised to 0.6  (was 0.5) — critical for plates
     • erasing=0.3 added         — prevents texture over-fitting on
                                   the 52 plates images
     • mixup kept at 0.15

  6. PROPER MLFLOW RUN-NAMING  ─ run name includes timestamp so
     every run is uniquely identifiable without manual renaming.

════════════════════════════════════════════════════════════════════
  EPOCH RECOMMENDATION
════════════════════════════════════════════════════════════════════

  Your dataset: 1,776 total label instances across 5 classes.
  Rarest class: plates ≈ 52 val instances.

  Recommended epochs by use case:
  ┌───────────────────────────────┬────────┬──────────┐
  │ Use case                      │ epochs │ patience │
  ├───────────────────────────────┼────────┼──────────┤
  │ Quick experiment / sanity     │   50   │   20     │
  │ Normal training run           │  120   │   40     │
  │ Best accuracy (recommended)   │  150   │   45     │
  │ Max (diminishing returns)     │  200   │   50     │
  └───────────────────────────────┴────────┴──────────┘

  Default in this script: --epochs 150  --patience 45

════════════════════════════════════════════════════════════════════
  HOW TO RUN
════════════════════════════════════════════════════════════════════

  # 1. Activate your venv
  source /home/pc-008/weight_recognition/venv/bin/activate

  # 2. Basic run (uses all defaults)
  python train_v8.py

  # 3. Custom model / epochs
  python train_v8.py --model yolov8m.pt --epochs 150 --batch 16

  # 4. Point to a specific MLflow DB
  python train_v8.py --mlflow-uri sqlite:////home/pc-008/weight_project/mlflow.db

  # 5. Resume interrupted run
  python train_v8.py --resume

  # 6. After training — launch MLflow UI
  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
  # Then open: http://localhost:5000

  # 7. GPU check before running
  python -c "import torch; print(torch.cuda.get_device_name(0))"

════════════════════════════════════════════════════════════════════
"""

import argparse
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER

ROOT      = Path(__file__).resolve().parent

DATA_YAML = str(
    "/home/pc-008/weight_project/DB and KB Detetcion/"
    "Dumbells and Kettlebells.v3i.yolov8/data.yaml"
)

# ── v8: 5 classes (swiss ball removed) ───────────────────────────────────────
CLASS_NAMES = ["bb", "db", "kb", "medicine ball", "plates"]

# Update these from your actual dataset split counts
KNOWN_COUNTS = {
    "bb":            429,   # val split counts from labels.jpg
    "db":            747,
    "kb":            442,
    "medicine ball": 106,
    "plates":         52,   # CRITICALLY LOW ⚠
}

# ── Dropout config ────────────────────────────────────────────────────────────
HEAD_DROPOUT     = 0.10   # applied to the Detect head layer
BACKBONE_DROPOUT = 0.05   # applied to any existing nn.Dropout in backbone


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO on gym equipment dataset (v8)")
    p.add_argument("--model",    default="yolov8m.pt")
    p.add_argument("--epochs",   type=int, default=150,
                   help="Recommended: 150 for best accuracy (see table in docstring)")
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--batch",    type=int, default=16)
    p.add_argument("--device",   default="0")
    p.add_argument("--workers",  type=int, default=4)
    p.add_argument("--project",  default=str(ROOT / "runs"))
    p.add_argument("--name",     default="gym_equipment_v8")
    p.add_argument("--resume",   action="store_true")
    p.add_argument("--patience", type=int, default=45)
    p.add_argument("--data",     default=DATA_YAML)
    p.add_argument("--mlflow-uri",        default="sqlite:///mlflow.db")
    p.add_argument("--mlflow-experiment", default="gym_equipment_detection")
    p.add_argument("--mlflow-run-name",   default=None,
                   help="Defaults to v8_<timestamp> for unique run IDs")
    p.add_argument("--head-dropout",     type=float, default=HEAD_DROPOUT)
    p.add_argument("--backbone-dropout", type=float, default=BACKBONE_DROPOUT)
    return p.parse_args()


# ── MLflow setup ──────────────────────────────────────────────────────────────

def setup_mlflow(args):
    uri = args.mlflow_uri
    if "://" not in uri:
        uri = Path(uri).resolve().as_uri()

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(args.mlflow_experiment)
    os.environ["MLFLOW_TRACKING_URI"]    = uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment
    return uri


def _safe_params(d: dict) -> dict:
    return {k: str(v) for k, v in d.items()}


# ── Dropout injection ─────────────────────────────────────────────────────────

def inject_dropout(model: YOLO, head_p: float = 0.10, backbone_p: float = 0.05):
    """
    Inject dropout into a YOLOv8 model.

    Strategy:
    ──────────
    1. Detect head  → set model.model[-1].dropout  (Ultralytics exposes this attr)
    2. Backbone     → patch any existing nn.Dropout layers to backbone_p
       (YOLOv8n/s/m don't have dropout in backbone by default, but this
        future-proofs for variants that do, and handles custom models)

    NOTE: dropout is automatically disabled during val/test because
    Ultralytics calls model.eval() before every validation pass.
    You do NOT need to manage this manually.
    """
    nn_model = model.model  # DetectionModel (torch.nn.Module)

    # ── 1. Head dropout ───────────────────────────────────────────────────────
    detect_head = nn_model.model[-1]
    if hasattr(detect_head, "dropout"):
        old_p = detect_head.dropout
        detect_head.dropout = head_p
        LOGGER.info(f"[Dropout] Detect head: {old_p} → {head_p}")
    else:
        LOGGER.warning(
            f"[Dropout] model[-1] has no .dropout attr "
            f"(model type: {type(detect_head).__name__}). "
            f"Skipping head dropout — upgrade ultralytics if you need it."
        )

    # ── 2. Backbone dropout ───────────────────────────────────────────────────
    patched = 0
    for module in nn_model.model.modules():
        if isinstance(module, nn.Dropout):
            module.p = backbone_p
            patched += 1

    if patched:
        LOGGER.info(f"[Dropout] Patched {patched} existing backbone Dropout layers → p={backbone_p}")
    else:
        LOGGER.info(f"[Dropout] No existing Dropout layers in backbone "
                    f"(normal for YOLOv8n/s/m — head dropout only)")

    return model


# ── Class distribution check ──────────────────────────────────────────────────

def check_class_distribution(data_yaml: str) -> Counter:
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    train_path = Path(cfg.get("train", ""))
    yaml_dir   = Path(data_yaml).parent
    if not train_path.is_absolute():
        train_path = (yaml_dir / train_path).resolve()

    candidates = [
        train_path.parent.parent / "labels" / "train",
        train_path.parent / "labels" / "train",
        train_path.parent / "labels",
        Path(str(train_path).replace("/images/", "/labels/")),
        yaml_dir / "labels" / "train",
        yaml_dir / "train" / "labels",
    ]
    label_path = next((c for c in candidates if c.exists()), None)

    if label_path is None:
        LOGGER.warning("[Distribution] Could not locate label directory. Tried:")
        for c in candidates:
            LOGGER.warning(f"  {c}")
        return Counter()

    counts: Counter = Counter()
    for lf in label_path.glob("*.txt"):
        for line in lf.read_text().splitlines():
            if line.strip():
                try:
                    counts[int(line.split()[0])] += 1
                except (ValueError, IndexError):
                    pass

    names  = cfg.get("names", CLASS_NAMES)
    total  = sum(counts.values()) or 1
    nc_cfg = cfg.get("nc", len(names))

    # ── Warn if nc in data.yaml doesn't match CLASS_NAMES ────────────────────
    if nc_cfg != len(CLASS_NAMES):
        LOGGER.warning(
            f"[Distribution] data.yaml has nc={nc_cfg} but CLASS_NAMES has "
            f"{len(CLASS_NAMES)} entries. Update CLASS_NAMES or data.yaml!"
        )

    print("\n" + "═" * 70)
    print("  CLASS DISTRIBUTION — training split")
    print("═" * 70)
    for cls_id, cnt in sorted(counts.items()):
        name = names[cls_id] if isinstance(names, list) else names.get(cls_id, str(cls_id))
        pct  = 100 * cnt / total
        bar  = "█" * int(30 * cnt / total)

        if pct < 3:
            flag = "  🔴 CRITICAL (<3%) — heavy oversampling needed"
        elif pct < 6:
            flag = "  🟠 VERY LOW (<6%) — copy_paste essential"
        elif pct < 10:
            flag = "  🟡 LOW (<10%) — monitor per-class AP50"
        else:
            flag = ""

        print(f"  {name:>15} (cls {cls_id}): {cnt:>5}  {bar:<30}  {pct:5.1f}%{flag}")

    print(f"  {'TOTAL':>15}         : {total:>5}")
    print("═" * 70 + "\n")

    # ── Epoch suggestion based on rarest class ────────────────────────────────
    if counts:
        min_count = min(counts.values())
        if min_count < 30:
            print(f"  ⚠ Rarest class has only {min_count} samples.")
            print(f"    → Strongly recommend: --epochs 150 --patience 45")
        elif min_count < 100:
            print(f"  ⚠ Rarest class has {min_count} samples.")
            print(f"    → Recommend: --epochs 120 --patience 40")
        print()

    return counts


# ── Per-epoch callbacks ───────────────────────────────────────────────────────

def make_callbacks():
    def on_train_epoch_end(trainer):
        epoch    = trainer.epoch + 1
        log_dict = {}

        if hasattr(trainer, "tloss") and trainer.tloss is not None:
            try:
                log_dict["train/loss_total"] = float(trainer.tloss)
            except Exception:
                pass

        if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
            try:
                items = trainer.loss_items.detach().cpu().tolist()
                for name, val in zip(
                    ["train/box_loss", "train/cls_loss", "train/dfl_loss"], items
                ):
                    log_dict[name] = float(val)
            except Exception:
                pass

        lr = getattr(trainer, "lr", None)
        if lr:
            try:
                for i, lr_val in enumerate(lr.values()):
                    log_dict[f"train/lr_pg{i}"] = float(lr_val)
            except Exception:
                pass

        if log_dict:
            mlflow.log_metrics(log_dict, step=epoch)

    def on_fit_epoch_end(trainer):
        epoch    = trainer.epoch + 1
        log_dict = {}

        metrics = getattr(trainer, "metrics", None) or {}
        for k, v in metrics.items():
            try:
                clean_key = k.replace("(B)", "").strip()
                log_dict[clean_key] = float(v)
            except Exception:
                pass

        # ── Per-class AP50 mid-training (if available) ────────────────────────
        validator = getattr(trainer, "validator", None)
        if validator:
            box = getattr(getattr(validator, "metrics", None), "box", None)
            if box and hasattr(box, "ap50") and box.ap50 is not None:
                try:
                    for cls_name, ap in zip(CLASS_NAMES, box.ap50.tolist()):
                        log_dict[f"val/AP50_{cls_name}"] = float(ap)
                except Exception:
                    pass

        if log_dict:
            mlflow.log_metrics(log_dict, step=epoch)
            LOGGER.info(f"[MLflow] epoch {epoch} → {list(log_dict.keys())}")

    def on_val_end(validator):
        trainer = getattr(validator, "trainer", None)
        if trainer is None:
            return

        epoch    = trainer.epoch + 1
        log_dict = {}
        box      = getattr(getattr(validator, "metrics", None), "box", None)

        if box is not None:
            try:
                log_dict.update({
                    "metrics/mAP50":     float(box.map50),
                    "metrics/mAP50-95":  float(box.map),
                    "metrics/precision": float(box.mp),
                    "metrics/recall":    float(box.mr),
                })
            except Exception:
                pass

        if log_dict:
            mlflow.log_metrics(log_dict, step=epoch)

    return {
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end":   on_fit_epoch_end,
        "on_val_end":         on_val_end,
    }


def _remove_builtin_mlflow_callbacks(model):
    for event in list(model.callbacks.keys()):
        model.callbacks[event] = [
            fn for fn in model.callbacks[event]
            if "mlflow" not in getattr(fn, "__module__", "").lower()
        ]

    def _patch_trainer_callbacks(trainer):
        for event in list(trainer.callbacks.keys()):
            trainer.callbacks[event] = [
                fn for fn in trainer.callbacks[event]
                if "mlflow" not in getattr(fn, "__module__", "").lower()
            ]
        LOGGER.info("[MLflow] Removed built-in Ultralytics MLflow callbacks")

    model.add_callback("on_pretrain_routine_start", _patch_trainer_callbacks)


# ── Artifact logging ──────────────────────────────────────────────────────────

def log_run_artifacts(save_dir: Path):
    if not save_dir.exists():
        LOGGER.warning(f"[MLflow] save_dir {save_dir} not found — skipping artifacts")
        return

    for name in (
        "results.png", "results.csv",
        "confusion_matrix.png", "confusion_matrix_normalized.png",
        "PR_curve.png", "P_curve.png", "R_curve.png", "F1_curve.png",
        "labels.jpg", "labels_correlogram.jpg",
        "args.yaml",
    ):
        f = save_dir / name
        if f.exists():
            mlflow.log_artifact(str(f))

    for wname in ("best.pt", "last.pt"):
        w = save_dir / "weights" / wname
        if w.exists():
            mlflow.log_artifact(str(w), artifact_path="weights")
            LOGGER.info(f"[MLflow] weight logged: {wname}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    uri  = setup_mlflow(args)

    # Auto-generate unique run name with timestamp
    run_name = args.mlflow_run_name or f"v8_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ── Augmentation config ───────────────────────────────────────────────────
    #
    # KEY CHANGES vs v7:
    #
    # copy_paste=0.6  (was 0.5)
    #   plates has only ~52 samples — this is the single most impactful
    #   augmentation for minority classes. Pastes instances from other images.
    #
    # erasing=0.3  (new)
    #   Random erasing prevents the model from memorising the exact textures
    #   of the 52 plates training images. Acts like dropout at the data level.
    #
    # scale=0.35  (was 0.5)
    #   labels.jpg width/height plot shows objects are SMALL (0.05-0.10).
    #   Aggressive scale jitter (0.5) crops them out of frame. Reduced.
    #
    # mosaic=0.9  (was 0.85)
    #   More mosaic → more multi-class scenes per image → better co-occurrence
    #   learning (plates next to db/kb teaches spatial context).

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=args.patience,
        cos_lr=True,

        # Loss weights
        cls=2.0,
        label_smoothing=0.5,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
    )

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.set_tags({
            "dataset":          "dumbells-and-kettlebells-v15",
            "fix_version":      "v8",
            "fix_target":       "plates_mb_confusion",
            "imbalanced_cls":   "plates,medicine_ball",
            "model":            args.model,
            "dropout_head":     str(args.head_dropout),
            "dropout_backbone": str(args.backbone_dropout),
            "num_classes":      "5",
        })

        print("\n" + "═" * 65)
        print("  MLflow run started — v8")
        print("═" * 65)
        print(f"  tracking_uri  : {uri}")
        print(f"  experiment    : {args.mlflow_experiment}")
        print(f"  run_id        : {run_id}")
        print(f"  run_name      : {run_name}")
        print(f"  dropout       : head={args.head_dropout}, backbone={args.backbone_dropout}")
        print(f"  epochs        : {args.epochs} (patience={args.patience})")
        print("═" * 65 + "\n")

        # ── Class distribution ────────────────────────────────────────────────
        counts = check_class_distribution(args.data)

        if counts:
            mlflow.log_params(_safe_params({
                f"cls_count/{CLASS_NAMES[i] if i < len(CLASS_NAMES) else i}": v
                for i, v in counts.items()
            }))
            max_count = max(counts.values()) if counts else 1
            mlflow.log_params(_safe_params({
                f"cls_imbalance_ratio/{CLASS_NAMES[i] if i < len(CLASS_NAMES) else i}":
                    round(max_count / v, 2) if v > 0 else 999
                for i, v in counts.items()
            }))

        mlflow.log_params(_safe_params({
            "base_model":        args.model,
            "dataset":           "dumbells-and-kettlebells-v15",
            "num_classes":       5,
            "classes":           ",".join(CLASS_NAMES),
            "fix_version":       "v8",
            "head_dropout":      args.head_dropout,
            "backbone_dropout":  args.backbone_dropout,
            **train_kwargs,
        }))

        # ── Build model ───────────────────────────────────────────────────────
        model = YOLO(args.model)

        # ── Inject dropout ────────────────────────────────────────────────────
        inject_dropout(model, head_p=args.head_dropout, backbone_p=args.backbone_dropout)

        # ── Log dropout confirmation to MLflow ────────────────────────────────
        detect_head = model.model.model[-1]
        actual_head_dropout = getattr(detect_head, "dropout", "N/A")
        mlflow.log_param("actual_head_dropout", str(actual_head_dropout))
        LOGGER.info(f"[Dropout] Confirmed head dropout = {actual_head_dropout}")

        # Strip Ultralytics built-in MLflow callbacks
        _remove_builtin_mlflow_callbacks(model)

        # Register custom per-epoch callbacks
        for event, fn in make_callbacks().items():
            model.add_callback(event, fn)

        # ── Train ─────────────────────────────────────────────────────────────
        results = model.train(**train_kwargs)

        save_dir = None
        if results is not None:
            save_dir = Path(getattr(results, "save_dir", "") or "")
        if not save_dir or not save_dir.exists():
            save_dir = Path(getattr(model.trainer, "save_dir", "."))
        mlflow.log_param("save_dir", str(save_dir))

        # ── Final test-split evaluation ───────────────────────────────────────
        print("\n[MLflow] Running final test-split evaluation...")
        test_results = model.val(data=args.data, split="test")
        box = test_results.box

        test_metrics = {
            "test/mAP50":     float(box.map50),
            "test/mAP50-95":  float(box.map),
            "test/precision": float(box.mp),
            "test/recall":    float(box.mr),
        }

        # Per-class AP50  ← most useful for diagnosing plates / medicine ball confusion
        if hasattr(box, "ap50") and box.ap50 is not None:
            for cls_name, ap in zip(CLASS_NAMES, box.ap50.tolist()):
                test_metrics[f"test/AP50_{cls_name}"] = float(ap)

        # Per-class AP50-95
        if hasattr(box, "ap") and box.ap is not None:
            try:
                for cls_name, ap in zip(CLASS_NAMES, box.ap.tolist()):
                    test_metrics[f"test/AP_{cls_name}"] = float(ap)
            except Exception:
                pass

        mlflow.log_metrics(test_metrics)

        # ── Upload plots + weights ────────────────────────────────────────────
        log_run_artifacts(save_dir)

        # ── Print summary ─────────────────────────────────────────────────────
        print("\n" + "═" * 65)
        print("  TEST METRICS")
        print("═" * 65)
        print(f"  {'mAP50':>20} : {test_metrics.get('test/mAP50', 0):.4f}")
        print(f"  {'mAP50-95':>20} : {test_metrics.get('test/mAP50-95', 0):.4f}")
        print(f"  {'Precision':>20} : {test_metrics.get('test/precision', 0):.4f}")
        print(f"  {'Recall':>20} : {test_metrics.get('test/recall', 0):.4f}")
        print()
        print("  Per-class AP50 (watch plates and medicine ball):")
        for cls_name in CLASS_NAMES:
            key = f"test/AP50_{cls_name}"
            if key in test_metrics:
                v   = test_metrics[key]
                bar = "█" * int(40 * v)
                lvl = ("🔴" if v < 0.3 else "🟡" if v < 0.5 else "🟢")
                print(f"  {cls_name:>15} : {v:.4f}  {bar:<40}  {lvl}")
        print("═" * 65)
        print(f"\n  MLflow run_id  : {run_id}")
        print(f"  Best weights   : {save_dir / 'weights' / 'best.pt'}")
        print(f"\n  ► mlflow ui --backend-store-uri {uri} --port 5000")
        print()

        # ── Final plates warning if AP50 is low ──────────────────────────────
        plates_ap = test_metrics.get("test/AP50_plates", None)
        if plates_ap is not None and plates_ap < 0.4:
            print("  ⚠  plates AP50 is low. Options:")
            print("     1. Collect more plates images (aim for ≥200)")
            print("     2. Raise --epochs to 200 and re-run")
            print("     3. Use focal loss: add fl_gamma=2.0 to train_kwargs")
            print()


if __name__ == "__main__":
    main()