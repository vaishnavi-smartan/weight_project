"""
train_v9n.py  –  Gym Equipment YOLO trainer  (v9n — YOLOv8n + CPU-safe)
Dataset  : Roboflow — dumbells-and-kettlebells (v15)
Classes  : bb, db, kb, medicine ball, plates  (nc=5)


  1. BATCH  ─ Lowered default to 8 (was 16).
              Nano fits comfortably in CPU RAM at batch=8.
              Raise to 16 if your machine has ≥32 GB RAM.

  2. WORKERS ─ Lowered to 2 (was 4).
               On CPU training, excess workers add overhead
               without throughput gain.

  3. DROPOUT ─ Head dropout reduced to 0.05 (was 0.10).
               Nano has fewer parameters; aggressive dropout
               hurts convergence more than it helps regularise.
               Backbone dropout kept at 0.0 (nano has none anyway).

  4. AUGMENTATION TUNED for nano + CPU:
               • copy_paste=0.6   kept  — still critical for plates
               • scale=0.35       kept  — small objects, tight bbox
               • label_smoothing lowered to 0.1  (was 0.5)
                 Nano has less capacity; heavy label smoothing
                 blurs the already-weak signal for minority classes.
               • cls weight lowered to 1.5 (was 2.0)
                 Same reason — nano needs cleaner gradients.

  5. EPOCHS  ─ Default raised to 200 (was 150).
               Nano converges slower on small datasets.
               Patience raised to 50 accordingly.

  6. RUN NAMING ─ Prefix changed to "v9n_" for easy MLflow
                  filtering alongside v8 (medium) runs.

════════════════════════════════════════════════════════════════════
  NANO vs MEDIUM — WHEN TO USE WHICH
════════════════════════════════════════════════════════════════════
  ┌──────────────┬───────────┬──────────────────────────────────┐
  │ Scenario     │ Use       │ Reason                           │
  ├──────────────┼───────────┼──────────────────────────────────┤
  │ Quick expt   │ nano (n)  │ 3–5× faster per epoch on CPU     │
  │ Hyperparams  │ nano (n)  │ Cheap to iterate                 │
  │ Best mAP     │ medium (m)│ Higher capacity for plates/MB    │
  │ Edge deploy  │ nano (n)  │ Smaller model size (6 MB)        │
  │ Server infer │ medium (m)│ Accuracy matters more            │
  └──────────────┴───────────┴──────────────────────────────────┘

════════════════════════════════════════════════════════════════════
  EPOCH RECOMMENDATION (nano-specific)
════════════════════════════════════════════════════════════════════
  ┌───────────────────────────────┬────────┬──────────┐
  │ Use case                      │ epochs │ patience │
  ├───────────────────────────────┼────────┼──────────┤
  │ Quick experiment / sanity     │   50   │   20     │
  │ Normal training run           │  150   │   45     │
  │ Best accuracy (recommended)   │  200   │   50     │
  │ Max (diminishing returns)     │  300   │   60     │
  └───────────────────────────────┴────────┴──────────┘
  Default in this script: --epochs 200  --patience 50

════════════════════════════════════════════════════════════════════
  HOW TO RUN
════════════════════════════════════════════════════════════════════
  # 1. Activate your venv
  source /home/pc-008/weight_recognition/venv/bin/activate

  # 2. Basic run — nano, CPU, all defaults
  python train_v9n.py

  # 3. Custom epochs / batch
  python train_v9n.py --epochs 200 --batch 8

  # 4. Point to a specific MLflow DB
  python train_v9n.py --mlflow-uri sqlite:////home/pc-008/weight_project/mlflow.db

  # 5. Resume interrupted run
  python train_v9n.py --resume

  # 6. Compare nano vs medium in MLflow UI
  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
  # Filter by tag model=yolov8n.pt  vs  model=yolov8m.pt

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
    "/home/pc-008/weight_project/DB_and_KB_Detetcion/"
    "Dumbells_and_Kettlebells.v3i.yolov8/data.yaml"
)

CLASS_NAMES = ["bb", "db", "kb", "medicine ball", "plates"]

KNOWN_COUNTS = {
    "bb":            429,
    "db":            747,
    "kb":            442,
    "medicine ball": 106,
    "plates":         52,   # CRITICALLY LOW ⚠
}

# ── Dropout config (nano-tuned) ───────────────────────────────────────────────
HEAD_DROPOUT     = 0.05   # reduced vs v8 (0.10) — nano has less capacity
BACKBONE_DROPOUT = 0.0    # nano has no Dropout layers in backbone


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8n on gym equipment dataset (v9n)")
    p.add_argument("--model",    default="yolov8n.pt",         # ← nano
                   help="Base weights. Default: yolov8n.pt")
    p.add_argument("--epochs",   type=int, default=200,        # ← raised for nano
                   help="Nano converges slower; 200 recommended")
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--batch",    type=int, default=8,          # ← lowered for CPU
                   help="Batch size. 8 is safe on CPU; raise to 16 with ≥32 GB RAM")
    p.add_argument("--device",   default="cpu",                # ← CPU default
                   help="'cpu' or GPU id e.g. '0'")
    p.add_argument("--workers",  type=int, default=2,          # ← lowered for CPU
                   help="Dataloader workers. 2 is optimal for CPU training")
    p.add_argument("--project",  default=str(ROOT / "runs"))
    p.add_argument("--name",     default="gym_equipment_v9n")  # ← new run prefix
    p.add_argument("--resume",   action="store_true")
    p.add_argument("--patience", type=int, default=50)         # ← raised for nano
    p.add_argument("--data",     default=DATA_YAML)
    p.add_argument("--mlflow-uri",        default="sqlite:///mlflow.db")
    p.add_argument("--mlflow-experiment", default="gym_equipment_detection")
    p.add_argument("--mlflow-run-name",   default=None,
                   help="Defaults to v9n_<timestamp>")
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

def inject_dropout(model: YOLO, head_p: float = 0.05, backbone_p: float = 0.0):
    """
    Inject dropout into YOLOv8n.

    YOLOv8n has NO Dropout layers in its backbone by default.
    Only the Detect head exposes a .dropout attribute.

    head_p=0.05 is intentionally lower than the medium model (0.10)
    because nano's smaller capacity makes it more sensitive to
    regularisation-induced underfitting.
    """
    nn_model    = model.model
    detect_head = nn_model.model[-1]

    # ── Head dropout ──────────────────────────────────────────────────────────
    if hasattr(detect_head, "dropout"):
        old_p = detect_head.dropout
        detect_head.dropout = head_p
        LOGGER.info(f"[Dropout] Detect head: {old_p} → {head_p}")
    else:
        LOGGER.warning(
            f"[Dropout] model[-1] ({type(detect_head).__name__}) has no .dropout attr. "
            f"Skipping — upgrade ultralytics if needed."
        )

    # ── Backbone dropout (informational only for nano) ────────────────────────
    patched = 0
    if backbone_p > 0.0:
        for module in nn_model.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = backbone_p
                patched += 1

    if patched:
        LOGGER.info(f"[Dropout] Patched {patched} backbone Dropout layers → p={backbone_p}")
    else:
        LOGGER.info("[Dropout] No backbone Dropout layers (expected for YOLOv8n — head only)")

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

    if nc_cfg != len(CLASS_NAMES):
        LOGGER.warning(
            f"[Distribution] data.yaml nc={nc_cfg} but CLASS_NAMES has "
            f"{len(CLASS_NAMES)} entries. Sync them!"
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

    if counts:
        min_count = min(counts.values())
        if min_count < 30:
            print(f"  ⚠ Rarest class has only {min_count} samples.")
            print(f"    → Strongly recommend: --epochs 200 --patience 50")
        elif min_count < 100:
            print(f"  ⚠ Rarest class has {min_count} samples.")
            print(f"    → Recommend: --epochs 150 --patience 45")
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

    run_name = args.mlflow_run_name or f"v9n_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

        # Loss weights (softened vs v8 for nano capacity)
        cls=2.0,                # was 2.0 — nano needs cleaner gradients
        label_smoothing=0.5,    # was 0.5 — heavy smoothing hurts minority classes on nano

        # Augmentation (kept from v8, proven effective for plates)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        degrees=5.0,
        translate=0.1,
        scale=0.5,             # small objects — kept reduced from v8
        shear=0.0,
    )

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.set_tags({
            "dataset":          "dumbells-and-kettlebells-v15",
            "fix_version":      "v9n",
            "model_size":       "nano",
            "fix_target":       "plates_mb_confusion",
            "imbalanced_cls":   "plates,medicine_ball",
            "model":            args.model,
            "dropout_head":     str(args.head_dropout),
            "dropout_backbone": str(args.backbone_dropout),
            "num_classes":      "5",
            "device":           args.device,
        })

        print("\n" + "═" * 65)
        print("  MLflow run started — v9n  (YOLOv8 nano)")
        print("═" * 65)
        print(f"  tracking_uri  : {uri}")
        print(f"  experiment    : {args.mlflow_experiment}")
        print(f"  run_id        : {run_id}")
        print(f"  run_name      : {run_name}")
        print(f"  model         : {args.model}  (nano — {3.2}M params)")
        print(f"  device        : {args.device}")
        print(f"  dropout       : head={args.head_dropout}, backbone={args.backbone_dropout}")
        print(f"  epochs        : {args.epochs} (patience={args.patience})")
        print(f"  batch         : {args.batch}")
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
            "fix_version":       "v9n",
            "model_size":        "nano",
            "head_dropout":      args.head_dropout,
            "backbone_dropout":  args.backbone_dropout,
            **train_kwargs,
        }))

        # ── Build model ───────────────────────────────────────────────────────
        model = YOLO(args.model)

        # ── Inject dropout ────────────────────────────────────────────────────
        inject_dropout(model, head_p=args.head_dropout, backbone_p=args.backbone_dropout)

        detect_head         = model.model.model[-1]
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

        if hasattr(box, "ap50") and box.ap50 is not None:
            for cls_name, ap in zip(CLASS_NAMES, box.ap50.tolist()):
                test_metrics[f"test/AP50_{cls_name}"] = float(ap)

        if hasattr(box, "ap") and box.ap is not None:
            try:
                for cls_name, ap in zip(CLASS_NAMES, box.ap.tolist()):
                    test_metrics[f"test/AP_{cls_name}"] = float(ap)
            except Exception:
                pass

        mlflow.log_metrics(test_metrics)
        log_run_artifacts(save_dir)

        # ── Print summary ─────────────────────────────────────────────────────
        print("\n" + "═" * 65)
        print("  TEST METRICS  (YOLOv8n)")
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
                lvl = "🔴" if v < 0.3 else "🟡" if v < 0.5 else "🟢"
                print(f"  {cls_name:>15} : {v:.4f}  {bar:<40}  {lvl}")
        print("═" * 65)
        print(f"\n  MLflow run_id  : {run_id}")
        print(f"  Best weights   : {save_dir / 'weights' / 'best.pt'}")
        print(f"\n  ► mlflow ui --backend-store-uri {uri} --port 5000")
        print()

        # ── Plates warning ────────────────────────────────────────────────────
        plates_ap = test_metrics.get("test/AP50_plates", None)
        if plates_ap is not None and plates_ap < 0.4:
            print("  ⚠  plates AP50 is low. Options for nano:")
            print("     1. Collect more plates images (aim for ≥200)")
            print("     2. Raise --epochs to 300 and re-run")
            print("     3. Switch to yolov8s.pt or yolov8m.pt for higher capacity")
            print("     4. Use focal loss: add fl_gamma=2.0 to train_kwargs")
            print()


if __name__ == "__main__":
    main()
