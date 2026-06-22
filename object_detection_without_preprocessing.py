"""
train_v8n_db_fix.py  –  Gym Equipment YOLO trainer  (v8n_db_fix — YOLOv8n, GPU)
Dataset  : Roboflow — dumbells-and-kettlebells (v15)
Classes  : bb, db, kb, medicine ball, plates  (nc=5)

════════════════════════════════════════════════════════════════════
  THIS IS A SHARED SCRIPT — 3 people use this.
  Keep changes small, isolated, and clearly commented so anyone
  reading this can quickly see what changed vs the base v9n script
  and why, without re-deriving the reasoning.
════════════════════════════════════════════════════════════════════

WHAT CHANGED vs train_v9n.py (base script)  —  ONLY 4 CHANGES:
────────────────────────────────────────────────────────────────────
  Evidence this experiment is responding to (from v9n training graphs
  + Precision-Confidence curve + confusion matrix):

    • db precision crashes from ~0.85 to 0.0 at confidence 0.85-0.9
      while bb/kb/mb/plates all hold near 1.0 confidence.
    • Confusion matrix: db has 27 background misclassifications —
      by far the worst of any class (bb=3, kb=4, mb=1).
    • db has the HIGHEST image count (747) of all classes, so this
      is NOT a data-quantity problem — it's a signal/training problem
      specific to db.
    • val/cls_loss curve is visibly spikier than the other loss curves,
      suggesting classification signal instability.

  1. label_smoothing: 0.5 → 0.1
     WHY: 0.5 is unusually high (typical range 0.0–0.1). It tells the
     model "don't be too confident even when right" — this lines up
     exactly with db's confident-precision collapse. Classes with
     fewer/cleaner shape variations (plates, mb) tolerate this fine,
     but db (which has TWO similar-looking objects — single dumbbell
     vs paired dumbbells) needs sharper signal to commit to a class.

  2. cls: 2.0 → 1.2
     WHY: 2.0 combined with label_smoothing=0.5 was pushing the
     classifier hard while simultaneously telling it to stay uncertain
     — a contradictory training signal. This likely caused the spiky
     val/cls_loss seen in the v9n graphs. 1.2 still emphasises
     classification over the ultralytics default (0.5) without
     fighting the (now lower) smoothing.

  3. copy_paste: 0.6 → 0.4 (kept — NOT removed)
     WHY: copy_paste is essential for plates (low sample count, 52
     images) so it stays on. Lowered slightly because db is the
     second-most copy-pasted class by volume in real gym scenes
     (paired dumbbells near a hand) — at 0.6 it was likely creating
     too many synthetic/unnatural db placements that don't match how
     db actually appears (held in hand, often as a pair of spheres).
     0.4 keeps the plates benefit while reducing synthetic db noise.

  4. close_mosaic: NEW — added, value 15
     WHY: db's natural shape is "two spheres connected by a handle,
     near a hand" — a compound shape. Mosaic stitches 4 images
     together and can fragment that shape unpredictably (cutting a
     dumbbell head off mid-image). Turning mosaic off for the final
     15 epochs lets the model see complete, undistorted db shapes
     right before best.pt is saved — directly supporting the goal of
     learning "two spheres near a hand = db" cleanly, without
     confusing it with bb (which is a single long bar, no paired
     spheres).

  GPU CHANGE (to match balanced_v6 script speed):
    • cache=True added — preloads dataset into RAM/disk cache.
      Same setting used in balanced_v6 which trains noticeably faster.
      Everything else (batch, workers, device) already matched.

  EVERYTHING ELSE IS UNCHANGED — same batch, workers, dropout,
  epochs/patience, hsv/scale/degrees augmentation, callbacks, and
  MLflow logging as the base v9n script. This keeps the experiment
  isolated and easy to compare against the previous run in MLflow.
════════════════════════════════════════════════════════════════════

  HOW TO RUN
════════════════════════════════════════════════════════════════════
  # 1. Activate your venv
  source /home/pc-008/weight_recognition/venv/bin/activate

  # 2. Basic run — nano, GPU 0, all defaults
  python train_v8n_db_fix.py

  # 3. Custom epochs / batch
  python train_v8n_db_fix.py --epochs 200 --batch 16

  # 4. Specify a different GPU, or multiple GPUs
  python train_v8n_db_fix.py --device 1
  python train_v8n_db_fix.py --device 0,1

  # 5. Fall back to CPU if no GPU available
  python train_v8n_db_fix.py --device cpu --batch 8 --workers 2

  # 6. Point to a specific MLflow DB
  python train_v8n_db_fix.py --mlflow-uri sqlite:////home/pc-008/weight_project/mlflow.db

  # 7. Resume interrupted run
  python train_v8n_db_fix.py --resume

  # 8. Compare this run vs the base v9n run in MLflow UI
  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
  # Filter by tag fix_version=v8n_db_fix  vs  fix_version=v9n
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
    "/home/smartan/weight_recognition/Yolo/new_dataset/test.v14i.yolov8/data.yaml"
)

CLASS_NAMES = ["bb", "db", "kb", "medicine ball", "plates"]

KNOWN_COUNTS = {
    "bb":            429,
    "db":            747,
    "kb":            442,
    "medicine ball": 106,
    "plates":         52,   # CRITICALLY LOW ⚠
}

# ── Dropout config (nano-tuned, unchanged from v9n) ───────────────────────────
HEAD_DROPOUT     = 0.05
BACKBONE_DROPOUT = 0.0


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8n on gym equipment dataset (v8n_db_fix)")
    p.add_argument("--model",    default="yolov8n.pt",
                   help="Base weights. Default: yolov8n.pt")
    p.add_argument("--epochs",   type=int, default=120,
                   help="Nano converges slower; 200 recommended")
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--batch",    type=int, default=16,
                   help="Batch size. 16 is safe on GPU with ≥8 GB VRAM; raise to 32 with more")
    p.add_argument("--device",   default="0",
                   help="GPU id e.g. '0' (default), multiple GPUs e.g. '0,1', or 'cpu'")
    p.add_argument("--workers",  type=int, default=8,
                   help="Dataloader workers. 8 is a good default on GPU training")
    p.add_argument("--project",  default="/home/smartan/weight_recognition/Yolo/new_exp_op")
    p.add_argument("--name",     default="gym_equipment_v8n_db_fix")
    p.add_argument("--resume",   action="store_true")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--data",     default=DATA_YAML)
    p.add_argument("--mlflow-uri",        default="sqlite:///mlflow.db")
    p.add_argument("--mlflow-experiment", default="gym_equipment_detection")
    p.add_argument("--mlflow-run-name",   default=None,
                   help="Defaults to v8n_db_fix_<timestamp>")
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


# ── Dropout injection (unchanged from v9n) ─────────────────────────────────────

def inject_dropout(model: YOLO, head_p: float = 0.05, backbone_p: float = 0.0):
    nn_model    = model.model
    detect_head = nn_model.model[-1]

    if hasattr(detect_head, "dropout"):
        old_p = detect_head.dropout
        detect_head.dropout = head_p
        LOGGER.info(f"[Dropout] Detect head: {old_p} → {head_p}")
    else:
        LOGGER.warning(
            f"[Dropout] model[-1] ({type(detect_head).__name__}) has no .dropout attr. "
            f"Skipping — upgrade ultralytics if needed."
        )

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


# ── Class distribution check (unchanged from v9n) ──────────────────────────────

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


# ── Per-epoch callbacks (unchanged from v9n) ───────────────────────────────────

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


# ── Artifact logging (unchanged from v9n) ───────────────────────────────────────

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

    # ── GPU availability check ──────────────────────────────────────────────
    if args.device != "cpu":
        if not torch.cuda.is_available():
            print("\n⚠  No CUDA GPU detected, but --device is set to "
                  f"'{args.device}'. Falling back to CPU automatically.")
            print("   To force CPU explicitly next time, use: --device cpu --batch 8 --workers 2\n")
            args.device = "cpu"
            if args.batch > 8:
                args.batch = 8
            if args.workers > 2:
                args.workers = 2
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"\n✅ GPU detected: {gpu_name}  ({gpu_mem:.1f} GB VRAM)")
            print(f"   Using device='{args.device}', batch={args.batch}, workers={args.workers}\n")

    uri  = setup_mlflow(args)

    run_name = args.mlflow_run_name or f"v8n_db_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

        # ── CHANGE 1 & 2 — Loss weights ─────────────────────────────────
        cls=1.2,                  # CHANGED from 2.0
        label_smoothing=0.1,      # CHANGED from 0.5

        # ── Augmentation ────────────────────────────────────────────────
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,

        # ── CHANGE 3 — copy_paste reduced ───────────────────────────────
        copy_paste=0.4,            # CHANGED from 0.6

        # ── CHANGE 4 — NEW: close_mosaic ────────────────────────────────
        close_mosaic=15,           # NEW

        # ── GPU CHANGE — cache added to match balanced_v6 train speed ───
        # Preloads dataset into RAM on first epoch; subsequent epochs
        # skip disk I/O entirely — same as balanced_v6 script which
        # trains noticeably faster because of this.
        # Use cache="disk" if your RAM is limited (< 16 GB).
        cache=True,                # ADDED
    )

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.set_tags({
            "dataset":          "dumbells-and-kettlebells-v15",
            "fix_version":      "v8n_db_fix",
            "base_version":     "v9n",
            "model_size":       "nano",
            "fix_target":       "db_confidence_collapse",
            "imbalanced_cls":   "plates,medicine_ball",
            "model":            args.model,
            "dropout_head":     str(args.head_dropout),
            "dropout_backbone": str(args.backbone_dropout),
            "num_classes":      "5",
            "device":           args.device,
            "changes":          "cls=1.2(was 2.0), label_smoothing=0.1(was 0.5), "
                                 "copy_paste=0.4(was 0.6), close_mosaic=15(new), cache=True(new)",
        })

        print("\n" + "═" * 65)
        print("  MLflow run started — v8n_db_fix  (YOLOv8 nano, db-targeted, GPU)")
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
        print()
        print("  CHANGES vs base v9n script:")
        print("    cls              : 2.0    → 1.2")
        print("    label_smoothing  : 0.5    → 0.1")
        print("    copy_paste       : 0.6    → 0.4")
        print("    close_mosaic     : (none) → 15")
        print("    cache            : (none) → True   ← GPU speed match")
        print("  Target: fix db precision collapse + background confusion")
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
            "fix_version":       "v8n_db_fix",
            "base_version":      "v9n",
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
        print("  TEST METRICS  (YOLOv8n — db_fix)")
        print("═" * 65)
        print(f"  {'mAP50':>20} : {test_metrics.get('test/mAP50', 0):.4f}")
        print(f"  {'mAP50-95':>20} : {test_metrics.get('test/mAP50-95', 0):.4f}")
        print(f"  {'Precision':>20} : {test_metrics.get('test/precision', 0):.4f}")
        print(f"  {'Recall':>20} : {test_metrics.get('test/recall', 0):.4f}")
        print()
        print("  Per-class AP50 (watch db specifically — that's the target):")
        for cls_name in CLASS_NAMES:
            key = f"test/AP50_{cls_name}"
            if key in test_metrics:
                v   = test_metrics[key]
                bar = "█" * int(40 * v)
                lvl = "🔴" if v < 0.3 else "🟡" if v < 0.5 else "🟢"
                marker = "  ← TARGET CLASS" if cls_name == "db" else ""
                print(f"  {cls_name:>15} : {v:.4f}  {bar:<40}  {lvl}{marker}")
        print("═" * 65)
        print(f"\n  MLflow run_id  : {run_id}")
        print(f"  Best weights   : {save_dir / 'weights' / 'best.pt'}")
        print(f"\n  ► Compare vs base v9n run in MLflow UI:")
        print(f"  ► mlflow ui --backend-store-uri {uri} --port 5000")
        print(f"  ► Filter: fix_version=v8n_db_fix  vs  fix_version=v9n")
        print()

        # ── db-specific check ─────────────────────────────────────────────────
        db_ap = test_metrics.get("test/AP50_db", None)
        if db_ap is not None:
            if db_ap < 0.5:
                print("  ⚠  db AP50 still low after this fix. Next steps to consider:")
                print("     1. Check db images for hand-occlusion — may need more")
                print("        examples of 'pair of dumbbells near hand' specifically")
                print("     2. Try copy_paste=0.2 (further reduce synthetic db noise)")
                print("     3. Inspect missed db detections — is it bb confusion")
                print("        (long-shape) or background confusion (no shape)?")
            else:
                print(f"  ✅ db AP50 improved to {db_ap:.4f} — fix appears to be working.")
                print(f"     Compare against previous v9n run's db AP50 in MLflow to confirm.")
            print()


if __name__ == "__main__":
    main()