"""
train_v8n_gpu.py  –  Gym Equipment YOLO trainer  (YOLOv8n + Head Dropout + Class Weights)
Dataset  : Roboflow — dumbells-and-kettlebells (v19)
Classes  : bb, db, kb, medicine ball, plates  (nc=5)

════════════════════════════════════════════════════════════════════
  DATASET COUNTS (balanced extraction — bb=198, db=156, kb=195, mb=193, plates=186)
════════════════════════════════════════════════════════════════════
  bb           : 198   ← majority class
  db           : 156   ← minority
  kb           : 195
  medicine ball: 193
  plates       : 186

════════════════════════════════════════════════════════════════════
  CLASS WEIGHTS  (raw inverse-frequency: max_count / class_count)
════════════════════════════════════════════════════════════════════
  bb           : 1.000   (198 / 198)  ← majority, baseline
  db           : 1.269   (198 / 156)
  kb           : 1.015   (198 / 195)
  medicine ball: 1.026   (198 / 193)
  plates       : 1.065   (198 / 186)

  Formula: weight[c] = max(class_counts) / class_counts[c]
  The class with the most instances (bb) gets weight 1.0 (no change).
  Every other class is boosted by exactly the ratio of how many fewer
  instances it has compared to the largest class — so, in effect, each
  class contributes an equal total amount of loss signal regardless of
  how many raw instances it has in the dataset.
  Injected into BCEWithLogitsLoss (bce_cls) inside the Detect head.
  Falls back gracefully with a warning if Ultralytics internals change.

════════════════════════════════════════════════════════════════════
  HOW TO RUN
════════════════════════════════════════════════════════════════════

  # 1. Activate venv
  source /home/pc-008/weight_recognition/venv/bin/activate

  # 2. Basic run — auto-detects GPU (inverse-frequency weights by default)
  python train_v8n_gpu.py

  # 3. Custom class weights (override the computed defaults)
  python train_v8n_gpu.py --class-weights 1.0 1.269 1.015 1.026 1.065

  # 4. Disable class weights entirely (ablation)
  python train_v8n_gpu.py --no-class-weights

  # 5. Force CPU (debug)
  python train_v8n_gpu.py --device cpu

  # 6. View MLflow UI
  mlflow ui \
    --backend-store-uri sqlite:////home/pc-008/weight_project/mlflow.db \
    --port 5000

════════════════════════════════════════════════════════════════════
"""

import argparse
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import mlflow
import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER

import types
import torch.nn as nn
import torch.nn.functional as F

# ── GPU auto-detection ────────────────────────────────────────────────────────
try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
    _GPU_COUNT      = torch.cuda.device_count()
except ImportError:
    _CUDA_AVAILABLE = False
    _GPU_COUNT      = 0

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).resolve().parent

DATA_YAML = (
    "/home/pc-008/weight_project/DB_and_KB_Detetcion/Dumbells and Kettlebells.v19i.yolov8_v5/data.yaml"
)

CLASS_NAMES = ["bb", "db", "kb", "medicine ball", "plates"]

# ── Dataset counts — UPDATED to match the new balanced extraction ───────────
KNOWN_COUNTS = {
    "bb":            278,   # majority
    "db":            232,   # minority
    "kb":            290,
    "medicine ball": 272,
    "plates":        282,
}

# ── Raw inverse-frequency weights ─────────────────────────────────────────
# Formula: weight[c] = max_count / class_count[c]
# The majority class (bb) gets weight 1.0; every other class is boosted by
# the exact ratio of "how many times fewer instances" it has vs. the
# majority class, so each class ends up contributing equally overall.
_max = max(KNOWN_COUNTS.values())   # 198 (bb)
DEFAULT_CLASS_WEIGHTS = [
    round(_max / KNOWN_COUNTS[c], 3) for c in CLASS_NAMES
]
# → [1.0, 1.0, 1.0, 1.0, 1.0]

HEAD_DROPOUT = 0.5
seed=0,   # fixes random seed for reproducibility
# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    default_device  = "0" if _CUDA_AVAILABLE else "cpu"
    default_batch   = 16  if _CUDA_AVAILABLE else 8
    default_workers = 8   if _CUDA_AVAILABLE else 2

    p = argparse.ArgumentParser(description="Train YOLOv8n — gym equipment + class weights")
    p.add_argument("--model",    default="yolov8n.pt")
    p.add_argument("--epochs",   type=int, default=200)
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--batch",    type=int, default=default_batch)
    p.add_argument("--device",   default=default_device)
    p.add_argument("--workers",  type=int, default=default_workers)
    p.add_argument("--project",  default=str(ROOT / "runs"))
    p.add_argument("--name",     default="gym_equipment_v8n_weighted")
    p.add_argument("--resume",   action="store_true")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--data",     default=DATA_YAML)
    p.add_argument("--mlflow-uri",
                   default="sqlite:////home/pc-008/weight_project/mlflow.db")
    p.add_argument("--mlflow-experiment", default="gym_equipment_detection")
    p.add_argument("--mlflow-run-name",   default=None)
    p.add_argument("--head-dropout", type=float, default=HEAD_DROPOUT)
    p.add_argument("--amp",   action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--seed", type=int, default=0)

    # ── Class weight args ─────────────────────────────────────────────────────
    p.add_argument(
        "--class-weights",
        nargs=5,
        type=float,
        default=DEFAULT_CLASS_WEIGHTS,
        metavar=("W_BB", "W_DB", "W_KB", "W_MB", "W_PLATES"),
        help=(
            "Per-class pos_weight for BCEWithLogitsLoss. "
            f"Default (raw inverse-frequency): {DEFAULT_CLASS_WEIGHTS}"
        ),
    )
    p.add_argument(
        "--no-class-weights",
        dest="use_class_weights",
        action="store_false",
        default=True,
        help="Disable class-weight injection (ablation run)",
    )

    return p.parse_args()


# ── GPU info ──────────────────────────────────────────────────────────────────

def get_gpu_info(device: str) -> dict:
    info = {"gpu_available": str(_CUDA_AVAILABLE), "gpu_count": str(_GPU_COUNT)}
    if _CUDA_AVAILABLE and device != "cpu":
        try:
            idx = int(device) if device.isdigit() else 0
            info["gpu_name"]    = torch.cuda.get_device_name(idx)
            info["gpu_vram_gb"] = str(
                round(torch.cuda.get_device_properties(idx).total_memory / 1e9, 1)
            )
        except Exception as e:
            info["gpu_info_error"] = str(e)
    return info


# ── MLflow setup ──────────────────────────────────────────────────────────────

def setup_mlflow(args):
    uri = args.mlflow_uri
    if "://" not in uri:
        uri = "sqlite:///" + str(Path(uri).resolve())
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(args.mlflow_experiment)
    os.environ["MLFLOW_TRACKING_URI"]    = uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment
    print(f"[MLflow] Tracking URI : {uri}")
    return uri


def _safe_params(d: dict) -> dict:
    return {k: str(v) for k, v in d.items()}


# ── Head-only dropout injection ───────────────────────────────────────────────

def inject_dropout(model: YOLO, head_p: float = 0.05):
    detect_head = model.model.model[-1]

    if not hasattr(detect_head, "cv2") or not hasattr(detect_head, "cv3"):
        LOGGER.warning(
            f"[Dropout] model[-1] ({type(detect_head).__name__}) missing cv2/cv3. Skipping."
        )
        return model

    detect_head.dropout = nn.Dropout2d(p=head_p)
    original_forward = detect_head.forward

    def forward_with_dropout(self, x):
        # x is a list of feature maps from different scales (P3, P4, P5)
        x = [self.dropout(xi) if self.training else xi for xi in x]
        return original_forward(x)

    detect_head.forward = types.MethodType(forward_with_dropout, detect_head)

    LOGGER.info(f"[Dropout] ✅ Injected nn.Dropout2d(p={head_p}) before Detect head convs")
    return model


# ── Class-weight injection ────────────────────────────────────────────────────

def make_class_weight_callback(class_weights: list, device: str):
    """
    Returns an on_train_start callback that injects per-class pos_weight
    into the BCEWithLogitsLoss used for classification inside the Detect head.

    Ultralytics YOLOv8 uses bce_cls (BCEWithLogitsLoss) for the cls branch.
    We replace it with a weighted version. By default all weights are 1.0
    (default: raw inverse-frequency); pass --class-weights to override.

    pos_weight[i] > 1  →  recall for class i is prioritised over precision.
    pos_weight[i] = 1  →  standard BCE (no change) — this is the new default.
    """
    dev = (
        torch.device(f"cuda:{device}" if device.isdigit() else "cuda")
        if _CUDA_AVAILABLE and device != "cpu"
        else torch.device("cpu")
    )
    w = torch.tensor(class_weights, dtype=torch.float32, device=dev)

    def _callback(trainer):
        injected = False

        # Strategy 1: search all modules for bce_cls attribute
        for module in trainer.model.modules():
            if hasattr(module, "bce_cls"):
                module.bce_cls = torch.nn.BCEWithLogitsLoss(
                    pos_weight=w, reduction="none"
                )
                LOGGER.info(
                    f"[ClassWeight] ✅ Injected pos_weight into bce_cls: "
                    f"{dict(zip(CLASS_NAMES, [round(x, 3) for x in class_weights]))}"
                )
                injected = True
                break

        # Strategy 2: try loss criterion directly
        if not injected:
            criterion = getattr(trainer, "criterion", None)
            if criterion is not None and hasattr(criterion, "bce_cls"):
                criterion.bce_cls = torch.nn.BCEWithLogitsLoss(
                    pos_weight=w, reduction="none"
                )
                LOGGER.info(
                    f"[ClassWeight] ✅ Injected pos_weight into criterion.bce_cls: "
                    f"{dict(zip(CLASS_NAMES, [round(x, 3) for x in class_weights]))}"
                )
                injected = True

        if not injected:
            LOGGER.warning(
                "[ClassWeight] ⚠ bce_cls not found in model or criterion. "
                "Class weights NOT applied. "
                "This may happen if Ultralytics changed internal loss naming. "
                "fl_gamma is still active as a fallback."
            )

    return _callback

# DROPOUT CALLBACK
def make_dropout_callback(head_p: float):
    def _callback(trainer):
        detect_head = trainer.model.model[-1]
        if not hasattr(detect_head, "cv2") or not hasattr(detect_head, "cv3"):
            LOGGER.warning("[Dropout] Detect head missing cv2/cv3. Skipping.")
            return

        detect_head.dropout = nn.Dropout2d(p=head_p)
        original_forward = detect_head.forward

        def forward_with_dropout(self, x):
            x = [self.dropout(xi) if self.training else xi for xi in x]
            return original_forward(x)

        detect_head.forward = types.MethodType(forward_with_dropout, detect_head)
        LOGGER.info(f"[Dropout] ✅ Injected p={head_p} on trainer.model (on_train_start)")

    return _callback


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
        LOGGER.warning("[Distribution] Could not locate label directory.")
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
            f"[Distribution] data.yaml nc={nc_cfg} ≠ CLASS_NAMES {len(CLASS_NAMES)}"
        )

    print("\n" + "═" * 70)
    print("  CLASS DISTRIBUTION — training split")
    print("═" * 70)
    for cls_id, cnt in sorted(counts.items()):
        name = names[cls_id] if isinstance(names, list) else names.get(cls_id, str(cls_id))
        pct  = 100 * cnt / total
        bar  = "█" * int(30 * cnt / total)
        if pct < 3:    flag = "  🔴 CRITICAL (<3%)"
        elif pct < 6:  flag = "  🟠 VERY LOW (<6%)"
        elif pct < 10: flag = "  🟡 LOW (<10%)"
        else:          flag = ""
        print(f"  {name:>15} (cls {cls_id}): {cnt:>5}  {bar:<30}  {pct:5.1f}%{flag}")
    print(f"  {'TOTAL':>15}         : {total:>5}")
    print("═" * 70 + "\n")
    return counts


# ── Per-epoch MLflow callbacks ────────────────────────────────────────────────

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

        if _CUDA_AVAILABLE:
            try:
                log_dict["gpu/memory_allocated_gb"] = round(
                    torch.cuda.memory_allocated() / 1e9, 3
                )
                log_dict["gpu/memory_reserved_gb"] = round(
                    torch.cuda.memory_reserved() / 1e9, 3
                )
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
                log_dict[k.replace("(B)", "").strip()] = float(v)
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
            if box is not None:
                try:
                    log_dict["val/mAP50"]     = float(box.map50)
                    log_dict["val/mAP50-95"]  = float(box.map)
                    log_dict["val/precision"] = float(box.mp)
                    log_dict["val/recall"]    = float(box.mr)
                except Exception:
                    pass

        if log_dict:
            mlflow.log_metrics(log_dict, step=epoch)
            LOGGER.info(f"[MLflow] epoch {epoch} → {list(log_dict.keys())}")

    return {
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end":   on_fit_epoch_end,
    }


# ── Strip built-in MLflow callbacks ──────────────────────────────────────────

def remove_builtin_mlflow_callbacks(model: YOLO):
    removed = 0
    for event in list(model.callbacks.keys()):
        before = len(model.callbacks[event])
        model.callbacks[event] = [
            fn for fn in model.callbacks[event]
            if "mlflow" not in getattr(fn, "__module__", "").lower()
        ]
        removed += before - len(model.callbacks[event])

    if removed:
        LOGGER.info(f"[MLflow] Stripped {removed} built-in Ultralytics MLflow callback(s)")
    else:
        LOGGER.info("[MLflow] No built-in Ultralytics MLflow callbacks found (already clean)")


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
            LOGGER.info(f"[MLflow] weight logged: {wname}  ({w.stat().st_size / 1e6:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    uri  = setup_mlflow(args)

    if _CUDA_AVAILABLE and args.device != "cpu":
        idx = int(args.device) if str(args.device).isdigit() else 0
        print(f"\n[GPU] ✅ Using: {torch.cuda.get_device_name(idx)}  "
              f"({torch.cuda.get_device_properties(idx).total_memory / 1e9:.1f} GB VRAM)")
        print(f"[GPU] AMP (mixed precision): {'ON ⚡' if args.amp else 'OFF'}")
    else:
        print(f"\n[GPU] ⚠ Running on CPU (CUDA available: {_CUDA_AVAILABLE})")

    # Print class weight plan
    print("\n" + "═" * 70)
    print("  CLASS WEIGHTS (raw inverse-frequency: max_count / class_count)")
    print("═" * 70)
    for name, w, cnt in zip(CLASS_NAMES, args.class_weights, KNOWN_COUNTS.values()):
        bar = "█" * int(w * 10)
        print(f"  {name:>15} (n={cnt:>5}): weight {w:.3f}  {bar}")
    print(f"  Class weights active: {args.use_class_weights}")
    print("═" * 70 + "\n")

    run_name = args.mlflow_run_name or f"v8n_weighted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
        lr0=0.01,   # initial learning rate
        lrf=0.0001,   # final LR = lr0 * lrf
        amp=args.amp if args.device != "cpu" else False,

        # Loss
        cls=2.0,
        label_smoothing=0.0,       # reduced from 0.5 — preserves plate signal
                
       # Augmentation
        hsv_h=0.1,
        hsv_s=0.3,
        hsv_v=0.5,
        degrees=120.0,
        fliplr=0.0,
        flipud=0.0,
        scale=0.1,
        translate=0.1,
    )
  
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        gpu_info = get_gpu_info(str(args.device))
        mlflow.set_tags({
            "dataset":          "dumbells-and-kettlebells-v19",
            "fix_version":      "v8n_invfreq_weighted",
            "model_size":       "nano",
            "model":            args.model,
            "dropout_head":     str(args.head_dropout),
            "num_classes":      "5",
            "device":           str(args.device),
            "amp":              str(args.amp),
            "class_weights":    str(args.use_class_weights),
            **gpu_info,
        })

        print("\n" + "═" * 65)
        print("  MLflow run started — v8n inverse-frequency weighted")
        print("═" * 65)
        print(f"  tracking_uri  : {uri}")
        print(f"  run_id        : {run_id}")
        print(f"  run_name      : {run_name}")
        print(f"  device        : {args.device}  |  AMP: {args.amp}")
        print(f"  epochs        : {args.epochs}  (patience={args.patience})")
        print(f"  batch         : {args.batch}")
        print("═" * 65 + "\n")

        # Log class distribution from actual label files
        counts = check_class_distribution(args.data)
        if counts:
            mlflow.log_params(_safe_params({
                f"cls_count/{CLASS_NAMES[i] if i < len(CLASS_NAMES) else i}": v
                for i, v in counts.items()
            }))
            max_count = max(counts.values())
            mlflow.log_params(_safe_params({
                f"cls_imbalance_ratio/{CLASS_NAMES[i] if i < len(CLASS_NAMES) else i}":
                    round(max_count / v, 2) if v > 0 else 999
                for i, v in counts.items()
            }))

        # Log class weights to MLflow
        mlflow.log_params(_safe_params({
            f"cls_weight/{CLASS_NAMES[i]}": w
            for i, w in enumerate(args.class_weights)
        }))
        mlflow.log_param("class_weights_active", str(args.use_class_weights))
        mlflow.log_param("weight_strategy",      "raw_inverse_frequency")

        mlflow.log_params(_safe_params({
            "base_model":    args.model,
            "dataset":       "dumbells-and-kettlebells-v19",
            "num_classes":   5,
            "classes":       ",".join(CLASS_NAMES),
            "fix_version":   "v8n_invfreq_weighted",
            "head_dropout":  args.head_dropout,
            "amp":           args.amp,
            **train_kwargs,
        }))

        # Build model
        model = YOLO(args.model)

        # Inject head-only dropout
        inject_dropout(model, head_p=args.head_dropout)
        actual_head_dropout = getattr(model.model.model[-1], "dropout", "N/A")
        mlflow.log_param("actual_head_dropout", str(actual_head_dropout))

        # Strip built-in MLflow callbacks
        remove_builtin_mlflow_callbacks(model)

        model.add_callback("on_train_start", make_dropout_callback(args.head_dropout))

        # Register class-weight injection callback
        if args.use_class_weights:
            weight_cb = make_class_weight_callback(args.class_weights, str(args.device))
            model.add_callback("on_train_start", weight_cb)
            LOGGER.info(
                f"[ClassWeight] Registered callback — weights: "
                f"{dict(zip(CLASS_NAMES, [round(w, 3) for w in args.class_weights]))}"
            )
        else:
            LOGGER.info("[ClassWeight] Skipped — --no-class-weights flag set")

        # Register per-epoch MLflow callbacks
        for event, fn in make_callbacks().items():
            model.add_callback(event, fn)

        # Train
        results = model.train(**train_kwargs)

        save_dir = None
        if results is not None:
            save_dir = Path(getattr(results, "save_dir", "") or "")
        if not save_dir or not save_dir.exists():
            save_dir = Path(getattr(model.trainer, "save_dir", "."))
        mlflow.log_param("save_dir", str(save_dir))

        # Final test-split evaluation
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

        # Summary
        print("\n" + "═" * 65)
        print("  TEST METRICS  (YOLOv8n — inverse-frequency class weighting)")
        print("═" * 65)
        print(f"  {'mAP50':>20} : {test_metrics.get('test/mAP50', 0):.4f}")
        print(f"  {'mAP50-95':>20} : {test_metrics.get('test/mAP50-95', 0):.4f}")
        print(f"  {'Precision':>20} : {test_metrics.get('test/precision', 0):.4f}")
        print(f"  {'Recall':>20} : {test_metrics.get('test/recall', 0):.4f}")
        print()
        print("  Per-class AP50:")
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
        print(f"\n  ► mlflow ui --backend-store-uri {uri} --port 5000\n")

        plates_ap = test_metrics.get("test/AP50_plates", None)
        if plates_ap is not None and plates_ap < 0.5:
            print("  ⚠  plates AP50 still below 0.5. Consider:")
            print("     1. Try --class-weights 2.69 1.0 1.5 1.47 5.22  (aggressive)")
            print("     2. Collect more plates images (aim ≥1000 instances)")
            print("     3. Upgrade to yolov8s.pt for more model capacity")


if __name__ == "__main__":
    main()
