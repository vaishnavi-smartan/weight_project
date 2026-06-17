"""
YOLOv8n Training — EXP2 : conservative_aug_v2
===============================================

Goal:
    Fix false positives from EXP1 — hands/heads/people being
    detected as gym equipment. Return to v6's augmentation
    philosophy but take advantage of the larger dataset.

What went wrong in EXP1 and what we fixed:
───────────────────────────────────────────
  ✘ mixup=0.1       → Model learned hand-holding-object patterns,
                       started firing on bare hands as db.
                       FIXED: mixup removed (back to 0.0)

  ✘ copy_paste=0.1  → Partial/cropped objects pasted in made model
                       fire on any partial round/elongated shape.
                       FIXED: copy_paste removed (back to 0.0)

  ✘ scale=0.4       → Too aggressive. Model associated tall vertical
                       shapes (people) with barbells.
                       FIXED: scale reduced to 0.25

  ✘ hsv_s=0.4,      → Round shapes of any colour variety started
    hsv_v=0.3         triggering as medicine ball (head detections).
                       FIXED: back to mild 0.2 / 0.2

  ✘ mosaic=0.8      → Too many cut-and-paste scene contexts. Model
                       learned to fire on partial shapes in any context.
                       FIXED: mosaic reduced to 0.6

  ✘ cls=1.5         → Too high. When wrong, the model now commits
                       harder to the wrong class (db instead of kb).
                       FIXED: gentle push at 1.2

  ✘ box=8.0         → Was not the problem. Unnecessary and can
                       destabilise small-object training.
                       FIXED: removed (use default)

  ✘ perspective=    → Added extra distortion that wasn't needed.
    0.0005             FIXED: removed

  ✘ degrees=15      → Back to v6's 10° — stable for overhead CCTV

What we kept from v6 that still makes sense:
─────────────────────────────────────────────
  ✔ cos_lr=True         — smooth LR decay
  ✔ fliplr=0.5          — horizontal flip still valid
  ✔ flipud=0.0          — vertical flip still wrong for gym
  ✔ shear=5             — mild, works fine
  ✔ translate=0.1       — slight increase from v6's 0.08
  ✔ close_mosaic=15     — clean final epochs for stable weights
  ✔ overlap_mask=True   — helps with overlapping db/kb
  ✔ cache=True          — faster training

New additions vs v6 (safe ones only):
──────────────────────────────────────
  ✔ iou=0.6             — higher IoU threshold reduces duplicate
                           boxes on the same object (was getting
                           2 db boxes on 1 dumbbell in EXP1)
  ✔ cls=1.2             — gentle nudge toward better classification
                           without over-committing to wrong class
  ✔ epochs=130          — more than v6's 120 since dataset is bigger
  ✔ close_mosaic=15     — helps final convergence
"""

import os
import re
import yaml
import mlflow
import matplotlib
matplotlib.use("Agg")

from ultralytics import YOLO

# ══════════════════════════════════════════════
#  1. PATHS & BASIC CONFIG
# ══════════════════════════════════════════════
DATA_YAML   = "/home/smartan/weight_recognition/Yolo/latest_data/test.v6i.yolov8/data.yaml"
BASE_MODEL  = "yolov8n.pt"
PROJECT_DIR = "/home/smartan/weight_recognition/Yolo/new_exp_op"
RUN_NAME    = "exp2_conservative_aug_v2"

# ══════════════════════════════════════════════
#  2. TRAINING HYPERPARAMETERS
# ══════════════════════════════════════════════
EPOCHS    = 130
IMGSZ     = 640
BATCH     = 16

LR0       = 0.008    # same as v6 — proven stable
LRF       = 0.005

WARMUP_EP = 5
PATIENCE  = 40

# ══════════════════════════════════════════════
#  3. CLASS INFO  (updated dataset)
# ══════════════════════════════════════════════
CLASS_COUNTS = {
    "bb":     631,
    "db":     817,
    "kb":     783,
    "mb":     467,
    "plates": 358,
}

CLASS_ORDER = ["bb", "db", "kb", "mb", "plates"]

# mb and plates still lowest — mild boost
WEIGHT_LIST = [1.0, 1.0, 1.0, 1.2, 1.2]

# ══════════════════════════════════════════════
#  4. MLFLOW CONFIG
# ══════════════════════════════════════════════
MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT  = "gym_equipment_detection"

os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════
def sanitize_key(name: str) -> str:
    clean = re.sub(r'[^\w\s\-\.:/]', '_', name)
    return clean.replace(" ", "_")

# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
def main():

    print("\n════════════════════════════════════")
    print(f"   {RUN_NAME.upper()}")
    print("════════════════════════════════════\n")

    print("[INFO] Class Counts:")
    for cls, cnt in CLASS_COUNTS.items():
        print(f"  {cls:<10}: {cnt}")
    print()
    print("[INFO] Class Weights:", WEIGHT_LIST)
    print()

    with open(DATA_YAML) as f:
        data_info = yaml.safe_load(f)

    class_names = [c for c in data_info.get("names") if c in CLASS_COUNTS]
    print("[INFO] Classes:", class_names)
    print()

    with mlflow.start_run(run_name=RUN_NAME):

        # ─────────────────────────────────────
        # LOG PARAMS TO MLFLOW
        # ─────────────────────────────────────
        mlflow.log_params({
            # Identity
            "base_model":           BASE_MODEL,
            "experiment_name":      RUN_NAME,

            # Dataset
            "total_images":         2425,
            "train_images":         1714,
            "val_images":           473,
            "test_images":          238,
            "roboflow_aug":         "none",

            # Training
            "epochs":               EPOCHS,
            "imgsz":                IMGSZ,
            "batch":                BATCH,
            "lr0":                  LR0,
            "lrf":                  LRF,
            "warmup_epochs":        WARMUP_EP,
            "patience":             PATIENCE,

            # Loss
            "cls_loss_weight":      1.2,
            "box_loss_weight":      "default",
            "iou_threshold":        0.6,

            # Augmentation
            "fliplr":               0.5,
            "flipud":               0.0,
            "degrees":              10,
            "shear":                5,
            "perspective":          0.0,
            "scale":                0.25,
            "translate":            0.1,
            "hsv_h":                0.015,
            "hsv_s":                0.2,
            "hsv_v":                0.2,
            "mosaic":               0.6,
            "mixup":                0.0,
            "copy_paste":           0.0,
            "close_mosaic":         15,

            # Fixes from EXP1
            "exp1_issues_fixed":    "hand=db, head=mb, person=bb, duplicate boxes",
        })

        # ─────────────────────────────────────
        # LOAD MODEL
        # ─────────────────────────────────────
        model = YOLO(BASE_MODEL)

        # ─────────────────────────────────────
        # TRAIN
        # ─────────────────────────────────────
        results = model.train(

            # ── Dataset ──────────────────────
            data    = DATA_YAML,

            # ── Core training ─────────────────
            epochs  = EPOCHS,
            imgsz   = IMGSZ,
            batch   = BATCH,

            # ── Optimisation ──────────────────
            lr0           = LR0,
            lrf           = LRF,
            warmup_epochs = WARMUP_EP,
            cos_lr        = True,

            # ── Loss ──────────────────────────
            # Gentle cls push — enough to improve db/kb separation
            # without over-committing to wrong class like EXP1
            cls = 1.2,

            # iou threshold — reduces duplicate boxes on same object
            # (was getting 2 db boxes on 1 dumbbell in EXP1)
            iou = 0.6,

            # ── Early stop ────────────────────
            patience = PATIENCE,

            # ── Output ────────────────────────
            project = PROJECT_DIR,
            name    = RUN_NAME,
            save    = True,
            verbose = True,

            # ── Misc ──────────────────────────
            cache        = True,
            overlap_mask = True,

            # ══════════════════════════════════
            # AUGMENTATIONS
            # Conservative — close to v6 philosophy
            # No Roboflow aug so we handle it all here
            # ══════════════════════════════════

            # Flip
            fliplr = 0.5,
            flipud = 0.0,     # disabled — gym orientation fixed

            # Rotation — back to v6's 10°
            degrees = 10,

            # Shear — mild, unchanged
            shear = 5,

            # Perspective — removed (was adding unnecessary distortion)
            perspective = 0.0,

            # Scale — slight increase from v6's 0.2
            # 0.25 handles distance variation without making
            # people look like barbells
            scale = 0.25,

            # Translation — tiny increase from v6's 0.08
            translate = 0.1,

            # Colour — mild, close to v6
            # Back from EXP1's 0.4/0.3 which caused head=mb
            hsv_h = 0.015,
            hsv_s = 0.2,
            hsv_v = 0.2,

            # Mosaic — moderate increase from v6's 0.5
            # Compensates for no Roboflow aug, but not aggressive like EXP1's 0.8
            mosaic = 0.6,

            # Mixup — removed (was causing hand=db in EXP1)
            mixup = 0.0,

            # Copy-paste — removed (was causing partial shape false fires)
            copy_paste = 0.0,

            # Close mosaic — turn off mosaic in final 15 epochs
            # Model sees clean images before saving best weights
            close_mosaic = 15,
        )

        # ─────────────────────────────────────
        # LOG FINAL METRICS TO MLFLOW
        # ─────────────────────────────────────
        if results and hasattr(results, "results_dict"):
            metrics  = results.results_dict
            loggable = {}
            for k, v in metrics.items():
                try:
                    loggable[sanitize_key(k)] = float(v)
                except (TypeError, ValueError):
                    pass
            if loggable:
                mlflow.log_metrics(loggable)
                print("\n[INFO] Final metrics logged to MLflow:")
                for k, v in loggable.items():
                    print(f"  {k}: {v:.4f}")

        # ─────────────────────────────────────
        # LOG BEST WEIGHTS AS ARTIFACT
        # ─────────────────────────────────────
        best_pt = f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt"
        if os.path.exists(best_pt):
            mlflow.log_artifact(best_pt, artifact_path="weights")
            print(f"\n[INFO] best.pt logged to MLflow artifacts")

        print("\n════════════════════════════════════")
        print("         TRAINING COMPLETE")
        print("════════════════════════════════════\n")
        print(f"[INFO] Best weights : {best_pt}")
        print(f"[INFO] MLflow run   : {RUN_NAME}")
        print(f"[INFO] Experiment   : {EXPERIMENT}\n")

        # ─────────────────────────────────────
        # INFERENCE REMINDER
        # ─────────────────────────────────────
        print("════════════════════════════════════")
        print("  INFERENCE TIP")
        print("════════════════════════════════════")
        print("  Use conf=0.35 or higher when testing.")
        print("  EXP1 false positives were mostly below 0.35.")
        print("  Example:")
        print(f"    model = YOLO('{best_pt}')")
        print("    model.predict(source=..., conf=0.35, iou=0.6)")
        print("════════════════════════════════════\n")


if __name__ == "__main__":
    main()