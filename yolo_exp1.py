"""
YOLOv8n Training — EXP1 : shape_aug_v1
========================================

Goal:
    Improve db / kb discrimination by teaching the model
    shape and hand-posture cues more aggressively.

Key changes from balanced_v6_minor_improvements:
    ✔ No Roboflow augmentation — all augmentation handled here
    ✔ Stronger scale range (0.4) — varied distances / zoom
    ✔ More rotation (15°) — overhead CCTV angle variation
    ✔ Perspective warp added — mimics real camera distortion
    ✔ Mosaic increased to 0.8 — compensates for no Roboflow aug
    ✔ Mild mixup (0.1) — helps overlapping equipment scenes
    ✔ copy_paste (0.1) — pastes objects across images, boosts rare classes
    ✔ Stronger HSV saturation/value — gym lighting variation
    ✔ cls loss weight raised to 1.5 — forces stronger shape classification
    ✔ box loss weight 8.0 — tighter localisation, better shape boundary
    ✔ close_mosaic=20 — clean final epochs for stable convergence
    ✔ overlap_mask=True — better handling of overlapping db/kb detections
    ✔ Epochs 120 — consistent with v6, early stop handles the rest
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
DATA_YAML   = "/home/smartan/weight_recognition/Yolo/latest_data/test.v8i.yolov8/data.yaml"
BASE_MODEL  = "yolov8n.pt"
PROJECT_DIR = "/home/smartan/weight_recognition/Yolo/new_exp_op"
RUN_NAME    = "exp1_shape_aug_v1"

# ══════════════════════════════════════════════
#  2. TRAINING HYPERPARAMETERS
# ══════════════════════════════════════════════
EPOCHS    = 120
IMGSZ     = 640
BATCH     = 16

LR0       = 0.01
LRF       = 0.005

WARMUP_EP = 5
PATIENCE  = 40

# ══════════════════════════════════════════════
#  3. CLASS INFO
# ══════════════════════════════════════════════
CLASS_COUNTS = {
    "bb":     631,
    "db":     817,
    "kb":     783,
    "mb":     467,
    "plates": 358,
}

CLASS_ORDER  = ["bb", "db", "kb", "mb", "plates"]

# Slight boost for mb and plates — still lowest count classes
WEIGHT_LIST  = [1.0, 1.0, 1.0, 1.2, 1.2]

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

    print("[INFO] Class Counts (updated dataset):")
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
            # Model
            "base_model":       BASE_MODEL,
            "experiment_name":  RUN_NAME,

            # Dataset
            "total_images":     2425,
            "train_images":     1714,
            "val_images":       473,
            "test_images":      238,
            "roboflow_aug":     "none",

            # Training
            "epochs":           120,
            "imgsz":            IMGSZ,
            "batch":            BATCH,
            "lr0":              LR0,
            "lrf":              LRF,
            "warmup_epochs":    WARMUP_EP,
            "patience":         PATIENCE,

            # Loss weights
            "cls_loss_weight":  1.5,
            "box_loss_weight":  8.0,

            # Augmentation
            "fliplr":           0.5,
            "flipud":           0.0,
            "degrees":          15,
            "shear":            5,
            "perspective":      0.0005,
            "scale":            0.4,
            "translate":        0.1,
            "hsv_h":            0.015,
            "hsv_s":            0.4,
            "hsv_v":            0.3,
            "mosaic":           0.8,
            "mixup":            0.1,
            "copy_paste":       0.1,
            "close_mosaic":     20,

            # Class weights
            "weight_mb":        1.2,
            "weight_plates":    1.2,
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

            # ── Loss weights ──────────────────
            # Higher cls → forces stronger shape/class discrimination
            cls = 1.5,
            # Higher box → tighter localisation, better shape boundary learning
            box = 8.0,

            # ── Early stop ────────────────────
            patience = PATIENCE,

            # ── Output ────────────────────────
            project = PROJECT_DIR,
            name    = RUN_NAME,
            save    = True,
            verbose = True,

            # ── Misc ──────────────────────────
            cache        = True,
            overlap_mask = True,   # better handling of overlapping db/kb detections

            # ══════════════════════════════════
            # AUGMENTATIONS
            # All augmentation is done here
            # since Roboflow aug is turned off
            # ══════════════════════════════════

            # Flip — horizontal only (vertical makes no sense for gym)
            fliplr = 0.5,
            flipud = 0.0,

            # Rotation — increased to 15° for CCTV angle variety
            degrees = 15,

            # Shear — mild
            shear = 5,

            # Perspective — mimics real overhead camera distortion
            # NEW vs v6
            perspective = 0.0005,

            # Scale — increased to 0.4 (was 0.2)
            # Equipment appears at many distances from camera
            scale = 0.4,

            # Translation
            translate = 0.1,

            # Colour — stronger sat/val for gym lighting variation
            # hsv_s and hsv_v increased vs v6
            hsv_h = 0.015,
            hsv_s = 0.4,
            hsv_v = 0.3,

            # Mosaic — increased to 0.8 (was 0.5)
            # Compensates for no Roboflow augmentation
            mosaic = 0.8,

            # Mixup — mild re-introduction (was 0.0 in v6)
            # Helps with overlapping equipment in frame
            mixup = 0.1,

            # Copy-paste — NEW vs v6
            # Pastes objects across images; boosts mb and plates
            copy_paste = 0.1,

            # Close mosaic — turn off mosaic in final 20 epochs
            # Lets model stabilise on clean images before saving best weights
            # NEW vs v6
            close_mosaic = 20,
        )

        # ─────────────────────────────────────
        # LOG FINAL METRICS TO MLFLOW
        # ─────────────────────────────────────
        if results and hasattr(results, "results_dict"):
            metrics = results.results_dict
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


if __name__ == "__main__":
    main()