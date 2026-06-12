"""
Testset Inference Script — Per-Class Folder Output
====================================================
Runs YOLOv8 inference on each class folder in the testset,
saves annotated images to separate output folders, and
prints per-class detection summary + confusion breakdown.

Testset structure expected:
    TESTSET_DIR/
    ├── bb/
    ├── db/
    ├── kb/
    ├── mb/              ← medicine ball folder
    └── plates/

Output structure created:
    OUTPUT_DIR/
    ├── bb/
    ├── db/
    ├── kb/
    ├── mb/
    ├── plates/
    ├── missed/          ← missed frames per class (if SAVE_MISSED=True)
    │   ├── bb/
    │   └── ...
    └── inference_report.txt
"""

import cv2
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_PATH  = "/home/pc-008/weight_project/DB_and_KB_Detetcion/scripts/runs/gym_equipment_v8n_gpu-6/weights/best.pt"
TESTSET_DIR = "/home/pc-008/weight_project/DB_and_KB_Detetcion/testset"
OUTPUT_DIR  = "/home/pc-008/weight_project/testset_inference_results"

CONF_THRESH  = 0.10
IOU_THRESH   = 0.45
IMG_SIZE     = 640

SAVE_ALL     = False   # True  → save every annotated image including correct detections
SAVE_MISSED  = True    # True  → save unannotated originals that got 0 detections to missed/<class>/

# Folder name on disk → display label used in report
CLASS_FOLDERS = {
    "bb":     "bb",
    "db":     "db",
    "kb":     "kb",
    "mb":     "medicine_ball",   # ← folder is 'mb' on your machine
    "plates": "plates",
}
# ─────────────────────────────────────────────────────────────


def run_inference():
    model = YOLO(MODEL_PATH)
    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    report_lines = []
    grand_total  = {"images": 0, "detected": 0, "missed": 0}

    # Per-class confusion matrix accumulator: confusion[true_cls][pred_cls] = count
    global_confusion = {}

    print(f"\n{'='*60}")
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Testset: {TESTSET_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Conf   : {CONF_THRESH}   IOU: {IOU_THRESH}")
    print(f"{'='*60}\n")

    for folder_name, display_name in CLASS_FOLDERS.items():
        cls_input_dir = Path(TESTSET_DIR) / folder_name
        if not cls_input_dir.exists():
            print(f"[SKIP] Folder not found: {cls_input_dir}")
            continue

        # Output dirs
        cls_output_dir = output_root / folder_name
        cls_output_dir.mkdir(parents=True, exist_ok=True)

        missed_dir = output_root / "missed" / folder_name
        if SAVE_MISSED:
            missed_dir.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(
            list(cls_input_dir.glob("*.jpg"))  +
            list(cls_input_dir.glob("*.jpeg")) +
            list(cls_input_dir.glob("*.png"))
        )
        if not img_paths:
            print(f"[SKIP] No images in {cls_input_dir}")
            continue

        n_total    = len(img_paths)
        n_detected = 0
        n_missed   = 0
        class_hits = {}          # pred_class → box count
        correct_hits = 0         # frames where expected class was among detections

        print(f"[{display_name.upper()}]  {n_total} images  (folder: {folder_name}/)")

        for img_path in img_paths:
            results = model.predict(
                source=str(img_path),
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                imgsz=IMG_SIZE,
                verbose=False,
            )[0]

            boxes      = results.boxes
            has_detect = boxes is not None and len(boxes) > 0

            if has_detect:
                n_detected += 1
                pred_names = set()
                for cls_id in boxes.cls.cpu().numpy().astype(int):
                    det_name = model.names[cls_id]
                    class_hits[det_name] = class_hits.get(det_name, 0) + 1
                    pred_names.add(det_name)
                # Check if correct class was detected in this frame
                if display_name in pred_names:
                    correct_hits += 1
                # Update global confusion
                if display_name not in global_confusion:
                    global_confusion[display_name] = {}
                for pn in pred_names:
                    global_confusion[display_name][pn] = global_confusion[display_name].get(pn, 0) + 1
            else:
                n_missed += 1
                if SAVE_MISSED:
                    import shutil
                    shutil.copy2(str(img_path), str(missed_dir / img_path.name))

            if SAVE_ALL or has_detect:
                annotated = results.plot()
                out_path  = cls_output_dir / img_path.name
                cv2.imwrite(str(out_path), annotated)

        recall       = (n_detected   / n_total * 100) if n_total > 0 else 0.0
        precision_cls = (correct_hits / n_detected * 100) if n_detected > 0 else 0.0

        # Sort hits: correct class first, then by count desc
        sorted_hits = dict(sorted(
            class_hits.items(),
            key=lambda x: (x[0] != display_name, -x[1])
        ))

        # Flag wrong-class detections
        wrong_hits = {k: v for k, v in class_hits.items() if k != display_name}

        summary = (
            f"  Total images          : {n_total}\n"
            f"  Frames with ≥1 box    : {n_detected}  ({recall:.1f}%)\n"
            f"  Missed (0 boxes)      : {n_missed}\n"
            f"  Correct class in frame: {correct_hits}  ({precision_cls:.1f}% of detected)\n"
            f"  All detected classes  : {sorted_hits}\n"
        )
        if wrong_hits:
            summary += f"  ⚠ Wrong-class boxes   : {wrong_hits}\n"

        print(summary)
        report_lines.append(f"CLASS: {display_name}  [{folder_name}/]")
        report_lines.append(summary)
        report_lines.append("-" * 50)

        grand_total["images"]   += n_total
        grand_total["detected"] += n_detected
        grand_total["missed"]   += n_missed

    # ── Grand summary ─────────────────────────────────────────
    overall_recall = (
        grand_total["detected"] / grand_total["images"] * 100
        if grand_total["images"] > 0 else 0.0
    )
    grand_line = (
        f"\nOVERALL\n"
        f"  Total images : {grand_total['images']}\n"
        f"  Detected     : {grand_total['detected']}  ({overall_recall:.1f}%)\n"
        f"  Missed       : {grand_total['missed']}\n"
    )

    # ── Cross-class confusion summary ────────────────────────
    confusion_lines = ["\nCROSS-CLASS CONFUSION SUMMARY"]
    confusion_lines.append("  (rows = true class, cols = what model predicted)\n")
    all_cls = list(CLASS_FOLDERS.values())
    true_pred_label = "True \\ Pred"
    header  = f"  {true_pred_label:<16}" + "".join(f"{c:<16}" for c in all_cls)
    confusion_lines.append(header)
    confusion_lines.append("  " + "-" * (16 * (len(all_cls) + 1)))
    for true_cls in all_cls:
        row_data = global_confusion.get(true_cls, {})
        row = f"  {true_cls:<16}"
        for pred_cls in all_cls:
            val = row_data.get(pred_cls, 0)
            flag = " ←!" if (val > 0 and pred_cls != true_cls) else ""
            row += f"{str(val) + flag:<16}"
        confusion_lines.append(row)
    confusion_text = "\n".join(confusion_lines)

    print("=" * 60)
    print(grand_line)
    print(confusion_text)

    report_lines.append(grand_line)
    report_lines.append(confusion_text)

    # ── Write report ──────────────────────────────────────────
    report_path = output_root / "inference_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model : {MODEL_PATH}\n")
        f.write(f"Conf  : {CONF_THRESH}   IOU: {IOU_THRESH}\n\n")
        f.write("\n".join(report_lines))

    print(f"\n✓ Report saved → {report_path}")
    print(f"✓ Annotated images → {OUTPUT_DIR}/<class>/")
    if SAVE_MISSED:
        print(f"✓ Missed frames    → {OUTPUT_DIR}/missed/<class>/\n")


if __name__ == "__main__":
    run_inference()