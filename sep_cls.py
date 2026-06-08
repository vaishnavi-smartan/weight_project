"""
extract_and_sort_frames.py
==========================
Extracts frames from videos and sorts them into class-specific folders
using a trained YOLOv8 model.

Classes: db (dumbbell), kb (kettlebell), mb (medicine ball), bb (barbell), plates

Usage:
    python extract_and_sort_frames.py --videos_dir /path/to/videos \
                                       --output_dir /path/to/output \
                                       --model /path/to/best.pt \
                                       [--frame_interval 5] \
                                       [--conf 0.4] \
                                       [--save_undetected]
"""

import os
import cv2
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────────
# Class mapping — must match your model's order
# ─────────────────────────────────────────────
CLASS_NAMES = {
    0: "bb",
    1: "db",
    2: "kb",
    3: "mb",       # medicine ball
    4: "plates",
}

UNDETECTED_FOLDER = "undetected"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract video frames and sort by detected class")
    parser.add_argument("--videos_dir",     type=str, required=True,  help="Folder containing input videos")
    parser.add_argument("--output_dir",     type=str, required=True,  help="Root output folder for sorted frames")
    parser.add_argument("--model",          type=str, required=True,  help="Path to trained YOLOv8 .pt weights")
    parser.add_argument("--frame_interval", type=int, default=5,      help="Extract every N-th frame (default: 5)")
    parser.add_argument("--conf",           type=float, default=0.4,  help="Detection confidence threshold (default: 0.4)")
    parser.add_argument("--img_size",       type=int, default=640,    help="Inference image size (default: 640)")
    parser.add_argument("--save_undetected",action="store_true",       help="Also save frames with no detections")
    parser.add_argument("--save_annotated", action="store_true",       help="Draw bounding boxes on saved frames")
    return parser.parse_args()


def create_output_folders(output_dir: str) -> dict:
    """Create one subfolder per class + an undetected folder."""
    folders = {}
    for name in list(CLASS_NAMES.values()) + [UNDETECTED_FOLDER]:
        folder = os.path.join(output_dir, name)
        os.makedirs(folder, exist_ok=True)
        folders[name] = folder
    return folders


def get_dominant_class(results, conf_threshold: float) -> str | None:
    """
    Given a single-image YOLO result, return the class label with
    the highest-confidence detection above the threshold.
    Returns None if nothing is detected above threshold.
    """
    best_conf  = 0.0
    best_class = None

    for box in results[0].boxes:
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = CLASS_NAMES.get(cls)
        if label is None:
            continue
        if conf >= conf_threshold and conf > best_conf:
            best_conf  = conf
            best_class = label

    return best_class


def annotate_frame(frame, results, conf_threshold: float):
    """Draw bounding boxes + labels on the frame."""
    for box in results[0].boxes:
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = CLASS_NAMES.get(cls, "unknown")
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, f"{label} {conf:.2f}",
            (x1, max(y1 - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    return frame


def process_video(video_path: str, model: YOLO, folders: dict, args) -> dict:
    """
    Extract frames from a single video and sort them into class folders.
    Returns a stats dict.
    """
    video_name = Path(video_path).stem
    cap        = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  [!] Cannot open video: {video_path}")
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video : {Path(video_path).name}  |  frames: {total_frames}  |  fps: {fps:.1f}")

    stats          = {cls: 0 for cls in list(CLASS_NAMES.values()) + [UNDETECTED_FOLDER]}
    frame_idx      = 0
    saved_idx      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Sample every N-th frame ──────────────────────────────────
        if frame_idx % args.frame_interval != 0:
            frame_idx += 1
            continue

        # ── Run inference ────────────────────────────────────────────
        results       = model(frame, imgsz=args.img_size, verbose=False)
        dominant_cls  = get_dominant_class(results, args.conf)

        # ── Decide where to save ─────────────────────────────────────
        if dominant_cls is None:
            if not args.save_undetected:
                frame_idx += 1
                continue
            dest_folder = folders[UNDETECTED_FOLDER]
            dominant_cls = UNDETECTED_FOLDER
        else:
            dest_folder = folders[dominant_cls]

        # ── Optionally annotate ──────────────────────────────────────
        save_frame = annotate_frame(frame.copy(), results, args.conf) \
                     if args.save_annotated else frame

        # ── Save frame ───────────────────────────────────────────────
        filename  = f"{video_name}_f{frame_idx:06d}.jpg"
        save_path = os.path.join(dest_folder, filename)
        cv2.imwrite(save_path, save_frame)
        stats[dominant_cls] += 1
        saved_idx            += 1

        frame_idx += 1

    cap.release()
    print(f"  Saved {saved_idx} frames from {total_frames} total  (interval={args.frame_interval})")
    return stats


def main():
    args = parse_args()

    # ── Validate paths ───────────────────────────────────────────────
    if not os.path.isdir(args.videos_dir):
        raise FileNotFoundError(f"Videos directory not found: {args.videos_dir}")
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model weights not found: {args.model}")

    # ── Load model ───────────────────────────────────────────────────
    print(f"\n[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    # ── Create output folders ────────────────────────────────────────
    folders = create_output_folders(args.output_dir)
    print(f"[INFO] Output root  : {args.output_dir}")
    print(f"[INFO] Class folders: {list(folders.keys())}\n")

    # ── Collect videos ───────────────────────────────────────────────
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv")
    video_files = [
        os.path.join(args.videos_dir, f)
        for f in sorted(os.listdir(args.videos_dir))
        if f.lower().endswith(video_extensions)
    ]

    if not video_files:
        print(f"[WARN] No videos found in: {args.videos_dir}")
        return

    print(f"[INFO] Found {len(video_files)} video(s) to process\n")

    # ── Process each video ───────────────────────────────────────────
    global_stats = {cls: 0 for cls in list(CLASS_NAMES.values()) + [UNDETECTED_FOLDER]}

    for i, vpath in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] Processing: {Path(vpath).name}")
        vstats = process_video(vpath, model, folders, args)
        for cls, count in vstats.items():
            global_stats[cls] += count
        print()

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 50)
    print("  FINAL FRAME COUNT PER CLASS")
    print("=" * 50)
    total = 0
    for cls, count in global_stats.items():
        print(f"  {cls:<15}: {count:>6} frames  →  {folders[cls]}")
        total += count
    print(f"  {'TOTAL':<15}: {total:>6} frames")
    print("=" * 50)
    print("\n[DONE]")


if __name__ == "__main__":
    main()