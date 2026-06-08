"""
Video Class Frame Counter
--------------------------
Runs YOLO inference on every frame of a video and
counts how many frames each class appears in.

Output example:
    ┌─────────────────────┬────────┬─────────┐
    │ Class               │ Frames │ % Total │
    ├─────────────────────┼────────┼─────────┤
    │ kb                  │  120   │  34.1%  │
    │ db                  │   85   │  24.1%  │
    │ bb                  │   60   │  17.0%  │
    │ medicine ball       │   40   │  11.4%  │
    │ plates              │   25   │   7.1%  │
    │ no_detection        │   22   │   6.3%  │
    └─────────────────────┴────────┴─────────┘
"""

import cv2
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_PATH = "/home/pc-008/weight_project/DB_and_KB_Detetcion/scripts/runs/gym_equipment_v72/weights/best.pt"
VIDEO_PATH = "/home/pc-008/weight_project/infer1_1.mp4"

CONF   = 0.10
IMGSZ  = 640

# Process every Nth frame to go faster (1 = every frame, 5 = every 5th frame)
EVERY_N_FRAMES = 1

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────

model = YOLO(MODEL_PATH)
class_names = model.names   # {0: 'bb', 1: 'db', 2: 'kb', ...}
print(f"✅ Model loaded  |  Classes: {list(class_names.values())}")

# ─────────────────────────────────────────────
# Open Video
# ─────────────────────────────────────────────

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Cannot open video: {VIDEO_PATH}")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS)
duration_sec = total_frames / fps if fps > 0 else 0

print(f"🎬 Video : {Path(VIDEO_PATH).name}")
print(f"   Frames: {total_frames}  |  FPS: {fps:.1f}  |  Duration: {duration_sec:.1f}s")
print(f"   Processing every {EVERY_N_FRAMES} frame(s)...\n")
print("─" * 55)

# ─────────────────────────────────────────────
# Counters
# ─────────────────────────────────────────────

# frames where each class appears (a frame counted once per class regardless of how many boxes)
class_frame_count = defaultdict(int)

# frames where MULTIPLE classes appear together
multi_class_frames = 0

# frames with zero detections
no_detection_frames = 0

processed = 0
frame_idx = 0

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    if (frame_idx - 1) % EVERY_N_FRAMES != 0:
        continue

    processed += 1

    # ── Inference ───────────────────────────────────
    results = model(frame, conf=CONF, imgsz=IMGSZ, verbose=False)
    boxes   = results[0].boxes

    # ── Unique classes in this frame ─────────────────
    detected = set()
    if boxes is not None and len(boxes) > 0:
        for cls_id in boxes.cls.cpu().numpy().astype(int):
            detected.add(class_names[cls_id])

    # ── Update counts ────────────────────────────────
    if len(detected) == 0:
        no_detection_frames += 1
    else:
        for cls in detected:
            class_frame_count[cls] += 1
        if len(detected) > 1:
            multi_class_frames += 1

    # ── Progress log every 50 frames ─────────────────
    if processed % 50 == 0:
        pct = (frame_idx / total_frames) * 100
        print(f"  Progress: {frame_idx}/{total_frames} frames  ({pct:.1f}%)", end="\r")

cap.release()
print(f"\n✅ Done processing {processed} frames.\n")

# ─────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────

# Build rows: all classes + no_detection
rows = []
for cls_name in class_names.values():
    count = class_frame_count.get(cls_name, 0)
    rows.append((cls_name, count))
rows.append(("no_detection", no_detection_frames))

# Sort by count descending
rows.sort(key=lambda x: x[1], reverse=True)

col1 = max(len(r[0]) for r in rows) + 2
col2 = 8
col3 = 9

divider = f"├─{'─'*col1}─┼─{'─'*col2}─┼─{'─'*col3}─┤"
top     = f"┌─{'─'*col1}─┬─{'─'*col2}─┬─{'─'*col3}─┐"
bottom  = f"└─{'─'*col1}─┴─{'─'*col2}─┴─{'─'*col3}─┘"
header  = f"│ {'Class':<{col1}} │ {'Frames':^{col2}} │ {'% Video':^{col3}} │"

print("=" * (col1 + col2 + col3 + 10))
print("  CLASS-WISE FRAME COUNT REPORT")
print("=" * (col1 + col2 + col3 + 10))
print(top)
print(header)
print(divider)

for cls_name, count in rows:
    pct = (count / processed * 100) if processed > 0 else 0
    print(f"│ {cls_name:<{col1}} │ {count:^{col2}} │ {pct:>7.1f}%  │")

print(bottom)

# ── Extra stats ─────────────────────────────────
print(f"\n  Total frames processed : {processed}")
print(f"  Frames with detections : {processed - no_detection_frames}")
print(f"  Frames, no detections  : {no_detection_frames}")
print(f"  Frames, multi-class    : {multi_class_frames}  (multiple classes in same frame)")
print(f"  Video duration         : {duration_sec:.1f}s  @  {fps:.1f} fps")

# ── Save report to txt ───────────────────────────
report_path = Path(VIDEO_PATH).parent / "frame_count_report.txt"
with open(report_path, "w") as f:
    f.write(f"Video : {VIDEO_PATH}\n")
    f.write(f"Model : {MODEL_PATH}\n\n")
    f.write(f"{'Class':<25} {'Frames':>8}   {'% Video':>8}\n")
    f.write("-" * 45 + "\n")
    for cls_name, count in rows:
        pct = (count / processed * 100) if processed > 0 else 0
        f.write(f"{cls_name:<25} {count:>8}   {pct:>7.1f}%\n")
    f.write("-" * 45 + "\n")
    f.write(f"\nTotal frames processed : {processed}\n")
    f.write(f"Frames with detections : {processed - no_detection_frames}\n")
    f.write(f"Frames, no detections  : {no_detection_frames}\n")
    f.write(f"Frames, multi-class    : {multi_class_frames}\n")

print(f"\n💾 Report saved to: {report_path}\n")
