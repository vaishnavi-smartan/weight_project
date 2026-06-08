import cv2
import os
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_PATH  = "/home/pc-008/weight_project/DB_and_KB_Detetcion/scripts/runs/gym_equipment_v82/weights/best.pt"
FRAMES_DIR  = "/home/pc-008/weight_project/frames/"   # folder of extracted frames
OUTPUT_DIR  = "/home/pc-008/weight_project/infer_out/" # save annotated images here

CONF        = 0.10
IMGSZ       = 640

DISPLAY_W   = 1280
DISPLAY_H   = 720

SAVE_OUTPUT = True   # set False to only display without saving

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────

model = YOLO(MODEL_PATH)

# ─────────────────────────────────────────────
# Collect Images
# ─────────────────────────────────────────────

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

frames_path = Path(FRAMES_DIR)
image_files = sorted([
    f for f in frames_path.iterdir()
    if f.suffix.lower() in SUPPORTED_EXT
])

if not image_files:
    print(f"❌ No images found in: {FRAMES_DIR}")
    exit()

print(f"✅ Found {len(image_files)} images in '{FRAMES_DIR}'")

# ─────────────────────────────────────────────
# Prepare Output Dir
# ─────────────────────────────────────────────

if SAVE_OUTPUT:
    out_path = Path(OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"💾 Saving annotated frames to: {OUTPUT_DIR}")

# ─────────────────────────────────────────────
# Display Setup
# ─────────────────────────────────────────────

window_name = "YOLO Detection - Frames"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, DISPLAY_W, DISPLAY_H)

print("\nControls:")
print("  SPACE / N  → Next frame")
print("  B          → Previous frame")
print("  A          → Auto-play toggle")
print("  + / =      → Faster auto-play")
print("  -          → Slower auto-play")
print("  Q          → Quit\n")

# ─────────────────────────────────────────────
# Resize Helper
# ─────────────────────────────────────────────

def resize_for_display(frame, max_w=DISPLAY_W, max_h=DISPLAY_H):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return frame

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

idx        = 0
auto_play  = True
delay_ms   = 100        # auto-play delay between frames

while 0 <= idx < len(image_files):

    img_path = image_files[idx]
    frame    = cv2.imread(str(img_path))

    if frame is None:
        print(f"⚠️  Could not read: {img_path.name}, skipping.")
        idx += 1
        continue

    # ── YOLO Inference ──────────────────────────────
    results   = model(frame, conf=CONF, imgsz=IMGSZ, verbose=False)
    boxes     = results[0].boxes
    det_count = len(boxes)
    annotated = results[0].plot()

    # ── Overlay Info ────────────────────────────────
    cv2.putText(annotated, f"File : {img_path.name}",
                (10, 35),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated, f"Frame: {idx + 1} / {len(image_files)}",
                (10, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated, f"Detections: {det_count}",
                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    mode_text = "AUTO" if auto_play else "MANUAL"
    cv2.putText(annotated, f"Mode: {mode_text}",
                (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ── Save Annotated Image ─────────────────────────
    if SAVE_OUTPUT:
        save_file = out_path / f"infer_{img_path.name}"
        cv2.imwrite(str(save_file), annotated)

    # ── Display ──────────────────────────────────────
    display = resize_for_display(annotated)
    cv2.imshow(window_name, display)

    print(f"[{idx+1:>4}/{len(image_files)}] {img_path.name}  |  Detections: {det_count}")

    # ── Key Handling ─────────────────────────────────
    wait = delay_ms if auto_play else 0   # 0 = wait indefinitely in manual mode
    key  = cv2.waitKey(wait) & 0xFF

    if key == ord("q"):
        print("👋 Quit.")
        break

    elif key == ord("a"):
        auto_play = not auto_play
        print(f"{'▶ Auto-play ON' if auto_play else '⏸ Manual mode'}")

    elif key in (ord(" "), ord("n")):   # next
        idx += 1

    elif key == ord("b"):               # back
        idx = max(0, idx - 1)

    elif key in (ord("+"), ord("=")):   # faster
        delay_ms = max(10, delay_ms - 20)
        print(f"⚡ Delay: {delay_ms}ms")

    elif key == ord("-"):               # slower
        delay_ms = min(2000, delay_ms + 20)
        print(f"🐢 Delay: {delay_ms}ms")

    else:
        if auto_play:
            idx += 1                    # advance automatically

# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────

cap_msg = f"✅ Done! Annotated images saved to: {OUTPUT_DIR}" if SAVE_OUTPUT else "✅ Program closed"
print(cap_msg)

cv2.destroyAllWindows()
