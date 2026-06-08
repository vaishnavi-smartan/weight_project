import cv2
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_PATH = "/home/pc-008/weight_project/DB_and_KB_Detetcion/scripts/runs/gym_equipment_v82/weights/best.pt"
VIDEO_PATH = "/home/pc-008/weight_project/infer2_1.mp4"
CONF       = 0.10
IMGSZ      = 640

DISPLAY_W  = 1280
DISPLAY_H  = 720

SEEK_SECS  = 5   # seconds to jump on fast-forward / rewind

# ─────────────────────────────────────────────
# Load model & open video
# ─────────────────────────────────────────────

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

fps        = cap.get(cv2.CAP_PROP_FPS) or 25
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
base_delay = int(1000 / fps)

speed_multiplier = 1.0
paused     = False
frame_count = 0

window_name = "YOLO Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, DISPLAY_W, DISPLAY_H)

# ─────────────────────────────────────────────
# Trackbar callback (seek)
# ─────────────────────────────────────────────

seeking = False   # flag so mouse-drag on trackbar isn't confused with button clicks

def on_trackbar(val):
    global frame_count, seeking
    seeking = True
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)
    frame_count = val

cv2.createTrackbar("Seek", window_name, 0, max(total_frames - 1, 1), on_trackbar)

# ─────────────────────────────────────────────
# Button definitions  (x, y, w, h, label, action)
# ─────────────────────────────────────────────

BUTTONS = [
    # label        action-key     x    y    w    h
    ("◀◀ -5s",    "rewind",      10,  DISPLAY_H - 60, 100, 44),
    ("⏮ -1f",     "prev_frame",  120, DISPLAY_H - 60,  90, 44),
    ("⏸ Pause",   "pause",       220, DISPLAY_H - 60, 110, 44),
    ("⏭ +1f",     "next_frame",  340, DISPLAY_H - 60,  90, 44),
    ("▶▶ +5s",    "forward",     440, DISPLAY_H - 60, 100, 44),
    ("🐢 -0.25x", "slower",      600, DISPLAY_H - 60, 110, 44),
    ("⚡ +0.25x", "faster",      720, DISPLAY_H - 60, 110, 44),
]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def resize_for_display(frame, max_w=DISPLAY_W, max_h=DISPLAY_H):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return frame


def draw_buttons(img, paused, speed):
    """Render button bar at the bottom of img (in-place)."""
    overlay = img.copy()
    for label, action, bx, by, bw, bh in BUTTONS:
        # Highlight pause button when paused
        if action == "pause":
            display_label = "▶ Resume" if paused else "⏸ Pause"
            color = (0, 180, 0) if paused else (40, 40, 200)
        else:
            display_label = label
            color = (30, 30, 30)

        # Button background (semi-transparent)
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, -1)
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)

        # Text (OpenCV doesn't support Unicode emoji, strip to ASCII-safe part)
        safe_label = display_label.encode("ascii", "ignore").decode()
        cv2.putText(overlay, safe_label, (bx + 6, by + bh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # Speed indicator
    cv2.putText(overlay, f"Speed: {speed:.2f}x",
                (DISPLAY_W - 180, DISPLAY_H - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Blend overlay
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)


def hit_button(mx, my):
    """Return action string if (mx,my) is inside a button, else None."""
    for _, action, bx, by, bw, bh in BUTTONS:
        if bx <= mx <= bx + bw and by <= my <= by + bh:
            return action
    return None


# ─────────────────────────────────────────────
# Mouse callback
# ─────────────────────────────────────────────

def on_mouse(event, mx, my, flags, param):
    global paused, speed_multiplier, frame_count, seeking

    if event == cv2.EVENT_LBUTTONUP:
        action = hit_button(mx, my)
        if action is None:
            return

        if action == "pause":
            paused = not paused

        elif action == "rewind":
            target = max(0, frame_count - int(SEEK_SECS * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            frame_count = target
            cv2.setTrackbarPos("Seek", window_name, frame_count)
            print(f"⏪ Rewound to frame {frame_count}")

        elif action == "forward":
            target = min(total_frames - 1, frame_count + int(SEEK_SECS * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            frame_count = target
            cv2.setTrackbarPos("Seek", window_name, frame_count)
            print(f"⏩ Fast-forwarded to frame {frame_count}")

        elif action == "prev_frame":
            target = max(0, frame_count - 2)   # -2 because read() will advance +1
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            frame_count = target
            paused = True   # freeze so user can step

        elif action == "next_frame":
            # just un-pause for one frame then re-pause
            paused = False
            seeking = True  # signal main loop to re-pause after one frame

        elif action == "slower":
            speed_multiplier = max(0.25, speed_multiplier - 0.25)
            print(f"🐢 Speed: {speed_multiplier:.2f}x")

        elif action == "faster":
            speed_multiplier = min(4.0, speed_multiplier + 0.25)
            print(f"⚡ Speed: {speed_multiplier:.2f}x")

cv2.setMouseCallback(window_name, on_mouse)

# ─────────────────────────────────────────────
# Print controls
# ─────────────────────────────────────────────

print("✅ Video started")
print("\nOn-screen buttons available at the bottom of the window.")
print("Keyboard shortcuts:")
print("  Q       → Quit")
print("  P       → Pause / Resume")
print("  + / =   → Speed +0.25x")
print("  -       → Speed -0.25x")
print("  F       → Fast-forward 5 s")
print("  R       → Rewind 5 s")
print("  RIGHT   → Next frame (step)")
print("  LEFT    → Previous frame (step)")

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

step_mode = False   # True when user steps frame-by-frame

while True:

    # ── Pause / step-mode wait ──────────────────
    if paused and not seeking:
        # Keep refreshing so buttons remain responsive
        key = cv2.waitKey(50) & 0xFF
        if   key == ord("p"):            paused = not paused
        elif key == ord("q"):            break
        elif key == 83 or key == ord("f"):  # RIGHT arrow or F → step forward
            target = min(total_frames - 1, frame_count)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            paused = False
            seeking = True  # re-pause after one frame
        elif key == 81 or key == ord("r"):  # LEFT arrow or R → step back
            target = max(0, frame_count - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            paused = False
            seeking = True
        continue

    # ── Read frame ─────────────────────────────
    ret, frame = cap.read()
    if not ret:
        print("✅ Video finished")
        break

    frame_count += 1
    seeking_was = seeking
    seeking = False

    # ── YOLO inference ──────────────────────────
    results  = model(frame, conf=CONF, imgsz=IMGSZ, verbose=False)
    boxes    = results[0].boxes
    det_count = len(boxes)
    annotated = results[0].plot()

    # ── Overlay text ────────────────────────────
    cv2.putText(annotated, f"Frame: {frame_count}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(annotated, f"Detections: {det_count}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # ── Resize to display ───────────────────────
    display = resize_for_display(annotated)

    # Pad to full DISPLAY_H so buttons sit in a fixed zone
    h, w = display.shape[:2]
    if h < DISPLAY_H:
        pad = np.zeros((DISPLAY_H - h, w, 3), dtype=np.uint8)
        display = np.vstack([display, pad])
    if w < DISPLAY_W:
        pad = np.zeros((display.shape[0], DISPLAY_W - w, 3), dtype=np.uint8)
        display = np.hstack([display, pad])

    # ── Draw button bar ──────────────────────────
    draw_buttons(display, paused, speed_multiplier)

    # ── Update seek trackbar ─────────────────────
    cv2.setTrackbarPos("Seek", window_name,
                       min(frame_count, total_frames - 1))

    # ── Show ────────────────────────────────────
    cv2.imshow(window_name, display)

    # Re-pause after a single-step
    if seeking_was:
        paused = True

    # ── Delay / keyboard ────────────────────────
    delay = max(1, int(base_delay / speed_multiplier))
    key   = cv2.waitKey(delay) & 0xFF

    if   key == ord("q"):  break
    elif key == ord("p"):  paused = not paused

    elif key in (ord("+"), ord("=")):
        speed_multiplier = min(speed_multiplier + 0.25, 4.0)
        print(f"⚡ Speed: {speed_multiplier:.2f}x")

    elif key == ord("-"):
        speed_multiplier = max(speed_multiplier - 0.25, 0.25)
        print(f"🐢 Speed: {speed_multiplier:.2f}x")

    elif key == ord("f") or key == 83:   # F or RIGHT arrow → fast-forward 5 s
        target = min(total_frames - 1, frame_count + int(SEEK_SECS * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        frame_count = target
        print(f"⏩ Jumped to frame {frame_count}")

    elif key == ord("r") or key == 81:   # R or LEFT arrow → rewind 5 s
        target = max(0, frame_count - int(SEEK_SECS * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        frame_count = target
        print(f"⏪ Jumped to frame {frame_count}")

# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────

cap.release()
cv2.destroyAllWindows()
print("✅ Program closed")
