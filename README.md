# Gym Equipment Detection — YOLOv8 Pipeline

End-to-end object detection pipeline for gym equipment using YOLOv8.  
Detects 5 classes: **BB** (barbell), **DB** (dumbbell), **KB** (kettlebell), **MB** (medicine ball), **Plates**.

---

## Project Structure

```
.
├── object_detection.py           # Model training with MLflow tracking
├── infer_video.py        # Real-time video inference with interactive playback controls
├── extract_frames.py     # Extract frames from video to disk
├── sort_frames.py        # Sort extracted frames into per-class folders via YOLO inference
├── frame_counter.py      # Count per-class detections across all frames of a video
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone / navigate to project directory
cd weight_project

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Verify GPU availability
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Scripts

### 1. `object_detection.py` — Model Training

Trains a YOLOv8 model on the Roboflow *Dumbells and Kettlebells v15* dataset with full MLflow experiment tracking.

**Key features:**
- Dropout injection into the Detect head (`p=0.10`) and backbone (`p=0.05`) for regularisation
- Per-epoch MLflow logging: losses, LR, per-class AP50, mAP50, precision, recall
- Final test-split evaluation with per-class AP50 breakdown
- Class distribution analysis and imbalance warnings at startup
- Automatic unique run naming with timestamps

**Usage:**

```bash
# Basic run (150 epochs, yolov8n.pt, batch=16)
python train_v8.py

# Custom model and epoch count
python object_detection.py --model yolov8n.pt --epochs 150 --batch 16

# Point to a specific MLflow database
python object_detection.py --mlflow-uri sqlite:////home/pc-008/weight_project/mlflow.db

# Resume an interrupted run
python object_detection.py --resume

# Launch MLflow UI after training
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Then open: http://localhost:5000
```



**All CLI arguments:**

| Argument              | Default                       | Description                          |
|-----------------------|-------------------------------|--------------------------------------|
| `--model`             | `yolov8n.pt`                  | Base YOLO model weights              |
| `--epochs`            | `150`                         | Training epochs                      |
| `--imgsz`             | `640`                         | Input image size                     |
| `--batch`             | `16`                          | Batch size                           |
| `--device`            | `0`                           | GPU device ID (`cpu` for CPU-only)   |
| `--workers`           | `4`                           | Dataloader workers                   |
| `--patience`          | `45`                          | Early stopping patience              |
| `--resume`            | `False`                       | Resume from last checkpoint          |
| `--mlflow-uri`        | `sqlite:///mlflow.db`         | MLflow tracking URI                  |
| `--mlflow-experiment` | `gym_equipment_detection`     | MLflow experiment name               |
| `--mlflow-run-name`   | `v8_<timestamp>`              | MLflow run display name              |
| `--head-dropout`      | `0.10`                        | Dropout prob for Detect head         |
| `--backbone-dropout`  | `0.05`                        | Dropout prob for backbone layers     |

---

### 2. `infer_video.py` — Interactive Video Inference

Runs real-time YOLO inference on a video with an interactive OpenCV player.

**Usage:**

```bash
# Edit MODEL_PATH and VIDEO_PATH at the top of the script, then:
python infer_video.py
```

**Controls:**

| Button / Key        | Action                         |
|---------------------|--------------------------------|
| `P` / ⏸ Pause       | Pause / Resume                 |
| `F` / `→` / ▶▶ +5s  | Fast-forward 5 seconds         |
| `R` / `←` / ◀◀ -5s  | Rewind 5 seconds               |
| ⏭ +1f               | Step forward one frame         |
| ⏮ -1f               | Step back one frame            |
| `+` / ⚡ +0.25x     | Increase playback speed        |
| `-` / 🐢 -0.25x     | Decrease playback speed        |
| Seek trackbar       | Jump to any frame              |
| `Q`                 | Quit                           |

**Config (top of script):**

```python
MODEL_PATH  = "runs/gym_equipment_v8/weights/best.pt"
VIDEO_PATH  = "infer.mp4"
CONF        = 0.10    # detection confidence threshold
IMGSZ       = 640
DISPLAY_W   = 1280
DISPLAY_H   = 720
SEEK_SECS   = 5       # seconds to jump on fast-forward/rewind
```

---

### 3. `extract_frames.py` — Video Frame Extractor-- for test set accuracy

Splits a video into individual image frames saved to disk.

**Usage:**

```bash
# Extract all frames (JPG, 95% quality)
python extract_frames.py --video input.mp4

# Extract at 1 frame per second, save as PNG
python extract_frames.py --video input.mp4 --fps 1 --format png

# Extract a specific time window
python extract_frames.py --video input.mp4 --start 10 --end 30

# Custom output directory and prefix
python extract_frames.py --video input.mp4 --output frames/ --prefix gym
```

**All CLI arguments:**

| Argument    | Default   | Description                                      |
|-------------|-----------|--------------------------------------------------|
| `--video`   | required  | Path to input video file                         |
| `--output`  | `frames/` | Output directory                                 |
| `--format`  | `jpg`     | Image format: `jpg`, `png`, `bmp`                |
| `--every`   | `1`       | Save every N-th frame                            |
| `--fps`     | `None`    | Target frames-per-second (overrides `--every`)   |
| `--quality` | `95`      | JPEG quality 1–100                               |
| `--start`   | `None`    | Start extraction at this timestamp (seconds)     |
| `--end`     | `None`    | Stop extraction at this timestamp (seconds)      |
| `--prefix`  | `frame`   | Filename prefix for saved images                 |

---

### 4. `sort_frames.py` — YOLO Frame Sorter

Runs inference on a folder of images and sorts them into per-class subfolders.

**Output structure:**

```
sorted_frames/
├── bb/               # Frames where BB was detected
├── db/               # Frames where DB was detected
├── kb/               # Frames where KB was detected
├── medicine_ball/    # Frames where MB was detected
├── plates/           # Frames where Plates were detected
├── multi/            # Frames with multiple different classes
└── no_detection/     # Frames with no detections
```

**Usage:**

```bash
# Edit the config block at the top of the script, then:
python sort_frames.py
```

**Config (top of script):**

```python
MODEL_PATH     = "runs/gym_equipment_v8/weights/best.pt"
FRAMES_DIR     = "frames/"           # input folder from extract_frames.py
OUTPUT_DIR     = "sorted_frames/"    # output root
CONF           = 0.10
IMGSZ          = 640
SAVE_ANNOTATED = True                # True = save with bounding boxes drawn
```

> **Note:** When multiple classes appear in the same frame, the image is copied into **each** matching class folder AND the `multi/` folder.

---











### 5. `frame_counter.py` — Video Class Frame Counter

Runs YOLO inference on every frame of a video and reports how many frames each class appears in.

**Usage:**

```bash
# Edit MODEL_PATH and VIDEO_PATH at the top of the script, then:
python frame_counter.py
```

**Sample output:**

```
══════════════════════════════════════════
  CLASS-WISE FRAME COUNT REPORT
══════════════════════════════════════════
┌─────────────────────┬────────┬─────────┐
│ Class               │ Frames │ % Video │
├─────────────────────┼────────┼─────────┤
│ db                  │  210   │  35.2%  │
│ kb                  │  180   │  30.2%  │
│ bb                  │   95   │  15.9%  │
│ medicine ball       │   62   │  10.4%  │
│ plates              │   30   │   5.0%  │
│ no_detection        │   19   │   3.2%  │
└─────────────────────┴────────┴─────────┘
```

A `frame_count_report.txt` is automatically saved next to the input video.

**Config (top of script):**

```python
MODEL_PATH     = "runs/gym_equipment_v8/weights/best.pt"
VIDEO_PATH     = "infer.mp4"
CONF           = 0.10
IMGSZ          = 640
EVERY_N_FRAMES = 1     # set > 1 to process faster (e.g. 5 = every 5th frame)
```

---

## Typical Workflow

```
Video input
    │
    ▼
extract_frames.py   ──→  frames/frame_000001.jpg ...
    │
    ▼
sort_frames.py      ──→  sorted_frames/kb/ , db/ , plates/ ...
    │
    ▼  (review per-class folders to check annotation quality)
    │
train_v8.py         ──→  runs/gym_equipment_v8/weights/best.pt
    │                     + MLflow run logged to mlflow.db
    ▼
infer_video.py      ──→  Interactive playback with live detections
    │
    ▼
frame_counter.py    ──→  frame_count_report.txt
```

---

## Classes & Dataset

| ID | Class         | Notes                                      |
|----|---------------|--------------------------------------------|
| 0  | bb            | Barbell                                    |
| 1  | db            | Dumbbell                                   |
| 2  | kb            | Kettlebell                                 |
| 3  | medicine ball | Often confused with plates — monitor AP50  |
| 4  | plates        | ⚠ Lowest sample count (~52 val instances)  |

**Dataset:** Roboflow — *Dumbells and Kettlebells v15*  
**Total label instances:** ~1,776 across all classes

> The `plates` class is critically underrepresented. Training uses `copy_paste=0.6` and `erasing=0.3` to compensate. Aim to collect ≥200 plates images for best results.

---

## MLflow Tracking

Every training run logs:

- **Parameters:** model, epochs, batch, augmentation config, dropout values, class counts, imbalance ratios
- **Per-epoch metrics:** train loss (box/cls/dfl), LR, val mAP50, precision, recall, per-class AP50
- **Test metrics:** mAP50, mAP50-95, precision, recall, per-class AP50 and AP50-95
- **Artifacts:** `best.pt`, `last.pt`, `results.png`, `confusion_matrix.png`, PR/F1 curves, `args.yaml`

```bash
# View all runs in the browser
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

---

## Notes

- **CPU-only training:** pass `--device cpu` to `train_v8.py`. Expect significantly longer run times.
- **Dropout behaviour:** dropout is automatically disabled during validation/inference via `model.eval()` — no manual toggling needed.
- The built-in Ultralytics MLflow callbacks are stripped and replaced with custom ones to avoid duplicate logging and schema conflicts.
