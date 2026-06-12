import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_PATH  = "/home/pc-008/weight_project/DB_and_KB_Detetcion/scripts/runs/gym_equipment_v8n_gpu-6/weights/best.pt"
TESTSET_DIR = "/home/pc-008/weight_project/DB_and_KB_Detetcion/testset/"
OUTPUT_DIR  = "/home/pc-008/weight_project/infer_out/"

CONF   = 0.10
IMGSZ  = 640
DEVICE = 0        # 0 = first GPU; use "cpu" to force CPU

SAVE_ANNOTATED = True

# Folder name aliases — maps your folder names to model class names
FOLDER_ALIASES = {
    "mb": "medicine ball",
}

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────

out_path = Path(OUTPUT_DIR)
out_path.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)
model.to(f"cuda:{DEVICE}")

# active classes: strip swiss ball entirely at class-map level
active = {k: v for k, v in model.names.items() if v.lower() != "swiss ball"}

orig_ids_sorted = sorted(active.keys())
orig_to_cm      = {orig: new for new, orig in enumerate(orig_ids_sorted)}
cm_to_name      = {new: active[orig] for orig, new in orig_to_cm.items()}
name_to_cm      = {v.lower(): k for k, v in cm_to_name.items()}

# excl_orig_ids used to filter predictions
excl_orig_ids = {k for k, v in model.names.items() if v.lower() == "swiss ball"}

for alias, real_name in FOLDER_ALIASES.items():
    if real_name.lower() in name_to_cm:
        name_to_cm[alias.lower()] = name_to_cm[real_name.lower()]

num_classes = len(cm_to_name)
BG          = num_classes

print(f"✅ Model loaded on GPU:{DEVICE}  |  {num_classes} classes:")
for i, name in cm_to_name.items():
    print(f"   [{i}] {name}")
print()

# ─────────────────────────────────────────────
# Collect images — folder name = GT class
# ─────────────────────────────────────────────

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
testset_path  = Path(TESTSET_DIR)

dataset         = []
skipped_folders = []

for folder in sorted(testset_path.iterdir()):
    if not folder.is_dir():
        continue
    folder_cls = folder.name.lower()
    if folder_cls not in name_to_cm:
        skipped_folders.append(folder.name)
        continue
    gt_cm_id = name_to_cm[folder_cls]
    images   = sorted([f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXT])
    for img in images:
        dataset.append((img, gt_cm_id))

if skipped_folders:
    print(f"⚠️  Skipped folders (no class match): {skipped_folders}")

print(f"✅ {len(dataset)} images found\n")

if not dataset:
    print("❌ No images found. Check TESTSET_DIR and folder names.")
    exit()

# ─────────────────────────────────────────────
# Confusion matrix  (num_classes+1) x (num_classes+1)
# ─────────────────────────────────────────────

cm = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

# ─────────────────────────────────────────────
# Inference loop
# ─────────────────────────────────────────────

total = len(dataset)

for idx, (img_path, gt_cm_id) in enumerate(dataset, 1):

    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  ⚠️  Skipped (unreadable): {img_path.name}")
        cm[gt_cm_id][BG] += 1
        continue

    results = model(frame, conf=CONF, imgsz=IMGSZ, device=DEVICE, verbose=False)
    boxes   = results[0].boxes

    pred_cm_id = BG
    pred_conf  = 0.0

    if len(boxes) > 0:
        confs   = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        valid = [(c, f) for c, f in zip(classes, confs) if c not in excl_orig_ids]

        if valid:
            best_orig_id, pred_conf = max(valid, key=lambda x: x[1])
            pred_cm_id = orig_to_cm[best_orig_id]

    cm[gt_cm_id][pred_cm_id] += 1

    if SAVE_ANNOTATED:
        annotated = results[0].plot()
        save_dir  = out_path / cm_to_name[gt_cm_id]
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / f"infer_{img_path.name}"), annotated)

    gt_name   = cm_to_name[gt_cm_id]
    pred_name = cm_to_name[pred_cm_id] if pred_cm_id != BG else "no_detection"
    status    = "✓" if pred_cm_id == gt_cm_id else "✗"
    print(f"  [{idx:>4}/{total}]  {status}  GT: {gt_name:<15}  "
          f"Pred: {pred_name:<15}  conf: {pred_conf:.2f}  | {img_path.name}")

# ─────────────────────────────────────────────
# Per-class stats
# ─────────────────────────────────────────────

print(f"\n{'─'*70}")
print(f"{'Class':<18} {'Total':>6} {'TP':>5} {'FP':>5} {'FN':>5} "
      f"{'Prec':>8} {'Rec':>8} {'F1':>8}")
print(f"{'─'*70}")

for i, name in cm_to_name.items():
    total_gt = int(cm[i].sum())
    tp = int(cm[i, i])
    fn = int(cm[i].sum()) - tp
    fp = int(cm[:, i].sum()) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"{name:<18} {total_gt:>6} {tp:>5} {fp:>5} {fn:>5} "
          f"{prec:>8.3f} {rec:>8.3f} {f1:>8.3f}")

print(f"{'─'*70}\n")

# ─────────────────────────────────────────────
# Confusion matrix plot
# ─────────────────────────────────────────────

labels   = [cm_to_name[i] for i in range(num_classes)] + ["no_detection"]
row_sums = cm.sum(axis=1, keepdims=True).astype(float)
row_sums[row_sums == 0] = 1
cm_norm  = cm / row_sums

fig, axes = plt.subplots(1, 2, figsize=(max(14, len(labels) * 1.8),
                                         max(6,  len(labels) * 1.2)))

for ax, data, fmt, title, vmax in [
    (axes[0], cm,      "d",    "Confusion Matrix — Counts",              None),
    (axes[1], cm_norm, ".2f",  "Confusion Matrix — Normalised (row=GT)", 1.0),
]:
    sns.heatmap(
        data, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="gray",
        annot_kws={"size": 11, "weight": "bold"},
        vmin=0, vmax=vmax, ax=ax
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted",   fontsize=11)
    ax.set_ylabel("Actual (GT)", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

plt.tight_layout()
cm_save = out_path / "confusion_matrix.png"
plt.savefig(cm_save, dpi=150, bbox_inches="tight")
plt.close()

print(f"📊 Confusion matrix saved → {cm_save}")
if SAVE_ANNOTATED:
    print(f"🖼️  Annotated images  saved → {OUTPUT_DIR}")
print("✅ Done!")