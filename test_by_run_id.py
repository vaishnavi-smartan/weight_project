"""
Run-ID Driven Testset Evaluation
=================================
Pass the training run ID (e.g. "38" for gym_equipment_v8n_weighted-38),
and this script automatically locates that run's best.pt, runs inference
on the fixed test set, and produces the same per-class summary +
sklearn classification report + charts as test_report1.py.

Usage:
    python test_by_run_id.py --run_id 38
    python test_by_run_id.py --run_id 38 --conf 0.35 --iou 0.5
    python test_by_run_id.py --model_path /custom/path/to/best.pt   # bypass run_id lookup

Testset structure expected (fixed, same every run):
    TESTSET_DIR/
    ├── bb/
    ├── db/
    ├── kb/
    ├── mb/
    └── plates/

Output structure created (one folder per run, so runs never overwrite each other):
    OUTPUT_ROOT/run_<run_id>/
    ├── bb/ db/ kb/ mb/ plates/     ← annotated detections
    ├── missed/<class>/             ← 0-detection frames
    ├── mis_detection/<class>/      ← top box = wrong class
    ├── charts/
    └── inference_report.txt
"""

import argparse
import cv2
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from ultralytics import YOLO
import mlflow
from mlflow.tracking import MlflowClient

# ─────────────────────────────────────────────────────────────
# FIXED PATHS — change these once, reuse for every run_id
# ─────────────────────────────────────────────────────────────
RUNS_BASE_DIR = "/home/pc-008/weight_project/DB_and_KB_Detetcion/scripts/runs"
RUN_NAME_TEMPLATE = "gym_equipment_v8n_weighted-{run_id}"   # folder name pattern
MLFLOW_TRACKING_URI = "sqlite:////home/pc-008/weight_project/mlflow.db"

# Test-set evaluation results are logged to their OWN experiment, kept
# separate from whatever experiment the training runs themselves live in.
TEST_EXPERIMENT_NAME = "gym_equipment_testset_evaluation"

TESTSET_DIR = "/home/pc-008/weight_project/DB_and_KB_Detetcion/equipments_infer"
OUTPUT_ROOT = "/home/pc-008/weight_project/infer_report_new"

DEFAULT_CONF_THRESH = 0.40
DEFAULT_IOU_THRESH  = 0.45
IMG_SIZE            = 640

SAVE_ALL       = False   # True → save every annotated image, not just detections
SAVE_MISSED    = True
SAVE_MISDETECT = True

CLASS_FOLDERS = {
    "bb":     "bb",
    "db":     "db",
    "kb":     "kb",
    "mb":     "mb",
    "plates": "plates",
}
ALL_LABELS = list(CLASS_FOLDERS.values()) + ["no_detection"]

PALETTE = {
    "bb":            "#4C72B0",
    "db":            "#DD8452",
    "kb":            "#55A868",
    "mb":            "#C44E52",
    "plates":        "#8172B3",
    "no_detection":  "#AAAAAA",
}
DEFAULT_COLOR = "#999999"
# ─────────────────────────────────────────────────────────────


def _color(cls_name):
    return PALETTE.get(cls_name, DEFAULT_COLOR)


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("_", "").replace("-", "")


_NORM_TO_DISPLAY = {}
for _folder_name, _display_name in CLASS_FOLDERS.items():
    _NORM_TO_DISPLAY[_normalize(_folder_name)]  = _display_name
    _NORM_TO_DISPLAY[_normalize(_display_name)] = _display_name

_NORM_TO_DISPLAY[_normalize("medicine ball")] = "mb"
_NORM_TO_DISPLAY[_normalize("medicineball")]  = "mb"
_NORM_TO_DISPLAY[_normalize("dumbbell")]      = "db"
_NORM_TO_DISPLAY[_normalize("dumbbells")]     = "db"
_NORM_TO_DISPLAY[_normalize("kettlebell")]    = "kb"
_NORM_TO_DISPLAY[_normalize("kettlebells")]   = "kb"
_NORM_TO_DISPLAY[_normalize("barbell")]       = "bb"
_NORM_TO_DISPLAY[_normalize("plate")]         = "plates"


def to_display(raw_name: str) -> str:
    mapped = _NORM_TO_DISPLAY.get(_normalize(raw_name))
    if mapped is None:
        print(f"  ⚠ WARNING: unrecognized model class name '{raw_name}' — "
              f"not in CLASS_FOLDERS or alias list. Add it to _NORM_TO_DISPLAY.")
        return raw_name
    return mapped


def _looks_like_mlflow_run_id(s: str) -> bool:
    """MLflow run IDs are 32-character lowercase hex strings, e.g.
    '6eef36523c5d491aa35f596e15f9a3cf'. The numeric folder suffixes
    ('38', '5', etc.) never match this pattern."""
    return len(s) == 32 and all(c in "0123456789abcdef" for c in s.lower())


def resolve_run_folder_name_from_mlflow(run_id: str) -> str:
    """Look up an MLflow run ID in mlflow.db and figure out which folder
    under RUNS_BASE_DIR it corresponds to.

    The mlflow.runName tag isn't always reliable — some runs were logged
    with a timestamp-based name (e.g. 'v8n_weighted_20260707_232341') that
    doesn't match the actual training-script folder naming on disk
    ('gym_equipment_v8n_weighted-38'). So instead of trusting runName
    blindly, this searches every param, tag, and the artifact_uri for
    anything matching the real folder pattern.
    """
    import re

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    try:
        run = client.get_run(run_id)
    except Exception as e:
        raise ValueError(
            f"Could not find MLflow run '{run_id}' in {MLFLOW_TRACKING_URI}: {e}"
        )

    folder_pattern = re.compile(r"gym_equipment_v8n_weighted-\d+")

    # Collect every string value we have on this run: params, tags, artifact_uri
    candidates = []
    candidates.extend(run.data.params.values())
    candidates.extend(run.data.tags.values())
    if run.info.artifact_uri:
        candidates.append(run.info.artifact_uri)

    for value in candidates:
        match = folder_pattern.search(str(value))
        if match:
            return match.group(0)

    # Nothing matched — dump everything so it's obvious what IS available,
    # rather than failing with no context.
    dump_lines = ["Could not find a 'gym_equipment_v8n_weighted-N' folder reference "
                  f"anywhere in MLflow run '{run_id}'.\n",
                  f"  run.info.run_name : {run.info.run_name}",
                  f"  run.info.artifact_uri : {run.info.artifact_uri}",
                  "  Params:"]
    for k, v in run.data.params.items():
        dump_lines.append(f"    {k} = {v}")
    dump_lines.append("  Tags:")
    for k, v in run.data.tags.items():
        dump_lines.append(f"    {k} = {v}")
    dump_lines.append(
        "\n  → Check the values above for whichever one references the actual "
        "run folder, then either rename that run's folder to match, or pass "
        "the folder name directly with --model_path."
    )
    raise ValueError("\n".join(dump_lines))


def get_training_run_description(run_id_arg: str, run_folder_name: str) -> str:
    """Fetch the description/note already logged on the TRAINING run,
    so you never have to retype it by hand for the test-set evaluation.

    Strategy:
      1. If --run_id was itself an MLflow hash, just read that run's note directly.
      2. Otherwise (numeric folder suffix like '38'), search every experiment
         for a run whose params/tags/artifact_uri reference this folder name,
         and read that run's note.

    Returns "" if nothing is found — the caller falls back to no description
    rather than failing, since a missing description shouldn't block testing.
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Case 1: run_id_arg IS the MLflow run — read its note directly.
    if run_id_arg and _looks_like_mlflow_run_id(run_id_arg):
        try:
            run = client.get_run(run_id_arg)
            note = run.data.tags.get("mlflow.note.content", "")
            if note:
                return note
        except Exception:
            pass  # fall through to the folder-name search below

    # Case 2: search all experiments for a run referencing this folder name.
    try:
        experiments = client.search_experiments()
    except AttributeError:
        experiments = client.list_experiments()  # older mlflow versions

    for exp in experiments:
        try:
            runs = client.search_runs([exp.experiment_id], max_results=5000)
        except Exception:
            continue
        for run in runs:
            candidates = list(run.data.params.values()) + list(run.data.tags.values())
            if run.info.artifact_uri:
                candidates.append(run.info.artifact_uri)
            if any(run_folder_name in str(v) for v in candidates):
                note = run.data.tags.get("mlflow.note.content", "")
                if note:
                    return note

    return ""


def resolve_model_path(args) -> Path:
    """Find best.pt either from an explicit --model_path or from --run_id.
    --run_id accepts either the numeric folder suffix ('38') or the full
    MLflow run ID (32-char hex) — the latter is resolved via mlflow.db."""
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"--model_path given but not found: {model_path}")
        return model_path

    if not args.run_id:
        raise ValueError("You must pass either --run_id or --model_path")

    if _looks_like_mlflow_run_id(args.run_id):
        print(f"  '{args.run_id}' looks like an MLflow run ID — resolving run name via mlflow.db...")
        run_folder_name = resolve_run_folder_name_from_mlflow(args.run_id)
        print(f"  → Resolved to run folder: {run_folder_name}")
        run_folder = Path(RUNS_BASE_DIR) / run_folder_name
    else:
        run_folder = Path(RUNS_BASE_DIR) / RUN_NAME_TEMPLATE.format(run_id=args.run_id)

    model_path = run_folder / "weights" / "best.pt"

    if not model_path.exists():
        # Helpful fallback: list what run folders DO exist so a typo is obvious
        available = sorted(p.name for p in Path(RUNS_BASE_DIR).glob("gym_equipment_v8n_weighted-*"))
        raise FileNotFoundError(
            f"best.pt not found at expected path:\n  {model_path}\n\n"
            f"Available run folders in {RUNS_BASE_DIR}:\n  " +
            "\n  ".join(available) if available else "  (none found)"
        )
    return model_path


# ═════════════════════════════════════════════════════════════
#  CHARTS
# ═════════════════════════════════════════════════════════════

def plot_classification_metrics(report_dict: dict, classes: list, charts_dir: Path):
    prec = [report_dict[c]["precision"] for c in classes]
    rec  = [report_dict[c]["recall"]    for c in classes]
    f1   = [report_dict[c]["f1-score"]  for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(9, len(classes) * 1.8), 5.5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    bars_p = ax.bar(x - width, prec, width, label="Precision", color="#4C72B0", alpha=0.88, edgecolor="white", linewidth=0.8)
    bars_r = ax.bar(x,         rec,  width, label="Recall",    color="#55A868", alpha=0.88, edgecolor="white", linewidth=0.8)
    bars_f = ax.bar(x + width, f1,   width, label="F1-Score",  color="#C44E52", alpha=0.88, edgecolor="white", linewidth=0.8)

    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Classification Report — Precision / Recall / F1 per Class",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=10, framealpha=0.7)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = charts_dir / "classification_report.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out.name}")
    return out


def plot_confusion_matrix(y_true: list, y_pred: list, labels: list, charts_dir: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm /= row_sums

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(labels) * 2.2),
                                             max(5,  len(labels) * 1.4)))
    fig.patch.set_facecolor("#F8F9FA")

    for ax, data, fmt, title, vmax in [
        (axes[0], cm,      "d",   "Confusion Matrix — Counts",     None),
        (axes[1], cm_norm, ".2f", "Confusion Matrix — Normalised", 1.0),
    ]:
        ax.set_facecolor("#F8F9FA")
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, linecolor="#CCCCCC",
                    annot_kws={"size": 10, "weight": "bold"},
                    vmin=0, vmax=vmax, ax=ax, cbar=True)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True (GT)", fontsize=10)
        ax.tick_params(axis="x", rotation=40, labelsize=9)
        ax.tick_params(axis="y", rotation=0,  labelsize=9)

    fig.suptitle("Confusion Matrix Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = charts_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out.name}")
    return out


def plot_detection_accuracy(class_stats: dict, charts_dir: Path):
    classes  = list(class_stats.keys())
    gt_vals  = [class_stats[c]["total"]      for c in classes]
    det_vals = [class_stats[c]["n_detected"] for c in classes]
    acc_vals = [
        min(class_stats[c]["n_detected"], class_stats[c]["total"]) / class_stats[c]["total"] * 100
        if class_stats[c]["total"] > 0 else 0
        for c in classes
    ]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(9, len(classes) * 1.8), 5.5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    bars_gt  = ax.bar(x - width/2, gt_vals,  width, label="Ground Truth",
                      color="#B0BEC5", edgecolor="white", linewidth=0.8)
    bars_det = ax.bar(x + width/2, det_vals, width, label="Detected",
                      color=[_color(c) for c in classes], edgecolor="white",
                      linewidth=0.8, alpha=0.9)

    for i, (g, d, a) in enumerate(zip(gt_vals, det_vals, acc_vals)):
        top = max(g, d) + 0.5
        color = "#1A5C1A" if a >= 90 else "#BA7517" if a >= 75 else "#AA0000"
        ax.text(i, top + ax.get_ylim()[1] * 0.01, f"{a:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=color)

    for bar in bars_gt:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, str(int(h)),
                ha="center", va="bottom", fontsize=8.5, color="#555555")
    for bar in bars_det:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, str(int(h)),
                ha="center", va="bottom", fontsize=8.5, color="#333333", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel("Frame Count", fontsize=11)
    ax.set_title("Detection Accuracy — Detected vs Ground Truth per Class",
                 fontsize=13, fontweight="bold", pad=14)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    patches = [
        mpatches.Patch(color="#1A5C1A", label="≥ 90% accurate"),
        mpatches.Patch(color="#BA7517", label="75–90%"),
        mpatches.Patch(color="#AA0000", label="< 75%"),
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + patches,
              labels=ax.get_legend_handles_labels()[1] + [p.get_label() for p in patches],
              fontsize=9, framealpha=0.7, loc="upper right")

    plt.tight_layout()
    out = charts_dir / "detection_accuracy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out.name}")
    return out


def plot_summary_dashboard(report_dict, class_stats, y_true, y_pred, labels, overall_acc, charts_dir, conf_thresh):
    classes = [c for c in labels if c != "no_detection"]

    fig = plt.figure(figsize=(20, 14), facecolor="#F0F2F5")
    fig.suptitle(
        f"Gym Equipment Detection — Inference Dashboard\n"
        f"Overall Accuracy: {overall_acc*100:.1f}%   |   "
        f"Classes: {len(classes)}   |   Conf ≥ {conf_thresh}",
        fontsize=15, fontweight="bold", y=0.98,
    )

    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.92, bottom=0.07)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor("#F8F9FA")
    prec = [report_dict[c]["precision"] for c in classes]
    rec  = [report_dict[c]["recall"]    for c in classes]
    f1   = [report_dict[c]["f1-score"]  for c in classes]
    x = np.arange(len(classes)); w = 0.25
    ax0.bar(x - w, prec, w, label="Precision", color="#4C72B0", alpha=0.88, edgecolor="white")
    ax0.bar(x,     rec,  w, label="Recall",    color="#55A868", alpha=0.88, edgecolor="white")
    ax0.bar(x + w, f1,   w, label="F1-Score",  color="#C44E52", alpha=0.88, edgecolor="white")
    for bars in [ax0.containers[0], ax0.containers[1], ax0.containers[2]]:
        ax0.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5, fontweight="bold")
    ax0.set_xticks(x); ax0.set_xticklabels(classes, fontsize=9)
    ax0.set_ylim(0, 1.18)
    ax0.set_title("Precision / Recall / F1", fontsize=11, fontweight="bold")
    ax0.legend(fontsize=8); ax0.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax0.set_axisbelow(True); ax0.spines[["top","right"]].set_visible(False)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor("#F8F9FA")
    cm_full = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm_full.astype(float)
    rs = cm_norm.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    cm_norm /= rs
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.4, linecolor="#CCCCCC",
                annot_kws={"size": 9, "weight": "bold"},
                vmin=0, vmax=1.0, ax=ax1, cbar=True)
    ax1.set_title("Confusion Matrix (Normalised)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Predicted", fontsize=9)
    ax1.set_ylabel("True (GT)", fontsize=9)
    ax1.tick_params(axis="x", rotation=35, labelsize=8)
    ax1.tick_params(axis="y", rotation=0,  labelsize=8)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#F8F9FA")
    gt_vals  = [class_stats[c]["total"]      for c in classes]
    det_vals = [class_stats[c]["n_detected"] for c in classes]
    x2 = np.arange(len(classes))
    ax2.bar(x2 - 0.2, gt_vals,  0.38, label="Ground Truth", color="#B0BEC5", edgecolor="white", linewidth=0.7)
    ax2.bar(x2 + 0.2, det_vals, 0.38, label="Detected",
            color=[_color(c) for c in classes], edgecolor="white", linewidth=0.7, alpha=0.9)
    for i, (g, d) in enumerate(zip(gt_vals, det_vals)):
        acc_i = min(g, d) / g * 100 if g > 0 else 0
        col = "#1A5C1A" if acc_i >= 90 else "#BA7517" if acc_i >= 75 else "#AA0000"
        ax2.text(i, max(g, d) + max(gt_vals) * 0.03, f"{acc_i:.0f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color=col)
    ax2.set_xticks(x2); ax2.set_xticklabels(classes, fontsize=9)
    ax2.set_title("Detected vs Ground Truth", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Frame Count", fontsize=9)
    ax2.legend(fontsize=8); ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True); ax2.spines[["top","right"]].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#F8F9FA")
    acc_vals = [
        min(class_stats[c]["n_detected"], class_stats[c]["total"]) / class_stats[c]["total"] * 100
        if class_stats[c]["total"] > 0 else 0
        for c in classes
    ]
    f1_vals = [report_dict[c]["f1-score"] * 100 for c in classes]
    y_pos = np.arange(len(classes))

    bars_a = ax3.barh(y_pos + 0.2, acc_vals, 0.35, label="Detection Acc %",
                      color=[_color(c) for c in classes], alpha=0.85, edgecolor="white")
    bars_f = ax3.barh(y_pos - 0.2, f1_vals,  0.35, label="F1 Score × 100",
                      color=[_color(c) for c in classes], alpha=0.45, edgecolor="white", hatch="//")

    for bar in bars_a:
        w2 = bar.get_width()
        ax3.text(w2 + 0.5, bar.get_y() + bar.get_height()/2, f"{w2:.1f}%",
                 va="center", fontsize=8.5, fontweight="bold")
    for bar in bars_f:
        w2 = bar.get_width()
        ax3.text(w2 + 0.5, bar.get_y() + bar.get_height()/2, f"{w2/100:.2f}",
                 va="center", fontsize=8.5, color="#444444")

    ax3.set_yticks(y_pos); ax3.set_yticklabels(classes, fontsize=10)
    ax3.set_xlim(0, 115)
    ax3.axvline(100, color="#AAAAAA", linestyle="--", linewidth=0.8)
    ax3.set_title("Accuracy % & F1 per Class", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Value", fontsize=9)
    ax3.legend(fontsize=8); ax3.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax3.set_axisbelow(True); ax3.spines[["top","right"]].set_visible(False)

    out = charts_dir / "summary_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out.name}")
    return out


def save_charts(y_true, y_pred, class_stats, report_dict, overall_acc, charts_dir, conf_thresh):
    charts_dir.mkdir(parents=True, exist_ok=True)
    classes_in_report = [c for c in CLASS_FOLDERS.values() if c in report_dict]

    print(f"\n{'─'*60}")
    print("  Saving charts...")
    plot_classification_metrics(report_dict, classes_in_report, charts_dir)
    plot_confusion_matrix(y_true, y_pred, ALL_LABELS, charts_dir)
    plot_detection_accuracy(class_stats, charts_dir)
    plot_summary_dashboard(report_dict, class_stats, y_true, y_pred, ALL_LABELS,
                           overall_acc, charts_dir, conf_thresh)
    print(f"{'─'*60}\n")


# ═════════════════════════════════════════════════════════════
#  MAIN INFERENCE
# ═════════════════════════════════════════════════════════════

def run_inference(model_path: Path, output_root: Path, conf_thresh: float, iou_thresh: float,
                  run_label: str = "", description: str = ""):
    model = YOLO(str(model_path))
    output_root.mkdir(parents=True, exist_ok=True)
    charts_dir = output_root / "charts"

    report_lines     = []
    grand_total      = {"images": 0, "detected": 0, "missed": 0}
    global_confusion = {}
    class_stats      = {}
    y_true_all       = []
    y_pred_all       = []

    print(f"\n{'='*60}")
    print(f"  Model  : {model_path}")
    print(f"  Testset: {TESTSET_DIR}")
    print(f"  Output : {output_root}")
    print(f"  Conf   : {conf_thresh}   IOU: {iou_thresh}")
    print(f"  Model classes (raw) : {model.names}")
    print(f"{'='*60}\n")

    for folder_name, display_name in CLASS_FOLDERS.items():
        cls_input_dir = Path(TESTSET_DIR) / folder_name
        if not cls_input_dir.exists():
            print(f"[SKIP] Folder not found: {cls_input_dir}")
            continue

        cls_output_dir = output_root / folder_name
        cls_output_dir.mkdir(parents=True, exist_ok=True)

        missed_dir = output_root / "missed" / folder_name
        if SAVE_MISSED:
            missed_dir.mkdir(parents=True, exist_ok=True)

        misdetect_dir = output_root / "mis_detection" / folder_name
        if SAVE_MISDETECT:
            misdetect_dir.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(
            list(cls_input_dir.glob("*.jpg"))  +
            list(cls_input_dir.glob("*.jpeg")) +
            list(cls_input_dir.glob("*.png"))
        )
        if not img_paths:
            print(f"[SKIP] No images in {cls_input_dir}")
            continue

        n_total, n_detected, n_missed = len(img_paths), 0, 0
        class_hits, correct_hits = {}, 0

        print(f"[{display_name.upper()}]  {n_total} images  (folder: {folder_name}/)")

        for img_path in img_paths:
            results = model.predict(source=str(img_path), conf=conf_thresh, iou=iou_thresh,
                                     imgsz=IMG_SIZE, verbose=False)[0]
            boxes = results.boxes
            has_detect = boxes is not None and len(boxes) > 0

            y_true_all.append(display_name)

            if has_detect:
                n_detected += 1
                pred_names = set()

                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                best_id = cls_ids[np.argmax(confs)]
                raw_pred_name = model.names[best_id]
                best_pred_name = to_display(raw_pred_name)
                y_pred_all.append(best_pred_name)

                if SAVE_MISDETECT and best_pred_name != display_name:
                    shutil.copy2(str(img_path), str(misdetect_dir / img_path.name))

                for cls_id in cls_ids:
                    det_name = to_display(model.names[cls_id])
                    class_hits[det_name] = class_hits.get(det_name, 0) + 1
                    pred_names.add(det_name)

                if display_name in pred_names:
                    correct_hits += 1

                global_confusion.setdefault(display_name, {})
                for pn in pred_names:
                    global_confusion[display_name][pn] = global_confusion[display_name].get(pn, 0) + 1
            else:
                n_missed += 1
                y_pred_all.append("no_detection")
                if SAVE_MISSED:
                    shutil.copy2(str(img_path), str(missed_dir / img_path.name))

            if SAVE_ALL or has_detect:
                annotated = results.plot()
                cv2.imwrite(str(cls_output_dir / img_path.name), annotated)

        class_stats[display_name] = {
            "total": n_total, "n_detected": n_detected,
            "n_missed": n_missed, "correct": correct_hits,
        }

        recall = correct_hits / n_total * 100 if n_total > 0 else 0.0
        sorted_hits = dict(sorted(class_hits.items(), key=lambda x: (x[0] != display_name, -x[1])))
        wrong_hits = {k: v for k, v in class_hits.items() if k != display_name}

        summary = (
            f"  Total images          : {n_total}\n"
            f"  Frames with ≥1 box    : {n_detected}  ({n_detected/n_total*100:.1f}%)\n"
            f"  Missed (0 boxes)      : {n_missed}\n"
            f"  Correct class in frame: {correct_hits}  ({recall:.1f}%)\n"
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

    overall_det_pct = grand_total["detected"] / grand_total["images"] * 100 if grand_total["images"] > 0 else 0.0
    grand_line = (
        f"\nOVERALL\n"
        f"  Total images : {grand_total['images']}\n"
        f"  Detected     : {grand_total['detected']}  ({overall_det_pct:.1f}%)\n"
        f"  Missed       : {grand_total['missed']}\n"
    )

    confusion_lines = ["\nCROSS-CLASS CONFUSION SUMMARY", "  (rows = true class, cols = what model predicted)\n"]
    all_cls = list(CLASS_FOLDERS.values())
    true_pred_label = "True \\ Pred"
    header = f"  {true_pred_label:<16}" + "".join(f"{c:<16}" for c in all_cls)
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

    sk_report_str = classification_report(y_true_all, y_pred_all, labels=ALL_LABELS, zero_division=0)
    overall_acc = accuracy_score(y_true_all, y_pred_all)

    clf_header = (
        "\n  ╔══════════════════════════════════════════════════════════════╗\n"
        "  ║              CLASSIFICATION REPORT  (sklearn)               ║\n"
        "  ╚══════════════════════════════════════════════════════════════╝\n"
    )
    print(clf_header)
    print(sk_report_str)
    print(f"  Overall Accuracy : {overall_acc*100:.2f}%\n")

    report_lines.append(clf_header)
    report_lines.append(sk_report_str)
    report_lines.append(f"  Overall Accuracy : {overall_acc*100:.2f}%\n")

    report_dict = classification_report(y_true_all, y_pred_all, labels=ALL_LABELS,
                                        zero_division=0, output_dict=True)
    save_charts(y_true_all, y_pred_all, class_stats, report_dict, overall_acc, charts_dir, conf_thresh)

    report_path = output_root / "inference_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model : {model_path}\n")
        f.write(f"Conf  : {conf_thresh}   IOU: {iou_thresh}\n\n")
        f.write("\n".join(report_lines))

    print(f"✓ Report saved       → {report_path}")
    print(f"✓ Annotated images   → {output_root}/<class>/")
    print(f"✓ Charts             → {output_root}/charts/")
    if SAVE_MISSED:
        print(f"✓ Missed frames      → {output_root}/missed/<class>/\n")

    # ── Log this evaluation to MLflow, under its own experiment ──────
    # Kept separate from whatever experiment the training runs live in,
    # so test-set results don't get mixed in with training metrics.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(TEST_EXPERIMENT_NAME)

    macro_f1 = report_dict["macro avg"]["f1-score"]

    with mlflow.start_run(run_name=f"eval_{run_label}" if run_label else None):
        if description:
            # Shows up as the run's description in the MLflow UI
            mlflow.set_tag("mlflow.note.content", description)
            mlflow.set_tag("description", description)

        mlflow.log_param("model_path", str(model_path))
        mlflow.log_param("source_run", run_label)
        mlflow.log_param("testset_dir", TESTSET_DIR)
        mlflow.log_param("conf_thresh", conf_thresh)
        mlflow.log_param("iou_thresh", iou_thresh)

        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("test_accuracy", overall_acc)

        # Per-class precision/recall/f1 for completeness, so individual
        # class regressions are visible in the MLflow UI too.
        for cls_name in ALL_LABELS:
            if cls_name in report_dict:
                mlflow.log_metric(f"{cls_name}_precision", report_dict[cls_name]["precision"])
                mlflow.log_metric(f"{cls_name}_recall",    report_dict[cls_name]["recall"])
                mlflow.log_metric(f"{cls_name}_f1",        report_dict[cls_name]["f1-score"])

        mlflow.log_artifact(str(report_path))
        if charts_dir.exists():
            mlflow.log_artifacts(str(charts_dir), artifact_path="charts")

        print(f"✓ Logged to MLflow   → experiment '{TEST_EXPERIMENT_NAME}'  "
              f"(macro_f1={macro_f1:.3f}, test_accuracy={overall_acc:.3f})\n")



def main():
    parser = argparse.ArgumentParser(description="Test a specific training run's best.pt against the fixed test set.")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Training run id, e.g. '38' for gym_equipment_v8n_weighted-38, "
                             "or a full MLflow run ID hash")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Explicit path to a .pt file, bypasses --run_id lookup")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESH, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESH, help="IOU threshold")
    parser.add_argument("--description", type=str, default=None,
                        help="Optional MANUAL override for the description logged to MLflow. "
                             "If omitted, the description already logged on the training run "
                             "is fetched and reused automatically.")
    args = parser.parse_args()

    model_path = resolve_model_path(args)

    # Use the resolved run folder name (e.g. "gym_equipment_v8n_weighted-38")
    # for the output directory, not the raw --run_id — this keeps output
    # folders readable even when an MLflow hex ID was passed in.
    run_label = model_path.parent.parent.name
    output_root = Path(OUTPUT_ROOT) / f"run_{run_label}"

    if args.description is not None:
        description = args.description
        print(f"  Using manually provided description: {description!r}")
    else:
        print(f"  No --description given — fetching the training run's own description from MLflow...")
        description = get_training_run_description(args.run_id, run_label)
        if description:
            print(f"  → Found training run description: {description!r}")
        else:
            print(f"  → No description found on the training run; logging without one.")

    run_inference(model_path, output_root, args.conf, args.iou,
                 run_label=run_label, description=description)


if __name__ == "__main__":
    main()
