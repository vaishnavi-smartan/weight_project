"""
extract_n_instances.py

Select whole images per class from a YOLO-format dataset until each class's
INSTANCE count (not image count) reaches n.

- Scans root_folder for label .txt files, matches each to its image (flat
  images/ + labels/ sibling folders, or same folder, or nested train/valid/test).
- For each class (read from labels_file: plain classes.txt OR a data.yaml with
  a 'names' key), copies whole, unmodified images into <output>/<class_name>/
  until that class has accumulated n instances.
- Image count is NOT constrained — one image can count toward multiple
  instances of the same class (e.g. 2 dumbbells annotated in one image counts
  as 2 db instances from a single copied file). Because whole images are
  copied rather than cropped, a class's final count can slightly exceed n if
  the last image needed to reach n contains more than one instance of it.
- If a class doesn't have n instances available in the dataset at all, it is
  NOT skipped — it collects everything available and reports how short it is.
- Every run wipes each class's output folder first, so re-running never
  accumulates on top of a previous run.

Usage:
    python extract_n_instances.py --root /path/to/dataset --labels /path/to/classes.txt --n 110
    python extract_n_instances.py --root /path/to/train/images --labels /path/to/data.yaml --n 110 --seed 42
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')


def load_class_names(labels_file):
    labels_file = Path(labels_file)

    if labels_file.suffix.lower() in ('.yaml', '.yml'):
        try:
            import yaml
        except ImportError:
            raise SystemExit("Reading data.yaml needs PyYAML. Install it with: pip install pyyaml")
        with open(labels_file, 'r') as f:
            data = yaml.safe_load(f)
        names = data.get('names')
        if names is None:
            raise ValueError(f"No 'names' key found in {labels_file}")
        # Roboflow yaml can store names as a list (index-ordered) or a dict {index: name}
        if isinstance(names, dict):
            names = [names[i] for i in sorted(names.keys(), key=int)]
        return list(names)

    # Plain text file: one class name per line, in class-index order
    with open(labels_file, 'r') as f:
        names = [line.strip() for line in f if line.strip()]
    if not names:
        raise ValueError(f"No class names found in {labels_file}")
    return names


def find_matching_image(label_path: Path, images_dir: Path = None):
    stem = label_path.stem
    # 1) explicit images_dir passed in (flat-folder case: images/ and labels/ are siblings)
    if images_dir is not None:
        for ext in IMG_EXTS:
            cand = images_dir / (stem + ext)
            if cand.exists():
                return cand
    # 2) same directory as the label file
    for ext in IMG_EXTS:
        cand = label_path.with_suffix(ext)
        if cand.exists():
            return cand
    # 3) sibling 'images' folder (common YOLO layout: .../labels/x.txt <-> .../images/x.jpg)
    if label_path.parent.name.lower() == 'labels':
        img_dir = label_path.parent.parent / 'images'
        for ext in IMG_EXTS:
            cand = img_dir / (stem + ext)
            if cand.exists():
                return cand
    return None


def resolve_images_and_labels_dirs(root: Path):
    """
    Figures out the images/ and labels/ folders from whatever path was passed in:
      - root/images + root/labels           (root = e.g. .../train/)
      - root itself is an 'images' folder    (root = .../train/images/)  -> sibling 'labels'
      - root itself is a 'labels' folder     (root = .../train/labels/)  -> sibling 'images'
    Returns (images_dir, labels_dir) or (None, None) if this pattern doesn't match.
    """
    if (root / 'images').is_dir() and (root / 'labels').is_dir():
        return root / 'images', root / 'labels'
    if root.name.lower() == 'images' and (root.parent / 'labels').is_dir():
        return root, root.parent / 'labels'
    if root.name.lower() == 'labels' and (root.parent / 'images').is_dir():
        return root.parent / 'images', root
    return None, None


def find_pairs(root_folder):
    root = Path(root_folder)
    pairs = []

    images_dir, labels_dir = resolve_images_and_labels_dirs(root)
    if images_dir is not None:
        # Flat-folder case: match every label file to its image by filename stem.
        print(f"Using images dir: {images_dir}")
        print(f"Using labels dir: {labels_dir}")
        for label_path in sorted(labels_dir.glob('*.txt')):
            img_path = find_matching_image(label_path, images_dir=images_dir)
            if img_path is not None:
                pairs.append((img_path, label_path))
            else:
                print(f"  Warning: no matching image for label {label_path.name}")
        return pairs

    # Fallback: recursive search (handles nested train/valid/test structures under root)
    for label_path in root.rglob('*.txt'):
        img_path = find_matching_image(label_path)
        if img_path is not None:
            pairs.append((img_path, label_path))
    return pairs


def select_instances(root_folder, labels_file, n, output_dir, seed=None):
    class_names = load_class_names(labels_file)
    output_dir = Path(output_dir)
    for name in class_names:
        class_dir = output_dir / name
        if class_dir.exists():
            shutil.rmtree(class_dir)  # wipe leftovers from any previous run
        class_dir.mkdir(parents=True, exist_ok=True)

    counts = defaultdict(int)
    pairs = find_pairs(root_folder)
    print(f"Found {len(pairs)} image-label pairs under {root_folder}")

    # Shuffle so that classes with surplus instances (e.g. db with 151 available
    # when n=110) get a spread across the whole dataset, not just the first
    # images encountered in filename order (which can cluster on a few video clips).
    rng = random.Random(seed)
    rng.shuffle(pairs)

    for img_path, label_path in pairs:
        if all(counts[name] >= n for name in class_names):
            break  # every class is already full, stop scanning

        with open(label_path, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        # Count how many instances of each class appear in THIS image.
        per_image_counts = defaultdict(int)
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            if cls_id < 0 or cls_id >= len(class_names):
                continue
            per_image_counts[class_names[cls_id]] += 1

        # Copy the WHOLE image (unmodified) into every class folder that still
        # needs more instances and that this image contributes to. Instance
        # count is what matters, not image count, so an image with 2 db boxes
        # adds 2 toward the db total even though it's a single file.
        for cls_name, instance_count in per_image_counts.items():
            if counts[cls_name] >= n:
                continue  # this class is already full, skip copying for it
            dest = output_dir / cls_name / img_path.name
            shutil.copy2(img_path, dest)
            counts[cls_name] += instance_count

    print("\nInstance counts collected:")
    for name in class_names:
        status = "OK" if counts[name] >= n else "SHORT"
        print(f"  {name}: {counts[name]}/{n}  [{status}]")

    short_classes = [name for name in class_names if counts[name] < n]
    if short_classes:
        print(f"\nWarning: dataset did not have enough instances for: {', '.join(short_classes)}")
    over_classes = [name for name in class_names if counts[name] > n]
    if over_classes:
        print(f"Note: these classes slightly exceeded {n} because the last image copied in "
              f"contained multiple instances at once (whole images aren't split): {', '.join(over_classes)}")


def main():
    parser = argparse.ArgumentParser(description="Select N instances per class from a YOLO dataset, copying whole images")
    parser.add_argument('--root', required=True, help="Root folder containing images and YOLO label .txt files")
    parser.add_argument('--labels', required=True, help="Path to classes/names file (one class name per line, in class-index order)")
    parser.add_argument('--n', type=int, required=True, help="Number of instances to extract per class")
    parser.add_argument('--output', default=None, help="Output folder (default: <root>/instance_samples)")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for shuffling image order (set for reproducible selection)")
    args = parser.parse_args()

    output_dir = args.output or str(Path(args.root) / "instance_samples")
    select_instances(args.root, args.labels, args.n, output_dir, seed=args.seed)


if __name__ == '__main__':
    main()
