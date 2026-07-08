import os
import shutil

# ---- CONFIG ----
IMAGES_DIR = r"/home/pc-008/weight_project/DB_and_KB_Detetcion/Dumbells and Kettlebells.v19i.yolov8_v3/train/images"
LABELS_DIR = r"/home/pc-008/weight_project/DB_and_KB_Detetcion/Dumbells and Kettlebells.v19i.yolov8_v3/train/labels"
OUTPUT_LABELS_DIR = r"/home/pc-008/weight_project/DB_and_KB_Detetcion/Dumbells and Kettlebells.v19i.yolov8_v3/train/labels_matched"
LABEL_EXTENSION = ".txt"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
# ----------------

def main():
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

    # Get base filenames (without extension) of all images
    image_basenames = {
        os.path.splitext(f)[0]
        for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    }

    if not image_basenames:
        print(f"No images found in {IMAGES_DIR}")
        return

    matched = 0
    missing = []

    for base_name in image_basenames:
        label_file = base_name + LABEL_EXTENSION
        src_path = os.path.join(LABELS_DIR, label_file)
        dest_path = os.path.join(OUTPUT_LABELS_DIR, label_file)

        if os.path.isfile(src_path):
            shutil.copy2(src_path, dest_path)
            matched += 1
        else:
            missing.append(label_file)

    print(f"Total images: {len(image_basenames)}")
    print(f"Labels matched & copied: {matched}")
    print(f"Labels missing: {len(missing)}")

    if missing:
        print("\nMissing label files (no match found):")
        for m in missing:
            print(f"  - {m}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
