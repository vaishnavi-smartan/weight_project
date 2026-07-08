import os
import random
import shutil

# ---- CONFIG ----
ROOT_DIR = "/home/pc-008/weight_project/DB_and_KB_Detetcion/new_class_exp/"   # <-- change this to your root folder path
NUM_SAMPLES = 110
VIDEO_EXTENSIONS = (".jpg", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm")
# ----------------

def get_subfolders(root_dir):
    return [
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f)) and f != "copy"
    ]

def main():
    copy_root = os.path.join(ROOT_DIR, "copy")
    os.makedirs(copy_root, exist_ok=True)

    subfolders = get_subfolders(ROOT_DIR)

    if not subfolders:
        print(f"No subfolders found in {ROOT_DIR}")
        return

    for folder_name in subfolders:
        src_folder = os.path.join(ROOT_DIR, folder_name)
        dest_folder = os.path.join(copy_root, folder_name)
        os.makedirs(dest_folder, exist_ok=True)

        # Collect all video files in this subfolder
        video_files = [
            f for f in os.listdir(src_folder)
            if f.lower().endswith(VIDEO_EXTENSIONS)
            and os.path.isfile(os.path.join(src_folder, f))
        ]

        if not video_files:
            print(f"[{folder_name}] No video files found. Skipping.")
            continue

        num_to_select = min(NUM_SAMPLES, len(video_files))
        selected_files = random.sample(video_files, num_to_select)

        for file_name in selected_files:
            src_path = os.path.join(src_folder, file_name)
            dest_path = os.path.join(dest_folder, file_name)
            shutil.copy2(src_path, dest_path)

        print(f"[{folder_name}] Copied {num_to_select}/{len(video_files)} videos -> {dest_folder}")

    print("\nDone. All copies are inside:", copy_root)

if __name__ == "__main__":
    main()
