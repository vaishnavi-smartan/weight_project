"""
Video Frame Extractor
---------------------
Splits video files into individual frames and saves them as images.

Requirements:
    pip install opencv-python

Usage:
    python extract_frames.py --video input.mp4
    python extract_frames.py --video input.mp4 --output frames/ --format png
    python extract_frames.py --video input.mp4 --fps 1 --quality 95
    python extract_frames.py --video input.mp4 --start 10 --end 30
"""

import cv2
import os
import argparse
from pathlib import Path


def extract_frames(
    video_path: str,
    output_dir: str = "frames",
    image_format: str = "jpg",
    every_n_frames: int = 1,
    fps: float = None,
    quality: int = 95,
    start_sec: float = None,
    end_sec: float = None,
    prefix: str = "frame",
):
    """
    Extract frames from a video file and save as images.

    Args:
        video_path     : Path to the input video file.
        output_dir     : Directory where frames will be saved.
        image_format   : Output image format — 'jpg', 'png', or 'bmp'.
        every_n_frames : Save every Nth frame (1 = all frames).
        fps            : Save frames at this rate (overrides every_n_frames).
        quality        : JPEG quality 1–100 (ignored for PNG).
        start_sec      : Start extraction at this timestamp (seconds).
        end_sec        : Stop extraction at this timestamp (seconds).
        prefix         : Filename prefix for saved frames.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Video metadata
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / video_fps if video_fps > 0 else 0

    print(f"\n{'='*50}")
    print(f"  Video  : {video_path.name}")
    print(f"  Size   : {width}x{height}")
    print(f"  FPS    : {video_fps:.2f}")
    print(f"  Frames : {total_frames}")
    print(f"  Length : {duration_sec:.1f}s")
    print(f"{'='*50}\n")

    # Resolve frame-skip from target fps
    if fps is not None:
        every_n_frames = max(1, round(video_fps / fps))
        print(f"Sampling at {fps} fps → saving every {every_n_frames} frame(s).")

    # Resolve start/end frame indices
    start_frame = int(start_sec * video_fps) if start_sec is not None else 0
    end_frame   = int(end_sec   * video_fps) if end_sec   is not None else total_frames
    start_frame = max(0, min(start_frame, total_frames))
    end_frame   = max(0, min(end_frame,   total_frames))

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Prepare output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JPEG / PNG encode params
    image_format = image_format.lower().lstrip(".")
    encode_params = []
    if image_format in ("jpg", "jpeg"):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif image_format == "png":
        # PNG compression 0-9 (0=none, 9=max); map quality 0-100 → 9-0
        png_compression = max(0, min(9, 9 - round(quality / 11)))
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]

    saved   = 0
    skipped = 0
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % every_n_frames == 0:
            filename = out_dir / f"{prefix}_{frame_idx:06d}.{image_format}"
            cv2.imwrite(str(filename), frame, encode_params)
            saved += 1
        else:
            skipped += 1

        frame_idx += 1

        if saved % 100 == 0 and saved > 0:
            print(f"  Saved {saved} frames...", end="\r")

    cap.release()

    print(f"\n✅ Done! Saved {saved} frames to '{out_dir}/'")
    print(f"   (Skipped {skipped} frames)\n")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video and save as images."
    )
    parser.add_argument("--video",   required=True,          help="Path to input video file")
    parser.add_argument("--output",  default="frames",       help="Output directory (default: frames/)")
    parser.add_argument("--format",  default="jpg",          help="Image format: jpg | png | bmp (default: jpg)")
    parser.add_argument("--every",   type=int,   default=1,  help="Save every N-th frame (default: 1 = all)")
    parser.add_argument("--fps",     type=float, default=None, help="Save this many frames per second (overrides --every)")
    parser.add_argument("--quality", type=int,   default=95, help="JPEG quality 1-100 (default: 95)")
    parser.add_argument("--start",   type=float, default=None, help="Start time in seconds")
    parser.add_argument("--end",     type=float, default=None, help="End time in seconds")
    parser.add_argument("--prefix",  default="frame",        help="Filename prefix (default: frame)")

    args = parser.parse_args()

    extract_frames(
        video_path    = args.video,
        output_dir    = args.output,
        image_format  = args.format,
        every_n_frames= args.every,
        fps           = args.fps,
        quality       = args.quality,
        start_sec     = args.start,
        end_sec       = args.end,
        prefix        = args.prefix,
    )


if __name__ == "__main__":
    main()
