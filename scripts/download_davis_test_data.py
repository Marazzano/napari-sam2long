#!/usr/bin/env python3
"""
Download DAVIS 2017 dataset subset and convert to napari-sam2long project format.

This creates a small test project with real video segmentation data
for testing the training pipeline.

Usage:
    python scripts/download_davis_test_data.py --output ./test_project --num-videos 3

The script will:
1. Download DAVIS 2017 trainval (480p) if not cached
2. Extract a subset of videos
3. Convert to napari-sam2long project structure
4. Create initial annotation run from ground truth masks
"""

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlretrieve

import cv2
import numpy as np


DAVIS_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
DAVIS_CACHE_DIR = Path.home() / ".cache" / "napari-sam2long" / "davis"


def download_with_progress(url: str, dest: Path) -> Path:
    """Download file with progress bar."""

    def report_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 // total_size)
        bar = "=" * (percent // 2) + ">" + " " * (50 - percent // 2)
        sys.stdout.write(f"\r[{bar}] {percent}%")
        sys.stdout.flush()

    print(f"Downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest, reporthook=report_hook)
    print()  # Newline after progress bar
    return dest


def get_davis_path() -> Path:
    """Get path to DAVIS dataset, downloading if necessary."""
    zip_path = DAVIS_CACHE_DIR / "DAVIS-2017-trainval-480p.zip"
    extract_path = DAVIS_CACHE_DIR / "DAVIS"

    if extract_path.exists() and (extract_path / "JPEGImages").exists():
        print(f"Using cached DAVIS at {extract_path}")
        return extract_path

    if not zip_path.exists():
        download_with_progress(DAVIS_URL, zip_path)

    print(f"Extracting to {DAVIS_CACHE_DIR}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DAVIS_CACHE_DIR)

    return extract_path


def list_davis_videos(davis_path: Path) -> list[str]:
    """List all video sequences in DAVIS."""
    jpeg_dir = davis_path / "JPEGImages" / "480p"
    return sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])


def convert_davis_to_project(
    davis_path: Path,
    output_path: Path,
    video_names: list[str],
    class_name: str = "object",
):
    """
    Convert DAVIS videos to napari-sam2long project format.

    Args:
        davis_path: Path to extracted DAVIS dataset
        output_path: Output project directory
        video_names: List of DAVIS sequence names to include
        class_name: Name for the segmentation class
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    jpeg_base = davis_path / "JPEGImages" / "480p"
    anno_base = davis_path / "Annotations" / "480p"

    # Create project.json
    project_config = {
        "class_names": [class_name],
        "video_extensions": [".mp4", ".avi", ".mov"],
        "source": "DAVIS-2017-trainval-480p",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path / "project.json", "w") as f:
        json.dump(project_config, f, indent=2)

    # Create manifest.json
    manifest = {"videos": {}}

    vids_dir = output_path / "vids"
    vids_dir.mkdir(exist_ok=True)

    for idx, video_name in enumerate(video_names):
        vid_id = f"vid_{idx + 1:03d}"
        jpeg_dir = jpeg_base / video_name
        anno_dir = anno_base / video_name

        if not jpeg_dir.exists():
            print(f"Warning: {video_name} not found, skipping")
            continue

        # Create video directory
        vid_dir = vids_dir / vid_id
        images_dir = vid_dir / "images"
        masks_dir = vid_dir / "masks" / class_name
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        # Copy frames and masks
        frame_files = sorted(jpeg_dir.glob("*.jpg"))
        frames_meta = []

        for seq_idx, frame_file in enumerate(frame_files):
            # Copy image
            dst_image = images_dir / f"frame_{seq_idx:05d}.jpg"
            shutil.copy(frame_file, dst_image)

            # Convert and copy mask (DAVIS uses colored PNGs, we need indexed)
            mask_file = anno_dir / frame_file.with_suffix(".png").name
            if mask_file.exists():
                # Read DAVIS mask (palette-based PNG)
                mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    # DAVIS masks: 0 = background, other values = object IDs
                    # Already indexed, just need to ensure uint8/uint16
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]  # Take first channel
                    mask = mask.astype(np.uint16)
                    dst_mask = masks_dir / f"frame_{seq_idx:05d}.png"
                    cv2.imwrite(str(dst_mask), mask)

            frames_meta.append({
                "seq": seq_idx,
                "orig": seq_idx,
                "file": f"frame_{seq_idx:05d}.jpg",
            })

        # Get frame dimensions
        if frame_files:
            sample = cv2.imread(str(frame_files[0]))
            h, w = sample.shape[:2]
        else:
            h, w = 480, 854

        # Create metadata.json
        metadata = {
            "frames": frames_meta,
            "step_size": 1,
            "reverse": False,
            "total_frames": len(frames_meta),
            "original_total": len(frames_meta),
            "fps": 24,
            "resolution": [w, h],
            "source": video_name,
        }
        with open(vid_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Add to manifest
        manifest["videos"][vid_id] = {
            "source": video_name,
            "status": "completed",  # Already has GT masks
            "frame_count": len(frames_meta),
            "added_at": datetime.now(timezone.utc).isoformat(),
        }

        print(f"  {vid_id}: {video_name} ({len(frames_meta)} frames)")

    # Write manifest
    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Create initial annotation run from the GT masks
    create_initial_run(output_path, list(manifest["videos"].keys()))

    print(f"\nProject created at: {output_path}")
    print(f"Videos: {len(manifest['videos'])}")
    print(f"Class: {class_name}")


def create_initial_run(project_path: Path, video_ids: list[str]):
    """Create an initial annotation run from the working masks."""
    runs_dir = project_path / "runs"
    run_dir = runs_dir / "annot_001_ground-truth"
    run_masks = run_dir / "masks"
    run_masks.mkdir(parents=True)

    # Copy masks from working directory
    vids_dir = project_path / "vids"
    for vid_id in video_ids:
        src_masks = vids_dir / vid_id / "masks"
        if src_masks.exists():
            dst_masks = run_masks / vid_id
            shutil.copytree(src_masks, dst_masks)

    # Create manifest
    manifest = {
        "schema_version": 1,
        "id": 1,
        "type": "annot",
        "label": "ground-truth",
        "name": "annot_001_ground-truth",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parents": [],
        "metadata": {
            "source": "DAVIS-2017 ground truth annotations",
        },
        "videos_included": video_ids,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created initial run: annot_001_ground-truth")


def main():
    parser = argparse.ArgumentParser(
        description="Download DAVIS dataset and create test project"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./test_davis_project"),
        help="Output project directory",
    )
    parser.add_argument(
        "--num-videos", "-n",
        type=int,
        default=3,
        help="Number of videos to include (default: 3)",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        help="Specific video names to include (overrides --num-videos)",
    )
    parser.add_argument(
        "--list-videos",
        action="store_true",
        help="List available DAVIS videos and exit",
    )
    parser.add_argument(
        "--class-name",
        default="object",
        help="Name for the segmentation class (default: object)",
    )

    args = parser.parse_args()

    # Download/get DAVIS
    print("Checking DAVIS dataset...")
    davis_path = get_davis_path()

    # List videos if requested
    all_videos = list_davis_videos(davis_path)

    if args.list_videos:
        print(f"\nAvailable DAVIS videos ({len(all_videos)}):")
        for v in all_videos:
            print(f"  {v}")
        return

    # Select videos
    if args.videos:
        video_names = args.videos
    else:
        # Select diverse videos (short ones first for testing)
        # These are good test cases: clear objects, manageable length
        preferred = ["bear", "blackswan", "bmx-trees", "boat", "breakdance"]
        video_names = [v for v in preferred if v in all_videos][:args.num_videos]
        if len(video_names) < args.num_videos:
            remaining = [v for v in all_videos if v not in video_names]
            video_names.extend(remaining[:args.num_videos - len(video_names)])

    print(f"\nConverting {len(video_names)} videos to project format:")
    convert_davis_to_project(
        davis_path,
        args.output,
        video_names,
        class_name=args.class_name,
    )


if __name__ == "__main__":
    main()
