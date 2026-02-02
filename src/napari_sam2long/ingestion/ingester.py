"""Single video frame extraction."""

import cv2
from pathlib import Path
from typing import Optional, Callable
import json


class VideoIngester:
    """Extract frames from a single video with reverse/strided sampling."""

    def __init__(self, config: "IngestionConfig"):
        """
        Initialize ingester.

        Args:
            config: IngestionConfig instance
        """
        self.config = config

    def run(self, progress_callback: Optional[Callable[[int, int], None]] = None) -> dict:
        """
        Extract frames and save metadata.

        Args:
            progress_callback: Optional callback fn(current, total)

        Returns:
            Metadata dict with frame mapping
        """
        output = Path(self.config.output_dir)
        images_dir = output / "images"
        masks_dir = output / "masks"

        # Create directory structure
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        for class_name in self.config.class_names:
            (masks_dir / class_name).mkdir(exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.config.video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate frame indices
        step = self.config.step_size
        if self.config.reverse:
            indices = list(range(total - 1, -1, -step))
        else:
            indices = list(range(0, total, step))

        # Build metadata (array format, not string-keyed dict)
        frames = []

        for seq_idx, orig_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, orig_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            filename = f"frame_{seq_idx:05d}.{self.config.image_format}"
            filepath = images_dir / filename

            if self.config.image_format.lower() in ("jpg", "jpeg"):
                cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.image_quality])
            else:
                cv2.imwrite(str(filepath), frame)

            frames.append({
                "seq": seq_idx,
                "orig": orig_idx,
                "file": filename,
            })

            if progress_callback:
                progress_callback(seq_idx + 1, len(indices))

        cap.release()

        # Save metadata
        metadata = {
            "frames": frames,
            "step_size": step,
            "reverse": self.config.reverse,
            "total_frames": len(frames),
            "original_total": total,
            "fps": fps,
            "resolution": [width, height],
            "source": Path(self.config.video_path).name,
        }

        with open(output / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata
