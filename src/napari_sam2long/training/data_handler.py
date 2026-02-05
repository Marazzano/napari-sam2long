"""
Training data handler for collecting samples from runs and exporting to COCO format.

Handles:
- Collecting mask/image pairs from annotation runs
- Train/val splits at the video level (to avoid data leakage)
- COCO format export for training
"""

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from ..ingestion.mask_io import read_indexed_mask, split_indexed_mask
from ..ingestion.run_manager import RunManager


@dataclass
class TrainingSample:
    """A single training sample (image + mask pair)."""

    image_path: Path
    mask_path: Path
    video_id: str
    frame_idx: int
    class_name: str

    def load_image(self) -> np.ndarray:
        """Load image as RGB numpy array."""
        img = cv2.imread(str(self.image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {self.image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_mask(self) -> np.ndarray:
        """Load mask as indexed numpy array (uint16)."""
        return read_indexed_mask(self.mask_path)

    def load_binary_masks(self) -> dict[int, np.ndarray]:
        """Load mask split into binary masks per instance."""
        return split_indexed_mask(self.load_mask())


class SplitManager:
    """
    Manages train/val splits at the video level.

    Splits are stored in project_root/splits.json and can be edited manually.
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.splits_path = self.project_root / "splits.json"
        self._splits: Optional[dict] = None

    @property
    def splits(self) -> dict:
        if self._splits is None:
            self._splits = self._load_or_create()
        return self._splits

    def _load_or_create(self) -> dict:
        if self.splits_path.exists():
            with open(self.splits_path) as f:
                return json.load(f)
        return {}

    def _save(self):
        with open(self.splits_path, "w") as f:
            json.dump(self._splits, f, indent=2)

    def get_split(self, name: str = "default") -> Optional[dict]:
        """Get a split by name. Returns None if not found."""
        return self.splits.get(name)

    def create_split(
        self,
        video_ids: list[str],
        val_ratio: float = 0.2,
        name: str = "default",
        seed: int = 42,
    ) -> dict:
        """
        Create a train/val split.

        Args:
            video_ids: List of all video IDs to split
            val_ratio: Fraction for validation (default 20%)
            name: Name for this split
            seed: Random seed for reproducibility

        Returns:
            Split dict with 'train' and 'val' video ID lists
        """
        shuffled = sorted(video_ids)  # Deterministic starting order
        random.Random(seed).shuffle(shuffled)

        n_val = max(1, int(len(shuffled) * val_ratio))
        split = {
            "train": shuffled[n_val:],
            "val": shuffled[:n_val],
            "created_at": datetime.now().isoformat(),
            "val_ratio": val_ratio,
            "seed": seed,
        }

        self._splits = self.splits  # Ensure loaded
        self._splits[name] = split
        self._save()

        return split

    def list_splits(self) -> list[str]:
        """List all available split names."""
        return list(self.splits.keys())


class TrainingDataHandler:
    """
    Collects training data from annotation runs and exports to COCO format.

    Usage:
        handler = TrainingDataHandler(project_root)
        handler.add_run("annot", 1)
        handler.add_run("annot", 2)  # Combine multiple runs

        # Export with auto-generated split
        train_json, val_json = handler.export_train_val_coco(
            output_dir,
            split_name="default",  # Auto-creates if doesn't exist
        )
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.run_manager = RunManager(project_root)
        self.split_manager = SplitManager(project_root)
        self._runs: list[tuple[str, int]] = []  # [(type, id), ...]
        self._samples: Optional[list[TrainingSample]] = None

        # Load project config for class names
        config_path = self.project_root / "project.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
            self.class_names = self.config.get("class_names", [])
        else:
            self.config = {}
            self.class_names = []

    def add_run(self, run_type: str, run_id: int):
        """
        Add a run to the training data collection.

        Args:
            run_type: Run type ('annot' or 'infer')
            run_id: Run ID number
        """
        # Validate run exists
        self.run_manager.get_run(run_type, run_id)
        self._runs.append((run_type, run_id))
        self._samples = None  # Invalidate cache

    def clear_runs(self):
        """Clear all added runs."""
        self._runs = []
        self._samples = None

    def _collect_samples(self) -> list[TrainingSample]:
        """Collect all samples from added runs."""
        samples = []

        for run_type, run_id in self._runs:
            run_masks_dir = self.run_manager.get_masks_dir(run_type, run_id)
            if not run_masks_dir.exists():
                continue

            # Iterate over videos
            for vid_dir in run_masks_dir.iterdir():
                if not vid_dir.is_dir():
                    continue
                video_id = vid_dir.name

                # Get images directory
                images_dir = self.project_root / "vids" / video_id / "images"
                if not images_dir.exists():
                    continue

                # Iterate over classes
                for class_dir in vid_dir.iterdir():
                    if not class_dir.is_dir():
                        continue
                    class_name = class_dir.name

                    # Iterate over mask files
                    for mask_path in sorted(class_dir.glob("*.png")):
                        frame_stem = mask_path.stem  # e.g., "frame_00000"

                        # Try common image extensions
                        image_path = None
                        for ext in [".jpg", ".jpeg", ".png"]:
                            candidate = images_dir / f"{frame_stem}{ext}"
                            if candidate.exists():
                                image_path = candidate
                                break

                        if image_path is None:
                            continue

                        # Extract frame index from filename
                        try:
                            frame_idx = int(frame_stem.split("_")[-1])
                        except (IndexError, ValueError):
                            frame_idx = 0

                        samples.append(TrainingSample(
                            image_path=image_path,
                            mask_path=mask_path,
                            video_id=video_id,
                            frame_idx=frame_idx,
                            class_name=class_name,
                        ))

        return samples

    @property
    def samples(self) -> list[TrainingSample]:
        """Get all collected samples (cached)."""
        if self._samples is None:
            self._samples = self._collect_samples()
        return self._samples

    def iter_samples(self, shuffle: bool = False) -> Iterator[TrainingSample]:
        """Iterate over training samples."""
        samples = self.samples.copy()
        if shuffle:
            random.shuffle(samples)
        yield from samples

    def get_video_ids(self) -> list[str]:
        """Get unique video IDs from collected samples."""
        return sorted(set(s.video_id for s in self.samples))

    def export_train_val_coco(
        self,
        output_dir: Path,
        split_name: str = "default",
        val_ratio: float = 0.2,
    ) -> tuple[Path, Path]:
        """
        Export samples to train and val COCO JSON files.

        Args:
            output_dir: Directory to write JSON files
            split_name: Name of split to use (auto-creates if doesn't exist)
            val_ratio: Validation ratio if creating new split

        Returns:
            (train_json_path, val_json_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get or create split
        split = self.split_manager.get_split(split_name)
        if split is None:
            video_ids = self.get_video_ids()
            split = self.split_manager.create_split(
                video_ids,
                val_ratio=val_ratio,
                name=split_name,
            )

        train_videos = set(split["train"])
        val_videos = set(split["val"])

        train_samples = [s for s in self.samples if s.video_id in train_videos]
        val_samples = [s for s in self.samples if s.video_id in val_videos]

        train_json = output_dir / "train.json"
        val_json = output_dir / "val.json"

        self._export_coco(train_samples, train_json)
        self._export_coco(val_samples, val_json)

        return train_json, val_json

    def export_coco(self, output_path: Path, samples: list[TrainingSample] = None) -> dict:
        """
        Export samples to single COCO JSON file.

        Args:
            output_path: Path for output JSON
            samples: Samples to export (default: all)

        Returns:
            COCO dict
        """
        samples = samples or self.samples
        return self._export_coco(samples, output_path)

    def _export_coco(self, samples: list[TrainingSample], output_path: Path) -> dict:
        """Internal method to export samples to COCO format."""
        # Build category mapping (1-indexed, 0 is background)
        categories = [
            {"id": i + 1, "name": name}
            for i, name in enumerate(self.class_names)
        ]
        cat_name_to_id = {c["name"]: c["id"] for c in categories}

        images = []
        annotations = []
        image_id = 0
        ann_id = 0

        for sample in samples:
            image_id += 1

            # Get image dimensions
            img = sample.load_image()
            h, w = img.shape[:2]

            images.append({
                "id": image_id,
                "file_name": str(sample.image_path.absolute()),
                "width": w,
                "height": h,
                "video_id": sample.video_id,
                "frame_idx": sample.frame_idx,
            })

            # Get category ID
            cat_id = cat_name_to_id.get(sample.class_name, 1)

            # Get binary masks for each instance
            try:
                binary_masks = sample.load_binary_masks()
            except Exception:
                continue

            for instance_id, binary_mask in binary_masks.items():
                ann_id += 1

                # Compute bounding box
                rows = np.any(binary_mask, axis=1)
                cols = np.any(binary_mask, axis=0)
                if not rows.any() or not cols.any():
                    continue

                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]

                # Area
                area = int(binary_mask.sum())

                # RLE encoding
                rle = self._mask_to_rle(binary_mask)

                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "segmentation": rle,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "instance_id": instance_id,
                })

        coco = {
            "info": {
                "description": "napari-sam2long training export",
                "date_created": datetime.now().isoformat(),
                "runs_used": [f"{t}_{i:03d}" for t, i in self._runs],
            },
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(coco, f)

        return coco

    def _mask_to_rle(self, binary_mask: np.ndarray) -> dict:
        """
        Convert binary mask to COCO RLE format.

        Args:
            binary_mask: Boolean array H×W

        Returns:
            RLE dict with 'counts' and 'size'
        """
        # Flatten in Fortran order (column-major)
        flat = binary_mask.flatten(order="F").astype(np.uint8)

        # Run-length encode
        runs = []
        prev = 0
        count = 0

        for val in flat:
            if val == prev:
                count += 1
            else:
                runs.append(count)
                count = 1
                prev = val
        runs.append(count)

        # COCO RLE starts with background run
        if flat[0] == 1:
            runs = [0] + runs

        return {
            "counts": runs,
            "size": [binary_mask.shape[0], binary_mask.shape[1]],
        }

    def get_stats(self) -> dict:
        """Get statistics about collected training data."""
        samples = self.samples

        by_video = {}
        by_class = {}

        for s in samples:
            by_video[s.video_id] = by_video.get(s.video_id, 0) + 1
            by_class[s.class_name] = by_class.get(s.class_name, 0) + 1

        return {
            "total_samples": len(samples),
            "runs_used": [f"{t}_{i:03d}" for t, i in self._runs],
            "num_videos": len(by_video),
            "samples_per_video": by_video,
            "samples_per_class": by_class,
        }
