# Multi-Model Active Learning Architecture Proposal

## Executive Summary

This document proposes an architecture to extend napari-sam2long into a **multi-model active learning platform**. The core loop:

```
SAM2 Propagation → Manual Refinement → Train Expert Model → Expert Inference →
Load & Refine → Propagate with SAM2 → Repeat
```

This creates a reinforcement loop where:
1. **SAM2** provides fast video propagation from sparse annotations
2. **Expert models** (UNet, Detectron2) learn from accumulated annotations
3. **Human refinement** improves model outputs iteratively
4. Each cycle produces better annotations with less manual effort

---

## Current State Analysis

### Existing Project Structure
```
project_root/
├── project.json          # {class_names, video_extensions, ...}
├── manifest.json         # {videos: {vid_001: {status, source, ...}}}
├── exports/coco/         # COCO format exports
└── vids/
    └── vid_001/
        ├── images/           # frame_00000.jpg, ...
        ├── metadata.json     # Frame mapping, video properties
        └── masks/
            └── class_name/   # frame_00000.png (indexed masks)
```

### Current Workflow
1. Load project → Load video frames
2. Draw masks with napari tools OR use SAM2 propagation
3. Approve anchor frames → Propagate with SAM2
4. Save masks → Export to COCO

### Key Strengths to Preserve
- Standard project structure with JSON configs
- Indexed PNG mask storage (supports multi-instance)
- COCO export for training compatibility
- Multi-layer, multi-class support
- Anchor-based SAM2 propagation

---

## Proposed Architecture

### 1. Extended Project Structure

```
project_root/
├── project.json              # Extended with model configs
├── manifest.json             # Extended with run tracking
├── vids/                     # Unchanged
│   └── vid_001/
│       ├── images/
│       ├── metadata.json
│       └── masks/            # "Working" masks (current napari state)
│
├── runs/                     # NEW: Version-controlled annotation/inference history
│   ├── runs_manifest.json    # Index of all runs with lineage
│   │
│   ├── run_001_manual/       # Manual annotation run
│   │   ├── run_meta.json
│   │   └── masks/
│   │       └── vid_001/
│   │           └── class_name/
│   │
│   ├── run_002_training/     # Training run (no masks, has checkpoints)
│   │   ├── run_meta.json
│   │   ├── checkpoints/
│   │   │   ├── epoch_10.pth
│   │   │   └── best.pth
│   │   └── metrics/
│   │       └── training_log.json
│   │
│   └── run_003_inference/    # Inference run
│       ├── run_meta.json
│       └── masks/
│           └── vid_001/
│               └── class_name/
│
└── models/                   # NEW: Model registry
    ├── models_manifest.json  # Registered models and their configs
    ├── unet/
    │   └── config.yaml
    └── detectron2/
        └── config.yaml
```

### 2. Run Types and Metadata

#### Run Manifest (`runs/runs_manifest.json`)
```json
{
  "runs": {
    "run_001_manual": {
      "id": "run_001_manual",
      "type": "manual",
      "created_at": "2025-01-15T10:30:00Z",
      "description": "Initial SAM2 propagation + manual refinement",
      "parent_run": null,
      "videos_included": ["vid_001", "vid_002"],
      "stats": {
        "total_frames": 200,
        "annotated_frames": 45
      }
    },
    "run_002_training": {
      "id": "run_002_training",
      "type": "training",
      "created_at": "2025-01-15T14:00:00Z",
      "description": "UNet training on run_001",
      "parent_run": "run_001_manual",
      "model_type": "unet",
      "input_runs": ["run_001_manual"],
      "checkpoint": "checkpoints/best.pth",
      "training_config": {
        "epochs": 50,
        "batch_size": 8,
        "learning_rate": 0.001
      },
      "final_metrics": {
        "val_iou": 0.72,
        "val_loss": 0.15
      }
    },
    "run_003_inference": {
      "id": "run_003_inference",
      "type": "inference",
      "created_at": "2025-01-15T16:00:00Z",
      "description": "UNet inference on all project videos",
      "parent_run": "run_002_training",
      "model_type": "unet",
      "model_run": "run_002_training",
      "checkpoint_used": "checkpoints/best.pth",
      "videos_included": ["vid_001", "vid_002", "vid_003"],
      "stats": {
        "total_frames": 500,
        "inference_time_sec": 120
      }
    },
    "run_004_manual": {
      "id": "run_004_manual",
      "type": "manual",
      "created_at": "2025-01-16T09:00:00Z",
      "description": "Refinement of UNet predictions + SAM2 propagation",
      "parent_run": "run_003_inference",
      "based_on_inference": "run_003_inference",
      "videos_included": ["vid_001", "vid_002", "vid_003"]
    }
  },
  "current_working_run": "run_004_manual",
  "lineage_graph": {
    "run_001_manual": [],
    "run_002_training": ["run_001_manual"],
    "run_003_inference": ["run_002_training"],
    "run_004_manual": ["run_003_inference"]
  }
}
```

#### Individual Run Metadata (`runs/run_XXX/run_meta.json`)
```json
{
  "id": "run_003_inference",
  "type": "inference",
  "created_at": "2025-01-15T16:00:00Z",
  "model_type": "unet",
  "model_run": "run_002_training",
  "checkpoint": "checkpoints/best.pth",
  "config": {
    "threshold": 0.5,
    "batch_size": 4
  },
  "videos": {
    "vid_001": {
      "frames_processed": 100,
      "avg_confidence": 0.82
    }
  }
}
```

---

### 3. Core Backend Components

#### 3.1 Run Manager (`src/napari_sam2long/ingestion/run_manager.py`)

```python
"""Run management for tracking annotation/training/inference history."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set
import json
import shutil

class RunType(Enum):
    MANUAL = "manual"
    TRAINING = "training"
    INFERENCE = "inference"

@dataclass
class RunInfo:
    """Metadata for a single run."""
    id: str
    type: RunType
    created_at: datetime
    description: str = ""
    parent_run: Optional[str] = None
    videos_included: List[str] = field(default_factory=list)

    # Type-specific fields
    model_type: Optional[str] = None          # For training/inference
    input_runs: List[str] = field(default_factory=list)  # For training
    checkpoint: Optional[str] = None          # For training (output) / inference (input)
    model_run: Optional[str] = None           # For inference
    training_config: Dict = field(default_factory=dict)
    final_metrics: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "parent_run": self.parent_run,
            "videos_included": self.videos_included,
            "model_type": self.model_type,
            "input_runs": self.input_runs,
            "checkpoint": self.checkpoint,
            "model_run": self.model_run,
            "training_config": self.training_config,
            "final_metrics": self.final_metrics,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunInfo":
        return cls(
            id=data["id"],
            type=RunType(data["type"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            description=data.get("description", ""),
            parent_run=data.get("parent_run"),
            videos_included=data.get("videos_included", []),
            model_type=data.get("model_type"),
            input_runs=data.get("input_runs", []),
            checkpoint=data.get("checkpoint"),
            model_run=data.get("model_run"),
            training_config=data.get("training_config", {}),
            final_metrics=data.get("final_metrics", {}),
        )


class RunManager:
    """
    Manages annotation/training/inference runs for a project.

    Provides:
    - Run creation and persistence
    - Lineage tracking (which run derived from which)
    - Mask versioning (copy masks between runs)
    - Checkpoint management
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.runs_dir = self.project_root / "runs"
        self.manifest_path = self.runs_dir / "runs_manifest.json"
        self._manifest = None

    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            self._manifest = self._load_or_create_manifest()
        return self._manifest

    def _load_or_create_manifest(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {
            "runs": {},
            "current_working_run": None,
            "lineage_graph": {}
        }

    def _save_manifest(self):
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def create_run(
        self,
        run_type: RunType,
        description: str = "",
        parent_run: Optional[str] = None,
        model_type: Optional[str] = None,
        copy_masks_from: Optional[str] = None,
    ) -> RunInfo:
        """
        Create a new run.

        Args:
            run_type: Type of run (manual, training, inference)
            description: Human-readable description
            parent_run: ID of parent run in lineage
            model_type: For training/inference runs
            copy_masks_from: Run ID to copy initial masks from

        Returns:
            RunInfo for the new run
        """
        # Generate run ID
        run_count = len(self.manifest["runs"]) + 1
        run_id = f"run_{run_count:03d}_{run_type.value}"

        # Create run directory
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create run info
        run_info = RunInfo(
            id=run_id,
            type=run_type,
            created_at=datetime.now(),
            description=description,
            parent_run=parent_run,
            model_type=model_type,
        )

        # Copy masks if specified
        if copy_masks_from and run_type != RunType.TRAINING:
            self._copy_masks(copy_masks_from, run_id)

        # Create subdirectories based on type
        if run_type == RunType.TRAINING:
            (run_dir / "checkpoints").mkdir(exist_ok=True)
            (run_dir / "metrics").mkdir(exist_ok=True)
        else:
            (run_dir / "masks").mkdir(exist_ok=True)

        # Save run metadata
        with open(run_dir / "run_meta.json", "w") as f:
            json.dump(run_info.to_dict(), f, indent=2)

        # Update manifest
        self.manifest["runs"][run_id] = run_info.to_dict()
        self.manifest["lineage_graph"][run_id] = [parent_run] if parent_run else []
        self._save_manifest()

        return run_info

    def get_run(self, run_id: str) -> Optional[RunInfo]:
        """Get run info by ID."""
        if run_id in self.manifest["runs"]:
            return RunInfo.from_dict(self.manifest["runs"][run_id])
        return None

    def list_runs(self, run_type: Optional[RunType] = None) -> List[RunInfo]:
        """List all runs, optionally filtered by type."""
        runs = [RunInfo.from_dict(r) for r in self.manifest["runs"].values()]
        if run_type:
            runs = [r for r in runs if r.type == run_type]
        return sorted(runs, key=lambda r: r.created_at)

    def get_run_dir(self, run_id: str) -> Path:
        """Get directory path for a run."""
        return self.runs_dir / run_id

    def get_masks_dir(self, run_id: str, video_id: str, class_name: str) -> Path:
        """Get masks directory for a specific run/video/class."""
        return self.runs_dir / run_id / "masks" / video_id / class_name

    def get_checkpoint_dir(self, run_id: str) -> Path:
        """Get checkpoints directory for a training run."""
        return self.runs_dir / run_id / "checkpoints"

    def _copy_masks(self, from_run: str, to_run: str):
        """Copy all masks from one run to another."""
        src = self.runs_dir / from_run / "masks"
        dst = self.runs_dir / to_run / "masks"
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)

    def save_working_to_run(self, run_id: str, video_ids: List[str] = None):
        """
        Save current working masks (vids/*/masks) to a run.

        Args:
            run_id: Target run ID
            video_ids: Optional list of video IDs to save (default: all)
        """
        run_masks = self.runs_dir / run_id / "masks"
        run_masks.mkdir(parents=True, exist_ok=True)

        vids_dir = self.project_root / "vids"
        for vid_dir in vids_dir.iterdir():
            if not vid_dir.is_dir():
                continue
            if video_ids and vid_dir.name not in video_ids:
                continue

            src_masks = vid_dir / "masks"
            if src_masks.exists():
                dst = run_masks / vid_dir.name
                shutil.copytree(src_masks, dst, dirs_exist_ok=True)

        # Update run info
        run_info = self.get_run(run_id)
        if run_info:
            run_info.videos_included = video_ids or [d.name for d in vids_dir.iterdir() if d.is_dir()]
            self.manifest["runs"][run_id] = run_info.to_dict()
            self._save_manifest()

    def load_run_to_working(self, run_id: str, video_ids: List[str] = None):
        """
        Load masks from a run into working directory (vids/*/masks).

        Args:
            run_id: Source run ID
            video_ids: Optional list of video IDs to load (default: all in run)
        """
        run_masks = self.runs_dir / run_id / "masks"
        if not run_masks.exists():
            raise ValueError(f"No masks found in run {run_id}")

        for vid_dir in run_masks.iterdir():
            if not vid_dir.is_dir():
                continue
            if video_ids and vid_dir.name not in video_ids:
                continue

            dst = self.project_root / "vids" / vid_dir.name / "masks"
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copytree(vid_dir, dst, dirs_exist_ok=True)

    def get_lineage(self, run_id: str) -> List[str]:
        """Get full lineage chain for a run (ancestors first)."""
        lineage = []
        current = run_id
        while current:
            lineage.insert(0, current)
            parents = self.manifest["lineage_graph"].get(current, [])
            current = parents[0] if parents else None
        return lineage

    def update_run_metrics(self, run_id: str, metrics: dict):
        """Update final metrics for a training run."""
        if run_id in self.manifest["runs"]:
            self.manifest["runs"][run_id]["final_metrics"] = metrics
            self._save_manifest()

            # Also update run_meta.json
            run_meta_path = self.runs_dir / run_id / "run_meta.json"
            if run_meta_path.exists():
                with open(run_meta_path) as f:
                    meta = json.load(f)
                meta["final_metrics"] = metrics
                with open(run_meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
```

#### 3.2 Training Data Handler (`src/napari_sam2long/training/data_handler.py`)

```python
"""Unified data handler for training expert models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import json
import numpy as np
import cv2

from ..ingestion.mask_io import read_indexed_mask, split_indexed_mask


@dataclass
class TrainingSample:
    """A single training sample (image + mask)."""
    image_path: Path
    mask_path: Path
    video_id: str
    frame_idx: int
    class_name: str

    def load_image(self) -> np.ndarray:
        """Load image as RGB numpy array."""
        img = cv2.imread(str(self.image_path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_mask(self) -> np.ndarray:
        """Load mask as indexed numpy array."""
        return read_indexed_mask(self.mask_path)

    def load_binary_masks(self) -> Dict[int, np.ndarray]:
        """Load mask split into binary masks per instance."""
        return split_indexed_mask(self.load_mask())


class TrainingDataHandler:
    """
    Collects and formats training data from project runs.

    Supports multiple output formats:
    - Raw: Iterator of (image, mask) numpy arrays
    - COCO: Standard COCO format for Detectron2
    - ImageFolder: Directory structure for torchvision

    Usage:
        handler = TrainingDataHandler(project_root)
        handler.add_run("run_001_manual")
        handler.add_run("run_004_manual")

        # Get samples
        for sample in handler.iter_samples():
            img = sample.load_image()
            mask = sample.load_mask()

        # Export to COCO
        handler.export_coco("train.json", split="train")

        # Export to image folder
        handler.export_image_folder("./data/train/")
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.runs_dir = self.project_root / "runs"
        self._run_ids: List[str] = []
        self._samples: Optional[List[TrainingSample]] = None

        # Load project config for class names
        with open(self.project_root / "project.json") as f:
            self.config = json.load(f)
        self.class_names = self.config.get("class_names", [])

    def add_run(self, run_id: str):
        """Add a run to the training data collection."""
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Run {run_id} not found")
        self._run_ids.append(run_id)
        self._samples = None  # Invalidate cache

    def clear_runs(self):
        """Clear all added runs."""
        self._run_ids = []
        self._samples = None

    def _collect_samples(self) -> List[TrainingSample]:
        """Collect all samples from added runs."""
        samples = []

        for run_id in self._run_ids:
            run_masks_dir = self.runs_dir / run_id / "masks"
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
                    for mask_path in class_dir.glob("*.png"):
                        frame_stem = mask_path.stem  # e.g., "frame_00000"
                        image_path = images_dir / f"{frame_stem}.jpg"

                        if not image_path.exists():
                            continue

                        # Extract frame index from filename
                        try:
                            frame_idx = int(frame_stem.split("_")[1])
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
    def samples(self) -> List[TrainingSample]:
        """Get all collected samples (cached)."""
        if self._samples is None:
            self._samples = self._collect_samples()
        return self._samples

    def iter_samples(self, shuffle: bool = False) -> Iterator[TrainingSample]:
        """Iterate over training samples."""
        samples = self.samples.copy()
        if shuffle:
            import random
            random.shuffle(samples)
        yield from samples

    def split_train_val(
        self,
        val_ratio: float = 0.2,
        by_video: bool = True
    ) -> Tuple[List[TrainingSample], List[TrainingSample]]:
        """
        Split samples into train and validation sets.

        Args:
            val_ratio: Fraction for validation
            by_video: If True, split by video (all frames from a video go to same set)

        Returns:
            (train_samples, val_samples)
        """
        import random

        if by_video:
            # Group by video
            by_vid = {}
            for s in self.samples:
                by_vid.setdefault(s.video_id, []).append(s)

            videos = list(by_vid.keys())
            random.shuffle(videos)

            n_val = max(1, int(len(videos) * val_ratio))
            val_videos = set(videos[:n_val])

            train = [s for s in self.samples if s.video_id not in val_videos]
            val = [s for s in self.samples if s.video_id in val_videos]
        else:
            samples = self.samples.copy()
            random.shuffle(samples)
            n_val = int(len(samples) * val_ratio)
            val = samples[:n_val]
            train = samples[n_val:]

        return train, val

    def export_coco(
        self,
        output_path: str,
        samples: Optional[List[TrainingSample]] = None,
        copy_images: bool = False,
        images_dir: Optional[str] = None,
    ) -> dict:
        """
        Export samples to COCO format.

        Args:
            output_path: Path for output JSON
            samples: Samples to export (default: all)
            copy_images: If True, copy images to images_dir
            images_dir: Directory for copied images

        Returns:
            COCO dict
        """
        from datetime import datetime
        from ..ingestion.coco_export import mask_to_rle, mask_to_bbox

        samples = samples or self.samples

        # Build category mapping
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

            img = sample.load_image()
            h, w = img.shape[:2]

            # Determine image filename
            if copy_images and images_dir:
                img_filename = f"{sample.video_id}_{sample.frame_idx:05d}.jpg"
                dst = Path(images_dir) / img_filename
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            else:
                img_filename = str(sample.image_path)

            images.append({
                "id": image_id,
                "file_name": img_filename,
                "width": w,
                "height": h,
                "video_id": sample.video_id,
                "frame_idx": sample.frame_idx,
            })

            # Get annotations from mask
            cat_id = cat_name_to_id.get(sample.class_name, 1)
            binary_masks = sample.load_binary_masks()

            for instance_id, binary_mask in binary_masks.items():
                ann_id += 1
                bbox = mask_to_bbox(binary_mask)
                area = int(binary_mask.sum())

                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "segmentation": mask_to_rle(binary_mask),
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "instance_id": instance_id,
                })

        coco = {
            "info": {
                "description": "napari-sam2long training export",
                "date_created": datetime.now().isoformat(),
            },
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        with open(output_path, "w") as f:
            json.dump(coco, f)

        return coco

    def export_image_folder(
        self,
        output_dir: str,
        samples: Optional[List[TrainingSample]] = None,
    ):
        """
        Export to image folder structure for torchvision.

        Structure:
            output_dir/
                images/
                    vid_001_00000.jpg
                masks/
                    vid_001_00000.png
        """
        samples = samples or self.samples
        output = Path(output_dir)
        (output / "images").mkdir(parents=True, exist_ok=True)
        (output / "masks").mkdir(parents=True, exist_ok=True)

        for sample in samples:
            basename = f"{sample.video_id}_{sample.frame_idx:05d}"

            # Copy image
            img = sample.load_image()
            cv2.imwrite(
                str(output / "images" / f"{basename}.jpg"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )

            # Copy mask
            mask = sample.load_mask()
            cv2.imwrite(str(output / "masks" / f"{basename}.png"), mask)

    def get_stats(self) -> dict:
        """Get statistics about collected training data."""
        samples = self.samples

        by_video = {}
        by_class = {}

        for s in samples:
            by_video.setdefault(s.video_id, 0)
            by_video[s.video_id] += 1
            by_class.setdefault(s.class_name, 0)
            by_class[s.class_name] += 1

        return {
            "total_samples": len(samples),
            "runs_used": self._run_ids.copy(),
            "videos": by_video,
            "classes": by_class,
        }
```

#### 3.3 Model Registry (`src/napari_sam2long/training/model_registry.py`)

```python
"""Registry and base class for expert models (UNet, Detectron2, etc.)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Type
import json
import numpy as np


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-3
    val_ratio: float = 0.2
    checkpoint_interval: int = 10
    early_stopping_patience: int = 10
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    extra: Dict = None  # Model-specific config

    def to_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "val_ratio": self.val_ratio,
            "checkpoint_interval": self.checkpoint_interval,
            "early_stopping_patience": self.early_stopping_patience,
            "device": self.device,
            "extra": self.extra or {},
        }


@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    batch_size: int = 4
    threshold: float = 0.5
    device: str = "auto"
    extra: Dict = None

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "threshold": self.threshold,
            "device": self.device,
            "extra": self.extra or {},
        }


class BaseExpertModel(ABC):
    """
    Abstract base class for expert segmentation models.

    Implementations must provide:
    - train(): Train on data from TrainingDataHandler
    - predict(): Run inference on images
    - load_checkpoint(): Load saved weights
    - save_checkpoint(): Save weights
    """

    name: str = "base"

    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.model = None

    @abstractmethod
    def train(
        self,
        train_samples: List,
        val_samples: List,
        config: TrainingConfig,
        checkpoint_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> Dict:
        """
        Train the model.

        Args:
            train_samples: List of TrainingSample for training
            val_samples: List of TrainingSample for validation
            config: Training configuration
            checkpoint_dir: Directory to save checkpoints
            progress_callback: Optional callback(epoch, metrics) for progress

        Returns:
            Final training metrics dict
        """
        pass

    @abstractmethod
    def predict(
        self,
        images: Iterator[np.ndarray],
        config: InferenceConfig,
    ) -> Iterator[np.ndarray]:
        """
        Run inference on images.

        Args:
            images: Iterator of RGB numpy arrays
            config: Inference configuration

        Yields:
            Indexed mask arrays (same shape as input, uint16)
        """
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model weights from checkpoint."""
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint_path: Path):
        """Save model weights to checkpoint."""
        pass

    @classmethod
    def get_default_config(cls) -> Dict:
        """Get default model-specific configuration."""
        return {}


class ModelRegistry:
    """Registry of available expert models."""

    _models: Dict[str, Type[BaseExpertModel]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class."""
        def decorator(model_cls: Type[BaseExpertModel]):
            model_cls.name = name
            cls._models[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseExpertModel]:
        """Get a model class by name."""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._models.keys())}")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model names."""
        return list(cls._models.keys())

    @classmethod
    def create(cls, name: str, num_classes: int, class_names: List[str]) -> BaseExpertModel:
        """Create a model instance by name."""
        model_cls = cls.get(name)
        return model_cls(num_classes, class_names)
```

#### 3.4 UNet Implementation (`src/napari_sam2long/training/models/unet.py`)

```python
"""UNet implementation for semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Iterator, List, Optional
import numpy as np

from ..model_registry import BaseExpertModel, ModelRegistry, TrainingConfig, InferenceConfig


class UNetBlock(nn.Module):
    """Double convolution block for UNet."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Standard UNet architecture."""

    def __init__(self, in_channels=3, num_classes=2, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        for feature in features:
            self.downs.append(UNetBlock(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = UNetBlock(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.ups.append(UNetBlock(feature * 2, feature))

        self.final = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)

        return self.final(x)


class SegmentationDataset(Dataset):
    """PyTorch dataset for segmentation training."""

    def __init__(self, samples, transform=None, target_size=(512, 512)):
        self.samples = samples
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = sample.load_image()
        mask = sample.load_mask()

        # Resize
        import cv2
        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


@ModelRegistry.register("unet")
class UNetModel(BaseExpertModel):
    """UNet expert model implementation."""

    name = "unet"

    def __init__(self, num_classes: int, class_names: List[str]):
        super().__init__(num_classes, class_names)
        # +1 for background class
        self.model = UNet(in_channels=3, num_classes=num_classes + 1)
        self.device = None

    def _get_device(self, config_device: str) -> torch.device:
        if config_device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(config_device)

    def train(
        self,
        train_samples: List,
        val_samples: List,
        config: TrainingConfig,
        checkpoint_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> Dict:
        self.device = self._get_device(config.device)
        self.model = self.model.to(self.device)

        # Create datasets
        train_ds = SegmentationDataset(train_samples)
        val_ds = SegmentationDataset(val_samples)

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        metrics_history = []

        for epoch in range(config.epochs):
            # Train
            self.model.train()
            train_loss = 0
            for images, masks in train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            self.model.eval()
            val_loss = 0
            val_iou = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    outputs = self.model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

                    # Compute IoU
                    preds = outputs.argmax(dim=1)
                    intersection = ((preds == masks) & (masks > 0)).sum().item()
                    union = ((preds > 0) | (masks > 0)).sum().item()
                    val_iou += intersection / max(union, 1)

            val_loss /= len(val_loader)
            val_iou /= len(val_loader)

            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_iou,
            }
            metrics_history.append(metrics)

            if progress_callback:
                progress_callback(epoch + 1, metrics)

            # Checkpointing
            if (epoch + 1) % config.checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1}.pth")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(checkpoint_dir / "best.pth")
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Save final checkpoint
        self.save_checkpoint(checkpoint_dir / "final.pth")

        return {
            "final_epoch": epoch + 1,
            "best_val_loss": best_val_loss,
            "final_val_iou": val_iou,
            "history": metrics_history,
        }

    def predict(
        self,
        images: Iterator[np.ndarray],
        config: InferenceConfig,
    ) -> Iterator[np.ndarray]:
        self.device = self._get_device(config.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            for image in images:
                # Preprocess
                orig_h, orig_w = image.shape[:2]
                import cv2
                resized = cv2.resize(image, (512, 512))
                tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                tensor = tensor.unsqueeze(0).to(self.device)

                # Predict
                output = self.model(tensor)
                pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

                # Resize back
                pred = cv2.resize(pred.astype(np.uint16), (orig_w, orig_h),
                                  interpolation=cv2.INTER_NEAREST)

                yield pred

    def load_checkpoint(self, checkpoint_path: Path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict)

    def save_checkpoint(self, checkpoint_path: Path):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)

    @classmethod
    def get_default_config(cls) -> Dict:
        return {
            "features": [64, 128, 256, 512],
            "target_size": [512, 512],
        }
```

#### 3.5 Detectron2 Implementation (`src/napari_sam2long/training/models/detectron2_model.py`)

```python
"""Detectron2 implementation for instance segmentation."""

from pathlib import Path
from typing import Dict, Iterator, List, Optional
import numpy as np
import json
import tempfile

from ..model_registry import BaseExpertModel, ModelRegistry, TrainingConfig, InferenceConfig


@ModelRegistry.register("detectron2")
class Detectron2Model(BaseExpertModel):
    """Detectron2 expert model for instance segmentation."""

    name = "detectron2"

    def __init__(self, num_classes: int, class_names: List[str]):
        super().__init__(num_classes, class_names)
        self.cfg = None
        self.predictor = None

    def _setup_config(self, checkpoint_dir: Path, config: TrainingConfig):
        """Set up Detectron2 configuration."""
        from detectron2.config import get_cfg
        from detectron2 import model_zoo

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        cfg.DATASETS.TRAIN = ("train_dataset",)
        cfg.DATASETS.TEST = ("val_dataset",)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.SOLVER.IMS_PER_BATCH = config.batch_size
        cfg.SOLVER.BASE_LR = config.learning_rate
        cfg.SOLVER.MAX_ITER = config.epochs * 100  # Approximate
        cfg.SOLVER.CHECKPOINT_PERIOD = config.checkpoint_interval * 100
        cfg.OUTPUT_DIR = str(checkpoint_dir)

        # Device
        if config.device == "auto":
            import torch
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            cfg.MODEL.DEVICE = config.device

        self.cfg = cfg
        return cfg

    def train(
        self,
        train_samples: List,
        val_samples: List,
        config: TrainingConfig,
        checkpoint_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> Dict:
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.engine import DefaultTrainer
        from detectron2.evaluation import COCOEvaluator

        # Export samples to COCO format for Detectron2
        from ..data_handler import TrainingDataHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create temporary handler to export COCO
            handler = TrainingDataHandler.__new__(TrainingDataHandler)
            handler.class_names = self.class_names

            train_json = tmpdir / "train.json"
            val_json = tmpdir / "val.json"
            images_dir = tmpdir / "images"
            images_dir.mkdir()

            # Export train samples
            handler._samples = train_samples
            handler._run_ids = []
            handler.export_coco(str(train_json), copy_images=True, images_dir=str(images_dir))

            # Export val samples
            handler._samples = val_samples
            handler.export_coco(str(val_json), copy_images=True, images_dir=str(images_dir))

            # Register datasets
            def get_train_dicts():
                with open(train_json) as f:
                    return json.load(f)

            def get_val_dicts():
                with open(val_json) as f:
                    return json.load(f)

            DatasetCatalog.register("train_dataset", get_train_dicts)
            DatasetCatalog.register("val_dataset", get_val_dicts)
            MetadataCatalog.get("train_dataset").set(thing_classes=self.class_names)
            MetadataCatalog.get("val_dataset").set(thing_classes=self.class_names)

            # Setup config
            cfg = self._setup_config(checkpoint_dir, config)

            # Train
            trainer = DefaultTrainer(cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()

            # Cleanup registered datasets
            DatasetCatalog.remove("train_dataset")
            DatasetCatalog.remove("val_dataset")

        # Get final metrics
        return {
            "final_checkpoint": str(checkpoint_dir / "model_final.pth"),
        }

    def predict(
        self,
        images: Iterator[np.ndarray],
        config: InferenceConfig,
    ) -> Iterator[np.ndarray]:
        from detectron2.engine import DefaultPredictor

        if self.predictor is None:
            raise RuntimeError("Must load checkpoint before prediction")

        for image in images:
            outputs = self.predictor(image)

            # Convert instances to indexed mask
            instances = outputs["instances"]
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint16)

            if len(instances) > 0:
                pred_masks = instances.pred_masks.cpu().numpy()
                scores = instances.scores.cpu().numpy()

                # Filter by threshold and assign instance IDs
                for i, (m, s) in enumerate(zip(pred_masks, scores)):
                    if s >= config.threshold:
                        mask[m] = i + 1

            yield mask

    def load_checkpoint(self, checkpoint_path: Path):
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2 import model_zoo

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.MODEL.WEIGHTS = str(checkpoint_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        import torch
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)

    def save_checkpoint(self, checkpoint_path: Path):
        # Detectron2 handles checkpointing internally during training
        pass

    @classmethod
    def get_default_config(cls) -> Dict:
        return {
            "backbone": "R_50_FPN_3x",
            "score_threshold": 0.5,
        }
```

---

### 4. UI Components

#### 4.1 Run Manager Widget (`src/napari_sam2long/widgets/run_manager_widget.py`)

```python
"""Widget for managing annotation/training/inference runs."""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QListWidget, QListWidgetItem,
    QDialog, QDialogButtonBox, QLineEdit, QTextEdit, QMessageBox,
    QProgressDialog,
)
from qtpy.QtCore import Qt, Signal
from pathlib import Path

from ..ingestion.run_manager import RunManager, RunType, RunInfo


class RunManagerWidget(QWidget):
    """Widget for viewing and managing runs."""

    run_selected = Signal(str)  # Emits run_id when user selects a run to load

    def __init__(self, parent=None):
        super().__init__(parent)
        self.run_manager: RunManager = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Run list
        self.run_list = QListWidget()
        self.run_list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(QLabel("Annotation/Inference Runs:"))
        layout.addWidget(self.run_list)

        # Run info display
        self.run_info = QTextEdit()
        self.run_info.setReadOnly(True)
        self.run_info.setMaximumHeight(100)
        layout.addWidget(self.run_info)

        # Action buttons
        btn_layout = QHBoxLayout()

        self.save_btn = QPushButton("Save Current → New Run")
        self.save_btn.clicked.connect(self._save_current_to_run)
        btn_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("Load Selected Run")
        self.load_btn.clicked.connect(self._load_selected_run)
        self.load_btn.setEnabled(False)
        btn_layout.addWidget(self.load_btn)

        layout.addLayout(btn_layout)

        # Training section
        train_group = QGroupBox("Expert Model Training")
        train_layout = QVBoxLayout(train_group)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["unet", "detectron2"])
        model_row.addWidget(self.model_combo)
        train_layout.addLayout(model_row)

        self.train_btn = QPushButton("Train on Selected Run(s)")
        self.train_btn.clicked.connect(self._start_training)
        train_layout.addWidget(self.train_btn)

        self.infer_btn = QPushButton("Run Inference (All Videos)")
        self.infer_btn.clicked.connect(self._run_inference)
        train_layout.addWidget(self.infer_btn)

        layout.addWidget(train_group)

    def set_project(self, project_root: Path):
        """Set the project and load runs."""
        self.run_manager = RunManager(project_root)
        self._refresh_run_list()

    def _refresh_run_list(self):
        self.run_list.clear()
        if not self.run_manager:
            return

        for run in self.run_manager.list_runs():
            item = QListWidgetItem(f"{run.id} ({run.type.value})")
            item.setData(Qt.UserRole, run.id)

            # Color code by type
            if run.type == RunType.MANUAL:
                item.setForeground(Qt.blue)
            elif run.type == RunType.TRAINING:
                item.setForeground(Qt.darkGreen)
            elif run.type == RunType.INFERENCE:
                item.setForeground(Qt.darkMagenta)

            self.run_list.addItem(item)

    def _on_selection_changed(self):
        items = self.run_list.selectedItems()
        self.load_btn.setEnabled(len(items) > 0)

        if items:
            run_id = items[0].data(Qt.UserRole)
            run = self.run_manager.get_run(run_id)
            if run:
                info = f"Type: {run.type.value}\n"
                info += f"Created: {run.created_at.strftime('%Y-%m-%d %H:%M')}\n"
                info += f"Description: {run.description}\n"
                if run.parent_run:
                    info += f"Parent: {run.parent_run}\n"
                if run.final_metrics:
                    info += f"Metrics: {run.final_metrics}"
                self.run_info.setText(info)

    def _save_current_to_run(self):
        """Save current working masks to a new run."""
        if not self.run_manager:
            return

        # Get description from user
        desc, ok = QInputDialog.getText(
            self, "Save Run", "Description for this annotation run:"
        )
        if not ok:
            return

        # Create new manual run
        run = self.run_manager.create_run(
            RunType.MANUAL,
            description=desc or "Manual annotation",
        )

        # Save working masks to run
        self.run_manager.save_working_to_run(run.id)

        self._refresh_run_list()
        QMessageBox.information(self, "Saved", f"Saved to run: {run.id}")

    def _load_selected_run(self):
        """Load selected run's masks into working directory."""
        items = self.run_list.selectedItems()
        if not items:
            return

        run_id = items[0].data(Qt.UserRole)
        run = self.run_manager.get_run(run_id)

        if run.type == RunType.TRAINING:
            QMessageBox.warning(
                self, "Cannot Load",
                "Training runs don't contain masks. Select a manual or inference run."
            )
            return

        reply = QMessageBox.question(
            self, "Load Run",
            f"Load masks from '{run_id}'?\n\nThis will overwrite current working masks.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.run_manager.load_run_to_working(run_id)
            self.run_selected.emit(run_id)
            QMessageBox.information(self, "Loaded", f"Loaded masks from: {run_id}")

    def _start_training(self):
        """Start training on selected run(s)."""
        # Implementation would spawn training dialog/worker
        pass

    def _run_inference(self):
        """Run inference using selected training run's model."""
        # Implementation would spawn inference dialog/worker
        pass
```

---

### 5. Integration with Main Widget

#### Changes to `_widget.py`

```python
# Add to imports
from .widgets.run_manager_widget import RunManagerWidget
from .ingestion.run_manager import RunManager, RunType

# Add to __init__ after project manager setup
def _setup_run_manager_ui(self):
    """Set up the Run Manager UI section."""
    self.run_manager_widget = RunManagerWidget(self)
    self.run_manager_widget.run_selected.connect(self._on_run_loaded)

    # Insert into layout
    main_layout = self.layout()
    if main_layout:
        # Find project manager group and insert after it
        main_layout.insertWidget(1, self.run_manager_widget)

def _on_run_loaded(self, run_id: str):
    """Handle when a run is loaded - reload current video."""
    if self.current_video_id:
        self.load_current_video()
    show_info(f"Loaded masks from run: {run_id}")

# Modify load_project_folder to initialize run manager
def load_project_folder(self):
    # ... existing code ...

    # Initialize run manager
    self.run_manager_widget.set_project(Path(folder))
```

---

### 6. Complete Active Learning Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    ACTIVE LEARNING LOOP                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. INITIAL ANNOTATION (run_001_manual)                          │
│     ┌──────────────────────────────────────────────┐             │
│     │ • Load project in napari                      │             │
│     │ • Draw sparse annotations on key frames       │             │
│     │ • Use SAM2 to propagate through video         │             │
│     │ • Refine propagated masks manually            │             │
│     │ • Save → Creates run_001_manual               │             │
│     └──────────────────────────────────────────────┘             │
│                           ↓                                       │
│  2. TRAIN EXPERT MODEL (run_002_training)                        │
│     ┌──────────────────────────────────────────────┐             │
│     │ • Select run_001_manual as training data      │             │
│     │ • Choose model (UNet / Detectron2)            │             │
│     │ • Configure training params                   │             │
│     │ • Train → Saves checkpoints to run_002        │             │
│     │ • View training metrics                       │             │
│     └──────────────────────────────────────────────┘             │
│                           ↓                                       │
│  3. EXPERT INFERENCE (run_003_inference)                         │
│     ┌──────────────────────────────────────────────┐             │
│     │ • Select run_002_training checkpoint          │             │
│     │ • Run inference on ALL project videos         │             │
│     │ • Saves predictions to run_003_inference      │             │
│     │ • Much faster than SAM2 propagation           │             │
│     └──────────────────────────────────────────────┘             │
│                           ↓                                       │
│  4. LOAD & REFINE (run_004_manual)                               │
│     ┌──────────────────────────────────────────────┐             │
│     │ • Load run_003_inference masks into napari    │             │
│     │ • Review model predictions                    │             │
│     │ • Fix errors with manual editing              │             │
│     │ • Use SAM2 propagation where needed           │             │
│     │ • Approve/refine more frames (higher volume)  │             │
│     │ • Save → Creates run_004_manual               │             │
│     └──────────────────────────────────────────────┘             │
│                           ↓                                       │
│  5. RETRAIN (run_005_training)                                   │
│     ┌──────────────────────────────────────────────┐             │
│     │ • Combine run_001 + run_004 as training data  │             │
│     │ • More data → Better model                    │             │
│     │ • Fine-tune from run_002 checkpoint           │             │
│     │ • Save new checkpoints                        │             │
│     └──────────────────────────────────────────────┘             │
│                           ↓                                       │
│                    ↻ REPEAT FROM STEP 3                          │
│                                                                   │
│  Each cycle:                                                      │
│  • Model improves with more training data                        │
│  • Inference covers more videos faster                           │
│  • Human effort focuses on corrections only                      │
│  • Quality improves exponentially                                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

### 7. File Structure Summary

```
src/napari_sam2long/
├── __init__.py
├── _widget.py                    # Main widget (extended)
├── export_curation.py            # Existing
├── napari.yaml                   # Existing
│
├── ingestion/                    # Existing + extended
│   ├── __init__.py
│   ├── project.py                # Existing
│   ├── config.py                 # Existing
│   ├── mask_io.py                # Existing
│   ├── coco_export.py            # Existing (add mask_to_rle, mask_to_bbox)
│   ├── ingester.py               # Existing
│   ├── organizer.py              # Existing
│   ├── cli.py                    # Existing
│   └── run_manager.py            # NEW: Run tracking
│
├── training/                     # NEW: Training subsystem
│   ├── __init__.py
│   ├── data_handler.py           # Training data collection
│   ├── model_registry.py         # Model abstraction
│   └── models/
│       ├── __init__.py
│       ├── unet.py               # UNet implementation
│       └── detectron2_model.py   # Detectron2 wrapper
│
└── widgets/                      # NEW: Modular widgets
    ├── __init__.py
    └── run_manager_widget.py     # Run management UI
```

---

### 8. Implementation Phases

#### Phase 1: Run Management (Foundation)
- [ ] `RunManager` class with create/load/save
- [ ] Run manifest persistence
- [ ] Basic UI for viewing runs
- [ ] "Save to Run" and "Load from Run" buttons

#### Phase 2: Training Data Handler
- [ ] `TrainingDataHandler` class
- [ ] Multi-run aggregation
- [ ] COCO export for training
- [ ] Train/val split utilities

#### Phase 3: Model Integration
- [ ] `BaseExpertModel` abstract class
- [ ] `ModelRegistry` for model discovery
- [ ] UNet implementation
- [ ] Detectron2 wrapper

#### Phase 4: Training UI
- [ ] Training configuration dialog
- [ ] Progress monitoring
- [ ] Checkpoint selection
- [ ] Metrics visualization

#### Phase 5: Inference Pipeline
- [ ] Batch inference runner
- [ ] Progress tracking
- [ ] Results → Run conversion

#### Phase 6: Polish
- [ ] Lineage visualization
- [ ] Run comparison tools
- [ ] Documentation

---

### 9. Key Design Decisions

1. **Runs are immutable snapshots**: Once created, a run's masks are never modified. Edits create new runs.

2. **Working directory is ephemeral**: `vids/*/masks` is the "scratch space" loaded from runs.

3. **Lineage tracking**: Every run knows its parent, enabling reproducibility.

4. **Model-agnostic interface**: `BaseExpertModel` allows easy addition of new models.

5. **COCO as interchange format**: Standardized format works with most training frameworks.

6. **Checkpoints are versioned**: Each training run keeps all checkpoints for comparison.

7. **UI is modular**: `RunManagerWidget` can be enabled/disabled without affecting core annotation.

---

### 10. Dependencies to Add

```toml
# pyproject.toml additions
[project.optional-dependencies]
training = [
    "torch>=2.0",
    "torchvision>=0.15",
    "detectron2>=0.6",  # Optional, for Detectron2 model
    "albumentations>=1.3",  # For data augmentation
]
```

---

This architecture preserves the spirit of the existing codebase while adding powerful active learning capabilities. The key insight is that runs provide version control for annotations, enabling the iterative refinement loop you described.
