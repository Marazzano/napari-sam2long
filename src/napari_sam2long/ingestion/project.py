"""Project class - the main interface for working with ingestion projects."""

from pathlib import Path
from typing import Optional, List, Dict, Any
import json
from datetime import datetime
import numpy as np
import cv2


class Project:
    """
    A project folder is the "unit of truth" for napari-sam2long.

    A project contains organized video data with extracted frames, masks,
    and metadata for batch annotation workflows.

    Usage:
        # Load existing project
        proj = Project.load("./processed_data/my_project")

        # Get next video to annotate
        vid = proj.next_pending()

        # Load frames as numpy stack
        frames = proj.load_frames(vid)

        # Update status
        proj.update_status(vid, "annotating")

        # ... perform annotation ...

        # Mark complete
        proj.update_status(vid, "completed")
    """

    def __init__(self, root: Path, config: dict, manifest: dict):
        """
        Initialize Project. Use Project.load() or Project.create() instead.

        Args:
            root: Project root directory
            config: Project configuration dict
            manifest: Project manifest dict
        """
        self.root = root
        self.config = config
        self.manifest = manifest

    @classmethod
    def load(cls, project_root: str) -> "Project":
        """
        Load a project from its root directory.

        Args:
            project_root: Path to project directory

        Returns:
            Project instance
        """
        root = Path(project_root)
        with open(root / "project.json") as f:
            config = json.load(f)
        with open(root / "manifest.json") as f:
            manifest = json.load(f)
        return cls(root, config, manifest)

    @classmethod
    def create(cls, project_root: str, config: "ProjectConfig") -> "Project":
        """
        Create a new project directory structure.

        Args:
            project_root: Path for new project
            config: ProjectConfig instance

        Returns:
            Project instance
        """
        root = Path(project_root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "vids").mkdir(exist_ok=True)
        (root / "exports" / "coco").mkdir(parents=True, exist_ok=True)

        config_dict = config.to_dict()
        manifest = {"videos": {}, "created_at": datetime.now().isoformat()}

        with open(root / "project.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        with open(root / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return cls(root, config_dict, manifest)

    # === Path Resolution ===

    def video_dir(self, video_id: str) -> Path:
        """Get directory for a specific video."""
        return self.root / "vids" / video_id

    def images_dir(self, video_id: str) -> Path:
        """Get images directory for a specific video."""
        return self.video_dir(video_id) / "images"

    def masks_dir(self, video_id: str, class_name: Optional[str] = None) -> Path:
        """
        Get masks directory for a specific video.

        Args:
            video_id: Video identifier
            class_name: Optional class name for class-specific mask dir

        Returns:
            Path to masks directory
        """
        base = self.video_dir(video_id) / "masks"
        return base / class_name if class_name else base

    def metadata_path(self, video_id: str) -> Path:
        """Get metadata.json path for a specific video."""
        return self.video_dir(video_id) / "metadata.json"

    def export_dir(self) -> Path:
        """Get COCO export directory."""
        return self.root / "exports" / "coco"

    # === Video Status Management ===

    def video_ids(self) -> List[str]:
        """List all video IDs in project."""
        return list(self.manifest["videos"].keys())

    def next_pending(self) -> Optional[str]:
        """
        Get next video with status='pending'.

        Returns:
            Video ID or None if no pending videos
        """
        for vid, info in self.manifest["videos"].items():
            if info.get("status") == "pending":
                return vid
        return None

    def update_status(self, video_id: str, status: str) -> None:
        """
        Update video status.

        Allowed transitions:
        - pending -> annotating -> completed
        - annotating -> pending (abandon)
        - completed -> pending (redo)

        Args:
            video_id: Video identifier
            status: New status (pending, annotating, or completed)
        """
        valid_statuses = {"pending", "annotating", "completed"}
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}")
        self.manifest["videos"][video_id]["status"] = status
        self._save_manifest()

    def _save_manifest(self) -> None:
        """Save manifest to disk."""
        with open(self.root / "manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=2)

    # === Frame/Mask Operations ===

    def load_metadata(self, video_id: str) -> dict:
        """
        Load per-video metadata (frame mapping, etc.).

        Args:
            video_id: Video identifier

        Returns:
            Metadata dict with frame mapping and video info
        """
        with open(self.metadata_path(video_id)) as f:
            return json.load(f)

    def load_frames(self, video_id: str) -> np.ndarray:
        """
        Load frame stack as numpy array.

        Args:
            video_id: Video identifier

        Returns:
            Numpy array of shape (T, H, W, C) in RGB order
        """
        image_files = sorted(self.images_dir(video_id).glob("frame_*.jpg"))
        return np.array([cv2.imread(str(f))[..., ::-1] for f in image_files])

    def class_names(self) -> List[str]:
        """Get list of class names from project config."""
        return self.config.get("class_names", [])

    # === Statistics ===

    def stats(self) -> Dict[str, int]:
        """
        Get project statistics.

        Returns:
            Dict with counts: total, pending, annotating, completed
        """
        videos = self.manifest["videos"]
        return {
            "total": len(videos),
            "pending": sum(1 for v in videos.values() if v["status"] == "pending"),
            "annotating": sum(1 for v in videos.values() if v["status"] == "annotating"),
            "completed": sum(1 for v in videos.values() if v["status"] == "completed"),
        }
