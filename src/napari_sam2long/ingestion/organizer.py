"""Batch video processing into project structure."""

from pathlib import Path
from typing import Optional, Callable
from datetime import datetime
import json

from .project import Project
from .config import ProjectConfig, IngestionConfig
from .ingester import VideoIngester


class ProjectOrganizer:
    """
    Batch process a folder of videos into organized project structure.

    Idempotent: safe to rerun. Skips existing videos unless force=True.
    """

    def __init__(self, config: ProjectConfig):
        """
        Initialize organizer.

        Args:
            config: ProjectConfig instance
        """
        self.config = config

    def organize(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        skip_existing: bool = True,
        force: bool = False,
    ) -> Project:
        """
        Process all videos in source_folder.

        Args:
            progress_callback: Optional callback fn(current, total, video_name)
            skip_existing: Skip videos already in manifest (default: True)
            force: Overwrite existing videos (default: False)

        Returns:
            Project instance
        """
        source = Path(self.config.source_folder)
        output = Path(self.config.output_folder)

        # Find video files
        video_files = []
        for ext in self.config.video_extensions:
            video_files.extend(source.glob(f"*{ext}"))
            video_files.extend(source.glob(f"*{ext.upper()}"))
        video_files = sorted(set(video_files))

        # Create or load project
        if (output / "project.json").exists() and not force:
            project = Project.load(str(output))
        else:
            project = Project.create(str(output), self.config)

        # Process each video
        for idx, video_path in enumerate(video_files):
            video_id = f"vid_{idx + 1:03d}"

            if progress_callback:
                progress_callback(idx + 1, len(video_files), video_path.name)

            # Skip if exists and not forcing
            if video_id in project.manifest["videos"] and skip_existing and not force:
                continue

            # Create video-level config
            video_config = IngestionConfig(
                video_path=str(video_path),
                output_dir=str(project.video_dir(video_id)),
                class_names=self.config.class_names,
                step_size=self.config.step_size,
                reverse=self.config.reverse,
                image_format=self.config.image_format,
                image_quality=self.config.image_quality,
            )

            # Ingest video
            ingester = VideoIngester(video_config)
            metadata = ingester.run()

            # Update manifest
            project.manifest["videos"][video_id] = {
                "source": video_path.name,
                "status": "pending",
                "frames_extracted": metadata["total_frames"],
                "original_frames": metadata["original_total"],
                "created_at": datetime.now().isoformat(),
            }

        project._save_manifest()
        return project
