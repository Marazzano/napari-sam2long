"""Configuration dataclasses for video ingestion."""

from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Optional
import json


@dataclass(frozen=True)
class ProjectConfig:
    """Configuration for a project (batch ingestion)."""
    source_folder: str
    output_folder: str
    class_names: Tuple[str, ...] = ()
    step_size: int = 10
    reverse: bool = True
    image_format: str = "jpg"
    image_quality: int = 95
    video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["class_names"] = list(d["class_names"])
        d["video_extensions"] = list(d["video_extensions"])
        return d

    @classmethod
    def from_json(cls, path: str) -> "ProjectConfig":
        with open(path) as f:
            d = json.load(f)
        d["class_names"] = tuple(d.get("class_names", []))
        d["video_extensions"] = tuple(d.get("video_extensions", [".mp4", ".avi", ".mov", ".mkv"]))
        return cls(**d)


@dataclass(frozen=True)
class IngestionConfig:
    """Configuration for single video ingestion."""
    video_path: str
    output_dir: str
    class_names: Tuple[str, ...] = ()
    step_size: int = 10
    reverse: bool = True
    image_format: str = "jpg"
    image_quality: int = 95
