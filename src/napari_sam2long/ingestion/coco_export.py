"""COCO format export with RLE encoding (default)."""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import cv2

from .mask_io import read_indexed_mask, split_indexed_mask


def mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    """
    Convert binary mask to uncompressed RLE.

    Args:
        binary_mask: Boolean or binary numpy array

    Returns:
        RLE dict with 'counts' and 'size'
    """
    flat = binary_mask.flatten(order='F')
    runs = []
    prev = 0
    count = 0
    for val in flat:
        if val != prev:
            runs.append(count)
            count = 1
            prev = val
        else:
            count += 1
    runs.append(count)
    if flat[0]:
        runs.insert(0, 0)
    return {"counts": runs, "size": list(binary_mask.shape)}


def compute_bbox(mask: np.ndarray) -> List[float]:
    """
    Compute [x, y, width, height] bounding box.

    Args:
        mask: Binary mask array

    Returns:
        COCO-format bbox [x, y, width, height]
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


class COCOExporter:
    """Export indexed masks to COCO annotation format."""

    def __init__(self, project: "Project"):
        """
        Initialize exporter.

        Args:
            project: Project instance
        """
        self.project = project

    def export(
        self,
        output_path: Optional[str] = None,
        video_ids: Optional[List[str]] = None,
        frames_by_video: Optional[Dict[str, List[int]]] = None,
        use_original_frame_ids: bool = True,
    ) -> Dict[str, Any]:
        """
        Export to COCO format.

        Args:
            output_path: Where to save. Defaults to project exports/coco/annotations.json
            video_ids: Specific videos to export. None = all.
            use_original_frame_ids: Map back to original video frame IDs

        Returns:
            COCO dict
        """
        if output_path is None:
            output_path = str(self.project.export_dir() / "annotations.json")

        if video_ids is None:
            video_ids = self.project.video_ids()

        coco = {
            "info": {"description": "napari-sam2long export", "date_created": datetime.now().isoformat()},
            "images": [],
            "annotations": [],
            "categories": [],
        }

        # Build categories from class names
        class_names = self.project.class_names()
        for idx, name in enumerate(class_names, 1):
            coco["categories"].append({"id": idx, "name": name})
        category_map = {name: idx for idx, name in enumerate(class_names, 1)}

        annotation_id = 1

        for video_id in video_ids:
            metadata = self.project.load_metadata(video_id)
            images_dir = self.project.images_dir(video_id)
            masks_dir = self.project.masks_dir(video_id)

            allowed_frames = None
            if frames_by_video and video_id in frames_by_video:
                allowed_frames = set(frames_by_video[video_id])

            for frame_info in metadata["frames"]:
                seq_id = frame_info["seq"]
                orig_id = frame_info["orig"]
                filename = frame_info["file"]

                if allowed_frames is not None and seq_id not in allowed_frames:
                    continue

                image_id = orig_id if use_original_frame_ids else seq_id

                # Get image dimensions
                img_path = images_dir / filename
                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]

                coco["images"].append({
                    "id": image_id,
                    "file_name": filename,
                    "width": w,
                    "height": h,
                    "video_id": video_id,
                    "seq_id": seq_id,
                })

                # Find masks for this frame
                frame_stem = filename.rsplit(".", 1)[0]  # frame_00000

                for class_name in class_names:
                    mask_path = masks_dir / class_name / f"{frame_stem}.png"
                    if not mask_path.exists():
                        continue

                    indexed_mask = read_indexed_mask(mask_path)
                    instances = split_indexed_mask(indexed_mask)

                    for inst_id, binary_mask in instances.items():
                        coco["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_map[class_name],
                            "segmentation": mask_to_rle(binary_mask),
                            "bbox": compute_bbox(binary_mask),
                            "area": float(binary_mask.sum()),
                            "iscrowd": 0,
                            "instance_id": inst_id,
                        })
                        annotation_id += 1

        with open(output_path, "w") as f:
            json.dump(coco, f)

        return coco
