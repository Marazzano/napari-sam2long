"""
Handler for Detectron2 (Mask R-CNN for instance segmentation).

Uses Facebook's Detectron2 library which provides:
- Instance segmentation (separate masks per object)
- Pretrained COCO weights
- Configurable backbone (ResNet, etc.)

Install: pip install detectron2
         (May require building from source depending on CUDA version)
"""

import json
import shutil
from pathlib import Path
from typing import Iterator, Optional, Callable

import numpy as np

from .base import (
    BaseModelHandler,
    TrainConfig,
    InferConfig,
    register_handler,
)


@register_handler("maskrcnn")
class Detectron2Handler(BaseModelHandler):
    """
    Mask R-CNN handler using Detectron2.

    Produces instance segmentation (each object gets unique ID).

    Config options via 'extra':
        config.extra = {
            "backbone": "R_50_FPN_3x",  # or "R_101_FPN_3x"
            "score_threshold": 0.5,
        }
    """

    name = "maskrcnn"

    def __init__(self, num_classes: int, class_names: list[str]):
        super().__init__(num_classes, class_names)
        self.cfg = None
        self.predictor = None
        self._train_dataset_name = None
        self._val_dataset_name = None

    def _get_device(self, config_device: str) -> str:
        """Resolve device string for Detectron2."""
        import torch

        if config_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return config_device

    def train(
        self,
        coco_train_json: Path,
        coco_val_json: Path,
        checkpoint_dir: Path,
        config: TrainConfig,
        progress_callback: Optional[Callable[[int, dict], None]] = None,
    ) -> dict:
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from detectron2.engine import DefaultTrainer
        from detectron2.data.datasets import register_coco_instances
        from detectron2.data import DatasetCatalog

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get image directory from COCO JSON
        with open(coco_train_json) as f:
            coco_data = json.load(f)

        if coco_data["images"]:
            # Images may come from multiple directories; use common parent
            img_dirs = set(
                str(Path(img["file_name"]).parent)
                for img in coco_data["images"]
            )
            if len(img_dirs) == 1:
                img_dir = img_dirs.pop()
            else:
                # Multiple image directories; use the common ancestor
                from os.path import commonpath
                img_dir = commonpath(list(img_dirs))
        else:
            raise ValueError("No images found in training COCO JSON")

        # Register datasets with unique names (avoid conflicts)
        import uuid
        uid = uuid.uuid4().hex[:8]
        self._train_dataset_name = f"train_{uid}"
        self._val_dataset_name = f"val_{uid}"

        # Unregister if exists (for reruns)
        for name in [self._train_dataset_name, self._val_dataset_name]:
            if name in DatasetCatalog:
                DatasetCatalog.remove(name)

        register_coco_instances(
            self._train_dataset_name,
            {},
            str(coco_train_json),
            img_dir,
        )
        register_coco_instances(
            self._val_dataset_name,
            {},
            str(coco_val_json),
            img_dir,
        )

        # Build config
        backbone = config.extra.get("backbone", "R_50_FPN_3x")
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(f"COCO-InstanceSegmentation/mask_rcnn_{backbone}.yaml")
        )
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            f"COCO-InstanceSegmentation/mask_rcnn_{backbone}.yaml"
        )

        cfg.DATASETS.TRAIN = (self._train_dataset_name,)
        cfg.DATASETS.TEST = (self._val_dataset_name,)

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.SOLVER.IMS_PER_BATCH = config.batch_size
        cfg.SOLVER.BASE_LR = config.learning_rate

        # Calculate iterations per epoch: num_images / batch_size
        num_train_images = len(coco_data["images"])
        iters_per_epoch = num_train_images // config.batch_size
        cfg.SOLVER.MAX_ITER = config.epochs * iters_per_epoch
        cfg.SOLVER.CHECKPOINT_PERIOD = max(iters_per_epoch, config.epochs * iters_per_epoch // 2)

        cfg.OUTPUT_DIR = str(checkpoint_dir.parent)  # Detectron2 creates its own structure
        cfg.MODEL.DEVICE = self._get_device(config.device)

        # Disable some augmentations for simplicity
        cfg.INPUT.MIN_SIZE_TRAIN = config.img_size[0]
        cfg.INPUT.MAX_SIZE_TRAIN = config.img_size[0]
        cfg.INPUT.MIN_SIZE_TEST = config.img_size[0]
        cfg.INPUT.MAX_SIZE_TEST = config.img_size[0]

        self.cfg = cfg

        # Use bitmask format for RLE-encoded masks (not polygons)
        from detectron2.data import DatasetMapper
        cfg.INPUT.MASK_FORMAT = "bitmask"

        # Train
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        # Copy model_final.pth to checkpoints/best.pth for consistency
        final_model = Path(cfg.OUTPUT_DIR) / "model_final.pth"
        if final_model.exists():
            shutil.copy(final_model, checkpoint_dir / "best.pth")
            shutil.copy(final_model, checkpoint_dir / "final.pth")

        # Cleanup registered datasets
        try:
            DatasetCatalog.remove(self._train_dataset_name)
            DatasetCatalog.remove(self._val_dataset_name)
        except Exception:
            pass  # Datasets may already be unregistered or never registered

        # Get final training loss from event storage
        final_loss = 0.0
        try:
            # Detectron2 stores metrics in EventStorage
            # storage.latest() returns (value, iteration) tuples
            if hasattr(trainer, 'storage') and trainer.storage:
                loss_data = trainer.storage.latest().get("total_loss", (0.0, 0))
                # Extract the value (first element of tuple)
                final_loss = loss_data[0] if isinstance(loss_data, tuple) else loss_data
        except Exception as e:
            print(f"Warning: Could not retrieve final loss: {e}")
            final_loss = 0.0

        return {
            "final_epoch": config.epochs,
            "best_checkpoint": str(checkpoint_dir / "best.pth"),
            "output_dir": cfg.OUTPUT_DIR,
            "best_val_loss": float(final_loss),
            "final_val_iou": 0.0,
            "train_loss": float(final_loss),
        }

    def predict(
        self,
        images: Iterator[tuple[str, np.ndarray]],
        config: InferConfig,
    ) -> Iterator[tuple[str, np.ndarray]]:
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        import cv2

        threshold = config.extra.get("score_threshold", config.threshold)

        for frame_id, img in images:
            # Detectron2 expects BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            outputs = self.predictor(img_bgr)
            instances = outputs["instances"].to("cpu")

            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint16)

            # Debug: print detection info for first frame
            if frame_id.endswith("frame_00000"):
                print(f"DEBUG {frame_id}: Detected {len(instances)} instances")
                if len(instances) > 0:
                    print(f"  Scores: {instances.scores.numpy()}")
                    print(f"  Classes: {instances.pred_classes.numpy()}")
                    print(f"  Threshold: {threshold}")

            if len(instances) > 0:
                pred_masks = instances.pred_masks.numpy()
                pred_classes = instances.pred_classes.numpy()
                scores = instances.scores.numpy()

                # Merge instances into semantic segmentation
                # Each pixel gets the class ID (1=head, 2=yolk, etc.)
                for m, cls, score in zip(pred_masks, pred_classes, scores):
                    if score >= threshold:
                        # cls is 0-indexed, add 1 to match COCO class IDs
                        class_id = int(cls) + 1
                        mask[m] = class_id

            yield frame_id, mask

    def load_checkpoint(self, path: Path):
        """Load model weights from checkpoint."""
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor

        backbone = "R_50_FPN_3x"  # Default backbone

        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(f"COCO-InstanceSegmentation/mask_rcnn_{backbone}.yaml")
        )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.MODEL.WEIGHTS = str(path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = self._get_device("auto")

        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)

    def save_checkpoint(self, path: Path):
        """Save model weights to checkpoint.

        Note: Detectron2 handles checkpointing internally during training.
        This method is mainly for interface compliance.
        """
        pass

    @classmethod
    def get_default_config(cls) -> dict:
        return {
            "backbone": "R_50_FPN_3x",
            "score_threshold": 0.5,
        }
