"""
Training module for napari-sam2long.

Provides a unified API for:
- Collecting training data from annotation runs
- Training expert models (UNet, Mask R-CNN)
- Running inference
- Evaluating results with COCO metrics

Quick Start:
    from napari_sam2long.training import (
        TrainingDataHandler,
        train_model,
        run_inference,
        evaluate_coco,
        list_handlers,
    )

    # 1. Collect training data
    handler = TrainingDataHandler(project_root)
    handler.add_run("annot", 1)
    train_json, val_json = handler.export_train_val_coco(output_dir)

    # 2. Train model
    metrics = train_model(
        handler_name="unet",
        coco_train=train_json,
        coco_val=val_json,
        checkpoint_dir=checkpoint_dir,
        num_classes=2,
        class_names=["cell", "debris"],
    )

    # 3. Run inference
    for frame_id, mask in run_inference(
        handler_name="unet",
        checkpoint=checkpoint_dir / "best.pth",
        images=image_iterator(),
        num_classes=2,
        class_names=["cell", "debris"],
    ):
        save_mask(mask, f"{frame_id}.png")

    # 4. Evaluate
    results = evaluate_coco(pred_json, gt_json)
    print(f"AP: {results['AP']}, AP50: {results['AP50']}")
"""

from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from .handlers import (
    BaseModelHandler,
    TrainConfig,
    InferConfig,
    get_handler,
    list_handlers,
    create_handler,
)
from .data_handler import TrainingDataHandler, SplitManager, TrainingSample
from .evaluator import evaluate_coco, evaluate_masks_simple, compute_iou, compute_dice


def train_model(
    handler_name: str,
    coco_train: Path,
    coco_val: Path,
    checkpoint_dir: Path,
    num_classes: int,
    class_names: list[str],
    config: TrainConfig = None,
    progress_callback: Optional[Callable[[int, dict], None]] = None,
) -> dict:
    """
    Train a model using the specified handler.

    Args:
        handler_name: Handler name ("unet", "maskrcnn")
        coco_train: Path to training COCO JSON
        coco_val: Path to validation COCO JSON
        checkpoint_dir: Where to save checkpoints
        num_classes: Number of foreground classes
        class_names: List of class names
        config: Training configuration (optional)
        progress_callback: Callback fn(epoch, metrics) for progress updates

    Returns:
        Final metrics dict
    """
    handler = create_handler(handler_name, num_classes, class_names)
    return handler.train(
        coco_train_json=Path(coco_train),
        coco_val_json=Path(coco_val),
        checkpoint_dir=Path(checkpoint_dir),
        config=config or TrainConfig(),
        progress_callback=progress_callback,
    )


def run_inference(
    handler_name: str,
    checkpoint: Path,
    images: Iterator[tuple[str, np.ndarray]],
    num_classes: int,
    class_names: list[str],
    config: InferConfig = None,
) -> Iterator[tuple[str, np.ndarray]]:
    """
    Run inference using the specified handler.

    Args:
        handler_name: Handler name ("unet", "maskrcnn")
        checkpoint: Path to model checkpoint
        images: Iterator of (frame_id, RGB image array) tuples
        num_classes: Number of foreground classes
        class_names: List of class names
        config: Inference configuration (optional)

    Yields:
        (frame_id, mask) tuples where mask is indexed uint16 array
    """
    handler = create_handler(handler_name, num_classes, class_names)
    handler.load_checkpoint(Path(checkpoint))
    yield from handler.predict(images, config or InferConfig())


__all__ = [
    # Handlers
    "BaseModelHandler",
    "TrainConfig",
    "InferConfig",
    "get_handler",
    "list_handlers",
    "create_handler",
    # Data handling
    "TrainingDataHandler",
    "SplitManager",
    "TrainingSample",
    # High-level API
    "train_model",
    "run_inference",
    # Evaluation
    "evaluate_coco",
    "evaluate_masks_simple",
    "compute_iou",
    "compute_dice",
]
