"""
Model handlers for training and inference.

Each handler encapsulates the full train/infer cycle for one model type.
Handlers are registered via decorator and accessed via the registry.

Usage:
    from napari_sam2long.training.handlers import get_handler, list_handlers

    # List available handlers
    print(list_handlers())  # ['unet', 'maskrcnn']

    # Create handler
    handler_cls = get_handler("unet")
    handler = handler_cls(num_classes=2, class_names=["cell", "debris"])

    # Train
    metrics = handler.train(train_json, val_json, checkpoint_dir, config)

    # Inference
    handler.load_checkpoint(checkpoint_dir / "best.pth")
    for frame_id, mask in handler.predict(images, infer_config):
        save_mask(mask, f"{frame_id}.png")
"""

from .base import (
    BaseModelHandler,
    TrainConfig,
    InferConfig,
    register_handler,
    get_handler,
    list_handlers,
    create_handler,
)

# Import handlers to trigger registration
from .smp_handler import SMPHandler

# Detectron2 is optional (heavy dependency)
try:
    from .detectron2_handler import Detectron2Handler
except ImportError:
    pass  # Detectron2 not installed

__all__ = [
    "BaseModelHandler",
    "TrainConfig",
    "InferConfig",
    "register_handler",
    "get_handler",
    "list_handlers",
    "create_handler",
    "SMPHandler",
]
