"""
Base class and registry for model handlers.

Each handler encapsulates:
- Loading COCO data in model-specific way
- Training loop
- Checkpoint save/load
- Inference loop

This allows adding new models with a single file that implements BaseModelHandler.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Callable
import numpy as np


@dataclass
class TrainConfig:
    """Configuration for model training."""

    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-3
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    img_size: tuple[int, int] = (512, 512)
    # Model-specific options
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "device": self.device,
            "img_size": self.img_size,
            "extra": self.extra,
        }


@dataclass
class InferConfig:
    """Configuration for model inference."""

    batch_size: int = 4
    threshold: float = 0.5
    device: str = "auto"
    img_size: tuple[int, int] = (512, 512)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "threshold": self.threshold,
            "device": self.device,
            "img_size": self.img_size,
            "extra": self.extra,
        }


class BaseModelHandler(ABC):
    """
    Abstract base class for model handlers.

    Each handler owns training + inference for one model type.
    Handlers read COCO format and produce indexed masks.

    Usage:
        handler = SMPHandler(num_classes=2, class_names=["cell", "debris"])
        metrics = handler.train(train_coco, val_coco, checkpoint_dir, config)
        handler.load_checkpoint(checkpoint_dir / "best.pth")
        for frame_id, mask in handler.predict(image_iterator, config):
            save_mask(mask, f"{frame_id}.png")
    """

    name: str = "base"

    def __init__(self, num_classes: int, class_names: list[str]):
        """
        Initialize handler.

        Args:
            num_classes: Number of foreground classes (excluding background)
            class_names: List of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names

    @abstractmethod
    def train(
        self,
        coco_train_json: Path,
        coco_val_json: Path,
        checkpoint_dir: Path,
        config: TrainConfig,
        progress_callback: Optional[Callable[[int, dict], None]] = None,
    ) -> dict:
        """
        Train model from COCO annotations.

        Args:
            coco_train_json: Path to training COCO JSON
            coco_val_json: Path to validation COCO JSON
            checkpoint_dir: Where to save checkpoints
            config: Training configuration
            progress_callback: Optional fn(epoch, metrics) for UI updates

        Returns:
            Final metrics dict {val_loss, val_iou, best_checkpoint, ...}
        """
        pass

    @abstractmethod
    def predict(
        self,
        images: Iterator[tuple[str, np.ndarray]],
        config: InferConfig,
    ) -> Iterator[tuple[str, np.ndarray]]:
        """
        Run inference on images.

        Args:
            images: Iterator of (frame_id, RGB numpy array H×W×3) tuples
            config: Inference configuration

        Yields:
            (frame_id, indexed_mask) tuples where mask is uint16 H×W
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path: Path):
        """Load model weights from checkpoint file."""
        pass

    @abstractmethod
    def save_checkpoint(self, path: Path):
        """Save model weights to checkpoint file."""
        pass

    @classmethod
    def get_default_config(cls) -> dict:
        """Get default model-specific configuration options."""
        return {}


# ---------------------------------------------------------------------------
# Handler Registry
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, type[BaseModelHandler]] = {}


def register_handler(name: str):
    """
    Decorator to register a handler class.

    Usage:
        @register_handler("unet")
        class UNetHandler(BaseModelHandler):
            ...
    """
    def decorator(cls: type[BaseModelHandler]):
        cls.name = name
        _HANDLERS[name] = cls
        return cls
    return decorator


def get_handler(name: str) -> type[BaseModelHandler]:
    """
    Get a handler class by name.

    Args:
        name: Handler name (e.g., "unet", "maskrcnn")

    Returns:
        Handler class (not instance)

    Raises:
        ValueError: If handler name not found
    """
    if name not in _HANDLERS:
        available = list(_HANDLERS.keys())
        raise ValueError(f"Unknown handler: '{name}'. Available: {available}")
    return _HANDLERS[name]


def list_handlers() -> list[str]:
    """List all registered handler names."""
    return list(_HANDLERS.keys())


def create_handler(
    name: str,
    num_classes: int,
    class_names: list[str],
) -> BaseModelHandler:
    """
    Create a handler instance by name.

    Args:
        name: Handler name
        num_classes: Number of foreground classes
        class_names: List of class names

    Returns:
        Handler instance
    """
    handler_cls = get_handler(name)
    return handler_cls(num_classes, class_names)
