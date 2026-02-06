"""
Handler for segmentation-models-pytorch (UNet, FPN, DeepLabV3+, etc.)

Uses the `segmentation-models-pytorch` library which provides:
- Multiple architectures: Unet, UnetPlusPlus, FPN, PSPNet, DeepLabV3, DeepLabV3+
- Multiple encoders: ResNet, EfficientNet, etc.
- Pretrained ImageNet weights

Install: pip install segmentation-models-pytorch
"""

import json
from pathlib import Path
from typing import Iterator, Optional, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .base import (
    BaseModelHandler,
    TrainConfig,
    InferConfig,
    register_handler,
)


class COCOSegmentationDataset(Dataset):
    """
    PyTorch dataset that reads COCO JSON for semantic segmentation.

    Converts instance masks to semantic segmentation by using category_id
    as the pixel value.
    """

    def __init__(
        self,
        coco_json: Path,
        img_size: tuple[int, int] = (512, 512),
        augment: bool = False,
    ):
        from pycocotools.coco import COCO

        self.coco = COCO(str(coco_json))
        self.img_ids = list(self.coco.imgs.keys())
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]

        # Load image
        img_path = img_info["file_name"]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]

        # Load annotations and create mask
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((orig_h, orig_w), dtype=np.int64)
        for ann in anns:
            m = self.coco.annToMask(ann)
            cat_id = ann["category_id"]
            mask[m > 0] = cat_id

        # Resize
        img = cv2.resize(img, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # Simple augmentation
        if self.augment:
            if np.random.random() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()

        # To tensor: (H, W, C) -> (C, H, W), normalized to [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        return img, mask


@register_handler("unet")
class SMPHandler(BaseModelHandler):
    """
    UNet handler using segmentation-models-pytorch.

    Supports multiple architectures and encoders via the 'extra' config:
        config.extra = {
            "architecture": "Unet",  # or "UnetPlusPlus", "FPN", "DeepLabV3Plus"
            "encoder": "resnet34",   # or "efficientnet-b0", etc.
        }
    """

    name = "unet"

    def __init__(self, num_classes: int, class_names: list[str]):
        super().__init__(num_classes, class_names)
        self.model = None
        self.device = None

    def _build_model(self, config: TrainConfig):
        """Build model from config."""
        import segmentation_models_pytorch as smp

        arch = config.extra.get("architecture", "Unet")
        encoder = config.extra.get("encoder", "resnet34")

        # Map architecture name to smp class
        arch_map = {
            "Unet": smp.Unet,
            "UnetPlusPlus": smp.UnetPlusPlus,
            "FPN": smp.FPN,
            "PSPNet": smp.PSPNet,
            "DeepLabV3": smp.DeepLabV3,
            "DeepLabV3Plus": smp.DeepLabV3Plus,
        }

        if arch not in arch_map:
            raise ValueError(f"Unknown architecture: {arch}. Available: {list(arch_map.keys())}")

        model_cls = arch_map[arch]
        self.model = model_cls(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.num_classes + 1,  # +1 for background (class 0)
        )

    def _get_device(self, config_device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if config_device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(config_device)

    def train(
        self,
        coco_train_json: Path,
        coco_val_json: Path,
        checkpoint_dir: Path,
        config: TrainConfig,
        progress_callback: Optional[Callable[[int, dict], None]] = None,
    ) -> dict:
        import segmentation_models_pytorch as smp

        # Build model
        self._build_model(config)
        self.device = self._get_device(config.device)
        self.model = self.model.to(self.device)

        # Create datasets
        train_ds = COCOSegmentationDataset(
            coco_train_json,
            img_size=config.img_size,
            augment=True,
        )
        val_ds = COCOSegmentationDataset(
            coco_val_json,
            img_size=config.img_size,
            augment=False,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if self.device.type == "cuda" else False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Loss and optimizer
        loss_fn = smp.losses.DiceLoss(mode="multiclass")
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        # Training loop
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        history = []

        for epoch in range(config.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for imgs, masks in train_loader:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                preds = self.model(imgs)
                loss = loss_fn(preds, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_iou = 0.0

            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs = imgs.to(self.device)
                    masks = masks.to(self.device)

                    preds = self.model(imgs)
                    loss = loss_fn(preds, masks)
                    val_loss += loss.item()

                    # Compute IoU (foreground only)
                    pred_mask = preds.argmax(dim=1)
                    for cls in range(1, self.num_classes + 1):
                        pred_cls = pred_mask == cls
                        gt_cls = masks == cls
                        intersection = (pred_cls & gt_cls).sum().item()
                        union = (pred_cls | gt_cls).sum().item()
                        if union > 0:
                            val_iou += intersection / union

            val_loss /= len(val_loader)
            val_iou /= len(val_loader) * self.num_classes  # Average over classes

            # Step scheduler
            scheduler.step(val_loss)

            # Log metrics
            metrics = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_iou": round(val_iou, 4),
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(metrics)

            if progress_callback:
                progress_callback(epoch + 1, metrics)

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(checkpoint_dir / "best.pth")

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1:03d}.pth")

        # Save final checkpoint
        self.save_checkpoint(checkpoint_dir / "final.pth")

        # Save training history
        history_path = checkpoint_dir.parent / "metrics" / "training_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        return {
            "final_epoch": config.epochs,
            "best_val_loss": round(best_val_loss, 4),
            "final_val_loss": round(val_loss, 4),
            "final_val_iou": round(val_iou, 4),
            "best_checkpoint": str(checkpoint_dir / "best.pth"),
            "history": history,
        }

    def predict(
        self,
        images: Iterator[tuple[str, np.ndarray]],
        config: InferConfig,
    ) -> Iterator[tuple[str, np.ndarray]]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        self.device = self._get_device(config.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            for frame_id, img in images:
                orig_h, orig_w = img.shape[:2]

                # Preprocess
                resized = cv2.resize(img, config.img_size)
                tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                tensor = tensor.unsqueeze(0).to(self.device)

                # Predict
                output = self.model(tensor)
                pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

                # Resize back to original size
                pred = cv2.resize(
                    pred.astype(np.uint16),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                )

                yield frame_id, pred

    def load_checkpoint(self, path: Path):
        """Load model weights from checkpoint."""
        if self.model is None:
            # Build with default config if not yet built
            self._build_model(TrainConfig())

        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)

    def save_checkpoint(self, path: Path):
        """Save model weights to checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    @classmethod
    def get_default_config(cls) -> dict:
        return {
            "architecture": "Unet",
            "encoder": "resnet34",
        }
