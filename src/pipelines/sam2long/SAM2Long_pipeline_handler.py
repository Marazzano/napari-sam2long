import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
from napari.utils.notifications import show_info, show_warning
from qtpy.QtWidgets import QWidget


# ============================================================================
# Core Helpers for Multi-Label Multi-Frame Conditioning
# ============================================================================

@dataclass(frozen=True)
class VideoPropagateConfig:
    """Configuration for video propagation with multi-anchor support."""
    prob_threshold: float = 0.50
    protect_anchors: bool = True
    max_anchors: int = 10
    max_total_masks: int = 200
    write_to_new_layer_if_non_numpy: bool = True
    output_layer_suffix: str = " [SAM2]"


def materialize_2d_slice(x) -> np.ndarray:
    """Convert sliced array to numpy. Only materializes the passed slice."""
    if hasattr(x, "compute") and callable(getattr(x, "compute")):
        x = x.compute()
    return np.asarray(x)


def get_label_slice_2d(label_layer, viewer, t: int) -> np.ndarray:
    """Get 2D mask from (T,H,W) or (T,Z,H,W) data."""
    data = label_layer.data
    ndim = getattr(data, "ndim", None)
    if ndim == 3:
        return materialize_2d_slice(data[int(t)])
    if ndim == 4:
        z = int(viewer.dims.current_step[1])
        return materialize_2d_slice(data[int(t), int(z)])
    raise ValueError(f"Unsupported ndim={ndim}")


def resolve_output_layer(viewer, src_label_layer, suffix=" [SAM2]"):
    """Return numpy-backed layer for writing. Create new if needed."""
    data = src_label_layer.data
    if isinstance(data, np.ndarray):
        return src_label_layer
    # Create new numpy output layer
    shape = data.shape
    out = np.zeros(shape, dtype=np.int32)
    name = f"{src_label_layer.name}{suffix}"
    existing = {lyr.name for lyr in viewer.layers}
    if name in existing:
        i = 2
        while f"{name} {i}" in existing:
            i += 1
        name = f"{name} {i}"
    return viewer.add_labels(out, name=name, opacity=src_label_layer.opacity)


def _logits_to_label_image(out_obj_ids, out_mask_logits, prob_threshold=0.5):
    """
    Convert SAM2 logits to label image using argmax (no overlap ambiguity).

    Args:
        out_obj_ids: List of object IDs from SAM2
        out_mask_logits: Tensor of shape [K, 1, H, W] or [K, H, W]
        prob_threshold: Minimum probability for a pixel to be assigned

    Returns:
        Label image of shape [H, W] with object IDs
    """
    import torch
    obj_ids = list(out_obj_ids)
    if len(obj_ids) == 0:
        raise ValueError("No object IDs")

    # Normalize shape to [K, H, W]
    logits = out_mask_logits
    if logits.dim() == 4:
        logits = logits.squeeze(1)  # [K,1,H,W] -> [K,H,W]

    probs = torch.sigmoid(logits).cpu().numpy()  # [K, H, W]
    K, H, W = probs.shape

    best_k = probs.argmax(axis=0)  # [H, W]
    best_p = probs.max(axis=0)     # [H, W]

    label_img = np.zeros((H, W), dtype=np.int32)
    confident = best_p >= prob_threshold

    # Map argmax index to object ID
    obj_ids_arr = np.array(obj_ids, dtype=np.int32)
    label_img[confident] = obj_ids_arr[best_k[confident]]

    return label_img


# ============================================================================
# SAM2Long Pipeline Class
# ============================================================================

class SAM2Long_pipeline(QWidget):
    def __init__(
        self,
        napari_viewer,
        main_window_object,
        checkpoint_path,
        model_cfg_name,
    ):
        # Clear and re-initialize hydra to ensure SAM2 configs are found
        from hydra.core.global_hydra import GlobalHydra
        from hydra import initialize_config_module
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        initialize_config_module("sam2", version_base="1.2")

        build_sam2_video_predictor = pytest.importorskip(
            "sam2.build_sam"
        ).build_sam2_video_predictor
        torch = pytest.importorskip("torch")

        super().__init__()
        self.viewer = napari_viewer
        self.mwo = main_window_object

        self.source_frame_dir = (
            None  # Will be set inside process volume function
        )

        ### Allow cpu and mps as well
        # select the device for computation
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")

        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

        sam2_checkpoint = checkpoint_path
        model_cfg = model_cfg_name

        self.predictor = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=device
        )

        self.preprocess_volume()

        self.inference_state = self.predictor.init_state(
            video_path=self.source_frame_dir.as_posix()
        )

        ### Additional parameters for SAM2Long
        self.inference_state["num_pathway"] = 3
        self.inference_state["iou_thre"] = 0.3
        self.inference_state["uncertainty"] = 1

        self.prompts = {}

        # Multi-anchor support: track approved frames
        self.approved_frames: set = set()

        # Multi-layer support: track initialized layers and obj_id mapping
        self.initialized_layers = []  # List of layer names
        self.layer_to_index = {}  # layer_name → layer_index
        self.obj_id_to_layer_label = {}  # SAM2 obj_id → (layer_name, label_id)

    def set_initialized_layers(self, layer_names: list):
        """Initialize tracking for multiple label layers.

        Args:
            layer_names: List of label layer names to track

        SAM2 obj_id mapping: layer_index * 1000 + label_id
        This ensures unique obj_ids across layers.
        """
        self.initialized_layers = list(layer_names)
        self.layer_to_index = {name: idx for idx, name in enumerate(layer_names)}

        # Print initialization summary
        print(f"\n{'='*60}")
        print(f"Initializing SAM2 state with {len(layer_names)} layer(s):")
        for layer_name in layer_names:
            layer = self.viewer.layers[layer_name]
            unique_labels = [int(x) for x in np.unique(layer.data) if x != 0]
            print(f"  - {layer_name}: {len(unique_labels)} label(s) {unique_labels if unique_labels else '(empty)'}")
        print(f"{'='*60}\n")

    def _map_to_sam_obj_id(self, layer_name: str, label_id: int) -> int:
        """Map (layer_name, label_id) to unique SAM2 obj_id."""
        layer_idx = self.layer_to_index[layer_name]
        sam_obj_id = layer_idx * 1000 + label_id
        self.obj_id_to_layer_label[sam_obj_id] = (layer_name, label_id)
        return sam_obj_id

    def _unmap_from_sam_obj_id(self, sam_obj_id: int) -> tuple:
        """Map SAM2 obj_id back to (layer_name, label_id)."""
        return self.obj_id_to_layer_label.get(sam_obj_id, (None, None))

    # ========================================================================
    # Approved Frames Management
    # ========================================================================

    def approve_frame(self, t: int):
        """Mark a frame as an approved anchor for propagation.

        Simply adds the frame to the approved set.
        Masks will be collected from label layers during propagation.
        """
        frame_idx = int(t)
        self.approved_frames.add(frame_idx)

    def unapprove_frame(self, t: int):
        """Remove a frame from the approved set."""
        self.approved_frames.discard(int(t))

    def clear_approved_frames(self):
        """Clear all approved frames."""
        self.approved_frames.clear()

    def get_approved_frames_sorted(self) -> list:
        """Get sorted list of approved frame indices."""
        return sorted(self.approved_frames)

    # ========================================================================
    # Volume Preprocessing
    # ========================================================================

    def preprocess_volume(self):
        """Save each frame as jpeg to a temp dir"""
        layer_name = self.mwo.image_layers_combo.currentText()
        layer = self.viewer.layers[layer_name]
        volume = layer.data

        self.source_frame_dir = Path(
            tempfile.mkdtemp(suffix="_naparisam2long")
        )
        # Save each slice as a separate image
        for i in range(volume.shape[0]):
            slice_path = os.path.join(self.source_frame_dir, f"{i:04d}.jpeg")

            if os.path.exists(slice_path):
                continue

            img_slice = volume[i]
            cv2.imwrite(slice_path, img_slice.squeeze())

        print("Frames generated.")

    # ========================================================================
    # Point Prompt Handling - REMOVED
    # ========================================================================
    # Point prompting has been removed in favor of direct mask drawing.
    # Users should use napari's built-in label tools (paintbrush, etc.)

    # ========================================================================
    # Video Propagation (Multi-Anchor Multi-Label Support)
    # ========================================================================

    def _check_label_consistency(self, anchors, src_layer):
        """Warn if labels appear in only one anchor."""
        all_labels = []
        for t in anchors:
            mask_2d = get_label_slice_2d(src_layer, self.viewer, t)
            all_labels.extend([int(x) for x in np.unique(mask_2d) if x != 0])

        counts = Counter(all_labels)
        lonely = [lbl for lbl, c in counts.items() if c == 1 and len(anchors) > 1]
        if lonely:
            show_warning(f"Labels {lonely} appear in only one anchor - tracking may be unstable.")

    def video_propagate(self, config: VideoPropagateConfig = None):
        """
        Propagate segmentation from approved anchor frames through the video.

        Supports multiple labels per frame, multiple anchor frames, and MULTIPLE LAYERS.
        Uses argmax-based merging to avoid overlap ambiguity.
        """
        if config is None:
            config = VideoPropagateConfig()

        import torch

        if not self.initialized_layers:
            show_info("No initialized layers. Please initialize first.")
            return

        # Get anchors - use approved frames if any, else use current frame
        anchors = self.get_approved_frames_sorted()
        if not anchors:
            # Fall back to current frame if no frames approved
            current_frame = int(self.viewer.dims.current_step[0])
            anchors = [current_frame]
            show_info(f"No approved anchors. Using current frame ({current_frame}) as anchor.")
            print(f"No approved frames. Using current frame {current_frame} as anchor.")

        # Validate anchors
        first_layer = self.viewer.layers[self.initialized_layers[0]]
        nT = first_layer.data.shape[0]
        anchors = [a for a in anchors if 0 <= a < nT]
        if not anchors:
            show_info("No valid anchor frames.")
            return

        if len(anchors) > config.max_anchors:
            show_info(f"Too many anchors ({len(anchors)}). Max is {config.max_anchors}.")
            return

        # Collect all SAM obj_ids from all initialized layers at anchor frames
        all_sam_obj_ids = set()
        layer_obj_map = defaultdict(set)  # layer_name → set of original label_ids

        for layer_name in self.initialized_layers:
            layer = self.viewer.layers[layer_name]
            for t in anchors:
                mask_2d = get_label_slice_2d(layer, self.viewer, t)
                for original_label_id in np.unique(mask_2d):
                    if original_label_id == 0:
                        continue
                    sam_obj_id = self._map_to_sam_obj_id(layer_name, int(original_label_id))
                    all_sam_obj_ids.add(sam_obj_id)
                    layer_obj_map[layer_name].add(int(original_label_id))

        if not all_sam_obj_ids:
            show_info("No labels/masks found in approved anchor frames. Please draw masks first.")
            return

        # Validate that we actually have masks to propagate
        total_mask_pixels = 0
        for layer_name in self.initialized_layers:
            layer = self.viewer.layers[layer_name]
            for t in anchors:
                mask_2d = get_label_slice_2d(layer, self.viewer, t)
                total_mask_pixels += (mask_2d > 0).sum()

        if total_mask_pixels == 0:
            show_info("No masks found in approved frames. Please draw masks before propagating.")
            return

        print(f"\nPropagating {len(self.initialized_layers)} layer(s) with {len(all_sam_obj_ids)} total objects:")
        for layer_name in self.initialized_layers:
            labels = sorted(layer_obj_map[layer_name])
            print(f"  - {layer_name}: labels {labels}")
        print(f"Anchor frames: {anchors}")
        print(f"Will propagate from frame {min(anchors)} to {nT-1} ({nT - min(anchors)} frames)\n")

        # Reset predictor state
        self.predictor.reset_state(self.inference_state)

        # Add all anchor masks to predictor
        for layer_name in self.initialized_layers:
            layer = self.viewer.layers[layer_name]
            for t in anchors:
                mask_2d = get_label_slice_2d(layer, self.viewer, t)
                for original_label_id in np.unique(mask_2d):
                    if original_label_id == 0:
                        continue
                    sam_obj_id = self._map_to_sam_obj_id(layer_name, int(original_label_id))
                    binary = (mask_2d == original_label_id).astype(np.float32)
                    self.predictor.add_new_mask(
                        inference_state=self.inference_state,
                        frame_idx=t,
                        obj_id=int(sam_obj_id),
                        mask=binary,
                    )

        anchor_set = set(anchors)

        # Propagate forward and backward from anchors
        for direction, reverse in [("forward", False), ("backward", True)]:
            print(f"Propagating {direction}...")
            for frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state, reverse=reverse
            ):
                # Progress: forward 0-50%, backward 50-100%
                base = 0 if not reverse else 50
                progress = base + int((frame_idx * 50) / nT)
                self.mwo.video_propagation_progressBar.setValue(progress)

                # Skip anchor frames if protecting
                if config.protect_anchors and frame_idx in anchor_set:
                    continue

                # Split SAM results by layer
                for layer_name in self.initialized_layers:
                    layer = self.viewer.layers[layer_name]
                    layer_data = layer.data

                    # Create empty mask for this frame
                    if layer_data.ndim == 3:
                        H, W = layer_data.shape[1], layer_data.shape[2]
                    elif layer_data.ndim == 4:
                        H, W = layer_data.shape[2], layer_data.shape[3]
                    else:
                        continue

                    frame_mask = np.zeros((H, W), dtype=np.int32)

                    # Extract masks for this layer's objects
                    for i, sam_obj_id in enumerate(out_obj_ids):
                        obj_layer_name, original_label_id = self._unmap_from_sam_obj_id(sam_obj_id)
                        if obj_layer_name != layer_name:
                            continue

                        out_mask = (out_mask_logits[i] > config.prob_threshold).cpu().numpy()
                        frame_mask[out_mask[0]] = original_label_id

                    # Write to layer
                    if layer_data.ndim == 3:
                        layer_data[frame_idx] = frame_mask
                    elif layer_data.ndim == 4:
                        z = int(self.viewer.dims.current_step[1])
                        layer_data[frame_idx, z] = frame_mask

                    layer.data = layer_data

        self.mwo.video_propagation_progressBar.setValue(100)
        print("Propagation complete (forward + backward).")

    # ========================================================================
    # Reset and Cleanup
    # ========================================================================

    def reset(self):
        self.predictor.reset_state(self.inference_state)
        for label_layer_name in self.mwo.get_checked_label_layers():
            if label_layer_name in self.viewer.layers:
                label_layer = self.viewer.layers[label_layer_name]
                label_layer_data = label_layer.data
                zero_mask = np.zeros(label_layer_data.shape, dtype=np.int32)
                label_layer.data = zero_mask

        self.prompts = {}  ### Empty prompts when resetting
        self.approved_frames.clear()  ### Clear approved frames on reset
        self.mwo.video_propagation_progressBar.setValue(0)

    def reinitialize_for_video(self):
        """Re-export frames and reinitialize SAM2 inference state for a new video."""
        # Clean up old temp frames
        self.delete_source_frame_dir()

        # Export new video frames to temp dir
        self.preprocess_volume()

        # Reinitialize SAM2 inference state with new frames
        self.inference_state = self.predictor.init_state(
            video_path=self.source_frame_dir.as_posix()
        )
        self.inference_state["num_pathway"] = 3
        self.inference_state["iou_thre"] = 0.3
        self.inference_state["uncertainty"] = 1

        # Clear all tracking state
        self.prompts = {}
        self.approved_frames.clear()
        self.initialized_layers.clear()
        self.obj_id_map.clear()
        self._next_sam_obj_id = 1

        print("SAM2 inference state reinitialized for new video.")

    def delete_source_frame_dir(self):
        """Deletes the temporary source frame directory"""
        if self.source_frame_dir:
            shutil.rmtree(self.source_frame_dir, ignore_errors=True)
