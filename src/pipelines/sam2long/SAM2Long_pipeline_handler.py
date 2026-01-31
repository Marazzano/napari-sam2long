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
        """Bake the point-refined mask into a permanent anchor.

        Steps:
        1. Get current mask from all initialized label layers
        2. Convert to binary mask per object
        3. Call add_new_mask() to commit as permanent anchor
        4. Clear points (now redundant - the mask is the anchor)

        Works across all initialized layers.
        """
        frame_idx = int(t)

        # Get all objects that have been prompted on this frame
        point_inputs = self.inference_state.get("point_inputs_per_obj", {})

        baked_objects = []

        # Process all initialized layers
        for layer_name in self.initialized_layers:
            if layer_name not in self.viewer.layers:
                continue

            layer = self.viewer.layers[layer_name]
            mask_2d = layer.data[frame_idx]

            # Find SAM obj_ids that belong to this layer
            for sam_obj_id in list(point_inputs.keys()):
                if frame_idx not in point_inputs.get(sam_obj_id, {}):
                    continue
                if point_inputs[sam_obj_id].get(frame_idx) is None:
                    continue

                # Check if this obj_id belongs to current layer
                obj_layer_name, original_label_id = self._unmap_from_sam_obj_id(sam_obj_id)
                if obj_layer_name != layer_name:
                    continue

                # Get the current mask from the label layer (the preview)
                binary_mask = (mask_2d == original_label_id).astype(np.float32)

                if binary_mask.sum() == 0:
                    continue  # No mask to bake

                # COMMIT: Bake mask into predictor as permanent anchor
                self.predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=int(sam_obj_id),
                    mask=binary_mask,
                )

                # Clear points for this object (now redundant)
                point_inputs[sam_obj_id].pop(frame_idx, None)

                # Clear from self.prompts too
                if sam_obj_id in self.prompts:
                    self.prompts[sam_obj_id] = [
                        p for p in self.prompts[sam_obj_id] if p[0] != frame_idx
                    ]

                baked_objects.append((layer_name, original_label_id))

        # Mark frame as approved
        self.approved_frames.add(frame_idx)

        return baked_objects  # Return list of (layer, label) tuples for UI feedback

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
    # Point Prompt Handling
    # ========================================================================

    def add_point(self, point_array, label_id, neg_or_pos=1, layer_name=None):
        """Add a point prompt for iterative mask refinement.

        Points accumulate within the same (object, frame) for iterative refinement.
        Coordinates are stored as (y, x) internally, converted to SAM's (x, y) at call site.

        Args:
            point_array: [frame_idx, y, x] coordinates
            label_id: Object/label ID to segment
            neg_or_pos: 1 for positive (include), 0 for negative (exclude)
            layer_name: Name of the label layer (required for multi-layer support)
        """
        ann_frame_idx = point_array[0]

        # Map (layer_name, label_id) to unique SAM2 obj_id
        if layer_name and layer_name in self.layer_to_index:
            ann_obj_id = self._map_to_sam_obj_id(layer_name, label_id)
        else:
            # Fallback for backward compatibility (single layer)
            ann_obj_id = label_id
            if not hasattr(self, 'initialized_layers') or not self.initialized_layers:
                # Old single-layer mode
                layer_name = self.mwo.output_layers_list.item(0).text() if self.mwo.output_layers_list.count() > 0 else None
        # Store as (y, x) internally - convert to SAM's (x, y) only at call site
        new_point_yx = (point_array[1], point_array[2])  # y, x
        new_label = neg_or_pos

        # Check if predictor already has points for this (obj, frame)
        # This is the ROBUST way - handles frame 10 → 15 → back to 10
        point_inputs = self.inference_state.get("point_inputs_per_obj", {})
        has_existing = (
            ann_obj_id in point_inputs
            and ann_frame_idx in point_inputs.get(ann_obj_id, {})
            and point_inputs[ann_obj_id].get(ann_frame_idx) is not None
        )
        clear_old = not has_existing

        # Store in self.prompts for UI bookkeeping only (not for propagation)
        new_point_xy = [new_point_yx[1], new_point_yx[0]]  # x, y for storage
        if ann_obj_id in self.prompts:
            found_frame = False
            for existing_list in self.prompts[ann_obj_id]:
                if existing_list[0] == ann_frame_idx:
                    # Append to existing frame's points
                    existing_list[1] = np.append(existing_list[1], [new_point_xy], axis=0)
                    existing_list[2] = np.append(existing_list[2], [new_label])
                    found_frame = True
                    break
            if not found_frame:
                # New frame for this object
                points = np.array([new_point_xy], dtype=np.float32)
                labels = np.array([new_label], np.int32)
                self.prompts[ann_obj_id].append([ann_frame_idx, points, labels])
        else:
            # New object
            points = np.array([new_point_xy], dtype=np.float32)
            labels = np.array([new_label], np.int32)
            self.prompts[ann_obj_id] = [[ann_frame_idx, points, labels]]

        # DO NOT call reset_state() - this destroys accumulated context!

        # Convert (y, x) to SAM's expected (x, y) at call site
        points_xy = np.array([[new_point_yx[1], new_point_yx[0]]], dtype=np.float32)

        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points_xy,  # (x, y) for SAM
            labels=np.array([new_label], np.int32),
            clear_old_points=clear_old,  # False if points already exist
        )

        # Update mask preview in label layer
        if not layer_name:
            # Try to get from active layer
            active = self.viewer.layers.selection.active
            if active and hasattr(active, 'name'):
                layer_name = active.name

        if not layer_name or layer_name not in self.viewer.layers:
            show_info("No valid label layer selected.")
            return

        layer = self.viewer.layers[layer_name]
        label_layer_data = layer.data

        # if image and label layer dimensions do not match, show info
        if (
            label_layer_data[0].shape
            != out_mask_logits[0][0].cpu().numpy().shape
        ):
            print("label", label_layer_data.shape)
            print("outmask", out_mask_logits[0][0].cpu().numpy().shape)
            show_info("Create a new labels layer.")
            return

        mask_for_this_frame = np.zeros(
            (label_layer_data.shape[1], label_layer_data.shape[2]),
            dtype=np.int32,
        )

        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            # Map SAM obj_id back to original label_id for this layer
            _, original_label_id = self._unmap_from_sam_obj_id(out_obj_id)
            if original_label_id is not None:
                mask_for_this_frame[out_mask[0]] = original_label_id
            else:
                # Fallback for backward compatibility
                mask_for_this_frame[out_mask[0]] = out_obj_id

        label_layer_data[ann_frame_idx, :, :] = mask_for_this_frame
        layer.data = label_layer_data

    def clear_points_for_current_object(self):
        """Clear accumulated points for current object on current frame.

        Note: This does NOT erase the mask preview from the label layer.
        The user can manually erase if needed, or the preview remains
        as a starting point for new prompts.
        """
        # Get currently active layer
        active_layer = self.viewer.layers.selection.active
        if not active_layer or not hasattr(active_layer, 'selected_label'):
            return

        layer_name = active_layer.name
        original_label_id = active_layer.selected_label
        frame_idx = int(self.viewer.dims.current_step[0])

        # Map to SAM obj_id
        if layer_name in self.layer_to_index:
            sam_obj_id = self._map_to_sam_obj_id(layer_name, original_label_id)
        else:
            sam_obj_id = original_label_id  # Fallback

        # Clear from self.prompts (UI bookkeeping only)
        if sam_obj_id in self.prompts:
            self.prompts[sam_obj_id] = [
                p for p in self.prompts[sam_obj_id] if p[0] != frame_idx
            ]

        # Clear from predictor state (the source of truth)
        point_inputs = self.inference_state.get("point_inputs_per_obj", {})
        if sam_obj_id in point_inputs:
            point_inputs[sam_obj_id].pop(frame_idx, None)

        # DO NOT clear the label layer - user may want to keep the preview!

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
            show_info("No labels found in anchor frames.")
            return

        print(f"\nPropagating {len(self.initialized_layers)} layer(s) with {len(all_sam_obj_ids)} total objects:")
        for layer_name in self.initialized_layers:
            labels = sorted(layer_obj_map[layer_name])
            print(f"  - {layer_name}: labels {labels}")
        print(f"Anchor frames: {anchors}\n")

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

        # Propagate through video
        for frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            self.inference_state, reverse=False
        ):
            progress = int((frame_idx * 100) / nT)
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
                        continue  # Skip objects from other layers

                    # Get mask and apply threshold
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
        print("Propagation complete.")

    # ========================================================================
    # Reset and Cleanup
    # ========================================================================

    def reset(self):
        self.predictor.reset_state(self.inference_state)
        label_layer_name = self.mwo.output_layers_combo.currentText()
        if (
            label_layer_name is not None and label_layer_name != ""
        ):  ### Reset label layer if label is not empty
            label_layer = self.viewer.layers[label_layer_name]
            label_layer_data = label_layer.data
            zero_mask = np.zeros(label_layer_data.shape, dtype=np.int32)
            label_layer.data = zero_mask

        self.prompts = {}  ### Empty prompts when resetting
        self.approved_frames.clear()  ### Clear approved frames on reset
        self.mwo.video_propagation_progressBar.setValue(0)

    def delete_source_frame_dir(self):
        """Deletes the temporary source frame directory"""
        if self.source_frame_dir:
            shutil.rmtree(self.source_frame_dir, ignore_errors=True)
