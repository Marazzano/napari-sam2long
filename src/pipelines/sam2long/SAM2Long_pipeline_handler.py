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

    # ========================================================================
    # Approved Frames Management
    # ========================================================================

    def approve_frame(self, t: int):
        """Mark a frame as approved/anchor for propagation."""
        self.approved_frames.add(int(t))

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

    def add_point(self, point_array, label_id, neg_or_pos=1):
        ann_frame_idx = point_array[0]
        ann_obj_id = label_id
        new_point = [point_array[2], point_array[1]]
        new_label = neg_or_pos
        check_if_our_z_is_new = True
        check_if_our_annotation_is_new = True

        # Check if in dict else add it

        # Object has been annotated before
        if ann_obj_id in self.prompts:
            all_list = []
            for existing_list in self.prompts[ann_obj_id]:

                # this frame has been annotated/prompted before
                if existing_list[0] == ann_frame_idx:
                    points = existing_list[1]
                    labels = list(existing_list[2])
                    points = np.append(points, [new_point], axis=0)
                    labels.append(new_label)
                    new_list = [
                        ann_frame_idx,
                        points,
                        np.array(labels, np.int32),
                    ]
                    all_list.append(new_list)
                    check_if_our_z_is_new = False
                # frame has not been annotated before
                else:
                    all_list.append(existing_list)

            self.prompts[ann_obj_id] = all_list
            check_if_our_annotation_is_new = False

        # Object has NOT been annotated before
        else:
            points = np.array(
                [[point_array[2], point_array[1]]], dtype=np.float32
            )
            labels = np.array([neg_or_pos], np.int32)
            self.prompts[ann_obj_id] = [[ann_frame_idx, points, labels]]

        # Object has been annotated but not in this frame
        if check_if_our_z_is_new and not (check_if_our_annotation_is_new):
            points = np.array(
                [[point_array[2], point_array[1]]], dtype=np.float32
            )
            labels = np.array([neg_or_pos], np.int32)
            existing_val = self.prompts[ann_obj_id]
            existing_val.append([ann_frame_idx, points, labels])
            self.prompts[ann_obj_id] = existing_val

        layer_name = self.mwo.output_layers_combo.currentText()
        layer = self.viewer.layers[layer_name]
        label_layer_data = layer.data

        self.predictor.reset_state(self.inference_state)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

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
            mask_for_this_frame[out_mask[0]] = out_obj_id

        label_layer_data[ann_frame_idx, :, :] = mask_for_this_frame
        layer.data = label_layer_data

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

        Supports multiple labels per frame and multiple anchor frames.
        Uses argmax-based merging to avoid overlap ambiguity.
        """
        if config is None:
            config = VideoPropagateConfig()

        import torch

        # Get source layer
        layer_name = self.mwo.output_layers_combo.currentText()
        src_layer = self.viewer.layers[layer_name]

        # Resolve output layer (creates new if source is dask-backed)
        output_layer = resolve_output_layer(
            self.viewer, src_layer, config.output_layer_suffix
        )
        output_data = output_layer.data

        # Get anchors - use approved frames if any, else use current frame
        anchors = self.get_approved_frames_sorted()
        if not anchors:
            # Fall back to current frame if no frames approved
            current_frame = int(self.viewer.dims.current_step[0])
            anchors = [current_frame]
            print(f"No approved frames. Using current frame {current_frame} as anchor.")

        # Validate anchors
        nT = output_data.shape[0]
        anchors = [a for a in anchors if 0 <= a < nT]
        if not anchors:
            show_info("No valid anchor frames.")
            return

        if len(anchors) > config.max_anchors:
            show_info(f"Too many anchors ({len(anchors)}). Max is {config.max_anchors}.")
            return

        # Check label consistency across anchors
        self._check_label_consistency(anchors, src_layer)

        # Collect all unique object IDs from anchor frames
        all_obj_ids = set()
        for t in anchors:
            mask_2d = get_label_slice_2d(src_layer, self.viewer, t)
            for obj_id in np.unique(mask_2d):
                if obj_id != 0:
                    all_obj_ids.add(int(obj_id))

        if not all_obj_ids:
            show_info("No labels found in anchor frames.")
            return

        print(f"Found {len(all_obj_ids)} objects across {len(anchors)} anchor frames: {sorted(all_obj_ids)}")

        # Reset predictor state
        self.predictor.reset_state(self.inference_state)

        # Add all anchor masks to predictor
        for t in anchors:
            mask_2d = get_label_slice_2d(src_layer, self.viewer, t)
            for obj_id in np.unique(mask_2d):
                if obj_id == 0:
                    continue
                binary = (mask_2d == obj_id).astype(np.float32)
                self.predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=t,
                    obj_id=int(obj_id),
                    mask=binary,
                )

        print(f"Propagating from {len(anchors)} anchor frames...")
        anchor_set = set(anchors)

        # Propagate through video
        for frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            self.inference_state, reverse=False
        ):
            progress = int((frame_idx * 100) / nT)
            self.mwo.video_propagation_progressBar.setValue(progress)

            # Skip anchor frames if protecting
            if config.protect_anchors and frame_idx in anchor_set:
                # Copy anchor mask to output if different layer
                if output_layer is not src_layer:
                    anchor_mask = get_label_slice_2d(src_layer, self.viewer, frame_idx)
                    if output_data.ndim == 3:
                        output_data[frame_idx] = anchor_mask
                    elif output_data.ndim == 4:
                        z = int(self.viewer.dims.current_step[1])
                        output_data[frame_idx, z] = anchor_mask
                continue

            # Merge logits to label image using argmax
            if len(out_obj_ids) > 0:
                pred_2d = _logits_to_label_image(
                    out_obj_ids, out_mask_logits, config.prob_threshold
                )
            else:
                # No objects predicted for this frame
                pred_2d = np.zeros(
                    (output_data.shape[-2], output_data.shape[-1]),
                    dtype=np.int32
                )

            # Write to output
            if output_data.ndim == 3:
                output_data[frame_idx] = pred_2d
            elif output_data.ndim == 4:
                z = int(self.viewer.dims.current_step[1])
                output_data[frame_idx, z] = pred_2d

        output_layer.data = output_data
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
