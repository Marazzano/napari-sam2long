# Imports
import glob
import os
import shutil

import napari
import requests
from napari.utils.notifications import show_info
from qtpy import uic
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from qtpy.QtCore import Qt

from pipelines.sam2long.SAM2Long_pipeline_handler import SAM2Long_pipeline


# Main Plugin class that is connected from outside at napari plugin entry point
class SAM2Long(QWidget):
    def __init__(self, napari_viewer):
        # Initializing
        super().__init__()
        self.viewer = napari_viewer
        self.appInstance = QApplication.instance()  ### For deleting tmp dir

        # Load the UI file - Main window
        script_dir = os.path.dirname(__file__)
        ui_file_name = "SAM2Long.ui"
        abs_file_path = os.path.join(
            script_dir, "..", "UI_files", ui_file_name
        )
        uic.loadUi(abs_file_path, self)

        # Get required children for functionality addition
        self.image_layers_combo = self.findChild(
            QComboBox, "image_layer_combo"
        )

        # Replace output_layers_combo with multi-select list
        old_combo = self.findChild(QComboBox, "output_layer_combo")
        self.output_layers_list = QListWidget()
        self.output_layers_list.setMaximumHeight(100)
        self.output_layers_list.setSelectionMode(QListWidget.NoSelection)  # Use checkboxes instead
        # Replace in layout
        parent_layout = old_combo.parent().layout()
        parent_layout.replaceWidget(old_combo, self.output_layers_list)
        old_combo.setParent(None)

        self.model_cbbox = self.findChild(QComboBox, "model_cbbox")

        self.initialize_btn = self.findChild(QPushButton, "Initialize_btn")
        self.video_propagation_progressBar = self.findChild(
            QProgressBar, "Propagation_progress"
        )
        self.video_propagate_btn = self.findChild(QPushButton, "Propagate_btn")
        self.reset_btn = self.findChild(QPushButton, "reset_btn")

        # Populate combo box - call
        self.populate_combo_box(self.image_layers_combo, "image")
        self.populate_label_layers_list()
        self.populate_model_combo()

        # Connect events to functions
        self.viewer.layers.events.inserted.connect(self.layer_changed)
        self.viewer.layers.events.removed.connect(self.layer_changed)
        self.viewer.layers.events.changed.connect(self.layer_changed)
        self.viewer.mouse_drag_callbacks.append(self.on_mouse_click)
        self.appInstance.lastWindowClosed.connect(
            self.delete_source_dir
        )  ### Delete tempory source frame directory when closing napari

        # Connect button to functions
        self.initialize_btn.clicked.connect(self.initialize_pipeline)
        self.video_propagate_btn.clicked.connect(self.video_propagate)
        self.reset_btn.clicked.connect(self.reset_everything)

        # ====================================================================
        # Multi-Anchor UI: Approve/Clear buttons and approved frames display
        # ====================================================================
        self._setup_anchor_ui()

    def _setup_anchor_ui(self):
        """Set up the UI elements for multi-anchor frame management."""
        # Create container widget for anchor controls
        anchor_container = QWidget()
        anchor_layout = QVBoxLayout(anchor_container)
        anchor_layout.setContentsMargins(0, 5, 0, 5)

        # Button row
        button_row = QHBoxLayout()
        self.approve_btn = QPushButton("Approve Frame")
        self.approve_btn.setToolTip(
            "Bake current mask as anchor (converts points → permanent mask)"
        )
        self.clear_points_btn = QPushButton("Clear Points")
        self.clear_points_btn.setToolTip(
            "Clear points for current label on this frame (keeps mask preview)"
        )
        self.clear_approved_btn = QPushButton("Clear Approved")
        self.clear_approved_btn.setToolTip("Clear all approved anchor frames")
        button_row.addWidget(self.approve_btn)
        button_row.addWidget(self.clear_points_btn)
        button_row.addWidget(self.clear_approved_btn)
        anchor_layout.addLayout(button_row)

        # Approved frames display
        self.approved_list_label = QLabel("Approved: []")
        self.approved_list_label.setWordWrap(True)
        anchor_layout.addWidget(self.approved_list_label)

        # Status HUD
        self.status_hud = QLabel("Label: - | Points: 0+ 0- | Frame: -")
        self.status_hud.setStyleSheet("font-family: monospace; color: #888;")
        anchor_layout.addWidget(self.status_hud)

        # Point mode indicator (positive/negative)
        self.negative_mode = False
        self.mode_label = QLabel("Mode: + (positive)")
        self.mode_label.setStyleSheet(
            "font-weight: bold; color: #2a2; padding: 2px;"
        )
        anchor_layout.addWidget(self.mode_label)

        # Connect buttons
        self.approve_btn.clicked.connect(self.approve_current_frame)
        self.clear_points_btn.clicked.connect(self.clear_current_points)
        self.clear_approved_btn.clicked.connect(self.clear_approved)

        # Connect frame change to update status HUD
        self.viewer.dims.events.current_step.connect(self.update_status_hud)

        # Bind Ctrl+Space to toggle negative mode
        @self.viewer.bind_key("Control-Space", overwrite=True)
        def toggle_negative_mode(viewer):
            self.negative_mode = not self.negative_mode
            self._update_mode_label()

        # Insert anchor controls into the main layout
        # Find the main layout and add our container
        main_layout = self.layout()
        if main_layout is not None:
            # Insert before the last item (usually stretch or propagate button)
            main_layout.insertWidget(main_layout.count() - 1, anchor_container)
        else:
            # Fallback: create a layout if none exists
            fallback_layout = QVBoxLayout(self)
            fallback_layout.addWidget(anchor_container)

    def approve_current_frame(self):
        """Approve the current frame as an anchor for propagation.

        This "bakes" the current mask preview into a permanent anchor:
        - Converts point prompts to a committed mask
        - Clears points (now redundant)
        - SAM2 tracks better from baked masks than raw points
        """
        if hasattr(self, "pipeline_object"):
            t = int(self.viewer.dims.current_step[0])
            baked_objects = self.pipeline_object.approve_frame(t)
            self.update_approved_display()
            self.update_status_hud()
            if baked_objects:
                show_info(f"Frame {t} approved. Baked labels: {baked_objects}")
            else:
                show_info(f"Frame {t} approved as anchor.")
        else:
            show_info("Please initialize pipeline first.")

    def clear_current_points(self):
        """Clear points for current object on current frame."""
        if hasattr(self, "pipeline_object"):
            self.pipeline_object.clear_points_for_current_object()
            self.update_status_hud()
            show_info("Points cleared (mask preview kept).")
        else:
            show_info("Please initialize pipeline first.")

    def clear_approved(self):
        """Clear all approved anchor frames."""
        if hasattr(self, "pipeline_object"):
            self.pipeline_object.clear_approved_frames()
            self.update_approved_display()
            show_info("Approved frames cleared.")
        else:
            show_info("Please initialize pipeline first.")

    def update_approved_display(self):
        """Update the display label showing approved frames."""
        if hasattr(self, "pipeline_object"):
            frames = self.pipeline_object.get_approved_frames_sorted()
            self.approved_list_label.setText(f"Approved: {frames}")
        else:
            self.approved_list_label.setText("Approved: []")

    def update_status_hud(self, event=None):
        """Update the status HUD showing current editing state."""
        if not hasattr(self, "pipeline_object"):
            self.status_hud.setText("Label: - | Points: 0+ 0- | Frame: -")
            return

        frame = int(self.viewer.dims.current_step[0])

        # Get currently active layer
        active_layer = self.viewer.layers.selection.active
        if not active_layer or not isinstance(active_layer, napari.layers.Labels):
            self.status_hud.setText(f"Label: - | Points: 0+ 0- | Frame: {frame}")
            return

        layer = active_layer
        label = layer.selected_label

        # Count points from inference_state (source of truth)
        point_inputs = self.pipeline_object.inference_state.get("point_inputs_per_obj", {})
        pos_count = neg_count = 0
        if label in point_inputs and frame in point_inputs.get(label, {}):
            pts = point_inputs[label].get(frame)
            if pts is not None and len(pts) > 1:
                labels_arr = pts[1]  # (points, labels) tuple
                if hasattr(labels_arr, '__len__'):
                    pos_count = sum(1 for l in labels_arr if l == 1)
                    neg_count = sum(1 for l in labels_arr if l == 0)

        approved = frame in self.pipeline_object.approved_frames
        approved_str = "Yes" if approved else "No"
        self.status_hud.setText(
            f"Layer: {layer.name} | Label: {label} | Points: {pos_count}+ {neg_count}- | "
            f"Frame: {frame} | Approved: {approved_str}"
        )

    # Function to populate combo boxes based on layers
    def populate_combo_box(self, combobx, layer_type="image"):
        ### Save last selected layer, so that input drop-down menu doesn't change whenever new layer is added to viewer
        current_text = combobx.currentText() if combobx else None

        # Clear the combo box first
        combobx.clear() if combobx else None

        if layer_type == "image":
            # Get all existing image layers from the napari viewer
            layers = [
                layer.name
                for layer in self.viewer.layers
                if isinstance(layer, napari.layers.Image)
                and len(layer.data.shape)
                in [3, 4]  # accept 3D grayscale or 4D color video
            ]
        else:
            raise ValueError(
                "Invalid layer_type. Expected 'image'."
            )

        combobx.addItems(layers)
        ### Keep last selected item
        if current_text:
            combobx.setCurrentText(current_text)

    def populate_label_layers_list(self):
        """Populate the multi-select list of label layers with checkboxes."""
        # Save currently checked layers
        checked_layers = set()
        for i in range(self.output_layers_list.count()):
            item = self.output_layers_list.item(i)
            if item.checkState() == Qt.Checked:
                checked_layers.add(item.text())

        # Clear and repopulate
        self.output_layers_list.clear()

        # Get all label layers
        label_layers = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]

        # Add as checkable items
        for layer_name in label_layers:
            item = QListWidgetItem(layer_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            # Restore checked state if it was checked before
            if layer_name in checked_layers:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.output_layers_list.addItem(item)

    def get_checked_label_layers(self):
        """Get list of checked label layer names."""
        checked = []
        for i in range(self.output_layers_list.count()):
            item = self.output_layers_list.item(i)
            if item.checkState() == Qt.Checked:
                checked.append(item.text())
        return checked

    # Function to handle layer changes
    def layer_changed(self):
        # Populate combo box and label list
        self.populate_combo_box(self.image_layers_combo, "image")
        self.populate_label_layers_list()

    def populate_model_combo(self):
        self.model_cbbox.clear()
        self.model_cbbox.addItems(
            [
                "sam2.1_hiera_base_plus",
                "sam2.1_hiera_tiny",
                "sam2.1_hiera_small",
                "sam2.1_hiera_large",
            ]
        )

    # Initialize pipeline
    BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    CHECKPOINTS = {
        "sam2.1_hiera_large": "sam2.1_hiera_large.pt",
        "sam2.1_hiera_small": "sam2.1_hiera_small.pt",
        "sam2.1_hiera_tiny": "sam2.1_hiera_tiny.pt",
        "sam2.1_hiera_base_plus": "sam2.1_hiera_base_plus.pt",
    }

    def initialize_pipeline(self):
        # Clean up any temporary directories
        self.cleanup_all_temp_dirs()

        # "Reset" napari's mouse_drag_callbacks; does not remove automatically from previous session when closing widget window
        if len(self.viewer.mouse_drag_callbacks) > 1:
            self.viewer.mouse_drag_callbacks.pop(0)

        if self.image_layers_combo.count() == 0:
            show_info("No input image.")

        # If pipeline has been initialized before, reset first
        if hasattr(self, "pipeline_object"):
            self.reset_everything()
            self.delete_source_dir()

        script_dir = os.path.dirname(__file__)
        model_map = {
            "sam2.1_hiera_large": (
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                "sam2.1_hiera_large.pt",
            ),
            "sam2.1_hiera_small": (
                "configs/sam2.1/sam2.1_hiera_s.yaml",
                "sam2.1_hiera_small.pt",
            ),
            "sam2.1_hiera_tiny": (
                "configs/sam2.1/sam2.1_hiera_t.yaml",
                "sam2.1_hiera_tiny.pt",
            ),
            "sam2.1_hiera_base_plus": (
                "configs/sam2.1/sam2.1_hiera_b+.yaml",
                "sam2.1_hiera_base_plus.pt",
            ),
        }

        selected_model = self.model_cbbox.currentText()

        if selected_model in model_map:
            model_cfg, checkpoint_name = model_map[selected_model]
            model_cfg_path = os.path.join("configs", "sam2.1", model_cfg)
            checkpoint_path = os.path.join(
                script_dir, "..", "model", checkpoint_name
            )

            # Check if the checkpoint file exists
            if not os.path.exists(checkpoint_path):
                print(
                    f"Checkpoint {checkpoint_name} not found. Downloading..."
                )
                self.download_checkpoint(checkpoint_name, checkpoint_path)
            print("Model_cfg ", model_cfg_path)
            self.pipeline_object = SAM2Long_pipeline(
                self.viewer,
                self,
                checkpoint_path,
                model_cfg,
            )

            # Initialize with checked label layers
            checked_layers = self.get_checked_label_layers()
            if not checked_layers:
                show_info("Please check at least one label layer before initializing.")
                return

            self.pipeline_object.set_initialized_layers(checked_layers)

            # Update approved frames display after initialization
            self.update_approved_display()
        else:
            print("Model not recognized.")

    def download_checkpoint(self, checkpoint_name, checkpoint_path):
        url = self.BASE_URL + checkpoint_name

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check if the download was successful

            with open(checkpoint_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"{checkpoint_name} downloaded successfully.")

        except requests.exceptions.RequestException as e:
            print(
                f"Failed to download {checkpoint_name} from {url}. Error: {e}"
            )

    def _update_mode_label(self):
        """Update the mode indicator label."""
        if self.negative_mode:
            self.mode_label.setText("Mode: - (negative)")
            self.mode_label.setStyleSheet(
                "font-weight: bold; color: #c22; padding: 2px;"
            )
        else:
            self.mode_label.setText("Mode: + (positive)")
            self.mode_label.setStyleSheet(
                "font-weight: bold; color: #2a2; padding: 2px;"
            )

    def on_mouse_click(self, layer, event):
        """Handle mouse clicks for point prompts.

        Keybindings:
        - Ctrl+left-click: add point (positive or negative based on mode)
        - Ctrl+Space: toggle between positive/negative mode

        Points accumulate for iterative mask refinement.
        Works on whichever label layer is currently active in napari.
        """
        # Check for left mouse button (button 1) with Ctrl modifier
        if event.button != 1:
            return  # Only handle left clicks

        if "Control" not in event.modifiers:
            return  # Must hold Ctrl to add points

        # Check that pipeline has been initialized
        if not hasattr(self, "pipeline_object"):
            show_info("Please initialize first.")
            return

        # Get currently active layer
        active_layer = self.viewer.layers.selection.active
        if not active_layer or not isinstance(active_layer, napari.layers.Labels):
            show_info("Please select a label layer first.")
            return

        # Check that this layer was initialized
        if not hasattr(self.pipeline_object, "initialized_layers"):
            show_info("Please initialize with label layers first.")
            return

        if active_layer.name not in self.pipeline_object.initialized_layers:
            show_info(f"Layer '{active_layer.name}' was not initialized. Please re-initialize.")
            return

        point = [
            int(event.position[0]),
            int(event.position[1]),
            int(event.position[2]),
        ]
        active_label = active_layer.selected_label

        # Use mode toggle to determine positive (1) or negative (0)
        neg_or_pos = 0 if self.negative_mode else 1
        self.pipeline_object.add_point(point, active_label, neg_or_pos=neg_or_pos, layer_name=active_layer.name)

        # Update status HUD after adding point
        self.update_status_hud()

    def video_propagate(self):
        if self.image_layers_combo.count() == 0:
            show_info("No input image.")
            return
        else:
            self.pipeline_object.video_propagate()

    def reset_everything(self):
        if hasattr(self, "pipeline_object"):
            self.pipeline_object.reset()
            self.update_approved_display()

    def delete_source_dir(self):
        """Deletes the temporary source frame directory when Napari closes."""
        if hasattr(self, "pipeline_object"):
            self.pipeline_object.delete_source_frame_dir()

    def cleanup_all_temp_dirs(self):
        """Delete potential temporary directories from previous sessions that have crashed"""
        for dir_path in glob.glob("/tmp/tmp*naparisam2long"):
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
