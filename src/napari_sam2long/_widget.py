# Imports
import glob
import os
import shutil

import napari
import numpy as np
import requests
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker
from qtpy import uic
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from qtpy.QtCore import Qt

from pipelines.sam2long.SAM2Long_pipeline_handler import SAM2Long_pipeline
from .export_curation import ExportConfigDialog
from .ingestion import Project
from .ingestion.mask_io import read_indexed_mask, write_indexed_mask


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

        # Project state tracking
        self.current_project = None
        self.current_video_id = None
        self.current_video_index = 0

        # Setup project manager UI (must be before getting children)
        self._setup_project_manager_ui()

        # Get required children for functionality addition
        self.image_layers_combo = self.findChild(
            QComboBox, "image_layer_combo"
        )

        # Replace output_layers_combo with multi-select list
        old_combo = self.findChild(QComboBox, "output_layer_combo")
        self.output_layers_list = QListWidget()
        self.output_layers_list.setMaximumHeight(60)  # Compact height
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
        self.image_layers_combo.currentTextChanged.connect(self.update_export_frame_list)
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
        self._setup_export_curation_ui()

    def _setup_project_manager_ui(self):
        """Set up the Project Manager UI section at the top."""
        # Create project manager group box
        project_group = QGroupBox("Project Manager")
        project_layout = QVBoxLayout()
        project_group.setLayout(project_layout)

        # Row 1: Load project button
        load_row = QHBoxLayout()
        self.load_project_btn = QPushButton("Load Project...")
        self.load_project_btn.setToolTip("Load an organized project folder")
        self.load_project_btn.clicked.connect(self.load_project_folder)
        load_row.addWidget(self.load_project_btn)
        project_layout.addLayout(load_row)

        # Row 2: Project info display
        self.project_info_label = QLabel("No project loaded")
        self.project_info_label.setStyleSheet("font-weight: bold;")
        project_layout.addWidget(self.project_info_label)

        # Row 3: Video info display
        self.video_info_label = QLabel("")
        project_layout.addWidget(self.video_info_label)

        # Row 4: Navigation buttons
        nav_row = QHBoxLayout()
        self.prev_video_btn = QPushButton("← Previous")
        self.prev_video_btn.setEnabled(False)
        self.prev_video_btn.clicked.connect(self.prev_video)

        self.next_video_btn = QPushButton("Next →")
        self.next_video_btn.setEnabled(False)
        self.next_video_btn.clicked.connect(self.next_video)

        self.complete_next_btn = QPushButton("✓ Complete & Next")
        self.complete_next_btn.setEnabled(False)
        self.complete_next_btn.clicked.connect(self.mark_complete_and_next)
        self.complete_next_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")

        nav_row.addWidget(self.prev_video_btn)
        nav_row.addWidget(self.next_video_btn)
        nav_row.addWidget(self.complete_next_btn)
        project_layout.addLayout(nav_row)

        # Insert at the top of the main layout
        main_layout = self.layout()
        if main_layout is not None:
            main_layout.insertWidget(0, project_group)

    def _setup_anchor_ui(self):
        """Set up the UI elements for multi-anchor frame management."""
        # Create grouped container for anchor controls
        anchor_group = QGroupBox("Anchor Frame Management")
        anchor_layout = QVBoxLayout()
        anchor_group.setLayout(anchor_layout)

        # Button row 1: Add/Remove Anchor
        button_row1 = QHBoxLayout()
        self.approve_btn = QPushButton("Add Anchor")
        self.approve_btn.setToolTip("Mark current frame as anchor for propagation")

        self.unapprove_btn = QPushButton("Remove Anchor")
        self.unapprove_btn.setToolTip("Remove current frame from anchors")

        button_row1.addWidget(self.approve_btn)
        button_row1.addWidget(self.unapprove_btn)
        anchor_layout.addLayout(button_row1)

        # Button row 1b: Approve/Unapprove frame for export
        approve_row = QHBoxLayout()
        self.approve_frame_btn = QPushButton("Approve Frame")
        self.approve_frame_btn.setToolTip("Approve current frame for export")
        self.approve_frame_btn.setStyleSheet("background-color: #4CAF50; color: white;")

        self.unapprove_frame_btn = QPushButton("Unapprove Frame")
        self.unapprove_frame_btn.setToolTip("Remove current frame from export approval")

        approve_row.addWidget(self.approve_frame_btn)
        approve_row.addWidget(self.unapprove_frame_btn)
        anchor_layout.addLayout(approve_row)

        # Button row 2: Clear anchors and clear current frame
        button_row2 = QHBoxLayout()
        self.clear_approved_btn = QPushButton("Clear All Anchors")
        self.clear_approved_btn.setToolTip("Clear all approved anchor frames")

        self.clear_frame_btn = QPushButton("Clear Current Frame")
        self.clear_frame_btn.setToolTip("Clear masks on current frame only (for redrawing)")
        self.clear_frame_btn.setStyleSheet("background-color: #f44336; color: white;")

        button_row2.addWidget(self.clear_approved_btn)
        button_row2.addWidget(self.clear_frame_btn)
        anchor_layout.addLayout(button_row2)

        # Approved frames display
        self.approved_list_label = QLabel("Anchors: []")
        self.approved_list_label.setWordWrap(True)
        self.approved_list_label.setStyleSheet("font-size: 9pt;")
        anchor_layout.addWidget(self.approved_list_label)

        # Status HUD
        self.status_hud = QLabel("Layer: - | Label: - | Frame: -")
        self.status_hud.setStyleSheet("font-family: monospace; color: #888; font-size: 9pt;")
        anchor_layout.addWidget(self.status_hud)

        # Connect buttons
        self.approve_btn.clicked.connect(self.approve_current_frame)
        self.unapprove_btn.clicked.connect(self.unapprove_current_frame)
        self.clear_approved_btn.clicked.connect(self.clear_approved)
        self.clear_frame_btn.clicked.connect(self.clear_current_frame)
        self.approve_frame_btn.clicked.connect(self.approve_frame_for_export)
        self.unapprove_frame_btn.clicked.connect(self.unapprove_frame_for_export)

        # Connect frame change to update status HUD
        self.viewer.dims.events.current_step.connect(self.update_status_hud)

        # Insert anchor controls into the main layout
        main_layout = self.layout()
        if main_layout is not None:
            main_layout.insertWidget(main_layout.count() - 1, anchor_group)
        else:
            fallback_layout = QVBoxLayout(self)
            fallback_layout.addWidget(anchor_group)

    def approve_current_frame(self):
        """Approve the current frame as an anchor for propagation."""
        if hasattr(self, "pipeline_object"):
            t = int(self.viewer.dims.current_step[0])

            # Check if we've hit the max anchor limit
            if len(self.pipeline_object.approved_frames) >= 10:
                show_info("Maximum of 10 anchor frames reached. Unapprove some frames first.")
                return

            self.pipeline_object.approve_frame(t)
            self.update_approved_display()
            self.update_status_hud()
            show_info(f"Frame {t} approved as anchor.")
        else:
            show_info("Please initialize pipeline first.")

    def unapprove_current_frame(self):
        """Remove current frame from approved anchors."""
        if hasattr(self, "pipeline_object"):
            t = int(self.viewer.dims.current_step[0])
            if t in self.pipeline_object.approved_frames:
                self.pipeline_object.unapprove_frame(t)
                self.update_approved_display()
                self.update_status_hud()
                show_info(f"Frame {t} unapproved.")
            else:
                show_info(f"Frame {t} is not approved.")
        else:
            show_info("Please initialize pipeline first.")

    def approve_frame_for_export(self):
        """Approve current frame for export."""
        t = int(self.viewer.dims.current_step[0])
        if hasattr(self, 'export_config_dialog'):
            item = self.export_config_dialog.export_panel.frame_items.get(t)
            if item is not None:
                item.setCheckState(Qt.Checked)
                show_info(f"Frame {t} approved for export.")
            else:
                show_info(f"Frame {t} not in export list. Load a video first.")
        else:
            show_info("Export not configured yet.")

    def unapprove_frame_for_export(self):
        """Remove current frame from export approval."""
        t = int(self.viewer.dims.current_step[0])
        if hasattr(self, 'export_config_dialog'):
            item = self.export_config_dialog.export_panel.frame_items.get(t)
            if item is not None:
                item.setCheckState(Qt.Unchecked)
                show_info(f"Frame {t} removed from export.")
            else:
                show_info(f"Frame {t} not in export list.")
        else:
            show_info("Export not configured yet.")

    def clear_approved(self):
        """Clear all approved anchor frames."""
        if hasattr(self, "pipeline_object"):
            self.pipeline_object.clear_approved_frames()
            self.update_approved_display()
            show_info("Approved frames cleared.")
        else:
            show_info("Please initialize pipeline first.")

    def clear_current_frame(self):
        """Clear all masks on the current frame across all checked label layers."""
        # Get current frame index
        current_frame = int(self.viewer.dims.current_step[0])

        # Get all checked label layers
        checked_layers = self.get_checked_label_layers()

        if not checked_layers:
            show_info("No label layers selected. Check layers in the Labels list.")
            return

        # Clear masks on current frame for each checked layer
        cleared_count = 0
        for layer_name in checked_layers:
            try:
                layer = self.viewer.layers[layer_name]
                if isinstance(layer, napari.layers.Labels):
                    # Clear the current frame
                    layer.data[current_frame] = 0
                    layer.refresh()
                    cleared_count += 1
            except KeyError:
                continue

        if cleared_count > 0:
            show_info(f"Cleared frame {current_frame} in {cleared_count} layer(s)")
        else:
            show_info("No layers to clear")

    def update_approved_display(self):
        """Update the display label showing approved frames."""
        if hasattr(self, "pipeline_object"):
            frames = self.pipeline_object.get_approved_frames_sorted()
            self.approved_list_label.setText(f"Anchors: {frames}")
        else:
            self.approved_list_label.setText("Anchors: []")

    def update_status_hud(self, event=None):
        """Update the status HUD showing current editing state."""
        if not hasattr(self, "pipeline_object"):
            self.status_hud.setText("Layer: - | Label: - | Frame: -")
            return

        frame = int(self.viewer.dims.current_step[0])

        # Get currently active layer
        active_layer = self.viewer.layers.selection.active
        if not active_layer or not isinstance(active_layer, napari.layers.Labels):
            self.status_hud.setText(f"Layer: - | Label: - | Frame: {frame}")
            return

        layer = active_layer
        label = layer.selected_label

        is_anchor = frame in self.pipeline_object.approved_frames
        anchor_str = "⚓ ANCHOR" if is_anchor else ""

        is_exported = hasattr(self, 'export_config_dialog') and frame in self.export_config_dialog.export_frames
        export_str = "✓ APPROVED" if is_exported else ""

        status_parts = [f"Layer: {layer.name}", f"Label: {label}", f"Frame: {frame}"]
        if anchor_str:
            status_parts.append(anchor_str)
        if export_str:
            status_parts.append(export_str)
        self.status_hud.setText(" | ".join(status_parts))

    def _setup_export_curation_ui(self):
        """
        Add the export configuration button to the UI.

        Opens a dialog for configuring export settings including
        frame selection and other COCO export options.
        """
        # Create the dialog (but don't show it yet)
        self.export_config_dialog = ExportConfigDialog(0, parent=self)

        # Create export section group
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        export_group.setLayout(export_layout)

        # Row 1: Export actions
        export_actions = QHBoxLayout()

        self.export_coco_btn = QPushButton("Export Approved")
        self.export_coco_btn.setEnabled(False)
        self.export_coco_btn.setToolTip("Export approved frames to COCO format")
        self.export_coco_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 8px;"
        )
        self.export_coco_btn.clicked.connect(self.export_to_coco)
        export_actions.addWidget(self.export_coco_btn)

        self.export_complete_btn = QPushButton("Mark Complete & Export")
        self.export_complete_btn.setEnabled(False)
        self.export_complete_btn.setToolTip(
            "Mark current video complete, export its masks, then go to next video"
        )
        self.export_complete_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;"
        )
        self.export_complete_btn.clicked.connect(self.mark_complete_and_export)
        export_actions.addWidget(self.export_complete_btn)

        export_layout.addLayout(export_actions)

        # Row 2: Configure export button (settings)
        self.configure_export_btn = QPushButton("Export Frames...")
        self.configure_export_btn.setToolTip("Open frame selection for export configuration")
        self.configure_export_btn.clicked.connect(self.open_export_config)
        export_layout.addWidget(self.configure_export_btn)

        # Row 3: Project-level export
        self.export_project_btn = QPushButton("Export Entire Project (COCO)")
        self.export_project_btn.setEnabled(False)
        self.export_project_btn.setToolTip(
            "Export saved masks for the entire project to a single COCO file"
        )
        self.export_project_btn.clicked.connect(self.export_project_to_coco)
        export_layout.addWidget(self.export_project_btn)

        # Insert at the very bottom of main layout
        main_layout = self.layout()
        if main_layout is not None:
            main_layout.addWidget(export_group)

    def open_export_config(self):
        """Open the export configuration dialog."""
        # Update the dialog with current layer info before showing
        self.update_export_frame_list()
        # Show the dialog
        self.export_config_dialog.exec_()

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

    def _get_current_image_layer(self):
        """
        Get the currently selected image layer from the combo box.

        Returns:
            napari.layers.Image or None: The selected layer, or None if invalid/empty.
        """
        layer_name = self.image_layers_combo.currentText()
        if not layer_name:
            return None

        try:
            layer = self.viewer.layers[layer_name]
            if isinstance(layer, napari.layers.Image):
                return layer
        except KeyError:
            pass  # Layer was deleted

        return None

    def update_export_frame_list(self):
        """
        Update the export configuration dialog to match the current image layer.

        Called when:
        - User selects a different image layer
        - Layers are added/removed
        - Pipeline is initialized
        - User opens the export config dialog
        """
        layer = self._get_current_image_layer()

        if layer is None:
            # No valid layer selected
            self.export_config_dialog.set_total_frames(0)
            return

        # Determine number of frames
        shape = layer.data.shape

        if len(shape) == 3:
            # Grayscale video: (T, Y, X)
            num_frames = shape[0]
        elif len(shape) == 4:
            # Color video: (T, Y, X, C) or (T, C, Y, X)
            num_frames = shape[0]
        else:
            # 2D image or invalid shape
            num_frames = 0

        self.export_config_dialog.set_total_frames(num_frames)

    # Function to handle layer changes
    def layer_changed(self):
        # Populate combo box and label list
        self.populate_combo_box(self.image_layers_combo, "image")
        self.populate_label_layers_list()
        # Update export frame list when layers change
        self.update_export_frame_list()

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
            # Update export frame list after pipeline initialization
            self.update_export_frame_list()
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

    def on_mouse_click(self, layer, event):
        """Mouse click callback - currently unused.

        Point prompting has been removed in favor of direct mask drawing with napari tools.
        """
        pass  # Placeholder for future mouse interactions

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

    # ====================================================================
    # Project Management Methods
    # ====================================================================

    def load_project_folder(self):
        """Open file dialog to select and load a project folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Project Folder",
            os.path.expanduser("~"),
        )

        if not folder:
            return

        try:
            # Load project
            self.current_project = Project.load(folder)
            self.current_video_index = 0

            # Update UI
            project_name = os.path.basename(folder)
            stats = self.current_project.stats()
            self.project_info_label.setText(
                f"Project: {project_name} | Videos: {stats['total']} "
                f"(Pending: {stats['pending']}, Completed: {stats['completed']})"
            )

            # Load first video
            self.load_current_video()

            # Enable navigation buttons
            self.update_navigation_buttons()

            # Enable export button
            if hasattr(self, 'export_coco_btn'):
                self.export_coco_btn.setEnabled(True)
            if hasattr(self, 'export_complete_btn'):
                self.export_complete_btn.setEnabled(True)
            if hasattr(self, 'export_project_btn'):
                self.export_project_btn.setEnabled(True)

            show_info(f"Loaded project: {project_name}")

        except Exception as e:
            show_info(f"Error loading project: {str(e)}")

    def load_current_video(self):
        """Load the current video into napari with auto-created label layers."""
        if self.current_project is None:
            return

        video_ids = self.current_project.video_ids()
        if self.current_video_index >= len(video_ids):
            show_info("No more videos in project")
            return

        # Save masks from previous video BEFORE clearing layers
        if hasattr(self, '_previous_video_id') and self._previous_video_id:
            self._save_masks_for_video(self._previous_video_id)

        # Clear existing layers
        self.viewer.layers.clear()

        # Clear approved anchors when switching videos
        if hasattr(self, "pipeline_object"):
            self.pipeline_object.clear_approved_frames()
            self.update_approved_display()

        self.current_video_id = video_ids[self.current_video_index]

        # Load frames
        frames = self.current_project.load_frames(self.current_video_id)
        metadata = self.current_project.load_metadata(self.current_video_id)

        # Add image layer
        self.viewer.add_image(frames, name=self.current_video_id)

        # Auto-create label layers for each class
        class_names = self.current_project.class_names()
        for class_name in class_names:
            # Create empty labels layer matching frame dimensions
            labels_data = np.zeros(frames.shape[:3], dtype=np.uint16)

            # Try to load existing masks if they exist
            self._load_existing_masks(self.current_video_id, class_name, labels_data, metadata)

            # Add labels layer
            self.viewer.add_labels(labels_data, name=class_name)

        # Update video info
        manifest_entry = self.current_project.manifest["videos"][self.current_video_id]
        self.video_info_label.setText(
            f"Video {self.current_video_index + 1}/{len(video_ids)}: "
            f"{manifest_entry['source']} | "
            f"Status: {manifest_entry['status']} | "
            f"Frames: {metadata['total_frames']}"
        )

        # Auto-check all label layers
        self.populate_label_layers_list()
        self._auto_check_all_labels()

        # Update export frame list
        self.update_export_frame_list()

        # Track for saving later
        self._previous_video_id = self.current_video_id

    def _load_existing_masks(self, video_id, class_name, labels_data, metadata):
        """Load existing masks from disk if they exist."""
        masks_dir = self.current_project.masks_dir(video_id, class_name)

        for frame_info in metadata["frames"]:
            seq_idx = frame_info["seq"]
            filename = frame_info["file"]
            frame_stem = filename.rsplit(".", 1)[0]  # frame_00000
            mask_path = masks_dir / f"{frame_stem}.png"

            if mask_path.exists():
                try:
                    mask = read_indexed_mask(mask_path)
                    labels_data[seq_idx] = mask
                except Exception as e:
                    print(f"Warning: Could not load mask {mask_path}: {e}")

    def _auto_check_all_labels(self):
        """Automatically check all label layers in the output list."""
        for i in range(self.output_layers_list.count()):
            item = self.output_layers_list.item(i)
            item.setCheckState(Qt.Checked)

    def save_current_masks(self):
        """Save current label layers to disk as indexed masks."""
        if self.current_project is None or self.current_video_id is None:
            return
        self._save_masks_for_video(self.current_video_id)

    def _save_masks_for_video(self, video_id):
        """Save masks for a specific video ID."""
        if self.current_project is None:
            return

        metadata = self.current_project.load_metadata(video_id)
        class_names = self.current_project.class_names()

        for class_name in class_names:
            # Get the label layer
            try:
                layer = self.viewer.layers[class_name]
                if not isinstance(layer, napari.layers.Labels):
                    continue
            except KeyError:
                continue

            masks_dir = self.current_project.masks_dir(video_id, class_name)
            masks_dir.mkdir(parents=True, exist_ok=True)

            # Save each frame
            for frame_info in metadata["frames"]:
                seq_idx = frame_info["seq"]
                filename = frame_info["file"]
                frame_stem = filename.rsplit(".", 1)[0]  # frame_00000

                # Bounds check to avoid IndexError
                if seq_idx >= layer.data.shape[0]:
                    continue

                mask = layer.data[seq_idx]
                mask_path = masks_dir / f"{frame_stem}.png"

                # Only save if mask has data
                if mask.max() > 0:
                    write_indexed_mask(mask, mask_path)

    def next_video(self):
        """Load next video in the project."""
        if self.current_project is None:
            return

        # Confirm before switching
        reply = QMessageBox.question(
            self,
            "Save and Continue?",
            "Save current masks and move to next video?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply != QMessageBox.Yes:
            return

        video_ids = self.current_project.video_ids()
        if self.current_video_index < len(video_ids) - 1:
            self.current_video_index += 1
            self.load_current_video()
            self.update_navigation_buttons()

    def prev_video(self):
        """Load previous video in the project."""
        if self.current_project is None:
            return

        # Confirm before switching
        reply = QMessageBox.question(
            self,
            "Save and Continue?",
            "Save current masks and move to previous video?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply != QMessageBox.Yes:
            return

        if self.current_video_index > 0:
            self.current_video_index -= 1
            self.load_current_video()
            self.update_navigation_buttons()

    def mark_complete_and_next(self):
        """Mark current video as completed and load next."""
        if self.current_project is None or self.current_video_id is None:
            return

        # Check if there are any masks
        has_masks = self._check_has_masks()

        if not has_masks:
            reply = QMessageBox.warning(
                self,
                "No Masks Found",
                "No masks detected in any label layer. Mark as complete anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        # Confirm completion
        reply = QMessageBox.question(
            self,
            "Mark Complete?",
            f"Mark video '{self.current_video_id}' as complete and save masks?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply != QMessageBox.Yes:
            return

        # Save current masks
        self.save_current_masks()

        # Update status
        self.current_project.update_status(self.current_video_id, "completed")

        # Move to next
        video_ids = self.current_project.video_ids()
        if self.current_video_index < len(video_ids) - 1:
            self.current_video_index += 1
            self.load_current_video()
        else:
            show_info("All videos completed!")

        self.update_navigation_buttons()

        # Update project info
        stats = self.current_project.stats()
        project_name = os.path.basename(str(self.current_project.root))
        self.project_info_label.setText(
            f"Project: {project_name} | Videos: {stats['total']} "
            f"(Pending: {stats['pending']}, Completed: {stats['completed']})"
        )

    def _check_has_masks(self):
        """Check if any label layers have non-zero masks."""
        class_names = self.current_project.class_names()
        for class_name in class_names:
            try:
                layer = self.viewer.layers[class_name]
                if isinstance(layer, napari.layers.Labels):
                    if layer.data.max() > 0:
                        return True
            except KeyError:
                continue
        return False

    def update_navigation_buttons(self):
        """Update enabled state of navigation buttons."""
        if self.current_project is None:
            self.prev_video_btn.setEnabled(False)
            self.next_video_btn.setEnabled(False)
            self.complete_next_btn.setEnabled(False)
            if hasattr(self, "export_coco_btn"):
                self.export_coco_btn.setEnabled(False)
            if hasattr(self, "export_complete_btn"):
                self.export_complete_btn.setEnabled(False)
            if hasattr(self, "export_project_btn"):
                self.export_project_btn.setEnabled(False)
            return

        video_ids = self.current_project.video_ids()

        self.prev_video_btn.setEnabled(self.current_video_index > 0)
        self.next_video_btn.setEnabled(self.current_video_index < len(video_ids) - 1)
        self.complete_next_btn.setEnabled(True)
        if hasattr(self, "export_coco_btn"):
            self.export_coco_btn.setEnabled(True)
        if hasattr(self, "export_complete_btn"):
            self.export_complete_btn.setEnabled(True)
        if hasattr(self, "export_project_btn"):
            self.export_project_btn.setEnabled(True)

    def _export_coco(self, video_ids=None, frames_by_video=None):
        """Run COCO export and return (output_path, coco_dict)."""
        from .ingestion.coco_export import COCOExporter

        exporter = COCOExporter(self.current_project)
        output_path = str(self.current_project.export_dir() / "annotations.json")
        coco = exporter.export(
            output_path=output_path,
            video_ids=video_ids,
            frames_by_video=frames_by_video,
        )
        return output_path, coco

    def mark_complete_and_export(self):
        """Mark current video complete, export its masks, then go to next."""
        if self.current_project is None or self.current_video_id is None:
            return

        if len(self.export_config_dialog.export_frames) == 0:
            QMessageBox.warning(
                self,
                "No Frames Selected",
                "No frames selected for export. Please configure frame export first.",
            )
            self.open_export_config()
            return

        # Check if there are any masks
        has_masks = self._check_has_masks()
        if not has_masks:
            reply = QMessageBox.warning(
                self,
                "No Masks Found",
                "No masks detected in any label layer. Mark as complete and export anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        reply = QMessageBox.question(
            self,
            "Mark Complete & Export",
            f"Mark video '{self.current_video_id}' as complete and export its masks?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        # Save current masks
        self.save_current_masks()

        # Update status
        self.current_project.update_status(self.current_video_id, "completed")

        # Export only current video (background)
        frames_by_video = {self.current_video_id: set(self.export_config_dialog.export_frames)}
        self.export_complete_btn.setEnabled(False)
        worker = thread_worker(
            lambda: self._export_coco(
                video_ids=[self.current_video_id],
                frames_by_video=frames_by_video,
            )
        )()
        self._export_worker = worker

        def _on_export_done(result):
            output_path, coco = result
            show_info(
                f"Export complete for '{self.current_video_id}'.\n\n"
                f"Images: {len(coco['images'])}\n"
                f"Annotations: {len(coco['annotations'])}\n"
                f"Categories: {len(coco['categories'])}\n\n"
                f"Saved to: {output_path}"
            )

            # Move to next
            video_ids = self.current_project.video_ids()
            if self.current_video_index < len(video_ids) - 1:
                self.current_video_index += 1
                self.load_current_video()
            else:
                show_info("All videos completed!")

            self.update_navigation_buttons()

            # Update project info
            stats = self.current_project.stats()
            project_name = os.path.basename(str(self.current_project.root))
            self.project_info_label.setText(
                f"Project: {project_name} | Videos: {stats['total']} "
                f"(Pending: {stats['pending']}, Completed: {stats['completed']})"
            )

        def _on_export_error(err):
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Error during export:\n{str(err)}",
            )
            self.update_navigation_buttons()

        worker.returned.connect(_on_export_done)
        worker.errored.connect(_on_export_error)
        worker.start()

        return

    def export_to_coco(self):
        """Export current video to COCO format."""
        if self.current_project is None:
            show_info("No project loaded")
            return

        # Save current video masks first
        if self.current_video_id:
            self.save_current_masks()

        if len(self.export_config_dialog.export_frames) == 0:
            QMessageBox.warning(
                self,
                "No Frames Selected",
                "No frames selected for export. Please configure frame export first.",
            )
            self.open_export_config()
            return

        frames_by_video = {self.current_video_id: set(self.export_config_dialog.export_frames)}

        self.export_coco_btn.setEnabled(False)
        worker = thread_worker(
            lambda: self._export_coco(
                video_ids=[self.current_video_id],
                frames_by_video=frames_by_video,
            )
        )()
        self._export_worker = worker

        def _on_export_done(result):
            output_path, coco = result

            show_info(
                f"Export complete for '{self.current_video_id}'.\n\n"
                f"Images: {len(coco['images'])}\n"
                f"Annotations: {len(coco['annotations'])}\n"
                f"Categories: {len(coco['categories'])}\n\n"
                f"Saved to: {output_path}"
            )

            self.update_navigation_buttons()

        def _on_export_error(err):
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Error during export:\n{str(err)}"
            )
            self.update_navigation_buttons()

        worker.returned.connect(_on_export_done)
        worker.errored.connect(_on_export_error)
        worker.start()

    def export_project_to_coco(self):
        """Export entire project to COCO format."""
        if self.current_project is None:
            show_info("No project loaded")
            return

        # Save current video masks first
        if self.current_video_id:
            self.save_current_masks()

        stats = self.current_project.stats()
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Export Entire Project (COCO)")
        dialog.setText(
            "Export videos to COCO format?\n\n"
            f"Total videos: {stats['total']}\n"
            f"Completed: {stats['completed']}\n"
            f"Pending: {stats['pending']}\n"
        )
        export_all_btn = dialog.addButton("Export All", QMessageBox.AcceptRole)
        export_completed_btn = dialog.addButton("Only Completed", QMessageBox.DestructiveRole)
        cancel_btn = dialog.addButton(QMessageBox.Cancel)
        dialog.exec_()

        clicked = dialog.clickedButton()
        if clicked is None or clicked == cancel_btn:
            return

        if clicked == export_completed_btn:
            video_ids = [
                vid for vid, info in self.current_project.manifest["videos"].items()
                if info["status"] == "completed"
            ]
            if not video_ids:
                show_info("No completed videos to export")
                return
        else:
            video_ids = None

        self.export_project_btn.setEnabled(False)
        worker = thread_worker(
            lambda: self._export_coco(video_ids=video_ids)
        )()
        self._export_worker = worker

        def _on_export_done(result):
            output_path, coco = result

            show_info(
                "Export complete!\n\n"
                f"Images: {len(coco['images'])}\n"
                f"Annotations: {len(coco['annotations'])}\n"
                f"Categories: {len(coco['categories'])}\n\n"
                f"Saved to: {output_path}"
            )

            self.update_navigation_buttons()

        def _on_export_error(err):
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Error during export:\n{str(err)}"
            )
            self.update_navigation_buttons()

        worker.returned.connect(_on_export_done)
        worker.errored.connect(_on_export_error)
        worker.start()

    # ====================================================================
    # End Project Management Methods
    # ====================================================================

    def delete_source_dir(self):
        """Deletes the temporary source frame directory when Napari closes."""
        if hasattr(self, "pipeline_object"):
            self.pipeline_object.delete_source_frame_dir()

    def cleanup_all_temp_dirs(self):
        """Delete potential temporary directories from previous sessions that have crashed"""
        for dir_path in glob.glob("/tmp/tmp*naparisam2long"):
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
