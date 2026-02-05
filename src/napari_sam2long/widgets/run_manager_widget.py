"""
Widget for managing annotation, training, and inference runs.

Provides UI for:
- Viewing and selecting runs
- Saving current masks to a new run
- Loading masks from a run
- Starting training
- Running inference
"""

from pathlib import Path
from typing import Optional

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QFormLayout,
    QTextEdit,
    QMessageBox,
    QProgressDialog,
    QInputDialog,
)
from qtpy.QtCore import Qt, Signal

from ..ingestion.run_manager import RunManager


class RunManagerWidget(QWidget):
    """
    Widget for viewing and managing runs.

    Signals:
        run_loaded: Emitted when user loads a run (run_type, run_id)
        training_requested: Emitted when user wants to train (run_ids_to_use)
        inference_requested: Emitted when user wants to run inference (train_run_id)
    """

    run_loaded = Signal(str, int)  # (run_type, run_id)
    training_requested = Signal(list)  # [(run_type, run_id), ...]
    inference_requested = Signal(int)  # train_run_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.run_manager: Optional[RunManager] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Run list section
        list_group = QGroupBox("Runs")
        list_layout = QVBoxLayout(list_group)

        # Filter dropdown
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "annot", "train", "infer"])
        self.filter_combo.currentTextChanged.connect(self._refresh_run_list)
        filter_row.addWidget(self.filter_combo)
        filter_row.addStretch()
        list_layout.addLayout(filter_row)

        # Run list
        self.run_list = QListWidget()
        self.run_list.setMaximumHeight(150)
        self.run_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.run_list.itemDoubleClicked.connect(self._load_selected_run)
        list_layout.addWidget(self.run_list)

        # Run info display
        self.run_info = QTextEdit()
        self.run_info.setReadOnly(True)
        self.run_info.setMaximumHeight(80)
        self.run_info.setStyleSheet("font-size: 10px;")
        list_layout.addWidget(self.run_info)

        layout.addWidget(list_group)

        # Action buttons section
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        # Row 1: Save/Load
        row1 = QHBoxLayout()
        self.save_btn = QPushButton("Save → New Run")
        self.save_btn.setToolTip("Save current working masks to a new annotation run")
        self.save_btn.clicked.connect(self._save_current_to_run)
        row1.addWidget(self.save_btn)

        self.load_btn = QPushButton("Load Selected")
        self.load_btn.setToolTip("Load masks from selected run into working directory")
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(self._load_selected_run)
        row1.addWidget(self.load_btn)
        actions_layout.addLayout(row1)

        # Row 2: Training
        row2 = QHBoxLayout()
        self.train_btn = QPushButton("Train Model...")
        self.train_btn.setToolTip("Train a model on selected annotation run(s)")
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self._start_training_dialog)
        row2.addWidget(self.train_btn)

        self.infer_btn = QPushButton("Run Inference...")
        self.infer_btn.setToolTip("Run inference using selected training run")
        self.infer_btn.setEnabled(False)
        self.infer_btn.clicked.connect(self._start_inference_dialog)
        row2.addWidget(self.infer_btn)
        actions_layout.addLayout(row2)

        layout.addWidget(actions_group)

    def set_project(self, project_root: Path):
        """Set the project and load runs."""
        self.run_manager = RunManager(project_root)
        self._refresh_run_list()

    def _refresh_run_list(self):
        """Refresh the run list based on current filter."""
        self.run_list.clear()
        self.run_info.clear()

        if not self.run_manager:
            return

        filter_type = self.filter_combo.currentText()
        run_type = None if filter_type == "All" else filter_type

        runs = self.run_manager.list_runs(run_type)

        for run in runs:
            # Format: "annot_001_label" with color coding
            display_name = run.get("name", f"{run['type']}_{run['id']:03d}")
            item = QListWidgetItem(display_name)
            item.setData(Qt.UserRole, (run["type"], run["id"]))

            # Color code by type
            if run["type"] == "annot":
                item.setForeground(Qt.blue)
            elif run["type"] == "train":
                item.setForeground(Qt.darkGreen)
            elif run["type"] == "infer":
                item.setForeground(Qt.darkMagenta)

            self.run_list.addItem(item)

    def _on_selection_changed(self):
        """Handle selection change in run list."""
        items = self.run_list.selectedItems()
        has_selection = len(items) > 0

        if has_selection:
            run_type, run_id = items[0].data(Qt.UserRole)
            run = self.run_manager.get_run(run_type, run_id)

            # Update info display
            info = f"Type: {run['type']}\n"
            info += f"ID: {run['id']}\n"
            if run.get("label"):
                info += f"Label: {run['label']}\n"
            if run.get("created_at"):
                info += f"Created: {run['created_at'][:19]}\n"
            if run.get("parents"):
                parents_str = ", ".join(
                    f"{p['type']}-{p['id']:03d}" for p in run["parents"]
                )
                info += f"Parents: {parents_str}\n"
            if run.get("final_metrics"):
                metrics = run["final_metrics"]
                if "val_iou" in metrics:
                    info += f"Val IoU: {metrics['val_iou']:.4f}\n"

            self.run_info.setText(info)

            # Enable/disable buttons based on run type
            self.load_btn.setEnabled(run_type in ("annot", "infer"))
            self.train_btn.setEnabled(run_type == "annot")
            self.infer_btn.setEnabled(run_type == "train")
        else:
            self.run_info.clear()
            self.load_btn.setEnabled(False)
            self.train_btn.setEnabled(False)
            self.infer_btn.setEnabled(False)

    def _save_current_to_run(self):
        """Save current working masks to a new annotation run."""
        if not self.run_manager:
            return

        # Get label from user
        label, ok = QInputDialog.getText(
            self,
            "Save to Run",
            "Label for this annotation run:",
            text="manual-annotation",
        )
        if not ok or not label:
            return

        # Get parent if any
        parent = None
        items = self.run_list.selectedItems()
        if items:
            run_type, run_id = items[0].data(Qt.UserRole)
            if run_type == "infer":
                reply = QMessageBox.question(
                    self,
                    "Set Parent?",
                    f"Set {run_type}_{run_id:03d} as parent of this run?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply == QMessageBox.Yes:
                    parent = f"{run_type}-{run_id:03d}"

        try:
            # Create run
            run_path = self.run_manager.create_run("annot", label=label, parent=parent)

            # Save working masks (this will be called by parent widget)
            # For now, just emit signal
            QMessageBox.information(
                self,
                "Run Created",
                f"Created run: {run_path.name}\n\n"
                "Note: Save your current masks using the main widget's save function, "
                "then use 'Load Selected' to copy them to this run.",
            )

            self._refresh_run_list()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create run:\n{str(e)}")

    def _load_selected_run(self):
        """Load masks from selected run."""
        items = self.run_list.selectedItems()
        if not items:
            return

        run_type, run_id = items[0].data(Qt.UserRole)

        if run_type == "train":
            QMessageBox.warning(
                self,
                "Cannot Load",
                "Training runs don't contain masks.\n"
                "Select an annotation or inference run.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Load Run",
            f"Load masks from {run_type}_{run_id:03d}?\n\n"
            "This will overwrite current working masks.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        try:
            self.run_manager.load_run_to_working(run_type, run_id)
            self.run_loaded.emit(run_type, run_id)
            QMessageBox.information(
                self,
                "Loaded",
                f"Masks loaded from {run_type}_{run_id:03d}.\n"
                "Reload your video to see the changes.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load run:\n{str(e)}")

    def _start_training_dialog(self):
        """Open training configuration dialog."""
        items = self.run_list.selectedItems()
        if not items:
            return

        run_type, run_id = items[0].data(Qt.UserRole)

        dialog = TrainingConfigDialog(self, run_type, run_id)
        if dialog.exec_():
            config = dialog.get_config()
            self.training_requested.emit([(run_type, run_id)])

    def _start_inference_dialog(self):
        """Open inference configuration dialog."""
        items = self.run_list.selectedItems()
        if not items:
            return

        run_type, run_id = items[0].data(Qt.UserRole)

        if run_type != "train":
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "Select a training run to run inference.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Run Inference",
            f"Run inference using train_{run_id:03d}?\n\n"
            "This will create a new inference run with predictions for all videos.",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.inference_requested.emit(run_id)


class TrainingConfigDialog(QDialog):
    """Dialog for configuring model training."""

    def __init__(self, parent, run_type: str, run_id: int):
        super().__init__(parent)
        self.run_type = run_type
        self.run_id = run_id
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Training Configuration")
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)

        # Form
        form = QFormLayout()

        self.model_combo = QComboBox()
        self.model_combo.addItems(["unet", "maskrcnn"])
        form.addRow("Model:", self.model_combo)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        form.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(8)
        form.addRow("Batch Size:", self.batch_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        form.addRow("Learning Rate:", self.lr_spin)

        layout.addLayout(form)

        # Info
        info = QLabel(f"Training on: {self.run_type}_{self.run_id:03d}")
        info.setStyleSheet("color: gray;")
        layout.addWidget(info)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self) -> dict:
        """Get training configuration from dialog."""
        return {
            "model": self.model_combo.currentText(),
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_spin.value(),
            "learning_rate": self.lr_spin.value(),
        }
