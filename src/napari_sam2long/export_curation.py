"""
Export curation panel for frame-level selection in COCO export.

This module provides a self-contained Qt widget that allows users to select
which frames should be included when exporting to COCO format.
"""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QLabel, QDialog, QDialogButtonBox
)
from qtpy.QtCore import Qt


class ExportCurationPanel(QWidget):
    """
    Frame-level export curation panel for COCO export.

    Provides:
    - Checkable list of frames
    - Bulk selection controls (Approve All, Clear All, Flip Selection)
    - Status HUD showing export count

    Attributes:
        export_frames (set[int]): Set of frame indices selected for export
        frame_items (dict[int, QListWidgetItem]): Mapping from frame index to list item
        total_frames (int): Total number of frames in current video
    """

    def __init__(self, total_frames: int = 0, parent=None):
        """
        Initialize the export curation panel.

        Args:
            total_frames: Number of frames in the current video layer.
                         Set to 0 for no layer selected.
            parent: Parent Qt widget
        """
        super().__init__(parent)

        # Core state
        self.export_frames: set[int] = set()
        self.frame_items: dict[int, QListWidgetItem] = {}
        self.total_frames: int = 0

        self._setup_ui()
        self.set_total_frames(total_frames)

    def _setup_ui(self):
        """Create the UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 5, 0, 5)

        # Bulk action buttons row
        button_row = QHBoxLayout()

        self.approve_all_btn = QPushButton("Approve All")
        self.approve_all_btn.setToolTip("Select all frames for export")
        self.approve_all_btn.clicked.connect(lambda: self._bulk_set(True))
        button_row.addWidget(self.approve_all_btn)

        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.setToolTip("Deselect all frames")
        self.clear_all_btn.clicked.connect(lambda: self._bulk_set(False))
        button_row.addWidget(self.clear_all_btn)

        self.flip_btn = QPushButton("Flip Selection")
        self.flip_btn.setToolTip("Invert current selection")
        self.flip_btn.clicked.connect(self._flip_selection)
        button_row.addWidget(self.flip_btn)

        main_layout.addLayout(button_row)

        # Frame list (checkable)
        self.frame_list = QListWidget()
        self.frame_list.setMaximumHeight(150)
        self.frame_list.setSelectionMode(QListWidget.NoSelection)
        self.frame_list.itemChanged.connect(self._on_item_changed)
        main_layout.addWidget(self.frame_list)

        # Status HUD
        self.status_label = QLabel("Exporting: 0 / 0 frames")
        self.status_label.setStyleSheet("font-family: monospace; color: #888;")
        main_layout.addWidget(self.status_label)

    def set_total_frames(self, total_frames: int):
        """
        Rebuild the frame list for a new video.

        Args:
            total_frames: Number of frames in the current video layer.
                         Set to 0 to clear the list (no layer selected).

        Default behavior: All frames approved for export.
        """
        self.total_frames = total_frames

        # Block signals during rebuild to avoid triggering itemChanged
        self.frame_list.blockSignals(True)
        self.frame_list.clear()
        self.frame_items.clear()

        if total_frames == 0:
            # No video loaded
            self.export_frames.clear()
            self.frame_list.blockSignals(False)
            self._update_status()
            return

        # Create checkable items for each frame (default: none selected)
        for t in range(total_frames):
            item = QListWidgetItem(f"Frame {t}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.frame_list.addItem(item)
            self.frame_items[t] = item

        # Initialize export set with none selected
        self.export_frames = set()

        self.frame_list.blockSignals(False)
        self._update_status()

    def _on_item_changed(self, item: QListWidgetItem):
        """Handle individual checkbox toggle using row index."""
        row = self.frame_list.row(item)
        if row == -1:
            return

        if item.checkState() == Qt.Checked:
            self.export_frames.add(row)
        else:
            self.export_frames.discard(row)

        self._update_status()

    def _bulk_set(self, state: bool):
        """
        Bulk approve or clear all frames.

        Args:
            state: True to approve all, False to clear all
        """
        self.frame_list.blockSignals(True)  # Prevent per-item signals

        qt_state = Qt.Checked if state else Qt.Unchecked

        for t in range(self.total_frames):
            item = self.frame_items.get(t)
            if item:
                item.setCheckState(qt_state)

        # Update export set
        if state:
            self.export_frames = set(range(self.total_frames))
        else:
            self.export_frames.clear()

        self.frame_list.blockSignals(False)
        self._update_status()

    def _flip_selection(self):
        """Invert current selection state."""
        self.frame_list.blockSignals(True)

        for t in range(self.total_frames):
            item = self.frame_items.get(t)
            if item:
                current = item.checkState()
                new_state = Qt.Unchecked if current == Qt.Checked else Qt.Checked
                item.setCheckState(new_state)

        # Invert export set
        all_frames = set(range(self.total_frames))
        self.export_frames = all_frames - self.export_frames

        self.frame_list.blockSignals(False)
        self._update_status()

    def _update_status(self):
        """Update the status label showing export count."""
        count = len(self.export_frames)
        total = self.total_frames
        self.status_label.setText(f"Exporting: {count} / {total} frames")


class ExportConfigDialog(QDialog):
    """
    Dialog for configuring export settings.

    Provides a spacious interface for:
    - Frame-level export curation
    - Future export options (COCO format settings, etc.)
    """

    def __init__(self, total_frames: int = 0, parent=None):
        """
        Initialize the export configuration dialog.

        Args:
            total_frames: Number of frames in the current video layer
            parent: Parent Qt widget
        """
        super().__init__(parent)

        self.setWindowTitle("Configure Export Settings")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        # Main layout
        main_layout = QVBoxLayout(self)

        # Section title
        title = QLabel("Frame Selection for Export")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        main_layout.addWidget(title)

        # Export curation panel (has more space here than in main widget)
        self.export_panel = ExportCurationPanel(total_frames, parent=self)
        # Remove max height constraint for more space in dialog
        self.export_panel.frame_list.setMaximumHeight(16777215)  # Qt max
        self.export_panel.frame_list.setMinimumHeight(250)
        main_layout.addWidget(self.export_panel)

        # Future: Add more export configuration options here
        # e.g., COCO format options, export path, etc.

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.accept)  # Close button
        main_layout.addWidget(button_box)

    def set_total_frames(self, total_frames: int):
        """
        Update the number of frames when video layer changes.

        Args:
            total_frames: Number of frames in the current video layer
        """
        self.export_panel.set_total_frames(total_frames)

    @property
    def export_frames(self):
        """Get the set of selected frame indices."""
        return self.export_panel.export_frames
