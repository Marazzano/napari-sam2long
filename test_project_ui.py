#!/usr/bin/env python
"""Test script to launch napari with the SAM2Long plugin and Project Manager."""

import napari

# Launch napari
viewer = napari.Viewer()

# Add the SAM2Long widget
viewer.window.add_plugin_dock_widget('napari-sam2long', 'SAM2Long')

print("\n" + "="*70)
print("NAPARI SAM2LONG - PROJECT MANAGER TEST")
print("="*70)
print("\nInstructions:")
print("1. Click 'Load Project...' in the Project Manager section")
print("2. Navigate to: /Users/marazzanocolon/coding/sam2_video_mask_generation/processed_project")
print("3. Click 'Select Folder'")
print("\nThe plugin should:")
print("✓ Load the first video automatically")
print("✓ Create 'fish' and 'rock' label layers")
print("✓ Auto-check both layers in the Labels list")
print("✓ Show project info and video progress")
print("\nThen you can:")
print("- Navigate between videos with Previous/Next buttons")
print("- Use 'Complete & Next' to mark video as done and move forward")
print("- SAM2Long will work on the auto-created label layers")
print("="*70 + "\n")

napari.run()
