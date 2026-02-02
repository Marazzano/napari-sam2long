#!/usr/bin/env python
"""Test script to organize videos without CLI installation."""

from napari_sam2long.ingestion import ProjectConfig, ProjectOrganizer

# Configure the project
config = ProjectConfig(
    source_folder="/Users/marazzanocolon/coding/sam2_video_mask_generation/raw_videos",
    output_folder="/Users/marazzanocolon/coding/napari-sam2long/test_project",
    class_names=("fish", "rock"),  # Add your class names here
    step_size=10,  # Extract every 10th frame
    reverse=True,  # Start from end of video
)

# Progress callback
def progress(current, total, video_name):
    print(f"[{current}/{total}] Processing {video_name}")

# Run organizer
print("Starting video organization...")
organizer = ProjectOrganizer(config)
project = organizer.organize(progress_callback=progress, skip_existing=True)

# Print results
stats = project.stats()
print(f"\n✓ Project created: {config.output_folder}")
print(f"  Videos: {stats['total']} total")
print(f"  Pending: {stats['pending']}")
print(f"  Completed: {stats['completed']}")
