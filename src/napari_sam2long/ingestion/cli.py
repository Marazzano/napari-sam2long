"""CLI entry points."""

import argparse
import sys


def organize_main():
    """napari-sam2long-organize: Batch process folder of videos."""
    parser = argparse.ArgumentParser(description="Organize videos for napari-sam2long annotation")
    parser.add_argument("source", help="Folder containing video files")
    parser.add_argument("-o", "--output", required=True, help="Output project folder")
    parser.add_argument("--classes", nargs="*", default=[], help="Class names for mask folders")
    parser.add_argument("--step", type=int, default=10, help="Extract every Nth frame (default: 10)")
    parser.add_argument("--no-reverse", action="store_true", help="Extract start-to-end (default: reverse)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing videos")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip already processed (default)")

    args = parser.parse_args()

    from .config import ProjectConfig
    from .organizer import ProjectOrganizer

    config = ProjectConfig(
        source_folder=args.source,
        output_folder=args.output,
        class_names=tuple(args.classes),
        step_size=args.step,
        reverse=not args.no_reverse,
    )

    def progress(curr, total, name):
        print(f"[{curr}/{total}] Processing {name}")

    organizer = ProjectOrganizer(config)
    project = organizer.organize(progress_callback=progress, force=args.force)

    stats = project.stats()
    print(f"\nProject created: {args.output}")
    print(f"Videos: {stats['total']} total, {stats['pending']} pending")


def ingest_main():
    """napari-sam2long-ingest: Process single video."""
    parser = argparse.ArgumentParser(description="Ingest single video")
    parser.add_argument("video", help="Video file path")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--classes", nargs="*", default=[], help="Class names")
    parser.add_argument("--step", type=int, default=10, help="Frame step")
    parser.add_argument("--no-reverse", action="store_true")

    args = parser.parse_args()

    from .config import IngestionConfig
    from .ingester import VideoIngester

    config = IngestionConfig(
        video_path=args.video,
        output_dir=args.output,
        class_names=tuple(args.classes),
        step_size=args.step,
        reverse=not args.no_reverse,
    )

    def progress(curr, total):
        pct = curr / total * 100
        print(f"\rExtracting: {curr}/{total} ({pct:.0f}%)", end="", flush=True)

    metadata = VideoIngester(config).run(progress_callback=progress)
    print(f"\nExtracted {metadata['total_frames']} frames to {args.output}")


def export_main():
    """napari-sam2long-export-coco: Export to COCO format."""
    parser = argparse.ArgumentParser(description="Export masks to COCO")
    parser.add_argument("project", help="Project folder")
    parser.add_argument("-o", "--output", help="Output file (default: project/exports/coco/annotations.json)")
    parser.add_argument("--videos", nargs="*", help="Specific video IDs (default: all)")

    args = parser.parse_args()

    from .project import Project
    from .coco_export import COCOExporter

    project = Project.load(args.project)
    exporter = COCOExporter(project)
    coco = exporter.export(output_path=args.output, video_ids=args.videos)

    print(f"Exported {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    print(f"Categories: {[c['name'] for c in coco['categories']]}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        sys.argv.pop(1)
        if cmd == "organize":
            organize_main()
        elif cmd == "ingest":
            ingest_main()
        elif cmd == "export":
            export_main()
