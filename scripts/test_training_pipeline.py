#!/usr/bin/env python3
"""
End-to-end test for the multi-model active learning pipeline.

This script tests the full workflow:
1. Load a test project (created by download_davis_test_data.py)
2. Collect training data from annotation run
3. Train a UNet model
4. Run inference on all videos
5. Evaluate predictions against ground truth
6. Create a new annotation run from predictions

Usage:
    # First, create test data:
    python scripts/download_davis_test_data.py --output ./test_davis_project -n 3

    # Then run this test:
    python scripts/test_training_pipeline.py --project ./test_davis_project

Requirements:
    pip install segmentation-models-pytorch pycocotools torch torchvision
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_run_manager(project_path: Path):
    """Test RunManager functionality."""
    print("\n" + "=" * 60)
    print("TEST: RunManager")
    print("=" * 60)

    from napari_sam2long.ingestion import RunManager

    mgr = RunManager(project_path)

    # List existing runs
    runs = mgr.list_runs()
    print(f"Existing runs: {len(runs)}")
    for run in runs:
        print(f"  - {run['name']} (type: {run['type']})")

    # Create initial annotation run if none exists
    if not runs:
        print("No existing runs — creating initial annotation run from working masks...")
        annot_run = mgr.create_run("annot", label="initial")
        video_ids = [d.name for d in (project_path / "vids").iterdir() if d.is_dir()]
        mgr.save_working_to_run("annot", 1, video_ids=video_ids)
        print(f"Created: {annot_run.name}")
        runs = mgr.list_runs()

    # Test creating another run
    next_id = max(r["id"] for r in mgr._get_existing_runs("annot")) + 1
    new_run = mgr.create_run("annot", label="test-run", parent=None)
    print(f"Created: {new_run.name}")

    # Test getting run
    run_info = mgr.get_run("annot", next_id)
    print(f"Retrieved: {run_info['name']}")

    print("✅ RunManager tests passed")
    return mgr


def test_data_handler(project_path: Path):
    """Test TrainingDataHandler functionality."""
    print("\n" + "=" * 60)
    print("TEST: TrainingDataHandler")
    print("=" * 60)

    from napari_sam2long.training import TrainingDataHandler

    handler = TrainingDataHandler(project_path)

    # Add ground truth run
    handler.add_run("annot", 1)

    # Get stats
    stats = handler.get_stats()
    print(f"Total samples: {stats['total_samples']}")
    print(f"Videos: {stats['num_videos']}")
    print(f"Samples per video: {stats['samples_per_video']}")

    assert stats["total_samples"] > 0, "No samples found"

    # Test sample loading
    sample = handler.samples[0]
    img = sample.load_image()
    mask = sample.load_mask()
    print(f"Sample image shape: {img.shape}")
    print(f"Sample mask shape: {mask.shape}, unique values: {len(set(mask.flatten()))}")

    # Test COCO export
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        train_json, val_json = handler.export_train_val_coco(tmpdir)

        print(f"Train JSON: {train_json}")
        print(f"Val JSON: {val_json}")

        # Verify COCO format
        with open(train_json) as f:
            coco = json.load(f)
        print(f"Train images: {len(coco['images'])}")
        print(f"Train annotations: {len(coco['annotations'])}")
        print(f"Categories: {[c['name'] for c in coco['categories']]}")

        assert len(coco["images"]) > 0, "No training images"
        assert len(coco["annotations"]) > 0, "No training annotations"

    print("✅ TrainingDataHandler tests passed")
    return handler


def test_training(project_path: Path, epochs: int = 2, model: str = "unet"):
    """Test model training."""
    print("\n" + "=" * 60)
    print(f"TEST: Model Training ({model})")
    print("=" * 60)

    from napari_sam2long.training import (
        TrainingDataHandler,
        train_model,
        TrainConfig,
        list_handlers,
    )
    from napari_sam2long.ingestion import RunManager

    # Check available handlers
    handlers = list_handlers()
    print(f"Available handlers: {handlers}")
    assert model in handlers, f"{model} handler not registered"

    # Prepare training data
    handler = TrainingDataHandler(project_path)
    handler.add_run("annot", 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        train_json, val_json = handler.export_train_val_coco(tmpdir / "coco")

        # Create training run
        run_mgr = RunManager(project_path)
        train_run = run_mgr.create_run(
            "train",
            label=f"{model}-test",
            parent="annot-001",
            metadata={"epochs": epochs, "model": model, "test": True},
        )
        checkpoint_dir = train_run / "checkpoints"

        # Train model (minimal epochs for testing)
        config = TrainConfig(
            epochs=epochs,
            batch_size=2,
            learning_rate=1e-3,
            img_size=(256, 256),  # Smaller for faster testing
        )

        def progress_callback(epoch, metrics):
            print(f"  Epoch {epoch}: loss={metrics['train_loss']:.4f}, val_iou={metrics['val_iou']:.4f}")

        print(f"\nTraining {model} for {epochs} epochs...")
        metrics = train_model(
            handler_name=model,
            coco_train=train_json,
            coco_val=val_json,
            checkpoint_dir=checkpoint_dir,
            num_classes=len(handler.class_names),
            class_names=handler.class_names,
            config=config,
            progress_callback=progress_callback,
        )

        print(f"\nFinal metrics:")
        print(f"  Best val loss: {metrics['best_val_loss']:.4f}")
        print(f"  Final val IoU: {metrics['final_val_iou']:.4f}")

        # Verify checkpoint exists
        best_checkpoint = checkpoint_dir / "best.pth"
        assert best_checkpoint.exists(), f"Checkpoint not found: {best_checkpoint}"
        print(f"  Checkpoint: {best_checkpoint}")

        # Update run manifest with metrics
        latest_train = run_mgr.get_latest("train")
        run_mgr.update_manifest("train", latest_train["id"], {
            "final_metrics": metrics,
        })

    print("✅ Training tests passed")
    return metrics


def test_inference(project_path: Path, model: str = "unet"):
    """Test model inference."""
    print("\n" + "=" * 60)
    print(f"TEST: Model Inference ({model})")
    print("=" * 60)

    import cv2
    from napari_sam2long.training import run_inference, InferConfig
    from napari_sam2long.ingestion import RunManager, Project

    run_mgr = RunManager(project_path)
    project = Project.load(project_path)

    # Get training run and checkpoint
    train_run = run_mgr.get_latest("train")
    checkpoint = Path(train_run["_path"]) / "checkpoints" / "best.pth"

    if not checkpoint.exists():
        print("⚠️ No checkpoint found, skipping inference test")
        return None

    # Create inference run
    train_id = train_run["id"]
    infer_run = run_mgr.create_run(
        "infer",
        label=f"{model}-test",
        parent=f"train-{train_id:03d}",
    )

    # Prepare image iterator
    def image_iterator():
        for vid_id in project.video_ids():
            vid_dir = project.root / "vids" / vid_id
            images_dir = vid_dir / "images"
            for img_path in sorted(images_dir.glob("*.jpg")):
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame_id = f"{vid_id}/{img_path.stem}"
                yield frame_id, img

    # Run inference
    config = InferConfig(
        batch_size=4,
        threshold=0.5,
        img_size=(256, 256),
    )

    print("Running inference...")
    masks_dir = infer_run / "masks"
    count = 0

    for frame_id, mask in run_inference(
        handler_name=model,
        checkpoint=checkpoint,
        images=image_iterator(),
        num_classes=len(project.class_names()),
        class_names=project.class_names(),
        config=config,
    ):
        # Save mask - split semantic mask by class
        vid_id, frame_name = frame_id.split("/")
        for class_idx, class_name in enumerate(project.class_names(), start=1):
            # Extract pixels for this class (class_idx = 1 for first class, 2 for second, etc.)
            class_mask = (mask == class_idx).astype(np.uint16) * class_idx
            out_dir = masks_dir / vid_id / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{frame_name}.png"
            cv2.imwrite(str(out_path), class_mask)
        count += 1

        if count % 20 == 0:
            print(f"  Processed {count} frames...")

    print(f"Inference complete: {count} frames")
    print(f"Output: {infer_run}")

    print("✅ Inference tests passed")
    return infer_run


def test_evaluation(project_path: Path):
    """Test COCO evaluation."""
    print("\n" + "=" * 60)
    print("TEST: Evaluation")
    print("=" * 60)

    from napari_sam2long.training import evaluate_masks_simple
    from napari_sam2long.ingestion import RunManager
    from napari_sam2long.ingestion.mask_io import read_indexed_mask

    run_mgr = RunManager(project_path)

    # Get ground truth and inference runs
    try:
        gt_run = run_mgr.get_run("annot", 1)
        infer_run = run_mgr.get_latest("infer")
    except FileNotFoundError:
        print("⚠️ Runs not found, skipping evaluation test")
        return None

    gt_masks_dir = Path(gt_run["_path"]) / "masks"
    pred_masks_dir = Path(infer_run["_path"]) / "masks"

    if not pred_masks_dir.exists():
        print("⚠️ No prediction masks found, skipping evaluation")
        return None

    # Collect masks for comparison
    gt_masks = {}
    pred_masks = {}

    for vid_dir in gt_masks_dir.iterdir():
        if not vid_dir.is_dir():
            continue

        for class_dir in vid_dir.iterdir():
            if not class_dir.is_dir():
                continue

            for mask_path in class_dir.glob("*.png"):
                frame_id = f"{vid_dir.name}/{class_dir.name}/{mask_path.stem}"
                gt_masks[frame_id] = read_indexed_mask(mask_path)

                # Get corresponding prediction
                pred_path = pred_masks_dir / vid_dir.name / class_dir.name / mask_path.name
                if pred_path.exists():
                    pred_masks[frame_id] = read_indexed_mask(pred_path)

    if not pred_masks:
        print("⚠️ No matching prediction masks found")
        return None

    # Evaluate
    results = evaluate_masks_simple(pred_masks, gt_masks)

    print(f"Evaluation results:")
    print(f"  Mean IoU: {results['mean_iou']:.4f}")
    print(f"  Mean Dice: {results['mean_dice']:.4f}")
    print(f"  Frames evaluated: {results['num_frames']}")

    print("✅ Evaluation tests passed")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test the multi-model active learning pipeline"
    )
    parser.add_argument(
        "--project", "-p",
        type=Path,
        required=True,
        help="Path to test project (created by download_davis_test_data.py)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2 for quick test)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (use existing checkpoint)",
    )
    parser.add_argument(
        "--model", "-m",
        default="unet",
        choices=["unet", "maskrcnn"],
        help="Model handler to use (default: unet)",
    )

    args = parser.parse_args()

    if not args.project.exists():
        print(f"Error: Project not found at {args.project}")
        print("Run: python scripts/download_davis_test_data.py --output ./test_davis_project")
        sys.exit(1)

    print("=" * 60)
    print("MULTI-MODEL ACTIVE LEARNING PIPELINE TEST")
    print("=" * 60)
    print(f"Project: {args.project}")

    # Import numpy here to check it's available
    global np
    import numpy as np

    # Run tests
    test_run_manager(args.project)
    test_data_handler(args.project)

    if not args.skip_training:
        test_training(args.project, epochs=args.epochs, model=args.model)

    test_inference(args.project, model=args.model)
    test_evaluation(args.project)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
