#!/usr/bin/env python3
"""
Unit tests for RunManager that don't require external data.

Usage:
    python scripts/test_run_manager_unit.py
"""

import json
import tempfile
from pathlib import Path

# Import directly from file to avoid napari dependency in __init__.py
src_path = Path(__file__).parent.parent / "src"

import importlib.util
spec = importlib.util.spec_from_file_location(
    "run_manager",
    src_path / "napari_sam2long" / "ingestion" / "run_manager.py"
)
run_manager_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_manager_module)
RunManager = run_manager_module.RunManager


def test_create_runs():
    """Test creating different run types."""
    print("Test: Create runs...")

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = RunManager(tmpdir)

        # Create annotation run
        run1 = mgr.create_run("annot", label="seed-data")
        assert run1.name == "annot_001_seed-data"
        assert (run1 / "masks").exists()
        assert (run1 / "manifest.json").exists()

        # Create training run with parent
        run2 = mgr.create_run("train", label="unet", parent="annot-001")
        assert "train_001_unet__from-annot-001" == run2.name
        assert (run2 / "checkpoints").exists()
        assert (run2 / "metrics").exists()

        # Create inference run
        run3 = mgr.create_run("infer", label="test", parent="train-001")
        assert "infer_001_test__from-train-001" == run3.name

        # Create second annotation run from inference
        run4 = mgr.create_run("annot", label="cleanup", parent="infer-001")
        assert "annot_002_cleanup__from-infer-001" == run4.name

        print("  ✅ Run creation works")


def test_list_runs():
    """Test listing runs."""
    print("Test: List runs...")

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = RunManager(tmpdir)

        # Create several runs
        mgr.create_run("annot", label="first")
        mgr.create_run("annot", label="second")
        mgr.create_run("train", label="model1", parent="annot-001")

        # List all
        all_runs = mgr.list_runs()
        assert len(all_runs) == 3

        # List by type
        annot_runs = mgr.list_runs("annot")
        assert len(annot_runs) == 2

        train_runs = mgr.list_runs("train")
        assert len(train_runs) == 1

        print("  ✅ List runs works")


def test_get_run():
    """Test getting run by ID."""
    print("Test: Get run...")

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = RunManager(tmpdir)

        mgr.create_run("annot", label="test-run")

        run = mgr.get_run("annot", 1)
        assert run["id"] == 1
        assert run["type"] == "annot"
        assert run["label"] == "test-run"

        # Test get_run_path
        path = mgr.get_run_path("annot", 1)
        assert path.exists()
        assert "annot_001_test-run" in path.name

        print("  ✅ Get run works")


def test_get_latest():
    """Test getting latest run."""
    print("Test: Get latest...")

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = RunManager(tmpdir)

        mgr.create_run("annot", label="first")
        mgr.create_run("annot", label="second")
        mgr.create_run("annot", label="third")

        latest = mgr.get_latest("annot")
        assert latest["id"] == 3
        assert latest["label"] == "third"

        print("  ✅ Get latest works")


def test_parent_validation():
    """Test parent validation."""
    print("Test: Parent validation...")

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = RunManager(tmpdir)

        # Should fail - parent doesn't exist
        try:
            mgr.create_run("train", parent="annot-001")
            assert False, "Should have raised error"
        except FileNotFoundError:
            pass

        # Create parent first
        mgr.create_run("annot", label="seed")

        # Now should work
        mgr.create_run("train", label="model", parent="annot-001")

        print("  ✅ Parent validation works")


def test_multi_parent():
    """Test multi-parent syntax."""
    print("Test: Multi-parent...")

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = RunManager(tmpdir)

        # Create multiple annotation runs
        mgr.create_run("annot", label="first")
        mgr.create_run("annot", label="second")

        # Train on both
        run = mgr.create_run("train", label="combined", parent="annot-001+002")
        assert "__from-annot-001+002" in run.name

        # Verify manifest
        manifest = json.loads((run / "manifest.json").read_text())
        assert len(manifest["parents"]) == 2
        assert manifest["parents"][0] == {"type": "annot", "id": 1}
        assert manifest["parents"][1] == {"type": "annot", "id": 2}

        print("  ✅ Multi-parent works")


def test_label_slugification():
    """Test label sanitization."""
    print("Test: Label slugification...")

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = RunManager(tmpdir)

        # Test various labels
        run1 = mgr.create_run("annot", label="Hello World!")
        assert "Hello-World" in run1.name

        run2 = mgr.create_run("annot", label="test__with--dashes")
        # Double underscores preserved, double dashes collapsed
        assert "test__with-dashes" in run2.name

        run3 = mgr.create_run("annot", label="  spaces  ")
        assert "spaces" in run3.name

        print("  ✅ Label slugification works")


def test_manifest_update():
    """Test updating manifest."""
    print("Test: Manifest update...")

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = RunManager(tmpdir)

        mgr.create_run("train", label="test")

        # Update with metrics
        mgr.update_manifest("train", 1, {
            "final_metrics": {"loss": 0.5, "iou": 0.8}
        })

        # Verify update
        run = mgr.get_run("train", 1)
        assert "final_metrics" in run
        assert run["final_metrics"]["iou"] == 0.8

        print("  ✅ Manifest update works")


def main():
    print("=" * 50)
    print("RunManager Unit Tests")
    print("=" * 50)

    test_create_runs()
    test_list_runs()
    test_get_run()
    test_get_latest()
    test_parent_validation()
    test_multi_parent()
    test_label_slugification()
    test_manifest_update()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED ✅")
    print("=" * 50)


if __name__ == "__main__":
    main()
