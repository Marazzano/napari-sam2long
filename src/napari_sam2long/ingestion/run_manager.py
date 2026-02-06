"""
Run directory management for tracking annotation, training, and inference runs.

Naming Convention: [type]_[id]_[label]__from-[parent-ref]
- type: annot, train, or infer
- id: Fixed-width 3-digit counter per type
- label: Human-readable slug (alphanumeric, underscore, hyphen)
- parent-ref: Lineage reference (e.g., __from-annot-001 or __from-infer-001+002)

Example:
    runs/
    ├── annot_001_seed/
    ├── train_001_unet__from-annot-001/
    ├── infer_001_eval__from-train-001/
    └── annot_002_fix-tail__from-infer-001/
"""

import json
import re
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


class RunManager:
    """
    Manages annotation/training/inference runs for a project.

    Provides:
    - Run creation with auto-incrementing IDs
    - Parent validation and lineage tracking
    - Manifest-based source of truth
    - Mask copying between runs and working directory
    """

    TYPES = ("annot", "train", "infer")

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.runs_dir = self.project_root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Internal Helpers
    # ---------------------------------------------------------

    def _slugify_label(self, label: str) -> Optional[str]:
        """Sanitize label: allow only alnum, underscore, hyphen."""
        if not label:
            return None
        # Replace unsafe chars and whitespace with '-'
        label = re.sub(r"[^A-Za-z0-9_-]", "-", label.strip())
        # Collapse multiple '-'
        return re.sub(r"-{2,}", "-", label).strip("-") or None

    def _get_existing_runs(self, run_type: str) -> list[dict]:
        """Returns sorted list of runs for a specific type."""
        # Match: type_NNN followed by underscore, double-underscore, or end of string
        pattern = re.compile(rf"^{re.escape(run_type)}_(\d{{3}})(?:_|__|$)")
        runs = []
        for p in self.runs_dir.iterdir():
            if p.is_dir():
                m = pattern.match(p.name)
                if m:
                    runs.append({
                        "type": run_type,
                        "id": int(m.group(1)),
                        "name": p.name,
                        "path": p,
                    })
        return sorted(runs, key=lambda x: x["id"])

    def _next_id(self, run_type: str) -> int:
        """Get next available ID for a run type."""
        existing = self._get_existing_runs(run_type)
        return 1 if not existing else existing[-1]["id"] + 1

    def _parse_parent_spec(self, parent: str) -> list[dict]:
        """
        Parse parent specification into structured dicts.

        Args:
            parent: String like 'annot-001' or 'infer-001+002'

        Returns:
            List of {"type": str, "id": int} dicts
        """
        if not parent:
            return []

        try:
            p_type, p_ids = parent.strip().split("-", 1)
        except ValueError:
            raise ValueError(f"Invalid parent '{parent}'. Use 'type-###' or 'type-###+###'")

        if p_type not in self.TYPES:
            raise ValueError(f"Invalid parent type '{p_type}'. Must be one of {self.TYPES}")

        parents = []
        seen = set()
        for token in p_ids.split("+"):
            token = token.strip()
            if not re.fullmatch(r"\d{1,3}", token):
                raise ValueError(f"Invalid ID '{token}'. Use digits like '001' or '1'.")
            pid = int(token)
            if pid not in seen:
                parents.append({"type": p_type, "id": pid})
                seen.add(pid)
        return parents

    def _validate_parents_exist(self, parents: list[dict]):
        """Ensure referenced parents actually exist on disk."""
        for pr in parents:
            existing = self._get_existing_runs(pr["type"])
            if not any(r["id"] == pr["id"] for r in existing):
                raise FileNotFoundError(f"Parent not found: {pr['type']}_{pr['id']:03d}")

    def _lineage_suffix(self, parents: list[dict]) -> str:
        """Generate the folder suffix: __from-infer-001+002"""
        if not parents:
            return ""
        types = {p["type"] for p in parents}
        if len(types) != 1:
            raise ValueError(f"Mixed parent types not supported in naming: {sorted(types)}")
        p_type = parents[0]["type"]
        ids = "+".join(f"{p['id']:03d}" for p in parents)
        return f"__from-{p_type}-{ids}"

    # ---------------------------------------------------------
    # Public API: Create
    # ---------------------------------------------------------

    def create_run(
        self,
        run_type: str,
        label: str = None,
        parent: str = None,
        metadata: dict = None,
    ) -> Path:
        """
        Create a new run directory with lineage tracking.

        Args:
            run_type: One of 'annot', 'train', 'infer'
            label: Human-readable label (e.g., 'unet-baseline', 'seed-data')
            parent: Parent reference (e.g., 'annot-001' or 'infer-001+002')
            metadata: Additional metadata to store in manifest

        Returns:
            Path to the created run directory

        Example:
            mgr.create_run("train", label="unet", parent="annot-001")
            # Creates: runs/train_001_unet__from-annot-001/
        """
        if run_type not in self.TYPES:
            raise ValueError(f"Invalid type. Must be one of {self.TYPES}")

        # 1. Prepare data
        label = self._slugify_label(label)
        parents = self._parse_parent_spec(parent)
        self._validate_parents_exist(parents)
        next_id = self._next_id(run_type)

        # 2. Build folder name
        folder_name = f"{run_type}_{next_id:03d}"
        if label:
            folder_name += f"_{label}"
        folder_name += self._lineage_suffix(parents)

        # 3. Create directory
        run_path = self.runs_dir / folder_name
        run_path.mkdir(exist_ok=False)

        # 4. Create subdirectories based on type
        if run_type == "train":
            (run_path / "checkpoints").mkdir()
            (run_path / "metrics").mkdir()
        else:
            (run_path / "masks").mkdir()

        # 5. Create stable manifest
        manifest = {
            "schema_version": 1,
            "id": next_id,
            "type": run_type,
            "label": label,
            "name": folder_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parents": parents,  # Stable IDs: [{"type": "annot", "id": 1}]
            "metadata": metadata or {},
        }
        (run_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

        return run_path

    # ---------------------------------------------------------
    # Public API: Read
    # ---------------------------------------------------------

    def get_run(self, run_type: str, run_id: int) -> dict:
        """
        Load manifest for a specific run by stable ID.

        Args:
            run_type: One of 'annot', 'train', 'infer'
            run_id: The run's numeric ID

        Returns:
            Manifest dict with run metadata
        """
        for run in self._get_existing_runs(run_type):
            if run["id"] == run_id:
                manifest_path = run["path"] / "manifest.json"
                if manifest_path.exists():
                    data = json.loads(manifest_path.read_text())
                    data["_path"] = str(run["path"])
                    return data
                else:
                    # Return basic info if manifest missing
                    return {
                        "id": run_id,
                        "type": run_type,
                        "name": run["name"],
                        "_path": str(run["path"]),
                    }
        raise FileNotFoundError(f"Run not found: {run_type}_{run_id:03d}")

    def get_run_path(self, run_type: str, run_id: int) -> Path:
        """
        Get directory path for a run by stable ID.

        Args:
            run_type: One of 'annot', 'train', 'infer'
            run_id: The run's numeric ID

        Returns:
            Path to the run directory
        """
        for run in self._get_existing_runs(run_type):
            if run["id"] == run_id:
                return run["path"]
        raise FileNotFoundError(f"Run not found: {run_type}_{run_id:03d}")

    def list_runs(self, run_type: str = None) -> list[dict]:
        """
        List all runs with their manifests.

        Args:
            run_type: Optional filter by type ('annot', 'train', 'infer')

        Returns:
            List of manifest dicts sorted by creation time
        """
        types = [run_type] if run_type else list(self.TYPES)
        results = []

        for t in types:
            for run in self._get_existing_runs(t):
                manifest_path = run["path"] / "manifest.json"
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    manifest["_path"] = str(run["path"])
                    results.append(manifest)
                else:
                    # Basic info if manifest missing
                    results.append({
                        "id": run["id"],
                        "type": t,
                        "name": run["name"],
                        "_path": str(run["path"]),
                    })

        return sorted(results, key=lambda x: x.get("created_at", ""))

    def get_latest(self, run_type: str) -> dict:
        """
        Get most recent run of a type.

        Args:
            run_type: One of 'annot', 'train', 'infer'

        Returns:
            Manifest dict for the latest run
        """
        runs = self._get_existing_runs(run_type)
        if not runs:
            raise FileNotFoundError(f"No {run_type} runs exist")
        return self.get_run(run_type, runs[-1]["id"])

    # ---------------------------------------------------------
    # Public API: Mask Operations
    # ---------------------------------------------------------

    def save_working_to_run(
        self,
        run_type: str,
        run_id: int,
        video_ids: list[str] = None,
    ):
        """
        Copy current working masks (vids/*/masks) to a run.

        Args:
            run_type: Target run type
            run_id: Target run ID
            video_ids: Optional list of video IDs to save (default: all)
        """
        run_path = self.get_run_path(run_type, run_id)
        run_masks = run_path / "masks"
        run_masks.mkdir(parents=True, exist_ok=True)

        vids_dir = self.project_root / "vids"
        if not vids_dir.exists():
            return

        for vid_dir in vids_dir.iterdir():
            if not vid_dir.is_dir():
                continue
            if video_ids and vid_dir.name not in video_ids:
                continue

            src_masks = vid_dir / "masks"
            if src_masks.exists():
                dst = run_masks / vid_dir.name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src_masks, dst)

        # Update manifest with video list
        manifest_path = run_path / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if video_ids:
                manifest["videos_included"] = video_ids
            else:
                manifest["videos_included"] = [
                    d.name for d in vids_dir.iterdir() if d.is_dir()
                ]
            manifest_path.write_text(json.dumps(manifest, indent=2))

    def load_run_to_working(
        self,
        run_type: str,
        run_id: int,
        video_ids: list[str] = None,
    ):
        """
        Load masks from a run into working directory (vids/*/masks).

        Args:
            run_type: Source run type
            run_id: Source run ID
            video_ids: Optional list of video IDs to load (default: all in run)
        """
        run_path = self.get_run_path(run_type, run_id)
        run_masks = run_path / "masks"

        if not run_masks.exists():
            raise ValueError(f"No masks found in run {run_type}_{run_id:03d}")

        for vid_dir in run_masks.iterdir():
            if not vid_dir.is_dir():
                continue
            if video_ids and vid_dir.name not in video_ids:
                continue

            dst = self.project_root / "vids" / vid_dir.name / "masks"
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copytree(vid_dir, dst, dirs_exist_ok=True)

    def get_masks_dir(self, run_type: str, run_id: int, video_id: str = None) -> Path:
        """
        Get masks directory for a run, optionally for a specific video.

        Args:
            run_type: Run type
            run_id: Run ID
            video_id: Optional video ID for specific video's masks

        Returns:
            Path to masks directory
        """
        run_path = self.get_run_path(run_type, run_id)
        masks_dir = run_path / "masks"
        if video_id:
            return masks_dir / video_id
        return masks_dir

    def get_checkpoints_dir(self, run_id: int) -> Path:
        """
        Get checkpoints directory for a training run.

        Args:
            run_id: Training run ID

        Returns:
            Path to checkpoints directory
        """
        run_path = self.get_run_path("train", run_id)
        return run_path / "checkpoints"

    # ---------------------------------------------------------
    # Public API: Manifest Updates
    # ---------------------------------------------------------

    def update_manifest(self, run_type: str, run_id: int, updates: dict):
        """
        Update fields in a run's manifest.

        Args:
            run_type: Run type
            run_id: Run ID
            updates: Dict of fields to update/add
        """
        run_path = self.get_run_path(run_type, run_id)
        manifest_path = run_path / "manifest.json"

        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())

        manifest.update(updates)
        manifest_path.write_text(json.dumps(manifest, indent=2))


# Example usage
if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = RunManager(tmpdir)

        # 1. Create annotation run
        p1 = mgr.create_run("annot", label="seed data")
        print(f"Created: {p1.name}")

        # 2. Train on annot-001
        p2 = mgr.create_run("train", label="unet-baseline", parent="annot-001")
        print(f"Created: {p2.name}")

        # 3. Inference from train-001
        p3 = mgr.create_run("infer", label="full-project", parent="train-001")
        print(f"Created: {p3.name}")

        # 4. Refine annotations from inference
        p4 = mgr.create_run("annot", label="cleanup", parent="infer-001")
        print(f"Created: {p4.name}")

        # List all runs
        print("\nAll runs:")
        for run in mgr.list_runs():
            print(f"  {run['name']}")
