"""
COCO evaluation metrics using pycocotools.

Provides standard COCO metrics (AP, AR, IoU) for comparing
predictions against ground truth annotations.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np


def evaluate_coco(
    pred_json: Path,
    gt_json: Path,
    iou_type: str = "segm",
) -> dict:
    """
    Evaluate predictions against ground truth using COCO metrics.

    Args:
        pred_json: Path to predictions COCO JSON (from inference)
        gt_json: Path to ground truth COCO JSON (from annotations)
        iou_type: Type of IoU to compute ('segm' for masks, 'bbox' for boxes)

    Returns:
        Dict with metrics:
            - AP: Average Precision @ IoU=0.50:0.95
            - AP50: AP @ IoU=0.50
            - AP75: AP @ IoU=0.75
            - AR: Average Recall @ IoU=0.50:0.95
            - per_category: Dict of per-category metrics
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # Load ground truth and predictions
    coco_gt = COCO(str(gt_json))
    coco_pred = coco_gt.loadRes(str(pred_json))

    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_pred, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract metrics
    # stats indices: [AP, AP50, AP75, AP_small, AP_medium, AP_large, AR1, AR10, AR100, ...]
    stats = coco_eval.stats

    results = {
        "AP": round(stats[0], 4),      # AP @ IoU=0.50:0.95, all areas
        "AP50": round(stats[1], 4),    # AP @ IoU=0.50
        "AP75": round(stats[2], 4),    # AP @ IoU=0.75
        "AP_small": round(stats[3], 4),
        "AP_medium": round(stats[4], 4),
        "AP_large": round(stats[5], 4),
        "AR1": round(stats[6], 4),     # AR @ 1 detection per image
        "AR10": round(stats[7], 4),    # AR @ 10 detections per image
        "AR100": round(stats[8], 4),   # AR @ 100 detections per image
        "AR_small": round(stats[9], 4),
        "AR_medium": round(stats[10], 4),
        "AR_large": round(stats[11], 4),
    }

    # Per-category evaluation
    if hasattr(coco_eval, "eval") and coco_eval.eval:
        per_cat = {}
        cat_ids = coco_gt.getCatIds()
        cat_names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}

        precision = coco_eval.eval["precision"]  # [T, R, K, A, M]
        # T: IoU thresholds, R: recall thresholds, K: categories, A: areas, M: max dets

        for k, cat_id in enumerate(cat_ids):
            # Mean precision across IoU thresholds, all areas, 100 max dets
            cat_precision = precision[:, :, k, 0, 2]  # [T, R]
            cat_precision = cat_precision[cat_precision > -1]
            if len(cat_precision) > 0:
                ap = np.mean(cat_precision)
                per_cat[cat_names.get(cat_id, str(cat_id))] = round(float(ap), 4)

        results["per_category"] = per_cat

    return results


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute IoU between two binary masks.

    Args:
        pred_mask: Predicted binary mask (H×W)
        gt_mask: Ground truth binary mask (H×W)

    Returns:
        IoU score (0 to 1)
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        return 0.0

    return float(intersection) / float(union)


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Dice coefficient between two binary masks.

    Args:
        pred_mask: Predicted binary mask (H×W)
        gt_mask: Ground truth binary mask (H×W)

    Returns:
        Dice score (0 to 1)
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    sum_masks = pred_mask.sum() + gt_mask.sum()

    if sum_masks == 0:
        return 0.0

    return float(2 * intersection) / float(sum_masks)


def evaluate_masks_simple(
    pred_masks: dict[str, np.ndarray],
    gt_masks: dict[str, np.ndarray],
) -> dict:
    """
    Simple mask evaluation without COCO format.

    Computes IoU and Dice for each frame, then averages.

    Args:
        pred_masks: Dict mapping frame_id to predicted mask (indexed)
        gt_masks: Dict mapping frame_id to ground truth mask (indexed)

    Returns:
        Dict with mean_iou, mean_dice, per_frame metrics
    """
    ious = []
    dices = []
    per_frame = {}

    common_frames = set(pred_masks.keys()) & set(gt_masks.keys())

    for frame_id in sorted(common_frames):
        pred = pred_masks[frame_id]
        gt = gt_masks[frame_id]

        # Convert indexed masks to binary (foreground vs background)
        pred_binary = pred > 0
        gt_binary = gt > 0

        iou = compute_iou(pred_binary, gt_binary)
        dice = compute_dice(pred_binary, gt_binary)

        ious.append(iou)
        dices.append(dice)
        per_frame[frame_id] = {"iou": round(iou, 4), "dice": round(dice, 4)}

    return {
        "mean_iou": round(np.mean(ious), 4) if ious else 0.0,
        "mean_dice": round(np.mean(dices), 4) if dices else 0.0,
        "num_frames": len(common_frames),
        "per_frame": per_frame,
    }


def create_coco_results(
    coco_gt_json: Path,
    predictions: dict[int, list[dict]],
    output_path: Optional[Path] = None,
) -> list[dict]:
    """
    Create COCO results format from predictions.

    Args:
        coco_gt_json: Path to ground truth COCO JSON (for image IDs)
        predictions: Dict mapping image_id to list of prediction dicts
                    Each prediction: {
                        "category_id": int,
                        "segmentation": RLE dict,
                        "score": float,
                        "bbox": [x, y, w, h] (optional)
                    }
        output_path: Optional path to save results JSON

    Returns:
        List of COCO result dicts
    """
    results = []
    ann_id = 0

    for image_id, preds in predictions.items():
        for pred in preds:
            ann_id += 1
            result = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": pred["category_id"],
                "segmentation": pred["segmentation"],
                "score": pred.get("score", 1.0),
            }
            if "bbox" in pred:
                result["bbox"] = pred["bbox"]
            results.append(result)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f)

    return results
