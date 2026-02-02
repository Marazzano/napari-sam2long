"""Internal utilities for indexed mask I/O."""

import numpy as np
from pathlib import Path
from typing import Union, Dict
import cv2


def read_indexed_mask(path: Union[str, Path]) -> np.ndarray:
    """
    Read indexed mask. Always returns uint16 for consistency.

    Args:
        path: Path to mask PNG file

    Returns:
        uint16 numpy array with instance IDs as pixel values
    """
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint16)


def write_indexed_mask(mask: np.ndarray, path: Union[str, Path]) -> None:
    """
    Write indexed mask. Uses 8-bit if max_id <= 255, else 16-bit.

    Args:
        mask: Indexed mask array where pixel values are instance IDs
        path: Output path for PNG file
    """
    max_id = int(mask.max())
    if max_id <= 255:
        out = mask.astype(np.uint8)
    elif max_id <= 65535:
        out = mask.astype(np.uint16)
    else:
        raise ValueError(f"Too many instances ({max_id}), max is 65535")
    cv2.imwrite(str(path), out)


def split_indexed_mask(mask: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Split indexed mask into binary masks per instance.

    Args:
        mask: Indexed mask where pixel values are instance IDs

    Returns:
        Dict mapping instance ID to binary mask (boolean array)
    """
    result = {}
    for inst_id in np.unique(mask):
        if inst_id == 0:
            continue
        result[int(inst_id)] = (mask == inst_id)
    return result
