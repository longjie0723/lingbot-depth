"""
Depth evaluation metrics.

All functions take depth maps in meters and a boolean mask of pixels to evaluate.
"""

from typing import *

import numpy as np
import torch

DELTA_THRESHOLDS = (1.05, 1.10, 1.25)
DISTANCE_BINS = (0.0, 0.5, 1.0, 2.0, 4.0, float('inf'))


def depth_errors(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor, min_pixels: int = 100) -> Dict[str, float]:
    """
    Error of `pred` against `ref` on `mask`.

    Returns n (pixel count), abs_rel, mae, med_ae, rmse (meters) and d<t (ratio of pixels with
    max(pred/ref, ref/pred) < t). Metrics are NaN when fewer than `min_pixels` pixels are valid.
    """
    mask = mask & torch.isfinite(pred) & (pred > 0) & (ref > 0)
    n = int(mask.sum())
    keys = ['abs_rel', 'mae', 'med_ae', 'rmse'] + [f'd{t:.2f}' for t in DELTA_THRESHOLDS]
    if n < min_pixels:
        return {'n': n, **{k: float('nan') for k in keys}}

    p, r = pred[mask].double(), ref[mask].double()
    abs_err = (p - r).abs()
    ratio = torch.maximum(p / r, r / p)
    out = {
        'n': n,
        'abs_rel': float((abs_err / r).mean()),
        'mae': float(abs_err.mean()),
        'med_ae': float(abs_err.median()),
        'rmse': float(((p - r) ** 2).mean().sqrt()),
    }
    for t in DELTA_THRESHOLDS:
        out[f'd{t:.2f}'] = float((ratio < t).double().mean())
    return out


def binned_errors(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor,
                  bins: Sequence[float] = DISTANCE_BINS, min_pixels: int = 100) -> Dict[str, float]:
    """MAE and AbsRel per reference-distance bin, keyed as '<lo>-<hi>m/<metric>'."""
    out = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        name = f'{lo:g}-{hi:g}m'
        errs = depth_errors(pred, ref, mask & (ref >= lo) & (ref < hi), min_pixels=min_pixels)
        out[f'{name}/n'] = errs['n']
        out[f'{name}/mae'] = errs['mae']
        out[f'{name}/abs_rel'] = errs['abs_rel']
    return out


def make_pseudo_holes(valid: np.ndarray, ratio: float, rng: np.random.Generator,
                      min_size: float = 0.05, max_size: float = 0.25, max_rects: int = 200) -> np.ndarray:
    """
    Pick random rectangles until about `ratio` of the valid pixels are covered.

    Returns a boolean mask (subset of `valid`) of pixels to hide from the model.
    """
    h, w = valid.shape
    total = valid.sum()
    holes = np.zeros_like(valid, dtype=bool)
    if total == 0:
        return holes
    for _ in range(max_rects):
        rh = int(h * rng.uniform(min_size, max_size))
        rw = int(w * rng.uniform(min_size, max_size))
        y = rng.integers(0, h - rh + 1)
        x = rng.integers(0, w - rw + 1)
        holes[y:y + rh, x:x + rw] = True
        if (holes & valid).sum() >= ratio * total:
            break
    return holes & valid
