"""
Per-sample evaluation shared by tools/eval_depth.py and the validation loop of training.

Comparisons per sample:
    pred_vs_gt/{all,meas,hole}  prediction vs GT on all / raw-valid / raw-missing pixels
    raw_vs_gt/meas              sensor depth vs GT (the sensor's own error, reference for pred_vs_gt/meas)
    pred_vs_raw/meas            how well the prediction preserves the measurement
    */pseudo                    with pseudo_hole_ratio > 0: valid raw pixels are hidden from the model and
                                the prediction there is compared with the hidden raw depth and GT
"""

import time
from typing import *

import numpy as np
import torch

from ..data.mdm_depth import Sample, load_sample
from .metrics import depth_errors, binned_errors, make_pseudo_holes


def subsample(samples: List[Sample], limit: Optional[int]) -> List[Sample]:
    """Evenly spaced subset of at most `limit` samples."""
    if limit is None or len(samples) <= limit:
        return samples
    idx = np.linspace(0, len(samples) - 1, limit).round().astype(int)
    return [samples[i] for i in idx]


def sync(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_model(model, image: np.ndarray, raw: np.ndarray, tokens: int, amp_dtype: torch.dtype, device: torch.device):
    """Returns depth (H, W), confidence (H, W) or None, and the elapsed time in ms."""
    image_t = torch.tensor(image / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)[None]
    raw_t = torch.tensor(raw, dtype=torch.float32, device=device)[None]
    sync(device)
    t0 = time.perf_counter()
    out = model.infer(image_t, depth_in=raw_t, num_tokens=tokens, apply_mask=False,
                      use_fp16=amp_dtype != torch.float32, amp_dtype=amp_dtype)
    sync(device)
    elapsed = (time.perf_counter() - t0) * 1000
    confidence = out['confidence'][0] if out.get('confidence') is not None else None
    return out['depth'][0], confidence, elapsed


def _add(row, prefix, metrics):
    for k, v in metrics.items():
        row[f'{prefix}/{k}'] = v


def evaluate_sample(model, sample: Sample, tokens: int, amp_dtype: torch.dtype, device: torch.device,
                    pseudo_hole_ratio: float = 0.0, rng: Optional[np.random.Generator] = None):
    """Returns a dict of metrics and (image, raw, pred, gt) numpy arrays for visualization."""
    data = load_sample(sample)
    image, raw, gt = data['image'], data['raw'], data['gt']

    pred, conf, elapsed = run_model(model, image, raw, tokens, amp_dtype, device)
    raw_t = torch.tensor(raw, device=device)
    raw_valid = raw_t > 0
    row = {
        'subset': sample.subset, 'key': sample.key, 'height': image.shape[0], 'width': image.shape[1],
        'time_ms': elapsed,
        'raw_valid_ratio': float(raw_valid.float().mean()),
    }
    if conf is not None:
        row['conf_mean_meas'] = float(conf[raw_valid].mean()) if raw_valid.any() else float('nan')

    _add(row, 'pred_vs_raw/meas', depth_errors(pred, raw_t, raw_valid))
    if gt is not None:
        gt_t = torch.tensor(gt, device=device)
        gt_valid = gt_t > 0
        row['gt_valid_ratio'] = float(gt_valid.float().mean())
        _add(row, 'pred_vs_gt/all', depth_errors(pred, gt_t, gt_valid))
        _add(row, 'pred_vs_gt/meas', depth_errors(pred, gt_t, gt_valid & raw_valid))
        _add(row, 'pred_vs_gt/hole', depth_errors(pred, gt_t, gt_valid & ~raw_valid))
        _add(row, 'raw_vs_gt/meas', depth_errors(raw_t, gt_t, gt_valid & raw_valid))
        _add(row, 'pred_vs_gt/meas_bins', binned_errors(pred, gt_t, gt_valid & raw_valid))
        _add(row, 'raw_vs_gt/meas_bins', binned_errors(raw_t, gt_t, gt_valid & raw_valid))

    if pseudo_hole_ratio > 0:
        rng = rng if rng is not None else np.random.default_rng(0)
        holes = make_pseudo_holes(raw > 0, pseudo_hole_ratio, rng)
        raw_masked = np.where(holes, 0.0, raw).astype(np.float32)
        pred_ph, _, _ = run_model(model, image, raw_masked, tokens, amp_dtype, device)
        holes_t = torch.tensor(holes, device=device)
        row['pseudo_hole_ratio'] = float(holes.sum() / max((raw > 0).sum(), 1))
        _add(row, 'pred_vs_raw/pseudo', depth_errors(pred_ph, raw_t, holes_t))
        if gt is not None:
            _add(row, 'pred_vs_gt/pseudo', depth_errors(pred_ph, gt_t, holes_t & gt_valid))
            _add(row, 'raw_vs_gt/pseudo', depth_errors(raw_t, gt_t, holes_t & gt_valid))

    return row, (image, raw, pred.float().cpu().numpy(), gt)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Per subset: mean and median (key@median) over samples of every metric."""
    summary = {}
    for subset in sorted({r['subset'] for r in rows}):
        sub = [r for r in rows if r['subset'] == subset]
        keys = sorted({k for r in sub for k in r if k not in ('subset', 'key')})
        stats = {'num_samples': len(sub)}
        for k in keys:
            vals = np.array([r.get(k, np.nan) for r in sub], dtype=np.float64)
            finite = np.isfinite(vals).any()
            stats[k] = float(np.nanmean(vals)) if finite else float('nan')
            stats[f'{k}@median'] = float(np.nanmedian(vals)) if finite else float('nan')
        times = np.array([r['time_ms'] for r in sub[1:]] or [sub[0]['time_ms']])  # skip warm-up sample
        stats['time_ms_median'] = float(np.median(times))
        summary[subset] = stats
    return summary
