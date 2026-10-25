"""
Training data pipeline for fusion training at native resolution.

Samples are never resized, so the measured depth keeps the sensor's own noise and hole patterns.
Images of different resolutions are batched separately: `BucketBatchSampler` picks a resolution
bucket and a random crop size for every batch, and the dataset crops each sample of the batch to
that size at a random position.
"""

from collections import defaultdict
from typing import *

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .mdm_depth import Sample, load_sample
from ..eval.metrics import make_pseudo_holes


def bucket_samples(samples: Sequence[Sample]) -> Dict[Tuple[str, int, int], List[int]]:
    """Group sample indices by (subset, height, width). The size is read once per camera directory."""
    size_of_dir = {}
    buckets = defaultdict(list)
    for i, s in enumerate(samples):
        key = s.raw.parent
        if key not in size_of_dir:
            with Image.open(s.raw) as im:
                w, h = im.size
            size_of_dir[key] = (h, w)
        h, w = size_of_dir[key]
        buckets[(s.subset, h, w)].append(i)
    return dict(buckets)


class BucketBatchSampler(Sampler):
    """
    Infinite sampler of batches [(index, crop_h, crop_w), ...] drawn from a single resolution bucket.

    Subsets are drawn according to `subset_weights`, buckets within a subset in proportion to their
    size. The crop height and width are drawn independently as fractions of the bucket resolution,
    which also varies the aspect ratio.
    """

    def __init__(self, buckets: Dict[Tuple[str, int, int], List[int]], batch_size: int,
                 subset_weights: Dict[str, float], crop_scale: Tuple[float, float] = (0.6, 1.0), seed: int = 0):
        self.buckets = buckets
        self.batch_size = batch_size
        self.crop_scale = crop_scale
        self.rng = np.random.default_rng(seed)
        subsets = sorted({k[0] for k in buckets if subset_weights.get(k[0], 0) > 0})
        if not subsets:
            raise ValueError(f'No samples for subsets with positive weight: {subset_weights}')
        weights = np.array([subset_weights[s] for s in subsets], dtype=np.float64)
        self.subsets = subsets
        self.subset_probs = weights / weights.sum()
        self.subset_buckets = {s: [k for k in buckets if k[0] == s] for s in subsets}

    def __iter__(self):
        while True:
            subset = self.subsets[self.rng.choice(len(self.subsets), p=self.subset_probs)]
            keys = self.subset_buckets[subset]
            sizes = np.array([len(self.buckets[k]) for k in keys], dtype=np.float64)
            key = keys[self.rng.choice(len(keys), p=sizes / sizes.sum())]
            _, h, w = key
            crop_h = int(round(h * self.rng.uniform(*self.crop_scale) / 2)) * 2
            crop_w = int(round(w * self.rng.uniform(*self.crop_scale) / 2)) * 2
            indices = self.rng.choice(self.buckets[key], size=self.batch_size, replace=len(self.buckets[key]) < self.batch_size)
            yield [(int(i), crop_h, crop_w) for i in indices]


class RGBDTrainDataset(Dataset):
    """
    Returns dict(image (3, h, w) float [0, 1], raw (h, w) meters, gt (h, w) meters), 0 = invalid.

    Augmentations: random crop, horizontal flip, color jitter, and extra rectangular holes in the
    measured depth (with probability `extra_hole_prob`, hiding up to `extra_hole_max` of the valid pixels).
    """

    def __init__(self, samples: Sequence[Sample], color_jitter: float = 0.2,
                 extra_hole_prob: float = 0.5, extra_hole_max: float = 0.3):
        self.samples = list(samples)
        self.color_jitter = color_jitter
        self.extra_hole_prob = extra_hole_prob
        self.extra_hole_max = extra_hole_max

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        index, crop_h, crop_w = item
        rng = np.random.default_rng()
        data = load_sample(self.samples[index])
        image, raw, gt = data['image'], data['raw'], data['gt']
        if gt is None:
            gt = np.zeros_like(raw)

        h, w = raw.shape
        crop_h, crop_w = min(crop_h, h), min(crop_w, w)
        y = rng.integers(0, h - crop_h + 1)
        x = rng.integers(0, w - crop_w + 1)
        image = image[y:y + crop_h, x:x + crop_w]
        raw = raw[y:y + crop_h, x:x + crop_w]
        gt = gt[y:y + crop_h, x:x + crop_w]

        if rng.random() < 0.5:
            image, raw, gt = image[:, ::-1], raw[:, ::-1], gt[:, ::-1]

        image = image.astype(np.float32) / 255.0
        if self.color_jitter > 0:
            j = self.color_jitter
            gray = image.mean(axis=2, keepdims=True)
            image = gray + (image - gray) * rng.uniform(1 - j, 1 + j)            # saturation
            image = image.mean() + (image - image.mean()) * rng.uniform(1 - j, 1 + j)  # contrast
            image = image * rng.uniform(1 - j, 1 + j)                              # brightness
            image = np.clip(image, 0.0, 1.0)

        if rng.random() < self.extra_hole_prob:
            holes = make_pseudo_holes(raw > 0, rng.uniform(0.0, self.extra_hole_max), rng)
            raw = np.where(holes, 0.0, raw)

        return {
            'image': torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            'raw': torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32)),
            'gt': torch.from_numpy(np.ascontiguousarray(gt, dtype=np.float32)),
        }
