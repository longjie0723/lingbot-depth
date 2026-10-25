"""
Sample indexing and loading for the LingBot-Depth dataset (robbyant/mdm_depth).

Supported layouts (as extracted by tools/stream_subset.py):
    RobbySimVal/val_view/<scene>/<cam>/NNNN_camC_{rgb.left.jpg, depth_left.png, rawdepth.left.png}, intrinsics.txt
    RobbySim/<view>/<scene>/<cam>/NNNN_camC_{left.jpg, depth.png, rmd2c.png}, RobbySim/intrinsics.txt
    RobbyReal/<scene>/<seq>/<camera>/{color,gtdepth,rawdepth}/<frame>.*, intrinsic.txt
    RobbyVla/<robot>/<seq>/<camera>/{color,gtdepth,rawdepth}/<frame>.*, intrinsic.txt
Depth PNGs are 16-bit millimeters; 0 and 65535 are treated as invalid.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import *

import cv2
import numpy as np

SUBSETS = ('RobbySimVal', 'RobbySim', 'RobbyReal', 'RobbyVla')
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
INVALID_DEPTH_MM = 65535


@dataclass
class Sample:
    subset: str
    key: str
    rgb: Path
    raw: Path
    gt: Optional[Path]
    intrinsics: Optional[Path]


def _index_sim(root: Path, subset: str, rgb_suffix: str, gt_suffix: str, raw_suffix: str,
               global_intrinsics: Optional[Path]) -> List[Sample]:
    samples = []
    for rgb in sorted((root / subset).rglob(f'*{rgb_suffix}')):
        stem = rgb.name[:-len(rgb_suffix)]
        raw = rgb.with_name(stem + raw_suffix)
        gt = rgb.with_name(stem + gt_suffix)
        if not raw.exists():
            continue
        intr = rgb.parent / 'intrinsics.txt'
        samples.append(Sample(
            subset=subset,
            key=str(rgb.relative_to(root).with_name(stem)),
            rgb=rgb,
            raw=raw,
            gt=gt if gt.exists() else None,
            intrinsics=intr if intr.exists() else global_intrinsics,
        ))
    return samples


def _index_real(root: Path, subset: str) -> List[Sample]:
    samples = []
    for raw_dir in sorted((root / subset).rglob('rawdepth')):
        cam_dir = raw_dir.parent
        intr = cam_dir / 'intrinsic.txt'
        colors = {p.stem: p for p in (cam_dir / 'color').glob('*') if p.suffix.lower() in IMAGE_EXTS}
        for raw in sorted(raw_dir.glob('*.png')):
            rgb = colors.get(raw.stem)
            if rgb is None:
                continue
            gt = cam_dir / 'gtdepth' / raw.name
            samples.append(Sample(
                subset=subset,
                key=str(raw.relative_to(root).parent.parent / raw.stem),
                rgb=rgb,
                raw=raw,
                gt=gt if gt.exists() else None,
                intrinsics=intr if intr.exists() else None,
            ))
    return samples


def index_samples(root: Union[str, Path], subsets: Sequence[str] = SUBSETS) -> List[Sample]:
    root = Path(root)
    samples = []
    for subset in subsets:
        if not (root / subset).exists():
            continue
        if subset == 'RobbySimVal':
            samples += _index_sim(root, subset, '_rgb.left.jpg', '_depth_left.png', '_rawdepth.left.png', None)
        elif subset == 'RobbySim':
            intr = root / subset / 'intrinsics.txt'
            samples += _index_sim(root, subset, '_left.jpg', '_depth.png', '_rmd2c.png', intr if intr.exists() else None)
        elif subset in ('RobbyReal', 'RobbyVla'):
            samples += _index_real(root, subset)
        else:
            raise ValueError(f'Unknown subset: {subset}')
    return samples


def load_depth_mm(path: Path) -> np.ndarray:
    """Load a 16-bit millimeter depth PNG as float32 meters, invalid pixels set to 0."""
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f'Failed to read depth: {path}')
    invalid = depth == INVALID_DEPTH_MM
    depth = depth.astype(np.float32) / 1000.0
    depth[invalid] = 0.0
    return depth


def load_intrinsics(path: Path) -> np.ndarray:
    """Load a pixel-unit 3x3 intrinsics matrix (9 values, or fx fy cx cy)."""
    values = np.loadtxt(path, dtype=np.float32).reshape(-1)
    if values.size == 9:
        return values.reshape(3, 3)
    if values.size == 4:
        fx, fy, cx, cy = values
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    raise ValueError(f'Unsupported intrinsics format ({values.size} values): {path}')


def load_sample(sample: Sample) -> Dict[str, Any]:
    """
    Returns:
        image: (H, W, 3) uint8 RGB
        raw: (H, W) float32 meters, 0 = invalid
        gt: (H, W) float32 meters, 0 = invalid, or None
        intrinsics: (3, 3) float32 pixel units, or None
    """
    image = cv2.imread(str(sample.rgb))
    if image is None:
        raise ValueError(f'Failed to read image: {sample.rgb}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    raw = load_depth_mm(sample.raw)
    gt = load_depth_mm(sample.gt) if sample.gt is not None else None
    for name, d in (('raw', raw), ('gt', gt)):
        if d is not None and d.shape != (h, w):
            raise ValueError(f'{name} depth shape {d.shape} != image shape {(h, w)}: {sample.key}')

    intrinsics = load_intrinsics(sample.intrinsics) if sample.intrinsics is not None else None
    return {'image': image, 'raw': raw, 'gt': gt, 'intrinsics': intrinsics}
