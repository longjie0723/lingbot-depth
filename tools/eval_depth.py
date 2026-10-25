#!/usr/bin/env python3
"""
Evaluate LingBot-Depth against ground truth and against the measured (raw) depth.

The comparisons are described in mdm/eval/runner.py.

Usage:
    python tools/eval_depth.py --data-root ../../data/mdm_depth --subsets RobbySimVal --limit 200 --out eval/base_t1200
    python tools/eval_depth.py --examples examples --out eval/examples --pseudo-hole-ratio 0.15
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from mdm.data.mdm_depth import Sample, index_samples
from mdm.eval.runner import evaluate_sample, subsample, summarize
from mdm.model.v2 import MDMModel

AMP_DTYPES = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
# (metric, aggregation over samples, column title)
SUMMARY_COLUMNS = [('abs_rel', 'mean', 'AbsRel mean'), ('abs_rel', 'median', 'AbsRel median'),
                   ('med_ae', 'median', 'MedAE median'), ('d1.05', 'mean', 'd1.05 mean')]


def index_examples(examples_dir: Path):
    samples = []
    for d in sorted(p for p in examples_dir.iterdir() if p.is_dir()):
        rgb = next((d / f'rgb{ext}' for ext in ('.png', '.jpg', '.jpeg') if (d / f'rgb{ext}').exists()), None)
        if rgb is None or not (d / 'raw_depth.png').exists():
            continue
        intr = d / 'intrinsics.txt'
        samples.append(Sample(subset='examples', key=d.name, rgb=rgb, raw=d / 'raw_depth.png', gt=None,
                              intrinsics=intr if intr.exists() else None))
    return samples




def colorize(depth, vmin, vmax):
    valid = np.isfinite(depth) & (depth > 0)
    norm = np.clip((np.where(valid, depth, 0) - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def save_vis(path, image, raw, pred, gt):
    ref = gt if gt is not None else raw
    valid = ref[ref > 0]
    vmin, vmax = (np.percentile(valid, 1), np.percentile(valid, 99)) if valid.size else (0, 1)
    tiles = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR), colorize(raw, vmin, vmax), colorize(pred, vmin, vmax)]
    if gt is not None:
        tiles.append(colorize(gt, vmin, vmax))
        err = np.where(gt > 0, np.abs(pred - gt) / np.maximum(gt, 1e-6), 0)
        err_vis = cv2.applyColorMap(np.clip(err / 0.1 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        err_vis[gt <= 0] = 0
        tiles.append(err_vis)  # relative error, saturates at 10%
    cv2.imwrite(str(path), np.concatenate(tiles, axis=1))












def format_summary(summary):
    comparisons = ['pred_vs_gt/all', 'pred_vs_gt/meas', 'raw_vs_gt/meas', 'pred_vs_gt/hole',
                   'pred_vs_raw/meas', 'pred_vs_raw/pseudo', 'pred_vs_gt/pseudo', 'raw_vs_gt/pseudo']
    lines = []
    for subset, stats in summary.items():
        lines.append(f'## {subset} ({stats["num_samples"]} samples, median {stats["time_ms_median"]:.0f} ms/frame)')
        lines.append('')
        lines.append('| comparison | ' + ' | '.join(title for _, _, title in SUMMARY_COLUMNS) + ' |')
        lines.append('|---|' + '---|' * len(SUMMARY_COLUMNS))
        for c in comparisons:
            if f'{c}/abs_rel' not in stats:
                continue
            vals = []
            for m, agg, _ in SUMMARY_COLUMNS:
                v = stats.get(f'{c}/{m}' + ('@median' if agg == 'median' else ''), float('nan'))
                vals.append(f'{v * 1000:.1f} mm' if m in ('mae', 'med_ae') else f'{v:.4f}')
            lines.append(f'| {c} | ' + ' | '.join(vals) + ' |')
        bins = sorted({k.split('/')[2] for k in stats if k.startswith('pred_vs_gt/meas_bins/') and k.endswith('/mae')},
                      key=lambda s: float(s.split('-')[0]))
        if bins:
            lines.append('')
            lines.append('| MAE by GT distance (measured pixels) | ' + ' | '.join(bins) + ' |')
            lines.append('|---|' + '---|' * len(bins))
            for c in ('pred_vs_gt/meas_bins', 'raw_vs_gt/meas_bins'):
                vals = [stats.get(f'{c}/{b}/mae', float('nan')) * 1000 for b in bins]
                lines.append(f'| {c.split("/")[0]} | ' + ' | '.join(f'{v:.1f} mm' for v in vals) + ' |')
        lines.append('')
    return '\n'.join(lines)


def write_summary(out_dir, rows, config):
    summary = summarize(rows)
    summary['_config'] = config
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    text = format_summary({k: v for k, v in summary.items() if not k.startswith('_')})
    (out_dir / 'summary.md').write_text(text)
    print()
    print(text)


def main():
    parser = argparse.ArgumentParser(description='Evaluate LingBot-Depth on mdm_depth subsets or repo examples')
    parser.add_argument('--data-root', type=str, default=None, help='Root of the extracted mdm_depth subset')
    parser.add_argument('--subsets', nargs='+', default=['RobbySimVal', 'RobbyReal', 'RobbyVla'])
    parser.add_argument('--examples', type=str, default=None, help='Evaluate the repo examples directory instead')
    parser.add_argument('--limit', type=int, default=None, help='Max samples per subset (evenly spaced)')
    parser.add_argument('--model', type=str, default='robbyant/lingbot-depth-pretrain-vitl-14-v0.5')
    parser.add_argument('--tokens', type=int, default=1200, help='Number of ViT tokens (1200 = resolution_level 0)')
    parser.add_argument('--amp-dtype', type=str, default='fp16', choices=list(AMP_DTYPES))
    parser.add_argument('--pseudo-hole-ratio', type=float, default=0.0, help='Fraction of valid raw pixels to hide')
    parser.add_argument('--save-vis', type=int, default=0, help='Save visualizations for the first N samples per subset')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--resummarize', action='store_true', help='Only rebuild the summary from <out>/per_sample.csv')
    args = parser.parse_args()
    out_dir = Path(args.out)

    if args.resummarize:
        rows = []
        with open(out_dir / 'per_sample.csv', newline='') as f:
            for r in csv.DictReader(f):
                rows.append({k: (v if k in ('subset', 'key') else float(v) if v != '' else float('nan'))
                             for k, v in r.items()})
        old = out_dir / 'summary.json'
        config = json.loads(old.read_text()).get('_config', {}) if old.exists() else {}
        write_summary(out_dir, rows, config)
        return 0

    if args.examples:
        samples = index_examples(Path(args.examples))
    elif args.data_root:
        samples = []
        for subset in args.subsets:
            samples += subsample(index_samples(args.data_root, [subset]), args.limit)
    else:
        parser.error('either --data-root or --examples is required')
    if not samples:
        parser.error('no samples found')

    device = torch.device(args.device)
    amp_dtype = AMP_DTYPES[args.amp_dtype]
    (out_dir / 'vis').mkdir(parents=True, exist_ok=True)

    print(f'{len(samples)} samples | model {args.model} | tokens {args.tokens} | amp {args.amp_dtype}')
    model = MDMModel.from_pretrained(args.model).to(device).eval()
    rng = np.random.default_rng(args.seed)

    rows = []
    vis_count = {}
    for i, sample in enumerate(samples):
        row, vis = evaluate_sample(model, sample, args.tokens, amp_dtype, device, args.pseudo_hole_ratio, rng)
        rows.append(row)
        if vis_count.get(sample.subset, 0) < args.save_vis:
            vis_count[sample.subset] = vis_count.get(sample.subset, 0) + 1
            save_vis(out_dir / 'vis' / f'{sample.subset}_{i:05d}.png', *vis)
        if (i + 1) % 50 == 0 or i + 1 == len(samples):
            print(f'  [{i + 1}/{len(samples)}] {sample.subset} {row["time_ms"]:.0f} ms', flush=True)

    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out_dir / 'per_sample.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    write_summary(out_dir, rows, vars(args))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
