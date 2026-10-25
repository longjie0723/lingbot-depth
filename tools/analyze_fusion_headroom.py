#!/usr/bin/env python3
"""
Estimate how much fusing the measured depth back into the prediction could help.

For each sample the model is run once, then several per-pixel fusion rules are scored against GT:
    pred          model output as-is
    gate@t        raw depth where |raw - pred| <= t * pred, otherwise pred (no learning, no propagation)
    oracle        on measured pixels, whichever of raw / pred is closer to GT; pred in holes (upper bound)
These are reference points for Phase 2, which has to beat `gate` and approach `oracle`.

Usage:
    python tools/analyze_fusion_headroom.py --data-root ../../data/mdm_depth --limit 200 --out eval/headroom.md
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from mdm.data.mdm_depth import index_samples, load_sample
from mdm.eval.metrics import depth_errors
from mdm.model.v2 import MDMModel

GATE_THRESHOLDS = (0.01, 0.02, 0.05)


def fuse(pred, raw, gt, rule):
    raw_valid = raw > 0
    if rule == 'pred':
        return pred
    if rule.startswith('gate@'):
        t = float(rule.split('@')[1])
        use_raw = raw_valid & ((raw - pred).abs() <= t * pred)
        return torch.where(use_raw, raw, pred)
    if rule == 'oracle':
        use_raw = raw_valid & ((raw - gt).abs() < (pred - gt).abs())
        return torch.where(use_raw, raw, pred)
    raise ValueError(rule)


def main():
    parser = argparse.ArgumentParser(description='Fusion headroom analysis')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--subsets', nargs='+', default=['RobbySimVal', 'RobbyReal', 'RobbyVla'])
    parser.add_argument('--limit', type=int, default=200, help='Max samples per subset (evenly spaced)')
    parser.add_argument('--model', type=str, default='robbyant/lingbot-depth-pretrain-vitl-14-v0.5')
    parser.add_argument('--tokens', type=int, default=1200)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out', type=str, default=None, help='Optional markdown output path')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = MDMModel.from_pretrained(args.model).to(device).eval()
    rules = ['pred'] + [f'gate@{t}' for t in GATE_THRESHOLDS] + ['oracle']
    regions = ['all', 'meas', 'hole']

    lines = [f'# Fusion headroom (tokens {args.tokens}, fp16)', '']
    for subset in args.subsets:
        samples = index_samples(args.data_root, [subset])
        samples = [s for s in samples if s.gt is not None]
        if len(samples) > args.limit:
            samples = [samples[i] for i in np.linspace(0, len(samples) - 1, args.limit).round().astype(int)]
        if not samples:
            continue

        scores = {(r, g): [] for r in rules for g in regions}
        for sample in samples:
            data = load_sample(sample)
            image = torch.tensor(data['image'] / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)[None]
            raw = torch.tensor(data['raw'], device=device)
            gt = torch.tensor(data['gt'], device=device)
            pred = model.infer(image, depth_in=raw[None], num_tokens=args.tokens, apply_mask=False,
                               amp_dtype=torch.float16)['depth'][0].float()
            masks = {'all': gt > 0, 'meas': (gt > 0) & (raw > 0), 'hole': (gt > 0) & (raw <= 0)}
            for rule in rules:
                fused = fuse(pred, raw, gt, rule)
                for region in regions:
                    scores[(rule, region)].append(depth_errors(fused, gt, masks[region]))

        lines += [f'## {subset} ({len(samples)} samples)', '',
                  '| rule | all: AbsRel med | all: MedAE med | all: d1.05 | meas: AbsRel med | meas: MedAE med | meas: d1.05 |',
                  '|---|---|---|---|---|---|---|']
        for rule in rules:
            cells = []
            for region in ('all', 'meas'):
                errs = scores[(rule, region)]
                agg = lambda k, f: f(np.array([e[k] for e in errs], dtype=np.float64))
                cells += [f'{agg("abs_rel", np.nanmedian):.4f}',
                          f'{agg("med_ae", np.nanmedian) * 1000:.1f} mm',
                          f'{agg("d1.05", np.nanmean):.3f}']
            lines.append(f'| {rule} | ' + ' | '.join(cells) + ' |')
        lines.append('')
        print('\n'.join(lines[-len(rules) - 5:]), flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
