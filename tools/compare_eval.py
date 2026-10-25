#!/usr/bin/env python3
"""
Compare several tools/eval_depth.py result directories side by side.

Each result is given as NAME=DIR[,DIR...]. Every per_sample.csv found under the directories is merged, so
separate eval invocations (e.g. simval/ and real/) count as one result.
The measured depth itself (raw_vs_gt) is added as a reference row, taken from the first result.

Usage:
    python tools/compare_eval.py base=results/phase0/simval_t1200,results/phase0/real_t1200 \\
        p2a=results/phase2/p2a_refiner_best \\
        --out results/phase2/compare.md
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# (comparison, metric, aggregation, title, scale, format)
COLUMNS = [
    ('all', 'abs_rel', 'median', 'all AbsRel', 1, '.4f'),
    ('all', 'med_ae', 'median', 'all MedAE', 1000, '.1f'),
    ('meas', 'abs_rel', 'median', 'meas AbsRel', 1, '.4f'),
    ('meas', 'med_ae', 'median', 'meas MedAE', 1000, '.1f'),
    ('meas', 'd1.05', 'mean', 'meas d1.05', 1, '.3f'),
    ('hole', 'abs_rel', 'median', 'hole AbsRel', 1, '.4f'),
    ('hole', 'med_ae', 'median', 'hole MedAE', 1000, '.1f'),
    ('pseudo', 'abs_rel', 'median', 'pseudo AbsRel', 1, '.4f'),
    ('pseudo', 'med_ae', 'median', 'pseudo MedAE', 1000, '.1f'),
]


def load_rows(result_dirs: str):
    rows = defaultdict(list)
    for path in sorted(p for d in result_dirs.split(',') for p in Path(d).rglob('per_sample.csv')):
        with open(path) as f:
            for row in csv.DictReader(f):
                rows[row['subset']].append(row)
    return rows


def column(rows, key):
    return np.array([float(r[key]) if r.get(key) not in (None, '') else np.nan for r in rows])


def aggregate(rows, key, how):
    values = column(rows, key)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.median(values)) if how == 'median' else float(np.mean(values))


def table_row(name, rows, source):
    cells = []
    for region, metric, how, _, scale, fmt in COLUMNS:
        value = aggregate(rows, f'{source}/{region}/{metric}', how)
        cells.append('-' if np.isnan(value) else f'{value * scale:{fmt}}')
    # Share of samples whose measured-pixel MedAE is lower than that of the raw depth.
    ours, raw = column(rows, f'{source}/meas/med_ae'), column(rows, 'raw_vs_gt/meas/med_ae')
    valid = np.isfinite(ours) & np.isfinite(raw)
    wins = f'{np.mean(ours[valid] < raw[valid]):.0%}' if source != 'raw_vs_gt' and valid.any() else '-'
    times = column(rows, 'time_ms')[1:]
    time = f'{np.median(times):.0f}' if source != 'raw_vs_gt' and len(times) else '-'
    return f'| {name} | ' + ' | '.join(cells) + f' | {wins} | {time} |'


def main():
    parser = argparse.ArgumentParser(description='Compare eval_depth.py results')
    parser.add_argument('results', nargs='+', help='NAME=DIR[,DIR...]')
    parser.add_argument('--out', type=str, default=None, help='Optional markdown output path')
    args = parser.parse_args()

    results = [(spec.split('=', 1)[0], load_rows(spec.split('=', 1)[1])) for spec in args.results]
    subsets = list(dict.fromkeys(s for _, rows in results for s in rows))
    lines = ['Per-sample medians (MedAE in mm) unless noted; d1.05 is a mean over samples. '
             '"beats raw" = share of samples with a lower measured-pixel MedAE than the raw depth.', '']
    for subset in subsets:
        n = len(next(rows[subset] for _, rows in results if subset in rows))
        lines += [f'## {subset} ({n} samples)', '',
                  '| model | ' + ' | '.join(c[3] for c in COLUMNS) + ' | beats raw | ms |',
                  '|---|' + '---|' * (len(COLUMNS) + 2)]
        first = True
        for name, rows in results:
            if subset not in rows:
                continue
            if first:
                lines.append(table_row('raw depth', rows[subset], 'raw_vs_gt'))
                first = False
            lines.append(table_row(name, rows[subset], 'pred_vs_gt'))
        lines.append('')
    text = '\n'.join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
