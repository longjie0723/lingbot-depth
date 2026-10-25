#!/usr/bin/env python3
"""
Measure steady-state inference latency of LingBot-Depth with a per-stage breakdown.

Stages are timed with CUDA events from forward hooks on the encoder, the neck and the heads.
Each configuration is warmed up first, and the output depth is compared with an fp32 run to
check that reduced precision does not change the result.

Usage:
    python tools/bench_latency.py --tokens 1200 2000 3600 --amp-dtype fp16 bf16 fp32
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch

from mdm.model.v2 import MDMModel

AMP_DTYPES = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}


class StageTimer:
    def __init__(self, modules):
        self.enabled = False
        self.starts = {}
        self.events = {name: [] for name in modules}
        for name, module in modules.items():
            module.register_forward_pre_hook(partial(self._pre, name))
            module.register_forward_hook(partial(self._post, name))

    def _pre(self, name, module, inputs):
        if self.enabled:
            self.starts[name] = torch.cuda.Event(enable_timing=True)
            self.starts[name].record()

    def _post(self, name, module, inputs, output):
        if self.enabled:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.events[name].append((self.starts.pop(name), end))

    def reset(self):
        self.events = {name: [] for name in self.events}

    def medians(self):
        return {name: float(np.median([s.elapsed_time(e) for s, e in evs])) for name, evs in self.events.items() if evs}


def load_input(example_dir: Path, size, device):
    rgb = next(example_dir / f'rgb{ext}' for ext in ('.png', '.jpg', '.jpeg') if (example_dir / f'rgb{ext}').exists())
    image = cv2.cvtColor(cv2.imread(str(rgb)), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(example_dir / 'raw_depth.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    if size is not None:
        h, w = size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    image_t = torch.tensor(image / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)[None]
    depth_t = torch.tensor(depth, dtype=torch.float32, device=device)[None]
    return image_t, depth_t


def run(model, image, depth, tokens, amp_dtype):
    return model.infer(image, depth_in=depth, num_tokens=tokens, apply_mask=False,
                       use_fp16=amp_dtype != torch.float32, amp_dtype=amp_dtype)['depth']


def main():
    parser = argparse.ArgumentParser(description='Benchmark LingBot-Depth inference latency')
    parser.add_argument('--model', type=str, default='robbyant/lingbot-depth-pretrain-vitl-14-v0.5')
    parser.add_argument('--model-kwargs', type=str, default=None,
                        help='JSON merged into the model config, e.g. \'{"depth_reinject": true, "refiner": {}}\'')
    parser.add_argument('--example', type=str, default='examples/0', help='Example directory used as input')
    parser.add_argument('--size', type=int, nargs=2, default=None, metavar=('H', 'W'), help='Resize input')
    parser.add_argument('--tokens', type=int, nargs='+', default=[1200, 2000, 3600])
    parser.add_argument('--amp-dtype', type=str, nargs='+', default=['fp16', 'bf16', 'fp32'], choices=list(AMP_DTYPES))
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out', type=str, default=None, help='Optional JSON output path')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f'GPU: {torch.cuda.get_device_name(device)} | torch {torch.__version__}')
    model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else None
    model = MDMModel.from_pretrained(args.model, model_kwargs=model_kwargs).to(device).eval()
    image, depth = load_input(Path(args.example), args.size, device)
    print(f'Input: {image.shape[-1]}x{image.shape[-2]} from {args.example}')

    stages = {'encoder': model.encoder, 'neck': model.neck, 'depth_head': model.depth_head, 'mask_head': model.mask_head}
    if hasattr(model, 'refiner'):
        stages['refiner'] = model.refiner
    timer = StageTimer(stages)

    results = []
    for tokens in args.tokens:
        reference = run(model, image, depth, tokens, torch.float32).float()
        for name in args.amp_dtype:
            amp_dtype = AMP_DTYPES[name]
            try:
                for _ in range(args.warmup):
                    run(model, image, depth, tokens, amp_dtype)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                timer.reset()
                timer.enabled = True
                totals = []
                for _ in range(args.iters):
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    pred = run(model, image, depth, tokens, amp_dtype)
                    torch.cuda.synchronize(device)
                    totals.append((time.perf_counter() - t0) * 1000)
                timer.enabled = False
                stages = timer.medians()
                valid = torch.isfinite(reference) & (reference > 0)
                rel_diff = ((pred.float() - reference).abs() / reference)[valid].mean().item()
                result = {
                    'tokens': tokens, 'amp': name,
                    'total_ms': float(np.median(totals)),
                    **{f'{k}_ms': v for k, v in stages.items()},
                    'peak_mem_gb': torch.cuda.max_memory_allocated(device) / 1e9,
                    'rel_diff_vs_fp32': rel_diff,
                }
            except Exception as e:  # e.g. an unsupported dtype on this GPU
                timer.enabled = False
                result = {'tokens': tokens, 'amp': name, 'error': f'{type(e).__name__}: {e}'}
            results.append(result)
            print(json.dumps(result), flush=True)

    print()
    print('| tokens | amp | total | encoder | neck | depth_head | mask_head | refiner | peak mem | rel diff vs fp32 |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in results:
        if 'error' in r:
            print(f'| {r["tokens"]} | {r["amp"]} | error: {r["error"][:60]} |')
            continue
        print(f'| {r["tokens"]} | {r["amp"]} | {r["total_ms"]:.1f} ms | {r["encoder_ms"]:.1f} ms | '
              f'{r["neck_ms"]:.1f} ms | {r["depth_head_ms"]:.1f} ms | {r["mask_head_ms"]:.1f} ms | {r.get("refiner_ms", 0):.1f} ms | '
              f'{r["peak_mem_gb"]:.2f} GB | {r["rel_diff_vs_fp32"]:.2e} |')

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({'gpu': torch.cuda.get_device_name(device), 'results': results}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
