#!/usr/bin/env python3
"""
Train the depth fusion modules (decoder depth re-injection and the anchored refiner) on top of a
frozen LingBot-Depth encoder.

Launch with torchrun, one process per GPU:
    torchrun --nproc_per_node 4 tools/train_fusion.py \\
        --train-root ../../data/mdm_depth_train --val-root ../../data/mdm_depth \\
        --trainable refiner --out runs/refiner_only

Checkpoints are written in the same format as the released models, so they load with
MDMModel.from_pretrained(<out>/best.pt) and work with tools/eval_depth.py --model <out>/best.pt.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from mdm.data.mdm_depth import index_samples
from mdm.data.train_dataset import BucketBatchSampler, RGBDTrainDataset, bucket_samples
from mdm.eval.runner import evaluate_sample, subsample, summarize
from mdm.model.v2 import MDMModel

VAL_SUBSETS = ('RobbySimVal', 'RobbyReal', 'RobbyVla')


def load_model(model_id, model_kwargs):
    """Build the model with the extra modules and load the released weights (new modules keep their init)."""
    path = model_id if Path(model_id).exists() else hf_hub_download(repo_id=model_id, repo_type='model', filename='model.pt')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = {**checkpoint['model_config'], **model_kwargs}
    model = MDMModel(**config)
    state_dict = model.adapt_state_dict(checkpoint['model'])
    result = model.load_state_dict(state_dict, strict=False)
    reshaped = {k for k, v in checkpoint['model'].items() if k in state_dict and state_dict[k].shape != v.shape}
    return model, config, set(state_dict) - reshaped, result


def masked_mean(x, mask):
    mask = mask.float()
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def gradient_loss(log_pred, log_gt, valid, num_scales=4):
    """Multi-scale L1 on the spatial gradients of the log-depth error."""
    diff = log_pred - log_gt
    loss = 0.0
    for s in range(num_scales):
        d, v = diff[:, ::2 ** s, ::2 ** s], valid[:, ::2 ** s, ::2 ** s]
        loss = loss + masked_mean((d[:, :, 1:] - d[:, :, :-1]).abs(), v[:, :, 1:] & v[:, :, :-1])
        loss = loss + masked_mean((d[:, 1:, :] - d[:, :-1, :]).abs(), v[:, 1:, :] & v[:, :-1, :])
    return loss / num_scales


def compute_losses(out, raw, gt, args):
    valid = gt > 0.01
    log_gt = gt.clamp_min(1e-3).log()
    log_pred = out['depth_reg'].float().clamp_min(1e-3).log()
    losses = {
        'depth': masked_mean((log_pred - log_gt).abs(), valid),
        'grad': gradient_loss(log_pred, log_gt, valid),
    }
    total = losses['depth'] + args.w_grad * losses['grad']
    if out.get('confidence') is not None and args.w_conf > 0:
        meas = valid & (raw > 0.01)
        target = ((raw - gt).abs() < args.conf_tau * gt).float()
        conf = out['confidence'].float().clamp(1e-4, 1 - 1e-4)
        losses['conf'] = masked_mean(F.binary_cross_entropy(conf, target, reduction='none'), meas)
        total = total + args.w_conf * losses['conf']
    if out.get('depth_base') is not None and args.w_base > 0:
        log_base = out['depth_base'].float().clamp_min(1e-3).log()
        losses['base'] = masked_mean((log_base - log_gt).abs(), valid)
        total = total + args.w_base * losses['base']
    losses['total'] = total
    return losses


@torch.no_grad()
def validate(model, val_samples, args, device):
    rows = []
    rng = np.random.default_rng(0)
    for sample in val_samples:
        row, _ = evaluate_sample(model, sample, args.tokens, torch.float16, device, args.val_pseudo_hole_ratio, rng)
        rows.append(row)
    summary = summarize(rows)
    metrics = {}
    for subset, stats in summary.items():
        for key in ('pred_vs_gt/all', 'pred_vs_gt/meas', 'pred_vs_gt/hole', 'pred_vs_gt/pseudo', 'pred_vs_raw/meas'):
            for m in ('abs_rel', 'med_ae'):
                if f'{key}/{m}@median' in stats:
                    metrics[f'{subset}/{key}/{m}'] = stats[f'{key}/{m}@median']
            if f'{key}/d1.05' in stats:
                metrics[f'{subset}/{key}/d1.05'] = stats[f'{key}/d1.05']
        if 'conf_mean_meas' in stats:
            metrics[f'{subset}/conf_mean_meas'] = stats['conf_mean_meas']
        metrics[f'{subset}/time_ms'] = stats['time_ms_median']
    # lower is better: mean over subsets of the per-sample median AbsRel on all GT pixels
    metrics['score'] = float(np.mean([metrics[f'{s}/pred_vs_gt/all/abs_rel'] for s in summary]))
    return metrics


def save_checkpoint(path, model, config, iteration, metrics):
    tmp = Path(str(path) + '.tmp')
    torch.save({'model_config': config, 'model': model.state_dict(), 'iteration': iteration, 'metrics': metrics}, tmp)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description='Train depth fusion modules on a frozen LingBot-Depth encoder')
    parser.add_argument('--train-root', type=str, required=True)
    parser.add_argument('--val-root', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--model', type=str, default='robbyant/lingbot-depth-pretrain-vitl-14-v0.5')
    parser.add_argument('--model-kwargs', type=str, default='{"depth_reinject": true, "refiner": {}}',
                        help='JSON merged into the model config')
    parser.add_argument('--trainable', nargs='+', default=['refiner'],
                        help='Parameter name prefixes to train, e.g. refiner neck depth_head')
    parser.add_argument('--subset-weights', type=str, default='RobbySim=0.5,RobbyReal=0.25,RobbyVla=0.25')
    parser.add_argument('--tokens', type=int, default=1200)
    parser.add_argument('--batch-size', type=int, default=4, help='Per GPU')
    parser.add_argument('--num-workers', type=int, default=3, help='Per GPU')
    parser.add_argument('--crop-scale', type=float, nargs=2, default=[0.6, 1.0])
    parser.add_argument('--extra-hole-prob', type=float, default=0.5)
    parser.add_argument('--iters', type=int, default=20000)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--lr-new', type=float, default=1e-4, help='LR of newly added parameters')
    parser.add_argument('--lr-pretrained', type=float, default=1e-5, help='LR of released parameters being fine-tuned')
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--w-grad', type=float, default=0.5)
    parser.add_argument('--w-conf', type=float, default=0.1)
    parser.add_argument('--w-base', type=float, default=0.0, help='Weight of the loss on the pre-refiner depth')
    parser.add_argument('--conf-tau', type=float, default=0.01, help='Relative error below which a measurement counts as reliable')
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--val-every', type=int, default=1000)
    parser.add_argument('--val-limit', type=int, default=100, help='Validation samples per subset')
    parser.add_argument('--val-pseudo-hole-ratio', type=float, default=0.0)
    parser.add_argument('--no-val-at-start', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    distributed = 'LOCAL_RANK' in os.environ
    if distributed:
        dist.init_process_group('nccl')
        rank, world_size, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ['LOCAL_RANK'])
    else:
        rank, world_size, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.manual_seed(args.seed + rank)
    is_main = rank == 0
    out_dir = Path(args.out)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'args.json').write_text(json.dumps(vars(args), indent=2))

    def log(msg):
        if is_main:
            print(msg, flush=True)

    # Model
    model, config, pretrained_keys, load_result = load_model(args.model, json.loads(args.model_kwargs))
    log(f'Missing keys (new modules): {sorted({k.split(".")[0] for k in load_result.missing_keys})}')
    new_params, pretrained_params = [], []
    for name, p in model.named_parameters():
        p.requires_grad = any(name.startswith(prefix) for prefix in args.trainable)
        if p.requires_grad:
            (pretrained_params if name in pretrained_keys else new_params).append(p)
    log(f'Trainable: {sum(p.numel() for p in new_params) / 1e6:.3f}M new, '
        f'{sum(p.numel() for p in pretrained_params) / 1e6:.3f}M pretrained')
    model.to(device).eval()  # no dropout / batch norm in the trained parts; keeps the frozen encoder deterministic
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank]) if distributed else model

    optimizer = torch.optim.AdamW(
        [{'params': new_params, 'lr': args.lr_new}, {'params': pretrained_params, 'lr': args.lr_pretrained}],
        weight_decay=args.weight_decay)
    base_lrs = [args.lr_new, args.lr_pretrained]

    def lr_factor(it):
        if it < args.warmup:
            return (it + 1) / args.warmup
        progress = (it - args.warmup) / max(1, args.iters - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scaler = torch.amp.GradScaler('cuda', init_scale=2.0 ** 10)
    trainable = new_params + pretrained_params

    # Data
    weights = {k: float(v) for k, v in (kv.split('=') for kv in args.subset_weights.split(','))}
    train_samples = index_samples(args.train_root, [s for s, w in weights.items() if w > 0])
    buckets = bucket_samples(train_samples)
    log(f'Train samples: {len(train_samples)} in {len(buckets)} buckets: '
        + ', '.join(f'{k[0]} {k[2]}x{k[1]}: {len(v)}' for k, v in sorted(buckets.items())))
    sampler = BucketBatchSampler(buckets, args.batch_size, weights, tuple(args.crop_scale), seed=args.seed + 1000 * rank)
    dataset = RGBDTrainDataset(train_samples, extra_hole_prob=args.extra_hole_prob)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0, prefetch_factor=4 if args.num_workers > 0 else None)
    val_samples = []
    if is_main:
        for subset in VAL_SUBSETS:
            val_samples += subsample(index_samples(args.val_root, [subset]), args.val_limit)
        log(f'Validation samples: {len(val_samples)}')

    def run_validation(iteration, best_score):
        if is_main:
            t0 = time.time()
            metrics = validate(model, val_samples, args, device)
            metrics['iteration'] = iteration
            with open(out_dir / 'val.jsonl', 'a') as f:
                f.write(json.dumps(metrics) + '\n')
            log(f'[val {iteration}] score {metrics["score"]:.4f} ({time.time() - t0:.0f}s) | ' + ' | '.join(
                f'{s}: all {metrics[f"{s}/pred_vs_gt/all/abs_rel"]:.4f} meas {metrics[f"{s}/pred_vs_gt/meas/med_ae"] * 1000:.1f}mm '
                f'hole {metrics.get(f"{s}/pred_vs_gt/hole/abs_rel", float("nan")):.4f}'
                for s in VAL_SUBSETS if f'{s}/pred_vs_gt/all/abs_rel' in metrics))
            if iteration == 0:
                best_score = metrics['score']  # checkpoints must beat the model before training
            else:
                save_checkpoint(out_dir / 'latest.pt', model, config, iteration, metrics)
                if metrics['score'] < best_score:
                    best_score = metrics['score']
                    save_checkpoint(out_dir / 'best.pt', model, config, iteration, metrics)
                    log(f'  new best {best_score:.4f}')
        if distributed:
            dist.barrier()
        return best_score

    # Train
    best_score = math.inf
    if not args.no_val_at_start:
        best_score = run_validation(0, best_score)
    running, t_last, t_data = {}, time.time(), 0.0
    loader_iter = iter(loader)
    for it in range(args.iters):
        t0 = time.time()
        batch = next(loader_iter)
        image = batch['image'].to(device, non_blocking=True)
        raw = batch['raw'].to(device, non_blocking=True)
        gt = batch['gt'].to(device, non_blocking=True)
        t_data += time.time() - t0

        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group['lr'] = base_lr * lr_factor(it)
        with torch.autocast('cuda', dtype=torch.float16):
            out = ddp_model(image, num_tokens=args.tokens, depth=raw)
        losses = compute_losses(out, raw, gt, args)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(losses['total']).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v)
        running['grad_norm'] = running.get('grad_norm', 0.0) + float(grad_norm)
        if (it + 1) % args.log_every == 0:
            n = args.log_every
            elapsed = time.time() - t_last
            record = {'iteration': it + 1, **{k: v / n for k, v in running.items()},
                      'lr_new': optimizer.param_groups[0]['lr'], 'sec_per_iter': elapsed / n, 'data_frac': t_data / elapsed,
                      'shape': list(image.shape[-2:]), 'max_mem_gb': torch.cuda.max_memory_allocated(device) / 1e9}
            if hasattr(model, 'refiner') and hasattr(model.refiner, 'prop_gate'):
                record['prop_gate'] = float(model.refiner.prop_gate)
            if is_main:
                with open(out_dir / 'train.jsonl', 'a') as f:
                    f.write(json.dumps(record) + '\n')
            log(f'[{it + 1}/{args.iters}] ' + ' '.join(
                f'{k} {v:.4f}' for k, v in record.items() if k not in ('iteration', 'shape')))
            running, t_last, t_data = {}, time.time(), 0.0

        if (it + 1) % args.val_every == 0 or it + 1 == args.iters:
            best_score = run_validation(it + 1, best_score)

    if distributed:
        del loader_iter, loader  # stop the workers and free cached blocks: NCCL shutdown can hit OOM on a full GPU
        torch.cuda.empty_cache()
        dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
