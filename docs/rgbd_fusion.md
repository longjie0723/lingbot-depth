# RGB-D fusion (branch `rgbd-fusion`)

LingBot-Depth uses the measured depth as an input to the encoder, but the decoder only sees RGB tokens,
so the output follows the measurement only loosely: on measured pixels the released model is less accurate
than the sensor itself (e.g. 6.9 mm vs 1.0 mm median error on RobbyVla / D405). This branch adds learned
fusion inside the model so that the output keeps the measured depth where it is reliable and fills the
holes consistently with it, at a small latency cost.

Research only: the fusion weights are trained on [robbyant/mdm_depth](https://huggingface.co/datasets/robbyant/mdm_depth)
(CC BY-NC-SA 4.0), so they must not be used commercially.

## Model changes

Both are options of `MDMModel` (stored in the checkpoint config), off by default, so released checkpoints
load and behave exactly as before.

- `depth_reinject=True`: neck levels 1-4 additionally take the log measured depth and its coverage, pooled
  to each level. The extra input weights are zero-initialized (`MDMModel.adapt_state_dict`), so a released
  checkpoint gives identical output until it is fine-tuned.
- `refiner={...}`: `AnchoredRefiner` (`mdm/model/modules_refiner.py`, ~51k parameters) refines the decoded
  depth at full resolution. It predicts a per-pixel confidence of the measurement, anchors confident pixels
  to it, and propagates the confident residual log(raw) - log(pred) coarse to fine with learned affinities,
  so holes take the offset of their surroundings while keeping the predicted shape. How much of the
  propagated residual the holes take is a single learned scalar (`gate='scalar'`) or a per-pixel value
  predicted from the guidance features and the local density of confident measurements (`gate='pixel'`).
  The model then also returns `depth_base` (before refinement) and `confidence`.
- `infer(..., amp_dtype=torch.float16)`: bf16 autocast is the default, but xformers attention does not
  support bf16 before Ampere; fp16 matches fp32 to ~2e-4 relative.

## Data

`tools/stream_subset.py` extracts a subset of the dataset archives by streaming (`curl | zstd | tar`),
without downloading the 50-320 GB archives. `mdm/data/mdm_depth.py` reads the extracted layouts.

- Eval subset: RobbySimVal (1557 frames, `--frame-stride 25`), RobbyReal RealSense D415/D435/D455 of rooms
  00001_01-04 (84 frames, `--frame-stride 100`), RobbyVla D405 (627 frames, every 10th frame).
- Training subset, native resolution, no scene overlap with the eval subset (125k frames, 132 GB):
  RobbySim (every 6th camera group, 55k), RobbyReal active-stereo cameras outside the eval rooms
  (`--include '^RobbyReal/(?!00001_0[1-4]_)' --frame-stride 10`, 27k), RobbyVla (`--frame-stride 20`
  excluding eval groups, 44k).

## Tools

| tool | purpose |
|---|---|
| `tools/eval_depth.py` | metrics vs GT on measured pixels / holes / all pixels and vs the raw depth, optionally on pseudo holes |
| `tools/compare_eval.py` | side-by-side table of several `eval_depth.py` results |
| `tools/analyze_fusion_headroom.py` | training-free references: per-pixel gate rule and oracle selection |
| `tools/bench_latency.py` | latency with a per-stage breakdown |
| `tools/train_fusion.py` | trains the fusion modules on a frozen encoder (DDP, fp16, resolution-bucketed batches) |

```bash
# train (2 GPUs): refiner + neck + depth head, per-pixel gate
torchrun --nproc_per_node 2 tools/train_fusion.py --train-root data/mdm_depth_train --val-root data/mdm_depth \
    --model-kwargs '{"depth_reinject": true, "refiner": {"gate": "pixel"}}' \
    --trainable refiner neck depth_head --w-base 0.5 --iters 20000 --out runs/full_pixgate

# evaluate (SimVal and Real/Vla are separate invocations so that the pseudo holes match across models)
python tools/eval_depth.py --data-root data/mdm_depth --subsets RobbySimVal --model runs/full_pixgate/latest.pt \
    --tokens 1200 --pseudo-hole-ratio 0.15 --out results/full_pixgate/simval
python tools/eval_depth.py --data-root data/mdm_depth --subsets RobbyReal RobbyVla --model runs/full_pixgate/latest.pt \
    --tokens 1200 --pseudo-hole-ratio 0.15 --out results/full_pixgate/real
python tools/compare_eval.py base=results/base/simval,results/base/real full_pixgate=results/full_pixgate
```

## Results

Full eval subset, 1200 tokens (resolution level 0), fp16, final checkpoints after 20k iterations
(batch 4 x 2 GPUs, encoder frozen). Per-sample medians: MedAE in mm on measured pixels (meas) and on
pixels without measurement (hole), and AbsRel on all pixels with GT.

| model | Real meas | Real hole | Real all | Vla meas | Vla hole | Vla all | SimVal meas | SimVal hole | SimVal all |
|---|---|---|---|---|---|---|---|---|---|
| raw depth | 4.0 | - | - | 1.0 | - | - | 4.0 | - | - |
| released model | 5.5 | 11.2 | 0.0108 | 6.9 | 10.1 | 0.1434 | 13.6 | 19.4 | 0.0160 |
| gate@0.05 (training-free) | 4.0 | 11.2 | 0.0123 | 2.0 | 10.1 | 0.1366 | 6.0 | 19.4 | 0.0124 |
| oracle on measured pixels | 2.3 | 11.2 | 0.0068 | 1.0 | 10.1 | 0.1097 | 3.3 | 19.4 | 0.0094 |
| refiner, scalar gate | 3.3 | 10.1 | 0.0107 | 1.4 | 7.8 | 0.0901 | 8.1 | 17.3 | 0.0134 |
| refiner + neck + head, scalar gate | 3.7 | 9.9 | 0.0092 | 1.1 | 4.6 | 0.0537 | 9.0 | 18.4 | 0.0134 |
| refiner, pixel gate | 3.3 | 11.3 | 0.0095 | 1.0 | 3.7 | 0.0665 | 9.7 | 19.4 | 0.0146 |
| **refiner + neck + head, pixel gate** | 3.7 | **9.6** | **0.0090** | **1.0** | **2.4** | **0.0386** | 9.2 | 18.2 | 0.0137 |

gate@0.05 takes the raw depth where it is within 5% of the prediction; the oracle takes whichever of raw and
prediction is closer to GT. Both only change measured pixels (`tools/analyze_fusion_headroom.py`).

- On the RealSense subsets (Real: D415/D435/D455, Vla: D405) the fused model keeps the measured depth
  (Vla at the sensor's own 1.0 mm; Real better than the raw depth on 76% of the frames) and fills the holes
  much more accurately than the released model (Vla hole error 10.1 -> 2.4 mm).
- The per-pixel gate removes the imprint of the sparse measurement pattern that the scalar gate left in the
  output: holes between measured blobs took only part of the propagated correction.
- RobbySimVal does not reach the training-free gate rule. The only raw depth shipped with the RobbySim
  training scenes (`*_rmd2c.png`) is much noisier than the RobbySimVal raw depth (56% vs 81% of pixels within
  1% of GT), so the model learns to distrust simulated measurements (mean confidence on measured pixels about
  0.5 on SimVal vs 0.8 on the real subsets). How much the model trusts the measurement is learned from the
  training sensor, so a product model should be trained on data from its own camera.

Latency on an RTX 2080 Ti (fp16, `tools/bench_latency.py`):

| input | released model | refiner + neck + head, pixel gate |
|---|---|---|
| 1280x960 | 105 ms | 122 ms |
| 1280x1024 | 109 ms | 127 ms |
