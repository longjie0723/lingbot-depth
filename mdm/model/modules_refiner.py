from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules_decoder import ResidualConvBlock


def propagate(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted sum over each pixel's 3x3 neighborhood. x: (B, 1, H, W), weights: (B, 9, H, W) summing to 1."""
    h, w = x.shape[-2:]
    padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
    neighbors = torch.cat([padded[:, :, dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)], dim=1)
    return (neighbors * weights).sum(dim=1, keepdim=True)


class AnchoredRefiner(nn.Module):
    """
    Refine the decoded depth with the measured depth at full resolution.

    Works in log depth. A per-pixel confidence decides how much each measured pixel is trusted.
    The confident residual log(raw) - log(pred) is propagated coarse to fine with learned
    affinities, and at every iteration the confident pixels are pulled back to their measured
    residual. At full resolution the output is anchored to the measurement in proportion to the
    confidence, so confident pixels keep the measured depth and holes inherit the offset of their
    surroundings while keeping the shape of the prediction. A gate decides how much of the propagated
    offset the holes take: a single learned scalar, or a per-pixel value predicted from the guidance
    features and the density of confident measurements around each pixel at every scale.

    Args:
        guide_dim: channels of the decoder feature used as guidance
        hidden_dim: channels of the half-resolution guidance network
        scales: downsampling factors (w.r.t. full resolution) of the propagation stages, coarse to fine
        iters: propagation iterations at each stage
        conf_bias_init: initial confidence logit; together with the closed propagation gate,
            a low value makes the module start close to identity
        gate: 'scalar' (one learned gate, starting at 0) or 'pixel' (per-pixel gate)
        gate_bias_init: initial logit of the per-pixel gate
    """

    def __init__(self,
        guide_dim: int = 32,
        hidden_dim: int = 32,
        num_blocks: int = 2,
        scales: Sequence[int] = (16, 8, 4, 2),
        iters: Sequence[int] = (8, 4, 4, 2),
        conf_bias_init: float = -5.0,
        gate: Literal['scalar', 'pixel'] = 'scalar',
        gate_bias_init: float = -5.0,
    ):
        super().__init__()
        assert len(scales) == len(iters)
        assert scales[-1] >= 2 and all(scales[i] % scales[-1] == 0 for i in range(len(scales)))
        assert gate in ('scalar', 'pixel')
        self.scales = list(scales)
        self.iters = list(iters)
        self.gate = gate

        # inputs: image (3), log pred - median (1), raw mask (1), residual (1), |residual| (1)
        self.stem = nn.Conv2d(7, hidden_dim, kernel_size=3, stride=2, padding=1, padding_mode='replicate')
        self.guide_proj = nn.Conv2d(guide_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(*[
            ResidualConvBlock(hidden_dim, hidden_dim, hidden_dim, activation='relu', in_norm='none', hidden_norm='none')
            for _ in range(num_blocks)
        ])
        self.conf_head = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, padding_mode='replicate')
        # full-resolution correction of the confidence from local residual statistics
        self.conf_local = nn.Conv2d(3, 1, kernel_size=3, padding=1, padding_mode='replicate')
        self.affinity_heads = nn.ModuleList([
            nn.Conv2d(hidden_dim, 9, kernel_size=3, padding=1, padding_mode='replicate') for _ in scales
        ])
        if gate == 'scalar':
            self.prop_gate = nn.Parameter(torch.zeros(1))
        else:
            # inputs: guidance features and the confidence density at every propagation scale
            self.gate_head = nn.Conv2d(hidden_dim + len(scales), 1, kernel_size=3, padding=1, padding_mode='replicate')
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, gate_bias_init)

        nn.init.zeros_(self.conf_head.weight)
        nn.init.constant_(self.conf_head.bias, conf_bias_init)
        nn.init.zeros_(self.conf_local.weight)
        nn.init.zeros_(self.conf_local.bias)
        for head in self.affinity_heads:  # start from isotropic diffusion
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, image: torch.Tensor, log_pred: torch.Tensor, log_raw: torch.Tensor, raw_mask: torch.Tensor,
                guide: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: (B, 3, H, W) RGB in [0, 1]
            log_pred: (B, 1, H, W) log of the decoded depth
            log_raw: (B, 1, H, W) log of the measured depth, any value where invalid
            raw_mask: (B, 1, H, W) 1 where the measurement is valid
            guide: (B, C, h, w) decoder feature at any resolution
        Returns:
            log depth (B, 1, H, W) and confidence (B, 1, H, W), 0 where there is no measurement
        """
        h, w = image.shape[-2:]
        half = (h // 2 + h % 2, w // 2 + w % 2)
        residual = torch.where(raw_mask > 0, log_raw - log_pred, torch.zeros_like(log_pred))
        log_pred_centered = log_pred - log_pred.flatten(1).median(dim=1).values[:, None, None, None]

        x = torch.cat([(image - 0.5) / 0.25, log_pred_centered, raw_mask, residual, residual.abs()], dim=1)
        feat = self.stem(x)
        feat = feat + self.guide_proj(F.interpolate(guide, size=feat.shape[-2:], mode='bilinear', align_corners=False))
        feat = self.blocks(feat)

        conf_logits = F.interpolate(self.conf_head(feat), size=(h, w), mode='bilinear', align_corners=False)
        conf_logits = conf_logits + self.conf_local(torch.cat([raw_mask, residual, residual.abs()], dim=1))
        conf = torch.sigmoid(conf_logits.float()) * raw_mask

        # coarse-to-fine anchored propagation of the confident residual
        anchored = conf * residual.float()
        x = None
        densities = []
        for scale, n_iter, head in zip(self.scales, self.iters, self.affinity_heads):
            size = (-(-h // scale), -(-w // scale))
            weight = F.adaptive_avg_pool2d(conf, size)
            densities.append(F.interpolate(weight, size=feat.shape[-2:], mode='bilinear', align_corners=False))
            value = F.adaptive_avg_pool2d(anchored, size) / weight.clamp_min(1e-6)
            if x is None:
                x = value * (weight > 1e-6)
            else:
                x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            affinity = torch.softmax(head(F.adaptive_avg_pool2d(feat, size)).float(), dim=1)
            for _ in range(n_iter):
                x = weight * value + (1 - weight) * propagate(x, affinity)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        if self.gate == 'scalar':
            gate = self.prop_gate
        else:
            gate_logits = self.gate_head(torch.cat([feat, torch.cat(densities, dim=1).to(feat.dtype)], dim=1))
            gate = torch.sigmoid(F.interpolate(gate_logits.float(), size=(h, w), mode='bilinear', align_corners=False))
        refined_residual = anchored + (1 - conf) * gate * x
        return log_pred.float() + refined_residual, conf
