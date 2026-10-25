from typing import *
from numbers import Number
from functools import partial
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.amp
import torch.version
from huggingface_hub import hf_hub_download

from .modules_rgbd_encoder import DINOv2_RGBD_Encoder
from .modules_decoder import MLP, ConvStack
from .modules_refiner import AnchoredRefiner
from ..utils.geo import depth_to_pointcloud, normalized_view_plane_uv


class MDMModel(nn.Module):
    encoder: Union[DINOv2_RGBD_Encoder]
    neck: ConvStack
    points_head: ConvStack
    mask_head: ConvStack
    scale_head: MLP
    onnx_compatible_mode: bool

    def __init__(self, 
        encoder: Dict[str, Any],
        neck: Dict[str, Any],
        depth_head: Dict[str, Any] = None,
        mask_head: Dict[str, Any] = None,
        normal_head: Dict[str, Any] = None,
        scale_head: Dict[str, Any] = None,
        remap_output: Literal['linear', 'sinh', 'exp', 'sinh_exp'] = 'linear',
        remap_depth_in: Literal['linear', 'log'] = 'log',
        remap_depth_out: Literal['linear', 'exp'] = 'exp',
        num_tokens_range: List[int] = [1200, 3600],
        depth_reinject: bool = False,
        refiner: Dict[str, Any] = None,
        **deprecated_kwargs
    ):
        super(MDMModel, self).__init__()
        if deprecated_kwargs:
            warnings.warn(f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}")

        self.remap_output = remap_output
        self.num_tokens_range = num_tokens_range
        self.remap_depth_in = remap_depth_in
        self.remap_depth_out = remap_depth_out
        self.depth_reinject = depth_reinject
        
        self.encoder = DINOv2_RGBD_Encoder(**encoder)
        
        if depth_reinject:
            # levels 1-4 of the neck take [log raw depth, raw coverage] in addition to the UV map
            neck = {**neck, 'dim_in': [neck['dim_in'][0]] + [d + 2 for d in neck['dim_in'][1:]]}
        self.neck = ConvStack(**neck)
        if depth_head is not None:
            self.depth_head = ConvStack(**depth_head) 
        if mask_head is not None:
            self.mask_head = ConvStack(**mask_head)
        if refiner is not None:
            self.refiner = AnchoredRefiner(**refiner)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: Union[str, Path, IO[bytes]], 
        model_kwargs: Optional[Dict[str, Any]] = None, 
        **hf_kwargs) -> 'MDMModel':
        if Path(pretrained_model_name_or_path).exists():
            checkpoint_path = pretrained_model_name_or_path
        else:
            checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                repo_type="model",
                filename="model.pt",
                **hf_kwargs
            )
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        model_config = checkpoint['model_config']
        if model_kwargs is not None:
            model_config.update(model_kwargs)
        model = cls(**model_config)
        state_dict = model.adapt_state_dict(checkpoint['model'])
        model.load_state_dict(state_dict, strict=False)
        
        return model

    def adapt_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Zero-pad neck input weights of a checkpoint without depth re-injection, so the extra inputs start unused."""
        state_dict = dict(state_dict)
        own = self.state_dict()
        for key, value in state_dict.items():
            if key.startswith('neck.input_blocks.') and key.endswith('.weight') and key in own \
                    and value.shape != own[key].shape and value.shape[1] < own[key].shape[1]:
                padded = torch.zeros_like(own[key])
                padded[:, :value.shape[1]] = value
                state_dict[key] = padded
        return state_dict
            
    def init_weights(self):
        self.encoder.init_weights()

    def enable_pytorch_native_sdpa(self):
        self.encoder.enable_pytorch_native_sdpa()
    
    def forward(self, 
                image: torch.Tensor, 
                num_tokens: Union[int, torch.LongTensor], 
                depth: Union[None, torch.Tensor]=None, 
                **kwargs) -> Dict[str, torch.Tensor]:
        batch_size, _, img_h, img_w = image.shape
        device, dtype = image.device, image.dtype
        
        assert depth is not None  # in this version, depth is required
        if depth.dim() == 3:
            depth = depth.unsqueeze(1) # from (B, H, W) to (B, 1, H, W)

        aspect_ratio = img_w / img_h
        base_h, base_w = (num_tokens / aspect_ratio) ** 0.5, (num_tokens * aspect_ratio) ** 0.5
        if isinstance(base_h, torch.Tensor):
            base_h, base_w = base_h.round().long(), base_w.round().long()
        else:
            base_h, base_w = round(base_h), round(base_w)

        # Backbones encoding
        features, cls_token, _, _ = self.encoder(image, depth, base_h, base_w, return_class_token=True, remap_depth_in=self.remap_depth_in, **kwargs)

        features = features + cls_token[..., None, None]
        features = [features, None, None, None, None]

        # Concat UVs for aspect ratio input
        for level in range(5):
            uv = normalized_view_plane_uv(width=base_w * 2 ** level, height=base_h * 2 ** level, aspect_ratio=aspect_ratio, dtype=dtype, device=device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat([features[level], uv], dim=1)

        # Re-inject the measured depth at every decoder level
        raw_mask, log_raw = self._log_depth(depth)
        if self.depth_reinject:
            for level in range(1, 5):
                size = (base_h * 2 ** level, base_w * 2 ** level)
                coverage = F.adaptive_avg_pool2d(raw_mask, size)
                log_raw_level = F.adaptive_avg_pool2d(log_raw * raw_mask, size) / coverage.clamp_min(1e-6)
                features[level] = torch.concat([features[level], log_raw_level.to(dtype), coverage.to(dtype)], dim=1)

        # Shared neck
        features = self.neck(features)
        guide = features[-1]

        # Heads decoding
        depth_reg, normal, mask = (getattr(self, head)(features)[-1] if hasattr(self, head) else None for head in ['depth_head', 'normal_head', 'mask_head'])
        metric_scale = self.scale_head(cls_token) if hasattr(self, 'scale_head') else None
        
        # Resize
        depth_reg, normal, mask = (F.interpolate(v, (img_h, img_w), mode='bilinear', align_corners=False, antialias=False) if v is not None else None for v in [depth_reg, normal, mask])
        
        # Remap output
        if depth_reg is not None:
            if self.remap_depth_out == 'exp':
                depth_reg = depth_reg.exp().squeeze(1)
            elif self.remap_depth_out == 'linear':
                depth_reg = depth_reg.squeeze(1)
            else:
                raise ValueError(f"Invalid remap_depth_out: {self.remap_depth_out}")
        if normal is not None:
            normal = normal.permute(0, 2, 3, 1)
            normal = F.normalize(normal, dim=-1)
        if mask is not None:
            mask_prob = mask.squeeze(1).sigmoid()
            # mask_logits = mask.squeeze(1)
        else:
            mask_prob = None
        if metric_scale is not None:
            metric_scale = metric_scale.squeeze(1).exp()

        # Refine with the measured depth at full resolution
        depth_base, confidence = depth_reg, None
        if depth_reg is not None and hasattr(self, 'refiner'):
            log_pred = depth_reg.float().clamp_min(1e-3).log().unsqueeze(1)
            log_depth, confidence = self.refiner(image, log_pred, log_raw, raw_mask, guide)
            depth_reg, confidence = log_depth.exp().squeeze(1), confidence.squeeze(1)

        return_dict = {
            'depth_reg': depth_reg,
            'depth_base': depth_base if hasattr(self, 'refiner') else None,
            'confidence': confidence,
            'normal': normal,
            'mask': mask_prob,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        return return_dict

    @staticmethod
    def _log_depth(depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Valid mask (depth > 0.01 m, finite) and log depth (0 where invalid) of a (B, 1, H, W) depth map."""
        depth = depth.float()
        mask = (torch.isfinite(depth) & (depth > 0.01)).float()
        log_depth = torch.where(mask > 0, depth.clamp_min(0.01).log(), torch.zeros_like(depth))
        return mask, log_depth

    @torch.inference_mode()
    def infer(
        self, 
        image: torch.Tensor, 
        depth_in: torch.Tensor = None,
        num_tokens: int = None,
        resolution_level: int = 9,
        apply_mask: bool = True,
        use_fp16: bool = True,
        intrinsics: Optional[torch.Tensor] = None,
        amp_dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False
        image = image.to(dtype=self.dtype, device=self.device)

        if (depth_in is not None) and (depth_in.dim() == 2):
            depth_in = depth_in.unsqueeze(0).to(dtype=self.dtype, device=self.device)

        original_height, original_width = image.shape[-2:]
        area = original_height * original_width
        aspect_ratio = original_width / original_height
        
        # Determine the number of base tokens to use
        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        # Forward pass
        with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_fp16 and self.dtype != amp_dtype):
            output = self.forward(image, num_tokens=num_tokens, depth=depth_in, **kwargs)
        depth_reg, mask, confidence = (output.get(k, None) for k in ['depth_reg', 'mask', 'confidence'])

        # Always process the output in fp32 precision
        depth_reg, mask, confidence = map(lambda x: x.float() if isinstance(x, torch.Tensor) else x, [depth_reg, mask, confidence])
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            if mask is not None:
                mask_binary = mask > 0.5
            else:
                mask_binary = None
                
            depth = depth_reg
            if intrinsics is not None:
                points = depth_to_pointcloud(depth, intrinsics)
            else:
                points = None

            # Apply mask
            if apply_mask and mask_binary is not None:
                points = torch.where(mask_binary[..., None], points, torch.inf) if points is not None else None
                depth = torch.where(mask_binary, depth, torch.inf) if depth is not None else None

        return_dict = {
            'points': points,
            'depth': depth,
            'mask': mask_binary,
            'confidence': confidence,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        if omit_batch_dim:
            return_dict = {k: v.squeeze(0) for k, v in return_dict.items()}

        return return_dict
    
    def forward_feat(self, 
                image: torch.Tensor, 
                num_tokens: Union[int, torch.LongTensor], 
                depth: Union[None, torch.Tensor]=None, 
                **kwargs) -> Dict[str, torch.Tensor]:
        batch_size, _, img_h, img_w = image.shape
        device, dtype = image.device, image.dtype
        
        assert depth is not None  # in this version, depth is required
        if depth.dim() == 3:
            depth = depth.unsqueeze(1) # from (B, H, W) to (B, 1, H, W)

        aspect_ratio = img_w / img_h
        base_h, base_w = (num_tokens / aspect_ratio) ** 0.5, (num_tokens * aspect_ratio) ** 0.5
        if isinstance(base_h, torch.Tensor):
            base_h, base_w = base_h.round().long(), base_w.round().long()
        else:
            base_h, base_w = round(base_h), round(base_w)

        # Backbones encoding
        features, cls_token, _, _ = self.encoder(image, depth, base_h, base_w, return_class_token=True, remap_depth_in=self.remap_depth_in, **kwargs)
        
        return features, cls_token


    @torch.inference_mode()
    def infer_feat(
        self, 
        image: torch.Tensor, 
        depth_in: torch.Tensor = None,
        num_tokens: int = None,
        resolution_level: int = 9,
        apply_mask: bool = True,
        use_fp16: bool = True,
        intrinsics: Optional[torch.Tensor] = None,
        amp_dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ):
        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False
        image = image.to(dtype=self.dtype, device=self.device)

        if (depth_in is not None) and (depth_in.dim() == 2):
            depth_in = depth_in.unsqueeze(0).to(dtype=self.dtype, device=self.device)

        original_height, original_width = image.shape[-2:]
        area = original_height * original_width
        aspect_ratio = original_width / original_height
        
        # Determine the number of base tokens to use
        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        # Forward pass
        with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_fp16 and self.dtype != amp_dtype):
            features, cls_token = self.forward_feat(image, num_tokens=num_tokens, depth=depth_in, **kwargs)
        
        return features, cls_token