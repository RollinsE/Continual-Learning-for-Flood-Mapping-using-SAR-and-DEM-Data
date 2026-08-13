from typing import List, Type

import torch
import torch.nn.functional as F
from torch import nn

from floods.models.base import Decoder


class SegFormerDecoder(Decoder):
    """Lightweight all-MLP-style SegFormer decoder over four encoder scales.

    Each encoder feature map is projected to a common embedding width, resized to
    the highest-resolution feature level, concatenated, and fused. The final
    project segmentation head performs the remaining upsampling to input size.
    """

    def __init__(self, input_size: int, feature_channels: List[int], feature_reductions: List[int],
                 act_layer: Type[nn.Module], norm_layer: Type[nn.Module],
                 embed_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__(input_size, feature_channels, feature_reductions, act_layer, norm_layer)
        if len(feature_channels) != 4 or len(feature_reductions) != 4:
            raise ValueError("SegFormer decoder requires exactly four encoder feature levels")
        self.channels = int(embed_dim)
        self.reduction = int(min(feature_reductions))
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(int(channels), self.channels, kernel_size=1, bias=False),
                norm_layer(self.channels),
                act_layer(),
            )
            for channels in feature_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(self.channels * 4, self.channels, kernel_size=1, bias=False),
            norm_layer(self.channels),
            act_layer(),
            nn.Dropout2d(float(dropout)),
        )

    @classmethod
    def required_indices(cls, encoder: str) -> List[int]:
        name = str(encoder).lower()
        if name.startswith(("mit_", "pvt_", "swin", "convnext", "efficientformer")):
            return [0, 1, 2, 3]
        return [1, 2, 3, 4]

    def out_channels(self) -> int:
        return self.channels

    def out_reduction(self) -> int:
        return self.reduction

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            value = projection(feature)
            if value.shape[-2:] != target_size:
                value = F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
            projected.append(value)
        return self.fuse(torch.cat(projected, dim=1))
