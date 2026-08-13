from typing import Iterable, List, Type

import torch
from torch import nn

from floods.models import Decoder


class UNetDecodeBlock(nn.Module):
    """UNet basic block, providing an upscale from the lower features and a skip connection
    from the encoder. This specific version adopts a residual skip similar to ResNets.
    """
    def __init__(self,
                 in_channels: int,
                 skip_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 act_layer: Type[nn.Module],
                 norm_layer: Type[nn.Module],
                 scale_factor: int = 2,
                 bilinear: bool = True):
        """Creates a new UNet block, with residual skips.

        Args:
            in_channels (int): number of input channels
            skip_channels (int): number of channels coming from the skip connection (usually 2 * input)
            out_channels (int): number of desired channels in output
            scale_factor (int, optional): How much should the input be scaled. Defaults to 2.
            bilinear (bool, optional): Upscale with bilinear and conv1x1 or transpose conv. Defaults to True.
            norm_layer (Type[nn.Module]: normalization layer.
            act_layer (Type[nn.Module]): activation layer.
        """
        super().__init__()
        self.upsample = self._upsampling(in_channels, mid_channels, factor=scale_factor, bilinear=bilinear)
        self.conv = self._upconv(mid_channels + skip_channels, out_channels, act_layer=act_layer, norm_layer=norm_layer)
        self.adapter = nn.Conv2d(mid_channels, out_channels, 1) if mid_channels != out_channels else nn.Identity()

    def _upsampling(self, in_channels: int, out_channels: int, factor: int, bilinear: bool = True):
        """Create the upsampling block using bilinear interpolation or transposed convolution.

        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            factor (int): upscaling factor
            bilinear (bool, optional): Use bilinear or upconvolutions. Defaults to True.

        Returns:
            nn.Module: upsampling block
        """
        if bilinear:
            return nn.Sequential(nn.Upsample(scale_factor=factor, mode="bilinear", align_corners=True),
                                 nn.Conv2d(in_channels, out_channels, kernel_size=1))
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=factor, stride=factor)

    def _upconv(self, in_channels: int, out_channels: int, act_layer: Type[nn.Module],
                norm_layer: Type[nn.Module]) -> nn.Sequential:
        """Creates a decoder block in the UNet standard architecture.
        Two conv3x3 with batch norms and activations.

        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            act_layer (Type[nn.Module]): activation layer
            norm_layer (Type[nn.Module]): normalization layer

        Returns:
            nn.Sequential: UNet basic decoder block.
        """
        mid_channels = out_channels
        return nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                             norm_layer(mid_channels), act_layer(),
                             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                             norm_layer(out_channels), act_layer())

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x2 = self.conv(torch.cat((x, skip), dim=1))
        x1 = self.adapter(x)
        return x1 + x2


class UNet(Decoder):
    """UNet architecture with dynamic adaptation to the encoder.
    Residual skip paths are used in the decoder to stabilise feature reconstruction.
    """
    def __init__(self,
                 input_size: int,
                 feature_channels: List[int],
                 feature_reductions: List[int],
                 act_layer: Type[nn.Module],
                 norm_layer: Type[nn.Module],
                 bilinear: bool = True,
                 num_classes: int = None,
                 drop_channels: bool = False,
                 dropout_prob: int = 0.5):
        super().__init__(input_size, feature_channels, feature_reductions, act_layer, norm_layer)
        # invert sequences to decode
        channels = feature_channels[::-1]
        reductions = feature_reductions[::-1] + [1]
        scaling_factors = [int(reductions[i] // reductions[i + 1]) for i in range(len(reductions) - 1)]
        # dinamically create the decoder blocks (some encoders do not have all 5 layers)
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(
                UNetDecodeBlock(in_channels=channels[i],
                                mid_channels=channels[i] // 2,
                                skip_channels=channels[i + 1],
                                out_channels=channels[i + 1],
                                act_layer=act_layer,
                                norm_layer=norm_layer,
                                scale_factor=scaling_factors[i],
                                bilinear=bilinear))
        drop_class = nn.Dropout2d if drop_channels else nn.Dropout
        self.dropout = drop_class(p=dropout_prob)
        self.channels = channels[-1]
        self.reduction = scaling_factors[-1]

    @classmethod
    def required_indices(cls, encoder: str) -> List[int]:
        if encoder.startswith("tresnet"):
            return None
        return [i for i in range(5)]

    def out_channels(self) -> int:
        return self.channels

    def out_reduction(self) -> int:
        return self.reduction

    def forward(self, features: Iterable[torch.Tensor]) -> torch.Tensor:
        # features = x1, x2, x3, x4, x5
        x, skips = features[-1], features[:-1]
        # we now start from the bottom and combine x with x4, x3, x2 ...
        for module, feature in zip(self.blocks, reversed(skips)):
            x = module(x, feature)
        return self.dropout(x)


class UNetPlusPlusConvBlock(nn.Module):
    """Two-convolution refinement block used by the nested U-Net++ decoder.

    The block keeps the implementation independent of segmentation-models-pytorch
    while matching the main U-Net decoder interface used by this repository.
    """

    def __init__(self, in_channels: int, out_channels: int, act_layer: Type[nn.Module],
                 norm_layer: Type[nn.Module]):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            act_layer(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            act_layer(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetPlusPlus(Decoder):
    """Nested U-Net++ decoder with dense skip connections.

    This decoder uses the same timm encoder features as the standard U-Net
    decoder, but it repeatedly refines each skip level by concatenating earlier
    decoder nodes at the same spatial scale with an upsampled deeper node. It is
    intended as a controlled architecture comparison: preprocessing, loss,
    sampler, optimiser and evaluation can remain unchanged while only the
    decoder changes.
    """

    def __init__(self,
                 input_size: int,
                 feature_channels: List[int],
                 feature_reductions: List[int],
                 act_layer: Type[nn.Module],
                 norm_layer: Type[nn.Module],
                 num_classes: int = None,
                 drop_channels: bool = False,
                 dropout_prob: float = 0.5):
        super().__init__(input_size, feature_channels, feature_reductions, act_layer, norm_layer)
        if len(feature_channels) < 2:
            raise ValueError("UNetPlusPlus requires at least two encoder feature levels")
        self.feature_channels = list(feature_channels)
        self.feature_reductions = list(feature_reductions)
        self.depth = len(feature_channels)
        self.blocks = nn.ModuleDict()

        # node_channels[(i, j)] stores channels for x_i_j. j=0 are raw encoder features.
        node_channels = {(i, 0): int(ch) for i, ch in enumerate(feature_channels)}
        for j in range(1, self.depth):
            for i in range(self.depth - j):
                same_scale_channels = sum(node_channels[(i, k)] for k in range(j))
                deeper_channels = node_channels[(i + 1, j - 1)]
                in_channels = same_scale_channels + deeper_channels
                out_channels = int(feature_channels[i])
                self.blocks[f"x{i}_{j}"] = UNetPlusPlusConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                node_channels[(i, j)] = out_channels

        drop_class = nn.Dropout2d if drop_channels else nn.Dropout
        self.dropout = drop_class(p=dropout_prob)
        self.channels = int(feature_channels[0])
        self.reduction = int(feature_reductions[0])

    @classmethod
    def required_indices(cls, encoder: str) -> List[int]:
        if encoder.startswith("tresnet"):
            return None
        return [i for i in range(5)]

    def out_channels(self) -> int:
        return self.channels

    def out_reduction(self) -> int:
        return self.reduction

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return torch.nn.functional.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=True)

    def forward(self, features: Iterable[torch.Tensor]) -> torch.Tensor:
        nodes = {(i, 0): feature for i, feature in enumerate(features)}
        for j in range(1, self.depth):
            for i in range(self.depth - j):
                same_scale = [nodes[(i, k)] for k in range(j)]
                deeper = self._resize_like(nodes[(i + 1, j - 1)], same_scale[0])
                x = torch.cat([*same_scale, deeper], dim=1)
                nodes[(i, j)] = self.blocks[f"x{i}_{j}"](x)
        return self.dropout(nodes[(0, self.depth - 1)])
