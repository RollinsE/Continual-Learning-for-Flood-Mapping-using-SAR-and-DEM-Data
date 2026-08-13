from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from timm.models.features import FeatureInfo
from torch import nn

from floods.models.base import Encoder


def _feature_info(channels: Sequence[int], reductions: Sequence[int]) -> FeatureInfo:
    records = [
        {"num_chs": int(ch), "reduction": int(red), "module": f"foundation.level{idx}"}
        for idx, (ch, red) in enumerate(zip(channels, reductions))
    ]
    return FeatureInfo(records, out_indices=tuple(range(len(records))))


def _square_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Convert [B,N,C] tokens to [B,C,H,W], dropping a prefix token if needed."""
    if tokens.ndim == 4:
        # Some providers already expose channels-first maps.
        if tokens.shape[1] < tokens.shape[-1] and tokens.shape[-1] > 32:
            return tokens.permute(0, 3, 1, 2).contiguous()
        return tokens
    if tokens.ndim != 3:
        raise ValueError(f"Expected provider tokens [B,N,C], got {tuple(tokens.shape)}")
    count = int(tokens.shape[1])
    side = int(round(math.sqrt(count)))
    if side * side != count:
        # timm ViTs commonly prepend one CLS token.
        count_without_prefix = count - 1
        side = int(round(math.sqrt(count_without_prefix)))
        if side * side != count_without_prefix:
            raise ValueError(f"Cannot reshape {count} provider tokens into a square feature map")
        tokens = tokens[:, 1:, :]
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side).contiguous()


def _pyramid_from_maps(maps: Sequence[torch.Tensor], input_size: int) -> list[torch.Tensor]:
    if len(maps) != 4:
        raise ValueError(f"Foundation feature pyramid requires four levels, got {len(maps)}")
    sizes = [max(1, int(input_size) // value) for value in (4, 8, 16, 32)]
    return [
        value if value.shape[-2:] == (size, size) else F.interpolate(
            value, size=(size, size), mode="bilinear", align_corners=False
        )
        for value, size in zip(maps, sizes)
    ]


def _select_four(values: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    if len(values) < 4:
        raise ValueError(f"Provider backbone returned only {len(values)} intermediate layers")
    last = len(values) - 1
    indices = sorted({int(round(last * fraction)) for fraction in (0.25, 0.50, 0.75, 1.0)})
    while len(indices) < 4:
        for index in range(len(values)):
            if index not in indices:
                indices.insert(0, index)
                if len(indices) == 4:
                    break
    return [values[index] for index in indices[-4:]]


class TerraMindFeatureEncoder(Encoder):
    """TerraMind raw-modality ViT adapted to the project's four-level decoder API."""

    def __init__(
        self,
        model_name: str,
        modalities: Sequence[str],
        *,
        input_size: int = 224,
        pretrained: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        try:
            from terratorch import BACKBONE_REGISTRY
        except ImportError:  # pragma: no cover - compatibility with older TerraTorch
            from terratorch.registry import BACKBONE_REGISTRY

        self.input_modalities = [str(value).lower() for value in modalities]
        terra_modalities = ["S1GRD"] + (["DEM"] if "dem" in self.input_modalities else [])
        self.input_size = int(input_size)
        try:
            self.backbone = BACKBONE_REGISTRY.build(
                model_name,
                pretrained=bool(pretrained),
                modalities=terra_modalities,
                img_size=self.input_size,
                merge_method="mean",
            )
        except TypeError:
            # TerraTorch releases before img_size became an explicit registry option.
            self.backbone = BACKBONE_REGISTRY.build(
                model_name,
                pretrained=bool(pretrained),
                modalities=terra_modalities,
                merge_method="mean",
            )
        channels = list(getattr(self.backbone, "out_channels", []))
        if not channels:
            dim = int(getattr(self.backbone, "dim", 768))
            depth = len(getattr(self.backbone, "encoder", [])) or 12
            channels = [dim] * depth
        # Select four transformer depths for the hierarchical decoder.
        depth = len(channels)
        indices = [max(0, int(round((level + 1) * depth / 4.0)) - 1) for level in range(4)]
        self._indices = indices
        selected_channels = [int(channels[index]) for index in self._indices]
        self._feature_info = _feature_info(selected_channels, (4, 8, 16, 32))
        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    @property
    def feature_info(self) -> FeatureInfo:
        return self._feature_info

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        index = {name: idx for idx, name in enumerate(self.input_modalities)}
        payload = {"S1GRD": x[:, [index["vv"], index["vh"]], ...]}
        if "dem" in index:
            payload["DEM"] = x[:, [index["dem"]], ...]
        values = self.backbone(payload)
        if isinstance(values, dict):
            values = list(values.values())
        if not isinstance(values, (list, tuple)):
            values = [values]
        selected = [values[index] for index in self._indices]
        maps = [_square_tokens(value) for value in selected]
        return _pyramid_from_maps(maps, self.input_size)


class TorchGeoViTFeatureEncoder(Encoder):
    """TorchGeo Sentinel-1 ViT weights with a lightweight four-scale feature adapter."""

    def __init__(self, model_name: str, weight_name: str, *, input_size: int = 224,
                 pretrained: bool = True, freeze: bool = False) -> None:
        super().__init__()
        import timm
        from torchgeo import models as tg_models

        self.input_size = int(input_size)
        self.model = timm.create_model(model_name, pretrained=False, in_chans=2, num_classes=0)
        if pretrained:
            enum_name = "ViTSmall16_Weights" if "small" in model_name else "ViTBase16_Weights"
            weight_enum = getattr(tg_models, enum_name)
            weights = getattr(weight_enum, weight_name)
            state = weights.get_state_dict(progress=True)
            self.model.load_state_dict(state, strict=False)
        dim = int(getattr(self.model, "embed_dim", getattr(self.model, "num_features", 768)))
        depth = len(getattr(self.model, "blocks", []))
        if depth < 4:
            raise ValueError(f"Unsupported timm ViT without four transformer blocks: {model_name}")
        self._indices = [max(0, int(round((level + 1) * depth / 4.0)) - 1) for level in range(4)]
        self._feature_info = _feature_info([dim] * 4, (4, 8, 16, 32))
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    @property
    def feature_info(self) -> FeatureInfo:
        return self._feature_info

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = F.interpolate(x[:, :2], size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        model = self.model
        x = model.patch_embed(x)
        if hasattr(model, "_pos_embed"):
            x = model._pos_embed(x)
        else:  # pragma: no cover - older timm fallback
            if getattr(model, "cls_token", None) is not None:
                cls = model.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls, x), dim=1)
            x = model.pos_drop(x + model.pos_embed)
        if hasattr(model, "patch_drop"):
            x = model.patch_drop(x)
        if hasattr(model, "norm_pre"):
            x = model.norm_pre(x)
        values = []
        for index, block in enumerate(model.blocks):
            x = block(x)
            if index in self._indices:
                values.append(x)
        if hasattr(model, "norm") and values:
            values[-1] = model.norm(values[-1])
        return _pyramid_from_maps([_square_tokens(value) for value in values], self.input_size)


class CROMAFeatureEncoder(Encoder):
    """CROMA SAR encoder adapted from its patch tokens to four decoder levels."""

    def __init__(self, *, input_size: int = 120, pretrained: bool = True, freeze: bool = False) -> None:
        super().__init__()
        from torchgeo.models import CROMABase_Weights, croma_base

        weights = CROMABase_Weights.CROMA_VIT if pretrained else None
        self.input_size = int(input_size)
        self.model = croma_base(weights=weights, modalities=["sar"], image_size=self.input_size)
        dim = int(getattr(self.model, "encoder_dim", 768))
        self._feature_info = _feature_info([dim] * 4, (4, 8, 16, 32))
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    @property
    def feature_info(self) -> FeatureInfo:
        return self._feature_info

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = F.interpolate(x[:, :2], size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        result = self.model(x_sar=x)
        tokens = result["sar_encodings"]
        base = _square_tokens(tokens)
        return _pyramid_from_maps([base, base, base, base], self.input_size)
