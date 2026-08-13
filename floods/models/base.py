from abc import abstractclassmethod, abstractmethod
from itertools import chain
from typing import Iterator, List, Tuple, Type

import torch
from torch import nn

try:
    from timm.models.features import FeatureInfo
except ImportError:  # pragma: no cover - keeps lightweight config/CLI imports usable
    class FeatureInfo:  # type: ignore[no-redef]
        pass


class Encoder(nn.Module):
    @property
    @abstractmethod
    def feature_info(self) -> FeatureInfo:
        ...


class Decoder(nn.Module):
    @abstractmethod
    def __init__(self, input_size: int, feature_channels: List[int], feature_reductions: List[int],
                 act_layer: Type[nn.Module], norm_layer: Type[nn.Module]):
        super().__init__()

    @abstractclassmethod
    def required_indices(cls, encoder: str) -> List[int]:
        ...

    @abstractmethod
    def out_channels(self) -> int:
        ...

    @abstractmethod
    def out_reduction(self) -> int:
        ...


class Head(nn.Module):
    @abstractmethod
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels


class Segmenter(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, head: Head, return_features: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.head = head
        self.return_features = return_features

    def forward(self, x: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        input_size = x.shape[-2:]
        encoded = self.encoder(x)
        features = self.decoder(encoded)
        out = self.head(features)
        if out.shape[-2:] != input_size:
            out = torch.nn.functional.interpolate(
                out.unsqueeze(1), size=input_size, mode="bilinear", align_corners=False
            ).squeeze(1)
        return (out, features) if self.return_features else out

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def encoder_params(self) -> Iterator[nn.Parameter]:
        return self.encoder.parameters()

    def decoder_params(self) -> Iterator[nn.Parameter]:
        return chain(self.decoder.parameters(), self.head.parameters())


class MultiBranchSegmenter(Segmenter):
    def __init__(self, encoder: Encoder, decoder: Decoder, head: Head, auxiliary: Head, return_features: bool = False):
        super().__init__(encoder, decoder, head, return_features=return_features)
        self.auxiliary = auxiliary

    def forward(self, x: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        x: Tuple[torch.Tensor] = self.encoder(x)
        features = self.decoder(x)
        aux = self.auxiliary(x[-1])
        out = self.head(features)
        if self.return_features:
            return (out, aux), features
        else:
            return out, aux
