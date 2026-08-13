"""Model construction shared by training, evaluation, and deployment.

This module deliberately contains no dataset or augmentation imports.  Deployment
must be able to reconstruct a trained network without importing Albumentations or
SciPy, because neither library is required for model architecture creation.
"""

from __future__ import annotations

from torch import nn

from floods.config import TrainConfig
from floods.models import create_decoder, create_encoder, create_multi_encoder, create_registered_encoder
from floods.models.base import MultiBranchSegmenter, Segmenter
from floods.models.modules import SegmentationHead
from floods.utils.common import get_logger

LOG = get_logger(__name__)


def prepare_model(config: TrainConfig, num_classes: int, stage: str = "train") -> nn.Module:
    """Construct a segmentation model from a resolved training configuration.

    The implementation is intentionally independent of data preparation and
    training augmentation so it is safe to import from the deployment runtime.
    """
    cfg = config.model

    # Registered providers and ordinary timm encoders share the same decoder path.
    enc_names = cfg.encoder.split(",")
    if getattr(cfg, "weights_source", "random") not in {"random", "imagenet"}:
        encoder = create_registered_encoder(config)
    elif len(enc_names) == 1:
        encoder = create_encoder(
            name=enc_names[0],
            decoder=cfg.decoder,
            pretrained=cfg.pretrained,
            freeze=cfg.freeze,
            output_stride=cfg.output_stride,
            act_layer=cfg.act,
            norm_layer=cfg.norm,
            channels=config.data.in_channels,
        )
    else:
        # Two encoders are supported: one for SAR channels and one for DEM.
        if len(enc_names) != 2:
            raise ValueError(f"Expected exactly two multimodal encoders, got: {cfg.encoder}")
        if config.data.in_channels < 3:
            raise ValueError("Multimodal SAR+DEM models require at least three input channels")
        LOG.info("Creating a multimodal encoder (%s, %s)", enc_names[0], enc_names[1])
        encoder = create_multi_encoder(
            sar_name=enc_names[0],
            dem_name=enc_names[1],
            channels=config.data.in_channels,
            config=cfg,
            return_features=False,
        )

    additional_args: dict = {}
    if cfg.decoder == "segformer" and getattr(cfg, "weights_source", "random") not in {"random", "imagenet"}:
        additional_args["embed_dim"] = int(getattr(cfg, "foundation_pyramid_channels", 256))

    decoder = create_decoder(
        name=cfg.decoder,
        input_size=config.image_size,
        feature_info=encoder.feature_info,
        act_layer=cfg.act,
        norm_layer=cfg.norm,
        **additional_args,
    )

    extract_features = False
    LOG.debug("Returning intermediate features: %s", str(extract_features))
    head = SegmentationHead(
        in_channels=decoder.out_channels(),
        num_classes=num_classes,
        upscale=decoder.out_reduction(),
    )
    if cfg.multibranch and stage != "test":
        auxiliary = SegmentationHead(
            in_channels=encoder.feature_info.channels()[-1],
            num_classes=num_classes,
            upscale=encoder.feature_info.reduction()[-1],
        )
        return MultiBranchSegmenter(
            encoder,
            decoder,
            head,
            auxiliary=auxiliary,
            return_features=extract_features,
        )
    return Segmenter(encoder, decoder, head, return_features=extract_features)
