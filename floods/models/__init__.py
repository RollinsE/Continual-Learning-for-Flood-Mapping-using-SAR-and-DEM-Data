from typing import TYPE_CHECKING, Type

from torch import nn

from floods.config import ModelConfig
from floods.models.base import Decoder, Encoder

if TYPE_CHECKING:  # pragma: no cover
    from timm.models.features import FeatureInfo
    from floods.models.encoders import MultiEncoder


def resolve_layer_factory(layer):
    """Return the callable layer factory expected by timm and decoder modules."""
    return layer.value if hasattr(layer, "value") else layer


def filter_encoder_args(encoder: str, pretrained: bool, **kwargs: dict) -> dict:
    """Remove constructor arguments unsupported by specific timm encoder families."""
    exclude = set()
    # DenseNets do not expose output-stride control.
    if encoder.startswith("dense"):
        exclude = exclude.union(["output_stride"])
    # TResNets use fixed activation and normalization internals.
    if encoder.startswith("tresnet"):
        exclude = exclude.union(["norm_layer", "act_layer", "output_stride"])
    # EfficientNets use their own activation and normalization configuration.
    if encoder.startswith("efficientnet") or encoder.startswith("mobilenet") or pretrained:
        exclude = exclude.union(["norm_layer", "act_layer"])
    if encoder.startswith(("mit_", "pvt_", "swin", "convnext", "efficientformer")):
        exclude = exclude.union(["norm_layer", "act_layer", "output_stride"])
    for arg in exclude:
        kwargs.pop(arg, None)
    return kwargs


def create_encoder(name: str,
                   decoder: str,
                   pretrained: bool,
                   freeze: bool,
                   output_stride: int,
                   act_layer: Type[nn.Module],
                   norm_layer: Type[nn.Module],
                   channels: int = 3,
                   **kwargs) -> Encoder:
    act_layer = resolve_layer_factory(act_layer)
    norm_layer = resolve_layer_factory(norm_layer)

    import timm
    from floods.models.decoders import available_decoders
    from floods.models.encoders import available_encoders

    # Validate model names before instantiation.
    if name not in available_encoders:
        raise ValueError(f"Encoder not supported: {name}")
    if decoder not in available_decoders:
        raise ValueError(f"Decoder not supported: {decoder}")
    # Build encoder-specific keyword arguments.
    additional_args = kwargs or {}
    additional_args.update(act_layer=act_layer, norm_layer=norm_layer, output_stride=output_stride)
    additional_args = filter_encoder_args(encoder=name, pretrained=pretrained, **additional_args)
    additional_args.update(in_chans=channels)
    # Select intermediate feature indices required by the decoder.
    indices = available_decoders[decoder].func.required_indices(encoder=name)
    if (name.startswith("dense")):
        additional_args.pop('act_layer')
    model = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=indices, **additional_args)
    # Freeze encoder parameters when requested.
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model


def create_decoder(name: str, input_size: int, feature_info: "FeatureInfo", act_layer: Type[nn.Module],
                   norm_layer: Type[nn.Module], **kwargs: dict) -> Decoder:
    act_layer = resolve_layer_factory(act_layer)
    norm_layer = resolve_layer_factory(norm_layer)

    from floods.models.decoders import available_decoders

    # Validate decoder availability before construction.
    if name not in available_decoders:
        raise ValueError(f"Decoder not implemented: {name}")
    # Instantiate the registered decoder with the shared parameters.
    decoder_class = available_decoders.get(name)
    decoder = decoder_class(input_size=input_size,
                            feature_channels=feature_info.channels(),
                            feature_reductions=feature_info.reduction(),
                            act_layer=act_layer,
                            norm_layer=norm_layer,
                            **kwargs)
    return decoder


def create_multi_encoder(sar_name: str, dem_name: str, channels: int, config: ModelConfig,
                         **kwargs: dict) -> "MultiEncoder":
    from floods.models.encoders import MultiEncoder

    encoder_a = create_encoder(name=sar_name,
                               decoder=config.decoder,
                               pretrained=config.pretrained,
                               freeze=config.freeze,
                               output_stride=config.output_stride,
                               act_layer=config.act,
                               norm_layer=config.norm,
                               channels=channels - 1)
    encoder_b = create_encoder(name=dem_name,
                               decoder=config.decoder,
                               pretrained=False,
                               freeze=False,
                               output_stride=config.output_stride,
                               act_layer=config.act,
                               norm_layer=config.norm,
                               channels=1)
    return MultiEncoder(
        encoder_a,
        encoder_b,
        act_layer=resolve_layer_factory(config.act),
        norm_layer=resolve_layer_factory(config.norm),
        **kwargs,
    )


def create_registered_encoder(config):
    """Build an encoder selected through the central pretrained-weight registry."""
    from floods.pretrained import get_weight_source
    from floods.models.foundation import CROMAFeatureEncoder, TerraMindFeatureEncoder, TorchGeoViTFeatureEncoder

    source = get_weight_source(config.model.weights_source)
    modalities = list(config.data.input_modalities)
    input_size = int(config.model.foundation_input_size or source.foundation_input_size or config.image_size)
    pretrained = bool(config.model.pretrained)
    freeze = bool(config.model.freeze)

    if source.adapter == "terramind":
        return TerraMindFeatureEncoder(
            source.default_encoder,
            modalities,
            input_size=input_size,
            pretrained=pretrained,
            freeze=freeze,
        )
    if source.adapter == "croma_sar_base":
        return CROMAFeatureEncoder(input_size=input_size, pretrained=pretrained, freeze=freeze)
    if source.adapter == "torchgeo_vit_small_fgmae":
        return TorchGeoViTFeatureEncoder(
            "vit_small_patch16_224", "SENTINEL1_GRD_FGMAE",
            input_size=input_size, pretrained=pretrained, freeze=freeze,
        )
    if source.adapter == "torchgeo_vit_base_fgmae":
        return TorchGeoViTFeatureEncoder(
            "vit_base_patch16_224", "SENTINEL1_GRD_FGMAE",
            input_size=input_size, pretrained=pretrained, freeze=freeze,
        )
    if source.adapter in {"torchgeo_resnet50_moco", "torchgeo_resnet50_decur"}:
        from torchgeo.models import ResNet50_Weights
        weight_name = "SENTINEL1_GRD_MOCO" if source.adapter.endswith("moco") else "SENTINEL1_GRD_DECUR"
        encoder = create_encoder(
            name="resnet50",
            decoder=config.model.decoder,
            pretrained=False,
            freeze=False,
            output_stride=config.model.output_stride,
            act_layer=config.model.act,
            norm_layer=config.model.norm,
            channels=2,
        )
        if pretrained:
            state = getattr(ResNet50_Weights, weight_name).get_state_dict(progress=True)
            encoder.load_state_dict(state, strict=False)
        if freeze:
            for parameter in encoder.parameters():
                parameter.requires_grad = False
        return encoder
    raise ValueError(f"No encoder adapter is implemented for weights_source={source.name!r}")
