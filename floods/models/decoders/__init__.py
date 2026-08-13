from functools import partial

from floods.models.decoders.deeplab import DeepLabV3, DeepLabV3Plus
from floods.models.decoders.pspnet import PSPNet
from floods.models.decoders.unet import UNet, UNetPlusPlus
from floods.models.decoders.segformer import SegFormerDecoder

__all__ = ["DeepLabV3", "DeepLabV3Plus", "UNet", "UNetPlusPlus", "PSPNet", "SegFormerDecoder"]

available_decoders = {
    "unet": partial(UNet, bilinear=True),
    "unetpp": partial(UNetPlusPlus),
    "pspnet": partial(PSPNet),
    "deeplabv3": partial(DeepLabV3),
    "deeplabv3p": partial(DeepLabV3Plus),
    "segformer": partial(SegFormerDecoder)
}
