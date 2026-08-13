import pytest
import torch

from floods.models.architectures.deeplabv3_plus import DeepLab, Xception
from floods.models.architectures.unet import UNet


def test_legacy_unet_uses_professional_block_names_and_preserves_output_shape():
    model = UNet(num_classes=1, in_channels=2).eval()
    image = torch.randn(1, 2, 65, 67)
    with torch.no_grad():
        output, auxiliary = model(image)
    assert output.shape == (1, 1, 65, 67)
    assert auxiliary is None


def test_deeplab_resnet34_supports_two_channel_input_without_pretrained_download():
    model = DeepLab(
        num_classes=1,
        in_channels=2,
        backbone="resnet34",
        pretrained=False,
        output_stride=16,
    ).eval()
    image = torch.randn(1, 2, 64, 64)
    with torch.no_grad():
        output, auxiliary = model(image)
    assert output.shape == (1, 1, 64, 64)
    assert auxiliary is None


def test_xception_middle_flow_blocks_are_registered():
    model = Xception(output_stride=16, in_channels=2, pretrained=False)
    for block_index in range(4, 20):
        assert hasattr(model, f"block{block_index}")


def test_deeplab_rejects_unknown_backbone():
    with pytest.raises(ValueError, match="Unsupported DeepLab backbone"):
        DeepLab(num_classes=1, backbone="not-a-backbone", pretrained=False)
