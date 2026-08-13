import pytest
import torch
from torch import nn

from floods.checkpoint_adaptation import adapt_input_state_dict


def test_zero_extra_channel_adaptation_preserves_old_channels_and_zeros_new():
    source_model = nn.Conv2d(3, 4, kernel_size=3, bias=False)
    target_model = nn.Conv2d(6, 4, kernel_size=3, bias=False)
    with torch.no_grad():
        source_model.weight.copy_(torch.arange(source_model.weight.numel()).reshape_as(source_model.weight))

    adapted, keys = adapt_input_state_dict(target_model, source_model.state_dict(), mode="zero_extra")

    assert keys == ["weight"]
    np_weight = adapted["weight"]
    torch.testing.assert_close(np_weight[:, :3], source_model.weight)
    torch.testing.assert_close(np_weight[:, 3:], torch.zeros_like(np_weight[:, 3:]))


def test_zero_extra_rejects_unrelated_shape_changes():
    source_model = nn.Conv2d(3, 4, kernel_size=3, bias=False)
    target_model = nn.Conv2d(6, 5, kernel_size=3, bias=False)
    with pytest.raises(RuntimeError, match="unsupported shape mismatches"):
        adapt_input_state_dict(target_model, source_model.state_dict(), mode="zero_extra")
