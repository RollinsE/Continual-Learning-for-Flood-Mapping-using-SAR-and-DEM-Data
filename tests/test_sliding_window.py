import torch

from floods.sliding_window import sliding_window_logits


class EchoModel(torch.nn.Module):
    def forward(self, x):
        return x[:, :1]


def test_sliding_window_restores_input_shape():
    image = torch.rand(1, 3, 73, 81)
    logits = sliding_window_logits(EchoModel(), image, window_size=32, overlap=8, window_batch_size=2)
    assert tuple(logits.shape) == (1, 73, 81)
