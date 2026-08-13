import torch
from floods.eval_collate import pad_segmentation_batch


def test_pad_segmentation_batch_preserves_index_extra():
    batch = [
        (torch.ones(3, 10, 12), torch.zeros(10, 12), 4),
        (torch.ones(3, 8, 9), torch.ones(8, 9), 7),
    ]
    x, y, index = pad_segmentation_batch(batch, size_divisor=8)
    assert tuple(x.shape) == (2, 3, 16, 16)
    assert tuple(y.shape) == (2, 16, 16)
    assert index.tolist() == [4, 7]
    assert int(y[1, 9, 9]) == 255


def test_pad_segmentation_batch_still_returns_two_items_without_extras():
    batch = [
        (torch.ones(2, 5, 6), torch.zeros(5, 6)),
        (torch.ones(2, 4, 6), torch.ones(4, 6)),
    ]
    result = pad_segmentation_batch(batch, size_divisor=4)
    assert len(result) == 2
    x, y = result
    assert tuple(x.shape) == (2, 2, 8, 8)
    assert tuple(y.shape) == (2, 8, 8)
