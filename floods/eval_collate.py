from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch


def _as_image_tensor(value) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor with shape [C,H,W] or [H,W], got {tuple(tensor.shape)}")
    return tensor.float()


def _as_mask_tensor(value) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"Expected mask tensor with shape [H,W] or [1,H,W], got {tuple(tensor.shape)}")
    return tensor.long()


def _ceil_to_multiple(value: int, divisor: int) -> int:
    if divisor <= 1:
        return int(value)
    return int(math.ceil(float(value) / float(divisor)) * divisor)


def pad_segmentation_batch(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
                           image_pad_value: float = 0.0,
                           mask_pad_value: int = 255,
                           size_divisor: int = 32):
    """Pad variable-sized segmentation samples so they can be batched safely.

    Some MMFlood splits may contain full-scene rasters or variable-sized tiles.
    PyTorch's default collate stacks tensors directly and fails when spatial
    sizes differ. This collate pads images to the largest height/width in the
    batch, rounded up to ``size_divisor`` for encoder-decoder models. Mask
    padding uses 255 so existing metrics ignore padded pixels.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    if len(batch[0]) < 2:
        raise ValueError("Expected dataset samples to contain at least image and mask")

    images = [_as_image_tensor(sample[0]) for sample in batch]
    masks = [_as_mask_tensor(sample[1]) for sample in batch]
    channels = images[0].shape[0]
    for idx, image in enumerate(images):
        if image.shape[0] != channels:
            raise ValueError(f"All images in a batch must have the same channel count; entry 0 has {channels}, entry {idx} has {image.shape[0]}")

    max_h = max(int(image.shape[-2]) for image in images)
    max_w = max(int(image.shape[-1]) for image in images)
    out_h = _ceil_to_multiple(max_h, size_divisor)
    out_w = _ceil_to_multiple(max_w, size_divisor)

    image_batch = images[0].new_full((len(images), channels, out_h, out_w), float(image_pad_value))
    mask_batch = masks[0].new_full((len(masks), out_h, out_w), int(mask_pad_value))

    for idx, (image, mask) in enumerate(zip(images, masks)):
        h, w = int(image.shape[-2]), int(image.shape[-1])
        if int(mask.shape[-2]) != h or int(mask.shape[-1]) != w:
            raise ValueError(
                f"Image/mask size mismatch at batch entry {idx}: image={tuple(image.shape)}, mask={tuple(mask.shape)}"
            )
        image_batch[idx, :, :h, :w] = image
        mask_batch[idx, :h, :w] = mask

    if len(batch[0]) <= 2:
        return image_batch, mask_batch

    extra_columns = []
    for extra_idx in range(2, len(batch[0])):
        values = [sample[extra_idx] for sample in batch]
        if all(isinstance(v, (int,)) for v in values):
            extra_columns.append(torch.as_tensor(values, dtype=torch.long))
        elif all(torch.is_tensor(v) for v in values):
            try:
                extra_columns.append(torch.stack(values))
            except RuntimeError:
                extra_columns.append(values)
        else:
            extra_columns.append(values)

    return (image_batch, mask_batch, *extra_columns)
