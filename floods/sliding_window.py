from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from floods.evaluation import BinaryThresholdSweep


def main_logits(output: Any) -> torch.Tensor:
    """Return segmentation logits as a ``B x H x W`` tensor."""
    logits = BinaryThresholdSweep._main_prediction(output)
    if logits.ndim == 4 and logits.shape[1] == 1:
        return logits[:, 0]
    if logits.ndim == 4 and logits.shape[1] > 1:
        return logits[:, 1]
    if logits.ndim == 3:
        return logits
    raise ValueError(f"Expected segmentation logits with 3 or 4 dimensions, got shape {tuple(logits.shape)}")


def probability_to_logit(probability: torch.Tensor) -> torch.Tensor:
    """Convert probabilities to logits with finite-value clamping."""
    eps = torch.finfo(probability.dtype).eps
    probability = probability.clamp(min=eps, max=1.0 - eps)
    return torch.log(probability / (1.0 - probability))


def _window_starts(length: int, window_size: int, stride: int) -> list[int]:
    if length <= window_size:
        return [0]
    starts = list(range(0, max(length - window_size + 1, 1), stride))
    last = length - window_size
    if starts[-1] != last:
        starts.append(last)
    return starts



def _blend_weights(window_size: int, mode: str, *, device: torch.device) -> torch.Tensor:
    """Return a positive 2-D blending window for overlap-add inference.

    ``uniform`` reproduces the historical arithmetic mean. ``cosine`` reduces
    the influence of less reliable window borders while retaining a small
    positive floor so outer-scene pixels remain covered.
    """
    mode = str(mode or "uniform").lower().replace("-", "_")
    if mode == "uniform":
        return torch.ones((window_size, window_size), dtype=torch.float32, device=device)
    if mode != "cosine":
        raise ValueError("blend_mode must be 'uniform' or 'cosine'")
    one_d = torch.hann_window(window_size, periodic=False, dtype=torch.float32, device=device)
    one_d = 0.05 + 0.95 * one_d
    return torch.outer(one_d, one_d)

def sliding_window_logits(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    window_size: int = 256,
    overlap: int = 64,
    window_batch_size: int = 4,
    blend_mode: str = "uniform",
) -> torch.Tensor:
    """Run memory-bounded sliding-window inference for one image.

    Args:
        model: Segmentation model returning logits or a structure containing logits.
        image: Tensor with shape ``1 x C x H x W``. Batch size must be one.
        window_size: Square inference-window size in pixels.
        overlap: Number of pixels shared between neighbouring windows.
        window_batch_size: Number of windows forwarded at once.

    Returns:
        A logits tensor with shape ``1 x H x W`` aligned to the input image.
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"sliding_window_logits expects one image with shape 1xCxHxW, got {tuple(image.shape)}")
    window_size = int(window_size)
    overlap = int(overlap)
    window_batch_size = max(int(window_batch_size), 1)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must be >= 0 and smaller than window_size")

    _, _, height, width = image.shape
    pad_h = max(window_size - height, 0)
    pad_w = max(window_size - width, 0)
    if pad_h or pad_w:
        image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    else:
        image_padded = image
    _, _, padded_h, padded_w = image_padded.shape
    stride = max(window_size - overlap, 1)
    rows = _window_starts(padded_h, window_size, stride)
    cols = _window_starts(padded_w, window_size, stride)

    accumulation = torch.zeros((1, padded_h, padded_w), dtype=torch.float32, device=image.device)
    counts = torch.zeros_like(accumulation)
    blend_weights = _blend_weights(window_size, blend_mode, device=image.device).unsqueeze(0)
    windows: list[torch.Tensor] = []
    coords: list[tuple[int, int]] = []

    def flush() -> None:
        nonlocal windows, coords, accumulation, counts
        if not windows:
            return
        batch = torch.cat(windows, dim=0)
        logits = main_logits(model(batch)).float()
        if logits.shape[-2:] != (window_size, window_size):
            logits = F.interpolate(logits.unsqueeze(1), size=(window_size, window_size), mode="bilinear", align_corners=False)[:, 0]
        for idx, (row, col) in enumerate(coords):
            accumulation[:, row:row + window_size, col:col + window_size] += logits[idx:idx + 1] * blend_weights
            counts[:, row:row + window_size, col:col + window_size] += blend_weights
        windows = []
        coords = []

    for row in rows:
        for col in cols:
            windows.append(image_padded[:, :, row:row + window_size, col:col + window_size])
            coords.append((row, col))
            if len(windows) >= window_batch_size:
                flush()
    flush()
    logits = accumulation / counts.clamp_min(1.0)
    return logits[:, :height, :width]


def ensemble_sliding_window_logits(
    models: Sequence[torch.nn.Module],
    image: torch.Tensor,
    *,
    method: str = "mean_prob",
    window_size: int = 256,
    overlap: int = 64,
    window_batch_size: int = 4,
    blend_mode: str = "uniform",
) -> torch.Tensor:
    """Run sliding-window inference for multiple models and combine their outputs."""
    method = str(method or "mean_prob").lower().replace("-", "_")
    if method not in {"mean_prob", "mean_logit"}:
        raise ValueError("method must be 'mean_prob' or 'mean_logit'")
    logits = [
        sliding_window_logits(model, image, window_size=window_size, overlap=overlap, window_batch_size=window_batch_size, blend_mode=blend_mode)
        for model in models
    ]
    if method == "mean_logit":
        return torch.stack(logits, dim=0).mean(dim=0)
    probabilities = torch.stack([torch.sigmoid(item) for item in logits], dim=0).mean(dim=0)
    return probability_to_logit(probabilities)
