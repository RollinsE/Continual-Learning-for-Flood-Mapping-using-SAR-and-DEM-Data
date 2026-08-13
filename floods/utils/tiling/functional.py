from pathlib import Path
from typing import Generator, List, Optional, Union

import numpy as np
from floods.utils.console import progress_iter

from floods.utils.gis import imread
from floods.utils.ml import entropy


def tile_overlapped(image: np.ndarray,
                    tile_size: Union[tuple, int],
                    overlap_threshold: int,
                    channels_first: bool = False) -> Generator[tuple, None, None]:
    """Generates a set of tiles with dynamically computed overlap, so that every tile is contained inside the image
    bounds.

    Args:
        image (np.ndarray): input image to be tiled.
        tile_size (Union[tuple, int], optional): size of the tile in pixels, assuming a square tile. Defaults to 256.
        overlap_threshold (int): Maximum allowed overlap before dropping a neighbouring tile.
        channels_first (bool, optional): whether the image has CxHxW format or HxWxC. Defaults to False.

    Raises:
        ValueError: when the image is smaller than a single tile

    Returns:
        Generator[int, int, int, int]: x, y coordinates with x and y offsets to crop windows
    """
    if len(image.shape) == 2:
        axis = 0 if channels_first else -1
        image = np.expand_dims(image, axis=axis)
    if channels_first:
        image = np.moveaxis(image, 0, -1)
    # From this point onward, data is handled as height, width, channels.
    height, width, channels = image.shape
    tile_h, tile_w = tile_size if isinstance(tile_size, tuple) else (tile_size, tile_size)
    if tile_h <= 0 or tile_w <= 0:
        raise ValueError(f"tile_size must contain positive dimensions, got {tile_size}")
    if overlap_threshold < 0:
        raise ValueError(f"overlap_threshold must be non-negative, got {overlap_threshold}")
    # Pad dimensions virtually when the image is smaller than a single tile.
    if height <= tile_h or width <= tile_w:
        height = max(height, tile_h)
        width = max(width, tile_w)
    # Number of tiles required on each axis.
    tile_count_h = int(np.ceil(height / tile_h))
    tile_count_w = int(np.ceil(width / tile_w))
    # Remainder after covering the image with non-overlapping tiles.
    remainder_h = (tile_count_h * tile_h) - height
    remainder_w = (tile_count_w * tile_w) - width
    # Divide the remainder across tiles as overlap.
    overlap_h = int(np.floor(remainder_h / float(tile_count_h - 1))) if tile_count_h > 1 else 0
    overlap_w = int(np.floor(remainder_w / float(tile_count_w - 1))) if tile_count_w > 1 else 0
    # Avoid near-duplicate edge tiles when the overlap would be too large.
    offset_h, offset_w = 0, 0
    if overlap_h >= overlap_threshold:
        tile_count_h -= 1
        overlap_h = 0
        offset_h = (height - (tile_count_h * tile_h)) // 2
    if overlap_w >= overlap_threshold:
        tile_count_w -= 1
        overlap_w = 0
        offset_w = (width - (tile_count_w * tile_w)) // 2

    # Compute crop windows row by row.
    for row in range(tile_count_h):
        for col in range(tile_count_w):
            # Starting indices include overlap and centering offsets.
            x = max(row * tile_h - overlap_h, 0) + offset_h
            y = max(col * tile_w - overlap_w, 0) + offset_w
            # Shift edge windows back inside image bounds.
            if (x + tile_h) >= height:
                x -= abs(x + tile_h - height)
            if (y + tile_w) >= width:
                y -= abs(y + tile_w - width)
            yield (row, col), (x, y, x + tile_h, y + tile_w)


def tile_fixed_overlap(image: np.ndarray,
                       tile_size: Union[tuple, int],
                       overlap: int,
                       channels_first: bool = False) -> Generator[tuple, None, None]:
    if len(image.shape) == 2:
        axis = 0 if channels_first else -1
        image = np.expand_dims(image, axis=axis)
    if channels_first:
        image = np.moveaxis(image, 0, -1)
    tile_dims = tile_size if isinstance(tile_size, tuple) else (tile_size, tile_size)
    tile_h, tile_w = tile_dims
    if tile_h <= 0 or tile_w <= 0:
        raise ValueError(f"tile_size must contain positive dimensions, got {tile_size}")
    if overlap < 0 or overlap >= min(tile_dims):
        raise ValueError(
            f"overlap must be non-negative and smaller than both tile dimensions, got {overlap}"
        )
    # Handle channels-last arrays from this point onward.
    height, width, _ = image.shape
    strides = [t - overlap for t in tile_dims]
    tiles_x, tiles_y = [int(np.ceil(dim / float(s))) for dim, s in zip((height, width), strides)]

    for tile_x in range(tiles_x):
        for tile_y in range(tiles_y):
            x = tile_x * strides[0]
            y = tile_y * strides[1]
            yield (tile_x, tile_y), (x, y, x + tile_h, y + tile_w)


def tile_body_water_ratio(image: np.ndarray, label_index: int = 1, smoothing: Optional[float] = 0.0) -> float:
    """
    Computes the body water ratio from the given image, with a smoothing factor if required.
    The smoothing factor should be between 0 and 1, given it multiplies the largest
    """
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D mask array, got shape {image.shape}")
    if image.shape[0] != 1:
        raise ValueError(f"Expected a single-channel mask, got shape {image.shape}")
    if not 0 <= smoothing <= 1:
        raise ValueError(f"smoothing must be between 0 and 1, got {smoothing}")

    flat = np.asarray(image).reshape(-1)
    valid = ~np.isnan(flat)
    valid &= flat != 255
    valid_pixels = int(np.count_nonzero(valid))
    if valid_pixels == 0:
        return 0.0
    flood_pixels = int(np.count_nonzero(flat[valid] == label_index))
    factor = smoothing * valid_pixels
    return (flood_pixels + factor) / (valid_pixels + factor)


def mask_body_ratio_from_threshold(labels: List[Path], ratio_threshold: float, label: str,
                                   cache_hash: str, cache_dir: Union[str, Path] = "data/cache",
                                   force_recompute: bool = False) -> np.ndarray:
    """
    Return a boolean mask selecting tiles above the configured water-body ratio.
    """
    if not labels:
        raise ValueError("No mask tiles were provided")
    target_file = Path(cache_dir) / f"mask_{label}_{cache_hash}.npy"
    target_file.parent.mkdir(parents=True, exist_ok=True)

    # Reuse cached filtering decisions when available.
    if target_file.exists() and target_file.is_file() and not force_recompute:
        mask = np.load(str(target_file))
        counts = np.bincount(mask.astype(np.uint8), minlength=2)
        return mask, counts

    # Build filtering decisions from each label tile.
    mask = np.zeros(len(labels), dtype=bool)
    for i, label_path in enumerate(progress_iter(labels, desc="Scanning mask tiles", unit="tile")):
        image = imread(label_path)
        ratio = tile_body_water_ratio(image)
        mask[i] = ratio >= ratio_threshold

    counts = np.bincount(mask.astype(np.uint8), minlength=2)
    # Cache the mask for repeated training runs with the same effective settings.
    np.save(str(target_file), mask)
    return mask, counts




def foreground_ratios_from_labels(labels: List[Path], cache_hash: str,
                                  cache_dir: Union[str, Path] = "data/cache",
                                  force_recompute: bool = False) -> np.ndarray:
    """Return foreground ratios for a list of mask tiles, using a cache when possible."""
    if not labels:
        raise ValueError("No mask tiles were provided")
    target_file = Path(cache_dir) / f"foreground-ratios_{cache_hash}.npy"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    if target_file.exists() and target_file.is_file() and not force_recompute:
        ratios = np.load(str(target_file))
        if len(ratios) == len(labels):
            return ratios
    ratios = np.zeros(len(labels), dtype=np.float32)
    for i, label_path in enumerate(progress_iter(labels, desc="Scanning mask tiles", unit="tile")):
        image = imread(label_path)
        ratios[i] = tile_body_water_ratio(image)
    np.save(str(target_file), ratios)
    return ratios

def weights_from_body_ratio(labels: List[Path], normalize: bool = True, smoothing: Optional[float] = 1.0) -> np.ndarray:
    """Computes sample weights from the body/water ratio (direct proportionality).

    Args:
        labels (List[Path]): list of files containing the masks
        normalize (bool, optional): whether to normalize outputs or not. Defaults to True.
        smoothing (Optional[float], optional): A factor to smooth out probabilities. Defaults to 1.0.

    Returns:
        np.ndarray: array containing floats, one for each sample, to be used in importance sampling
    """
    # Compute raw flooded coverage for each tile.
    # Normalise the resulting sample weights when requested.
    weights = np.zeros(len(labels), dtype=np.float32)
    for i, label_path in enumerate(progress_iter(labels, desc="Scanning mask tiles", unit="tile")):
        image = imread(label_path)
        weights[i] = tile_body_water_ratio(image, smoothing=smoothing)
    if normalize:
        weights /= weights.max()
    return weights


def entropy_weights(labels: List[Path], smoothing: float = 0.8) -> np.ndarray:
    """Computes the entropy from the given list of labels (binary labels).

    Args:
        labels (List[Path]): list of filenames to be read
        smoothing (float, optional): Value to smooth out the final array. Defaults to 0.8.

    Returns:
        np.ndarray: array of smoothed entropy values (max = 1.0, min = 0.0)
    """
    if not 0 <= smoothing <= 1:
        raise ValueError(f"smoothing must be between 0 and 1, got {smoothing}")
    minval = 1.0 - smoothing
    entropies = list()
    for label_path in progress_iter(labels, desc="Scanning mask entropy", unit="tile"):
        label = imread(label_path)
        entropies.append(entropy(label))

    entropies = np.array(entropies)
    return np.clip(entropies * smoothing + minval, 0, 1)
