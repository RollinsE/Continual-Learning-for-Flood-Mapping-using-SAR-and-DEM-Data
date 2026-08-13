from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from torch import Tensor

from floods.utils.ml import identity


def plot_confusion_matrix(cm: np.ndarray,
                          destination: Path,
                          labels: List[str],
                          title: str = "confusion matrix",
                          normalize: bool = True) -> None:
    """Utility function that plots a confusion matrix and stores it as PNG in the given destination folder.

    Args:
        cm (np.ndarray): confusion matrix as square array
        destination (Path): path to the destination, it must point to an image file
        labels (List[str]): labels for the axes
        title (str, optional): Title to assign to the plot. Defaults to "confusion matrix".
        normalize (bool, optional): normalize values. Defaults to True.
    """
    fig = plt.figure(figsize=(6, 6))
    if normalize:
        cm /= cm.max()
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(title)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    fig.savefig(str(destination))


def mask_to_rgb(mask: np.ndarray, palette: Dict[int, tuple]) -> np.ndarray:
    """Given an input batch, or single picture with dimensions [B, H, W] or [H, W], the utility generates
    an equivalent [B, H, W, 3] or [H, W, 3] array corresponding to an RGB version.
    The conversion uses the given palette, which should be provided as simple dictionary of indices and tuples, lists
    or arrays indicating a single RGB color. (e.g. {0: (255, 255, 255)})
    Args:
        mask (np.ndarray): input mask of indices. Each index should be present in the palette
        palette (Dict[int, tuple]): dictionary of pairs <index - color>, where colors can be provided in RGB tuple fmt
    Returns:
        np.ndarray: tensor containing the RGB version of the input index tensor
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for index, color in palette.items():
        lut[index] = np.array(color, dtype=np.uint8)
    return lut[mask]


def make_grid(inputs: np.ndarray, rgb_true: np.ndarray, rgb_pred: np.ndarray) -> np.ndarray:
    if inputs.ndim != 3:
        raise ValueError(f"Expected one channels-last RGB image, got shape {inputs.shape}")
    if not (inputs.shape == rgb_true.shape == rgb_pred.shape):
        raise ValueError(f"Grid image shapes do not match: {inputs.shape}, {rgb_true.shape}, {rgb_pred.shape}")
    return np.concatenate((inputs, rgb_true, rgb_pred), axis=1).astype(np.uint8)


def save_grid(inputs: Tensor,
              targets: Tensor,
              preds: Tensor,
              filepath: Path,
              filename: str,
              palette: Dict[int, tuple],
              offset: int = 0,
              inverse_transform: Callable = identity,
              image_transform: Callable = None) -> None:
    if targets.shape != preds.shape:
        raise ValueError(f"Target/prediction shape mismatch: {targets.shape} != {preds.shape}")
    if inputs.ndim < 3:
        raise ValueError(f"Expected channels-first image tensor, got shape {inputs.shape}")
    if inputs.ndim == 4:
        for i in range(inputs.shape[0]):
            save_grid(inputs[i], targets[i], preds[i], filepath=filepath, filename=filename, palette=palette, offset=i)
    else:
        image = inverse_transform(inputs)
        if image_transform is not None:
            image = image_transform(image)
        # Targets and predictions still include a channel dimension.
        if targets.ndim > 2:
            targets = targets.squeeze(0)
            preds = preds.squeeze(0)
        rgb_true = mask_to_rgb(targets.cpu().numpy(), palette=palette)
        rgb_pred = mask_to_rgb(preds.cpu().numpy(), palette=palette)
        grid = make_grid(image, rgb_true, rgb_pred)
        plt.imsave(filepath / f"{filename}-{offset}.png", grid)
