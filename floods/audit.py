from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from floods.utils.console import progress_iter

from floods.utils.common import get_logger
from floods.utils.gis import imread

LOG = get_logger(__name__)


def _import_pyplot():
    """Import matplotlib in a non-interactive backend suitable for CLI runs."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _event_id_from_name(name: str) -> str:
    import re
    match = re.search(r"(EMSR\d+)", name)
    return match.group(1) if match else "unknown"


def _array_quality(path: Path) -> dict:
    array = imread(path)
    return {
        "shape": tuple(int(v) for v in array.shape),
        "dtype": str(array.dtype),
        "nan_pixels": int(np.count_nonzero(np.isnan(array))) if np.issubdtype(array.dtype, np.floating) else 0,
        "inf_pixels": int(np.count_nonzero(~np.isfinite(array))) if np.issubdtype(array.dtype, np.floating) else 0,
        "min": float(np.nanmin(array)) if array.size else float("nan"),
        "max": float(np.nanmax(array)) if array.size else float("nan"),
    }


def _mask_row(path: Path, processed_data_dir: Path, split: str) -> dict:
    mask = imread(path).squeeze()
    valid = mask != 255
    valid_pixels = int(np.count_nonzero(valid))
    fg_pixels = int(np.count_nonzero((mask == 1) & valid))
    ignore_pixels = int(np.count_nonzero(mask == 255))
    fg_ratio = float(fg_pixels / valid_pixels) if valid_pixels else 0.0
    sar_path = processed_data_dir / split / "sar" / path.name
    dem_path = processed_data_dir / split / "dem" / path.name
    row = {
        "file": path.name,
        "event_id": _event_id_from_name(path.name),
        "fg_pixels": fg_pixels,
        "fg_ratio": fg_ratio,
        "valid_pixels": valid_pixels,
        "ignore_pixels": ignore_pixels,
        "unique_values": tuple(int(v) for v in np.unique(mask)),
        "mask_shape": tuple(int(v) for v in mask.shape),
    }
    if sar_path.exists():
        q = _array_quality(sar_path)
        row.update({f"sar_{k}": v for k, v in q.items()})
    if dem_path.exists():
        q = _array_quality(dem_path)
        row.update({f"dem_{k}": v for k, v in q.items()})
    return row


def audit_split(processed_data_dir: Path, split: str) -> pd.DataFrame:
    mask_dir = processed_data_dir / split / "mask"
    mask_paths = sorted(mask_dir.glob("*.tif"))
    if not mask_paths:
        raise FileNotFoundError(f"No mask tiles found for split '{split}' under {mask_dir}")
    rows = [_mask_row(path, processed_data_dir, split) for path in progress_iter(mask_paths, desc=f"Audit {split}", unit="mask", colour="green")]
    df = pd.DataFrame(rows)
    df.insert(0, "split", split)
    return df


def _write_histogram(df: pd.DataFrame, output_dir: Path) -> None:
    try:
        plt = _import_pyplot()
        plt.figure(figsize=(8, 5))
        for split, group in df.groupby("split"):
            plt.hist(group["fg_ratio"].dropna(), bins=100, alpha=0.55, label=split)
        plt.xlabel("Foreground/flood pixel ratio")
        plt.ylabel("Number of tiles")
        plt.title("Foreground ratio distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "foreground_ratio_distribution.png", dpi=150)
        plt.close()
    except Exception as exc:
        LOG.warning("Foreground histogram was not written: %s", exc)


def _write_overlays(processed_data_dir: Path, output_dir: Path, splits: Iterable[str], samples_per_split: int) -> None:
    if samples_per_split <= 0:
        return
    try:
        plt = _import_pyplot()
        overlay_dir = output_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(1337)
        for split in splits:
            sar_dir = processed_data_dir / split / "sar"
            mask_dir = processed_data_dir / split / "mask"
            sar_paths = sorted(sar_dir.glob("*.tif"))
            if not sar_paths:
                continue
            for sar_path in rng.sample(sar_paths, min(samples_per_split, len(sar_paths))):
                mask_path = mask_dir / sar_path.name
                if not mask_path.exists():
                    continue
                sar = imread(sar_path)
                mask = imread(mask_path).squeeze()
                image = sar[0] if sar.ndim == 3 else sar.squeeze()
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(image, cmap="gray")
                plt.title(f"{split} SAR")
                plt.axis("off")
                plt.subplot(1, 3, 2)
                plt.imshow(mask == 1, cmap="gray")
                plt.title("Mask")
                plt.axis("off")
                plt.subplot(1, 3, 3)
                plt.imshow(image, cmap="gray")
                plt.imshow(mask == 1, alpha=0.35)
                plt.title("Overlay")
                plt.axis("off")
                plt.suptitle(sar_path.name)
                plt.tight_layout()
                plt.savefig(overlay_dir / f"{split}_{sar_path.stem}.png", dpi=130)
                plt.close()
    except Exception as exc:
        LOG.warning("Overlay samples were not written: %s", exc)


def audit_dataset(processed_data_dir: Path, output_dir: Path, splits: List[str], samples_per_split: int = 8, write_plots: bool = True) -> pd.DataFrame:
    processed_data_dir = Path(processed_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = [audit_split(processed_data_dir, split) for split in splits]
    df = pd.concat(frames, ignore_index=True)
    csv_path = output_dir / "foreground_ratio_summary.csv"
    df.to_csv(csv_path, index=False)
    summary = df.groupby("split")["fg_ratio"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    summary_path = output_dir / "foreground_ratio_describe.csv"
    summary.to_csv(summary_path)
    empty = df.assign(is_empty=df["fg_pixels"] == 0).groupby("split")["is_empty"].agg(["sum", "count"])
    empty["empty_percent"] = 100.0 * empty["sum"] / empty["count"]
    empty.to_csv(output_dir / "empty_tile_counts.csv")
    bins = [-1e-12, 0.0, 0.005, 0.02, 0.10, float("inf")]
    labels = ["empty", "tiny", "small", "medium", "large"]
    df_bins = df.copy()
    df_bins["foreground_bin"] = pd.cut(df_bins["fg_ratio"], bins=bins, labels=labels, include_lowest=True, right=True)
    bin_counts = df_bins.groupby(["split", "foreground_bin"], observed=False).size().rename("tiles").reset_index()
    bin_counts.to_csv(output_dir / "foreground_ratio_bins.csv", index=False)
    event_counts = df.groupby(["split", "event_id"]).agg(tiles=("file", "count"), mean_fg_ratio=("fg_ratio", "mean"), fg_pixels=("fg_pixels", "sum"), valid_pixels=("valid_pixels", "sum")).reset_index()
    event_counts.to_csv(output_dir / "event_tile_summary.csv", index=False)
    unique_values = df.groupby(["split", "unique_values"]).size().rename("tiles").reset_index()
    unique_values.to_csv(output_dir / "mask_unique_values.csv", index=False)
    if write_plots:
        _write_histogram(df, output_dir)
        _write_overlays(processed_data_dir, output_dir, splits, samples_per_split)
    else:
        LOG.info("Audit plots disabled; CSV summaries were written")
    LOG.info("Dataset audit written to: %s", output_dir)
    for split, row in empty.iterrows():
        LOG.info("%s empty tiles: %d/%d (%.2f%%)", split, int(row["sum"]), int(row["count"]), float(row["empty_percent"]))
    LOG.info("Foreground-ratio bin counts written to: %s", output_dir / "foreground_ratio_bins.csv")
    return df
