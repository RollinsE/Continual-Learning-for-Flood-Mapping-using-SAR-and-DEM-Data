from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _tile_keys(path_value) -> set[str]:
    """Return stable identifiers used to match dataset tiles to audit rows."""
    if path_value is None:
        return set()
    text = str(path_value).strip()
    if not text or text.lower() == "nan":
        return set()
    path = Path(text)
    keys = {text.replace("\\", "/"), path.name, path.stem}
    return {k for k in keys if k}


def _audit_row_keys(row: pd.Series) -> set[str]:
    """Extract possible tile identifiers from an error-audit CSV row."""
    keys: set[str] = set()
    for column in ("mask_path", "image_path", "dem_path", "file", "filename", "tile", "tile_id"):
        if column in row:
            keys.update(_tile_keys(row[column]))
    return keys


def selected_hard_example_indices(label_files: Sequence[str],
                                  hard_example_csv: str,
                                  hard_example_categories: Iterable[str],
                                  hard_example_fg_bins: Iterable[str],
                                  hard_example_max_f1: float) -> set[int]:
    """Match audited hard examples to the current training dataset.

    The expected input is the ``tile_error_metrics.csv`` produced by
    ``floodmap error-audit --split train``. Rows are selected when their
    ``error_category`` is in ``hard_example_categories`` or when their
    foreground-size bin is listed in ``hard_example_fg_bins`` and tile F1 is
    not greater than ``hard_example_max_f1``. Matching is done by full path,
    file name, and stem so the CSV remains portable across filesystems and runtime locations.
    """
    if not hard_example_csv:
        raise ValueError("hard_example_csv is required when hard-example sampling is enabled")
    csv_path = Path(str(hard_example_csv)).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Hard-example CSV does not exist: {csv_path}")
    audit = pd.read_csv(csv_path)
    if audit.empty:
        raise ValueError(f"Hard-example CSV is empty: {csv_path}")

    categories = {str(v).strip() for v in hard_example_categories if str(v).strip()}
    fg_bins = {str(v).strip().lower() for v in hard_example_fg_bins if str(v).strip()}
    max_f1 = float(hard_example_max_f1)

    selected_rows = []
    for _, row in audit.iterrows():
        category = str(row.get("error_category", "")).strip()
        foreground_bin = str(row.get("foreground_bin", "")).strip().lower()
        try:
            f1 = float(row.get("f1", np.nan))
        except (TypeError, ValueError):
            f1 = np.nan
        by_category = category in categories
        by_bin = foreground_bin in fg_bins and (np.isnan(f1) or f1 <= max_f1)
        if by_category or by_bin:
            selected_rows.append(row)

    if not selected_rows:
        raise ValueError(
            "No rows in hard-example CSV matched the selected categories or foreground-bin criteria."
        )

    lookup: dict[str, int] = {}
    for idx, label_path in enumerate(label_files):
        for key in _tile_keys(label_path):
            lookup[key] = idx

    indices: set[int] = set()
    for row in selected_rows:
        row_indices = {lookup[key] for key in _audit_row_keys(row) if key in lookup}
        indices.update(row_indices)

    if not indices:
        raise ValueError(
            "Hard-example CSV rows matched the selection rules, but none matched this training dataset. "
            "Run error-audit on the same processed-data-dir and the train split."
        )
    return indices
