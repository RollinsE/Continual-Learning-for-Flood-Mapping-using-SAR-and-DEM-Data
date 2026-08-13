from pathlib import Path

import pandas as pd

from floods.hard_examples import selected_hard_example_indices


class DummyDataset:
    def __init__(self):
        self.label_files = [
            "/data/train/mask/EMSR001_0_0.tif",
            "/data/train/mask/EMSR001_0_512.tif",
            "/data/train/mask/EMSR002_0_0.tif",
            "/data/train/mask/EMSR002_0_512.tif",
        ]

    def __len__(self):
        return len(self.label_files)


def test_selected_hard_examples_match_by_file_and_category(tmp_path):
    csv_path = tmp_path / "tile_error_metrics.csv"
    pd.DataFrame([
        {"file": "EMSR001_0_0.tif", "error_category": "false_negative_low_recall", "foreground_bin": "large", "f1": 0.10},
        {"file": "EMSR001_0_512.tif", "error_category": "partial_overlap", "foreground_bin": "tiny", "f1": 0.20},
        {"file": "EMSR002_0_0.tif", "error_category": "partial_overlap", "foreground_bin": "large", "f1": 0.80},
    ]).to_csv(csv_path, index=False)

    indices = selected_hard_example_indices(
        label_files=DummyDataset().label_files,
        hard_example_csv=str(csv_path),
        hard_example_categories=["false_negative_low_recall"],
        hard_example_fg_bins=["tiny", "small"],
        hard_example_max_f1=0.30,
    )
    assert indices == {0, 1}

