import pandas as pd

from floods.error_audit import _select_overlay_rows


def _rows():
    return pd.DataFrame(
        [
            {"error_category": "false_positive_empty", "fp_pixels": 10, "pred_ratio": 0.2, "iou": 0.0, "fg_ratio": 0.0, "valid_pixels": 100},
            {"error_category": "false_positive_empty", "fp_pixels": 5, "pred_ratio": 0.1, "iou": 0.0, "fg_ratio": 0.0, "valid_pixels": 100},
            {"error_category": "poor_overlap", "fp_pixels": 1, "pred_ratio": 0.1, "iou": 0.1, "fg_ratio": 0.3, "valid_pixels": 100},
            {"error_category": "poor_overlap", "fp_pixels": 1, "pred_ratio": 0.1, "iou": 0.2, "fg_ratio": 0.4, "valid_pixels": 100},
        ]
    )


def test_overlay_selection_accepts_primary_argument():
    selected = _select_overlay_rows(_rows(), max_per_category=1)
    assert len(selected) == 2


def test_overlay_selection_accepts_cli_alias():
    selected = _select_overlay_rows(_rows(), max_overlays_per_category=1)
    assert len(selected) == 2


def test_crop_to_common_shape_uses_smallest_extent():
    import numpy as np
    from floods.error_audit import _crop_to_common_shape

    a = np.zeros((1440, 1568), dtype=np.float32)
    b = np.zeros((1410, 1545), dtype=np.float32)
    c = np.zeros((1420, 1600), dtype=np.float32)

    cropped = _crop_to_common_shape(a, b, c)

    assert [arr.shape for arr in cropped] == [(1410, 1545), (1410, 1545), (1410, 1545)]
