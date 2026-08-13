import numpy as np
import pytest

from floods.utils.tiling.functional import tile_fixed_overlap, tile_overlapped


def test_dynamic_overlap_uses_rectangular_tile_width_in_coordinates():
    image = np.zeros((100, 160, 2), dtype=np.float32)
    windows = list(tile_overlapped(image, tile_size=(64, 80), overlap_threshold=32))

    assert windows[0] == ((0, 0), (0, 0, 64, 80))
    assert windows[1] == ((0, 1), (0, 80, 64, 160))
    assert all((x2 - x1, y2 - y1) == (64, 80) for _, (x1, y1, x2, y2) in windows)


def test_fixed_overlap_uses_consistent_row_column_coordinate_order():
    image = np.zeros((100, 160, 2), dtype=np.float32)
    windows = list(tile_fixed_overlap(image, tile_size=(64, 80), overlap=16))

    assert windows[0] == ((0, 0), (0, 0, 64, 80))
    assert windows[1] == ((0, 1), (0, 64, 64, 144))


def test_tilers_reject_invalid_dimensions_and_overlap():
    image = np.zeros((100, 160, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="positive dimensions"):
        list(tile_overlapped(image, tile_size=(0, 80), overlap_threshold=32))
    with pytest.raises(ValueError, match="overlap must"):
        list(tile_fixed_overlap(image, tile_size=(64, 80), overlap=64))
