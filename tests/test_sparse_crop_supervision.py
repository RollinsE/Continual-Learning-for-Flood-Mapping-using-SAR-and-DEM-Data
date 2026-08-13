import numpy as np

from floods.sparse_crops import (
    MODE_FLOOD_CENTERED,
    MODE_HARD_BACKGROUND,
    MODE_NORMAL,
    SparseFloodCropSupervision,
)


def _image(size=512, channels=3):
    yy, xx = np.mgrid[:size, :size]
    image = np.zeros((size, size, channels), dtype=np.float32)
    image[..., 0] = yy
    image[..., 1] = xx
    image[..., 2] = yy + xx
    return image


def test_flood_centered_crop_is_resized_and_keeps_flood():
    np.random.seed(7)
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[240:248, 240:248] = 1
    original_fg = int(mask.sum())
    transform = SparseFloodCropSupervision(
        target_size=512,
        crop_sizes=[256],
        normal_fraction=0.0,
        flood_centered_fraction=1.0,
        hard_background_fraction=0.0,
    )

    image_out, mask_out, metadata = transform(_image(), mask)

    assert image_out.shape == (512, 512, 3)
    assert mask_out.shape == (512, 512)
    assert metadata.requested_mode == MODE_FLOOD_CENTERED
    assert metadata.applied_mode == MODE_FLOOD_CENTERED
    assert metadata.crop_size == 256
    assert int(mask_out.sum()) > original_fg


def test_hard_background_crop_comes_from_flood_tile_but_excludes_flood():
    np.random.seed(11)
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[0:40, 0:40] = 1
    transform = SparseFloodCropSupervision(
        target_size=512,
        crop_sizes=[256],
        normal_fraction=0.0,
        flood_centered_fraction=0.0,
        hard_background_fraction=1.0,
        attempts=100,
        hard_background_max_fg_ratio=0.0,
    )

    image_out, mask_out, metadata = transform(_image(), mask)

    assert image_out.shape == (512, 512, 3)
    assert mask_out.shape == (512, 512)
    assert metadata.requested_mode == MODE_HARD_BACKGROUND
    assert metadata.applied_mode == MODE_HARD_BACKGROUND
    assert metadata.crop_size == 256
    assert int(mask_out.sum()) == 0


def test_empty_tile_remains_full_size_normal_sample():
    np.random.seed(3)
    mask = np.zeros((512, 512), dtype=np.uint8)
    transform = SparseFloodCropSupervision(
        target_size=512,
        crop_sizes=[256],
        normal_fraction=0.0,
        flood_centered_fraction=0.0,
        hard_background_fraction=1.0,
    )

    image_out, mask_out, metadata = transform(_image(), mask)

    assert image_out.shape == (512, 512, 3)
    assert mask_out.shape == (512, 512)
    assert metadata.requested_mode == MODE_NORMAL
    assert metadata.applied_mode == MODE_NORMAL
    assert metadata.crop_size == 0


def test_fraction_mix_is_normalized():
    transform = SparseFloodCropSupervision(
        target_size=512,
        crop_sizes=[256],
        normal_fraction=2.0,
        flood_centered_fraction=1.0,
        hard_background_fraction=1.0,
    )
    assert transform.normal_fraction == 0.5
    assert transform.flood_centered_fraction == 0.25
    assert transform.hard_background_fraction == 0.25
