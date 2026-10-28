"""Tests for preprocessing, normalisation, augmentation, continual-learning and deployment conventions."""
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from floods.continual import (CLTask, ReplayBuffer, _next_position_from_checkpoint, _summarise_cl_matrix,
                              parse_year_ranges, pixel_uncertainty, sample_replay_indices)
from floods.inference_transforms import detect_sar_units, eval_transforms as inference_eval_transforms, prepare_sar_input
from floods.preproc import _check_same_grid, _clip_dem, _prepare_subset_output, _sar_db10, _sar_log1p
from floods.transforms import (ClipNormalize, PhysicalGaussianBlur, RobustMinMaxNormalize, SARRadiometricJitter,
                               db_to_sar, sar_to_db)


# --------------------------------------------------------------------------- preprocessing


def test_sar_transforms_mark_nodata_as_nan():
    linear = np.array([[[0.0, 0.1, -1.0, np.nan, 1.0]]], dtype=np.float32)
    db = _sar_db10(linear)
    assert np.isnan(db[0, 0, [0, 2, 3]]).all()
    np.testing.assert_allclose(db[0, 0, [1, 4]], [-10.0, 0.0], atol=1e-5)
    assert np.isnan(_sar_log1p(linear)[0, 0, 0])


def test_dem_voids_become_nan_and_valid_values_are_clipped():
    dem = np.array([-32768.0, -9999.0, -150.0, 50.0, 7000.0, np.nan], dtype=np.float32)
    out = _clip_dem(dem)
    assert np.isnan(out[[0, 1, 5]]).all()
    np.testing.assert_allclose(out[[2, 3, 4]], [-100.0, 50.0, 6000.0])


def test_existing_processed_split_is_refused_unless_overwriting(tmp_path):
    (tmp_path / "train" / "sar").mkdir(parents=True)
    (tmp_path / "train" / "sar" / "a.tif").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        _prepare_subset_output(tmp_path / "train", overwrite=False)
    _prepare_subset_output(tmp_path / "train", overwrite=True)
    assert not (tmp_path / "train" / "sar").exists()


def _tif(path: Path, data: np.ndarray, transform) -> Path:
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                       dtype=str(data.dtype), crs="EPSG:32631", transform=transform) as dst:
        dst.write(data[None])
    return path


def test_grid_check_rejects_shifted_dem(tmp_path):
    grid = from_origin(0, 100, 10, 10)
    sar = _tif(tmp_path / "sar.tif", np.ones((10, 10), np.float32), grid)
    mask = _tif(tmp_path / "mask.tif", np.zeros((10, 10), np.uint8), grid)
    same = _tif(tmp_path / "dem_ok.tif", np.ones((10, 10), np.float32), grid)
    shifted = _tif(tmp_path / "dem_bad.tif", np.ones((10, 10), np.float32), from_origin(10, 100, 10, 10))
    _check_same_grid(sar, same, mask)
    with pytest.raises(ValueError, match="not on the SAR grid"):
        _check_same_grid(sar, shifted, mask)


# --------------------------------------------------------------------------- normalisation and augmentation


def test_robust_minmax_standardises_and_zero_fills_missing():
    rng = np.random.default_rng(0)
    image = np.dstack([rng.normal(-12, 4, (64, 64)), rng.uniform(0, 800, (64, 64))]).astype(np.float32)
    image[:4] = np.nan
    lo, hi = np.nanpercentile(image.reshape(-1, 2), [1, 99], axis=0)
    scaled = np.clip((image[4:] - lo) / (hi - lo), 0, 1)
    mean, std = scaled.reshape(-1, 2).mean(0), scaled.reshape(-1, 2).std(0)
    out = RobustMinMaxNormalize(mean=tuple(mean), std=tuple(std), clip_min=tuple(lo), clip_max=tuple(hi))(image=image)["image"]
    assert np.all(out[:4] == 0.0)
    np.testing.assert_allclose(out[4:].reshape(-1, 2).std(0), [1.0, 1.0], atol=0.05)


def test_clip_normalize_matches_manual_formula():
    image = np.random.default_rng(1).normal(-12, 4, (16, 16, 2)).astype(np.float32)
    out = ClipNormalize(mean=(-12, -18), std=(4, 5), clip_min=(-25, -30), clip_max=(0, -5))(image=image)["image"]
    expected = (np.clip(image, [-25, -30], [0, -5]) - [-12, -18]) / [4, 5]
    np.testing.assert_allclose(out, expected, atol=1e-4)


def test_inference_transform_matches_training_transform():
    image = np.random.default_rng(2).normal(-12, 4, (16, 16, 2)).astype(np.float32)
    image[0, 0, 0] = np.nan
    kwargs = dict(mean=(0.4, 0.5), std=(0.2, 0.25), clip_min=(-25, -30), clip_max=(0, -5))
    train = RobustMinMaxNormalize(**kwargs)(image=image)["image"]
    infer = inference_eval_transforms(**kwargs, normalization_mode="robust_minmax")(image=image)["image"].numpy()
    np.testing.assert_allclose(np.moveaxis(infer, 0, -1), train, atol=1e-5)


@pytest.mark.parametrize("units", ["db10", "log1p", "linear"])
def test_sar_jitter_is_additive_in_db_and_keeps_nan(units):
    linear = np.full((32, 32, 2), 0.05, np.float32)
    linear[0, 0, 0] = np.nan
    x = db_to_sar(10 * np.log10(linear), units)
    jitter = SARRadiometricJitter(source_sar_transform=units, p=1.0, noise_db=0.0, seed=3)
    y = jitter(image=x, mask=None)["image"]
    delta = sar_to_db(y, units) - sar_to_db(x, units)
    assert np.isnan(y[0, 0, 0])
    # Without noise every pixel of a channel moves by the same dB offset.
    assert np.nanstd(delta[..., 0]) < 1e-3 and np.nanstd(delta[..., 1]) < 1e-3
    assert np.nanmax(np.abs(delta)) <= 1.5 + 1e-4


def test_physical_blur_does_not_clip_to_unit_range():
    image = np.full((32, 32, 3), -15.0, np.float32)
    image[..., 2] = 450.0
    out = PhysicalGaussianBlur(p=1.0, seed=0)(image=image, mask=None)["image"]
    np.testing.assert_allclose(out[..., 0], -15.0, atol=1e-4)
    np.testing.assert_allclose(out[..., 2], 450.0, atol=1e-3)


def test_dataset_sets_ignore_label_where_inputs_are_missing():
    from floods.datasets.flood import FloodDataset
    image = np.zeros((4, 4, 2), np.float32)
    image[0, 0, 1] = np.nan
    label = np.ones((4, 4), np.uint8)
    out = FloodDataset._ignore_missing(image, label, 255)
    assert out[0, 0] == 255 and (out != 255).sum() == 15


# --------------------------------------------------------------------------- continual learning


def test_margin_is_twice_least_confidence_for_sigmoid_output():
    prob = torch.rand(1000)
    torch.testing.assert_close(pixel_uncertainty(prob, "margin"), 2 * pixel_uncertainty(prob, "least_confidence"))


def test_no_replay_strategies_never_replay():
    buffer = ReplayBuffer(10, seed=0)
    buffer.add_many(range(10))
    assert sample_replay_indices(buffer, "finetune", None, 4) == []
    assert sample_replay_indices(buffer, "joint", None, 4) == []


def test_uncertainty_sampling_is_stochastic_and_hybrid_mixes():
    buffer = ReplayBuffer(50, seed=0)
    buffer.add_many(range(50))
    scores = {i: float(i) for i in range(50)}
    draws = {tuple(sorted(sample_replay_indices(buffer, "entropy", scores, 5))) for _ in range(20)}
    assert len(draws) > 1, "proportional sampling should not return the same tiles every batch"
    top = sample_replay_indices(buffer, "entropy", scores, 5, sampling="topk")
    assert sorted(top) == [45, 46, 47, 48, 49]
    hybrid = sample_replay_indices(buffer, "hybrid", scores, 6)
    assert len(hybrid) == 6 and len(set(hybrid)) == 6


def test_replay_buffer_state_round_trip():
    buffer = ReplayBuffer(5, seed=1)
    buffer.add_many(range(20))
    state = buffer.state_dict()
    restored = ReplayBuffer(5, seed=99)
    restored.load_state_dict(state)
    assert restored.indices == buffer.indices and restored.seen == 20
    buffer.add_many(range(20, 40))
    restored.add_many(range(20, 40))
    assert restored.indices == buffer.indices


def test_resume_position_reruns_unfinished_task_evaluation():
    assert _next_position_from_checkpoint({"completed_task_id": 2, "completed_epoch_in_task": 5, "task_end_done": True}, 5) == (3, 1)
    assert _next_position_from_checkpoint({"completed_task_id": 2, "completed_epoch_in_task": 5, "task_end_done": False}, 5) == (2, 6)
    assert _next_position_from_checkpoint({"completed_task_id": 2, "completed_epoch_in_task": 3, "task_end_done": False}, 5) == (2, 4)


def test_cl_summary_excludes_final_task_from_forgetting():
    def row(after, task, f1):
        return {"after_task": after, "eval_task": task, "metrics": {"f1": f1}}
    matrix = [row(1, 1, 0.6), row(2, 1, 0.4), row(2, 2, 0.7), row(3, 1, 0.5), row(3, 2, 0.6), row(3, 3, 0.8)]
    summary = _summarise_cl_matrix(matrix, "f1")
    assert summary["mean_final_task_score"] == pytest.approx((0.5 + 0.6 + 0.8) / 3)
    assert summary["mean_forgetting"] == pytest.approx(((0.6 - 0.5) + (0.7 - 0.6)) / 2)
    assert summary["backward_transfer"] == pytest.approx(((0.5 - 0.6) + (0.6 - 0.7)) / 2)
    assert summary["mean_plasticity"] == pytest.approx((0.6 + 0.7 + 0.8) / 3)


def test_overlapping_year_ranges_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        parse_year_ranges(["2014-2018", "2018-2020"])


def test_cl_task_val_indices_default():
    task = CLTask(1, "2014", 2014, 2014, [0], [1])
    assert task.val_indices is None


def test_cli_defaults_include_baselines_and_drop_margin():
    from floods.cli import build_cli_parser
    args = build_cli_parser().parse_args(["continual-train", "--config", "c.yaml", "--activations-json-path", "a.json"])
    assert args.strategies == ["finetune", "random", "least_confidence", "entropy", "hybrid", "joint"]
    assert args.uncertainty_sampling == "proportional" and args.cl_threshold is None


# --------------------------------------------------------------------------- deployment input units


def test_sar_unit_detection():
    rng = np.random.default_rng(0)
    linear = rng.gamma(2.0, 0.03, (64, 64)).astype(np.float32)
    assert detect_sar_units(linear, linear * 0.2) == "linear"
    assert detect_sar_units(10 * np.log10(linear), 10 * np.log10(linear * 0.2)) == "db10"


def test_linear_and_db_uploads_convert_to_the_same_model_input():
    rng = np.random.default_rng(1)
    vv = rng.gamma(2.0, 0.03, (32, 32)).astype(np.float32)
    vh = vv * 0.2
    vv[:, :3] = 0.0  # swath edge
    vh[:, :3] = 0.0
    a_vv, a_vh, units_a = prepare_sar_input(vv, vh, "auto", "db10")
    with np.errstate(divide="ignore"):
        db_vv = np.where(vv > 0, 10 * np.log10(np.maximum(vv, 1e-12)), np.nan)
        db_vh = np.where(vh > 0, 10 * np.log10(np.maximum(vh, 1e-12)), np.nan)
    b_vv, b_vh, units_b = prepare_sar_input(db_vv, db_vh, "auto", "db10")
    assert (units_a, units_b) == ("linear", "db10")
    assert np.isnan(a_vv[:, :3]).all() and np.isnan(b_vv[:, :3]).all()
    np.testing.assert_allclose(a_vv[:, 3:], b_vv[:, 3:], atol=1e-4)
    log1p_vv, _, _ = prepare_sar_input(vv, vh, "linear", "log1p")
    np.testing.assert_allclose(log1p_vv[:, 3:], np.log10(1 + vv[:, 3:]), atol=1e-5)
