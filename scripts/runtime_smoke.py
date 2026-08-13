"""Runtime smoke checks for the Flood Extent Mapping CLI package.

The script builds a tiny synthetic processed dataset and exercises the main
low-cost paths that should work before any long-running training or evaluation
job is started.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from floods.audit import audit_dataset
from floods.cli import _load_config_from_yaml
from floods.config import TrainConfig
from floods.evaluation import BatchAverageMetrics, BinaryThresholdSweep
from floods.normalization import fit_normalization_stats
from floods.prepare import eval_transforms, prepare_datasets, prepare_model, prepare_stratified_sampler, train_transforms_base, train_transforms_dem, train_transforms_sar
from floods.sliding_window import sliding_window_logits


class TinyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]


def _write_synthetic_dataset(root: Path) -> None:
    shutil.rmtree(root, ignore_errors=True)
    for split, count in [("train", 8), ("val", 4), ("test", 3)]:
        for subdir in ["sar", "mask", "dem"]:
            (root / split / subdir).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            name = f"tile_{index:03d}.tif"
            height = width = 64
            sar = (np.random.randn(2, height, width) * 0.1).astype("float32")
            dem = (np.random.rand(1, height, width) * 100).astype("float32")
            mask = np.zeros((1, height, width), dtype="uint8")
            if index % 3 != 0:
                mask[0, 16:32, 16:32] = 1
            if index == count - 1:
                mask[0, 0:4, 0:4] = 255
            base_profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "transform": from_origin(0, 0, 1, 1),
            }
            with rasterio.open(root / split / "sar" / name, "w", **dict(base_profile, count=2, dtype="float32")) as dst:
                dst.write(sar)
            with rasterio.open(root / split / "dem" / name, "w", **dict(base_profile, count=1, dtype="float32")) as dst:
                dst.write(dem)
            with rasterio.open(root / split / "mask" / name, "w", **dict(base_profile, count=1, dtype="uint8")) as dst:
                dst.write(mask)


def _check_transforms() -> None:
    image = np.random.randn(64, 64, 3).astype("float32")
    mask = np.zeros((64, 64), dtype="uint8")
    mask[16:32, 16:32] = 1
    norm = eval_transforms(
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        clip_min=(-1.0, -1.0, -1.0),
        clip_max=(1.0, 1.0, 1.0),
        normalization_mode="robust_percentile",
    )
    for profile in ["none", "geometric", "sar_radiometric", "standard", "crop_aware", "deformation", "composite"]:
        out = train_transforms_base(64, augmentation_profile=profile)(image=image.copy(), mask=mask.copy())
        out = train_transforms_sar(augmentation_profile=profile)(image=out["image"], mask=out["mask"])
        out = train_transforms_dem()(image=out["image"], mask=out["mask"])
        out = norm(image=out["image"], mask=out["mask"])
        assert "image" in out and "mask" in out
        assert tuple(out["image"].shape[-2:]) == (64, 64)


def _check_dataset_loading(repo_root: Path, dataset_root: Path) -> None:
    config = _load_config_from_yaml(repo_root / "configs" / "train_unet_resnet34_vv_vh_dem.yaml", TrainConfig)
    config.data.path = str(dataset_root)
    config.image_size = 64
    config.data.in_channels = 3
    config.data.include_dem = True
    config.data.train_mask_body_ratio = 0.0
    config.data.val_mask_body_ratio = 0.0
    config.data.weighted_sampling = False
    config.data.event_balanced_sampling = False
    config.data.augmentation_profile = "geometric"
    stats_path = dataset_root.parent / "normalization_stats.json"
    fit_normalization_stats(dataset_root, stats_path, split="train", input_modalities=["vv", "vh", "dem"], max_pixels_per_file=64)
    config.data.normalization_mode = "robust_percentile"
    config.data.normalization_stats_path = str(stats_path)
    train_set, val_set = prepare_datasets(config=config, use_rgb=False)
    assert len(train_set) == 8
    assert len(val_set) == 4
    image, mask = train_set[0]
    assert tuple(image.shape[-2:]) == (64, 64)
    assert tuple(mask.shape[-2:]) == (64, 64)

    # Exercise the real sparse-crop path at a tiny smoke-test scale.
    config.data.sparse_crop_supervision = True
    config.data.sparse_crop_sizes = [32, 48]
    config.data.sparse_crop_normal_fraction = 0.0
    config.data.sparse_crop_flood_fraction = 1.0
    config.data.sparse_crop_hard_background_fraction = 0.0
    crop_train_set, _ = prepare_datasets(config=config, use_rgb=False)
    crop_sample = crop_train_set[1]
    assert len(crop_sample) == 5
    crop_image, crop_mask, requested_mode, applied_mode, crop_size = crop_sample
    assert tuple(crop_image.shape[-2:]) == (64, 64)
    assert tuple(crop_mask.shape[-2:]) == (64, 64)
    assert requested_mode == 1 and applied_mode == 1
    assert crop_size in {32, 48}

    sampler = prepare_stratified_sampler(
        train_set,
        cache_hash="runtime_smoke",
        cache_dir=str(dataset_root.parent / "cache"),
        samples_multiplier=0.5,
    )
    assert len(sampler) == 4


def _check_metrics() -> None:
    sweep = BinaryThresholdSweep(thresholds=[0.2, 0.5], device="cpu")
    y_true = torch.tensor([[[0, 1], [1, 255]], [[0, 0], [0, 0]]])
    logits = torch.tensor([[[[-1.0, 2.0], [0.1, 3.0]]], [[[1.0, -2.0], [-3.0, -4.0]]]])
    sweep.update(y_true, logits)
    rows = sweep.compute()
    assert len(rows) == 2
    assert all(-1.0 <= row.mcc <= 1.0 for row in rows)
    batch_metrics = BatchAverageMetrics(threshold=0.5)
    batch_metrics.update(y_true, logits)
    assert "f1" in batch_metrics.compute()


def _check_model_construction(repo_root: Path, dataset_root: Path) -> None:
    for config_name, decoder in [
        ("train_unet_resnet34_vv_vh_dem.yaml", "unet"),
        ("train_deeplabv3p_resnet34_vv_vh_dem.yaml", "deeplabv3p"),
    ]:
        config = _load_config_from_yaml(repo_root / "configs" / config_name, TrainConfig)
        config.data.path = str(dataset_root)
        config.image_size = 64
        config.model.encoder = "resnet10t"
        config.model.decoder = decoder
        config.model.pretrained = False
        config.data.in_channels = 3
        config.data.include_dem = True
        model = prepare_model(config, num_classes=1)
        assert model is not None


def _check_sliding_window() -> None:
    model = TinyModel()
    image = torch.rand(1, 3, 70, 80)
    logits = sliding_window_logits(model, image, window_size=32, overlap=8, window_batch_size=2)
    assert tuple(logits.shape) == (1, 70, 80)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    work_root = repo_root / ".runtime_smoke"
    dataset_root = work_root / "processed"
    audit_root = work_root / "audit"
    _write_synthetic_dataset(dataset_root)
    audit_dataset(dataset_root, audit_root, ["train", "val", "test"], samples_per_split=0, write_plots=False)
    _check_transforms()
    _check_dataset_loading(repo_root, dataset_root)
    _check_metrics()
    _check_sliding_window()
    _check_model_construction(repo_root, dataset_root)
    subprocess.run([sys.executable, "-m", "floods.cli", "evaluate", "--help"], check=True, stdout=subprocess.DEVNULL)
    print("runtime_smoke_ok")


if __name__ == "__main__":
    main()
