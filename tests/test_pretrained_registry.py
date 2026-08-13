from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from floods.cli import _apply_training_overrides, build_cli_parser
from floods.config import TrainConfig
from floods.event_cv import _aggregate_results, run_event_cross_validation
from floods.pretrained import parse_candidate_spec, resolve_candidate, resolve_model_spec


def test_terramind_registry_supports_s1_with_optional_dem():
    s1 = resolve_model_spec(weights_source="terramind_v1_tiny", modalities=["vv", "vh"])
    s1_dem = resolve_model_spec(weights_source="terramind_v1_base", modalities=["vv", "vh", "dem"])
    assert s1.decoder == "segformer"
    assert s1.normalization_mode == "terramind_v1"
    assert s1_dem.encoder == "terramind_v1_base"
    assert s1_dem.modalities == ("vv", "vh", "dem")


def test_croma_rejects_dem_without_package_modification():
    with pytest.raises(ValueError, match="does not support"):
        resolve_model_spec(weights_source="croma_sar_base", modalities=["vv", "vh", "dem"])


def test_candidate_syntax_keeps_provider_and_optional_label():
    candidate = parse_candidate_spec("imagenet:unet:resnet34:reference")
    resolved = resolve_candidate(candidate, ["vv", "vh"])
    assert resolved.weights_source == "imagenet"
    assert resolved.decoder == "unet"
    assert resolved.encoder == "resnet34"
    assert resolved.label == "reference"


def test_train_cli_resolves_terramind_and_dem_from_neutral_config(tmp_path):
    parser = build_cli_parser()
    args = parser.parse_args([
        "train", "--config", str(tmp_path / "training_defaults.yaml"),
        "--weights-source", "terramind_v1_tiny",
        "--input-modalities", "vv", "vh", "dem",
    ])
    config = TrainConfig()
    config.data.in_channels = 2
    config.data.input_modalities = ["vv", "vh"]
    config.data.normalization_mode = "robust_percentile"
    resolved = _apply_training_overrides(config, args)
    assert resolved.model.weights_source == "terramind_v1_tiny"
    assert resolved.model.pretrained is True
    assert resolved.model.encoder == "terramind_v1_tiny"
    assert resolved.model.decoder == "segformer"
    assert resolved.data.input_modalities == ["vv", "vh", "dem"]
    assert resolved.data.normalization_mode == "terramind_v1"
    assert resolved.data.normalization_stats_path is None


def test_event_cv_candidate_plan_is_provider_aware(tmp_path):
    mask_dir = tmp_path / "processed" / "train" / "mask"
    mask_dir.mkdir(parents=True)
    for event in ("EMSR001", "EMSR002", "EMSR003", "EMSR004"):
        (mask_dir / f"{event}_tile_000.tif").touch()
    result = run_event_cross_validation(
        TrainConfig(),
        processed_data_dir=tmp_path / "processed",
        output_dir=tmp_path / "cv",
        candidates=["terramind_v1_tiny"],
        modality_sets=["vv+vh+dem"],
        folds=2,
        plan_only=True,
    )
    assert result["planned_runs"] == 2
    manifest = (tmp_path / "cv" / "cv_manifest.json").read_text()
    assert "terramind_v1_tiny" in manifest
    assert "terramind_v1" in manifest


def test_cv_summary_includes_median_best_epoch(tmp_path):
    pd.DataFrame([
        {
            "candidate": "terramind_v1_tiny", "weights_source": "terramind_v1_tiny",
            "provider": "terratorch", "decoder": "segformer", "encoder": "terramind_v1_tiny",
            "modalities": "vv+vh+dem", "normalization_mode": "terramind_v1",
            "fold": 0, "best_epoch": 3, "event_macro_f1": 0.4, "worst_event_f1": 0.1,
            "threshold": 0.7,
        },
        {
            "candidate": "terramind_v1_tiny", "weights_source": "terramind_v1_tiny",
            "provider": "terratorch", "decoder": "segformer", "encoder": "terramind_v1_tiny",
            "modalities": "vv+vh+dem", "normalization_mode": "terramind_v1",
            "fold": 1, "best_epoch": 5, "event_macro_f1": 0.5, "worst_event_f1": 0.2,
            "threshold": 0.8,
        },
    ]).to_csv(tmp_path / "cv_results.csv", index=False)
    _aggregate_results(tmp_path)
    row = pd.read_csv(tmp_path / "cv_summary.csv").iloc[0]
    assert float(row["median_best_epoch"]) == 4.0


def test_provider_normalizer_converts_linear_sar_and_standardizes_dem():
    pytest.importorskip("albumentations")
    from floods.transforms import ProviderSARNormalize
    transform = ProviderSARNormalize(
        ["vv", "vh", "dem"], source_sar_transform="linear",
        sar_mean=[-10.0, -20.0], sar_std=[2.0, 4.0], dem_mean=100.0, dem_std=50.0,
    )
    image = np.asarray([[[0.1, 0.01, 150.0]]], dtype=np.float32)
    result = transform(image=image, mask=np.zeros((1, 1), dtype=np.uint8))["image"]
    assert result.shape == (1, 1, 3)
    assert result[0, 0, 0] == pytest.approx(0.0, abs=1e-5)
    assert result[0, 0, 1] == pytest.approx(0.0, abs=1e-5)
    assert result[0, 0, 2] == pytest.approx(1.0, abs=1e-5)


def test_standard_requirements_include_provider_dependencies():
    for name in ("requirements.txt", "requirements-colab.txt"):
        text = Path(name).read_text()
        assert "-r requirements-foundation.txt" in text
    foundation = Path("requirements-foundation.txt").read_text()
    assert "torchgeo==0.9.0" in foundation
    assert "terratorch==1.2.8" in foundation


def test_event_cv_cli_accepts_shared_training_hyperparameters(tmp_path):
    parser = build_cli_parser()
    args = parser.parse_args([
        "event-cv", "--config", str(tmp_path / "training_defaults.yaml"),
        "--processed-data-dir", str(tmp_path / "processed"),
        "--output-dir", str(tmp_path / "cv"),
        "--candidates", "terramind_v1_tiny",
        "--modality-sets", "vv+vh+dem",
        "--encoder-lr", "1e-5", "--decoder-lr", "1e-4",
        "--weight-decay", "1e-4", "--optimizer", "adamw",
        "--scheduler", "poly", "--loss", "bce_tversky",
        "--loss-alpha", "0.3", "--loss-beta", "0.7",
        "--bce-weight", "0.5", "--tversky-weight", "0.5",
    ])
    assert args.encoder_lr == pytest.approx(1e-5)
    assert args.decoder_lr == pytest.approx(1e-4)
    assert args.optimizer == "adamw"
    assert args.loss == "bce_tversky"
