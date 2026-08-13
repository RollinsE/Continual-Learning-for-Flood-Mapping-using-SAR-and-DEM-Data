from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import yaml

from floods.deployment import (
    _deployment_settings,
    _load_deployment_context,
    write_deployment_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_deployment_sources(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    stats_path = root / "training_normalization_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "processed_data_dir": "/content/mmflood_processed",
                "preserve_channel_stats_from": "/content/drive/MyDrive/private/normalization_stats.json",
                "channels": [
                    {
                        "channel": "vv",
                        "clip_min": 0.0,
                        "clip_max": 1.0,
                        "mean": 0.2,
                        "std": 0.1,
                        "raw_mean": 0.2,
                        "raw_std": 0.1,
                        "robust_mean": 0.2,
                        "robust_std": 0.1,
                    },
                    {
                        "channel": "vh",
                        "clip_min": 0.0,
                        "clip_max": 1.0,
                        "mean": 0.3,
                        "std": 0.1,
                        "raw_mean": 0.3,
                        "raw_std": 0.1,
                        "robust_mean": 0.3,
                        "robust_std": 0.1,
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = yaml.safe_load(
        (REPO_ROOT / "configs" / "training_defaults.yaml").read_text(encoding="utf-8")
    )
    payload["data"]["normalization_stats_path"] = str(stats_path)
    payload["data"]["normalization_mode"] = "robust_percentile"
    config_path = root / "training_config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    checkpoint_path = root / "selected_model.pth"
    checkpoint_path.write_bytes(b"portable-checkpoint")
    return config_path, checkpoint_path, stats_path


def test_export_deployment_creates_self_contained_relative_bundle(tmp_path: Path):
    config_path, checkpoint_path, stats_path = _write_deployment_sources(tmp_path / "source")
    manifest_path = tmp_path / "bundle" / "deployment_manifest.yaml"

    write_deployment_manifest(
        output_file=manifest_path,
        configs=[config_path],
        checkpoints=[checkpoint_path],
        threshold=0.8,
        input_modalities=["vv", "vh"],
    )

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["bundle"]["portable"] is True
    assert not Path(manifest["members"][0]["config"]).is_absolute()
    assert not Path(manifest["members"][0]["checkpoint"]).is_absolute()
    assert not Path(manifest["inputs"]["normalization_stats_path"]).is_absolute()

    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert str(config_path) not in manifest_text
    assert str(checkpoint_path) not in manifest_text
    assert str(stats_path) not in manifest_text

    bundled_config = manifest_path.parent / manifest["members"][0]["config"]
    bundled_checkpoint = manifest_path.parent / manifest["members"][0]["checkpoint"]
    bundled_stats = manifest_path.parent / manifest["inputs"]["normalization_stats_path"]
    bundled_config_text = bundled_config.read_text(encoding="utf-8")
    bundled_payload = yaml.safe_load(bundled_config_text)
    assert str(stats_path) not in bundled_config_text
    assert bundled_payload["data"]["path"] == "."
    assert bundled_payload["data"]["cache_dir"] == "cache"
    assert not Path(bundled_payload["data"]["normalization_stats_path"]).is_absolute()
    assert bundled_payload["output_folder"] == "outputs"
    assert bundled_payload["resume"] is False
    assert bundled_payload["resume_from"] is None
    assert bundled_checkpoint.read_bytes() == checkpoint_path.read_bytes()
    bundled_stats_payload = json.loads(bundled_stats.read_text(encoding="utf-8"))
    assert "processed_data_dir" not in bundled_stats_payload
    assert "preserve_channel_stats_from" not in bundled_stats_payload
    assert bundled_stats_payload["channels"] == json.loads(stats_path.read_text(encoding="utf-8"))["channels"]
    bundled_stats_text = bundled_stats.read_text(encoding="utf-8")
    assert "/content/mmflood_processed" not in bundled_stats_text
    assert "/content/drive/" not in bundled_stats_text

    inventory_path = manifest_path.parent / "deployment_bundle.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert inventory["portable"] is True
    assert len(inventory["files"]) == 3
    checkpoint_record = next(item for item in inventory["files"] if item["role"] == "member_01_checkpoint")
    assert checkpoint_record["sha256"] == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    stats_record = next(item for item in inventory["files"] if item["role"] == "normalization_stats")
    assert stats_record["sha256"] == hashlib.sha256(bundled_stats.read_bytes()).hexdigest()
    assert stats_record["size_bytes"] == bundled_stats.stat().st_size
    assert (manifest_path.parent / "DEPLOYMENT_README.md").is_file()


def test_portable_bundle_still_loads_after_directory_is_moved(tmp_path: Path, monkeypatch):
    config_path, checkpoint_path, _ = _write_deployment_sources(tmp_path / "source")
    original_bundle = tmp_path / "original" / "deployment"
    manifest_path = original_bundle / "deployment_manifest.yaml"
    write_deployment_manifest(
        output_file=manifest_path,
        configs=[config_path],
        checkpoints=[checkpoint_path],
        threshold=0.8,
        input_modalities=["vv", "vh"],
    )

    moved_bundle = tmp_path / "different_machine" / "copied_deployment"
    moved_bundle.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(original_bundle), str(moved_bundle))
    moved_manifest = moved_bundle / "deployment_manifest.yaml"

    captured = {}

    def fake_load_models(configs, checkpoints, device):
        captured["checkpoints"] = list(checkpoints)
        return [object()]

    monkeypatch.setattr("floods.deployment._load_models", fake_load_models)
    manifest, configs, models, _ = _load_deployment_context(moved_manifest, "cpu")
    settings = _deployment_settings(manifest, configs, moved_manifest)

    assert len(models) == 1
    assert captured["checkpoints"][0].is_file()
    assert moved_bundle in captured["checkpoints"][0].parents
    assert settings["normalization_stats_path"].is_file()
    assert moved_bundle in settings["normalization_stats_path"].parents


def test_reference_only_mode_is_explicit_and_environment_dependent(tmp_path: Path):
    config_path, checkpoint_path, _ = _write_deployment_sources(tmp_path / "source")
    manifest_path = tmp_path / "reference" / "deployment_manifest.yaml"
    write_deployment_manifest(
        output_file=manifest_path,
        configs=[config_path],
        checkpoints=[checkpoint_path],
        threshold=0.8,
        input_modalities=["vv", "vh"],
        copy_assets=False,
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["bundle"]["portable"] is False
    assert Path(manifest["members"][0]["config"]).is_absolute()
    assert Path(manifest["members"][0]["checkpoint"]).is_absolute()
    assert not (manifest_path.parent / "assets").exists()
