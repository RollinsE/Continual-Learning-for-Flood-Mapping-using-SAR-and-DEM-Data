from __future__ import annotations

import builtins
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from floods.derived_features import derive_feature_channels


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_vv_vh_config_load_does_not_import_scipy_or_derived_processing():
    script = r'''
import builtins
import sys
from pathlib import Path
import yaml

real_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "scipy" or name.startswith("scipy."):
        raise AssertionError(f"unexpected SciPy import: {name}")
    if name == "floods.derived_features" or name.startswith("floods.derived_features."):
        raise AssertionError(f"unexpected derived-feature import: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
from floods.config import TrainConfig
payload = yaml.safe_load(Path("configs/training_defaults.yaml").read_text(encoding="utf-8"))
config = TrainConfig(**payload)
assert config.data.input_modalities == ["vv", "vh"]
print("lightweight_config_ok")
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "lightweight_config_ok" in result.stdout


def test_deployment_config_loader_does_not_require_scipy(tmp_path: Path):
    payload = yaml.safe_load(
        (REPO_ROOT / "configs" / "training_defaults.yaml").read_text(encoding="utf-8")
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    script = r'''
import builtins
import sys
from pathlib import Path

real_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "scipy" or name.startswith("scipy."):
        raise AssertionError(f"unexpected SciPy import: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
from floods.deployment import _load_train_config
config = _load_train_config(Path(sys.argv[1]))
assert config.data.input_modalities == ["vv", "vh"]
print("deployment_config_ok")
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(config_path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "deployment_config_ok" in result.stdout


def test_ratio_and_slope_derivation_do_not_import_scipy(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy" or name.startswith("scipy."):
            raise AssertionError(f"unexpected SciPy import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    vv = np.full((8, 8), 0.1, dtype=np.float32)
    vh = np.full((8, 8), 0.01, dtype=np.float32)
    dem = np.tile(np.arange(8, dtype=np.float32), (8, 1))
    result = derive_feature_channels(
        np.stack([vv, vh]),
        dem,
        pixel_size_x=1.0,
        pixel_size_y=1.0,
        modalities=["vv_vh_log_ratio", "dem_slope"],
    )
    assert set(result) == {"vv_vh_log_ratio", "dem_slope"}
    np.testing.assert_allclose(result["vv_vh_log_ratio"], 10.0, atol=1e-5)


def test_full_deployment_context_reaches_model_loader_without_scipy(tmp_path: Path, monkeypatch):
    import floods.deployment as deployment

    payload = yaml.safe_load(
        (REPO_ROOT / "configs" / "training_defaults.yaml").read_text(encoding="utf-8")
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_bytes(b"placeholder")
    manifest_path = tmp_path / "deployment_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "model_name": "smoke",
                "mode": "single",
                "members": [
                    {
                        "config": str(config_path),
                        "checkpoint": str(checkpoint_path),
                    }
                ],
                "inputs": {"modalities": ["vv", "vh"]},
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    def fake_load_models(configs, checkpoints, device):
        captured["modalities"] = configs[0].data.input_modalities
        captured["checkpoints"] = checkpoints
        captured["device"] = str(device)
        return [object()]

    monkeypatch.setattr(deployment, "_load_models", fake_load_models)
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy" or name.startswith("scipy."):
            raise AssertionError(f"unexpected SciPy import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    manifest, configs, models, device = deployment._load_deployment_context(
        manifest_path,
        "cpu",
    )
    assert manifest["model_name"] == "smoke"
    assert configs[0].data.input_modalities == ["vv", "vh"]
    assert len(models) == 1
    assert str(device) == "cpu"
    assert captured["modalities"] == ["vv", "vh"]
