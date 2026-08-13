from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from floods.inference_transforms import eval_transforms


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_inference_transform_matches_saved_stats_formula():
    transform = eval_transforms(
        mean=(2.0, 4.0),
        std=(2.0, 4.0),
        clip_min=(0.0, 0.0),
        clip_max=(10.0, 20.0),
        normalization_mode="stats",
    )
    image = np.array([[[4.0, 8.0], [20.0, -5.0]]], dtype=np.float32)
    result = transform(image=image, mask=np.zeros((1, 2), dtype=np.uint8))
    tensor = result["image"].numpy()
    expected = np.array(
        [
            [[1.0, 4.0]],
            [[1.0, -1.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(tensor, expected, rtol=0.0, atol=1e-6)


def test_inference_transform_matches_robust_minmax_formula():
    transform = eval_transforms(
        mean=(0.5,),
        std=(0.25,),
        clip_min=(0.0,),
        clip_max=(10.0,),
        normalization_mode="robust_minmax",
    )
    image = np.array([[[-5.0], [5.0], [20.0]]], dtype=np.float32)
    tensor = transform(image=image)["image"].numpy()
    expected = np.array([[[-2.0, 0.0, 2.0]]], dtype=np.float32)
    np.testing.assert_allclose(tensor, expected, rtol=0.0, atol=1e-6)


def test_vv_vh_deployment_runtime_does_not_import_training_augmentation_or_scipy(tmp_path: Path):
    stats_path = tmp_path / "normalization_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "split": "train",
                "input_modalities": ["vv", "vh"],
                "files_used": 1,
                "channels": [
                    {
                        "channel": "vv",
                        "clip_min": 0.0,
                        "clip_max": 1.0,
                        "mean": 0.25,
                        "std": 0.25,
                        "raw_mean": 0.25,
                        "raw_std": 0.25,
                        "robust_mean": 0.25,
                        "robust_std": 0.25,
                    },
                    {
                        "channel": "vh",
                        "clip_min": 0.0,
                        "clip_max": 1.0,
                        "mean": 0.5,
                        "std": 0.5,
                        "raw_mean": 0.5,
                        "raw_std": 0.5,
                        "robust_mean": 0.5,
                        "robust_std": 0.5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    script = r'''
import builtins
import sys
from pathlib import Path

import numpy as np
import torch

real_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    blocked = ("albumentations", "scipy", "floods.prepare")
    if any(name == item or name.startswith(item + ".") for item in blocked):
        raise AssertionError(f"unexpected deployment import: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
from floods.deployment import _build_tensor_from_arrays
from floods.model_factory import prepare_model
from floods.inference_transforms import eval_transforms

vv = np.full((4, 5), 0.5, dtype=np.float32)
vh = np.full((4, 5), 0.25, dtype=np.float32)
tensor, channels = _build_tensor_from_arrays(
    vv,
    vh,
    None,
    {"crs": None, "transform": None},
    ["vv", "vh"],
    "stats",
    Path(sys.argv[1]),
    torch.device("cpu"),
)
assert tensor.shape == (1, 2, 4, 5)
assert torch.isfinite(tensor).all()
assert set(channels) == {"vv", "vh"}
assert "floods.prepare" not in sys.modules
assert "albumentations" not in sys.modules
assert "scipy" not in sys.modules
print("deployment_runtime_isolated")
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(stats_path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "deployment_runtime_isolated" in result.stdout
