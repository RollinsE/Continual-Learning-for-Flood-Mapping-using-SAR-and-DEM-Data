from pathlib import Path

import pytest

from floods.colab import best_checkpoint, find_file, run, unzip, zip_dir


def test_zip_round_trip_keeps_relative_paths(tmp_path: Path):
    src = tmp_path / "processed"
    (src / "train").mkdir(parents=True)
    (src / "train" / "EMSR1-0_0_0.tif").write_bytes(b"x" * 10)
    (src / "activations.json").write_text("{}", encoding="utf-8")
    archive = zip_dir(src, tmp_path / "drive" / "processed.zip")
    restored = unzip(archive, tmp_path / "restored")
    assert (restored / "train" / "EMSR1-0_0_0.tif").read_bytes() == b"x" * 10
    assert find_file(restored, "activations.json") == restored / "activations.json"


def test_best_checkpoint_picks_highest_score(tmp_path: Path):
    models = tmp_path / "run" / "models"
    models.mkdir(parents=True)
    for name in ["model-004_best_f1-0.4301.pth", "model-006_best_f1-0.4452.pth", "model-005_best_f1-0.4400.pth", "last.ckpt"]:
        (models / name).write_text("x", encoding="utf-8")
    assert best_checkpoint(tmp_path / "run").name == "model-006_best_f1-0.4452.pth"
    assert best_checkpoint(tmp_path / "missing") is None


def test_run_streams_and_reports_failure(capsys):
    assert run("printf 'a\\rb\\n'; echo done") == 0
    assert "done" in capsys.readouterr().out
    with pytest.raises(RuntimeError):
        run("exit 3")
    assert run("exit 3", check=False) == 3
