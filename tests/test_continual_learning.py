import json
from pathlib import Path

from floods.cli import build_cli_parser
from floods.continual import ReplayBuffer, build_cl_tasks, event_id_from_path, load_event_years, parse_year_ranges


class DummyDataset:
    def __init__(self, names):
        self.image_files = names


def test_parse_year_ranges_accepts_ranges_and_single_years():
    assert parse_year_ranges(["2014-2017", "2021"]) == [(2014, 2017), (2021, 2021)]


def test_event_year_loading_and_task_assignment(tmp_path: Path):
    metadata = {
        "EMSR100": {"start": "2014-01-02T00:00:00"},
        "EMSR200": {"start": "2020-05-01T00:00:00"},
    }
    path = tmp_path / "activations.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    years = load_event_years(path)
    assert years == {"EMSR100": 2014, "EMSR200": 2020}
    train = DummyDataset(["/data/EMSR100-0-0_0_0.tif", "/data/EMSR200-0-0_0_0.tif"])
    eval_ds = DummyDataset(["/data/EMSR200-0-1_0_0.tif"])
    tasks = build_cl_tasks(train, eval_ds, years, [(2014, 2017), (2020, 2021)])
    assert tasks[0].train_indices == [0]
    assert tasks[0].eval_indices == []
    assert tasks[1].train_indices == [1]
    assert tasks[1].eval_indices == [0]


def test_replay_buffer_is_bounded():
    buffer = ReplayBuffer(max_size=3, seed=1)
    buffer.add_many(range(10))
    assert len(buffer) == 3
    assert all(isinstance(v, int) for v in buffer.indices)


def test_event_id_from_path_extracts_emsr_code():
    assert event_id_from_path("/tmp/EMSR470-0-3_0_0.tif") == "EMSR470"


def test_continual_train_cli_parses_core_options():
    parser = build_cli_parser()
    args = parser.parse_args([
        "continual-train",
        "--config", "configs/train_segmentation_vv_vh_dem.yaml",
        "--activations-json-path", "data/activations.json",
        "--strategies", "random", "entropy",
        "--task-year-ranges", "2014-2017", "2018-2019", "2020-2021",
        "--epochs-per-task", "3",
        "--replay-buffer-size", "50",
        "--replay-batch-size", "8",
        "--cl-eval-split", "val",
    ])
    assert args.command == "continual-train"
    assert args.strategies == ["random", "entropy"]
    assert args.epochs_per_task == 3
    assert args.replay_buffer_size == 50
    assert args.replay_batch_size == 8



def test_continual_train_cli_parses_ensemble_options():
    parser = build_cli_parser()
    args = parser.parse_args([
        "continual-train",
        "--config", "configs/train_segmentation_vv_vh_dem.yaml",
        "--activations-json-path", "data/activations.json",
        "--strategies", "random",
        "--cl-model-mode", "ensemble",
        "--ensemble-members", "unet:resnet50", "deeplabv3p:resnet50",
        "--ensemble-method", "mean_logit",
    ])
    assert args.cl_model_mode == "ensemble"
    assert args.ensemble_members == ["unet:resnet50", "deeplabv3p:resnet50"]
    assert args.ensemble_method == "mean_logit"


def test_region_tasks_from_country_and_coordinates(tmp_path: Path):
    from floods.cl_regions import load_event_regions, parse_region_tasks
    from floods.continual import build_region_tasks

    metadata = {
        "EMSR100": {"start": "2014-01-02", "country": "Italy"},
        "EMSR200": {"start": "2020-05-01", "country": "Viet Nam"},
        "EMSR300": {"start": "2019-03-01", "country": "Mozambique"},
        "EMSR400": {"start": "2019-03-01", "lat": -33.9, "lon": 151.2},  # no country -> coordinates (Sydney)
    }
    path = tmp_path / "activations.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    regions, _ = load_event_regions(path)
    assert regions == {"EMSR100": "europe", "EMSR200": "asia", "EMSR300": "africa", "EMSR400": "oceania"}
    specs = parse_region_tasks(["europe", "asia", "africa+oceania"])
    assert specs[2] == ("africa+oceania", ("africa", "oceania"))
    train = DummyDataset([f"/data/{e}-0-0_0_0.tif" for e in metadata])
    tasks = build_region_tasks(train, train, regions, specs)
    assert [t.train_indices for t in tasks] == [[0], [1], [2, 3]]


def test_region_task_parsing_rejects_overlap_and_unknown():
    import pytest
    from floods.cl_regions import parse_region_tasks

    with pytest.raises(ValueError):
        parse_region_tasks(["europe", "europe+asia"])
    with pytest.raises(ValueError):
        parse_region_tasks(["atlantis"])
