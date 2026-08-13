from pathlib import Path

from floods.cli import _load_config_from_yaml
from floods.config import TrainConfig


def test_training_configs_load():
    for path in Path('configs').glob('train_*.yaml'):
        config = _load_config_from_yaml(path, TrainConfig)
        assert config.image_size == 256
        assert config.data.in_channels == 3
        assert config.trainer.monitor.name == 'f1'
        assert config.loss.target.name == 'bce_tversky'

import yaml
from floods.utils.common import store_config


def test_store_config_writes_plain_yaml(tmp_path):
    cfg = TrainConfig()
    cfg.trainer.monitor = cfg.trainer.val_metrics[0]
    path = tmp_path / 'config.yaml'
    store_config(cfg, path)
    text = path.read_text()
    assert '!!python' not in text
    loaded = yaml.safe_load(text)
    assert loaded['loss']['target'] == cfg.loss.target.name
    assert loaded['optimizer']['target'] == cfg.optimizer.target.name
