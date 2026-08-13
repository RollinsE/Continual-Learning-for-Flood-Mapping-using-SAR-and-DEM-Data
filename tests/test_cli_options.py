from floods.cli import build_cli_parser


def test_preprocess_tile_size_cli_accepts_supported_sizes():
    parser = build_cli_parser()
    args = parser.parse_args([
        'preprocess', '--config', 'configs/preprocess_mmflood.yaml',
        '--tile-size', '128', '--tile-max-overlap', '64'
    ])
    assert args.tile_size == 128
    assert args.tile_max_overlap == 64


def test_train_architecture_aliases():
    parser = build_cli_parser()
    args = parser.parse_args([
        'train', '--config', 'configs/train_segmentation_vv_vh_dem.yaml',
        '--architecture', 'deeplabv3p', '--backbone', 'resnet34',
        '--pretrained', '--image-size', '256'
    ])
    assert args.decoder == 'deeplabv3p'
    assert args.encoder == 'resnet34'
    assert args.pretrained is True
    assert args.image_size == 256


def test_train_resume_extension_options():
    parser = build_cli_parser()
    args = parser.parse_args([
        'train', '--config', 'configs/train_segmentation_vv_vh_dem.yaml',
        '--resume', '--extend-epochs', '10', '--reset-early-stopping'
    ])
    assert args.resume is True
    assert args.extend_epochs == 10
    assert args.reset_early_stopping_on_resume is True


def test_train_keep_early_stopping_option():
    parser = build_cli_parser()
    args = parser.parse_args([
        'train', '--config', 'configs/train_segmentation_vv_vh_dem.yaml',
        '--resume', '--epochs', '25', '--keep-early-stopping-state'
    ])
    assert args.resume is True
    assert args.epochs == 25
    assert args.reset_early_stopping_on_resume is False


def test_evaluate_accepts_no_pretrained_override():
    parser = build_cli_parser()
    args = parser.parse_args([
        'evaluate', '--config', 'run/config.yaml', '--checkpoint', 'model.pth',
        '--no-pretrained'
    ])
    assert args.pretrained is False


def test_ensemble_evaluate_accepts_no_pretrained_override():
    parser = build_cli_parser()
    args = parser.parse_args([
        'ensemble-evaluate', '--configs', 'a.yaml', 'b.yaml', '--checkpoints', 'a.pth', 'b.pth',
        '--no-pretrained'
    ])
    assert args.pretrained is False

from floods.cli import _apply_training_overrides, _load_config_from_yaml
from floods.config import TrainConfig


def test_stratified_sampler_cli_disables_event_balanced_config():
    parser = build_cli_parser()
    args = parser.parse_args([
        'train', '--config', 'configs/train_segmentation_vv_vh_dem.yaml',
        '--stratified-sampling'
    ])
    cfg = _load_config_from_yaml(args.config, TrainConfig)
    cfg.data.event_balanced_sampling = True
    cfg = _apply_training_overrides(cfg, args)
    assert cfg.data.stratified_sampling is True
    assert cfg.data.event_balanced_sampling is False
    assert cfg.data.weighted_sampling is False
    assert cfg.data.foreground_balanced_sampling is False


def test_hard_example_sampler_cli_disables_other_samplers():
    parser = build_cli_parser()
    args = parser.parse_args([
        'train', '--config', 'configs/train_segmentation_vv_vh_dem.yaml',
        '--hard-example-sampling', '--hard-example-csv', 'train_audit/tile_error_metrics.csv',
        '--hard-example-categories', 'false_negative_low_recall', 'poor_overlap',
        '--hard-example-fg-bins', 'tiny', 'small',
        '--hard-example-weight', '5.0', '--hard-example-max-fraction', '0.55'
    ])
    cfg = _load_config_from_yaml(args.config, TrainConfig)
    cfg.data.event_balanced_sampling = True
    cfg = _apply_training_overrides(cfg, args)
    assert cfg.data.hard_example_sampling is True
    assert cfg.data.event_balanced_sampling is False
    assert cfg.data.hard_example_csv == 'train_audit/tile_error_metrics.csv'
    assert cfg.data.hard_example_categories == ['false_negative_low_recall', 'poor_overlap']
    assert cfg.data.hard_example_fg_bins == ['tiny', 'small']
    assert cfg.data.hard_example_weight == 5.0
    assert cfg.data.hard_example_max_fraction == 0.55



def test_compare_models_cli_parses_single_and_ensemble():
    from floods.cli import build_cli_parser
    parser = build_cli_parser()
    args = parser.parse_args([
        "compare-models",
        "--output-dir", "/tmp/compare",
        "--model", "unet", "a.yaml", "a.pth",
        "--ensemble", "ens", "mean_logit", "a.yaml:a.pth", "b.yaml:b.pth",
        "--split", "val",
    ])
    assert args.command == "compare-models"
    assert args.model == [["unet", "a.yaml", "a.pth"]]
    assert args.ensemble == [["ens", "mean_logit", "a.yaml:a.pth", "b.yaml:b.pth"]]


def test_export_deployment_cli_parses_manifest_options():
    from floods.cli import build_cli_parser
    parser = build_cli_parser()
    args = parser.parse_args([
        "export-deployment",
        "--output-file", "/tmp/deploy.yaml",
        "--configs", "a.yaml", "b.yaml",
        "--checkpoints", "a.pth", "b.pth",
        "--threshold", "0.45",
    ])
    assert args.command == "export-deployment"
    assert args.threshold == 0.45
    assert len(args.configs) == 2
    assert args.copy_assets is True
    assert args.assets_directory == "assets"

    reference_args = parser.parse_args([
        "export-deployment",
        "--output-file", "/tmp/deploy.yaml",
        "--config", "a.yaml",
        "--checkpoint", "a.pth",
        "--threshold", "0.45",
        "--reference-only",
    ])
    assert reference_args.copy_assets is False


def test_compare_models_confusion_matrix_toggle_is_available():
    parser = build_cli_parser()
    args = parser.parse_args([
        "compare-models",
        "--output-dir", "/tmp/compare",
        "--model", "unet", "a.yaml", "a.pth",
        "--no-confusion-matrix-plots",
    ])
    assert args.command == "compare-models"
    assert args.confusion_matrix_plots is False


def test_predict_scene_cli_parses_manifest_inputs_and_outputs():
    parser = build_cli_parser()
    args = parser.parse_args([
        "predict-scene",
        "--manifest", "/tmp/deploy.yaml",
        "--sar-path", "/tmp/sar.tif",
        "--dem-path", "/tmp/dem.tif",
        "--output-mask", "/tmp/mask.tif",
        "--output-probability", "/tmp/prob.tif",
        "--cpu",
    ])
    assert args.command == "predict-scene"
    assert str(args.manifest).endswith("deploy.yaml")
    assert str(args.output_mask).endswith("mask.tif")
    assert str(args.output_probability).endswith("prob.tif")
    assert args.deployment_device == "cpu"


def test_discover_scene_cli_parses_output_file():
    parser = build_cli_parser()
    args = parser.parse_args([
        "discover-scene",
        "--scene-dir", "/tmp/EMSR001",
        "--output-file", "/tmp/inventory.csv",
    ])
    assert args.command == "discover-scene"
    assert str(args.scene_dir).endswith("EMSR001")
    assert str(args.output_file).endswith("inventory.csv")


def test_predict_scene_cli_parses_folder_visual_options():
    parser = build_cli_parser()
    args = parser.parse_args([
        "predict-scene",
        "--manifest", "/tmp/deploy.yaml",
        "--scene-dir", "/tmp/EMSR001",
        "--dem-dir", "/tmp/EMSR001",
        "--sar-selection", "latest",
        "--output-dir", "/tmp/predictions",
        "--write-html-report",
        "--display-inline",
        "--write-previews",
        "--write-overlay",
        "--write-uncertainty",
        "--explain",
        "--explain-per-modality",
        "--cpu",
    ])
    assert args.command == "predict-scene"
    assert str(args.scene_dir).endswith("EMSR001")
    assert args.sar_selection == "latest"
    assert args.write_html_report is True
    assert args.display_inline is True
    assert args.explain_per_modality is True
    assert args.deployment_device == "cpu"


def test_predict_scene_cli_parses_mask_and_output_mode_options():
    parser = build_cli_parser()
    args = parser.parse_args([
        "predict-scene",
        "--manifest", "/tmp/deploy.yaml",
        "--sar-path", "/tmp/sar.tif",
        "--dem-path", "/tmp/dem.tif",
        "--mask-path", "/tmp/gt.tif",
        "--evaluate",
        "--output-mode", "concise",
        "--prediction-only",
        "--cpu",
    ])
    assert args.command == "predict-scene"
    assert str(args.mask_path).endswith("gt.tif")
    assert args.evaluate is True
    assert args.output_mode == "concise"
    assert args.prediction_only is True


def test_predict_scene_cli_parses_input_csv_naming_and_mosaic_options():
    parser = build_cli_parser()
    args = parser.parse_args([
        "predict-scene",
        "--manifest", "/tmp/deploy.yaml",
        "--input-csv", "/tmp/scenes.csv",
        "--scene-id", "EMSR001",
        "--candidate-prefix", "demo",
        "--candidate-name-template", "{scene_id}_{date}_{stem}",
        "--mosaic-compatible-sar-tiles",
        "--mosaic-undated",
        "--cpu",
    ])
    assert args.command == "predict-scene"
    assert str(args.input_csv).endswith("scenes.csv")
    assert args.scene_id == "EMSR001"
    assert args.candidate_prefix == "demo"
    assert args.candidate_name_template == "{scene_id}_{date}_{stem}"
    assert args.mosaic_compatible_sar_tiles is True
    assert args.mosaic_undated is True


def test_discover_scene_cli_parses_naming_options():
    parser = build_cli_parser()
    args = parser.parse_args([
        "discover-scene",
        "--scene-dir", "/tmp/EMSR001",
        "--scene-id", "EMSR001",
        "--candidate-prefix", "candidate",
        "--candidate-name-template", "{scene_id}_{stem}",
    ])
    assert args.command == "discover-scene"
    assert args.scene_id == "EMSR001"
    assert args.candidate_prefix == "candidate"
    assert args.candidate_name_template == "{scene_id}_{stem}"


def test_predict_scene_mosaic_mode_defaults_to_smart():
    parser = build_cli_parser()
    args = parser.parse_args([
        "predict-scene",
        "--manifest", "/tmp/deploy.yaml",
        "--sar-path", "/tmp/sar.tif",
        "--dem-path", "/tmp/dem.tif",
        "--cpu",
    ])
    assert args.mosaic_mode == "smart"


def test_audit_deployment_errors_cli_accepts_multiple_directories(tmp_path):
    from floods.cli import build_cli_parser
    parser = build_cli_parser()
    args = parser.parse_args([
        "audit-deployment-errors",
        "--deployment-dir", str(tmp_path / "a"),
        "--deployment-dir", str(tmp_path / "b"),
        "--output-dir", str(tmp_path / "out"),
        "--max-montages", "12",
    ])
    assert args.command == "audit-deployment-errors"
    assert len(args.deployment_dirs) == 2
    assert args.max_montages == 12


def test_predict_scene_parses_window_diagnostics_option():
    parser = build_cli_parser()
    args = parser.parse_args([
        "predict-scene",
        "--manifest", "/tmp/deploy.yaml",
        "--sar-path", "/tmp/sar.tif",
        "--write-window-diagnostics",
        "--cpu",
    ])
    assert args.write_window_diagnostics is True


def test_train_sparse_crop_options_are_available():
    parser = build_cli_parser()
    args = parser.parse_args([
        'train', '--config', 'configs/train_segmentation_vv_vh_dem.yaml',
        '--sparse-crop-supervision',
        '--sparse-crop-normal-fraction', '0.5',
        '--sparse-crop-flood-fraction', '0.25',
        '--sparse-crop-hard-background-fraction', '0.25',
        '--sparse-crop-sizes', '256', '320', '384', '448',
        '--sparse-crop-attempts', '32',
        '--sparse-crop-hard-background-max-fg-ratio', '0.001',
    ])
    assert args.sparse_crop_supervision is True
    assert args.sparse_crop_sizes == [256, 320, 384, 448]
    assert args.sparse_crop_attempts == 32


def test_train_sparse_crop_overrides_reach_config():
    parser = build_cli_parser()
    args = parser.parse_args([
        'train', '--config', 'configs/train_segmentation_vv_vh_dem.yaml',
        '--image-size', '512',
        '--sparse-crop-supervision',
        '--sparse-crop-sizes', '256', '384',
    ])
    cfg = _load_config_from_yaml(args.config, TrainConfig)
    cfg = _apply_training_overrides(cfg, args)
    assert cfg.data.sparse_crop_supervision is True
    assert cfg.data.sparse_crop_sizes == [256, 384]


def test_hard_negative_region_sampler_cli_disables_other_samplers():
    parser = build_cli_parser()
    args = parser.parse_args([
        'train', '--config', 'configs/train_segmentation_vv_vh_dem.yaml',
        '--hard-negative-region-sampling',
        '--hard-negative-manifest', 'mining/hard_negative_regions.csv',
        '--hard-negative-region-weight', '3.0',
        '--hard-negative-region-max-fraction', '0.30',
        '--hard-negative-crop-probability', '0.9',
    ])
    cfg = _load_config_from_yaml(args.config, TrainConfig)
    cfg.data.event_balanced_sampling = True
    cfg = _apply_training_overrides(cfg, args)
    assert cfg.data.hard_negative_region_sampling is True
    assert cfg.data.event_balanced_sampling is False
    assert cfg.data.hard_example_sampling is False
    assert cfg.data.hard_negative_manifest == 'mining/hard_negative_regions.csv'
    assert cfg.data.hard_negative_region_weight == 3.0
    assert cfg.data.hard_negative_region_max_fraction == 0.30
    assert cfg.data.hard_negative_crop_probability == 0.9


def test_mine_hard_negatives_cli_parses_region_options():
    parser = build_cli_parser()
    args = parser.parse_args([
        'mine-hard-negatives', '--config', 'run/config.yaml',
        '--checkpoint', 'best.pth', '--processed-data-dir', '/data',
        '--output-dir', '/out', '--threshold', '0.65',
        '--crop-sizes', '256', '320', '--min-fp-pixels', '200',
        '--max-regions-per-tile', '2', '--no-pretrained', '--gpu',
    ])
    assert args.command == 'mine-hard-negatives'
    assert args.threshold == 0.65
    assert args.crop_sizes == [256, 320]
    assert args.min_fp_pixels == 200
    assert args.max_regions_per_tile == 2
    assert args.pretrained is False
    assert args.cpu is False


def test_train_init_checkpoint_is_distinct_from_resume():
    parser = build_cli_parser()
    args = parser.parse_args([
        'train', '--config', 'configs/train_segmentation_vv_vh_dem.yaml',
        '--init-checkpoint', 'baseline_best.pth', '--no-resume'
    ])
    cfg = _load_config_from_yaml(args.config, TrainConfig)
    cfg = _apply_training_overrides(cfg, args)
    assert cfg.init_checkpoint == 'baseline_best.pth'
    assert cfg.resume is False
    assert cfg.resume_from is None


def test_derived_feature_commands_and_channel_adaptation_parse():
    parser = build_cli_parser()
    derive = parser.parse_args([
        "derive-features",
        "--processed-data-dir", "/tmp/mmflood_processed",
        "--tpi-radius-pixels", "15",
    ])
    assert derive.command == "derive-features"
    assert derive.tpi_radius_pixels == 15

    audit = parser.parse_args([
        "audit-feature-separability",
        "--processed-data-dir", "/tmp/mmflood_processed",
        "--output-dir", "/content/audit",
    ])
    assert audit.command == "audit-feature-separability"
    assert audit.extended_modalities[-3:] == ["vv_vh_log_ratio", "dem_slope", "dem_tpi"]

    train = parser.parse_args([
        "train",
        "--config", "configs/train_segmentation_vv_vh_dem.yaml",
        "--input-modalities", "vv", "vh", "dem", "vv_vh_log_ratio", "dem_slope", "dem_tpi",
        "--init-channel-adaptation", "zero_extra",
    ])
    assert train.input_modalities[-1] == "dem_tpi"
    assert train.init_channel_adaptation == "zero_extra"


def test_fit_normalization_parses_preserved_baseline_stats():
    parser = build_cli_parser()
    args = parser.parse_args([
        "fit-normalization",
        "--processed-data-dir", "/tmp/mmflood_processed",
        "--output-file", "/content/six_stats.json",
        "--input-modalities", "vv", "vh", "dem", "vv_vh_log_ratio", "dem_slope", "dem_tpi",
        "--preserve-channel-stats-from", "/content/baseline_stats.json",
    ])
    assert str(args.preserve_channel_stats_from).endswith("baseline_stats.json")


def test_prepare_derived_experiment_cli_parses():
    parser = build_cli_parser()
    args = parser.parse_args([
        "prepare-derived-experiment",
        "--base-config", "baseline/config.yaml",
        "--baseline-checkpoint", "baseline/model.pth",
        "--normalization-stats-path", "derived/stats.json",
        "--output-config", "derived/config.yaml",
        "--run-id", "derived_v088",
        "--artifacts-dir", "runs",
    ])
    assert args.command == "prepare-derived-experiment"
    assert args.max_skipped_batch_fraction == 0.02


def test_water_prior_audit_cli_parses_sweep_options():
    parser = build_cli_parser()
    args = parser.parse_args([
        "audit-water-prior",
        "--config", "run/config.yaml",
        "--checkpoint", "model.pth",
        "--processed-data-dir", "/content/processed",
        "--output-dir", "/content/output",
        "--prior-cache-dir", "/content/prior",
        "--include-events", "EMSR342",
        "--model-thresholds", "0.45", "0.50",
        "--occurrence-thresholds", "90", "95",
        "--penalty-strengths", "0.5", "1.0",
        "--min-component-areas", "96",
        "--no-pretrained",
        "--no-amp",
        "--gpu",
    ])
    assert args.command == "audit-water-prior"
    assert args.include_events == ["EMSR342"]
    assert args.model_thresholds == [0.45, 0.50]
    assert args.occurrence_thresholds == [90, 95]
    assert args.penalty_strengths == [0.5, 1.0]
    assert args.min_component_areas == [96]
    assert args.pretrained is False
    assert args.amp is False
    assert args.cpu is False


def test_hysteresis_audit_cli_parses_sweep_options():
    parser = build_cli_parser()
    args = parser.parse_args([
        "audit-hysteresis-postprocess",
        "--config", "run/config.yaml",
        "--checkpoint", "model.pth",
        "--processed-data-dir", "/content/processed",
        "--output-dir", "/content/output",
        "--include-events", "EMSR342",
        "--fixed-thresholds", "0.30", "0.50",
        "--low-thresholds", "0.20", "0.30",
        "--high-thresholds", "0.50", "0.60",
        "--min-seed-pixels", "1", "16",
        "--min-component-areas", "96",
        "--max-empty-fp-rate-increase", "0.05",
        "--no-pretrained",
        "--no-amp",
        "--gpu",
    ])
    assert args.command == "audit-hysteresis-postprocess"
    assert args.include_events == ["EMSR342"]
    assert args.fixed_thresholds == [0.30, 0.50]
    assert args.low_thresholds == [0.20, 0.30]
    assert args.high_thresholds == [0.50, 0.60]
    assert args.min_seed_pixels == [1, 16]
    assert args.min_component_areas == [96]
    assert args.max_empty_fp_rate_increase == 0.05
    assert args.pretrained is False
    assert args.amp is False
    assert args.cpu is False
