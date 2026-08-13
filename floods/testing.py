from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from floods.config.testing import TestConfig
from floods.config.training import TrainConfig
from floods.datasets.flood import FloodDataset, RGBFloodDataset
from floods.logging.functional import plot_confusion_matrix
from floods.logging.tensorboard import TensorBoardLogger
from floods.prepare import inverse_transform, prepare_evaluation_dataset, prepare_model, prepare_test_metrics
from floods.trainer.callbacks import DisplaySamples
from floods.trainer.flood import FloodTrainer
from floods.utils.common import check_or_make_dir, get_logger, init_experiment, load_config, print_config
from floods.utils.gis import as_image, rgb_ratio
from floods.utils.ml import find_best_checkpoint, load_class_weights, seed_everything, seed_worker
from floods.utils.tiling import SmoothTiler

LOG = get_logger(__name__)


def test(test_config: TestConfig):
    if test_config.name is None:
        raise ValueError("An experiment name is required for testing")

    log_name = "output-test.log"
    exp_id, out_folder, model_folder, logs_folder = init_experiment(config=test_config, log_name=log_name)
    config_path = out_folder / "config.yaml"
    config: TrainConfig = load_config(path=config_path, config_class=TrainConfig)
    print_config(LOG, config)

    accelerator = Accelerator(fp16=config.trainer.amp, cpu=config.trainer.cpu)
    accelerator.wait_for_everyone()

    LOG.info("Using seed: %d", config.seed)
    seed_everything(config.seed)
    LOG.info("Loading test dataset")
    num_classes = 1
    test_dataset, modalities, use_rgb = prepare_evaluation_dataset(config, split="test")
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,  # Full-size test rasters are evaluated one at a time.
                             shuffle=False,
                             num_workers=test_config.trainer.num_workers,
                             worker_init_fn=seed_worker)
    # Reconstruct the architecture without downloading encoder weights.
    LOG.info("Preparing model")
    config.model.pretrained = False
    model = prepare_model(config=config, num_classes=num_classes, stage="test")
    # Use the requested checkpoint or select the best checkpoint from the run.
    if test_config.checkpoint_path is not None:
        ckpt_path = Path(test_config.checkpoint_path)
    else:
        ckpt_path = find_best_checkpoint(model_folder)
    # Multi-branch checkpoints may contain auxiliary parameters.
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    strict_load = not config.model.multibranch
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"), strict=strict_load)
    LOG.info("Model restored from: %s", str(ckpt_path))

    weights = None
    if config.data.class_weights:
        weights = load_class_weights(Path(config.data.class_weights), device=accelerator.device, normalize=False)
        LOG.info("Using class weights: %s", str(weights))
    loss = config.loss.instantiate(ignore_index=255, weight=weights)
    logger = TensorBoardLogger(log_folder=logs_folder, comment=config.comment, filename_suffix="-test")

    # Limit visual exports because full-size predictions are memory intensive.
    LOG.info("Storing predicted images: %s", str(test_config.store_predictions).lower())
    num_samples = int(test_config.store_predictions) * test_config.prediction_count
    LOG.info("Storing batches: %s", str(num_samples))

    tiler = SmoothTiler(tile_size=test_config.image_size,
                        channels_first=True,
                        batch_size=test_config.trainer.batch_size,
                        mirrored=False)
    trainer = FloodTrainer(accelerator=accelerator,
                           model=model,
                           criterion=loss,
                           tiler=tiler,
                           categories=test_dataset.categories(),
                           logger=logger,
                           sample_batches=num_samples,
                           stage="test",
                           debug=test_config.debug)
    image_trf = as_image if use_rgb else rgb_ratio
    slice_at = 3 if use_rgb else 2
    trainer.add_callback(DisplaySamples(inverse_transform=inverse_transform(test_dataset.mean(), test_dataset.std()),
                                        mask_palette=test_dataset.palette(),
                                        image_transform=image_trf,
                                        slice_at=slice_at,
                                        stage="test"))
    # Test metrics extend validation metrics with a confusion matrix.
    eval_metrics = prepare_test_metrics(config=test_config, device=accelerator.device)

    predictions_path = check_or_make_dir(out_folder / "images")
    losses, _ = trainer.predict(test_dataloader=test_loader,
                                metrics=eval_metrics,
                                logger_exclude=["conf_mat"],
                                output_path=predictions_path)
    scores = trainer.current_scores["test"]
    LOG.info("Testing completed, average loss: %.4f", np.mean(losses))

    LOG.info("Test metrics")
    classwise = dict()
    for i, (name, score) in enumerate(scores.items()):
        # Keep console output to scalar metrics; class-wise tensors remain available for plots.
        if score.ndim == 0:
            LOG.info(f"{name:<20s}: {score.item():.4f}")
        elif name != "conf_mat":
            classwise[name] = score


    LOG.info("Writing confusion matrix")
    cm_name = f"cm_{Path(ckpt_path).stem}"
    plot_folder = check_or_make_dir(out_folder / "plots")
    plot_confusion_matrix(scores["conf_mat"].cpu().numpy(),
                          destination=plot_folder / f"{cm_name}.png",
                          labels=trainer.categories.values(),
                          title=cm_name,
                          normalize=False)
    LOG.info("Testing complete")
