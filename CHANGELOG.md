# Changelog

## 0.15.20 — 2026-08-13

- Fixed equal-area collection mosaicking under newer Rasterio/GDAL versions by keeping valid background value `0` separate from NoData handling during binary-mask reprojection.
- Preserved coverage through the existing explicit coverage raster, avoiding background pixels being reinterpreted as flood.
- Updated collection preview colormap handling for current Matplotlib releases without changing the map palette or output semantics.

## 0.15.19 — 2026-08-13

- Reworked the public README and Streamlit guide for clearer, more natural technical writing while keeping the same functionality and documented limitations.
- Updated the package description to reflect Sentinel-1 SAR/DEM training, evaluation, continual learning and deployment without marketing-style wording.
- Corrected the Colab guide so its archive example uses the current `flood_extent_mapping_v<version>.zip` release name.

## 0.15.18 — 2026-08-13

- Made no-input-coverage areas explicit in whole-area Streamlit and HTML-report previews so mosaic gaps cannot be mistaken for valid non-flood predictions.
- Standardised collection previews on the viridis probability/flood palette with neutral grey reserved for no-data pixels.
- Added a concise map legend beneath whole-area previews without changing inference, geospatial aggregation, or GeoTIFF outputs.

## 0.15.17 — 2026-08-13

- Sanitized portable deployment normalization assets so local training provenance paths such as `processed_data_dir` and `preserve_channel_stats_from` are not copied into public bundles.
- Preserved the fitted channel statistics used by inference and left the original training normalization artifact untouched.
- Updated deployment-bundle inventory hashes/sizes from the sanitized copy and added regression coverage for path-neutral normalization metadata.

## 0.15.16 — 2026-08-12

- Renamed the public software identity to **Flood Extent Mapping** and the CLI to `floodmap`; MMFlood remains identified as the reference dataset/upstream project where appropriate.
- Added Streamlit multi-model bundle discovery and modality-driven input forms.
- Added automatic SAR band inspection, safe VV/VH single-band pairing, multi-tile upload handling, and DEM/mask matching by canonical tile ID.
- Fixed candidate-specific mask handling for deployment CSV inputs.
- Replaced shape-only VH/DEM alignment with geospatial reprojection to the SAR grid.
- Added post-inference equal-area collection mosaics so tiles with different source resolutions can be combined without resampling SAR before inference.
- Added area-based collection summaries, whole-area downloads, equal-area labelled evaluation, pooled fallback metrics and a collection HTML report.
- Removed the last shape-only mask-alignment fallback; evaluation now requires geospatial metadata whenever a mask grid differs from its SAR grid.
- Added GitHub CI and expanded release-hygiene regression coverage.

## 0.15.8 - 2026-08-07

- Completed a GitHub-readiness cleanup across source comments, public logs, documentation, Streamlit error handling, and release metadata without changing model behaviour.
- Removed conversational/inherited development comments and corrected wording in preprocessing, metrics, tiling, testing, logging, and model-summary modules.
- Replaced Colab-specific wording in core runtime logs with environment-neutral interruption and notebook/shell language; Colab-specific instructions remain isolated in `docs/COLAB.md`.
- Fixed Streamlit deployment distribution so the portable checkpoint is not excluded by the repository-wide model-artifact ignore rules and remains tracked through Git LFS.
- Added Streamlit upload validation for all optional rasters, friendly public error reporting, server-side traceback logging, and cleanup of the previous session workspace before a new prediction.
- Made event-CV examples path-neutral, removed stale release-number instructions from active documentation, and tightened the code-quality audit to flag genuine release artefacts rather than legitimate compatibility terminology.
- Added explicit attribution to the original MMFlood codebase and publication in the README and notice.
- Fixed two latent compatibility-code defects found during the GitHub audit: DeepLab/Xception middle-flow registration and backbone validation, plus rectangular/fixed-overlap tiling coordinate handling.
- Standardised compatibility U-Net block/class naming without changing checkpoint state-dict keys, and added regression tests for the repaired architectures and tilers.
- Expanded citation and licensing metadata, documented Git LFS staging for the public Streamlit bundle, and made the README standard-Python-first rather than notebook-first.
- Kept automated tests in the source repository while excluding the `tests` package from built wheels, and aligned contribution checks with the Streamlit source tree.
- Replaced legacy `yaml.UnsafeLoader` fallbacks with `yaml.FullLoader`, retaining tuple-compatible historical configuration loading without permitting arbitrary Python object construction.

## 0.15.7 - 2026-08-07

- Added a first-party Streamlit web interface that calls the same `predict_scene` deployment engine used by the CLI rather than maintaining separate inference logic.
- Added SAR upload, optional DEM and ground-truth mask upload, CPU inference, prediction/evaluation metrics, preview maps, embedded HTML report and downloadable GeoTIFF/report/output ZIP artefacts.
- Added a dedicated `streamlit_app/requirements.txt` so Community Cloud installs a lean deployment environment instead of the full training/foundation-model dependency set.
- Added portable-bundle discovery through `streamlit_app/deployment/deployment_manifest.yaml` with an environment-variable override for other standard runtimes.
- Added Streamlit Community Cloud configuration and Git LFS tracking rules for bundled checkpoint assets.
- Retained portable deployment bundles, notebook-safe reports, deployment dependency isolation and event-CV resume handling from earlier releases.

## 0.15.6 - 2026-08-06

- Fixed Colab/Jupyter inline-report CSS leakage that could restyle earlier console output as white blocks after `--display-inline` rendered a report.
- Scoped every deployment-report selector beneath a dedicated MMFlood report root instead of using global `body`, `pre`, `code`, `table`, `img`, heading, and universal selectors.
- Inline display now strips document-level `html`, `head`, `body`, and doctype markup before passing the saved report to IPython, while preserving the scoped report styles and content.
- Applied the same notebook-safe isolation to deployment summary reports and added regression tests that reject unscoped notebook selectors.
- Retained portable deployment bundles, deployment dependency isolation, and event-CV interruption/resume handling from earlier releases.

## 0.15.5 - 2026-08-06

- Changed `export-deployment` to create a self-contained portable deployment bundle by default. Configurations, checkpoints and normalization statistics are copied beside the manifest under an `assets/` directory.
- Deployment manifests now use paths relative to the manifest directory and continue to work after the whole bundle is moved to another directory, computer or mounted storage location.
- Added a bundle inventory with SHA-256 hashes and file sizes, plus a generated deployment README with path-neutral CLI usage.
- Added explicit `--reference-only` for advanced environment-dependent manifests; portability is now the safe default rather than an optional convention.
- Added regression tests that export a bundle, move it to a different directory and load the model configuration, checkpoint and normalization assets exclusively through relative paths.
- Retained the deployment dependency isolation from v0.15.4 and the event-CV interruption/resume repair from v0.15.2.

## 0.15.4 - 2026-08-06

- Removed the training augmentation stack from the deployment import path: `predict-scene` no longer imports `floods.prepare`, Albumentations, or SciPy for ordinary VV/VH inference.
- Moved model reconstruction into the dependency-light `floods.model_factory` module shared by training and deployment.
- Added deterministic NumPy/PyTorch inference transforms that reproduce saved clipping and normalization without Albumentations.
- Added deployment-runtime isolation tests that block `albumentations`, `scipy`, and `floods.prepare` while building a VV/VH inference tensor.
- Retained the v0.15.3 modality/SciPy isolation and the v0.15.2 event-CV resume repair.

## 0.15.3 - 2026-08-06

- Decoupled modality validation from derived-feature processing so loading VV/VH training and deployment configurations no longer imports SciPy, Rasterio, or feature-generation code.
- Added a lightweight `floods.modalities` module and migrated CLI, configuration, event-CV, dataset, normalisation, pretrained-model and deployment callers to it.
- Made SciPy lazy and feature-specific: it is imported only when `dem_tpi` is actually calculated; VV/VH ratio and DEM slope derivation do not require SciPy.
- Added `floodmap doctor` with NumPy, SciPy, Rasterio and PyTorch runtime probes, including detection of mixed NumPy installations that fail in `numpy.char`.
- Added a tested Colab numeric-stack constraints file and a full installer that reinstalls NumPy/SciPy/Rasterio together before package installation.
- Added deployment/configuration regression tests that explicitly block SciPy imports, plus tests for lightweight derived-feature paths and the environment report.
- Retained the 0.15.2 event-CV interruption/resume repair.

## 0.15.2 - 2026-07-31

- Fixed event-CV resume handling so interrupted folds are never written or treated as completed.
- Re-running the same event-CV command now resumes the interrupted fold from `models/last.ckpt` and stops the matrix immediately when an interruption occurs.
- Added automatic repair for 0.15.1 `cv_result.json` files whose recorded `stop_reason` is `interrupted`; stale matrix rows are removed and the marker is archived before resume.
- Added explicit versioned completion markers, regression tests, clearer CLI/documentation wording, and repository-release cleanup.

## 0.15.1 - 2026-07-29

- Fixed reloading saved provider-aware run configurations whose normalisation mode is `terramind_v1` or `ssl4eo_s1`.
- Restored out-of-fold calibration, evaluation reconstruction, resume and deployment loading for TerraMind and SSL4EO/FG-MAE runs.
- Added regression coverage for provider-normalisation configuration round trips.

## 0.15.0 - 2026-07-29

- Unified ordinary training, event-level cross-validation, out-of-fold calibration, evaluation reconstruction and deployment around one pretrained-model registry.
- Added CLI-selectable ImageNet, SSL4EO/DeCUR Sentinel-1, FG-MAE SAR, CROMA SAR, and TerraMind Tiny/Small/Base sources without requiring users to edit package configurations.
- Added `--candidates SOURCE[:DECODER[:ENCODER[:LABEL]]]` to `event-cv` and `--candidate` to `calibrate-event-cv`; legacy architecture flags remain readable.
- Preserved deterministic complete-event folds, fold-specific leakage-free normalisation, event-macro checkpoint selection, worst-event reporting and strict test isolation across every registered architecture.
- Added neutral `configs/training_defaults.yaml`; resolved run settings continue to be written automatically to each run directory.
- Included TorchGeo, TerraTorch, Hugging Face Hub and Safetensors in the standard requirements and Conda environment.
- Added provider-aware CV manifests/results, median best-epoch summaries, provider input normalisation and regression tests.
- Added shared event-CV CLI overrides for encoder/decoder learning rates, optimiser, scheduler, loss, loss parameters, weight decay and augmentation without editing package files.

## 0.11.0 - 2026-07-28

- Added a locked, path-neutral reproduction workflow for the selected U-Net/ResNet34 VV+VH model.
- Added reference preprocessing and final-training configurations, a model card, data contract, machine-readable reference metrics, environment capture, CI, contribution guidance, citation metadata, and security guidance.
- Rewrote the README around the supported production profile and removed generated cache files and obsolete per-release notes from the distribution.
- Standardised the final evaluation command on batch size 1 for memory-safe inference on a 15 GB Colab GPU.

## 0.10.4

- Added `calibrate-event-cv` for one-pass, evaluation-only threshold calibration from completed out-of-fold checkpoints.
- Selects a single threshold by pooled out-of-fold event-macro F1, with worst-event F1 and global F1 as tie-breakers.
- Writes fold, event, and pooled threshold tables plus a machine-readable `calibration.json`.
- Loads each fold's preserved run configuration and leakage-free normalization statistics; no retraining occurs.

## 0.10.3

- Replaced the unsupported `segformer:mit_b0` event-CV default with the timm-supported `segformer:pvt_v2_b0`.
- Added PVT feature-index and encoder-argument handling for the SegFormer decoder.
- Added event-CV architecture preflight validation before fold normalization is fitted.
- Updated event-CV examples and regression tests.


## 0.10.2

- Added leakage-controlled EMSR event cross-validation with deterministic tile-balanced folds.
- Added fold-specific normalization fitted only on the training events in each fold.
- Added event-macro F1 checkpoint selection, worst-event F1 reporting, per-event metrics, and cumulative ranked summaries.
- Added fresh architecture/modality comparisons for U-Net, DeepLabV3+, and a SegFormer-style MiT decoder across VV+VH and VV+VH+DEM.
- Added staged Colab execution, completed-run skipping, interrupted-run resume, and protection against reusing an output directory with a different fold plan.
- Added regression tests for fold disjointness, event-macro metrics, staged result preservation, plan compatibility, CLI parsing, and resume signatures.

## 0.10.1

- Added `audit-modality-ablation` for checkpoint-level VV/VH/DEM occlusion audits.
- Reports full-split and target-event threshold sweeps, fixed-threshold metrics, per-tile deltas, and normalized-channel ablation semantics.

## 0.10.0

- Added `audit-domain-failure-link` to connect target-event errors with tile-domain features and quantify training-analogue coverage.
- Split deployable sensor/terrain statistics from mask-conditioned input summaries in `audit-domain-shift`.
- Domain-shift diagnosis now uses deployable all-pixel sensor/terrain features rather than any mask-conditioned statistics.
- Added nearest-neighbour support diagnostics, failure-feature correlations, failure classifier outputs, and training-analogue exports.

## 0.9.9

- Added `audit-domain-shift` for direct train-versus-target-event analysis across VV/VH/DEM pixel distributions, normalization clipping, tile composition, flood geometry, and connected-component structure.
- Added separate sensor/terrain, label-geometry, and combined tile-level domain classifiers so target-event failures are not described as generic domain shift without evidence.
- Added closest-training-event ranking, target feature z-scores, compact plots, persistent CSV/JSON outputs, and Colab progress logging.
- Added CLI and synthetic-raster regression tests for target-event selection and diagnostic output generation.

## 0.9.8

- Reduced plain Colab progress output to roughly ten updates per training or validation pass instead of about forty.
- Suppressed redundant command heartbeats whenever recent progress or other meaningful output is already visible.
- Kept complete threshold-sweep tables in `output.log` while printing only the best operating point to the console.
- Added a clear session separator when appending to an existing run log, including resumed runs.
- Added regression tests for progress cadence, heartbeat suppression, file-only detail records, and session boundaries.

## 0.9.7

- Fixed interrupted training resumes so a partial epoch no longer overwrites the last completed checkpoint or causes the remainder of that epoch to be skipped.
- Partial interrupted state is preserved separately as `interrupted_partial.ckpt`; first-epoch interruptions restart the same epoch.
- Added a warning when the hard-negative sampler cap is at or below the natural hard-negative tile prevalence.
- Updated the hard-negative fine-tuning guidance to use a controlled sampling mass and crop probability.

## 0.9.6

- Fixed undefined `norm_mode` metadata in `error-audit` and `ensemble-error-audit`.
- Added audit normalization-mode regression tests.

## 0.9.5

- Added a Colab-safe dependency file that preserves the preinstalled PyTorch/CUDA stack.
- Added a logged Colab installer with before/after PyTorch runtime checks.
- Prevented deleted-current-directory failures by documenting `/content` as the installation base.

## 0.9.4 — Pipeline-wide CLI logging and AMP recovery

- Standardised immediate timestamped console and file logging across every `mmflood` subcommand.
- Added command lifecycle records, elapsed time, tracebacks, and configurable running heartbeats.
- Added common `--log-file`, `--log-level`, `--plain-progress`, `--dynamic-progress`, and `--heartbeat-seconds` options to every command.
- Replaced remaining pipeline `print` calls with logger output.
- Added one-shot full-precision retry for AMP batches with non-finite gradients.
- Added overflow, recovery, skip, and gradient-norm telemetry plus resumable numerical-stability counters.
- Updated README, Colab, GroupDRO, logging and release documentation to use direct CLI commands.
- Added regression tests for CLI logging and AMP recovery.

## 0.9.3 — Event-level GroupDRO training

- Added `--group-dro` to optimise an event-robust objective over EMSR training events instead of ordinary average loss.
- Added exponentiated-gradient event weights, an ERM warm-up, a minimum event-probability floor, and per-epoch event-weight/loss reporting.
- Added strict resumable checkpoint support for GroupDRO state and training-plan validation.
- Added event-indexed dataset wrapping without changing existing sampler indices or validation behaviour.
- Added unit tests for event extraction, group aggregation, robust weighting, batch metadata, CLI parsing, and resume signatures.

## 0.9.2 — Seeded hysteresis post-processing audit

- Added `audit-hysteresis-postprocess` for connected low-threshold region growth anchored by high-confidence seed pixels.
- Added fixed-threshold, recall, and empty-scene false-positive comparators so threshold tuning is not misreported as a hysteresis gain.
- Added tile-, event-, and aggregate-level audit outputs plus regression tests.

## 0.9.1 — Water-prior decision correction

- Fixed `audit-water-prior` so model-threshold tuning is not misreported as a permanent-water-prior gain.
- Added explicit prior/no-prior selection, matched-threshold incremental metrics, and `recall_guard_eligible` to the sweep CSV.
- Recommendations now compare the best recall-eligible prior setting against the best recall-eligible no-prior setting.
- Added regression tests for the threshold-only gain failure mode.

## 0.9.0 — Permanent-water prior audit

- Added `audit-water-prior` to align JRC Global Surface Water occurrence to processed MMFlood tiles.
- Added hard exclusion and soft probability-penalty sweeps without retraining the model.
- Added explicit baseline reference, recall-guarded model selection, event comparisons, label-overlap diagnostics, and persistent prior caching.
- Added offline-cache and incomplete-coverage controls plus source-period leakage warnings.
- Added Planetary Computer/STAC dependencies and unit tests for prior adjustment, selection, local alignment, and CLI parsing.

## 0.8.9 — CRS-aware terrain derivatives

- Fixed DEM slope derivation for EPSG:4326 tiles by converting geographic pixel spacing from degrees to metres before computing gradients.
- Added projected-CRS unit conversion and support for rotated affine transforms.
- Derived rasters now carry schema version 2 plus metre-spacing metadata.
- `derive-features` automatically regenerates stale schema-v1 rasters, so the previously saturated slope files are not silently reused.
- Deployment-time derived features now use the same CRS-aware slope calculation as processed training tiles.
- Added regression tests for geographic slope, projected slope, and stale-feature regeneration.

## 0.8.8 — Derived flood-context channels

- Added reproducible VV-VH log-ratio, DEM slope, and local topographic-position GeoTIFF generation with `floodmap derive-features`.
- Added train-to-validation, global and event-level feature-separability audits before deep training.
- Added explicit ordered `data.input_modalities` support across training, evaluation, error audits, ensemble evaluation, normalization, and deployment.
- Added six-channel normalization fitting that can preserve the retained baseline VV/VH/DEM statistics exactly.
- Added `zero_extra` checkpoint input adaptation so a six-channel model starts from the retained three-channel behaviour while learning added channels from zero.
- Added `floodmap prepare-derived-experiment` to create a controlled warm-start config with specialist samplers and crops disabled.
- Added a configurable non-finite batch budget so unstable runs abort instead of silently skipping most optimiser updates.
- Added derived-channel, normalization, configuration, checkpoint-adaptation, and feature-audit tests.

## 0.8.7 - Audit-guided hard-positive regions

- Added `mine-hard-positives`, hard-positive region crop supervision/sampling, CLI flags, and strict resume coverage.
- Recommended hard-positive runs disable AMP after non-finite gradients in tile-level replay.

## 0.8.6 - Resume provenance checkpoint handling

- Fixed strict resume for fine-tuning runs whose preserved `config.yaml` contains `init_checkpoint`.
- Resume now validates the saved training plan first, then clears `init_checkpoint` only in the runtime configuration so state is restored exclusively from `last.ckpt` or `--resume-from`.
- The original run configuration remains unchanged for provenance.
- Added regression coverage for resuming hard-example runs initialised from a baseline checkpoint.

## 0.8.5 - Strict resume plan preservation

- Resume now automatically loads the authoritative `config.yaml` from the existing run directory.
- Resume refuses sampler, epoch-target, optimizer, loss, augmentation, or model-plan changes.
- Checkpoints embed a training-plan signature and reject incompatible future resumes.
- The original run config is preserved; resume invocations are written to `last_resume_config.yaml`.
- Training logs now print target epochs, patience, sampler, batch size, and branch learning rates before dataset loading.


## 0.8.4 — Audit-guided hard-negative regions

- Added `floodmap mine-hard-negatives` to mine high-confidence false-positive crop regions from a labelled training split.
- Added `hard_negative_regions.csv` manifests with exact tile coordinates, crop sizes, false-positive counts, confidence, and foreground contamination checks.
- Added audit-guided hard-negative region crop supervision and a capped replacement sampler.
- Added `--init-checkpoint` for genuine fresh fine-tuning from model weights without restoring optimiser, scheduler, epoch, or early-stopping state.
- Added strict incompatibility checks between random sparse crop supervision and audit-guided region crops.
- Added crop-usage logging for `audit-hard-negative` samples.
- Added unit and CLI tests for mining manifests, region crops, sampler activation, and initialisation checkpoints.

## 0.8.3

- Made `--resume` strict: a missing checkpoint now raises a clear error instead of silently restarting at epoch 1.
- Resolves the resume checkpoint before model construction and skips redundant pretrained-weight downloads.
- Falls back to the newest `epoch_*.ckpt` when `last.ckpt` is unavailable.
- Writes `last.ckpt` atomically to reduce checkpoint corruption risk on Google Drive.
- Restores checkpoint callback state consistently after resume.

## 0.8.2

- Made ordinary `mmflood` CLI commands stream logs immediately in Google Colab, notebook shell cells, redirected output, and `tee` pipelines.
- Added automatic line-buffered stdout/stderr configuration; `python -u` and custom subprocess wrappers are no longer required.
- Added newline-based progress heartbeats when dynamic carriage-return bars are unsuitable, while preserving tqdm bars in real terminals.
- Added visible mask-scan progress during sampler preparation so long dataset setup stages never appear frozen.
- Avoided tqdm logging redirection in captured consoles, preventing notebook output from being held until a command finishes.

## 0.8.1

- Added real sparse-flood crop supervision after SAR/DEM stacking, with configurable full-tile, flood-centred, and hard-background modes.
- Added variable crop sizes that are resized back to the configured model input size; the default 512px experiment uses 256, 320, 384, and 448px crops.
- Empty tiles remain full-size samples, while hard-background crops are searched only inside flood-containing tiles.
- Added epoch-level crop telemetry for requested/applied modes, fallbacks, and crop-size distribution.
- Reworked the legacy `crop_aware` augmentation profile so it activates the real crop-supervision path instead of attempting a no-op full-size crop.
- Added CLI/config controls for crop fractions, sizes, search attempts, valid-pixel requirements, and hard-background flood tolerance.
- Made epoch summaries report the same best-threshold validation operating point used for checkpoint selection, removing the previous fixed-threshold versus threshold-sweep ambiguity.
- Added sparse-crop unit tests and continual-learning batch compatibility.

## 0.8.0 - 2026-07-21

- Promoted the audited square-root tempered event sampler into real training.
- Added `data.event_balance_power` and `--event-balance-power`; `0.5` assigns event mass in proportion to the square root of event size.
- Added `data.event_tile_weight_cap` and `--event-tile-weight-cap` to prevent one- and two-tile events from dominating optimisation.
- Added a controlled U-Net ResNet50 training configuration using the audited `power=0.5`, `cap=5` profile.
- Kept model, loss, augmentation, normalization and dataset filtering unchanged so the first retraining run isolates the sampler correction.

## 0.7.9

- Added controlled training-sampler comparison: configured event balancing, square-root tempered event balancing, and plain shuffled sampling.
- Added exposure Gini, top-share concentration, mean unique tiles per epoch, and per-profile negative-cluster coverage.
- Added a 5x median per-tile weight cap to the audit-only tempered profile.

## 0.7.8

- Added `audit-training-exposure` to simulate the exact configured sampler without retraining.
- Reports per-epoch and per-batch flood composition, tile selection frequency, unseen/over-repeated tiles, and effective samples per epoch.
- Added unsupervised hard-negative proxy clustering from VV/VH/DEM statistics and texture descriptors.
- Reports empty and near-empty validation tile counts so false-positive evaluation can be targeted next.

## 0.7.7

- Added selectable sliding-window overlap blending: `uniform` (historical behaviour) and `cosine` (downweights less reliable window borders).
- Added `--window-blend` to deployment manifest export and prediction override.
- Corrected window diagnostics so a large raw probability jump is not treated as proof of a stitching bug; diagnostics now report each seam's percentile within the full-scene gradient distribution and its excess over the scene median.
- Preserved uniform blending as the compatibility default so existing validation/deployment results remain reproducible.

## 0.7.6

- Added `--write-window-diagnostics` to deployment inference.
- Writes the exact sliding-window grid, overlap-count map, per-window VV/VH/DEM/probability statistics, padding information, and probability seam scores without rerunning inference.
- Stores diagnostic paths and summary values in candidate metadata so rectangular artefacts can be separated from model-training failures.

## 0.7.5

- Added `audit-deployment-errors` for evidence-based false-positive/false-negative forensic analysis across labelled deployment runs.
- Added worst-candidate ranking CSV, detailed JSON, HTML report, per-candidate probability/SAR/DEM error statistics, and forensic montages.
- Diagnostics distinguish overprediction, missed-flood behaviour, near-threshold errors, and confident errors without pretending to infer land-cover classes unavailable in the inputs.

## 0.7.4

- Added `audit-training-data` to test whether train, validation, and test tiles are genuinely isolated at exact-tile, source-scene, mapped-area, and EMSR-event levels.
- Added per-split and per-event/area/scene flood-fraction summaries, empty-tile counts, and machine-readable JSON/CSV outputs.
- Added optional `--fail-on-leakage` for reproducible training checks.

## 0.6.5

- Added optional deployment mask inputs (`--mask-path`, `--mask-dir`) for labelled-scene evaluation.
- Added deployment evaluation metrics and confusion-matrix/error-overlay outputs when a mask is supplied.
- Added concise/standard/verbose deployment console modes and a compact final result table.
- Added `deploy_scene_colab(...)` for true notebook-native inline display.
- Changed deployment flood-mask GeoTIFFs so 0 remains a valid non-flood class rather than nodata.


## 0.6.3

- Added `discover-scene` to inventory event folders that contain multiple SAR files and group deployable VV/VH candidates.
- Expanded `predict-scene` to accept either a direct two-band SAR GeoTIFF or a scene folder with SAR acquisition selection.
- Added deployment visual outputs: input previews, false-colour composites, flood-probability heatmaps, binary-mask overlays, ensemble uncertainty maps, HTML reports, inline notebook display, and optional per-modality occlusion explanations.
- Deployment outputs now include prediction metadata and scene-level summaries for multi-acquisition folders.

## 0.6.0

- Added a cleaned continual-learning command, `floodmap continual-train`, for chronological rehearsal experiments.
- Implemented replay strategies: random, least-confidence, margin, and entropy sampling.
- Added year-range task construction from MMFlood activation metadata, reservoir replay-buffer management, mixed-validation checkpoint selection, and task-by-task evaluation matrices.
- Continual-learning runs reuse the same preprocessing, mask ignore handling, train-fitted normalization, model builders, losses, threshold sweeps, and global pixel metrics as the main segmentation pipeline.
- Added tests for CL task construction, replay-buffer bounds, EMSR event parsing, and CLI options.

## 0.5.9

- Fixed overlay generation for full-raster and sliding-window audits when model outputs are stride-padded but source GeoTIFFs keep their original shape.
- Overlay arrays are cropped to their shared valid extent for plotting only; metrics and CSV outputs are unchanged.

## 0.5.8

- Fixed `ensemble-error-audit` overlay selection so `--max-overlays-per-category` is passed to the shared overlay selector correctly.
- Added a compatibility alias in the overlay selector to keep single-model and ensemble audit code paths consistent.

## 0.5.7

- Added audit-based hard-example sampling for second-stage fine-tuning.
- New training CLI options: `--hard-example-sampling`, `--hard-example-csv`, `--hard-example-categories`, `--hard-example-fg-bins`, `--hard-example-max-f1`, `--hard-example-weight`, and `--hard-example-max-fraction`.
- Hard-example sampling reads train-split `tile_error_metrics.csv` from `floodmap error-audit` and oversamples matched training tiles without deleting any data.
- Sampler validation now treats hard-example sampling as mutually exclusive with event-balanced, stratified, foreground-balanced, and entropy-weighted sampling.

## 0.5.6

- Made training sampler selection explicit: only one of weighted, foreground-balanced, foreground-ratio stratified, or event-balanced sampling can be active.
- Positive sampler CLI flags now disable the other sampler modes, so `--stratified-sampling` no longer inherits an event-balanced sampler from a base config.
- Saved run configs are now plain YAML and can be loaded with `yaml.safe_load`.
- Polynomial LR scheduling no longer increases learning rates when fine-tuning below the configured end learning rate.
- Cleaned comments and help text in the touched training, config, transform, and scheduler code.
- Removed bytecode and pytest cache artifacts from the package.

## 0.5.5

- Preserved sample index fields in padded evaluation batches so error-audit datasets return `(x, y, index)` correctly.

## 0.5.4

- Added `--pretrained` and `--no-pretrained` overrides to `evaluate`, `ensemble-evaluate`, `error-audit`, and `ensemble-error-audit`.
- Checkpoint evaluation can build the architecture without downloading ImageNet or Hugging Face weights before loading a full checkpoint.

## 0.5.3

- Added explicit run-extension support for resumed training.
- Added `--extend-epochs` for workflows that continue a run for additional epochs.
- Added `--reset-early-stopping` and `--keep-early-stopping-state` to control patience state after resume.
- Clarified that `--epochs` remains the total target epoch count when resuming.

## 0.5.2

- Replaced the size-specific preprocessing template with `configs/preprocess_mmflood.yaml`.
- Made preprocessing tile size a runtime CLI choice with `--tile-size 128`, `256`, or `512`.
- Added derived overlap handling when `--tile-max-overlap` is omitted.
- Added architecture/backbone CLI aliases: `--architecture`/`--decoder` and `--backbone`/`--encoder`.
- Exposed model options from the training CLI: `--pretrained`, `--freeze-encoder`, `--output-stride`, `--activation`, `--norm-layer`, and `--dropout2d`.
- Added `configs/train_segmentation_vv_vh_dem.yaml` as a generic training template.

## 0.5.1

- Added `DistributedLogger.warning(...)` alias so non-finite batch skip logging does not crash training.
- Kept `warn(...)` as a wrapper around `warning(...)`.
