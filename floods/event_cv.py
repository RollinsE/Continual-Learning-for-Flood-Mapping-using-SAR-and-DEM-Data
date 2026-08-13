from __future__ import annotations

import csv
import json
import logging
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from floods.config import TrainConfig
from floods.config.training import Losses, Optimizers, Schedulers
from floods.modalities import canonicalize_modalities
from floods.normalization import fit_normalization_stats
from floods.pretrained import (
    CandidateSpec, ResolvedModelSpec, apply_resolved_model_to_config,
    parse_candidate_spec, resolve_candidate,
)
from floods.utils.common import get_logger

LOG = get_logger(__name__)


_SUCCESSFUL_CV_STOP_REASONS = frozenset({"completed", "early_stopping"})
_CV_RESULT_SCHEMA_VERSION = 2


def _cv_result_is_complete(payload: dict) -> bool:
    """Return whether a per-fold result marker represents a successful run.

    Version 0.15.1 treated any ``cv_result.json`` as complete, including files
    written after an interrupted training call.  Explicit status is used for
    new markers, while legacy markers remain readable when their stop reason
    is unambiguously successful.
    """
    result = payload.get("result")
    if not isinstance(result, dict):
        return False
    stop_reason = str(result.get("stop_reason") or "").strip().lower()
    status = payload.get("status")
    if status is None:
        return stop_reason in _SUCCESSFUL_CV_STOP_REASONS
    return (
        str(status).strip().lower() == "completed"
        and stop_reason in _SUCCESSFUL_CV_STOP_REASONS
    )


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON through a sibling temporary file before replacing the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _archive_incomplete_result_marker(result_json: Path, payload: dict | None = None) -> Path:
    """Move a stale/incomplete completion marker out of the skip path."""
    archived = result_json.with_name("cv_result.interrupted.json")
    if archived.exists():
        archived.unlink()
    if payload is None:
        result_json.replace(archived)
    else:
        _write_json_atomic(archived, payload)
        result_json.unlink(missing_ok=True)
    return archived


def _remove_run_rows(rows: list[dict], run_id: str) -> list[dict]:
    return [row for row in rows if str(row.get("run_id")) != str(run_id)]


def _write_result_matrices(output_dir: Path, result_rows: list[dict], event_rows: list[dict]) -> None:
    """Persist deduplicated result matrices and rebuild the summary."""
    results_path = output_dir / "cv_results.csv"
    if result_rows:
        pd.DataFrame(result_rows).drop_duplicates(subset=["run_id"], keep="last").to_csv(
            results_path, index=False
        )
    elif results_path.exists():
        results_path.unlink()

    events_path = output_dir / "cv_event_results.csv"
    if event_rows:
        pd.DataFrame(event_rows).drop_duplicates(
            subset=["run_id", "event_id"], keep="last"
        ).to_csv(events_path, index=False)
    elif events_path.exists():
        events_path.unlink()

    summary_path = output_dir / "cv_summary.csv"
    if results_path.exists():
        _aggregate_results(output_dir)
    elif summary_path.exists():
        summary_path.unlink()


@dataclass(frozen=True)
class ArchitectureSpec:
    decoder: str
    encoder: str
    label: str


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("_") or "unnamed"


def parse_architecture_spec(value: str) -> ArchitectureSpec:
    parts = [part.strip() for part in str(value).split(":") if part.strip()]
    if len(parts) not in {2, 3}:
        raise ValueError(
            "Architecture specifications must be decoder:encoder[:label], "
            "for example unet:resnet34 or segformer:pvt_v2_b0:segformer_pvtv2_b0"
        )
    decoder, encoder = parts[:2]
    label = parts[2] if len(parts) == 3 else f"{decoder}_{encoder}"
    return ArchitectureSpec(decoder=decoder, encoder=encoder, label=_safe_name(label))


def validate_architecture_specs(specs: Sequence[ArchitectureSpec]) -> None:
    """Fail before normalization when a decoder or timm encoder is unavailable."""
    from floods.models.decoders import available_decoders
    try:
        from floods.models.encoders import available_encoders
    except ModuleNotFoundError as exc:
        if exc.name != "timm":
            raise
        # ``timm`` is a required runtime dependency, but keeping this validation
        # lightweight allows plan/test tooling to inspect known candidates before
        # the full environment is installed.
        available_encoders = {
            "resnet18": None, "resnet34": None, "resnet50": None,
            "pvt_v2_b0": None, "convnext_tiny": None, "efficientnet_b3": None,
        }

    errors: list[str] = []
    for spec in specs:
        if spec.decoder not in available_decoders:
            errors.append(f"decoder {spec.decoder!r} is not implemented")
        if spec.encoder not in available_encoders:
            errors.append(
                f"encoder {spec.encoder!r} is not available in the installed timm version"
            )
    if errors:
        available_hint = ", ".join(
            name for name in ("resnet34", "pvt_v2_b0", "convnext_tiny")
            if name in available_encoders
        )
        suffix = f" Known compatible examples: {available_hint}." if available_hint else ""
        raise ValueError("Invalid event-CV architecture specification: " + "; ".join(errors) + suffix)


def parse_modality_set(value: str) -> list[str]:
    tokens = [part for part in re.split(r"[+,\s]+", str(value).strip()) if part]
    modalities = canonicalize_modalities(tokens)
    if len(modalities) < 2:
        raise ValueError(f"A modality set must contain at least two channels: {value!r}")
    return modalities


def _event_id(path: Path) -> str:
    match = re.search(r"(EMSR\d+)", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not extract EMSR event ID from {path}")
    return match.group(1).upper()


def discover_event_counts(processed_data_dir: Path, split: str = "train") -> dict[str, int]:
    mask_dir = Path(processed_data_dir) / split / "mask"
    files = sorted(mask_dir.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No masks found under {mask_dir}")
    counts: dict[str, int] = {}
    for path in files:
        event = _event_id(path)
        counts[event] = counts.get(event, 0) + 1
    if len(counts) < 2:
        raise ValueError("Event cross-validation requires at least two events")
    return counts


def build_balanced_folds(event_counts: dict[str, int], n_splits: int, seed: int = 42) -> list[list[str]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > len(event_counts):
        raise ValueError("n_splits cannot exceed the number of events")
    # Deterministic greedy bin packing. Seed is used only as a stable tie breaker.
    import random

    rng = random.Random(int(seed))
    items = list(event_counts.items())
    rng.shuffle(items)
    items.sort(key=lambda item: item[1], reverse=True)
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    totals = [0 for _ in range(n_splits)]
    for event, count in items:
        target = min(range(n_splits), key=lambda idx: (totals[idx], len(folds[idx]), idx))
        folds[target].append(event)
        totals[target] += int(count)
    for fold in folds:
        fold.sort()
    return folds


def write_fold_plan(output_dir: Path, event_counts: dict[str, int], folds: Sequence[Sequence[str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "events": event_counts,
        "folds": [list(fold) for fold in folds],
        "fold_tile_counts": [sum(event_counts[event] for event in fold) for fold in folds],
    }
    (output_dir / "folds.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output_dir / "fold_assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "fold", "tile_count"])
        writer.writeheader()
        for fold_idx, fold in enumerate(folds):
            for event in fold:
                writer.writerow({"event_id": event, "fold": fold_idx, "tile_count": event_counts[event]})


def _read_best_event_rows(run_dir: Path, best_epoch: int | None) -> list[dict]:
    path = run_dir / "event_validation_metrics.csv"
    if not path.exists() or best_epoch is None:
        return []
    frame = pd.read_csv(path)
    frame = frame[frame["epoch"] == int(best_epoch)]
    return frame.to_dict(orient="records")


def _read_best_history_row(run_dir: Path, best_epoch: int | None) -> dict:
    path = run_dir / "event_validation_history.csv"
    if not path.exists() or best_epoch is None:
        return {}
    frame = pd.read_csv(path)
    selected = frame[frame["epoch"] == int(best_epoch)]
    if selected.empty:
        return {}
    return selected.iloc[-1].to_dict()


def _disable_checkpoint_repair_interventions(config: TrainConfig) -> None:
    data = config.data
    for name in (
        "weighted_sampling", "foreground_balanced_sampling", "stratified_sampling",
        "event_balanced_sampling", "hard_example_sampling", "hard_positive_region_sampling",
        "hard_negative_region_sampling", "sparse_crop_supervision", "modality_dropout",
    ):
        setattr(data, name, False)
    config.trainer.group_dro = False


def _normalization_path(output_dir: Path, modalities: Sequence[str], fold_idx: int) -> Path:
    modality_label = "_".join(modalities)
    return output_dir / "normalization" / modality_label / f"fold_{fold_idx:02d}.json"


def _load_existing_rows(path: Path) -> list[dict]:
    """Load prior matrix rows so staged invocations never erase completed folds."""
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    return frame.to_dict(orient="records")


def _close_run_log_handler(run_dir: Path) -> None:
    """Detach the per-run file handler after an in-process CV training run.

    The event-CV command calls the trainer in-process so interactive and captured environments receive live output without subprocess buffering. Each training run attaches its own
    ``output.log`` handler; removing it here prevents later runs from writing into
    every earlier run log while retaining the command-level matrix log.
    """
    target = (Path(run_dir) / "output.log").resolve()
    root = logging.getLogger()
    for handler in list(root.handlers):
        base = getattr(handler, "baseFilename", None)
        if base is None or Path(base).resolve() != target:
            continue
        root.removeHandler(handler)
        handler.flush()
        handler.close()


def _aggregate_results(output_dir: Path) -> None:
    results_path = output_dir / "cv_results.csv"
    if not results_path.exists():
        return
    frame = pd.read_csv(results_path)
    if frame.empty:
        return
    group_cols = [
        "candidate", "weights_source", "provider", "decoder", "encoder",
        "modalities", "normalization_mode",
    ]
    # Older result matrices remain readable after an in-place package upgrade.
    if "candidate" not in frame.columns and "architecture" in frame.columns:
        frame["candidate"] = frame["architecture"]
    for column, default in (("weights_source", "legacy"), ("provider", "legacy"), ("normalization_mode", "legacy")):
        if column not in frame.columns:
            frame[column] = default
    summary = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            folds_completed=("fold", "nunique"),
            mean_event_macro_f1=("event_macro_f1", "mean"),
            std_event_macro_f1=("event_macro_f1", "std"),
            worst_fold_event_macro_f1=("event_macro_f1", "min"),
            mean_worst_event_f1=("worst_event_f1", "mean"),
            absolute_worst_event_f1=("worst_event_f1", "min"),
            median_best_epoch=("best_epoch", "median"),
            mean_threshold=("threshold", "mean"),
        )
        .reset_index()
        .sort_values(["mean_event_macro_f1", "absolute_worst_event_f1"], ascending=[False, False])
    )
    summary.to_csv(output_dir / "cv_summary.csv", index=False)


def run_event_cross_validation(
    base_config: TrainConfig,
    *,
    processed_data_dir: Path,
    output_dir: Path,
    candidates: Sequence[str] | None = None,
    architectures: Sequence[str] | None = None,
    modality_sets: Sequence[str],
    folds: int = 5,
    fold_indices: Iterable[int] | None = None,
    source_split: str = "train",
    seed: int = 42,
    epochs: int | None = None,
    patience: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    encoder_lr: float | None = None,
    decoder_lr: float | None = None,
    weight_decay: float | None = None,
    optimizer: str | None = None,
    scheduler: str | None = None,
    loss: str | None = None,
    loss_alpha: float | None = None,
    loss_beta: float | None = None,
    bce_weight: float | None = None,
    tversky_weight: float | None = None,
    augmentation_profile: str | None = None,
    pretrained: bool | None = None,
    amp: bool | None = None,
    cpu: bool | None = None,
    thresholds: Sequence[float] | None = None,
    q_min: float = 1.0,
    q_max: float = 99.0,
    max_pixels_per_file: int = 4096,
    plan_only: bool = False,
    skip_completed: bool = True,
) -> dict:
    """Run event-separated CV with the same provider registry used by ``train``.

    New callers use ``--candidates``.  ``--architectures`` remains a compatibility
    bridge for matrices produced before pretrained sources became first-class.
    """
    processed_data_dir = Path(processed_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    event_counts = discover_event_counts(processed_data_dir, split=source_split)
    fold_plan = build_balanced_folds(event_counts, int(folds), seed=int(seed))
    invariant_manifest = {
        "processed_data_dir": str(processed_data_dir.resolve()),
        "source_split": source_split,
        "folds": len(fold_plan),
        "seed": int(seed),
        "event_counts": event_counts,
        "fold_plan": [list(fold) for fold in fold_plan],
        "normalization_q_min": float(q_min),
        "normalization_q_max": float(q_max),
        "normalization_max_pixels_per_file": int(max_pixels_per_file),
    }
    manifest_path = output_dir / "cv_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_invariants = previous.get("invariants", {})
        if previous_invariants and previous_invariants != invariant_manifest:
            changed = sorted(
                key for key in set(previous_invariants) | set(invariant_manifest)
                if previous_invariants.get(key) != invariant_manifest.get(key)
            )
            raise ValueError(
                "Existing event-CV output directory was created with a different fold or "
                f"normalization plan (changed: {changed}). Use a new --output-dir."
            )
    write_fold_plan(output_dir, event_counts, fold_plan)

    raw_candidates: list[CandidateSpec] = []
    if candidates:
        raw_candidates.extend(parse_candidate_spec(value) for value in candidates)
    if architectures:
        legacy_source = "imagenet" if pretrained is not False else "random"
        for value in architectures:
            architecture = parse_architecture_spec(value)
            raw_candidates.append(CandidateSpec(
                weights_source=legacy_source,
                decoder=architecture.decoder,
                encoder=architecture.encoder,
                label=architecture.label,
            ))
    if not raw_candidates:
        raw_candidates = [
            parse_candidate_spec("imagenet:unet:resnet34"),
            parse_candidate_spec("imagenet:deeplabv3p:resnet34"),
            parse_candidate_spec("imagenet:segformer:pvt_v2_b0"),
        ]

    modality_specs = [parse_modality_set(value) for value in modality_sets]
    resolved_matrix: list[tuple[ResolvedModelSpec, list[str]]] = []
    for modality_values in modality_specs:
        for candidate in raw_candidates:
            resolved = resolve_candidate(candidate, modality_values, legacy_pretrained=pretrained)
            resolved_matrix.append((resolved, modality_values))

    # Validate ordinary timm entries before spending time fitting normalization.
    ordinary = [
        ArchitectureSpec(value.decoder, value.encoder, value.label)
        for value, _ in resolved_matrix if value.adapter == "timm"
    ]
    if ordinary and not plan_only:
        validate_architecture_specs(ordinary)

    selected_folds = list(range(len(fold_plan))) if fold_indices is None else sorted(set(int(v) for v in fold_indices))
    invalid = [idx for idx in selected_folds if idx < 0 or idx >= len(fold_plan)]
    if invalid:
        raise ValueError(f"Fold indices outside [0, {len(fold_plan) - 1}]: {invalid}")

    manifest = {
        "schema_version": 3,
        "invariants": invariant_manifest,
        "last_invocation": {
            "selected_folds": selected_folds,
            "candidates": [value.to_dict() for value, _ in resolved_matrix],
            "modality_sets": modality_specs,
            "plan_only": bool(plan_only),
        },
        "selection_metric": "best_event_macro_f1",
        "secondary_metric": "worst_event_f1",
        "normalization": (
            "provider-fixed statistics for registered provider modes; otherwise fitted "
            "separately on each fold's training events"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    planned_runs = len(selected_folds) * len(resolved_matrix)
    LOG.info(
        "Event CV plan: events=%d | folds=%d | selected_folds=%s | candidates=%d | runs=%d",
        len(event_counts), len(fold_plan), selected_folds, len(resolved_matrix), planned_runs,
    )
    for resolved, mods in resolved_matrix:
        LOG.info(
            "Candidate: label=%s | weights_source=%s | provider=%s | architecture=%s:%s | "
            "modalities=%s | normalization=%s",
            resolved.label, resolved.weights_source, resolved.provider, resolved.decoder,
            resolved.encoder, mods, resolved.normalization_mode,
        )
    for idx, held_out in enumerate(fold_plan):
        LOG.info(
            "Fold %d: validation events=%d | tiles=%d | %s",
            idx, len(held_out), sum(event_counts[event] for event in held_out), ", ".join(held_out),
        )
    if plan_only:
        return {"planned_runs": planned_runs, "output_dir": str(output_dir), "plan_only": True}

    from floods.training import train

    result_rows: list[dict] = _load_existing_rows(output_dir / "cv_results.csv")
    event_rows: list[dict] = _load_existing_rows(output_dir / "cv_event_results.csv")
    for resolved, modalities in resolved_matrix:
        modality_label = "_".join(modalities)
        for fold_idx in selected_folds:
            held_out = list(fold_plan[fold_idx])
            provider_fixed = resolved.normalization_mode in {"terramind_v1", "ssl4eo_s1"}
            stats_path = None
            if not provider_fixed:
                stats_path = _normalization_path(output_dir, modalities, fold_idx)
                if not stats_path.exists():
                    LOG.info(
                        "Fitting leakage-free normalization: fold=%d | candidate=%s | modalities=%s | excluded events=%s",
                        fold_idx, resolved.label, modalities, held_out,
                    )
                    fit_normalization_stats(
                        processed_data_dir=processed_data_dir,
                        output_file=stats_path,
                        split=source_split,
                        input_modalities=modalities,
                        q_min=float(q_min), q_max=float(q_max),
                        max_pixels_per_file=int(max_pixels_per_file),
                        seed=int(seed) + fold_idx,
                        ignore_mask_255=True,
                        exclude_events=held_out,
                    )

            run_id = _safe_name(f"eventcv_{resolved.label}_{modality_label}_fold{fold_idx:02d}")
            run_root = output_dir / "runs"
            run_dir = run_root / run_id
            result_json = run_dir / "cv_result.json"
            if not result_json.exists() and any(
                str(row.get("run_id")) == run_id for row in result_rows
            ):
                LOG.warning(
                    "Removing stale matrix rows for %s because no successful per-run completion marker exists.",
                    run_id,
                )
                result_rows = _remove_run_rows(result_rows, run_id)
                event_rows = _remove_run_rows(event_rows, run_id)
                _write_result_matrices(output_dir, result_rows, event_rows)

            if result_json.exists():
                try:
                    payload = json.loads(result_json.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    LOG.warning(
                        "Ignoring invalid CV result marker for %s (%s); the run will resume when a checkpoint exists.",
                        run_id, exc,
                    )
                    payload = None

                if payload is not None and skip_completed and _cv_result_is_complete(payload):
                    LOG.info("Skipping completed CV run: %s", run_id)
                    result_rows.append(payload["result"])
                    event_rows.extend(payload.get("events", []))
                    continue

                if payload is not None and not _cv_result_is_complete(payload):
                    stop_reason = payload.get("result", {}).get("stop_reason", "unknown")
                    archived = _archive_incomplete_result_marker(result_json, payload)
                    LOG.warning(
                        "Found stale incomplete CV result marker for %s (stop_reason=%s). "
                        "Archived it to %s and will resume from the last completed checkpoint.",
                        run_id, stop_reason, archived,
                    )
                    result_rows = _remove_run_rows(result_rows, run_id)
                    event_rows = _remove_run_rows(event_rows, run_id)
                    _write_result_matrices(output_dir, result_rows, event_rows)
                elif payload is None:
                    archived = _archive_incomplete_result_marker(result_json)
                    LOG.warning("Archived invalid CV result marker to %s", archived)
                    result_rows = _remove_run_rows(result_rows, run_id)
                    event_rows = _remove_run_rows(event_rows, run_id)
                    _write_result_matrices(output_dir, result_rows, event_rows)
                elif not skip_completed:
                    LOG.warning("Rerunning completed CV run from fresh initialisation: %s", run_id)
                    shutil.rmtree(run_dir)

            config = deepcopy(base_config)
            config.name = run_id
            config.output_folder = str(run_root)
            config.seed = int(seed) + fold_idx
            config.init_checkpoint = None
            resume_path = run_dir / "models" / "last.ckpt"
            config.resume = resume_path.exists()
            config.resume_from = None
            if config.resume:
                LOG.info("Resuming interrupted CV run from %s", resume_path)
            config.data.path = str(processed_data_dir)
            config.data.train_source_split = source_split
            config.data.val_source_split = source_split
            config.data.train_include_events = []
            config.data.train_exclude_events = held_out
            config.data.val_include_events = held_out
            config.data.val_exclude_events = []
            apply_resolved_model_to_config(config, resolved, evaluation=False, force_normalization=True)
            config.model.freeze = False
            config.model.multibranch = False
            config.data.normalization_stats_path = None if provider_fixed else str(stats_path)
            if epochs is not None:
                config.trainer.max_epochs = int(epochs)
            if patience is not None:
                config.trainer.patience = int(patience)
            if batch_size is not None:
                config.trainer.batch_size = int(batch_size)
            if num_workers is not None:
                config.trainer.num_workers = int(num_workers)
            if encoder_lr is not None:
                config.optimizer.encoder_lr = float(encoder_lr)
            if decoder_lr is not None:
                config.optimizer.decoder_lr = float(decoder_lr)
            if weight_decay is not None:
                config.optimizer.weight_decay = float(weight_decay)
            if optimizer is not None:
                config.optimizer.target = Optimizers[str(optimizer).lower()]
            if scheduler is not None:
                config.scheduler.target = Schedulers[str(scheduler).lower()]
            if loss is not None:
                config.loss.target = Losses[str(loss).lower()]
            if loss_alpha is not None:
                config.loss.alpha = float(loss_alpha)
            if loss_beta is not None:
                config.loss.beta = float(loss_beta)
            if bce_weight is not None:
                config.loss.bce_weight = float(bce_weight)
            if tversky_weight is not None:
                config.loss.tversky_weight = float(tversky_weight)
            if augmentation_profile is not None and resolved.normalization_mode not in {"terramind_v1", "ssl4eo_s1"}:
                config.data.augmentation_profile = str(augmentation_profile)
            if amp is not None:
                config.trainer.amp = bool(amp)
            if cpu is not None:
                config.trainer.cpu = bool(cpu)
            config.trainer.threshold_sweep = True
            config.trainer.monitor_threshold_sweep = True
            config.trainer.threshold_metric = "f1"
            config.trainer.event_macro_validation = True
            if thresholds:
                config.trainer.thresholds = [float(v) for v in thresholds]
            config.trainer.save_last = True
            config.trainer.save_epoch_checkpoints = False
            config.trainer.progress_label = f"CV {resolved.label} {modality_label} fold {fold_idx}"
            config.visualize = False
            _disable_checkpoint_repair_interventions(config)
            config.data.refresh_cache_hash()

            LOG.info(
                "Starting CV run %s | fold=%d | held_out=%s | weights_source=%s | architecture=%s:%s | modalities=%s",
                run_id, fold_idx, held_out, resolved.weights_source, resolved.decoder, resolved.encoder, modalities,
            )
            try:
                train_result = train(config)
            finally:
                _close_run_log_handler(run_dir)

            stop_reason = str(train_result.get("stop_reason") or "").strip().lower()
            if stop_reason == "interrupted":
                result_rows = _remove_run_rows(result_rows, run_id)
                event_rows = _remove_run_rows(event_rows, run_id)
                _write_result_matrices(output_dir, result_rows, event_rows)
                LOG.warning(
                    "CV run %s was interrupted. No completion marker was written; "
                    "rerun the same command to resume from models/last.ckpt.",
                    run_id,
                )
                raise KeyboardInterrupt
            if stop_reason not in _SUCCESSFUL_CV_STOP_REASONS:
                raise RuntimeError(
                    f"CV run {run_id} stopped without successful completion "
                    f"(stop_reason={stop_reason or 'unknown'}). No completion marker was written."
                )

            best_epoch = train_result.get("best_epoch")
            history = _read_best_history_row(run_dir, best_epoch)
            per_event = _read_best_event_rows(run_dir, best_epoch)
            row = {
                "run_id": run_id,
                "fold": fold_idx,
                "held_out_events": ",".join(held_out),
                "candidate": resolved.label,
                "architecture": resolved.label,
                "weights_source": resolved.weights_source,
                "provider": resolved.provider,
                "adapter": resolved.adapter,
                "decoder": resolved.decoder,
                "encoder": resolved.encoder,
                "modalities": "+".join(modalities),
                "normalization_mode": resolved.normalization_mode,
                "best_epoch": best_epoch,
                "checkpoint": train_result.get("best_checkpoint"),
                "event_macro_f1": history.get("macro_f1", train_result.get("best_score")),
                "event_macro_iou": history.get("macro_iou"),
                "worst_event_f1": history.get("worst_event_f1"),
                "threshold": history.get("threshold"),
                "stop_reason": train_result.get("stop_reason"),
            }
            enriched_events = [
                {
                    "run_id": run_id, "fold": fold_idx,
                    "candidate": resolved.label,
                    "weights_source": resolved.weights_source,
                    "modalities": "+".join(modalities), **event_row,
                }
                for event_row in per_event
            ]
            result_rows.append(row)
            event_rows.extend(enriched_events)
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(
                result_json,
                {
                    "schema_version": _CV_RESULT_SCHEMA_VERSION,
                    "status": "completed",
                    "result": row,
                    "events": enriched_events,
                },
            )

            _write_result_matrices(output_dir, result_rows, event_rows)

    final_results = pd.DataFrame(result_rows).drop_duplicates(subset=["run_id"], keep="last")
    _write_result_matrices(output_dir, result_rows, event_rows)
    return {
        "planned_runs": planned_runs,
        "completed_runs": int(len(final_results)),
        "output_dir": str(output_dir),
        "summary_csv": str(output_dir / "cv_summary.csv"),
    }

