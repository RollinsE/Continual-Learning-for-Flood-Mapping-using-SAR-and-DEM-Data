from __future__ import annotations

import logging

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


BIN_NAMES = ["empty", "tiny", "small", "medium", "large", "very_large"]
BIN_EDGES = np.asarray([0.0, 0.001, 0.005, 0.02, 0.10, 0.25, 1.000001], dtype=np.float64)



LOG = logging.getLogger(__name__)

def _bin_index(ratio: float) -> int:
    if ratio <= 0.0:
        return 0
    return int(np.clip(np.searchsorted(BIN_EDGES[1:], ratio, side="right"), 1, len(BIN_NAMES) - 1))


def _safe_stats(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}


def _event_id(path) -> str:
    import re
    match = re.search(r"(EMSR\d+)", str(path), flags=re.IGNORECASE)
    return match.group(1).upper() if match else "UNKNOWN"


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.all(values == 0):
        return 0.0
    values = np.sort(np.clip(values, 0.0, None))
    n = values.size
    return float((2.0 * np.dot(np.arange(1, n + 1), values) / (n * values.sum())) - (n + 1.0) / n)


def _top_share(values: np.ndarray, fraction: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = values.sum()
    if values.size == 0 or total <= 0:
        return 0.0
    n = max(1, int(np.ceil(values.size * float(fraction))))
    return float(np.sort(values)[-n:].sum() / total)


def _tempered_event_weights(label_files, ratios: np.ndarray, fg_bin_edges: Sequence[float],
                            fg_bin_sample_weights: Sequence[float], event_alpha: float,
                            max_weight_ratio: float | None) -> np.ndarray:
    """Return event-aware tile weights with event mass proportional to n_event ** alpha.

    alpha=0 reproduces equal event mass; alpha=1 approaches equal tile mass before
    foreground-bin correction. An optional cap prevents tiny events from producing
    extreme per-tile probabilities.
    """
    edges = [float(v) for v in fg_bin_edges]
    if len(edges) != 4:
        raise ValueError("fg_bin_edges must contain exactly four values")
    target = np.asarray(fg_bin_sample_weights, dtype=np.float64)
    if target.size != 5 or np.any(target < 0) or target.sum() <= 0:
        raise ValueError("fg_bin_sample_weights must contain five non-negative values")
    target /= target.sum()
    masks = [
        ratios <= edges[0],
        (ratios > edges[0]) & (ratios < edges[1]),
        (ratios >= edges[1]) & (ratios < edges[2]),
        (ratios >= edges[2]) & (ratios < edges[3]),
        ratios >= edges[3],
    ]
    events = np.asarray([_event_id(p) for p in label_files])
    unique_events = sorted(set(events.tolist()))
    sizes = np.asarray([np.count_nonzero(events == e) for e in unique_events], dtype=np.float64)
    masses = np.power(sizes, float(event_alpha))
    masses /= masses.sum()
    weights = np.zeros(len(label_files), dtype=np.float64)
    for event, event_mass in zip(unique_events, masses):
        event_mask = events == event
        available = np.asarray([np.any(event_mask & m) for m in masks], dtype=bool)
        local = target.copy()
        local[~available] = 0.0
        if local.sum() <= 0:
            weights[event_mask] = event_mass / np.count_nonzero(event_mask)
            continue
        local /= local.sum()
        for bi, mask in enumerate(masks):
            selected = event_mask & mask
            count = int(np.count_nonzero(selected))
            if count:
                weights[selected] = event_mass * local[bi] / count
    positive = weights[weights > 0]
    if max_weight_ratio is not None and positive.size:
        cap = float(np.median(positive)) * float(max_weight_ratio)
        weights = np.minimum(weights, cap)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise RuntimeError("Invalid tempered event sampler weights")
    return weights / weights.sum()


def _descriptor_row(dataset, index: int, fg_ratio: float) -> dict:
    from floods.utils.gis import imread
    sar = imread(dataset.image_files[index], channels_first=False).astype(np.float32)
    if sar.ndim == 2:
        sar = sar[..., None]
    row = {"index": index, "tile_id": Path(dataset.image_files[index]).stem,
           "foreground_ratio": float(fg_ratio), "foreground_bin": BIN_NAMES[_bin_index(float(fg_ratio))]}
    for c, name in enumerate(["vv", "vh"][: sar.shape[-1]]):
        vals = sar[..., c]
        finite = vals[np.isfinite(vals)]
        row[f"{name}_mean"] = float(finite.mean()) if finite.size else None
        row[f"{name}_std"] = float(finite.std()) if finite.size else None
        row[f"{name}_p10"] = float(np.percentile(finite, 10)) if finite.size else None
        row[f"{name}_p90"] = float(np.percentile(finite, 90)) if finite.size else None
        if finite.size and vals.shape[0] > 1 and vals.shape[1] > 1:
            gy, gx = np.gradient(np.nan_to_num(vals, nan=float(finite.mean())))
            row[f"{name}_texture"] = float(np.mean(np.sqrt(gx * gx + gy * gy)))
        else:
            row[f"{name}_texture"] = None
    if getattr(dataset, "_include_dem", False):
        dem = imread(dataset.dem_files[index], channels_first=False).astype(np.float32).squeeze()
        finite = dem[np.isfinite(dem)]
        row["dem_mean"] = float(finite.mean()) if finite.size else None
        row["dem_std"] = float(finite.std()) if finite.size else None
        row["dem_relief"] = float(finite.max() - finite.min()) if finite.size else None
    return row


def _simple_kmeans(matrix: np.ndarray, k: int, seed: int, iterations: int = 50) -> np.ndarray:
    if len(matrix) == 0:
        return np.empty(0, dtype=int)
    k = max(1, min(int(k), len(matrix)))
    rng = np.random.default_rng(seed)
    centres = matrix[rng.choice(len(matrix), size=k, replace=False)].copy()
    labels = np.zeros(len(matrix), dtype=int)
    for _ in range(iterations):
        distances = ((matrix[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for j in range(k):
            members = matrix[labels == j]
            if len(members):
                centres[j] = members.mean(axis=0)
    return labels


def _simulate_profile(profile: str, train_set, ratios: np.ndarray, config, epochs: int, seed: int):
    from floods.training import prepare_training_sampler
    samples_per_epoch = int(round(len(train_set) * float(config.data.weighted_samples_multiplier)))
    if profile == "configured":
        _, sampler = prepare_training_sampler(config, train_set)
        probability = None
    elif profile == "tempered":
        probability = _tempered_event_weights(train_set.label_files, ratios, config.data.fg_bin_edges,
                                               config.data.fg_bin_sample_weights, event_alpha=0.5,
                                               max_weight_ratio=5.0)
        sampler = None
    elif profile == "shuffle":
        probability = None
        sampler = None
        samples_per_epoch = len(train_set)
    else:
        raise ValueError(f"Unknown sampler profile: {profile}")

    counts = np.zeros(len(train_set), dtype=np.int64)
    epoch_rows, batch_rows = [], []
    batch_size = int(config.trainer.batch_size)
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    for epoch in range(int(epochs)):
        if profile == "shuffle":
            indices = torch.randperm(len(train_set), generator=generator).tolist()
        elif profile == "tempered":
            indices = rng.choice(len(train_set), size=samples_per_epoch, replace=True, p=probability).tolist()
        elif sampler is None:
            indices = torch.randperm(len(train_set), generator=generator).tolist()
        else:
            torch.manual_seed(seed + epoch)
            indices = list(iter(sampler))
        epoch_counts = np.bincount(indices, minlength=len(train_set))
        counts += epoch_counts
        bins = Counter(BIN_NAMES[_bin_index(float(ratios[i]))] for i in indices)
        epoch_rows.append({"profile": profile, "epoch": epoch + 1, "samples": len(indices),
                           "unique_tiles": int(np.count_nonzero(epoch_counts)),
                           "repeated_draws": int(len(indices) - np.count_nonzero(epoch_counts)),
                           "mean_foreground_ratio": float(np.mean(ratios[indices])) if indices else 0.0,
                           **{f"n_{name}": int(bins.get(name, 0)) for name in BIN_NAMES}})
        for batch_no, start in enumerate(range(0, len(indices), batch_size), start=1):
            idx = indices[start:start + batch_size]
            if len(idx) < batch_size:
                continue
            b = Counter(BIN_NAMES[_bin_index(float(ratios[i]))] for i in idx)
            batch_rows.append({"profile": profile, "epoch": epoch + 1, "batch": batch_no,
                               "mean_foreground_ratio": float(np.mean(ratios[idx])),
                               "max_foreground_ratio": float(np.max(ratios[idx])),
                               "unique_tiles": len(set(idx)),
                               **{f"n_{name}": int(b.get(name, 0)) for name in BIN_NAMES}})
    summary = {
        "profile": profile,
        "samples_per_epoch": int(epoch_rows[0]["samples"]) if epoch_rows else 0,
        "mean_unique_tiles_per_epoch": float(np.mean([r["unique_tiles"] for r in epoch_rows])) if epoch_rows else 0.0,
        "mean_repeated_draws_per_epoch": float(np.mean([r["repeated_draws"] for r in epoch_rows])) if epoch_rows else 0.0,
        "unique_tiles_seen": int(np.count_nonzero(counts)),
        "never_sampled_tiles": int(np.count_nonzero(counts == 0)),
        "max_tile_repetitions": int(counts.max()) if counts.size else 0,
        "exposure_gini": _gini(counts),
        "top_1pct_share": _top_share(counts, 0.01),
        "top_5pct_share": _top_share(counts, 0.05),
        "top_10pct_share": _top_share(counts, 0.10),
        "mean_sampled_foreground_ratio": float(np.average(ratios, weights=counts)) if counts.sum() else 0.0,
    }
    return counts, epoch_rows, batch_rows, summary


def audit_training_exposure(config, output_dir: Path, epochs: int = 10, seed: int = 42,
                            negative_max_ratio: float = 0.001, negative_clusters: int = 8,
                            sampler_profiles: Sequence[str] = ("configured", "tempered", "shuffle")) -> dict:
    from floods.prepare import foreground_ratios_from_labels, prepare_datasets

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    use_rgb = (config.data.in_channels - int(config.data.include_dem)) == 3
    train_set, valid_set = prepare_datasets(config=config, use_rgb=use_rgb)
    ratios = foreground_ratios_from_labels(train_set.label_files, cache_hash=config.data.cache_hash,
                                           cache_dir=config.data.cache_dir, force_recompute=config.data.clear_cache)
    val_ratios = foreground_ratios_from_labels(valid_set.label_files,
                                               cache_hash=f"{config.data.cache_hash}_val_audit",
                                               cache_dir=config.data.cache_dir,
                                               force_recompute=config.data.clear_cache)
    profiles = list(dict.fromkeys(str(p).lower() for p in sampler_profiles))
    all_epoch_rows, all_batch_rows, comparison_rows = [], [], []
    profile_counts = {}
    for profile in profiles:
        counts, epoch_rows, batch_rows, profile_summary = _simulate_profile(profile, train_set, ratios, config, epochs, seed)
        profile_counts[profile] = counts
        all_epoch_rows.extend(epoch_rows); all_batch_rows.extend(batch_rows); comparison_rows.append(profile_summary)

    configured_counts = profile_counts.get("configured", next(iter(profile_counts.values())))
    tile_rows = []
    for i, path in enumerate(train_set.image_files):
        row = {"index": i, "tile_id": Path(path).stem, "event": _event_id(path),
               "foreground_ratio": float(ratios[i]), "foreground_bin": BIN_NAMES[_bin_index(float(ratios[i]))]}
        for profile in profiles:
            row[f"times_sampled_{profile}"] = int(profile_counts[profile][i])
            row[f"sampled_per_epoch_{profile}"] = float(profile_counts[profile][i] / max(int(epochs), 1))
        tile_rows.append(row)

    negative_indices = np.flatnonzero(ratios <= float(negative_max_ratio)).tolist()
    descriptor_rows = [_descriptor_row(train_set, i, float(ratios[i])) for i in negative_indices]
    feature_names = [k for k in (descriptor_rows[0].keys() if descriptor_rows else [])
                     if k not in {"index", "tile_id", "foreground_ratio", "foreground_bin"}]
    if descriptor_rows and feature_names:
        matrix = np.asarray([[r.get(k, np.nan) for k in feature_names] for r in descriptor_rows], dtype=np.float64)
        med = np.nanmedian(matrix, axis=0); inds = np.where(~np.isfinite(matrix)); matrix[inds] = np.take(med, inds[1])
        scale = np.nanstd(matrix, axis=0); scale[scale == 0] = 1.0
        labels = _simple_kmeans((matrix - np.nanmean(matrix, axis=0)) / scale, negative_clusters, seed)
        for row, label in zip(descriptor_rows, labels):
            row["negative_cluster"] = int(label)
            for profile in profiles:
                row[f"sampled_per_epoch_{profile}"] = float(profile_counts[profile][row["index"]] / max(int(epochs), 1))

    cluster_summary = []
    for cluster in sorted({r.get("negative_cluster") for r in descriptor_rows if "negative_cluster" in r}):
        members = [r for r in descriptor_rows if r.get("negative_cluster") == cluster]
        row = {"negative_cluster": int(cluster), "tiles": len(members)}
        for profile in profiles:
            vals = [m[f"sampled_per_epoch_{profile}"] for m in members]
            row[f"mean_sampled_per_epoch_{profile}"] = float(np.mean(vals))
            row[f"never_sampled_tiles_{profile}"] = int(sum(profile_counts[profile][m["index"]] == 0 for m in members))
        row.update({f"{name}_mean": _safe_stats([m.get(name) for m in members if m.get(name) is not None])["mean"]
                    for name in feature_names})
        cluster_summary.append(row)

    def write_csv(name: str, rows: Sequence[dict]):
        if not rows: return
        with (output_dir / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)

    write_csv("training_exposure_sampler_comparison.csv", comparison_rows)
    write_csv("training_exposure_epochs.csv", all_epoch_rows)
    write_csv("training_exposure_batches.csv", all_batch_rows)
    write_csv("training_exposure_tiles.csv", tile_rows)
    write_csv("training_negative_descriptors.csv", descriptor_rows)
    write_csv("training_negative_clusters.csv", cluster_summary)

    summary = {"epochs_simulated": int(epochs), "batch_size": int(config.trainer.batch_size),
               "train_tiles": len(train_set), "validation_tiles": len(valid_set),
               "profiles": comparison_rows,
               "train_foreground_distribution": {name: int(sum(_bin_index(float(r)) == i for r in ratios)) for i, name in enumerate(BIN_NAMES)},
               "validation_empty_tiles": int(np.count_nonzero(val_ratios <= 0.0)),
               "validation_near_empty_tiles": int(np.count_nonzero(val_ratios <= negative_max_ratio)),
               "negative_proxy_threshold": float(negative_max_ratio), "negative_proxy_tiles": len(negative_indices),
               "negative_clusters": cluster_summary,
               "tempered_profile": {"event_alpha": 0.5, "max_weight_ratio": 5.0}}
    (output_dir / "training_exposure_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("Training exposure audit compared sampler profiles over %d epoch(s):", epochs)
    for row in comparison_rows:
        LOG.info(
            "  %s: unique/epoch=%.1f | never_seen=%d | max_repeats=%d | gini=%.3f | top5%%=%.1f%%",
            row["profile"], row["mean_unique_tiles_per_epoch"], row["never_sampled_tiles"],
            row["max_tile_repetitions"], row["exposure_gini"], 100.0 * row["top_5pct_share"],
        )
    LOG.info("Validation empty tiles: %d | near-empty: %d", summary["validation_empty_tiles"], summary["validation_near_empty_tiles"])
    LOG.info("Negative proxy tiles clustered: %d into %d group(s)", len(negative_indices), len(cluster_summary))
    LOG.info("Training exposure audit report: %s", output_dir / "training_exposure_audit.json")
    return summary
