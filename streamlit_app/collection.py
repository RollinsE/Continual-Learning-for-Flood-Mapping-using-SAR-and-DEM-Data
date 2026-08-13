from __future__ import annotations

import base64
import html
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, transform_bounds


TARGET_EQUAL_AREA_CRS = "EPSG:6933"
MAX_MOSAIC_CELLS = 20_000_000


def _projected_resolution(src: rasterio.io.DatasetReader, target_crs: str) -> tuple[float, float]:
    transform, _, _ = calculate_default_transform(
        src.crs,
        target_crs,
        src.width,
        src.height,
        *src.bounds,
    )
    return abs(float(transform.a)), abs(float(transform.e))


def _target_grid(paths: Sequence[Path], target_crs: str = TARGET_EQUAL_AREA_CRS) -> dict[str, Any]:
    if not paths:
        raise ValueError("No raster paths were supplied for collection mosaicking.")
    bounds_list: list[tuple[float, float, float, float]] = []
    resolutions: list[tuple[float, float]] = []
    for path in paths:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"{path.name} has no CRS and cannot be placed safely in a collection mosaic.")
            bounds_list.append(tuple(float(v) for v in transform_bounds(src.crs, target_crs, *src.bounds, densify_pts=21)))
            resolutions.append(_projected_resolution(src, target_crs))

    # Use the coarsest native-equivalent resolution. This avoids inventing spatial
    # detail that is not present in every source tile and keeps Community Cloud
    # memory use predictable.
    res_x = max(value[0] for value in resolutions)
    res_y = max(value[1] for value in resolutions)
    left = min(value[0] for value in bounds_list)
    bottom = min(value[1] for value in bounds_list)
    right = max(value[2] for value in bounds_list)
    top = max(value[3] for value in bounds_list)

    width = max(1, int(math.ceil((right - left) / res_x)))
    height = max(1, int(math.ceil((top - bottom) / res_y)))
    cells = width * height
    if cells > MAX_MOSAIC_CELLS:
        scale = math.sqrt(cells / MAX_MOSAIC_CELLS)
        res_x *= scale
        res_y *= scale
        width = max(1, int(math.ceil((right - left) / res_x)))
        height = max(1, int(math.ceil((top - bottom) / res_y)))

    transform = from_origin(left, top, res_x, res_y)
    return {
        "crs": target_crs,
        "transform": transform,
        "width": width,
        "height": height,
        "res_x": res_x,
        "res_y": res_y,
    }


def build_collection_mosaic(predictions: Sequence[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    """Reproject per-tile outputs after inference and create one whole-area mosaic.

    Input SAR is never resampled before model inference. Only completed prediction
    products are reprojected, so the model receives exactly the native tile data.
    Binary masks use nearest-neighbour resampling; probability rasters use bilinear
    resampling and are averaged in overlaps.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_paths = [Path(item["output_mask"]) for item in predictions if item.get("output_mask")]
    if len(mask_paths) < 2:
        return {"created": False, "reason": "A collection mosaic requires at least two prediction tiles."}
    if len(mask_paths) != len(predictions):
        return {"created": False, "reason": "One or more prediction tiles did not produce a flood mask."}

    try:
        grid = _target_grid(mask_paths)
    except Exception as exc:
        return {"created": False, "reason": str(exc)}

    height = int(grid["height"])
    width = int(grid["width"])
    dst_transform = grid["transform"]
    dst_crs = grid["crs"]

    flood_union = np.zeros((height, width), dtype=np.uint8)
    coverage = np.zeros((height, width), dtype=np.uint16)
    probability_sum = np.zeros((height, width), dtype=np.float32)
    probability_count = np.zeros((height, width), dtype=np.uint16)

    for item in predictions:
        mask_path = Path(item["output_mask"])
        with rasterio.open(mask_path) as src:
            tmp_mask = np.zeros((height, width), dtype=np.uint8)
            tmp_cov = np.zeros((height, width), dtype=np.uint8)
            reproject(
                source=rasterio.band(src, 1),
                destination=tmp_mask,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
            reproject(
                source=np.ones((src.height, src.width), dtype=np.uint8),
                destination=tmp_cov,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
            valid = tmp_cov > 0
            coverage[valid] += 1
            flood_union[valid] = np.maximum(flood_union[valid], (tmp_mask[valid] > 0).astype(np.uint8))

        prob_value = item.get("output_probability")
        if prob_value and Path(prob_value).is_file():
            with rasterio.open(prob_value) as src:
                tmp_prob = np.full((height, width), np.nan, dtype=np.float32)
                tmp_cov = np.zeros((height, width), dtype=np.uint8)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=tmp_prob,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
                reproject(
                    source=np.ones((src.height, src.width), dtype=np.uint8),
                    destination=tmp_cov,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )
                valid = (tmp_cov > 0) & np.isfinite(tmp_prob)
                probability_sum[valid] += tmp_prob[valid]
                probability_count[valid] += 1

    valid_area = coverage > 0
    mask_out = np.full((height, width), 255, dtype=np.uint8)
    mask_out[valid_area] = flood_union[valid_area]
    probability_out = np.full((height, width), np.nan, dtype=np.float32)
    prob_valid = probability_count > 0
    probability_out[prob_valid] = probability_sum[prob_valid] / probability_count[prob_valid]

    mask_path = output_dir / "collection_flood_mask.tif"
    probability_path = output_dir / "collection_flood_probability.tif"
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "crs": dst_crs,
        "transform": dst_transform,
        "compress": "deflate",
    }
    with rasterio.open(mask_path, "w", **profile, dtype="uint8", nodata=255) as dst:
        dst.write(mask_out, 1)
    with rasterio.open(probability_path, "w", **profile, dtype="float32", nodata=np.nan) as dst:
        dst.write(probability_out, 1)

    pixel_area_km2 = abs(float(grid["res_x"]) * float(grid["res_y"])) / 1_000_000.0
    mapped_pixels = int(valid_area.sum())
    flood_pixels = int(((flood_union == 1) & valid_area).sum())
    mapped_area_km2 = float(mapped_pixels * pixel_area_km2)
    flood_area_km2 = float(flood_pixels * pixel_area_km2)
    flood_fraction = float(flood_area_km2 / mapped_area_km2) if mapped_area_km2 > 0 else 0.0

    preview_mask = output_dir / "collection_flood_mask_preview.png"
    preview_probability = output_dir / "collection_flood_probability_preview.png"
    _save_preview(mask_out, preview_mask, discrete=True)
    _save_preview(probability_out, preview_probability, discrete=False)

    return {
        "created": True,
        "crs": dst_crs,
        "resolution_m": [float(grid["res_x"]), float(grid["res_y"])],
        "width": width,
        "height": height,
        "mapped_pixels": mapped_pixels,
        "flood_pixels": flood_pixels,
        "mapped_area_km2": mapped_area_km2,
        "flood_area_km2": flood_area_km2,
        "flood_fraction": flood_fraction,
        "output_mask": str(mask_path),
        "output_probability": str(probability_path),
        "mask_preview": str(preview_mask),
        "probability_preview": str(preview_probability),
        "overlap_pixels": int((coverage > 1).sum()),
    }


def _save_preview(values: np.ndarray, path: Path, *, discrete: bool) -> Path:
    import matplotlib.pyplot as plt

    arr = np.asarray(values)
    max_dim = max(arr.shape)
    stride = max(1, int(math.ceil(max_dim / 1600)))
    view = arr[::stride, ::stride]
    if discrete:
        view = np.where(view == 255, np.nan, view.astype(np.float32))

    # Keep the familiar viridis flood palette, but render no-data explicitly.
    # A visible neutral background prevents pixels outside the uploaded tile
    # footprints from being mistaken for valid non-flood predictions.
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#bdbdbd")
    masked = np.ma.masked_invalid(view.astype(np.float32, copy=False))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(
        masked,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest" if discrete else "bilinear",
    )
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return path



def build_equal_area_evaluation(
    predictions: Sequence[dict[str, Any]],
    mosaic: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any] | None:
    """Evaluate labelled collection coverage on the same equal-area grid as the mosaic.

    Per-tile masks are reprojected with nearest-neighbour interpolation after model
    inference. This prevents source tiles with smaller pixels from receiving more
    weight merely because they contain more pixels per square kilometre. Partial
    mask uploads are supported: only pixels covered by a matched ground-truth mask
    are included in the collection evaluation.
    """
    if not mosaic.get("created"):
        return None
    labelled = [
        item for item in predictions
        if item.get("mask_path") and Path(str(item["mask_path"])).is_file()
    ]
    if not labelled:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = Path(str(mosaic["output_mask"]))
    with rasterio.open(prediction_path) as pred_src:
        pred = pred_src.read(1)
        dst_crs = pred_src.crs
        dst_transform = pred_src.transform
        height = pred_src.height
        width = pred_src.width
        pixel_area_km2 = abs(float(dst_transform.a) * float(dst_transform.e)) / 1_000_000.0
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "crs": dst_crs,
            "transform": dst_transform,
        }

    flood_votes = np.zeros((height, width), dtype=np.uint16)
    background_votes = np.zeros((height, width), dtype=np.uint16)

    for item in labelled:
        mask_path = Path(str(item["mask_path"]))
        with rasterio.open(mask_path) as src:
            if src.crs is None:
                raise ValueError(
                    f"{mask_path.name} has no CRS and cannot be included in whole-area evaluation."
                )
            raw = src.read(1)
            normalised = np.full(raw.shape, 255, dtype=np.uint8)
            valid = ~np.isclose(raw, 255)
            if src.nodata is not None:
                valid &= ~np.isclose(raw, src.nodata)
            normalised[valid] = np.where(np.isclose(raw[valid], 1), 1, 0).astype(np.uint8)
            reprojected = np.full((height, width), 255, dtype=np.uint8)
            reproject(
                source=normalised,
                destination=reprojected,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=255,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=255,
                resampling=Resampling.nearest,
            )
            tile_valid = reprojected != 255
            flood_votes[tile_valid & (reprojected == 1)] += 1
            background_votes[tile_valid & (reprojected == 0)] += 1

    vote_total = flood_votes + background_votes
    labelled_area = vote_total > 0
    if not np.any(labelled_area):
        return None

    # Overlapping labels should normally agree. Majority voting avoids double
    # counting while retaining a deterministic result; ties are flagged and
    # resolved as flood so disagreements are not silently converted to background.
    truth = np.full((height, width), 255, dtype=np.uint8)
    truth[labelled_area] = (flood_votes[labelled_area] >= background_votes[labelled_area]).astype(np.uint8)
    conflict = (flood_votes > 0) & (background_votes > 0)

    valid = labelled_area & (pred != 255)
    pred_bin = pred == 1
    truth_bin = truth == 1
    tp = int((valid & pred_bin & truth_bin).sum())
    tn = int((valid & ~pred_bin & ~truth_bin).sum())
    fp = int((valid & pred_bin & ~truth_bin).sum())
    fn = int((valid & ~pred_bin & truth_bin).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0

    truth_path = output_dir / "collection_ground_truth.tif"
    gt_profile = dict(profile, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(truth_path, "w", **gt_profile) as dst:
        dst.write(truth, 1)

    overlay = np.full((height, width, 3), 0.92, dtype=np.float32)
    overlay[~valid] = [1.0, 1.0, 1.0]
    overlay[valid & pred_bin & truth_bin] = [0.0, 0.75, 0.0]
    overlay[valid & pred_bin & ~truth_bin] = [0.95, 0.1, 0.1]
    overlay[valid & ~pred_bin & truth_bin] = [0.1, 0.3, 0.95]
    overlay_path = output_dir / "collection_evaluation_overlay.png"
    _save_rgb_preview(overlay, overlay_path)

    valid_pixels = int(valid.sum())
    return {
        "mode": "equal_area",
        "tiles": len(labelled),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "mcc": mcc,
        "valid_pixels": valid_pixels,
        "labelled_area_km2": float(valid_pixels * pixel_area_km2),
        "conflict_pixels": int((conflict & valid).sum()),
        "ground_truth_mosaic": str(truth_path),
        "evaluation_overlay": str(overlay_path),
    }


def _save_rgb_preview(values: np.ndarray, path: Path) -> Path:
    import matplotlib.pyplot as plt

    arr = np.asarray(values, dtype=np.float32)
    max_dim = max(arr.shape[:2])
    stride = max(1, int(math.ceil(max_dim / 1600)))
    view = arr[::stride, ::stride]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(np.clip(view, 0.0, 1.0), interpolation="nearest")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return path

def pooled_evaluation(predictions: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    evaluations = [item.get("evaluation_metrics") for item in predictions if item.get("evaluation_metrics")]
    if not evaluations:
        return None
    tp = sum(int(item.get("tp") or 0) for item in evaluations)
    fp = sum(int(item.get("fp") or 0) for item in evaluations)
    fn = sum(int(item.get("fn") or 0) for item in evaluations)
    tn = sum(int(item.get("tn") or 0) for item in evaluations)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
    return {
        "mode": "source_grid_pooled",
        "tiles": len(evaluations),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "mcc": mcc,
    }


def write_collection_report(
    path: Path,
    predictions: Sequence[dict[str, Any]],
    mosaic: dict[str, Any],
    evaluation: dict[str, Any] | None,
    model_summary: dict[str, Any],
) -> Path:
    path = Path(path)
    def img_tag(value: str | None, caption: str) -> str:
        if not value or not Path(value).is_file():
            return ""
        encoded = base64.b64encode(Path(value).read_bytes()).decode("ascii")
        return f"<figure><img src='data:image/png;base64,{encoded}' alt='{html.escape(caption)}'><figcaption>{html.escape(caption)}</figcaption></figure>"

    metrics = ""
    if evaluation:
        if evaluation.get("mode") == "equal_area":
            evaluation_note = (
                f"Area-consistent metrics on the post-inference equal-area grid across "
                f"{evaluation['tiles']} matched ground-truth tile(s); labelled area "
                f"{float(evaluation.get('labelled_area_km2') or 0.0):.3f} km²."
            )
            evaluation_title = "Whole-area labelled evaluation"
            evaluation_image = img_tag(evaluation.get("evaluation_overlay"), "Evaluation overlay — green TP, red FP, blue FN")
        else:
            evaluation_note = f"Pixel-weighted pooled metrics across {evaluation['tiles']} matched ground-truth tile(s)."
            evaluation_title = "Labelled-tile evaluation"
            evaluation_image = ""
        metrics = f"""
        <h2>{evaluation_title}</h2>
        <p>{html.escape(evaluation_note)}</p>
        <table><tr><th>F1</th><th>IoU</th><th>Precision</th><th>Recall</th><th>MCC</th></tr>
        <tr><td>{evaluation['f1']:.4f}</td><td>{evaluation['iou']:.4f}</td><td>{evaluation['precision']:.4f}</td><td>{evaluation['recall']:.4f}</td><td>{evaluation['mcc']:.4f}</td></tr></table>
        {evaluation_image}
        """

    rows = "".join(
        f"<tr><td>{html.escape(str(item.get('candidate_id') or 'tile'))}</td>"
        f"<td>{int(item.get('flood_pixels') or 0):,}</td>"
        f"<td>{float(item.get('flood_fraction') or 0.0) * 100:.2f}%</td>"
        f"<td>{'–' if item.get('flood_area_km2') is None else f'{float(item.get('flood_area_km2')):.3f}'}</td></tr>"
        for item in predictions
    )
    mosaic_summary = ""
    if mosaic.get("created"):
        mosaic_summary = f"""
        <h2>Whole-area prediction</h2>
        <div class='cards'>
          <div><strong>Tiles processed</strong><span>{len(predictions)}</span></div>
          <div><strong>Mapped area</strong><span>{float(mosaic['mapped_area_km2']):.3f} km²</span></div>
          <div><strong>Predicted flood area</strong><span>{float(mosaic['flood_area_km2']):.3f} km²</span></div>
          <div><strong>Flood coverage</strong><span>{float(mosaic['flood_fraction']) * 100:.2f}%</span></div>
        </div>
        <p>Collection mosaics are created after inference in equal-area CRS {html.escape(str(mosaic.get('crs')))}. Binary predictions use nearest-neighbour reprojection and union in overlaps; probability surfaces use bilinear reprojection and are averaged in overlaps.</p>
        <p><strong>Map legend:</strong> flood-mask yellow = predicted flood, purple = predicted non-flood, grey = no input coverage. Grey also marks no input coverage on the probability map.</p>
        <div class='grid'>{img_tag(mosaic.get('mask_preview'), 'Whole-area flood mask')}{img_tag(mosaic.get('probability_preview'), 'Whole-area flood probability')}</div>
        """

    content = f"""<!doctype html><html><head><meta charset='utf-8'><title>Flood Extent Mapping report</title>
<style>body{{font-family:Arial,sans-serif;margin:28px;line-height:1.45;color:#202124}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 0}}.cards div{{border:1px solid #ddd;border-radius:8px;padding:12px}}.cards strong,.cards span{{display:block}}.cards span{{font-size:1.3rem;margin-top:6px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}}figure{{margin:0}}img{{width:100%;height:auto}}figcaption{{font-weight:600;margin-top:6px}}</style></head><body>
<h1>Flood Extent Mapping</h1><p>Deployment model: {html.escape(str(model_summary.get('name') or 'model'))} · modalities: {html.escape(', '.join(model_summary.get('modalities') or []))}</p>
{mosaic_summary}{metrics}
<h2>Per-tile predictions</h2><table><tr><th>Tile</th><th>Flood pixels</th><th>Flood fraction</th><th>Approx. flood area (km²)</th></tr>{rows}</table>
</body></html>"""
    path.write_text(content, encoding="utf-8")
    return path
