from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import rasterio
from floods.modalities import (
    BASE_MODALITIES,
    DERIVED_MODALITIES,
    RGB_MODALITIES,
    SUPPORTED_MODALITIES,
    canonicalize_modalities,
    canonicalize_modality,
    has_derived_modalities,
    infer_default_modalities,
    resolve_input_modalities,
)
from floods.utils.common import get_logger
from floods.utils.console import progress_iter
from floods.utils.gis import imread

LOG = get_logger(__name__)

DERIVED_BAND_ORDER = DERIVED_MODALITIES
DERIVED_SCHEMA_VERSION = 2

@dataclass(frozen=True)
class DerivedFeatureParameters:
    log_ratio_eps: float = 1e-6
    tpi_radius_pixels: int = 15

    def validate(self) -> "DerivedFeatureParameters":
        if not math.isfinite(float(self.log_ratio_eps)) or float(self.log_ratio_eps) <= 0:
            raise ValueError("log_ratio_eps must be a positive finite value")
        if int(self.tpi_radius_pixels) < 1:
            raise ValueError("tpi_radius_pixels must be at least 1")
        return self




def pixel_spacing_meters(transform, crs, width: int, height: int) -> tuple[float, float]:
    """Return horizontal pixel spacing in metres for projected or geographic rasters.

    DEM elevations are measured in metres, so slope gradients must also use metre
    horizontal spacing. EPSG:4326 transforms expose degrees per pixel; treating
    those values as metres saturates almost every slope near 90 degrees.
    """
    if transform is None:
        raise ValueError("Raster transform is required to derive metre-scale DEM slope")
    if crs is None:
        raise ValueError("Raster CRS is required to derive metre-scale DEM slope")

    width = int(width)
    height = int(height)
    if width < 1 or height < 1:
        raise ValueError("Raster width and height must be positive")

    # Pixel basis vectors in CRS units. This also handles rotated transforms.
    x_dx = float(transform.a)
    x_dy = float(transform.d)
    y_dx = float(transform.b)
    y_dy = float(transform.e)

    if crs.is_geographic:
        # Centre latitude is sufficient at tile scale. Convert longitude and
        # latitude degrees independently using standard ellipsoidal approximations.
        _, centre_lat = transform * (width / 2.0, height / 2.0)
        latitude = math.radians(max(-89.999999, min(89.999999, float(centre_lat))))
        metres_per_degree_lat = (
            111132.92
            - 559.82 * math.cos(2.0 * latitude)
            + 1.175 * math.cos(4.0 * latitude)
            - 0.0023 * math.cos(6.0 * latitude)
        )
        metres_per_degree_lon = (
            111412.84 * math.cos(latitude)
            - 93.5 * math.cos(3.0 * latitude)
            + 0.118 * math.cos(5.0 * latitude)
        )
        x_metres = math.hypot(
            x_dx * metres_per_degree_lon,
            x_dy * metres_per_degree_lat,
        )
        y_metres = math.hypot(
            y_dx * metres_per_degree_lon,
            y_dy * metres_per_degree_lat,
        )
    elif crs.is_projected:
        _, unit_to_metres = crs.linear_units_factor
        factor = float(unit_to_metres)
        x_metres = math.hypot(x_dx, x_dy) * factor
        y_metres = math.hypot(y_dx, y_dy) * factor
    else:
        raise ValueError(f"Unsupported CRS for metre-scale slope derivation: {crs}")

    if not math.isfinite(x_metres) or x_metres <= 0:
        raise ValueError(f"Invalid horizontal pixel spacing: {x_metres}")
    if not math.isfinite(y_metres) or y_metres <= 0:
        raise ValueError(f"Invalid vertical pixel spacing: {y_metres}")
    return x_metres, y_metres


def _derived_file_is_current(path: Path, params: DerivedFeatureParameters) -> bool:
    """Return True only for a complete derived raster written by this schema."""
    try:
        with rasterio.open(path) as src:
            tags = src.tags()
            return (
                src.count == len(DERIVED_BAND_ORDER)
                and tuple(src.descriptions) == DERIVED_BAND_ORDER
                and tags.get("derived_schema_version") == str(DERIVED_SCHEMA_VERSION)
                and tags.get("log_ratio_eps") == str(params.log_ratio_eps)
                and tags.get("tpi_radius_pixels") == str(params.tpi_radius_pixels)
            )
    except (OSError, rasterio.errors.RasterioError):
        return False




def _channels_last(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported raster shape: {arr.shape}")
    channels_first = arr.shape[0] <= 12 and (
        arr.shape[-1] > 12
        or (arr.shape[1] == arr.shape[2] and arr.shape[0] != arr.shape[-1])
    )
    if channels_first:
        return np.moveaxis(arr, 0, -1)
    return arr


def _read_raster_channels(path: Path) -> np.ndarray:
    arr = imread(path, channels_first=True)
    return _channels_last(arr).astype(np.float32, copy=False)


def processed_modality_paths(processed_data_dir: Path, split: str, stem: str) -> dict[str, Path]:
    root = Path(processed_data_dir) / split
    return {
        "sar": root / "sar" / f"{stem}.tif",
        "dem": root / "dem" / f"{stem}.tif",
        "derived": root / "derived" / f"{stem}.tif",
        "mask": root / "mask" / f"{stem}.tif",
    }


def read_processed_modalities(
    processed_data_dir: Path,
    split: str,
    stem: str,
    modalities: Sequence[str],
) -> np.ndarray:
    """Read requested processed channels in their declared order as H,W,C float32."""
    modalities = canonicalize_modalities(modalities)
    paths = processed_modality_paths(processed_data_dir, split, stem)
    arrays: dict[str, np.ndarray] = {}

    if any(m in {"vv", "vh", "r", "g", "b"} for m in modalities):
        if not paths["sar"].exists():
            raise FileNotFoundError(f"Missing SAR tile: {paths['sar']}")
        arrays["sar"] = _read_raster_channels(paths["sar"])

    if any(m == "dem" for m in modalities):
        if not paths["dem"].exists():
            raise FileNotFoundError(f"Missing DEM tile: {paths['dem']}")
        arrays["dem"] = _read_raster_channels(paths["dem"])

    if any(m in DERIVED_MODALITIES for m in modalities):
        if not paths["derived"].exists():
            raise FileNotFoundError(
                f"Missing derived-feature tile: {paths['derived']}. "
                "Run `floodmap derive-features` before fitting stats or training."
            )
        arrays["derived"] = _read_raster_channels(paths["derived"])
        if arrays["derived"].shape[-1] < len(DERIVED_BAND_ORDER):
            raise ValueError(
                f"Derived tile {paths['derived']} has {arrays['derived'].shape[-1]} band(s); "
                f"expected {len(DERIVED_BAND_ORDER)} in order {DERIVED_BAND_ORDER}."
            )

    band_indices = {
        "vv": ("sar", 0),
        "vh": ("sar", 1),
        "r": ("sar", 0),
        "g": ("sar", 1),
        "b": ("sar", 2),
        "dem": ("dem", 0),
        "vv_vh_log_ratio": ("derived", 0),
        "dem_slope": ("derived", 1),
        "dem_tpi": ("derived", 2),
    }
    channels: list[np.ndarray] = []
    spatial_shape: tuple[int, int] | None = None
    for modality in modalities:
        group, band = band_indices[modality]
        arr = arrays[group]
        if band >= arr.shape[-1]:
            raise ValueError(
                f"Tile {paths[group]} does not contain band {band + 1} required for modality '{modality}'"
            )
        channel = arr[..., band]
        if spatial_shape is None:
            spatial_shape = channel.shape
        elif channel.shape != spatial_shape:
            raise ValueError(
                f"Spatial mismatch while loading {stem}: modality {modality} has {channel.shape}, expected {spatial_shape}"
            )
        channels.append(channel)
    return np.stack(channels, axis=-1).astype(np.float32, copy=False)


def derive_feature_channels(
    sar: np.ndarray,
    dem: np.ndarray | None,
    *,
    pixel_size_x: float,
    pixel_size_y: float,
    modalities: Sequence[str],
    params: DerivedFeatureParameters | None = None,
) -> dict[str, np.ndarray]:
    """Compute only the requested derived channels.

    SciPy is imported only when ``dem_tpi`` is requested. VV/VH ratio and DEM
    slope therefore remain available in lean inference environments where SciPy
    is intentionally absent.
    """
    params = (params or DerivedFeatureParameters()).validate()
    requested = canonicalize_modalities(modalities)
    unsupported = [name for name in requested if name not in DERIVED_MODALITIES]
    if unsupported:
        raise ValueError(
            f"derive_feature_channels accepts only derived modalities; got {unsupported}"
        )

    sar = _channels_last(np.asarray(sar, dtype=np.float32))
    if sar.shape[-1] < 2:
        raise ValueError(f"SAR input must contain VV and VH bands; got shape {sar.shape}")
    vv = sar[..., 0]
    vh = sar[..., 1]
    result: dict[str, np.ndarray] = {}

    if "vv_vh_log_ratio" in requested:
        eps = float(params.log_ratio_eps)
        ratio = 10.0 * np.log10(np.maximum(vv, eps)) - 10.0 * np.log10(np.maximum(vh, eps))
        finite_sar = np.isfinite(vv) & np.isfinite(vh)
        result["vv_vh_log_ratio"] = np.where(finite_sar, ratio, np.nan).astype(
            np.float32,
            copy=False,
        )

    terrain_requested = any(name in {"dem_slope", "dem_tpi"} for name in requested)
    if terrain_requested:
        if dem is None:
            raise ValueError("DEM is required for dem_slope or dem_tpi")
        dem_array = _channels_last(np.asarray(dem, dtype=np.float32))[..., 0]
        if sar.shape[:2] != dem_array.shape:
            raise ValueError(f"SAR and DEM shapes differ: {sar.shape[:2]} != {dem_array.shape}")
        valid_dem = np.isfinite(dem_array)
        if not valid_dem.any():
            raise ValueError("DEM contains no finite pixels")
        fill_value = float(np.nanmedian(dem_array[valid_dem]))
        dem_filled = np.where(valid_dem, dem_array, fill_value).astype(np.float32, copy=False)

        if "dem_slope" in requested:
            x_res = max(abs(float(pixel_size_x)), 1e-6)
            y_res = max(abs(float(pixel_size_y)), 1e-6)
            grad_y, grad_x = np.gradient(dem_filled, y_res, x_res)
            slope = np.degrees(np.arctan(np.hypot(grad_x, grad_y))).astype(np.float32, copy=False)
            result["dem_slope"] = np.where(valid_dem, slope, np.nan).astype(np.float32, copy=False)

        if "dem_tpi" in requested:
            try:
                from scipy.ndimage import uniform_filter
            except (ImportError, AttributeError) as exc:
                raise RuntimeError(
                    "DEM TPI derivation requires a working SciPy installation. "
                    "Run `floodmap doctor` and reinstall the tested numeric stack before "
                    "using dem_tpi or build-derived-features. VV/VH deployment does not "
                    "require SciPy."
                ) from exc
            radius = int(params.tpi_radius_pixels)
            size = 2 * radius + 1
            local_mean = uniform_filter(dem_filled, size=size, mode="nearest")
            tpi = (dem_filled - local_mean).astype(np.float32, copy=False)
            result["dem_tpi"] = np.where(valid_dem, tpi, np.nan).astype(np.float32, copy=False)

    return result


def derive_feature_stack(
    sar: np.ndarray,
    dem: np.ndarray,
    *,
    pixel_size_x: float,
    pixel_size_y: float,
    params: DerivedFeatureParameters | None = None,
) -> np.ndarray:
    """Create VV/VH log-ratio, DEM slope, and local DEM TPI.

    ``pixel_size_x`` and ``pixel_size_y`` must be expressed in metres.
    """
    channels = derive_feature_channels(
        sar,
        dem,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
        modalities=DERIVED_BAND_ORDER,
        params=params,
    )
    return np.stack([channels[name] for name in DERIVED_BAND_ORDER], axis=0).astype(
        np.float32,
        copy=False,
    )


def build_derived_features(
    processed_data_dir: Path,
    *,
    splits: Sequence[str] = ("train", "val", "test"),
    log_ratio_eps: float = 1e-6,
    tpi_radius_pixels: int = 15,
    overwrite: bool = False,
) -> dict:
    """Write reproducible three-band derived-feature GeoTIFFs beside processed tiles."""
    processed_data_dir = Path(processed_data_dir)
    params = DerivedFeatureParameters(
        log_ratio_eps=float(log_ratio_eps),
        tpi_radius_pixels=int(tpi_radius_pixels),
    ).validate()
    split_summaries: dict[str, dict[str, int]] = {}

    for split in splits:
        sar_dir = processed_data_dir / split / "sar"
        dem_dir = processed_data_dir / split / "dem"
        out_dir = processed_data_dir / split / "derived"
        sar_paths = sorted(sar_dir.glob("*.tif"))
        if not sar_paths:
            raise FileNotFoundError(f"No SAR tiles found under {sar_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        written = skipped = stale_rewritten = 0
        for sar_path in progress_iter(sar_paths, desc=f"Derive features {split}", unit="tile", colour="cyan"):
            stem = sar_path.stem
            dem_path = dem_dir / f"{stem}.tif"
            out_path = out_dir / f"{stem}.tif"
            if not dem_path.exists():
                raise FileNotFoundError(f"Missing DEM tile for {stem}: {dem_path}")
            if out_path.exists() and not overwrite:
                if _derived_file_is_current(out_path, params):
                    skipped += 1
                    continue
                stale_rewritten += 1
            with rasterio.open(sar_path) as sar_src, rasterio.open(dem_path) as dem_src:
                sar = sar_src.read().astype(np.float32, copy=False)
                dem = dem_src.read(1).astype(np.float32, copy=False)
                if sar_src.width != dem_src.width or sar_src.height != dem_src.height:
                    raise ValueError(f"SAR/DEM shape mismatch for {stem}")
                if sar_src.transform != dem_src.transform or sar_src.crs != dem_src.crs:
                    raise ValueError(f"SAR/DEM geospatial mismatch for {stem}")
                pixel_size_x_m, pixel_size_y_m = pixel_spacing_meters(
                    sar_src.transform,
                    sar_src.crs,
                    sar_src.width,
                    sar_src.height,
                )
                derived = derive_feature_stack(
                    sar,
                    dem,
                    pixel_size_x=pixel_size_x_m,
                    pixel_size_y=pixel_size_y_m,
                    params=params,
                )
                profile = sar_src.profile.copy()
                if not profile.get("tiled", False):
                    profile.pop("blockxsize", None)
                    profile.pop("blockysize", None)
                profile.update(
                    count=len(DERIVED_BAND_ORDER),
                    dtype="float32",
                    compress="deflate",
                    predictor=3,
                    nodata=np.nan,
                )
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(derived)
                    for idx, name in enumerate(DERIVED_BAND_ORDER, start=1):
                        dst.set_band_description(idx, name)
                    dst.update_tags(
                        derived_schema_version=str(DERIVED_SCHEMA_VERSION),
                        slope_horizontal_units="metres",
                        pixel_size_x_metres=f"{pixel_size_x_m:.9f}",
                        pixel_size_y_metres=f"{pixel_size_y_m:.9f}",
                        log_ratio_eps=str(params.log_ratio_eps),
                        tpi_radius_pixels=str(params.tpi_radius_pixels),
                    )
            written += 1
        split_summaries[str(split)] = {
            "sar_tiles": len(sar_paths),
            "written": written,
            "skipped_existing": skipped,
            "stale_rewritten": stale_rewritten,
        }
        LOG.info(
            "Derived features ready: split=%s | tiles=%d | written=%d | skipped=%d | stale_rewritten=%d | output=%s",
            split,
            len(sar_paths),
            written,
            skipped,
            stale_rewritten,
            out_dir,
        )

    manifest = {
        "schema_version": DERIVED_SCHEMA_VERSION,
        "processed_data_dir": str(processed_data_dir),
        "band_order": list(DERIVED_BAND_ORDER),
        "formulas": {
            "vv_vh_log_ratio": "10*log10(max(VV,eps)) - 10*log10(max(VH,eps))",
            "dem_slope": "degrees(arctan(sqrt((dDEM/dx_m)^2 + (dDEM/dy_m)^2)))",
            "dem_tpi": "DEM - local_mean(DEM)",
        },
        "parameters": {
            "log_ratio_eps": params.log_ratio_eps,
            "tpi_radius_pixels": params.tpi_radius_pixels,
            "tpi_window_pixels": 2 * params.tpi_radius_pixels + 1,
            "slope_horizontal_units": "metres",
            "geographic_spacing_method": "tile-centre ellipsoidal metres-per-degree approximation",
        },
        "splits": split_summaries,
    }
    manifest_path = processed_data_dir / "derived_features_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOG.info("Derived-feature manifest written to: %s", manifest_path)
    return manifest
