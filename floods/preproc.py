import json
import logging
import os
import shutil
from glob import glob
from pathlib import Path
from typing import Callable, Counter, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import rasterio
from joblib import Parallel, delayed
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.windows import Window
from rasterio.warp import reproject
from skimage.restoration import denoise_nl_means
from floods.utils.console import progress_iter

from floods.config.preproc import ImageType, PreparationConfig, StatsConfig
from floods.datasets.flood import FloodDataset
from floods.utils.common import check_or_make_dir, print_config
from floods.utils.gis import imread, mask_raster, write_window
from floods.utils.ml import F16_EPS, identity
from floods.utils.tiling import DynamicOverlapTiler, SingleImageTiler, Tiler

LOG = logging.getLogger(__name__)


class MaskPreprocessor:
    """Map raw flood masks to 0/1/255 and optionally clean the flood class.

    Output mask values are fixed across the project:
    0 = valid non-flood/background, 1 = flood/foreground, 255 = ignore/nodata.
    Morphology is applied only to the binary flood class. Ignore pixels are
    restored afterwards so nodata areas cannot become training labels.
    """

    def __init__(self,
                 kernel_size: int = 5,
                 channels_first: bool = True,
                 flood_values: Optional[List[int]] = None,
                 background_values: Optional[List[int]] = None,
                 ignore_values: Optional[List[int]] = None,
                 apply_morphology: bool = True,
                 preserve_ignore: bool = True) -> None:
        self.kernel = self.create_round_kernel(kernel_size=kernel_size)
        self.channels_first = channels_first
        self.flood_values = set(int(v) for v in (flood_values or [1]))
        self.background_values = set(int(v) for v in (background_values or [0, 2]))
        self.ignore_values = set(int(v) for v in (ignore_values or [255]))
        self.apply_morphology = bool(apply_morphology)
        self.preserve_ignore = bool(preserve_ignore)

    @staticmethod
    def create_round_kernel(kernel_size: int):
        """Create a compact circular structuring element for binary masks."""
        center = kernel_size // 2
        radius = min(center, kernel_size - center)
        yy, xx = np.ogrid[:kernel_size, :kernel_size]
        distance = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
        return (distance <= radius).astype(np.uint8)

    def _to_channels_last_2d(self, image: np.ndarray) -> np.ndarray:
        if self.channels_first and image.ndim == 3:
            image = image.transpose(1, 2, 0)
        if image.ndim == 3:
            if image.shape[-1] != 1:
                raise ValueError(f"Expected a single-channel mask, got shape {image.shape}")
            image = image[..., 0]
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got shape {image.shape}")
        return image

    def _remap(self, raw_mask: np.ndarray) -> np.ndarray:
        raw_mask = raw_mask.astype(np.int32, copy=False)
        output = np.full(raw_mask.shape, 255, dtype=np.uint8)
        if self.background_values:
            output[np.isin(raw_mask, list(self.background_values))] = 0
        if self.flood_values:
            output[np.isin(raw_mask, list(self.flood_values))] = 1
        if self.ignore_values:
            output[np.isin(raw_mask, list(self.ignore_values))] = 255
        return output

    def _process_mask(self, image: np.ndarray):
        raw_mask = self._to_channels_last_2d(image)
        mapped_mask = self._remap(raw_mask)
        ignore_mask = mapped_mask == 255
        flood_mask = (mapped_mask == 1).astype(np.uint8)
        if self.apply_morphology:
            flood_mask = cv2.morphologyEx(flood_mask, cv2.MORPH_CLOSE, self.kernel)
        output = flood_mask.astype(np.uint8)
        if self.preserve_ignore:
            output[ignore_mask] = 255
        return np.expand_dims(output, axis=0 if self.channels_first else -1)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self._process_mask(image)


class MorphologyTransform(MaskPreprocessor):
    """Alias for mask preprocessing used by existing imports."""


def _dims(image: np.ndarray) -> Tuple[int, int]:
    """Return the first two dimensions of a channels-last array.

    Args:
        image (np.ndarray): array representing an image

    Returns:
        Tuple[int, int]: height and width
    """
    return image.shape[:2]


def _extract_emsr(path: Union[str, Path]) -> str:
    """Transforms string like 'EMSR345-0-0' into 'EMSR345'

    Args:
        image_id (str): image stem

    Returns:
        str: EMSR code of the image
    """
    image_id = Path(path).stem
    return image_id.split("-")[0]


def _delete_group(*paths: Tuple[Path, ...]) -> None:
    """Delete a group of aligned tile files."""
    for path in paths:
        if os.path.exists(str(path)):
            os.remove(path)


_PROCESSED_MODALITY_DIRS = ("sar", "dem", "mask", "derived", "weight", "context")


def _prepare_subset_output(subset_dir: Path, overwrite: bool = False) -> None:
    """Refuse to tile into a split folder that already holds tiles, unless overwriting.

    Tiles are named by scene and pixel offset, so a re-run with different settings
    (tile size, overlap, transforms) would otherwise mix old and new tiles silently.
    """
    subset_dir = Path(subset_dir)
    existing = [subset_dir / name for name in _PROCESSED_MODALITY_DIRS if (subset_dir / name).exists()]
    has_files = any(any(path.rglob("*.tif")) for path in existing)
    if not has_files:
        return
    if not overwrite:
        raise FileExistsError(
            f"{subset_dir} already contains processed tiles. Use a new --data-processed folder, "
            "or set overwrite_existing: true (CLI --overwrite-existing) to delete and rebuild this split."
        )
    for path in existing:
        LOG.info("Removing existing processed tiles: %s", path)
        shutil.rmtree(path)


def _gather_files(sar_glob: Path,
                  dem_glob: Path,
                  mask_glob: Path,
                  check_stems: bool = True,
                  subset: Set[str] = None) -> Tuple[list, list, list]:
    """Gathers files from the given glob definitions, then checks for consistency.

    Args:
        sar_glob (Path): Glob pattern for SAR rasters.
        dem_glob (Path): Glob pattern for DEM rasters.
        mask_glob (Path): Glob pattern for mask rasters.
        check_stems (bool, optional): Whether to validate matching file stems. Defaults to True.

    Returns:
        Tuple[list, list, list]: tuple of lists, with matching paths at each index i
    """
    # Collect source files and validate alignment across modalities.
    sar_files = sorted(glob(str(sar_glob)))
    dem_files = sorted(glob(str(dem_glob)))
    msk_files = sorted(glob(str(mask_glob)))
    # Ensure files exist, counts match, and stems align in sorted order.
    if not sar_files:
        raise FileNotFoundError(f"No SAR rasters found for pattern: {sar_glob}")
    if not (len(sar_files) == len(msk_files) == len(dem_files)):
        raise ValueError(
            "Raw raster count mismatch: "
            f"SAR={len(sar_files)}, DEM={len(dem_files)}, mask={len(msk_files)}"
        )
    if check_stems:
        for paths in zip(sar_files, dem_files, msk_files):
            names = [Path(p).stem for p in paths]
            if names.count(names[0]) != len(names):
                raise ValueError(f"Raw raster stem mismatch: {names}")
    if subset:
        sar_files = list(filter(lambda p: _extract_emsr(p) in subset, sar_files))
        dem_files = list(filter(lambda p: _extract_emsr(p) in subset, dem_files))
        msk_files = list(filter(lambda p: _extract_emsr(p) in subset, msk_files))

    return sar_files, dem_files, msk_files


def _sar_nodata_to_nan(data: np.ndarray) -> np.ndarray:
    """Return float32 linear backscatter with no-data marked as NaN.

    Sentinel-1 GRD products encode no-data (e.g. outside the swath) as 0, and
    linear backscatter is strictly positive, so non-positive and non-finite
    values are treated as missing rather than as very dark (water-like) pixels.
    """
    data = np.asarray(data, dtype=np.float32)
    return np.where(np.isfinite(data) & (data > 0.0), data, np.nan).astype(np.float32, copy=False)


def _sar_db10(data: np.ndarray):
    """Convert positive linear SAR backscatter to dB; no-data becomes NaN."""
    data = _sar_nodata_to_nan(data)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (10.0 * np.log10(np.maximum(data, F16_EPS))).astype(np.float32, copy=False)


def _sar_log1p(data: np.ndarray):
    """log10(1 + x) transform for positive SAR values; no-data becomes NaN."""
    data = _sar_nodata_to_nan(data)
    with np.errstate(invalid="ignore"):
        return np.log10(1.0 + data + F16_EPS).astype(np.float32, copy=False)


def _sar_linear(data: np.ndarray):
    """Keep linear SAR values; no-data becomes NaN."""
    return _sar_nodata_to_nan(data)


def _build_sar_transform(config: PreparationConfig):
    """Return the SAR transform selected by preprocessing configuration."""
    # SAR transform is controlled explicitly by sar_transform.
    if bool(getattr(config, "colorize", False)):
        return _rgb_ratio
    mode = str(getattr(config, "sar_transform", "db10") or "linear").lower().replace("-", "_")
    if mode in {"linear", "none"}:
        return _sar_linear
    if mode in {"db10", "db", "decibel"}:
        return _sar_db10
    if mode in {"log1p", "log"}:
        return _sar_log1p
    if bool(getattr(config, "decibel", False)):
        return _sar_db10
    raise ValueError(f"Unsupported sar_transform: {mode}")


def _rgb_ratio(data: np.ndarray):
    """False-color RGB formula taken directly from Sentinel-hub.
    """
    vv, vh = data[0], data[1]
    r = vv / 0.28
    g = vh / 0.06
    b = vh / vv / 0.49
    return np.clip(np.stack((r, g, b), axis=0), 0, 1)


# Elevations outside this range are treated as DEM voids/fill values (e.g. -32768, -9999).
DEM_VALID_MIN = -1000.0
DEM_VALID_MAX = 9000.0


def _dem_nodata_to_nan(data: np.ndarray) -> np.ndarray:
    """Return float32 DEM values with voids and implausible fill values marked as NaN."""
    data = np.asarray(data, dtype=np.float32)
    valid = np.isfinite(data) & (data >= DEM_VALID_MIN) & (data <= DEM_VALID_MAX)
    return np.where(valid, data, np.nan).astype(np.float32, copy=False)


def _clip_dem(data: np.ndarray):
    """Mark DEM voids as NaN, then clip valid elevations to [-100, 6000] metres."""
    data = _dem_nodata_to_nan(data)
    with np.errstate(invalid="ignore"):
        return np.clip(data, -100.0, 6000.0).astype(np.float32, copy=False)


def _prepare_source_array(image: np.ndarray, image_type: ImageType, src_nodata) -> np.ndarray:
    """Mark declared source no-data as NaN for continuous rasters (SAR/DEM)."""
    if image_type == ImageType.MASK:
        return image
    image = np.asarray(image, dtype=np.float32).copy()
    if src_nodata is not None:
        try:
            nodata_value = float(src_nodata)
        except (TypeError, ValueError):
            nodata_value = None
        if nodata_value is not None and np.isfinite(nodata_value):
            image[image == nodata_value] = np.nan
    return image


def _output_profile_for(profile: dict, processed_img: np.ndarray, image_type: ImageType) -> dict:
    """Return an output profile whose dtype/nodata match the processed array."""
    profile = profile.copy()
    profile.update(height=processed_img.shape[1], width=processed_img.shape[2], count=processed_img.shape[0], driver="GTiff")
    if image_type == ImageType.MASK:
        profile.update(dtype="uint8", nodata=255)
    else:
        profile.update(dtype="float32", nodata=float("nan"))
    if not profile.get("tiled", False):
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
    return profile


def _grid_signature(path: Path) -> dict:
    with rasterio.open(str(path), mode="r") as src:
        return {"width": src.width, "height": src.height, "transform": src.transform, "crs": src.crs}


def _check_same_grid(sar_path: Path, dem_path: Path, mask_path: Path) -> None:
    """Raise when SAR, DEM and mask do not share one pixel grid.

    Without ``align_to_reference_grid`` the three rasters are tiled by pixel
    index, so any difference in shape, transform or CRS silently pairs the
    wrong ground locations.
    """
    reference = _grid_signature(sar_path)
    for label, path in (("DEM", dem_path), ("mask", mask_path)):
        other = _grid_signature(path)
        problems = []
        if (other["width"], other["height"]) != (reference["width"], reference["height"]):
            problems.append(f"shape {other['height']}x{other['width']} != {reference['height']}x{reference['width']}")
        if reference["crs"] is not None and other["crs"] is not None and other["crs"] != reference["crs"]:
            problems.append(f"CRS {other['crs']} != {reference['crs']}")
        if not other["transform"].almost_equals(reference["transform"], precision=1e-6):
            problems.append("affine transform differs")
        if problems:
            raise ValueError(
                f"{label} raster {Path(path).name} is not on the SAR grid of {Path(sar_path).name}: "
                + "; ".join(problems)
                + ". Re-run with align_to_reference_grid: true (CLI --align-to-reference-grid) to reproject it onto the SAR grid."
            )


def _process_tiff(image_id: str,
                  source_path: Path,
                  dst_path: Path,
                  image_type: ImageType,
                  tiling_fn: Tiler,
                  process_fn: Optional[Callable] = None,
                  scale: float = 1.0,
                  resampling: Resampling = Resampling.bilinear,
                  is_context: bool = False,
                  name_suffix: str = "") -> Tuple[int, int]:
    """Read, validate, preprocess, and tile one source raster.

    Args:
        image_id (str): emsr-like code identifier of the tuple.
        source_path (Path): path to the image.
        dst_path (Path): path where to store the tiles.
        image_type (ImageType): sar, dem or mask.
        tiling_fn (Callable): callable for the tiling operation, yields coordinates.
        process_fn (Optional[Callable], optional): optional callable for mask processing. Defaults to None.
        scale (Optional[float]): optional value to up/downscale the image before tiling.
        resampling (Optional[Resampling]): resampling strategy, defaults to bilinear.
        is_context (Optional[bool]): whether the current image is a context image (should be downsampled).
        name_suffix (Optional[str]): optional suffix to add at the end of the file (useful for multi-scale).

    Returns:
        Tuple[int, int]: number of tiles for each axis
    """
    group, _ = image_type.value
    process_fn = process_fn or identity
    # Create destination folders for each modality and optional context tiles.
    ctx_dir = "context" if is_context else ""
    root_dir = Path(dst_path) / group / ctx_dir
    check_or_make_dir(root_dir)
    # Read the raster and adjust the transform when resized output is required.
    with rasterio.open(str(source_path), mode="r", driver="GTiff") as dataset:
        # Multiscale tiles are resized proportionally; context tiles are resized to tile size.
        orig_height = dataset.height
        orig_width = dataset.width
        if is_context:
            out_shape = (dataset.count, tiling_fn.tile_size, tiling_fn.tile_size)
        else:
            out_shape = (dataset.count, int(dataset.height * scale), int(dataset.width * scale))
        image = dataset.read(out_shape=out_shape, resampling=resampling)
        _, height, width = image.shape
        # given the possible resize, both transform and dimensions need to be updated
        transform = dataset.transform * dataset.transform.scale(dataset.width / width, dataset.height / height)
        image = _prepare_source_array(image, image_type, dataset.nodata)
        # Transform the image to extract the number of channels, which can change after processing.
        processed_img = process_fn(image)
        # Output dtype/nodata follow the processed array: float32 + NaN for SAR/DEM, uint8 + 255 for masks.
        profile = _output_profile_for(dataset.profile, processed_img, image_type)
        # create and open an in-memory file to store results
        # during preprocessing
        with MemoryFile() as memfile:
            with memfile.open(**profile) as processed:
                processed.write(processed_img)
                processed.transform = transform
                generator = tiling_fn(image)
                # store result, using target raster (which could be either)
                for (tile_row, tile_col), coords in generator:
                    x1, y1, x2, y2 = coords
                    window = Window.from_slices(rows=(x1, x2), cols=(y1, y2))
                    if is_context:
                        tile_path = root_dir / f"{image_id}_{orig_height}_{orig_width}{name_suffix}.tif"
                    else:
                        tile_path = root_dir / f"{image_id}_{x1}_{y1}{name_suffix}.tif"
                    write_window(window, processed, path=tile_path)
    # Return the number of tile rows and columns written for this raster.
    return tile_row + 1, tile_col + 1



def _read_reprojected_to_reference(source_path: Path,
                                   reference_profile: dict,
                                   reference_transform,
                                   reference_crs,
                                   resampling: Resampling,
                                   scale: float = 1.0,
                                   is_mask: bool = False) -> tuple[np.ndarray, dict]:
    """Read a raster on the SAR reference grid, optionally with scaled output."""
    out_height = int(reference_profile["height"] * scale)
    out_width = int(reference_profile["width"] * scale)
    out_transform = reference_transform * reference_transform.scale(reference_profile["width"] / out_width,
                                                                    reference_profile["height"] / out_height)
    with rasterio.open(str(source_path), mode="r", driver="GTiff") as src:
        # Areas outside the source coverage must become no-data, never a real value:
        # 255 (ignore) for masks and NaN for continuous rasters. A 0 fill would
        # otherwise turn uncovered mask areas into "background" labels.
        if is_mask:
            destination = np.full((src.count, out_height, out_width), 255, dtype=np.uint8)
            dst_nodata = 255
        else:
            destination = np.full((src.count, out_height, out_width), np.nan, dtype=np.float32)
            dst_nodata = np.nan
        for band_idx in range(1, src.count + 1):
            band = src.read(band_idx)
            if is_mask:
                band = band.astype(np.uint8, copy=False) if band.dtype != np.uint8 else band
                src_nodata = src.nodata if src.nodata is not None else 255
            else:
                band = _prepare_source_array(band[None], ImageType.SAR, src.nodata)[0]
                src_nodata = np.nan
            reproject(
                source=band,
                destination=destination[band_idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=out_transform,
                dst_crs=reference_crs,
                resampling=resampling,
                src_nodata=src_nodata,
                dst_nodata=dst_nodata,
            )
        profile = src.profile.copy()
    profile.update(height=out_height, width=out_width, transform=out_transform, crs=reference_crs, count=destination.shape[0], driver="GTiff")
    if not profile.get("tiled", False):
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
    return destination, profile


def _write_tiled_array(image_id: str,
                       array: np.ndarray,
                       profile: dict,
                       dst_path: Path,
                       image_type: ImageType,
                       tiling_fn: Tiler,
                       process_fn: Optional[Callable] = None,
                       is_context: bool = False,
                       name_suffix: str = "") -> Tuple[int, int]:
    """Write an already-aligned raster array into GeoTIFF tiles."""
    group, _ = image_type.value
    process_fn = process_fn or identity
    ctx_dir = "context" if is_context else ""
    root_dir = Path(dst_path) / group / ctx_dir
    check_or_make_dir(root_dir)
    processed_img = process_fn(array)
    profile = _output_profile_for(profile, processed_img, image_type)
    tile_row = tile_col = 0
    with MemoryFile() as memfile:
        with memfile.open(**profile) as processed:
            processed.write(processed_img)
            for (tile_row, tile_col), coords in tiling_fn(processed_img):
                x1, y1, x2, y2 = coords
                window = Window.from_slices(rows=(x1, x2), cols=(y1, y2))
                tile_path = root_dir / f"{image_id}_{x1}_{y1}{name_suffix}.tif"
                write_window(window, processed, path=tile_path)
    return tile_row + 1, tile_col + 1


def _process_aligned_triplet(image_id: str,
                             sar_path: Path,
                             dem_path: Path,
                             mask_path: Path,
                             subset_dir: Path,
                             tiling_fn: Tiler,
                             sar_process: Optional[Callable],
                             dem_process: Optional[Callable],
                             mask_process: Callable,
                             scale: float,
                             name_suffix: str) -> tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Reproject DEM and mask to the SAR grid before applying transforms and tiling."""
    with rasterio.open(str(sar_path), mode="r", driver="GTiff") as sar_src:
        reference_profile = sar_src.profile.copy()
        reference_transform = sar_src.transform
        reference_crs = sar_src.crs
        out_shape = (sar_src.count, int(sar_src.height * scale), int(sar_src.width * scale))
        sar_array = sar_src.read(out_shape=out_shape, resampling=Resampling.bilinear)
        sar_array = _prepare_source_array(sar_array, ImageType.SAR, sar_src.nodata)
        sar_transform = reference_transform * reference_transform.scale(sar_src.width / out_shape[2], sar_src.height / out_shape[1])
        sar_profile = reference_profile.copy()
        sar_profile.update(height=out_shape[1], width=out_shape[2], transform=sar_transform, count=sar_array.shape[0], driver="GTiff")
        if not sar_profile.get("tiled", False):
            sar_profile.pop("blockxsize", None)
            sar_profile.pop("blockysize", None)
    dem_array, dem_profile = _read_reprojected_to_reference(dem_path, reference_profile, reference_transform, reference_crs, Resampling.bilinear, scale=scale)
    mask_array, mask_profile = _read_reprojected_to_reference(mask_path, reference_profile, reference_transform, reference_crs, Resampling.nearest, scale=scale, is_mask=True)
    sar_tiles = _write_tiled_array(image_id, sar_array, sar_profile, subset_dir, ImageType.SAR, tiling_fn, process_fn=sar_process, name_suffix=name_suffix)
    dem_tiles = _write_tiled_array(image_id, dem_array, dem_profile, subset_dir, ImageType.DEM, tiling_fn, process_fn=dem_process, name_suffix=name_suffix)
    mask_tiles = _write_tiled_array(image_id, mask_array, mask_profile, subset_dir, ImageType.MASK, tiling_fn, process_fn=mask_process, name_suffix=name_suffix)
    return sar_tiles, dem_tiles, mask_tiles


def _file_count(path: Path) -> int:
    return len(list(path.glob("*.tif"))) if path.exists() else 0


def _mask_unique_values(mask_dir: Path, max_files: int = 250) -> List[int]:
    values: Set[int] = set()
    for path in sorted(mask_dir.glob("*.tif"))[:max_files]:
        try:
            values.update(int(v) for v in np.unique(imread(path).squeeze()))
        except Exception:
            continue
    return sorted(values)


def _write_preprocessing_manifest(config: PreparationConfig,
                                  processed_root: Path,
                                  raw_counts_by_split: Dict[str, int],
                                  tile_counts_by_split: Dict[str, Dict[str, int]]) -> None:
    """Write a reproducibility manifest for a processed MMFlood dataset."""
    manifest = {
        "data_source": str(config.data_source),
        "data_processed": str(config.data_processed),
        "summary_file": str(config.summary_file),
        "subsets": sorted(list(config.subset)),
        "tile_size": int(config.tile_size),
        "tile_max_overlap": int(config.tile_max_overlap),
        "scale": [int(v) for v in config.scale],
        "tiling": bool(config.tiling),
        "sar_transform": str(getattr(config, "sar_transform", "db10")),
        "sar_units": {"db10": "dB (10*log10 linear)", "log1p": "log10(1 + linear)", "linear": "linear"}.get(
            str(getattr(config, "sar_transform", "db10")), str(getattr(config, "sar_transform", "db10"))),
        "nodata": {
            "sar": "NaN (source no-data, non-finite or non-positive linear backscatter)",
            "dem": f"NaN (source no-data, non-finite, or outside [{DEM_VALID_MIN}, {DEM_VALID_MAX}] m)",
            "mask": 255,
            "mask_ignores_sar_nodata": True,
        },
        "nodata_schema_version": 2,
        "clip_dem": bool(config.clip_dem),
        "morphology": bool(config.morphology),
        "morphology_splits": sorted({str(s).lower() for s in (getattr(config, "morphology_splits", None) or ["train"])}),
        "morph_kernel": int(config.morph_kernel),
        "nan_threshold": float(config.nan_threshold),
        "mask_mapping": {
            "flood_values": [int(v) for v in config.mask_flood_values],
            "background_values": [int(v) for v in config.mask_background_values],
            "ignore_values": [int(v) for v in config.mask_ignore_values],
            "preserve_ignore": bool(config.preserve_mask_ignore),
            "output_values": {"background": 0, "flood": 1, "ignore": 255},
        },
        "align_to_reference_grid": bool(getattr(config, "align_to_reference_grid", False)),
        "raw_rasters_by_split": raw_counts_by_split,
        "processed_tile_counts_by_split": tile_counts_by_split,
    }
    output_path = processed_root / "preprocessing_manifest.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    LOG.info("Preprocessing manifest written to: %s", output_path)

def preprocess_data(config: PreparationConfig):
    LOG.info("Preprocessing dataset")
    if config.tiling:
        if config.tile_size is None:
            raise ValueError("Tiling is enabled, so provide --tile-size 128, 256, or 512, or set tile_size in the config.")
        if config.tile_max_overlap is None:
            config.tile_max_overlap = min(128, max(1, int(config.tile_size) // 2))
            LOG.info("tile_max_overlap not set; using derived value %s for tile_size %s", config.tile_max_overlap, config.tile_size)
        if int(config.tile_max_overlap) >= int(config.tile_size):
            raise ValueError("tile_max_overlap must be smaller than tile_size")
    print_config(LOG, config)

    # Prepare destination folders and load split assignments.
    dst_dir = check_or_make_dir(config.data_processed)
    code2subset = dict()
    with open(config.summary_file, "r", encoding="utf-8") as f:
        for k, v in json.load(f).items():
            code2subset[k] = v["subset"]

    LOG.info(f"EMSR activations: {dict(Counter(code2subset.values()))}")
    raw_counts_by_split: Dict[str, int] = {}
    tile_counts_by_split: Dict[str, Dict[str, int]] = {}

    for subset in config.subset:
        emsr_codes = {k for k, v in code2subset.items() if v == subset}
        subset_dir = check_or_make_dir(dst_dir / subset)
        is_test_set = subset == "test"

        if config.tiling:
            _prepare_subset_output(subset_dir, overwrite=bool(getattr(config, "overwrite_existing", False)))
            # Find images and sanity checks
            LOG.info("Processing split=%s", subset)
            sar_files, dem_files, msk_files = _gather_files(sar_glob=config.data_source / "*" / "s1_raw" / "*.tif",
                                                            dem_glob=config.data_source / "*" / "DEM" / "*.tif",
                                                            mask_glob=config.data_source / "*" / "mask" / "*.tif",
                                                            subset=emsr_codes)
            LOG.info("Raw scenes: %d", len(sar_files))
            raw_counts_by_split[subset] = len(sar_files)
            # prepare processing instances
            # Training and validation rasters are tiled; test rasters are kept whole for sliding-window evaluation.
            make_context = bool(config.make_context) and not is_test_set
            if not is_test_set:
                tiler = DynamicOverlapTiler(tile_size=config.tile_size,
                                            overlap_threshold=config.tile_max_overlap,
                                            channels_first=True)
                available_scales = config.scale
            else:
                available_scales = [1]
                tiler = SingleImageTiler(tile_size=config.tile_size, channels_first=True)
            LOG.info("Tiling strategy: %s", type(tiler).__name__)
            sar_process = _build_sar_transform(config)
            dem_process = _clip_dem if config.clip_dem else _dem_nodata_to_nan
            # Morphological clean-up is a training-label choice. Validation and test
            # masks are kept as published so reported metrics use the official labels.
            morphology_splits = {str(s).lower() for s in (getattr(config, "morphology_splits", None) or ["train"])}
            apply_morphology = bool(config.morphology) and subset in morphology_splits
            LOG.info("Mask morphology for split=%s: %s", subset, "enabled" if apply_morphology else "disabled")
            mask_preprocessor = MaskPreprocessor(kernel_size=config.morph_kernel,
                                                 channels_first=True,
                                                 flood_values=config.mask_flood_values,
                                                 background_values=config.mask_background_values,
                                                 ignore_values=config.mask_ignore_values,
                                                 apply_morphology=apply_morphology,
                                                 preserve_ignore=config.preserve_mask_ignore)
            # iterate over the different required scales
            # values are reversed: a scale factor of 2 (x2) means a tile 1024x1024
            # this is equivalent to a tile 512x512, on the image downscaled by 1/2
            for tile_scale in available_scales:
                LOG.info("Processing raw dataset | scale=x%s", tile_scale)
                image_scale = 1.0 / tile_scale
                name_suffix = "" if tile_scale == 1 else f"_x{tile_scale}"
                # tile the triplets of images into NxN chips
                for sar_path, dem_path, msk_path in progress_iter(list(zip(sar_files, dem_files, msk_files))):
                    image_id = Path(sar_path).stem
                    if bool(getattr(config, "align_to_reference_grid", False)):
                        sar_tiles, dem_tiles, msk_tiles = _process_aligned_triplet(
                            image_id, sar_path, dem_path, msk_path, subset_dir, tiler,
                            sar_process=sar_process, dem_process=dem_process, mask_process=mask_preprocessor,
                            scale=image_scale, name_suffix=name_suffix)
                    else:
                        _check_same_grid(Path(sar_path), Path(dem_path), Path(msk_path))
                        dem_tiles = _process_tiff(image_id,
                                                  dem_path,
                                                  subset_dir,
                                                  image_type=ImageType.DEM,
                                                  tiling_fn=tiler,
                                                  process_fn=dem_process,
                                                  scale=image_scale,
                                                  resampling=Resampling.bilinear,
                                                  name_suffix=name_suffix)
                        sar_tiles = _process_tiff(image_id,
                                                  sar_path,
                                                  subset_dir,
                                                  image_type=ImageType.SAR,
                                                  tiling_fn=tiler,
                                                  process_fn=sar_process,
                                                  scale=image_scale,
                                                  resampling=Resampling.bilinear,
                                                  name_suffix=name_suffix)
                        msk_tiles = _process_tiff(image_id,
                                                  msk_path,
                                                  subset_dir,
                                                  image_type=ImageType.MASK,
                                                  tiling_fn=tiler,
                                                  process_fn=mask_preprocessor,
                                                  scale=image_scale,
                                                  resampling=Resampling.nearest,
                                                  name_suffix=name_suffix)
                    if not (sar_tiles == dem_tiles == msk_tiles):
                        raise RuntimeError(
                            f"Tile-count mismatch for {image_id}: "
                            f"SAR={sar_tiles}, DEM={dem_tiles}, mask={msk_tiles}"
                        )
                    # Generate context rasters once at the base scale when requested.
                    if make_context and tile_scale == 1:
                        _process_tiff(image_id,
                                      dem_path,
                                      subset_dir,
                                      image_type=ImageType.DEM,
                                      tiling_fn=tiler,
                                      process_fn=dem_process,
                                      scale=1,
                                      resampling=Resampling.bilinear,
                                      is_context=True)
                        _process_tiff(image_id,
                                      sar_path,
                                      subset_dir,
                                      image_type=ImageType.SAR,
                                      tiling_fn=tiler,
                                      process_fn=sar_process,
                                      scale=1,
                                      resampling=Resampling.bilinear,
                                      is_context=True)
                        _process_tiff(image_id,
                                      msk_path,
                                      subset_dir,
                                      image_type=ImageType.MASK,
                                      tiling_fn=tiler,
                                      process_fn=mask_preprocessor,
                                      scale=1,
                                      resampling=Resampling.nearest,
                                      is_context=True)

        LOG.info("Tiling complete")
        # From here, assume tiles are done and present in dst_dir
        # continue with the preprocessing
        tile_paths = _gather_files(sar_glob=subset_dir / "sar" / "*.tif",
                                   dem_glob=subset_dir / "dem" / "*.tif",
                                   mask_glob=subset_dir / "mask" / "*.tif",
                                   check_stems=True)

        valid, removed = 0, 0

        for sar_path, dem_path, msk_path in progress_iter(list(zip(*tile_paths))):
            image = imread(sar_path)
            is_context = Path(sar_path).stem.endswith("_full")
            # SAR no-data (outside swath, or non-positive linear backscatter) is NaN after
            # preprocessing. Mostly-empty training/validation tiles are removed; test scenes are kept whole.
            nan_mask = ~np.isfinite(image).all(axis=0)
            empty_pixels = np.count_nonzero(nan_mask)
            tile_pixels = float(nan_mask.size)
            if not (is_test_set or is_context) and (empty_pixels / tile_pixels) >= config.nan_threshold:
                _delete_group(sar_path, dem_path, msk_path)
                removed += 1
            else:
                # SAR/DEM keep NaN as their declared no-data value; only the label is
                # changed so losses and metrics ignore pixels without SAR data.
                # DEM voids stay NaN in the DEM tile and are ignored by the data loader
                # only when the DEM channel is actually used.
                if empty_pixels > 0:
                    mask_raster(msk_path, mask=nan_mask, mask_value=ImageType.MASK.value[-1])
                valid += 1

        kept_percent = valid / float(valid + removed) * 100.0 if (valid + removed) else 0.0
        LOG.info("Tile validation: kept=%d | removed=%d | kept_percent=%.2f%%", valid, removed, kept_percent)
        tile_counts_by_split[subset] = {
            "sar": _file_count(subset_dir / "sar"),
            "dem": _file_count(subset_dir / "dem"),
            "mask": _file_count(subset_dir / "mask"),
            "valid_tiles": valid,
            "removed_tiles": removed,
            "mask_unique_values_sample": _mask_unique_values(subset_dir / "mask"),
        }
    _write_preprocessing_manifest(config, dst_dir, raw_counts_by_split, tile_counts_by_split)
    LOG.info("Preprocessing complete")


def compute_statistics(config: StatsConfig):
    """Log channel statistics of processed tiles for one split.

    Values are read exactly as stored (already in the preprocessing units, e.g. dB),
    so no transform is re-applied. Only finite pixels whose label is not 255 are
    counted, and mean/std are global over all counted pixels. Training uses
    ``floodmap fit-normalization``; this command is a diagnostic summary.
    """
    LOG.info("Computing dataset statistics on split=%s", config.subset)
    print_config(LOG, config)

    data_root = Path(config.data_root)
    sar_paths = sorted(list(glob(str(data_root / config.subset / "sar" / "*.tif"))))
    dem_paths = sorted(list(glob(str(data_root / config.subset / "dem" / "*.tif"))))
    msk_paths = sorted(list(glob(str(data_root / config.subset / "mask" / "*.tif"))))

    if not sar_paths:
        raise FileNotFoundError(f"No processed SAR tiles found under: {data_root / config.subset / 'sar'}")
    if not (len(sar_paths) == len(dem_paths) == len(msk_paths)):
        raise ValueError(
            f"Processed tile count mismatch: SAR={len(sar_paths)}, DEM={len(dem_paths)}, mask={len(msk_paths)}"
        )

    count = total = total_sq = ch_min = ch_max = None
    for sar_path, dem_path, mask_path in progress_iter(list(zip(sar_paths, dem_paths, msk_paths))):
        stems = {Path(sar_path).stem, Path(dem_path).stem, Path(mask_path).stem}
        if len(stems) != 1:
            raise ValueError(f"Tile stem mismatch: SAR={sar_path}, DEM={dem_path}, mask={mask_path}")
        sar = imread(sar_path, channels_first=False).astype(np.float64)
        dem = imread(dem_path, channels_first=False).astype(np.float64)
        mask = imread(mask_path, channels_first=False)
        mask = mask.reshape(_dims(mask))
        if not (_dims(sar) == _dims(dem) == _dims(mask)):
            raise ValueError(
                f"Raster shape mismatch for {Path(sar_path).stem}: "
                f"SAR={_dims(sar)}, DEM={_dims(dem)}, mask={_dims(mask)}"
            )
        values = np.concatenate((sar, dem), axis=-1).reshape(-1, sar.shape[-1] + dem.shape[-1])
        labelled = (mask.reshape(-1) != 255)[:, None]
        valid = np.isfinite(values) & labelled
        if count is None:
            channels = values.shape[-1]
            count = np.zeros(channels, dtype=np.int64)
            total = np.zeros(channels, dtype=np.float64)
            total_sq = np.zeros(channels, dtype=np.float64)
            ch_min = np.full(channels, np.inf)
            ch_max = np.full(channels, -np.inf)
        safe = np.where(valid, values, 0.0)
        count += valid.sum(axis=0)
        total += safe.sum(axis=0)
        total_sq += (safe ** 2).sum(axis=0)
        ch_min = np.minimum(ch_min, np.where(valid, values, np.inf).min(axis=0))
        ch_max = np.maximum(ch_max, np.where(valid, values, -np.inf).max(axis=0))

    denom = np.maximum(count, 1)
    ch_avg = total / denom
    ch_std = np.sqrt(np.maximum(total_sq / denom - ch_avg ** 2, 0.0))
    LOG.info("Valid pixels per channel: %s", count)
    LOG.info("Channel-wise max: %s", ch_max)
    LOG.info("Channel-wise min: %s", ch_min)
    LOG.info("Channel-wise mean: %s", ch_avg)
    LOG.info("Channel-wise standard deviation: %s", ch_std)


def _processed_sar_to_linear(values: np.ndarray, sar_transform: str) -> np.ndarray:
    """Invert the preprocessing SAR transform back to linear backscatter."""
    mode = str(sar_transform or "db10").lower()
    values = np.asarray(values, dtype=np.float32)
    if mode == "db10":
        return np.power(10.0, values / 10.0).astype(np.float32)
    if mode == "log1p":
        return (np.power(10.0, values) - 1.0).astype(np.float32)
    return values


def generate_pseudolabels(config: PreparationConfig):
    """Generate training pseudo-label weight rasters from SAR thresholds.

    Output values: 0 = background, 1 = only one of threshold/label says flood,
    2 = threshold and label agree on flood, 255 = ignore (label ignore or no SAR data).
    """
    LOG.info("Generating pseudo-label weight rasters")
    data_path = Path(config.data_processed)
    if not data_path.exists() or not data_path.is_dir():
        raise FileNotFoundError(f"Processed data directory not found: {data_path}")
    manifest_path = data_path / "preprocessing_manifest.json"
    sar_transform = "log1p"
    if manifest_path.exists():
        sar_transform = json.loads(manifest_path.read_text(encoding="utf-8")).get("sar_transform", sar_transform)
    else:
        LOG.warning("preprocessing_manifest.json not found; assuming log1p SAR tiles for pseudo-labelling.")
    LOG.info("Pseudo-label thresholds are applied to linear backscatter (source tiles: %s)", sar_transform)

    # Pseudo-label weights are generated only for the training split.
    dataset = FloodDataset(path=data_path,
                           subset="train",
                           include_dem=False,
                           transform_base=None)
    # Prepare the output directory for generated pseudo-label weights.
    result_path = data_path / "train" / "weight"
    check_or_make_dir(result_path)
    morph_kernel = MorphologyTransform().create_round_kernel(kernel_size=config.morph_kernel)

    def _denoise(image: np.ndarray) -> np.ndarray:
        try:
            return denoise_nl_means(image, h=0.1, channel_axis=-1)
        except TypeError:  # scikit-image < 0.19
            return denoise_nl_means(image, h=0.1, multichannel=True)

    # Use indexed processing so output filenames remain aligned with source tiles.
    def process_image(index: int):
        image = imread(dataset.image_files[index], channels_first=False).astype(np.float32)
        label, profile = imread(dataset.label_files[index], return_metadata=True)
        label = label.squeeze(0).astype(np.uint8)
        invalid = ~np.isfinite(image).all(axis=-1)
        image = _processed_sar_to_linear(np.nan_to_num(image, nan=0.0), sar_transform)
        # Scale VV and VH before threshold-based pseudo-label generation.
        image[:, :, 0] *= config.vv_multiplier
        image[:, :, 1] *= config.vh_multiplier
        # Denoise SAR inputs and clean the binary mask with morphological opening.
        denoised = _denoise(image)
        flooded = ((denoised[:, :, 0] <= 0.1) * (denoised[:, :, 1] <= 0.1)).astype(np.uint8)
        flooded = cv2.morphologyEx(flooded, cv2.MORPH_OPEN, morph_kernel)
        # 0 = background, 1 = one source says flood, 2 = both agree; ignore stays 255.
        result = (flooded.astype(np.int16) + np.where(label == 1, 1, 0)).astype(np.uint8)
        result[(label == 255) | invalid] = 255
        # Preserve source georeferencing in the generated weight raster.
        image_name = Path(dataset.image_files[index]).name
        out_profile = dict(profile)
        out_profile.update(dtype="uint8", nodata=255, count=1)
        with rasterio.open(str(result_path / image_name), "w", **out_profile) as dst:
            dst.write(result[np.newaxis, ...])

    # Process tiles in parallel while preserving deterministic output names.
    Parallel(n_jobs=12)(delayed(process_image)(i) for i in progress_iter(range(len(dataset))))
    LOG.info("Validating generated pseudo-label rasters")
    result_images = glob(str(result_path / "*.tif"))
    if len(result_images) != len(dataset):
        raise RuntimeError(
            f"Pseudo-label output count mismatch: dataset={len(dataset)}, generated={len(result_images)}"
        )
    LOG.info("Pseudo-label generation complete")
