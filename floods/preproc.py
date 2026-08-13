import json
import logging
import os
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


def _sar_db10(data: np.ndarray):
    """Convert positive SAR backscatter values to 10*log10 units."""
    return 10.0 * np.log10(np.maximum(data, F16_EPS))


def _sar_log1p(data: np.ndarray):
    """Apply a numerically stable log1p transform to positive SAR values."""
    return np.log10(1.0 + np.maximum(data, 0.0) + F16_EPS)


def _build_sar_transform(config: PreparationConfig):
    """Return the SAR transform selected by preprocessing configuration."""
    # SAR transform is controlled explicitly by sar_transform.
    if bool(getattr(config, "colorize", False)):
        return _rgb_ratio
    mode = str(getattr(config, "sar_transform", "db10") or "linear").lower().replace("-", "_")
    if mode in {"linear", "none"}:
        return None
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


def _clip_dem(data: np.ndarray):
    """
        MinMax the DEM between -100 and 6000 meters
    """
    data[data < -100] = -100
    data[data > 6000] = 6000
    return data


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
        profile: dict = dataset.profile.copy()
        # Some source rasters contain block size metadata without tiled output enabled.
        # Rasterio/GDAL rejects that combination when creating in-memory GeoTIFFs.
        if not profile.get("tiled", False):
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
        # Transform the image to extract the number of channels, which can change after processing.
        processed_img = process_fn(image)
        channels = processed_img.shape[0]
        profile.update(height=height, width=width, count=channels, driver="GTiff")
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
                                   scale: float = 1.0) -> tuple[np.ndarray, dict]:
    """Read a raster on the SAR reference grid, optionally with scaled output."""
    out_height = int(reference_profile["height"] * scale)
    out_width = int(reference_profile["width"] * scale)
    out_transform = reference_transform * reference_transform.scale(reference_profile["width"] / out_width,
                                                                    reference_profile["height"] / out_height)
    with rasterio.open(str(source_path), mode="r", driver="GTiff") as src:
        destination = np.zeros((src.count, out_height, out_width), dtype=src.dtypes[0])
        for band_idx in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band_idx),
                destination=destination[band_idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=out_transform,
                dst_crs=reference_crs,
                resampling=resampling,
                src_nodata=src.nodata,
                dst_nodata=src.nodata if src.nodata is not None else 0,
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
    profile = profile.copy()
    profile.update(height=processed_img.shape[1], width=processed_img.shape[2], count=processed_img.shape[0], driver="GTiff")
    if not profile.get("tiled", False):
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
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
        sar_transform = reference_transform * reference_transform.scale(sar_src.width / out_shape[2], sar_src.height / out_shape[1])
        sar_profile = reference_profile.copy()
        sar_profile.update(height=out_shape[1], width=out_shape[2], transform=sar_transform, count=sar_array.shape[0], driver="GTiff")
        if not sar_profile.get("tiled", False):
            sar_profile.pop("blockxsize", None)
            sar_profile.pop("blockysize", None)
    dem_array, dem_profile = _read_reprojected_to_reference(dem_path, reference_profile, reference_transform, reference_crs, Resampling.bilinear, scale=scale)
    mask_array, mask_profile = _read_reprojected_to_reference(mask_path, reference_profile, reference_transform, reference_crs, Resampling.nearest, scale=scale)
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
        "clip_dem": bool(config.clip_dem),
        "morphology": bool(config.morphology),
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
            dem_process = _clip_dem if config.clip_dem else None
            mask_preprocessor = MaskPreprocessor(kernel_size=config.morph_kernel,
                                                 channels_first=True,
                                                 flood_values=config.mask_flood_values,
                                                 background_values=config.mask_background_values,
                                                 ignore_values=config.mask_ignore_values,
                                                 apply_morphology=config.morphology,
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
                                   check_stems=False)

        valid, removed = 0, 0
        tile_area = config.tile_size * config.tile_size

        for sar_path, dem_path, msk_path in progress_iter(list(zip(*tile_paths))):
            image = imread(sar_path)
            is_context = Path(sar_path).stem.endswith("_full")
            # remove mostly nan images, using the configured percentage (excluding test images)
            # tile files are directly deleted, careful about this
            nan_mask = np.isnan(image.sum(axis=0))
            empty_pixels = np.count_nonzero(nan_mask)
            if not (is_test_set or is_context) and (empty_pixels / float(tile_area)) >= config.nan_threshold:
                _delete_group(sar_path, dem_path, msk_path)
                removed += 1
            # otherwise update nans into an actual ignore index
            # for sar and dem is not important, but the mask should be 255 for losses
            else:
                if empty_pixels > 0:
                    mask_raster(sar_path, mask=nan_mask, mask_value=ImageType.SAR.value[-1])
                    mask_raster(dem_path, mask=nan_mask, mask_value=ImageType.DEM.value[-1])
                    mask_raster(msk_path, mask=nan_mask, mask_value=ImageType.MASK.value[-1])
                valid += 1

        LOG.info("Tile validation: kept=%d | removed=%d | kept_percent=%.2f%%", valid, removed, valid / float(valid + removed) * 100.0)
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
    """Compute channel statistics for the configured processed-data subset."""
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

    pixel_count = 0
    ch_max = None
    ch_min = None
    ch_avg = None
    ch_std = None
    # First pass: accumulate channel extrema and means.
    LOG.info("Computing channel minima, maxima, and means")
    for sar_path, dem_path, mask_path in progress_iter(list(zip(sar_paths, dem_paths, msk_paths))):
        image_id = Path(sar_path).stem
        if not (_extract_emsr(sar_path) == _extract_emsr(dem_path) == _extract_emsr(mask_path)):
            raise ValueError(
                f"Image ID mismatch: SAR={sar_path}, DEM={dem_path}, mask={mask_path}"
            )

        # Read aligned SAR, DEM, and mask rasters.
        sar = _sar_db10(imread(sar_path, channels_first=False))
        dem = _clip_dem(imread(dem_path, channels_first=False))
        mask = imread(mask_path, channels_first=False)
        mask = mask.reshape(_dims(mask))
        if not (_dims(sar) == _dims(dem) == _dims(mask)):
            raise ValueError(
                f"Raster shape mismatch for {image_id}: "
                f"SAR={_dims(sar)}, DEM={_dims(dem)}, mask={_dims(mask)}"
            )

        # Initialise channel accumulators from the first valid triplet.
        channel_count = sar.shape[-1] + dem.shape[-1]
        if ch_max is None:
            ch_max = np.ones(channel_count) * np.finfo(np.float32).min
            ch_min = np.ones(channel_count) * np.finfo(np.float32).max
            ch_avg = np.zeros(channel_count, dtype=np.float32)
            ch_std = np.zeros(channel_count, dtype=np.float32)

        valid_pixels = mask.flatten() != 255
        sar = sar.reshape((-1, sar.shape[-1]))[valid_pixels]
        dem = dem.reshape((-1, dem.shape[-1]))[valid_pixels]

        pixel_count += sar.shape[0]
        ch_max = np.maximum(ch_max, np.concatenate((sar.max(axis=0), dem.max(axis=0)), axis=-1))
        ch_min = np.minimum(ch_min, np.concatenate((sar.min(axis=0), dem.min(axis=0)), axis=-1))
        ch_avg += np.concatenate((sar.sum(axis=0), dem.sum(axis=0)), axis=-1)
    ch_avg /= float(pixel_count)

    # Use a second pass so the standard deviation is computed from the fitted means.
    LOG.info("Computing channel standard deviations")
    for sar_path, dem_path, mask_path in progress_iter(list(zip(sar_paths, dem_paths, msk_paths))):
        image_id = Path(sar_path).stem
        # Read aligned SAR, DEM, and mask rasters.
        sar = _sar_db10(imread(sar_path, channels_first=False))
        dem = _clip_dem(imread(dem_path, channels_first=False))
        mask = imread(mask_path, channels_first=False)
        mask = mask.reshape(_dims(mask))
        if not (_dims(sar) == _dims(dem) == _dims(mask)):
            raise ValueError(
                f"Raster shape mismatch for {image_id}: "
                f"SAR={_dims(sar)}, DEM={_dims(dem)}, mask={_dims(mask)}"
            )
        # Flatten spatial dimensions while preserving channels.
        img_channels = sar.shape[-1]
        dem_channels = dem.shape[-1]
        valid_pixels = mask.flatten() != 255
        sar = sar.reshape((-1, img_channels))[valid_pixels]
        dem = dem.reshape((-1, dem_channels))[valid_pixels]
        # Accumulate per-channel variance terms.
        image_std = ((sar - ch_avg[:img_channels])**2).sum(axis=0) / float(sar.shape[0])
        dem_std = ((dem - ch_avg[img_channels:])**2).sum(axis=0) / float(sar.shape[0])
        ch_std += np.concatenate((image_std, dem_std), axis=-1)
    # Convert the accumulated variance to standard deviation.
    ch_std = np.sqrt(ch_std / len(sar_paths))
    LOG.info("Channel-wise max: %s", ch_max)
    LOG.info("Channel-wise min: %s", ch_min)
    LOG.info("Channel-wise mean: %s", ch_avg)
    LOG.info("Channel-wise standard deviation: %s", ch_std)
    LOG.info("Normalised mean: %s", (ch_avg - ch_min) / (ch_max - ch_min))
    LOG.info("Normalised standard deviation: %s", ch_std / (ch_max - ch_min))


def generate_pseudolabels(config: PreparationConfig):
    """Generate training pseudo-label weight rasters from SAR thresholds."""
    LOG.info("Generating pseudo-label weight rasters")
    data_path = Path(config.data_processed)
    if not data_path.exists() or not data_path.is_dir():
        raise FileNotFoundError(f"Processed data directory not found: {data_path}")

    # Pseudo-label weights are generated only for the training split.
    dataset = FloodDataset(path=data_path,
                           subset="train",
                           include_dem=False,
                           transform_base=None)
    # Prepare the output directory for generated pseudo-label weights.
    result_path = data_path / "train" / "weight"
    check_or_make_dir(result_path)
    morph_kernel = MorphologyTransform().create_round_kernel(kernel_size=config.morph_kernel)

    # Use indexed processing so output filenames remain aligned with source tiles.
    def process_image(index: int):
        image = imread(dataset.image_files[index], channels_first=False)
        label, profile = imread(dataset.label_files[index], return_metadata=True)
        label = label.squeeze(0).astype(np.uint8)
        # Scale VV and VH before threshold-based pseudo-label generation.
        image[:, :, 0] *= config.vv_multiplier
        image[:, :, 1] *= config.vh_multiplier
        # Denoise SAR inputs and clean the binary mask with morphological opening.
        denoised = denoise_nl_means(image, h=0.1, multichannel=True)
        flooded = ((denoised[:, :, 0] <= 0.1) * (denoised[:, :, 1] <= 0.1)).astype(np.uint8)
        flooded = cv2.morphologyEx(flooded, cv2.MORPH_OPEN, morph_kernel)
        # Encode background as 0, threshold union as 1, and threshold/label intersection as 2.
        result = flooded + label
        # Preserve source georeferencing in the generated weight raster.
        image_name = Path(dataset.image_files[index]).name
        with rasterio.open(str(result_path / image_name), "w", **profile) as dst:
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
