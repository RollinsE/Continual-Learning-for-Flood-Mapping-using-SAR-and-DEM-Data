from enum import Enum
from pathlib import Path
from typing import List, Optional, Set

try:
    from pydantic.v1 import Field, validator
except ImportError:  # pragma: no cover
    from pydantic import Field, validator

from floods.config.base import EnvConfig


class ImageType(Enum):
    SAR = ("sar", 0)
    DEM = ("dem", 0)
    MASK = ("mask", 255)


class StatsConfig(EnvConfig):
    data_root: Path = Field(description="Path where processed tiles are stored")
    subset: str = Field("train", description="Subset used to compute statistics")


class PreparationConfig(EnvConfig):
    data_source: Path = Field(description="Path containing SAR, DEM, and mask folders")
    data_processed: Path = Field(description="Path where processed tiles will be written")
    subset: Set[str] = Field(default_factory=lambda: {"train", "val", "test"},
                             description="Subsets to preprocess; each subset must exist in the split metadata")
    summary_file: str = Field(description="JSON file containing dataset metadata and split assignments")
    tiling: bool = Field(True, description="Whether to generate tiles and preprocess masks")
    scale: List[int] = Field([1], description="Scaling multipliers for each tile (before resizing to tile_size).")
    tile_size: Optional[int] = Field(None, description="Tile/window size in pixels. Set from CLI with --tile-size; supported values are 128, 256, or 512")
    tile_max_overlap: Optional[int] = Field(None, description="Maximum dynamic overlap before skipping a neighbouring tile. If omitted, a safe default is derived from tile_size")
    make_context: bool = Field(False, description="Whether to generate the context-based variant of the set")
    decibel: bool = Field(True, description="Apply a log10 transformation to the SAR signal")
    colorize: bool = Field(False, description="Apply an RGB ratio transformation (decibel takes priority)")
    clip_dem: bool = Field(True, description="Whether to apply min-max normalization to the DEM")
    morphology: bool = Field(True, description="Whether to apply morphological cleanup to the flood class")
    morph_kernel: int = Field(5, description="Kernel size for binary flood-mask morphology")
    nan_threshold: float = Field(0.75, description="Maximum invalid-pixel fraction before discarding a tile")
    sar_transform: str = Field("db10", description="SAR intensity transform: linear, db10, or log1p")
    mask_flood_values: List[int] = Field([1], description="Raw mask values to map to flood/foreground (output value 1)")
    mask_background_values: List[int] = Field([0, 2], description="Raw mask values to map to non-flood/background (output value 0)")
    mask_ignore_values: List[int] = Field([255], description="Raw mask values to preserve as ignore/nodata (output value 255)")
    preserve_mask_ignore: bool = Field(True, description="Restore ignore pixels after morphology so ignored areas never become training labels")
    align_to_reference_grid: bool = Field(False, description="Reproject DEM and mask rasters to the SAR raster grid before tiling")
    vv_multiplier: float = Field(5.0, description="Fixed multiplier for threshold-based pseudolabeling (1st channel)")
    vh_multiplier: float = Field(10.0, description="Fixed multiplier for threshold-based pseudolabeling (2nd channel)")


    @validator("tile_size")
    def validate_tile_size(cls, v):
        if v is None:
            return v
        value = int(v)
        allowed = {128, 256, 512}
        if value not in allowed:
            raise ValueError(f"tile_size must be one of {sorted(allowed)}")
        return value

    @validator("tile_max_overlap")
    def validate_tile_overlap(cls, v):
        if v is None:
            return v
        value = int(v)
        if value <= 0:
            raise ValueError("tile_max_overlap must be positive")
        return value

    @validator("sar_transform")
    def validate_sar_transform(cls, v):
        value = str(v or "linear").strip().lower().replace("-", "_")
        aliases = {"none": "linear", "db": "db10", "decibel": "db10", "log": "log1p"}
        value = aliases.get(value, value)
        allowed = {"linear", "db10", "log1p"}
        if value not in allowed:
            raise ValueError(f"sar_transform must be one of {sorted(allowed)}")
        return value

    @validator("mask_flood_values", "mask_background_values", "mask_ignore_values", pre=True)
    def validate_mask_value_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = [int(x.strip()) for x in v.split(",") if x.strip()]
        return [int(x) for x in v]

    @validator("subset", pre=True, always=True)
    def subset_exists(cls, v):
        allowed = {"train", "test", "val"}
        if not v:
            raise ValueError("Specify a subset before running")
        if not set(v) <= allowed:
            raise ValueError(f"subsets must belong to: {allowed}")
        return set(v)
