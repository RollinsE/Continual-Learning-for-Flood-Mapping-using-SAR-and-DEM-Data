"""Permanent-water mask from the JRC Global Surface Water occurrence layer.

Flood labels describe water that is there because of the flood, so lakes, reservoirs
and rivers that are almost always wet count as background. The SAR model sees them
as open water like any other, which makes permanent water the most common source of
false positives. This module reads JRC occurrence (the percentage of observations
1984-2020 in which a pixel was water) onto the prediction grid so those pixels can be
excluded before thresholding.

Only the standard library and rasterio are used. The layer is read from Microsoft
Planetary Computer, so an internet connection is needed.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/jrc-gsw"
JRC_COLLECTION = "jrc-gsw"
JRC_ASSET = "occurrence"
DEFAULT_OCCURRENCE_THRESHOLD = 50
SOURCE_NOTE = ("JRC Global Surface Water occurrence 1984-2020 (EC JRC/Google), "
               "hosted by Microsoft Planetary Computer")

__all__ = ["DEFAULT_OCCURRENCE_THRESHOLD", "occurrence_on_grid", "permanent_water_mask", "water_from_occurrence"]


def _request_json(url: str, payload: Optional[Mapping[str, Any]] = None, timeout: float = 60.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _signed(href: str, token: str) -> str:
    return f"{href}{'&' if '?' in href else '?'}{token}"


def occurrence_on_grid(crs: Any, transform: Any, shape: Sequence[int]) -> np.ndarray:
    """JRC occurrence (0-100 %, 255 = no data) resampled onto a raster grid (nearest neighbour)."""
    import rasterio
    from rasterio.transform import array_bounds
    from rasterio.warp import Resampling, reproject, transform_bounds

    height, width = int(shape[0]), int(shape[1])
    west, south, east, north = array_bounds(height, width, transform)
    bbox = transform_bounds(crs, "EPSG:4326", west, south, east, north, densify_pts=21)
    items = _request_json(STAC_SEARCH_URL, {"collections": [JRC_COLLECTION], "bbox": list(bbox), "limit": 50}).get("features", [])
    hrefs = [item.get("assets", {}).get(JRC_ASSET, {}).get("href") for item in items]
    hrefs = [href for href in hrefs if href]
    if not hrefs:
        raise RuntimeError(f"No JRC surface-water tiles overlap the scene (bbox {tuple(round(v, 4) for v in bbox)}).")
    token = _request_json(SAS_TOKEN_URL)["token"]

    occurrence = np.full((height, width), 255, dtype=np.uint8)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2"):
        for href in hrefs:
            with rasterio.open(_signed(href, token)) as source:
                warped = np.full((height, width), 255, dtype=np.uint8)
                reproject(source=rasterio.band(source, 1), destination=warped,
                          src_transform=source.transform, src_crs=source.crs, src_nodata=255,
                          dst_transform=transform, dst_crs=crs, dst_nodata=255,
                          resampling=Resampling.nearest)
            covered = warped <= 100
            occurrence[covered] = warped[covered]
    return occurrence


def water_from_occurrence(occurrence: np.ndarray, threshold: int = DEFAULT_OCCURRENCE_THRESHOLD) -> np.ndarray:
    """Boolean mask of pixels that were water in at least ``threshold`` % of observations."""
    occurrence = np.asarray(occurrence)
    return (occurrence <= 100) & (occurrence >= int(threshold))


def permanent_water_mask(profile: Mapping[str, Any], shape: Sequence[int],
                         threshold: int = DEFAULT_OCCURRENCE_THRESHOLD) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return (water mask, summary) for a raster described by a rasterio profile."""
    if not 0 < int(threshold) <= 100:
        raise ValueError("The permanent-water occurrence threshold must be between 1 and 100 %.")
    occurrence = occurrence_on_grid(profile["crs"], profile["transform"], shape)
    water = water_from_occurrence(occurrence, threshold)
    info = {
        "applied": True,
        "source": SOURCE_NOTE,
        "occurrence_threshold_percent": int(threshold),
        "layer_coverage": float(np.mean(occurrence <= 100)),
        "water_pixels": int(water.sum()),
    }
    return water, info
