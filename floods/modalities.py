"""Lightweight modality naming and validation utilities.

This module intentionally depends only on the Python standard library. Configuration
loading, CLI parsing, checkpoint inspection, and VV/VH deployment must not import
SciPy, Rasterio, OpenCV, or derived-feature processing code merely to validate a
list of channel names.
"""
from __future__ import annotations

from typing import Sequence

BASE_MODALITIES = ("vv", "vh", "dem")
DERIVED_MODALITIES = ("vv_vh_log_ratio", "dem_slope", "dem_tpi")
RGB_MODALITIES = ("r", "g", "b")
SUPPORTED_MODALITIES = BASE_MODALITIES + DERIVED_MODALITIES + RGB_MODALITIES

_MODALITY_ALIASES = {
    "ratio": "vv_vh_log_ratio",
    "vv_vh_ratio": "vv_vh_log_ratio",
    "vv-vh-ratio": "vv_vh_log_ratio",
    "vvvh_ratio": "vv_vh_log_ratio",
    "log_ratio": "vv_vh_log_ratio",
    "slope": "dem_slope",
    "terrain_slope": "dem_slope",
    "relative_elevation": "dem_tpi",
    "local_relative_elevation": "dem_tpi",
    "topographic_position": "dem_tpi",
    "tpi": "dem_tpi",
}


def canonicalize_modality(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "_")
    key = _MODALITY_ALIASES.get(key, key)
    if key not in SUPPORTED_MODALITIES:
        raise ValueError(
            f"Unsupported modality '{name}'. Supported: {', '.join(SUPPORTED_MODALITIES)}"
        )
    return key


def canonicalize_modalities(modalities: Sequence[str]) -> list[str]:
    result: list[str] = []
    for item in modalities:
        key = canonicalize_modality(item)
        if key in result:
            raise ValueError(f"Duplicate input modality after alias resolution: {key}")
        result.append(key)
    return result


def infer_default_modalities(
    in_channels: int,
    include_dem: bool,
    use_rgb: bool = False,
) -> list[str]:
    if use_rgb:
        result = ["r", "g", "b"]
        if include_dem:
            result.append("dem")
        if len(result) != int(in_channels):
            raise ValueError(
                f"Legacy RGB configuration declares in_channels={in_channels}, but resolves to {result}. "
                "Set data.input_modalities explicitly."
            )
        return result

    result = ["vv", "vh"] + (["dem"] if include_dem else [])
    if len(result) != int(in_channels):
        raise ValueError(
            f"Legacy flood configuration declares in_channels={in_channels}, but resolves to {result}. "
            "Set data.input_modalities explicitly for derived or non-standard channels."
        )
    return result


def resolve_input_modalities(
    input_modalities: Sequence[str] | None,
    *,
    in_channels: int,
    include_dem: bool,
    use_rgb: bool = False,
) -> list[str]:
    if input_modalities:
        result = canonicalize_modalities(input_modalities)
        if len(result) != int(in_channels):
            raise ValueError(
                f"data.in_channels={in_channels}, but data.input_modalities contains "
                f"{len(result)} channels: {result}"
            )
        return result
    return infer_default_modalities(
        in_channels=in_channels,
        include_dem=include_dem,
        use_rgb=use_rgb,
    )


def has_derived_modalities(modalities: Sequence[str]) -> bool:
    return bool(set(canonicalize_modalities(modalities)) & set(DERIVED_MODALITIES))


__all__ = [
    "BASE_MODALITIES",
    "DERIVED_MODALITIES",
    "RGB_MODALITIES",
    "SUPPORTED_MODALITIES",
    "canonicalize_modality",
    "canonicalize_modalities",
    "infer_default_modalities",
    "resolve_input_modalities",
    "has_derived_modalities",
]
