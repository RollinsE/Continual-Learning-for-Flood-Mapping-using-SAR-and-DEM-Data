"""Flood Extent Mapping command-line training, evaluation, and deployment package."""

from __future__ import annotations

import os

# Prevent Albumentations from performing an update check during package import.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from floods.version import __version__

__all__ = ["__version__"]
