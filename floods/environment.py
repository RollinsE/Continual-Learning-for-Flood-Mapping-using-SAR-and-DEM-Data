"""Runtime dependency diagnostics for Flood Extent Mapping."""
from __future__ import annotations

import importlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _check_module(name: str, probe) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "version": _distribution_version(name),
        "status": "failed",
        "detail": "",
    }
    try:
        module = importlib.import_module(name.replace("-", "_"))
        probe(module)
    except Exception as exc:  # diagnostics must report every binary/import failure
        row["detail"] = f"{type(exc).__name__}: {exc}"
    else:
        row["status"] = "ok"
        row["detail"] = "import and runtime probe passed"
    return row


def run_environment_checks() -> dict[str, Any]:
    """Run small import/runtime probes for the deployment and training stack."""

    def probe_numpy(np):
        values = np.arange(9, dtype=np.float32).reshape(3, 3)
        assert float(values.mean()) == 4.0
        # This catches mixed NumPy installations like the missing `_center`
        # failure observed after an interrupted in-place package upgrade.
        centred = np.char.center(np.array(["x"]), 3)
        assert centred.shape == (1,)

    def probe_scipy(scipy):
        import numpy as np
        from scipy.ndimage import uniform_filter

        values = np.arange(9, dtype=np.float32).reshape(3, 3)
        result = uniform_filter(values, size=3)
        assert result.shape == values.shape

    def probe_rasterio(rasterio):
        import numpy as np
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin

        with MemoryFile() as mem:
            with mem.open(
                driver="GTiff",
                height=2,
                width=2,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_origin(0, 2, 1, 1),
            ) as dst:
                dst.write(np.ones((1, 2, 2), dtype=np.float32))
            with mem.open() as src:
                assert src.read(1).shape == (2, 2)

    def probe_torch(torch):
        values = torch.tensor([1.0, 2.0])
        assert float(values.sum()) == 3.0

    checks = [
        _check_module("numpy", probe_numpy),
        _check_module("scipy", probe_scipy),
        _check_module("rasterio", probe_rasterio),
        _check_module("torch", probe_torch),
    ]
    try:
        from floods import __version__
    except Exception:
        package_version = None
    else:
        package_version = __version__

    return {
        "status": "ok" if all(row["status"] == "ok" for row in checks) else "failed",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "floodmap_version": package_version,
        "checks": checks,
    }


def write_environment_report(payload: dict[str, Any], output_file: Path) -> Path:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_file


def format_environment_report(payload: dict[str, Any]) -> str:
    lines = [
        f"Flood Extent Mapping environment: {payload['status'].upper()}",
        f"Flood Extent Mapping: {payload.get('floodmap_version') or payload.get('mmflood_version') or 'unknown'} | Python: {payload['python']}",
        "component | version | status | detail",
    ]
    for row in payload["checks"]:
        lines.append(
            f"{row['name']} | {row.get('version') or 'not installed'} | "
            f"{row['status']} | {row['detail']}"
        )
    if payload["status"] != "ok":
        lines.extend(
            [
                "",
                "The numeric environment is inconsistent. Recreate or repair the Python environment using the repository requirements and constraints for the target runtime.",
                "For Google Colab, use: bash scripts/install_colab.sh",
                "Restart the Python process after replacing NumPy/SciPy binary packages.",
            ]
        )
    return "\n".join(lines)
