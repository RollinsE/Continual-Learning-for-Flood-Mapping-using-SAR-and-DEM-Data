#!/usr/bin/env python3
"""Capture a portable environment record for a training or evaluation run."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:  # pragma: no cover - defensive metadata collection
        return f"unavailable: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    record: dict[str, object] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pip_freeze": _command_output([sys.executable, "-m", "pip", "freeze"]).splitlines(),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_status": _command_output(["git", "status", "--short"]),
    }

    try:
        import torch

        record["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover
        record["torch"] = {"unavailable": str(exc)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Environment record written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
