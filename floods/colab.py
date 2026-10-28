"""Notebook helpers for running the pipeline on Google Colab.

Standard library only (tqdm is used when available), so the module can be
imported before the package's dependencies are installed:

    import sys; sys.path.insert(0, REPO_DIR)
    from floods.colab import run, unzip, zip_dir

``run`` streams a shell command through a pseudo-terminal. IPython's ``%%bash``
and ``!`` read output line by line with a 64 KB limit, so long tqdm lines
(which use carriage returns, not newlines) stall the cell output or kill it.
"""
from __future__ import annotations

import os
import pty
import re
import select
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable, Optional

__all__ = ["run", "unzip", "zip_dir", "find_file", "best_checkpoint"]


def _bar(total: int, desc: str, unit: str = "file"):
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, desc=desc, unit=unit, colour="green", mininterval=1.0)
    except Exception:  # pragma: no cover - tqdm missing
        return None


def run(command: str, check: bool = True, env: Optional[dict] = None) -> int:
    """Run ``command`` with bash and stream its output live, including progress bars.

    Environment variables set with ``os.environ`` in the notebook are visible to the command.
    Raises ``RuntimeError`` on a non-zero exit when ``check`` is true.
    """
    master, slave = pty.openpty()
    full_env = dict(os.environ if env is None else env)
    full_env.setdefault("PYTHONUNBUFFERED", "1")
    full_env.setdefault("COLUMNS", "120")
    process = subprocess.Popen(["bash", "-c", "set -o pipefail\n" + command], stdin=subprocess.DEVNULL,
                               stdout=slave, stderr=slave, env=full_env, close_fds=True)
    os.close(slave)
    out = sys.stdout
    try:
        while True:
            ready, _, _ = select.select([master], [], [], 0.5)
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:  # child closed the terminal
                    chunk = b""
                if not chunk:
                    if process.poll() is not None:
                        break
                    continue
                out.write(chunk.decode("utf-8", errors="replace"))
                out.flush()
            elif process.poll() is not None:
                break
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    finally:
        os.close(master)
    code = process.wait()
    if check and code != 0:
        raise RuntimeError(f"Command failed with exit code {code}")
    return code


def _ensure_7z() -> str:
    exe = shutil.which("7z")
    if exe:
        return exe
    subprocess.run(["apt-get", "-qq", "install", "-y", "p7zip-full"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    exe = shutil.which("7z")
    if not exe:
        raise RuntimeError("7z is needed for this zip (Deflate64) but could not be installed.")
    return exe


def unzip(zip_path: str | Path, dest: str | Path, desc: Optional[str] = None) -> Path:
    """Extract ``zip_path`` into ``dest`` with a green progress bar.

    Falls back to 7-Zip for compression methods Python cannot read (e.g. Deflate64,
    which Windows uses for large archives).
    """
    zip_path, dest = Path(zip_path), Path(dest)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    dest.mkdir(parents=True, exist_ok=True)
    desc = desc or f"unzip {zip_path.name}"
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        supported = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA}
        if all(m.compress_type in supported for m in members):
            bar = _bar(sum(m.file_size for m in members), desc, unit="B")
            if bar is not None:
                bar.unit_scale = True
            for member in members:
                archive.extract(member, dest)
                if bar is not None:
                    bar.update(member.file_size)
            if bar is not None:
                bar.close()
            return dest
    exe = _ensure_7z()
    bar = _bar(100, desc + " (7z)", unit="%")
    process = subprocess.Popen([exe, "x", "-y", "-bsp1", "-bso0", f"-o{dest}", str(zip_path)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    buffer, last = b"", 0
    if process.stdout is None:  # pragma: no cover - Popen always sets it with stdout=PIPE
        raise RuntimeError("7z output pipe missing")
    while True:
        chunk = process.stdout.read(256)
        if not chunk:
            break
        buffer = (buffer + chunk)[-512:]
        found = re.findall(rb"(\d{1,3})%", buffer)
        if found and bar is not None:
            pct = min(100, int(found[-1]))
            if pct > last:
                bar.update(pct - last)
                last = pct
    code = process.wait()
    if bar is not None:
        bar.update(100 - last)
        bar.close()
    if code != 0:
        raise RuntimeError(f"7z failed with exit code {code}")
    return dest


def zip_dir(source: str | Path, zip_path: str | Path, desc: Optional[str] = None) -> Path:
    """Zip the contents of ``source`` (paths relative to it) into ``zip_path``.

    The archive is written locally first and then copied, which is much faster than
    writing a zip directly onto Google Drive.
    """
    source, zip_path = Path(source), Path(zip_path)
    files = [p for p in sorted(source.rglob("*")) if p.is_file()]
    local = Path("/content") / (zip_path.name + ".partial") if Path("/content").exists() else zip_path.with_suffix(".partial")
    local.parent.mkdir(parents=True, exist_ok=True)
    bar = _bar(len(files), desc or f"zip {source.name}")
    with zipfile.ZipFile(local, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for path in files:
            archive.write(path, path.relative_to(source))
            if bar is not None:
                bar.update(1)
    if bar is not None:
        bar.close()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying {local.stat().st_size / 2**30:.2f} GiB to {zip_path} ...", flush=True)
    start = time.time()
    shutil.copyfile(local, str(zip_path) + ".tmp")
    os.replace(str(zip_path) + ".tmp", zip_path)
    local.unlink()
    print(f"Done in {time.time() - start:.0f}s", flush=True)
    return zip_path


def find_file(root: str | Path, name: str) -> Optional[Path]:
    """Return the shallowest file called ``name`` under ``root`` (or None)."""
    hits = sorted(Path(root).rglob(name), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def best_checkpoint(run_dir: str | Path, patterns: Iterable[str] = ("model-*_best_*.pth",)) -> Optional[Path]:
    """Best checkpoint of a training run: the highest-scoring ``model-XXX_best_<metric>-<score>.pth``."""
    models = Path(run_dir) / "models"
    candidates = [p for pattern in patterns for p in models.glob(pattern)]

    def score(path: Path) -> float:
        match = re.search(r"-([0-9]*\.?[0-9]+)\.pth$", path.name)
        return float(match.group(1)) if match else float("-inf")

    return max(candidates, key=lambda p: (score(p), p.stat().st_mtime)) if candidates else None
