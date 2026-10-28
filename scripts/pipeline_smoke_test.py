"""End-to-end smoke test on tiny synthetic Sentinel-1 scenes (CPU, a few minutes).

Run this before any expensive job:

    python scripts/pipeline_smoke_test.py            # keeps outputs in .pipeline_smoke/
    python scripts/pipeline_smoke_test.py --clean    # delete outputs afterwards

It builds an MMFlood-like raw folder (linear VV/VH with a no-data swath edge, a DEM
with a void, masks with 0/1/255), then runs the real CLI commands:

    preprocess -> fit-normalization -> train -> evaluate -> continual-train (+ resume)
    -> export-deployment -> predict-scene (linear and dB uploads)

and checks the properties the pipeline is meant to guarantee. Every check
prints PASS or raises with a message. The models are tiny and trained for one epoch,
so metric values are meaningless; only the plumbing is tested.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

SIZE = 300
EVENTS = {
    # event: (split, start date)
    "EMSR101": ("train", "2015-03-01"), "EMSR102": ("train", "2016-06-01"),
    "EMSR201": ("train", "2018-02-01"), "EMSR202": ("train", "2019-05-01"),
    "EMSR301": ("train", "2020-04-01"), "EMSR302": ("train", "2021-01-01"),
    "EMSR103": ("val", "2016-09-01"), "EMSR203": ("val", "2019-09-01"), "EMSR303": ("val", "2021-06-01"),
    "EMSR104": ("test", "2017-01-01"), "EMSR204": ("test", "2018-11-01"), "EMSR304": ("test", "2020-10-01"),
}
STEPS: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"FAIL: {message}")
    print(f"  PASS  {message}")


def run(*args: str, expect_fail: bool = False) -> subprocess.CompletedProcess:
    command = [sys.executable, "-m", "floods.cli", *args]
    STEPS.append(args[0])
    start = time.time()
    print(f"\n$ floodmap {' '.join(args)}")
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.time() - start
    if expect_fail:
        if result.returncode == 0:
            raise AssertionError(f"FAIL: expected `{args[0]}` to fail but it succeeded")
        print(f"  (failed as expected in {elapsed:.1f}s)")
        return result
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-6000:])
        raise AssertionError(f"FAIL: `floodmap {args[0]}` exited with {result.returncode}")
    print(f"  ok in {elapsed:.1f}s")
    return result


def _write(path: Path, data: np.ndarray, dtype: str, nodata=None, descriptions=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = data if data.ndim == 3 else data[None]
    profile = dict(driver="GTiff", height=data.shape[1], width=data.shape[2], count=data.shape[0], dtype=dtype,
                   crs="EPSG:32631", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=nodata)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype))
        if descriptions:
            for index, text in enumerate(descriptions, start=1):
                dst.set_band_description(index, text)


def make_scene(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return linear VV/VH (2,H,W), DEM (H,W) and raw mask (H,W)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    centre_y, centre_x = rng.integers(80, 220, size=2)
    radius = rng.integers(35, 70)
    flood = ((yy - centre_y) ** 2 + (xx - centre_x) ** 2) < radius ** 2
    speckle = rng.gamma(4.0, 0.25, size=(2, SIZE, SIZE))
    vv = np.where(flood, 0.004, 0.09) * speckle[0]
    vh = np.where(flood, 0.001, 0.02) * speckle[1]
    sar = np.stack([vv, vh]).astype(np.float32)
    sar[:, :, :25] = 0.0  # swath edge: Sentinel-1 GRD no-data is 0
    dem = (120.0 + 0.3 * xx + 0.2 * yy - 25.0 * flood + rng.normal(0, 2, (SIZE, SIZE))).astype(np.float32)
    dem[200:215, 200:215] = -32768.0  # DEM void
    mask = flood.astype(np.uint8)
    mask[:5, :] = 255
    return sar, dem, mask


def build_raw(root: Path) -> Path:
    for index, (event, (split, start)) in enumerate(EVENTS.items()):
        sar, dem, mask = make_scene(index)
        stem = f"{event}-0"
        _write(root / event / "s1_raw" / f"{stem}.tif", sar, "float32")
        _write(root / event / "DEM" / f"{stem}.tif", dem, "float32")
        _write(root / event / "mask" / f"{stem}.tif", mask, "uint8")
    summary = {event: {"subset": split, "start": f"{start}T00:00:00"} for event, (split, start) in EVENTS.items()}
    summary_path = root / "activations.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / ".pipeline_smoke")
    parser.add_argument("--clean", action="store_true", help="Delete the work directory at the end.")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    shutil.rmtree(work, ignore_errors=True)
    raw, processed, runs = work / "raw", work / "processed", work / "runs"
    started = time.time()

    print("== Building synthetic raw scenes")
    summary = build_raw(raw)

    # ------------------------------------------------------------------ preprocessing
    run("preprocess", "--config", "configs/preprocess_mmflood.yaml", "--raw-data-dir", str(raw),
        "--processed-data-dir", str(processed), "--summary-file", str(summary), "--tile-size", "128")
    manifest = json.loads((processed / "preprocessing_manifest.json").read_text())
    check(manifest["sar_transform"] == "db10", "processed SAR tiles are in dB")
    check(manifest.get("morphology_splits") == ["train"], "mask morphology applies to train only")
    sar_tiles = sorted((processed / "train" / "sar").glob("*.tif"))
    check(len(sar_tiles) > 0, f"train tiles written ({len(sar_tiles)})")
    edge_tile = next(p for p in sar_tiles if p.stem.endswith("_0_0"))
    sar = read(edge_tile)
    label = read(processed / "train" / "mask" / edge_tile.name)[0]
    nan_px = ~np.isfinite(sar).all(axis=0)
    check(nan_px[:, :25].all(), "swath-edge zeros became NaN in SAR tiles")
    check(np.all(label[nan_px] == 255), "labels are 255 (ignore) wherever SAR has no data")
    finite = sar[np.isfinite(sar)]
    check(finite.max() < 15 and finite.min() > -60, f"SAR values look like dB (range {finite.min():.1f} to {finite.max():.1f})")
    dem_tiles = [read(p) for p in (processed / "train" / "dem").glob("*.tif")]
    check(any(np.isnan(d).any() for d in dem_tiles), "DEM void (-32768) stored as NaN")
    check(all(np.nanmin(d) >= -100 for d in dem_tiles if np.isfinite(d).any()), "no DEM fill values leak into tiles")
    test_mask = read(processed / "test" / "mask" / "EMSR104-0_0_0.tif")[0]
    raw_mask = read(raw / "EMSR104" / "mask" / "EMSR104-0.tif")[0]
    test_sar = read(processed / "test" / "sar" / "EMSR104-0_0_0.tif")
    valid = np.isfinite(test_sar).all(axis=0)
    check(np.array_equal(test_mask[valid], raw_mask[valid]), "test labels are the published masks (no morphology)")

    run("preprocess", "--config", "configs/preprocess_mmflood.yaml", "--raw-data-dir", str(raw),
        "--processed-data-dir", str(processed), "--summary-file", str(summary), "--tile-size", "128", expect_fail=True)
    check(True, "re-running into a non-empty folder is refused")
    run("preprocess", "--config", "configs/preprocess_mmflood.yaml", "--raw-data-dir", str(raw),
        "--processed-data-dir", str(processed), "--summary-file", str(summary), "--tile-size", "128", "--overwrite-existing")
    check(len(list((processed / "train" / "sar").glob("*.tif"))) == len(sar_tiles), "--overwrite-existing rebuilds the same tiles")

    # ------------------------------------------------------------------ normalisation + training
    stats = work / "normalization_stats.json"
    run("fit-normalization", "--processed-data-dir", str(processed), "--output-file", str(stats),
        "--input-modalities", "vv", "vh", "dem", "--max-pixels-per-file", "2000")
    from floods.normalization import load_normalization_stats
    mean, std, cmin, cmax = load_normalization_stats(stats, ["vv", "vh", "dem"], mode="robust_minmax")
    check(cmin[0] < 0 and cmax[0] < 15, f"VV clip range is in dB ({cmin[0]:.1f}..{cmax[0]:.1f})")

    common = ["--config", "configs/train_unet_resnet34_vv_vh_dem.yaml", "--processed-data-dir", str(processed),
              "--output-folder", str(runs), "--image-size", "128", "--batch-size", "4", "--num-workers", "2",
              "--encoder", "resnet10t", "--decoder", "unet", "--no-pretrained", "--cpu", "--no-amp",
              "--input-modalities", "vv", "vh", "dem", "--normalization-mode", "robust_minmax",
              "--normalization-stats-path", str(stats), "--augmentation-profile", "composite",
              "--thresholds", "0.3", "0.5", "0.7", "--threshold-metric", "f1"]
    run("train", *common, "--run-id", "smoke_train", "--epochs", "1", "--threshold-sweep",
        "--no-weighted-sampling", "--no-stratified-sampling", "--no-event-balanced-sampling")
    run_dir = runs / "smoke_train"
    checkpoints = sorted((run_dir / "models").glob("*.pth"))
    check(bool(checkpoints), f"training wrote a checkpoint ({checkpoints[0].name if checkpoints else '-'})")
    import yaml
    saved = yaml.safe_load((run_dir / "config.yaml").read_text())
    check(saved["data"]["source_sar_transform"] == "db10", "run config records the SAR units (db10)")
    check(saved["data"]["normalization_mode"] == "robust_minmax", "run config uses robust_minmax")

    run("evaluate", "--config", str(run_dir / "config.yaml"), "--checkpoint", str(checkpoints[0]),
        "--processed-data-dir", str(processed), "--split", "test", "--cpu", "--no-amp", "--no-pretrained",
        "--inference-mode", "sliding_window", "--window-size", "128", "--window-overlap", "32", "--window-batch-size", "2",
        "--thresholds", "0.3", "0.5", "0.7")

    # ------------------------------------------------------------------ continual learning
    cl_args = [*common, "--run-id", "smoke_cl", "--activations-json-path", str(summary),
               "--strategies", "finetune", "random", "entropy", "hybrid", "joint",
               "--task-year-ranges", "2014-2017", "2018-2019", "2020-2021", "--epochs-per-task", "1",
               "--replay-buffer-size", "6", "--replay-batch-size", "2",
               "--cl-eval-split", "test", "--cl-eval-inference-mode", "sliding_window",
               "--window-size", "128", "--window-overlap", "32", "--window-batch-size", "2"]
    run("continual-train", *cl_args)
    for strategy in ["finetune", "random", "entropy", "hybrid", "joint"]:
        result = json.loads((runs / f"smoke_cl_{strategy}" / "cl_results.json").read_text())
        rows = [r for r in result["detailed_matrix"] if r.get("metrics")]
        check(len(result["threshold_selection"]) == 3, f"{strategy}: one threshold chosen per task (3)")
        check(all(r["threshold_source"] == "val_seen_tasks" for r in result["detailed_matrix"]),
              f"{strategy}: test cells use the validation-selected threshold")
        check(len({(r["after_task"], r["threshold"]) for r in rows}) == len({r["after_task"] for r in rows}),
              f"{strategy}: every cell in an evaluation row shares one threshold")
        check(all((runs / f"smoke_cl_{strategy}" / "models" / f"after_task_{t}.pth").exists() for t in (1, 2, 3)),
              f"{strategy}: per-task checkpoints saved")
        check(result["protocol"]["uses_replay"] == (strategy not in {"finetune", "joint"}), f"{strategy}: replay on/off as expected")
    joint = json.loads((runs / "smoke_cl_joint" / "cl_results.json").read_text())
    check(joint["tasks"][2]["train_samples"] > 0 and joint["protocol"]["cumulative_training"], "joint trains cumulatively")
    check((runs / "smoke_cl_cl_comparison.csv").exists(), "strategy comparison CSV written")
    task_ckpt = runs / "smoke_cl_random" / "models" / "after_task_1.pth"
    before = task_ckpt.stat().st_mtime
    resume_args = list(cl_args)
    start = resume_args.index("--strategies")
    end = resume_args.index("--task-year-ranges")
    resume_args[start:end] = ["--strategies", "random"]
    run("continual-train", *resume_args, "--resume")
    check(task_ckpt.stat().st_mtime == before, "--resume on a finished CL run does not retrain any task")

    # ------------------------------------------------------------------ deployment
    bundle = work / "bundle" / "deployment_manifest.yaml"
    run("export-deployment", "--output-file", str(bundle), "--configs", str(run_dir / "config.yaml"),
        "--checkpoints", str(checkpoints[0]), "--threshold", "0.5", "--input-modalities", "vv", "vh", "dem",
        "--normalization-stats-path", str(stats), "--window-size", "128", "--window-overlap", "32")
    bundle_manifest = yaml.safe_load(bundle.read_text())
    check(bundle_manifest["inputs"]["sar_units"] == "db10", "deployment bundle records training SAR units")

    sar_linear, dem, _ = make_scene(99)
    scene = work / "scene"
    _write(scene / "scene_linear.tif", sar_linear, "float32", descriptions=["VV", "VH"])
    with np.errstate(divide="ignore"):
        sar_db = np.where(sar_linear > 0, 10 * np.log10(np.maximum(sar_linear, 1e-10)), np.nan).astype(np.float32)
    _write(scene / "scene_db.tif", sar_db, "float32", descriptions=["VV", "VH"])
    _write(scene / "dem.tif", dem, "float32")
    outputs = {}
    for name in ("scene_linear", "scene_db"):
        out_dir = work / f"predict_{name}"
        run("predict-scene", "--manifest", str(bundle), "--sar-path", str(scene / f"{name}.tif"),
            "--dem-path", str(scene / "dem.tif"), "--output-dir", str(out_dir), "--cpu", "--prediction-only",
            "--write-probability", "--output-prefix", name)
        outputs[name] = read(out_dir / f"{name}_flood_mask.tif")[0]
        with rasterio.open(out_dir / f"{name}_flood_mask.tif") as src:
            check(src.nodata == 255, f"{name}: output mask declares nodata=255")
    check(np.all(outputs["scene_linear"][:, :25] == 255), "no-data swath edge is 255 in the output mask, not flood")
    check(np.all(outputs["scene_linear"][200:215, 200:215] == 255), "DEM void is 255 in the output mask")
    agreement = float(np.mean(outputs["scene_linear"] == outputs["scene_db"]))
    check(agreement > 0.999, f"linear and dB uploads give the same prediction (agreement {agreement:.4f})")

    print(f"\nALL SMOKE CHECKS PASSED in {time.time() - started:.0f}s ({len(STEPS)} CLI runs). Outputs: {work}")
    if args.clean:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
