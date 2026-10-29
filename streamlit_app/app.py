from __future__ import annotations

import io
import logging
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import yaml

from floods.deployment import predict_scene
from streamlit_app.collection import (
    build_collection_mosaic,
    build_equal_area_evaluation,
    pooled_evaluation,
    write_collection_report,
)
from streamlit_app.raster_inputs import (
    canonical_tile_id,
    inspect_upload,
    prepare_sar_candidates,
    stage_auxiliary_uploads,
    write_candidate_csv,
)


APP_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
LOG = logging.getLogger("floodmap.streamlit")


def _deployment_manifests() -> list[Path]:
    configured = os.environ.get("FLOODMAP_DEPLOYMENT_MANIFEST", "").strip()
    if configured:
        return [Path(configured).expanduser().resolve()]
    manifests = sorted((APP_DIR / "deployments").glob("*/deployment_manifest.yaml"))
    single_bundle = APP_DIR / "deployment" / "deployment_manifest.yaml"
    if single_bundle.is_file():
        manifests.append(single_bundle)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(path.resolve() for path in manifests if path.is_file()))


def _load_manifest_summary(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    inputs = data.get("inputs") or {}
    operating = data.get("operating_point") or {}
    inference = data.get("inference") or {}
    members = data.get("members") or []
    return {
        "name": data.get("name") or data.get("model_name") or path.parent.name,
        "mode": data.get("mode") or ("ensemble" if len(members) > 1 else "single"),
        "members": len(members),
        "modalities": list(inputs.get("modalities") or []),
        "threshold": float(operating.get("threshold", 0.5)),
        "window_size": int(inference.get("window_size") or 512),
        "window_overlap": int(inference.get("window_overlap") or 128),
        "portable": bool((data.get("bundle") or {}).get("portable")),
    }


def _validate_upload_size(uploaded, label: str) -> None:
    if uploaded is not None and uploaded.size > MAX_UPLOAD_BYTES:
        raise ValueError(f"{label} exceeds the 200 MB per-file upload limit.")


def _clear_previous_workspace() -> None:
    previous = st.session_state.pop("floodmap_work_dir", None)
    st.session_state.pop("floodmap_result", None)
    if previous:
        shutil.rmtree(previous, ignore_errors=True)


def _reset_session() -> None:
    """Remove the previous results and empty the upload widgets."""
    _clear_previous_workspace()
    st.session_state["floodmap_upload_round"] = st.session_state.get("floodmap_upload_round", 0) + 1


def _natural_key(value) -> list:
    """Sort key that puts EMSR107-5-2 before EMSR107-5-10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def _upload_signature(uploads) -> list[tuple[str, int]]:
    return sorted((str(item.name), int(item.size)) for item in uploads or [])


def _zip_directory(root: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root))
    return buffer.getvalue()


def _metric(value) -> str:
    return "–" if value is None else f"{float(value):.4f}"


def _find_preview_for_prediction(prediction: dict, suffix: str) -> Path | None:
    mask_value = prediction.get("output_mask")
    if not mask_value:
        return None
    output_dir = Path(mask_value).parent
    matches = sorted((output_dir / "previews").glob(f"*{suffix}"))
    return matches[0] if matches else None


def _needs_dem(modalities: list[str]) -> bool:
    return any(name in {"dem", "dem_slope", "dem_tpi"} for name in modalities)


def _display_results(payload: dict) -> None:
    summary = payload["summary"]
    predictions = summary.get("predictions") or []
    mosaic = payload.get("mosaic") or {}
    evaluation = payload.get("evaluation")
    manifest_summary = payload["manifest_summary"]
    output_dir = Path(summary["output_dir"])

    st.success("Prediction complete")
    water_states = [p.get("permanent_water") for p in predictions if p.get("permanent_water")]
    failed_water = [w for w in water_states if not w.get("applied")]
    if failed_water:
        st.warning("The permanent-water layer could not be read for "
                   f"{len(failed_water)} of {len(predictions)} tile(s), so those predictions include permanent water. "
                   f"Reason: {failed_water[0].get('error', 'unknown')}")
    elif water_states:
        removed = sum(int(w.get("removed_flood_pixels", 0)) for w in water_states)
        st.caption(f"Permanent water removed (JRC Global Surface Water, ≥50% occurrence): {removed:,} predicted pixels.")
    st.subheader("Flood prediction")
    if mosaic.get("created"):
        cols = st.columns(5)
        cols[0].metric("Tiles processed", f"{len(predictions)}/{payload['expected_tiles']}")
        cols[1].metric("Mapped area", f"{float(mosaic['mapped_area_km2']):.3f} km²")
        cols[2].metric("Predicted flood area", f"{float(mosaic['flood_area_km2']):.3f} km²")
        cols[3].metric("Flood coverage", f"{float(mosaic['flood_fraction']) * 100:.2f}%")
        cols[4].metric("Threshold", f"{manifest_summary['threshold']:.2f}")
        st.caption(
            "Whole-area metrics come from a post-inference equal-area mosaic. "
            "Native SAR tiles are not resampled before model inference."
        )
        image_cols = st.columns(2)
        image_cols[0].image(mosaic["mask_preview"], caption="Whole-area flood mask", use_container_width=True)
        image_cols[1].image(mosaic["probability_preview"], caption="Whole-area flood probability", use_container_width=True)
        st.caption(
            "Map legend: flood-mask yellow = predicted flood, purple = predicted non-flood, "
            "grey = no input coverage. Grey also marks no input coverage on the probability map."
        )
    elif len(predictions) == 1:
        prediction = predictions[0]
        cols = st.columns(4)
        cols[0].metric("Flood pixels", f"{int(prediction.get('flood_pixels') or 0):,}")
        cols[1].metric("Flood fraction", f"{float(prediction.get('flood_fraction') or 0.0) * 100:.2f}%")
        area = prediction.get("flood_area_km2")
        cols[2].metric("Approx. flood area", "–" if area is None else f"{float(area):.3f} km²")
        cols[3].metric("Threshold", f"{manifest_summary['threshold']:.2f}")
    else:
        st.warning(
            "Per-tile inference completed, but a safe whole-area mosaic could not be created: "
            f"{mosaic.get('reason') or 'unknown geospatial compatibility issue'}"
        )

    if evaluation:
        if evaluation.get("mode") == "equal_area":
            st.subheader("Whole-area labelled evaluation")
        else:
            st.subheader("Pooled labelled-tile evaluation")
        eval_cols = st.columns(5)
        eval_cols[0].metric("F1", _metric(evaluation.get("f1")))
        eval_cols[1].metric("IoU", _metric(evaluation.get("iou")))
        eval_cols[2].metric("Precision", _metric(evaluation.get("precision")))
        eval_cols[3].metric("Recall", _metric(evaluation.get("recall")))
        eval_cols[4].metric("MCC", _metric(evaluation.get("mcc")))
        if evaluation.get("mode") == "equal_area":
            st.caption(
                f"Metrics use {evaluation['tiles']} matched mask tile(s) on the same post-inference equal-area grid "
                f"as the collection mosaic ({float(evaluation.get('labelled_area_km2') or 0.0):.3f} km² labelled). "
                "Masks never influence model inference."
            )
            overlay = evaluation.get("evaluation_overlay")
            if overlay and Path(str(overlay)).is_file():
                st.image(
                    str(overlay),
                    caption="Whole-area evaluation overlay — green TP, red FP, blue FN",
                    use_container_width=True,
                )
            if int(evaluation.get("conflict_pixels") or 0) > 0:
                st.warning(
                    f"{int(evaluation['conflict_pixels']):,} equal-area pixel(s) had conflicting overlapping "
                    "ground-truth labels; majority voting was used and ties were retained as flood."
                )
        else:
            st.caption(
                f"Metrics pool confusion counts from {evaluation['tiles']} matched mask tile(s). "
                "They are pixel-weighted across the matched source grids; masks never influence model inference."
            )

    if len(predictions) > 1:
        st.subheader("Per-tile results")
        predictions = sorted(predictions, key=lambda item: _natural_key(item.get("candidate_id")))
        rows = []
        for item in predictions:
            area = item.get("flood_area_km2")
            rows.append(
                {
                    "Tile": item.get("candidate_id"),
                    "Flood pixels": int(item.get("flood_pixels") or 0),
                    "Flood %": f"{float(item.get('flood_fraction') or 0.0) * 100:.2f}",
                    "Approx. area km²": "–" if area is None else f"{float(area):.3f}",
                    "Labelled": "yes" if item.get("evaluation_metrics") else "no",
                }
            )
        st.dataframe(rows, hide_index=True, use_container_width=True)
        tile_ids = [str(item.get("candidate_id")) for item in predictions]
        selected_id = st.selectbox("Inspect tile", tile_ids)
        selected = next(item for item in predictions if str(item.get("candidate_id")) == selected_id)
    else:
        selected = predictions[0] if predictions else None

    if selected:
        overlay = _find_preview_for_prediction(selected, "_binary_mask_overlay.png")
        probability = _find_preview_for_prediction(selected, "_flood_probability_heatmap.png")
        error_overlay = _find_preview_for_prediction(selected, "_error_overlay.png")
        image_cols = st.columns(2)
        if overlay:
            image_cols[0].image(str(overlay), caption="Binary flood-mask overlay", use_container_width=True)
        if probability:
            image_cols[1].image(str(probability), caption="Flood probability", use_container_width=True)
        if error_overlay:
            st.image(str(error_overlay), caption="Evaluation overlay — green TP, red FP, blue FN", use_container_width=True)
        report_value = selected.get("visual_report")
        report_path = Path(report_value) if report_value else None
        if report_path and report_path.is_file():
            with st.expander("Open full deployment report"):
                components.html(report_path.read_text(encoding="utf-8"), height=950, scrolling=True)

    st.subheader("Downloads")
    report_path = Path(payload["collection_report"])
    if mosaic.get("created"):
        cols = st.columns(4)
        cols[0].download_button(
            "Whole-area flood mask",
            data=Path(mosaic["output_mask"]).read_bytes(),
            file_name=Path(mosaic["output_mask"]).name,
            mime="image/tiff",
        )
        cols[1].download_button(
            "Whole-area probability",
            data=Path(mosaic["output_probability"]).read_bytes(),
            file_name=Path(mosaic["output_probability"]).name,
            mime="image/tiff",
        )
        cols[2].download_button(
            "Collection HTML report",
            data=report_path.read_bytes(),
            file_name=report_path.name,
            mime="text/html",
        )
        cols[3].download_button(
            "All outputs (.zip)",
            data=_zip_directory(output_dir),
            file_name="flood_extent_mapping_outputs.zip",
            mime="application/zip",
        )
    else:
        cols = st.columns(2)
        cols[0].download_button(
            "Collection HTML report",
            data=report_path.read_bytes(),
            file_name=report_path.name,
            mime="text/html",
        )
        cols[1].download_button(
            "All outputs (.zip)",
            data=_zip_directory(output_dir),
            file_name="flood_extent_mapping_outputs.zip",
            mime="application/zip",
        )
    st.caption("Prediction outputs are temporary on Community Cloud; download anything you want to keep.")


st.set_page_config(page_title="Flood Extent Mapping", page_icon="🌊", layout="wide")
st.title("Flood Extent Mapping")
st.caption(
    "Generate flood-extent predictions from geospatial imagery using trained segmentation models. "
    "The application validates model-specific inputs, matches auxiliary rasters by tile identifier, "
    "and keeps model inference in the standard deployment engine."
)

manifest_paths = _deployment_manifests()
if not manifest_paths:
    st.error("No portable deployment bundle is available to the application.")
    st.code(
        "Stage one or more exported bundles under:\n"
        "streamlit_app/deployments/<model-name>/deployment_manifest.yaml",
        language="text",
    )
    st.stop()

manifest_records = []
for path in manifest_paths:
    try:
        summary = _load_manifest_summary(path)
        manifest_records.append((path, summary))
    except Exception:
        LOG.exception("Deployment manifest loading failed: %s", path)

if not manifest_records:
    st.error("Deployment manifests were found, but none could be read.")
    st.stop()

if len(manifest_records) > 1:
    labels = [f"{summary['name']} · {'+'.join(summary['modalities']) or 'modalities not declared'}" for _, summary in manifest_records]
    selected_label = st.selectbox("Deployment model", labels)
    manifest_path, manifest_summary = manifest_records[labels.index(selected_label)]
else:
    manifest_path, manifest_summary = manifest_records[0]

with st.sidebar:
    st.header("Deployment")
    st.write(f"**Model:** {manifest_summary['name']}")
    st.write(f"**Mode:** {manifest_summary['mode']}")
    st.write(f"**Members:** {manifest_summary['members']}")
    st.write(f"**Modalities:** {', '.join(manifest_summary['modalities']) or 'not declared'}")
    st.write(f"**Threshold:** {manifest_summary['threshold']:.2f}")
    st.write(f"**Window:** {manifest_summary['window_size']} px")
    st.write(f"**Overlap:** {manifest_summary['window_overlap']} px")
    st.success("Portable bundle" if manifest_summary["portable"] else "Deployment manifest")
    st.divider()
    st.caption("Community Cloud runs inference on CPU. Generated files are temporary for each app session.")
    st.info(
        "Decision-support output only. Review predictions alongside source imagery and official information; "
        "do not use the model as the sole basis for life-safety decisions."
    )

st.subheader("Inputs")
upload_round = st.session_state.get("floodmap_upload_round", 0)
st.write(
    "Upload one SAR tile or a complete set of tiles. The application inspects band counts automatically; "
    "separate single-band VV/VH files are paired when their polarization can be identified safely."
)
sar_uploads = st.file_uploader(
    "Sentinel-1 SAR GeoTIFF(s)",
    type=["tif", "tiff"],
    accept_multiple_files=True,
    help="Select one or more SAR GeoTIFFs. You do not need to declare whether a file contains one band or a combined VV/VH pair.",
    key=f"floodmap_sar_uploads_{upload_round}",
)
_unit_labels = {"Detect automatically": "auto", "Linear power (σ⁰/γ⁰)": "linear", "Decibels (dB)": "db"}
st.session_state["floodmap_sar_units"] = _unit_labels[st.radio(
    "SAR backscatter units",
    list(_unit_labels),
    horizontal=True,
    help="Uploads are converted to the units the model was trained on. Zero/NaN/no-data pixels are treated as missing.",
)]
st.session_state["floodmap_permanent_water"] = st.checkbox(
    "Remove permanent water",
    value=True,
    help="Lakes, reservoirs and rivers that the JRC Global Surface Water layer records as water at least half the time "
         "(1984-2020) are shown as non-flood. Flood labels exclude permanent water, and removing it cut false positives "
         "by about a quarter in validation. Needs an internet connection; if the layer cannot be read, the prediction "
         "is shown unmasked with a warning.",
)

needs_dem = _needs_dem(manifest_summary["modalities"])
if needs_dem:
    dem_uploads = st.file_uploader(
        "DEM GeoTIFF(s)",
        type=["tif", "tiff"],
        accept_multiple_files=True,
        help="DEM files are matched to SAR tiles by canonical tile identifier, not upload order.",
        key=f"floodmap_dem_uploads_{upload_round}",
    )
else:
    dem_uploads = []
    st.caption("This deployment uses SAR modalities only; no DEM upload is required.")

mask_uploads = st.file_uploader(
    "Ground-truth mask GeoTIFF(s) (optional)",
    type=["tif", "tiff"],
    accept_multiple_files=True,
    help="Masks are optional and used only for evaluation. They are matched to SAR tiles by canonical tile identifier.",
    key=f"floodmap_mask_uploads_{upload_round}",
)

if sar_uploads:
    try:
        infos = [inspect_upload(item) for item in sar_uploads]
        combined_count = sum(info.count >= 2 for info in infos)
        single_count = sum(info.count == 1 for info in infos)
        st.caption(
            f"Detected {len(infos)} SAR file(s): {combined_count} combined/multiband, {single_count} single-band. "
            "Final VV/VH pairing is validated when you run the model."
        )
    except Exception as exc:
        st.warning(f"One or more SAR files could not be inspected yet: {exc}")

current_signature = _upload_signature(list(sar_uploads or []) + list(dem_uploads or []) + list(mask_uploads or []))
button_cols = st.columns([1, 1, 4])
run_clicked = button_cols[0].button("Run flood mapping", type="primary", disabled=not sar_uploads)
button_cols[1].button(
    "Clear results and uploads",
    on_click=_reset_session,
    disabled=not (sar_uploads or dem_uploads or mask_uploads or st.session_state.get("floodmap_result")),
    help="Remove the current results and uploaded files to start a new test.",
)

if run_clicked and sar_uploads:
    all_uploads = list(sar_uploads) + list(dem_uploads or []) + list(mask_uploads or [])
    try:
        for uploaded in all_uploads:
            _validate_upload_size(uploaded, str(uploaded.name))
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    _clear_previous_workspace()
    work_dir = Path(tempfile.mkdtemp(prefix="floodmap_streamlit_"))
    st.session_state["floodmap_work_dir"] = str(work_dir)
    input_root = work_dir / "inputs"
    output_dir = work_dir / "outputs"

    try:
        candidates, sar_errors, sar_warnings = prepare_sar_candidates(sar_uploads, input_root)
        dem_paths, dem_errors = stage_auxiliary_uploads(dem_uploads or [], input_root, "dem")
        mask_paths, mask_errors = stage_auxiliary_uploads(mask_uploads or [], input_root, "mask")
        errors = sar_errors + dem_errors + mask_errors
        if not candidates:
            errors.append("No deployable SAR tile could be constructed from the uploaded files.")
        if needs_dem:
            missing_dem = [candidate.candidate_id for candidate in candidates if candidate.candidate_id not in dem_paths]
            if missing_dem:
                errors.append("Missing DEM for required tile(s): " + ", ".join(missing_dem))

        candidate_ids = {candidate.candidate_id for candidate in candidates}
        unmatched_dem = sorted(set(dem_paths) - candidate_ids)
        unmatched_masks = sorted(set(mask_paths) - candidate_ids)
        if unmatched_dem:
            sar_warnings.append("Unmatched DEM tile(s): " + ", ".join(unmatched_dem))
        if unmatched_masks:
            sar_warnings.append("Unmatched mask tile(s): " + ", ".join(unmatched_masks))

        st.subheader("Input matching")
        match_cols = st.columns(4)
        match_cols[0].metric("SAR tiles", len(candidates))
        match_cols[1].metric("DEM matched", f"{sum(c.candidate_id in dem_paths for c in candidates)}/{len(candidates)}" if needs_dem else "Not required")
        match_cols[2].metric("Masks matched", f"{sum(c.candidate_id in mask_paths for c in candidates)}/{len(candidates)}")
        match_cols[3].metric("SAR layout", f"{sum(c.kind == 'multiband_vv_vh' for c in candidates)} combined / {sum(c.kind == 'separate_vv_vh' for c in candidates)} paired")
        for warning in sar_warnings:
            st.warning(warning)
        if errors:
            for message in errors:
                st.error(message)
            shutil.rmtree(work_dir, ignore_errors=True)
            st.session_state.pop("floodmap_work_dir", None)
            st.stop()

        input_csv = write_candidate_csv(work_dir / "input_candidates.csv", candidates, dem_paths, mask_paths)
        with st.status("Running flood mapping…", expanded=True) as status:
            summary = predict_scene(
                manifest_path=manifest_path,
                input_csv=input_csv,
                evaluate=bool(mask_paths),
                output_dir=output_dir,
                device="cpu",
                mosaic_mode="off",
                write_probability=True,
                write_previews=True,
                write_overlay=True,
                write_html_report=True,
                display_inline=False,
                output_mode="concise",
                input_sar_units=st.session_state.get("floodmap_sar_units", "auto"),
                permanent_water_threshold=50 if st.session_state.get("floodmap_permanent_water", True) else None,
            )
            predictions = summary.get("predictions") or []
            mosaic = build_collection_mosaic(predictions, output_dir / "collection") if len(predictions) > 1 else {
                "created": False,
                "reason": "Single-tile prediction does not require a collection mosaic.",
            }
            source_grid_evaluation = pooled_evaluation(predictions)
            evaluation = (
                build_equal_area_evaluation(predictions, mosaic, output_dir / "collection")
                if mosaic.get("created")
                else None
            ) or source_grid_evaluation
            collection_report = write_collection_report(
                output_dir / "collection_report.html",
                predictions,
                mosaic,
                evaluation,
                manifest_summary,
            )
            status.update(label="Prediction complete", state="complete", expanded=False)
    except Exception:
        LOG.exception("Flood Extent Mapping Streamlit prediction failed")
        shutil.rmtree(work_dir, ignore_errors=True)
        st.session_state.pop("floodmap_work_dir", None)
        st.error("Prediction failed. Check the server log for the technical cause; uploaded data were not modified.")
        st.stop()

    payload = {
        "summary": summary,
        "mosaic": mosaic,
        "evaluation": evaluation,
        "collection_report": str(collection_report),
        "manifest_summary": manifest_summary,
        "expected_tiles": len(candidates),
        "upload_signature": current_signature,
    }
    st.session_state["floodmap_result"] = payload

payload = st.session_state.get("floodmap_result")
if payload:
    if payload.get("upload_signature") != current_signature:
        st.info("These results are for the previous set of files. Click **Run flood mapping** to process the current "
                "uploads, or **Clear results and uploads** to start again.")
    _display_results(payload)
