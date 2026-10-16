"""Screen 1 - CT scan upload (wireframe: upload.py)."""
import streamlit as st

import config
from services.api_client import BackendError, run_preprocessing, upload_scan
from services.scan_reader import ScanReadError, read_series_info


def _signature(files):
    return tuple((f.name, f.size) for f in files)


def _analyse(files):
    """Parse once per distinct selection; cache result in session_state."""
    sig = _signature(files)
    if st.session_state.get("upload_sig") != sig:
        st.session_state.upload_sig = sig
        st.session_state.series_info = None
        st.session_state.upload_error = None
        with st.spinner("Reading scan information…"):
            try:
                st.session_state.series_info = read_series_info([(f.name, f.getvalue()) for f in files])
            except ScanReadError as exc:
                st.session_state.upload_error = str(exc)


def _run_pipeline(files):
    bar = st.progress(0, text="Uploading… 0%")

    def on_progress(done, total):
        pct = int(done / total * 100)
        bar.progress(pct, text=f"Uploading… {pct}%")

    try:
        scan_id = upload_scan([(f.name, f.getvalue()) for f in files], on_progress)
        with st.spinner("Preprocessing… this can take a minute."):
            run_preprocessing(scan_id)
    except BackendError as exc:
        bar.empty()
        st.error(str(exc))
        return
    st.session_state.scan_id = scan_id
    st.session_state.stage = "viewer"
    st.rerun()


def render():
    st.title("CT Scan Upload — 3D Reconstruction & Disease Analysis")
    st.caption("Upload a CT series (.dcm files) or a single NIfTI volume (.nii / .nii.gz).")

    files = st.file_uploader(
        "Drag & drop CT series (.dcm / .nii) — or Browse files",
        type=config.UPLOADER_TYPES,
        accept_multiple_files=True,
    )

    if not files:
        for key in ("upload_sig", "series_info", "upload_error"):
            st.session_state.pop(key, None)
        st.info("No scan selected yet.")
        st.button("Run Reconstruction →", disabled=True)
        return

    _analyse(files)
    info = st.session_state.get("series_info")
    error = st.session_state.get("upload_error")

    if error:
        st.error(error)
    if info:
        st.subheader("Series info")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patient / Scan ID", info.scan_id)
        c2.metric("Slice count", info.slice_count)
        c3.metric("Modality", info.modality)
        c4.metric("Format", info.file_format)
        st.caption(f"Dimensions: {' × '.join(map(str, info.dimensions))} · Files: {info.file_count}")

    if st.button("Run Reconstruction →", type="primary", disabled=info is None):
        _run_pipeline(files)
