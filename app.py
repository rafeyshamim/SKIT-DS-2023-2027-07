"""Entry point:  streamlit run app.py
Linear flow: Upload -> Viewer -> Results (no auth for MVP)."""
import streamlit as st

st.set_page_config(page_title="3D CT Reconstruction & Analysis", page_icon="🩻", layout="wide")

from views import upload  # noqa: E402  (after set_page_config)

STAGES = ["upload", "viewer", "results"]
LABELS = {"upload": "1 · Upload", "viewer": "2 · 3D Viewer", "results": "3 · Results & Report"}

st.session_state.setdefault("stage", "upload")
stage = st.session_state.stage

st.markdown(" → ".join(f"**{LABELS[s]}**" if s == stage else LABELS[s] for s in STAGES))
st.divider()

if stage == "upload":
    upload.render()
else:
    # Screens 2 and 3 are separate tasks (Oct–Dec sprints); placeholder keeps the flow testable.
    st.success(f"Scan {st.session_state.get('scan_id')} uploaded and preprocessed.")
    st.info(f"{LABELS[stage]} is not built yet.")
    if st.button("← Upload another scan"):
        for key in ("scan_id", "upload_sig", "series_info", "upload_error"):
            st.session_state.pop(key, None)
        st.session_state.stage = "upload"
        st.rerun()
