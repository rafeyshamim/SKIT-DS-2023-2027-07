"""Entry point:  streamlit run app.py
Linear flow: Upload -> Viewer -> Results (no auth for MVP)."""
import streamlit as st

st.set_page_config(page_title="3D CT Reconstruction & Analysis", page_icon="🩻", layout="wide")

from views import upload, viewer, results  # noqa: E402  (after set_page_config)

STAGES = ["upload", "viewer", "results"]
LABELS = {"upload": "1 · Upload", "viewer": "2 · 3D Viewer", "results": "3 · Results & Report"}

st.session_state.setdefault("stage", "upload")
stage = st.session_state.stage

st.markdown(" → ".join(f"**{LABELS[s]}**" if s == stage else LABELS[s] for s in STAGES))
st.divider()

if stage == "upload":
    upload.render()
elif stage == "viewer":
    viewer.render()
elif stage == "results":
    results.render()

